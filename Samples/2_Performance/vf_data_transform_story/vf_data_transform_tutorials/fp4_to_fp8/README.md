# FP4 E2M1 转 FP8 E4M3FN：raw-field 映射

## 引言

Ascend 950 没有 FP4 E2M1 到 FP8 E4M3FN 的单次浮点 Cast 接口。若先把 FP4 解码成更高精度数值，再转换为 FP8，会引入多级转换和更宽的中间数据通路。本教程直接变换 sign、exponent 和 mantissa 的 raw field，使用整数位操作生成 FP8 raw byte。

为了让字段映射可由移位和掩码精确表达，本 VF 写出 `y=x/64`。下游需要执行 `x=y*64`，或将 64 合并到已有反量化 scale。最终数据流为：

```text
packed FP4 E2M1 ND
  -> LoadAlign(DIST_US_B8)
  -> ShiftRight / ShiftLeft / Select / And
  -> StoreAlign(DIST_NORM_B8)
```

## 支持范围

- 硬件：Ascend 950，`dav-3510`，64 个 AIV。
- 软件：CANN 9.2.0，Release 构建，ASC `-O3`。
- 性能模型：Ascend950PR_9589 CAMODEL V100，Vector 频率 1.65 GHz。
- 输入：packed FP4 E2M1，每个 byte 先存 low nibble，再存 high nibble。
- 输出：FP8 E4M3FN raw byte，数值语义为输入除以 64。
- 形状：`full`、`perf` 使用 `N=4096`，`tail` 使用 `N=4099`；逻辑元素数按完整 packed byte 补齐。
- 验证：按 raw-field 公式生成 golden，逐 bit 比较并检查输出 guard。

## raw-field 映射原理

E2M1 nibble 由 1 个 sign bit、2 个 exponent bit 和 1 个 mantissa bit 组成。对应的 E4M3FN `x/64` raw byte 可以通过算术移位保留符号，再以 `0x9C` 清理与目标字段无关的 bit。这个过程不调用浮点 Cast。

`DIST_US_B8` 每次读取 128 个 packed byte，并把每个 byte 展开到相邻的两个 B8 lane。B16 pair mask 随后在每对 lane 中选择 low/high 候选值。`trace` case 中每个 AIV 处理 65536 个逻辑 FP4，即 32768 B 输入和 65536 B 输出，共执行 256 组转换。

## 优化路径与目录

| 阶段 | 对应技巧 | 本阶段变化 |
| --- | --- | --- |
| `0_raw_field_mapping` | raw-field 映射替代浮点 Cast | 用算术移位和位掩码直接生成两路 FP8 raw byte |
| `1_unpack_packed_byte` | `DIST_US_B8` 展开 packed byte | 相邻 lane 配合 Select，替代 Interleave |
| `2_shift_select_and` | 三次 shift + Select + And | 把两次 And 合并为 Select 后的一次 And |
| `3_hoist_constants` | 循环外提移位量和 bit mask | 三个常量寄存器仅创建一次 |
| `4_fold_scale` | 下游折叠 `×64` scale | 不在当前 VF 中增加乘法与中间写回 |

五个目录都包含完整的 VF、kernel、数据搬运和 host 代码，可独立构建和验证；公共目录只提供 ACL 生命周期管理及输入输出等辅助能力。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DNPU_ARCH=dav-3510 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vf_data_transform_fp4_to_fp8_tutorial -j4
```

## 基准实现：0_raw_field_mapping

源码：[0_raw_field_mapping/raw_field_mapping.asc](./0_raw_field_mapping/raw_field_mapping.asc)，目标：`vf_data_transform_fp4_to_fp8_0_raw_field_mapping`。

起点不把 FP4 解码为中间浮点类型，而是直接按编码字段计算。high nibble 算术右移 2 位；low nibble 先左移 4 位，再算术右移 2 位；两个候选值分别与 `0x9C` 按位与。寄存器按 `int8_t` 执行算术右移，以保留 nibble sign。

本阶段使用普通 B8 Load，low/high 两路结果经 Interleave 合成连续输出。这样首先验证 raw-field 公式，不把 Load 排布变化混入同一步。

CANNsim 实测为 1301 cycles，有效输入吞吐为 25.19 B/cycle，有效输出吞吐为 50.37 B/cycle。

## 优化阶段一：1_unpack_packed_byte

源码：[1_unpack_packed_byte/unpack_packed_byte.asc](./1_unpack_packed_byte/unpack_packed_byte.asc)，目标：`vf_data_transform_fp4_to_fp8_1_unpack_packed_byte`。

专用 Load 将 packed byte 展开到相邻 B8 lane：

```cpp
AscendC::Reg::LoadAlign<uint8_t, AscendC::Reg::LoadDist::DIST_US_B8>(
    loadReg, input + packedOffset);
```

每个 byte 的 low/high nibble 现在可以通过 B16 pair mask 的一次 `Select` 合成，不再执行额外 Interleave。本阶段仍分别对 low/high 候选值执行 And，以便单独观察 Load 排布优化。

CANNsim 降至 826 cycles，相比阶段 0 减少 36.5%；有效输出吞吐提高到 79.34 B/cycle。

## 优化阶段二：2_shift_select_and

源码：[2_shift_select_and/shift_select_and.asc](./2_shift_select_and/shift_select_and.asc)，目标：`vf_data_transform_fp4_to_fp8_2_shift_select_and`。

两个候选值使用相同的 `0x9C` 掩码，而 Select 每个 lane 只保留其中一路，因此可以先 Select，再执行一次 And：

```cpp
AscendC::Reg::ShiftRight(highScaledReg, loadReg, shiftRightReg, fullB8Mask);
AscendC::Reg::ShiftLeft(lowShiftReg, loadReg, shiftLeftReg, fullB8Mask);
AscendC::Reg::ShiftRight(lowScaledReg, lowShiftReg, shiftRightReg, fullB8Mask);
AscendC::Reg::Select(selectedReg, lowScaledReg, highScaledReg, pairB16Mask);
AscendC::Reg::And(outputReg, selectedReg, rawFieldMaskReg, fullB8Mask);
```

热路径由此减少一次 And。CANNsim 降至 698 cycles，相比阶段 1 减少 15.5%，有效输出吞吐提高到 93.89 B/cycle。

## 优化阶段三：3_hoist_constants

源码：[3_hoist_constants/hoist_constants.asc](./3_hoist_constants/hoist_constants.asc)，目标：`vf_data_transform_fp4_to_fp8_3_hoist_constants`。

`shiftRight=2`、`shiftLeft=4` 和 `rawFieldMask=0x9C` 与循环次数无关，在进入热循环前各执行一次 `Duplicate`：

```cpp
AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(
    shiftRightReg, 2, fullB8Mask);
AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(
    shiftLeftReg, 4, fullB8Mask);
AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(
    rawFieldMaskReg, static_cast<int8_t>(0x9C), fullB8Mask);
```

CANNsim 仍为 698 cycles，说明当前 `-O3` 已对前一阶段的循环不变量完成等价优化。显式外提固定了常量寄存器的生命周期，避免复用代码时依赖编译器的隐式判断。

## 优化阶段四：4_fold_scale

源码：[4_fold_scale/fold_scale.asc](./4_fold_scale/fold_scale.asc)，目标：`vf_data_transform_fp4_to_fp8`。

raw-field 数据通路输出的是 `x/64`。若下游已有 `value * scale`，只需把其 scale 预先乘以 64，即可恢复原始数值，不需要在本 VF 中增加一次乘法、额外寄存器或中间写回。

这是跨算子的接口优化，本阶段 VF 热路径与阶段 3 相同，host 日志显式给出 `downstream_scale=64`。

CANNsim 仍为 698 cycles。局部 kernel 的指令数不变；只有下游本来就存在 scale 时，`×64` 才能无额外 Vector 开销地折叠。

## 性能结果

以下数据由 CANNsim 在 Ascend 950 上使用 `--case trace` 采集。每个 AIV 处理一个 65536 元素 tile；`VF cycles` 取 core 0 业务 kernel 的 `avg_cycles`。

| 阶段 | VF cycles | 相比上一步 | 输入 B/cycle | 输出 B/cycle |
| --- | ---: | ---: | ---: | ---: |
| `0_raw_field_mapping` | 1301 | 起点 | 25.19 | 50.37 |
| `1_unpack_packed_byte` | 826 | 减少 36.5% | 39.67 | 79.34 |
| `2_shift_select_and` | 698 | 减少 15.5% | 46.95 | 93.89 |
| `3_hoist_constants` | 698 | 持平 | 46.95 | 93.89 |
| `4_fold_scale` | 698 | 持平 | 46.95 | 93.89 |

从 raw-field 映射起点到最终阶段，VF cycles 从 1301 降至 698，局部数据通路提速 1.86 倍，cycles 减少 46.3%。最终阶段整体 Vec IPC 为 1.97。阶段 3 的显式外提和阶段 4 的跨算子契约不增加当前 VF 指令，因此性能不回退；阶段 4 的端到端收益需要结合实际下游算子评估。

## 运行与验证

运行最终阶段：

```bash
./build/Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/\
fp4_to_fp8/4_fold_scale/vf_data_transform_fp4_to_fp8 --case all
```

验证程序比较 `x/64` 对应的 FP8 E4M3FN raw byte，不在当前 kernel 中执行下游 `×64`。通过时，`full`、`tail` 和 `perf` 均输出 `status=PASS`、`bin_match=true`，host 信息同时包含 `downstream_scale=64`。

## 总结与复用建议

raw-field 映射适用于硬件没有直接 Cast，且源格式与目标格式之间存在可证明字段关系的转换。复用时必须同时确认 raw-field 公式、packed nibble 顺序和 `×64` 补偿契约，并对零、次正规数、最大有限值及符号位逐 bit 验证。若下游无法折叠 scale，应把恢复缩放的计算和搬运代价计入端到端性能，而不能只比较本 VF 的 cycles。
