# packed INT4 转 BF16：UNPACK4 与单路 Cast

## 引言

INT4 权重或中间结果通常以一个 byte 存放两个有符号 nibble。转换为 BF16 时，关键不是先用通用位操作拆分 low/high nibble，而是让 Load 直接生成 Cast 可以消费的寄存器排布，再用一条 Cast 完成数值转换。

本教程从 `DIST_UNPACK4_B8` Load 开始，依次说明单路 INT4→BF16 Cast、packed-byte 与 logical-element 视图分离，以及 packed Cast 所需的 B8 predicate view。最终热路径为：

```text
packed INT4 ND
  -> LoadAlign(DIST_UNPACK4_B8)
  -> Cast(ZERO)
  -> StoreAlign(DIST_NORM_B16)
```

## 支持范围

- 硬件：Ascend 950，`dav-3510`，64 个 AIV。
- 软件：CANN 9.2.0，Release 构建，ASC `-O3`。
- 性能模型：Ascend950PR_9589 CAMODEL V100，Vector 频率 1.65 GHz。
- 输入：连续 packed signed INT4，每个 byte 先存 low nibble，再存 high nibble。
- 输出：连续 BF16，输入范围 `-8..7` 可精确表示。
- 形状：`full`、`perf` 使用 `N=4096`，`tail` 使用 `N=4099`；逻辑元素数按完整 packed byte 补齐。
- 验证：对 BF16 raw bits 逐 bit 比较，并检查输出 guard。

## 硬件架构基础

一个 256 B Vector 寄存器可容纳 128 个 BF16 元素。`DIST_UNPACK4_B8` 每次从 UB 紧凑读取 64 B，即 128 个逻辑 INT4，并将每个 byte 的 low/high nibble 展开到 Cast 可直接消费的位置。Load 已完成 nibble 展开后，一次 `Cast<bfloat16_t, int4x2_t>` 就能产生顺序连续的 BF16 寄存器，不再需要额外交织。

`trace` case 中每个 AIV 处理 40960 个逻辑 INT4，即 20480 B 输入和 81920 B 输出，共执行 320 组转换。最终每组仅保留一次 UNPACK4 Load、一次 Cast 和一次普通 B16 Store。

## 优化路径与目录

| 阶段 | 对应技巧 | 本阶段变化 |
| --- | --- | --- |
| `0_unpack4_load` | UNPACK4 Load 展开 packed nibble | 以专用 Load 作为起点，消除显式 nibble 拆分 |
| `1_single_cast` | 单路 Cast 得到连续结果 | INT4 直接转换为 BF16，移除 HALF 中间类型 |
| `2_separate_views` | 分离 packed-byte 与 logical-element view | 循环边界改由 packed 输入 byte 数推导 |
| `3_b8_predicate` | B8 predicate view 服务 packed Cast | 明确 predicate 粒度并将 mask 放在热循环外复用 |

四个目录都包含完整的 VF、kernel、数据搬运和 host 代码，可独立构建和验证；公共目录只提供 ACL 生命周期管理及输入输出等辅助能力。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DNPU_ARCH=dav-3510 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vf_data_transform_int4_to_bf16_tutorial -j4
```

## 基准实现：0_unpack4_load

源码：[0_unpack4_load/unpack4_load.asc](./0_unpack4_load/unpack4_load.asc)，目标：`vf_data_transform_int4_to_bf16_0_unpack4_load`。

起点直接使用专用 Load：

```cpp
AscendC::Reg::LoadAlign<int4x2_t, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(
    inputReg, input + packedOffset);
```

一次 Load 读取 64 B packed input，并把 128 个有符号 nibble 放到数值转换可以直接消费的位置，因此热循环中不需要 `ShiftLeft`、`ShiftRight` 或 `Select` 拆分 low/high nibble。为了单独观察 Load 侧的变化，本阶段仍通过 INT4→HALF→BF16 两级 Cast 生成完整 BF16 输出。

CANNsim 实测为 699 cycles，有效输入吞吐为 29.30 B/cycle，有效输出吞吐为 117.20 B/cycle。

## 优化阶段一：1_single_cast

源码：[1_single_cast/single_cast.asc](./1_single_cast/single_cast.asc)，目标：`vf_data_transform_int4_to_bf16_1_single_cast`。

UNPACK4 生成的寄存器可以直接参与 INT4→BF16 Cast：

```cpp
AscendC::Reg::Cast<bfloat16_t, int4x2_t, kInt4ToBf16CastTraitZero>(
    outputReg, inputReg, fullPackedMask);
```

Cast 使用 `RegLayout::ZERO`、`MaskMergeMode::ZEROING` 和 `CAST_RINT`。这一步去掉 HALF 中间寄存器及第二次 Cast；结果已经按 `x[0], x[1], ...` 连续排列，随后使用一次 `DIST_NORM_B16` Store 写出。

CANNsim 降至 453 cycles，相比阶段 0 减少 35.2%，有效输出吞吐提升至 180.84 B/cycle。直接 Cast 是本路径的主要性能收益。

## 优化阶段二：2_separate_views

源码：[2_separate_views/separate_views.asc](./2_separate_views/separate_views.asc)，目标：`vf_data_transform_int4_to_bf16_2_separate_views`。

packed INT4 输入和 BF16 输出的计数单位不同：输入 GM/UB Tensor 以物理 byte 表示，输出 Tensor 以逻辑元素表示。进入 VF 后才把 packed UB 基址解释为 `int4x2_t*`，循环次数由 packed input view 推导：

```cpp
const uint32_t computePackedBytes = static_cast<uint32_t>(inputUb.Layout().Size());
const uint16_t groupTimes =
    static_cast<uint16_t>(computePackedBytes / kPackedBytesPerGroup);
```

这样可以避免把逻辑 INT4 数量误作物理 byte 数量。该修改不增加 Vector 指令，也不改变最终 Load/Cast/Store 链，CANNsim 仍为 453 cycles。

## 优化阶段三：3_b8_predicate

源码：[3_b8_predicate/b8_predicate.asc](./3_b8_predicate/b8_predicate.asc)，目标：`vf_data_transform_int4_to_bf16`。

packed INT4 的一个物理单元是一个 byte，因此 Cast 必须使用 B8 predicate view：

```cpp
AscendC::Reg::MaskReg fullPackedMask =
    AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
```

包含 `int4x2_t` 的 Cast 固定采用 `ZEROING`，不使用不支持的 `MERGING`。mask 在 VF 循环外创建，各组重复使用。该阶段的热路径已收敛为一次 UNPACK4 Load、一次 Cast 和一次 Store。

CANNsim 为 453 cycles。前一阶段在 `-O3` 下已经得到等价的循环不变量处理，因此显式外提不改变 cycle；它的作用是固定正确的 predicate 语义，使代码可以直接复用。

## 性能结果

以下数据由 CANNsim 在 Ascend 950 上使用 `--case trace` 采集。每个 AIV 处理一个 40960 元素 tile；`VF cycles` 取 core 0 业务 kernel 的 `avg_cycles`。

| 阶段 | VF cycles | 相比上一步 | 输入 B/cycle | 输出 B/cycle |
| --- | ---: | ---: | ---: | ---: |
| `0_unpack4_load` | 699 | 起点 | 29.30 | 117.20 |
| `1_single_cast` | 453 | 减少 35.2% | 45.21 | 180.84 |
| `2_separate_views` | 453 | 持平 | 45.21 | 180.84 |
| `3_b8_predicate` | 453 | 持平 | 45.21 | 180.84 |

从 UNPACK4 起点到最终阶段，VF cycles 从 699 降至 453，局部数据通路提速 1.54 倍。最终阶段整体 Vec IPC 为 0.87，主数据通路已收敛为每 128 个逻辑元素一次 Load、一次 Cast 和一次 Store。后两步不改变指令数量，但分别固定了 Tensor 计数单位和 predicate 粒度，避免集成时因视图错误破坏结果或越界访问。

## 运行与验证

运行最终阶段：

```bash
./build/Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/\
int4_to_bf16/3_b8_predicate/vf_data_transform_int4_to_bf16 --case all
```

数据生成覆盖 `-8..7` 以及 nibble 边界。验证通过时，`full`、`tail` 和 `perf` 均输出 `status=PASS`，并包含 `bin_match=true`。

## 总结与复用建议

该转换的性能核心是 UNPACK4 Load 和单路 INT4→BF16 Cast。复用时必须同时确认 packed byte 中 low/high nibble 的顺序、signed INT4 的符号扩展规则、输入输出 Tensor 的计数单位和 B8 predicate view。若输入打包格式不同，应先重新验证 UNPACK4 的 lane 顺序，再调整 golden 与数据通路。
