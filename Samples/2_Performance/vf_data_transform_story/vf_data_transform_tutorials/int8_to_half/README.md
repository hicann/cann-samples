# INT8 转 HALF：紧凑载入与交织写出

## 引言

INT8 转 HALF 是反量化数据通路中的 1:2 扩展转换。一个 B8 Vector 寄存器可容纳 256 个 INT8，而一个 B16 寄存器只能容纳 128 个 HALF，因此扩展后需要形成两路结果并恢复连续顺序。

四个阶段依次使用紧凑 B8 Load、ZERO/ONE 互补 widening Cast、双输入 INTLV B16 Store 和共享 B8 predicate，最终形成下面的数据通路：

```text
INT8 ND
  -> LoadAlign(DIST_NORM)
  -> Cast(ZERO) + Cast(ONE)
  -> StoreAlign(DIST_INTLV_B16)
  -> HALF ND
```

每个阶段都是完整、独立的可执行样例。四个版本使用相同的多核切分、32768 元素 tile 和双缓冲流水，只调整 VF 热循环中的数据通路。

## 支持范围

- 硬件：Ascend 950，编译架构为 `dav-3510`。
- 软件：CANN 9.2.0，Release 构建，ASC 使用 `-O3`。
- 输入/输出：连续二维 INT8 ND 输入，HALF ND 输出。所有 INT8 均可由 HALF 精确表示。
- 转换语义：`CAST_RINT`、`ZEROING`；INT8 到 HALF 不产生实际小数舍入。
- 测试用例：`full`、`tail`、`perf`、`trace`。当运行环境有 64 个 AIV 时，对应 shape 分别为 `(2048, 4096)`、`(2049, 4099)`、`(4096, 4096)` 和 `(512, 4096)`。
- 尾块处理：kernel 将有效元素补齐到 256 元素组后进入 VF，写回 GM 时只复制有效范围。

## 硬件架构基础

Ascend 950 的 Vector 寄存器宽 256 B，一个寄存器可容纳 256 个 INT8 或 128 个 HALF。普通 B8 Load 一次保留 256 个紧凑输入。两路 Cast 分别使用 `RegLayout::ZERO` 和 `RegLayout::ONE`，从同一个 B8 寄存器取得 `x[0], x[2], ...` 和 `x[1], x[3], ...`，各生成一个 HALF 寄存器。

两路结果在寄存器中分别连续，但尚未恢复输入的逐元素顺序。`DIST_INTLV_B16` Store 可以直接接收这两个寄存器，在写回时得到 `x[0], x[1], x[2], x[3], ...`，无需先执行寄存器 Interleave。

`trace` 用例中每个 AIV 处理一个 32768 元素 tile，共 128 个 256 元素组。最终阶段的主指令规模为 128 次 Load、256 次 Cast 和 128 次双输入 Store。这个规模用于判断显式 UnPack、Interleave 和重复写回是否已经消除，不代表不同 Store 模式具有相同单条时延。

## 目录与构建目标

| 阶段 | 主要变化 | 构建目标 |
| --- | --- | --- |
| `0_compact_b8_load` | 紧凑 B8 Load 基线 | `vf_data_transform_int8_to_half_0_compact_b8_load` |
| `1_complementary_cast` | ZERO/ONE 两路 widening Cast | `vf_data_transform_int8_to_half_1_complementary_cast` |
| `2_intlv_b16_store` | 双输入 INTLV B16 Store | `vf_data_transform_int8_to_half_2_intlv_b16_store` |
| `3_b8_cast_mask` | 两路 Cast 共享 B8 predicate | `vf_data_transform_int8_to_half` |

四个目录中的 `.asc` 文件都包含完整的 VF、kernel、数据搬运和 host 调用代码，不引用其他阶段的实现。公共目录只提供 ACL 运行时和数据文件读写等辅助能力。

在仓库根目录构建全部阶段：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DNPU_ARCH=dav-3510 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vf_data_transform_int8_to_half_tutorial -j4
```

## 基准实现：0_compact_b8_load

源码：[0_compact_b8_load/compact_b8_load.asc](./0_compact_b8_load/compact_b8_load.asc)。

基线使用普通 B8 Load，一次读入 256 个连续 INT8，并保持输入的紧凑排布：

```cpp
LoadAlign<int8_t, LoadDist::DIST_NORM>(inputReg, input + groupOffset);
```

本阶段尚未使用 ZERO/ONE 互补 Cast，因此通过两次 UnPack 显式取得低半区和高半区，再分别转换、普通写出：

```cpp
UnPack<int16_t, int8_t, HighLowPart::LOWEST>(unpackedInputReg0, inputReg);
UnPack<int16_t, int8_t, HighLowPart::HIGHEST>(unpackedInputReg1, inputReg);
Cast<half, int8_t, kInt8ToHalfCastTraitZero>(outputReg0, inputReg0B8, fullB8Mask);
Cast<half, int8_t, kInt8ToHalfCastTraitZero>(outputReg1, inputReg1B8, fullB8Mask);
```

每个 tile 包含 256 次 UnPack 和 256 次普通 Store。CANNsim 测得 446 cycles；这些显式拆分操作是后续互补 Cast 要消除的对象。

## 优化阶段一：1_complementary_cast

源码：[1_complementary_cast/complementary_cast.asc](./1_complementary_cast/complementary_cast.asc)。

两路 Cast 直接读取同一个紧凑 B8 寄存器。ZERO layout 选择偶数元素，ONE layout 选择奇数元素：

```cpp
Cast<half, int8_t, kInt8ToHalfCastTraitZero>(outputReg0, inputReg, fullB8Mask);
Cast<half, int8_t, kInt8ToHalfCastTraitOne>(outputReg1, inputReg, fullB8Mask);
```

UnPack 已经删除。为把交织写回留到下一阶段，本阶段先在寄存器中执行 Interleave，再用两次普通 B16 Store 写出：

```cpp
Interleave(interleavedOutputReg0, interleavedOutputReg1, outputReg0, outputReg1);
StoreAlign<half, StoreDist::DIST_NORM_B16>(
    output + groupOffset, interleavedOutputReg0, fullB16Mask);
StoreAlign<half, StoreDist::DIST_NORM_B16>(
    output + groupOffset + kElementsPerGroup / 2U, interleavedOutputReg1, fullB16Mask);
```

CANNsim 为 574 cycles，比基线增加 28.7%。这是一个依赖尚未闭合的过渡阶段：互补 Cast 只有与 INTLV Store 配合，才能避免新增的 Interleave 和第二次 Store，因此本阶段用于解释 lane 关系，不适合单独集成。

## 优化阶段二：2_intlv_b16_store

源码：[2_intlv_b16_store/intlv_b16_store.asc](./2_intlv_b16_store/intlv_b16_store.asc)。

`DIST_INTLV_B16` Store 直接接收偶数和奇数 HALF 寄存器，在写回阶段恢复连续顺序：

```cpp
StoreAlign<half, StoreDist::DIST_INTLV_B16>(
    output + groupOffset, outputReg0, outputReg1, fullB16Mask);
```

这一步同时删除每组一次 Interleave 和第二次普通 Store，主数据通路收敛为一次 Load、两次 Cast、一次 Store。CANNsim 降至 458 cycles，较上一阶段减少 20.2%，等效提速 1.25 倍。交织 Store 的单条执行时间高于普通 Store，因此指令数减少并不保证相对所有过渡实现都更快。

## 优化阶段三：3_b8_cast_mask

源码：[3_b8_cast_mask/b8_cast_mask.asc](./3_b8_cast_mask/b8_cast_mask.asc)。

两次 widening Cast 的源操作数都是紧凑 B8 视图，需要覆盖 256 个 source lane。若误用 B16 mask，只会覆盖一半 source lane，使奇数输出错误清零。最终阶段在循环外创建一个 `B8 ALL` mask，并由两路 Cast 共享：

```cpp
MaskReg fullB8Mask = CreateMask<int8_t, MaskPattern::ALL>();

Cast<half, int8_t, kInt8ToHalfCastTraitZero>(outputReg0, inputReg, fullB8Mask);
Cast<half, int8_t, kInt8ToHalfCastTraitOne>(outputReg1, inputReg, fullB8Mask);
```

CANNsim 仍为 458 cycles。当前 `-O3` 已对前一阶段循环内的固定 predicate 做出等价处理，因此本步没有额外 cycle 收益；显式使用 B8 mask 仍是保证 widening Cast 正确性和可复用性的必要条件。

## 性能结果

性能数据使用 CANN 9.2.0 的 Ascend950PR_9589 CAMODEL V100 统一采集。构建类型为 Release，执行 `trace` 用例，每个 AIV 处理 32768 个 INT8。`VF cycles` 取业务 kernel core 0 的 `avg_cycles`；有效 B/cycle 只按本 tile 的有效输入、输出字节数计算。

```bash
bash Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/scripts/profile_tutorial.sh \
  cannsim int8_to_half build /tmp/vf_data_transform_cannsim/int8_to_half trace
```

| 阶段 | VF cycles | 相对上一阶段 | 输入 B/cycle | 输出 B/cycle | 每 tile 的主要冗余 |
| --- | ---: | ---: | ---: | ---: | --- |
| `0_compact_b8_load` | 446 | 基线 | 73.47 | 146.94 | 256 UnPack + 256 Store |
| `1_complementary_cast` | 574 | 0.78x | 57.09 | 114.17 | 128 Interleave + 256 Store |
| `2_intlv_b16_store` | 458 | 1.25x | 71.55 | 143.09 | B8 mask 在循环内创建 |
| `3_b8_cast_mask` | 458 | 1.00x | 71.55 | 143.09 | 目标数据通路 |

表中全部数值来自当前四个阶段随源码保留的同批 CANNsim 报告。另一次单版本采集得到 439 cycles，但采集批次不同；为保证阶段加速比口径一致，此处统一采用可从阶段报告直接复核的 458 cycles，不混用跨批次数据。

从依赖完整性看，阶段一的性能回退来自“互补 Cast 已启用、INTLV Store 尚未启用”的寄存器 Interleave 过渡。阶段二闭合这项依赖后，cycle 从 574 降至 458。最终版本相对紧凑 Load 过渡基线仍增加 2.7%，说明当前独立转换在该 CAMODEL 上受 INTLV Store 时延限制。实际集成时，若两路 HALF 能直接进入下游计算并省去交织写回，紧凑 Load 和互补 Cast 的指令规模优势才更容易转化为端到端收益。

## 运行与验证

四个阶段均支持 `full`、`tail`、`perf`、`trace` 和 `all`。以最终阶段为例：

```bash
./build/Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/int8_to_half/3_b8_cast_mask/vf_data_transform_int8_to_half --case all
```

程序生成固定 INT8 输入和 HALF golden，对输出 raw bits 逐 bit 比较，并检查带 guard 的完整输出区。`tail` 用例同时覆盖非整 tile 和不足 256 元素的最后一组。验证成功时输出包含：

```text
[HOST][tail] status=PASS ... bin_match=true
```

## 总结与复用建议

INT8 到 HALF 的四步路径围绕 1:2 扩展排布展开：普通 B8 Load 保持输入紧凑，ZERO/ONE Cast 分别生成偶数和奇数 HALF，双输入 INTLV B16 Store 在写回时恢复连续顺序，B8 predicate 则保证两次 widening Cast 覆盖完整 source lane。

复用时需要同时核对 Load 排布、Cast layout、predicate 的源数据粒度和 Store 的交织顺序，不能只替换输入输出类型。若下游只接受连续 HALF，应在目标硬件和真实 shape 上比较 INTLV Store 与简单 UNPACK 路径；若下游可直接消费两路寄存器，应优先融合后续计算并删除中间写回。
