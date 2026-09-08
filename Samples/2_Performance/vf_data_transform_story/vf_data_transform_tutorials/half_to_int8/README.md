# HALF 转 INT8：DINTLV 与互补 lane 合并

## 引言

HALF 转 INT8 常用于量化输出和算子边界的数据类型转换。本教程使用 `CAST_RINT` 和 `NO_SAT`，逐元素完成：

```text
y[i] = int8_rint(x[i])
```

优化目标是减少窄化转换前后的寄存器整理和 UB 写回。四个阶段依次使用 DINTLV 载入、ZERO/ONE 互补 Cast、原位 Or 和连续 B8 Store，最终形成下面的数据通路：

```text
HALF ND
  -> LoadAlign(DIST_DINTLV_B16)
  -> Cast(ZERO) + Cast(ONE)
  -> Or
  -> StoreAlign(DIST_NORM_B8)
  -> INT8 ND
```

每个阶段都是完整、独立的可执行样例。为便于比较，四个版本使用相同的多核切分、32768 元素 tile 和双缓冲流水，只调整 VF 热循环中的数据通路。

## 支持范围

- 硬件：Ascend 950，编译架构为 `dav-3510`。
- 软件：CANN 9.2.0，Release 构建，ASC 使用 `-O3`。
- 输入/输出：连续二维 ND，输入为 HALF，输出为 INT8。
- 转换语义：`CAST_RINT`、`NO_SAT`、`ZEROING`。输入数据应处于 INT8 可表示范围内。
- 测试用例：`full`、`tail`、`perf`、`trace`。当运行环境有 64 个 AIV 时，对应 shape 分别为 `(2048, 4096)`、`(2049, 4099)`、`(4096, 4096)` 和 `(512, 4096)`。
- 尾块处理：kernel 将有效元素补齐到 256 元素组后进入 VF，写回 GM 时只复制有效范围。

## 硬件架构基础

Ascend 950 的 Vector 寄存器宽 256 B，一个寄存器可容纳 128 个 HALF 或 256 个 INT8。HALF 到 INT8 的位宽比为 2:1，因此每轮以 256 个元素为一组：输入占两个 B16 寄存器，输出正好占一个 B8 寄存器。

`DIST_DINTLV_B16` 一次读取连续 256 个 HALF，并按偶数、奇数位置生成两个 B16 寄存器。若两路 Cast 分别使用 `RegLayout::ZERO` 和 `RegLayout::ONE`，窄化结果会落到一个 B8 视图中的互补 lane；两路结果按位合并后即可连续写出。

`trace` 用例中每个 AIV 处理一个 32768 元素 tile，共 128 个 256 元素组。对选定的数据通路，最终阶段的结构下限是 128 次双输出 Load、256 次 Cast、128 次 Or 和 128 次 Store。该下限用于判断多余的数据整理和写回是否已经消除，不等同于脱离指令时延和流水约束的 cycle 理论值。

## 目录与构建目标

| 阶段 | 主要变化 | 构建目标 |
| --- | --- | --- |
| `0_dintlv_load` | DINTLV 双输出载入基线 | `vf_data_transform_half_to_int8_0_dintlv_load` |
| `1_complementary_cast` | ZERO/ONE 互补 Cast | `vf_data_transform_half_to_int8_1_complementary_cast` |
| `2_or_merge` | 原位 Or 合并 | `vf_data_transform_half_to_int8_2_or_merge` |
| `3_contiguous_store` | 单次连续 B8 Store | `vf_data_transform_half_to_int8` |

四个目录中的 `.asc` 文件都包含完整的 VF、kernel、数据搬运和 host 调用代码，不引用其他阶段的实现。公共目录只提供 ACL 运行时和数据文件读写等辅助能力。

在仓库根目录构建全部阶段：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DNPU_ARCH=dav-3510 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vf_data_transform_half_to_int8_tutorial -j4
```

## 基准实现：0_dintlv_load

源码：[0_dintlv_load/dintlv_load.asc](./0_dintlv_load/dintlv_load.asc)。

基线首先使用 DINTLV B16 Load，把连续输入拆成偶数、奇数两路：

```cpp
LoadAlign<half, LoadDist::DIST_DINTLV_B16>(
    inputReg0, inputReg1, input + groupOffset);
```

这一阶段尚未使用互补 Cast，两路都采用 ZERO layout。窄化结果的有效 B8 lane 排布相同，必须分别 Pack，再通过 Interleave 恢复原始元素顺序：

```cpp
Cast<int8_t, half, kHalfToInt8CastTrait>(outputReg0, inputReg0, fullB16Mask);
Cast<int8_t, half, kHalfToInt8CastTrait>(outputReg1, inputReg1, fullB16Mask);
Pack<uint8_t, uint16_t, HighLowPart::LOWEST>(packedOutputReg0, outputReg0B16);
Pack<uint8_t, uint16_t, HighLowPart::LOWEST>(packedOutputReg1, outputReg1B16);
Interleave(interleavedOutputReg0, interleavedOutputReg1, packedOutputReg0, packedOutputReg1);
```

为把最后的写回优化留到阶段三，前三个阶段都使用低半区和高半区的两次 masked Store。基线每个 tile 包含 256 次 Pack、128 次 Interleave 和 256 次 Store，CANNsim 测得 780 cycles；额外的寄存器整理是首要瓶颈。

## 优化阶段一：1_complementary_cast

源码：[1_complementary_cast/complementary_cast.asc](./1_complementary_cast/complementary_cast.asc)。

第二路 Cast 改用 `RegLayout::ONE`，与第一路的 `RegLayout::ZERO` 形成互补 B8 lane。两路都保持 `NO_SAT`、`ZEROING` 和 `CAST_RINT`：

```cpp
constexpr CastTrait kHalfToInt8CastTrait = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait kHalfToInt8CastTraitOne = {
    RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
```

ZEROING 保证每个 B8 lane 最多只有一路非零，因此本阶段先用 B8 Add 得到完整输出。这里不会发生算术溢出：每个 lane 都是转换结果与零相加。写回方式与基线保持一致，以便单独观察互补 layout 消除 Pack 和 Interleave 的收益。

```cpp
Cast<int8_t, half, kHalfToInt8CastTrait>(outputReg0, inputReg0, fullB16Mask);
Cast<int8_t, half, kHalfToInt8CastTraitOne>(outputReg1, inputReg1, fullB16Mask);
Add<int8_t>(mergedOutputReg, outputReg0, outputReg1, fullB8Mask);
```

CANNsim 为 445 cycles，较基线减少 42.9%，等效提速 1.75 倍。Pack 和 Interleave 已经消失，当前主要开销转为算术合并以及同一地址上的两次分区 Store。

## 优化阶段二：2_or_merge

源码：[2_or_merge/or_merge.asc](./2_or_merge/or_merge.asc)。

互补 lane 的合并本质是按位拼接。将 Add 改为 B8 Or，并原位复用第一路输出寄存器：

```cpp
Or(outputReg0, outputReg0, outputReg1, fullB8Mask);
```

这一写法直接表达“有效位互不重叠”的数据关系，同时不再需要第三个合并结果寄存器。该阶段仍保留两次分区 Store，避免混入连续写回带来的收益。

CANNsim 为 433 cycles，较上一阶段减少 2.7%，等效提速 1.03 倍。热循环中的合并指令已收敛为每组一次 Or，瓶颈进一步集中到重复 Store。

## 优化阶段三：3_contiguous_store

源码：[3_contiguous_store/contiguous_store.asc](./3_contiguous_store/contiguous_store.asc)。

Or 之后，一个 B8 寄存器已经按 `y[0] ... y[255]` 连续排布。输出地址按 256 B 递增且满足对齐要求，可以用完整 B8 mask 一次写出：

```cpp
StoreAlign<int8_t, StoreDist::DIST_NORM_B8>(
    output + groupOffset, outputReg0, fullB8Mask);
```

这一步把每组两次 masked Store 减为一次连续 Store，同时删除低半区和高半区 mask。CANNsim 为 315 cycles，较上一阶段减少 27.3%，等效提速 1.37 倍。最终热循环达到前述结构下限，继续优化时应优先考虑与上下游计算融合，减少 UB 或 GM 往返。

## 性能结果

性能数据使用 CANN 9.2.0 的 Ascend950PR_9589 CAMODEL V100 采集。构建类型为 Release，执行 `trace` 用例，每个 AIV 处理 32768 个 HALF。`VF cycles` 取业务 kernel core 0 的 `avg_cycles`；有效 B/cycle 只按本 tile 的有效输入、输出字节数计算。

```bash
bash Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/scripts/profile_tutorial.sh \
  cannsim half_to_int8 build /tmp/vf_data_transform_cannsim/half_to_int8 trace
```

| 阶段 | VF cycles | 相对上一阶段 | 输入 B/cycle | 输出 B/cycle | 每 tile 的主要冗余 |
| --- | ---: | ---: | ---: | ---: | --- |
| `0_dintlv_load` | 780 | 基线 | 84.02 | 42.01 | 256 Pack + 128 Interleave |
| `1_complementary_cast` | 445 | 1.75x | 147.27 | 73.64 | Add 合并、256 Store |
| `2_or_merge` | 433 | 1.03x | 151.35 | 75.68 | 256 Store |
| `3_contiguous_store` | 315 | 1.37x | 208.05 | 104.03 | 无额外整理，128 Store |

从 DINTLV 基线到最终版本，VF cycles 从 780 降到 315，减少 59.6%，等效提速 2.48 倍。最终阶段每个 256 元素组只保留一次双输出 Load、两次 Cast、一次 Or 和一次 Store，与本教程的数据通路结构下限一致。当前没有可直接对照且转换语义完全相同的加速库接口，因此不列不等价的库性能数据。

## 运行与验证

四个阶段均支持 `full`、`tail`、`perf`、`trace` 和 `all`。以最终阶段为例：

```bash
./build/Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/half_to_int8/3_contiguous_store/vf_data_transform_half_to_int8 --case all
```

程序生成包含整数、正负小数和 `.5` 边界值的固定输入，使用 `numpy.rint` 生成 golden。host 对有效输出和带 guard 的完整存储区逐 bit 比较；`tail` 用例同时覆盖非整 tile 和不足 256 元素的最后一组。验证成功时输出包含：

```text
[HOST][tail] status=PASS ... bin_match=true
```

## 总结与复用建议

HALF 到 INT8 的四步优化围绕同一条 lane 数据流展开：DINTLV 将连续输入拆成偶、奇两路，ZERO/ONE Cast 让窄化结果落入互补 B8 lane，Or 用一次按位操作完成合并，连续 B8 Store 再把完整寄存器一次写回。

复用代码时需要同时核对输入分组、Cast layout、舍入与饱和语义以及输出对齐条件。若输入可能超出 INT8 范围，应按业务要求改用相应的饱和策略并重新生成 golden；若下游计算能够直接消费 INT8 寄存器，融合计算并省去中间写回通常比继续压缩单次 Cast 更有价值。
