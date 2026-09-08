# FP32 转 FP8 E4M3FN：DINTLV 与 B16 pack 写出

## 引言

FP32 转 FP8 E4M3FN 是低精度推理和通信压缩中常见的 4:1 窄化转换。本教程使用 `CAST_RINT`、`NO_SAT` 和 `ZEROING` 完成转换，重点减少窄化前后的寄存器整理和 UB 写回。

四个阶段依次使用 DINTLV B32 载入、ZERO/TWO 互补 Cast、B16 pack Store 和共享 B8 mask，最终形成下面的数据通路：

```text
FP32 ND
  -> LoadAlign(DIST_DINTLV_B32)
  -> Cast(ZERO) + Cast(TWO)
  -> Or
  -> StoreAlign(DIST_PACK_B16)
  -> FP8 E4M3FN raw bytes
```

每个阶段都是完整、独立的可执行样例。四个版本使用相同的多核切分、20480 元素 tile 和双缓冲流水，只调整 VF 热循环中的数据通路。

## 支持范围

- 硬件：Ascend 950，编译架构为 `dav-3510`。
- 软件：CANN 9.2.0，Release 构建，ASC 使用 `-O3`。
- 输入/输出：连续二维 FP32 ND 输入，FP8 E4M3FN raw byte 输出。
- 转换语义：`CAST_RINT`、`NO_SAT`、`ZEROING`。修改舍入或饱和规则后需要重新生成 golden。
- 测试用例：`full`、`tail`、`perf`、`trace`。当运行环境有 64 个 AIV 时，对应 shape 分别为 `(1280, 4096)`、`(1281, 4099)`、`(2560, 4096)` 和 `(320, 4096)`。
- 尾块处理：kernel 将有效元素补齐到 128 元素组后进入 VF，写回 GM 时只复制有效范围。

## 硬件架构基础

Ascend 950 的 Vector 寄存器宽 256 B，一个寄存器可容纳 64 个 FP32 或 256 个 B8。FP32 到 FP8 的位宽比为 4:1，因此每轮以 128 个元素为一组：输入占两个 B32 寄存器，输出只占 128 B。

`DIST_DINTLV_B32` 一次读取连续 128 个 FP32，并把两个 64 元素通道放入两个 B32 寄存器。两路 Cast 分别采用 `RegLayout::ZERO` 和 `RegLayout::TWO` 后，有效 FP8 会落在一个 B8 视图的互补 lane。Or 可以直接合并两路，`DIST_PACK_B16` Store 再将相邻 B16 中的低 B8 紧排为连续输出。

`trace` 用例中每个 AIV 处理一个 20480 元素 tile，共 160 个 128 元素组。最终阶段的主指令规模为 160 次双输出 Load、320 次 Cast、160 次 Or 和 160 次 Store。这个规模用于判断显式 Pack、Interleave 和重复写回是否已经消除，不等同于脱离指令时延和流水约束的 cycle 理论值。

## 目录与构建目标

| 阶段 | 主要变化 | 构建目标 |
| --- | --- | --- |
| `0_dintlv_b32_load` | DINTLV B32 双输出载入基线 | `vf_data_transform_fp32_to_fp8_0_dintlv_b32_load` |
| `1_complementary_cast` | ZERO/TWO 互补 Cast 与 Or | `vf_data_transform_fp32_to_fp8_1_complementary_cast` |
| `2_pack_b16_store` | B16 pack Store 连续写出 | `vf_data_transform_fp32_to_fp8_2_pack_b16_store` |
| `3_shared_b8_mask` | Or 和 pack Store 共享 B8 mask | `vf_data_transform_fp32_to_fp8` |

四个目录中的 `.asc` 文件都包含完整的 VF、kernel、数据搬运和 host 调用代码，不引用其他阶段的实现。公共目录只提供 ACL 运行时和数据文件读写等辅助能力。

在仓库根目录构建全部阶段：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DNPU_ARCH=dav-3510 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vf_data_transform_fp32_to_fp8_tutorial -j4
```

## 基准实现：0_dintlv_b32_load

源码：[0_dintlv_b32_load/dintlv_b32_load.asc](./0_dintlv_b32_load/dintlv_b32_load.asc)。

基线首先使用 DINTLV B32 Load，一次把连续输入拆成两个 64 元素通道：

```cpp
LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
    evenInputReg, oddInputReg, input + groupOffset);
```

本阶段尚未使用互补 Cast，两路都采用 ZERO layout。两路转换结果中的有效 byte 排布相同，需要各执行两级 Pack，再通过 Interleave 恢复原始元素顺序：

```cpp
Cast<fp8_e4m3fn_t, float, kFp32ToFp8CastTrait>(evenOutputReg, evenInputReg, fullB32Mask);
Cast<fp8_e4m3fn_t, float, kFp32ToFp8CastTrait>(oddOutputReg, oddInputReg, fullB32Mask);
Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(evenPackedB16, evenOutputB32);
Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(oddPackedB16, oddOutputB32);
Pack<uint8_t, uint16_t, HighLowPart::LOWEST>(evenPackedB8, evenPackedB16);
Pack<uint8_t, uint16_t, HighLowPart::LOWEST>(oddPackedB8, oddPackedB16);
Interleave(interleavedOutputReg0, interleavedOutputReg1, evenPackedB8, oddPackedB8);
```

每个 tile 因而包含 640 次 Pack 和 160 次 Interleave。CANNsim 测得 1250 cycles，显式寄存器整理是首要瓶颈。

## 优化阶段一：1_complementary_cast

源码：[1_complementary_cast/complementary_cast.asc](./1_complementary_cast/complementary_cast.asc)。

第二路 Cast 改用 `RegLayout::TWO`，与第一路的 `RegLayout::ZERO` 形成互补 B8 lane。两路均保持 `NO_SAT`、`ZEROING` 和 `CAST_RINT`：

```cpp
constexpr CastTrait kFp32ToFp8CastTrait = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait kFp32ToFp8CastTraitTwo = {
    RegLayout::TWO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
```

ZEROING 保证每个有效 B8 lane 只由一路提供数据，可以直接用 Or 合并。由于本阶段尚未引入 pack Store，合并结果仍通过一次显式 B16→B8 Pack 和普通 B8 Store 写出：

```cpp
Or(mergedOutputReg, evenOutputB8, oddOutputB8, fullB8Mask);
Pack<uint8_t, uint16_t, HighLowPart::LOWEST>(packedOutputReg, mergedOutputB16);
StoreAlign<uint8_t, StoreDist::DIST_NORM_B8>(
    output + groupOffset, packedOutputReg, halfB8Mask);
```

CANNsim 为 379 cycles，较基线减少 69.7%，等效提速 3.30 倍。ZERO/TWO layout 与 Or 共同消除了两级双路 Pack 和 Interleave，热循环只剩最后一级显式压缩。

## 优化阶段二：2_pack_b16_store

源码：[2_pack_b16_store/pack_b16_store.asc](./2_pack_b16_store/pack_b16_store.asc)。

Or 之后的有效 byte 位于相邻 B16 的低 8 bit。`DIST_PACK_B16` Store 可以在写回时完成紧排，不需要独立的 Pack 寄存器：

```cpp
StoreAlign<uint8_t, StoreDist::DIST_PACK_B16>(
    output + groupOffset, mergedOutputReg, storeB8Mask);
```

该阶段把每组的显式 Pack 删除，主数据通路已经收敛到双输出 Load、两次 Cast、一次 Or 和一次 Store。CANNsim 仍为 379 cycles，说明当前 tile 上显式 Pack 与 pack Store 的差异没有改变关键路径；但中间寄存器和显式整理指令已经消除，代码可直接作为 B16 pack 写出模板复用。

## 优化阶段三：3_shared_b8_mask

源码：[3_shared_b8_mask/shared_b8_mask.asc](./3_shared_b8_mask/shared_b8_mask.asc)。

Or 和 B16 pack Store 都消费合并前的完整 B8 lane，因此可以共享同一个 `B8 ALL` mask。B32 Cast mask 与 B8 mask 都是循环不变量，应在 VF 循环前创建：

```cpp
MaskReg fullB32Mask = CreateMask<float, MaskPattern::ALL>();
MaskReg fullB8Mask = CreateMask<uint8_t, MaskPattern::ALL>();

Or(mergedOutputReg, evenOutputB8, oddOutputB8, fullB8Mask);
StoreAlign<uint8_t, StoreDist::DIST_PACK_B16>(
    output + groupOffset, mergedOutputReg, fullB8Mask);
```

CANNsim 仍为 379 cycles。当前 `-O3` 已对前一阶段的固定 predicate 做出等价处理，因此这一改动没有额外 cycle 收益；显式共享仍能准确表达 predicate 粒度，避免复用时误把 128 B 输出长度当成 pack Store 的源 mask 范围。

## 性能结果

性能数据使用 CANN 9.2.0 的 Ascend950PR_9589 CAMODEL V100 采集。构建类型为 Release，执行 `trace` 用例，每个 AIV 处理 20480 个 FP32。`VF cycles` 取业务 kernel core 0 的 `avg_cycles`；有效 B/cycle 只按本 tile 的有效输入、输出字节数计算。

```bash
bash Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/scripts/profile_tutorial.sh \
  cannsim fp32_to_fp8 build /tmp/vf_data_transform_cannsim/fp32_to_fp8 trace
```

| 阶段 | VF cycles | 相对上一阶段 | 输入 B/cycle | 输出 B/cycle | 每 tile 的主要冗余 |
| --- | ---: | ---: | ---: | ---: | --- |
| `0_dintlv_b32_load` | 1250 | 基线 | 65.54 | 16.38 | 640 Pack + 160 Interleave |
| `1_complementary_cast` | 379 | 3.30x | 216.15 | 54.04 | 160 个显式 Pack |
| `2_pack_b16_store` | 379 | 1.00x | 216.15 | 54.04 | mask 在循环内分别创建 |
| `3_shared_b8_mask` | 379 | 1.00x | 216.15 | 目标数据通路 |

从 DINTLV 基线到最终版本，VF cycles 从 1250 降到 379，减少 69.7%，等效提速 3.30 倍。阶段二和阶段三的代码变化没有在当前 CAMODEL 上转化为额外 cycle 收益，但分别消除了显式 Pack，并明确了 pack Store 的 predicate 视图。最终阶段达到本教程的主指令规模目标。当前没有可直接对照且转换语义完全相同的加速库接口，因此不列不等价的库性能数据。

## 运行与验证

四个阶段均支持 `full`、`tail`、`perf`、`trace` 和 `all`。以最终阶段为例：

```bash
./build/Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/fp32_to_fp8/3_shared_b8_mask/vf_data_transform_fp32_to_fp8 --case all
```

程序生成固定 FP32 输入和 E4M3FN golden，对有效输出逐 bit 比较，并检查带 guard 的完整输出区。`tail` 用例同时覆盖非整 tile 和不足 128 元素的最后一组。验证成功时输出包含：

```text
[HOST][tail] status=PASS ... bin_match=true
```

## 总结与复用建议

FP32 到 FP8 的四步路径围绕 4:1 窄化排布展开：DINTLV B32 Load 将连续输入拆成两路，ZERO/TWO Cast 把结果放入互补 B8 lane，Or 完成合并，B16 pack Store 在写回时完成最后一级紧排，同一个 B8 mask 同时描述 Or 和 Store 的有效源 lane。

复用代码时应把 DINTLV_B32、ZERO/TWO layout、Or 和 PACK_B16 Store 作为整体核对。目标 FP8 格式、舍入模式或饱和规则改变后，应同时更新 CastTrait 和 golden；若下游能够直接消费转换后的寄存器，融合计算并减少 UB 或 GM 往返通常比继续调整单条 Cast 更有价值。
