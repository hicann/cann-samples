# BF16 RoPE 与 ND2NZ：Pack+Or 融合写出

## 引言

旋转位置编码（RoPE）把一行前后两个 64 元素半段视为 real 和 imag，在 FP32 域执行旋转，再回落到 BF16：

```text
first  = real * cos - imag * sin
second = imag * cos + real * sin
```

当下游直接消费 NZ 布局时，可以在寄存器中把两路 BF16 结果合成一整行，并用 `DATA_BLOCK_COPY` 跨步写出，避免产生中间 ND layout。本教程先实现 `Pack+Or` 融合路径，再通过 UB padding 消除 NZ Store 的 bank conflict。

最终数据流如下：

```text
BF16 ND
  -> LoadAlign(DIST_UNPACK_B16)
  -> Cast<float, bfloat16_t>
  -> Mul / Sub / Add
  -> Cast<bfloat16_t, float>
  -> Pack(LOWEST / HIGHEST)
  -> Or
  -> StoreAlign(DATA_BLOCK_COPY, stride=97)
```

## 支持范围

- 硬件：Ascend 950，`dav-3510`，64 个 AIV。
- 软件：CANN 9.2.0，Release 构建，ASC `-O3`。
- 数据类型：输入和输出为 BF16，sin/cos 为 FP32。
- Shape：`Dk=N=128`，`M` 可变，输出的 `M` 维按 16 对齐。
- 布局：输入为连续 ND，输出逻辑 shape 为 `[8, M_align, 16]`。
- 数值语义：在 FP32 域执行 RoPE，按 round-to-nearest-even 回落到 BF16。
- 验证：逐 bit 比较 NZ golden，并检查尾部 padding 和输出 guard。

## 硬件架构基础

每个半行包含 64 个 BF16。`DIST_UNPACK_B16` 把 BF16 展开到 Cast 可消费的 B32 lane，随后完成 FP32 乘加。两路结果回落到 BF16 后，每个 B32 slot 只有低 B16 有效：

- `Pack<uint16_t, uint32_t, LOWEST>` 将 first 的 64 个有效 BF16 紧排到目标寄存器低半段。
- `Pack<uint16_t, uint32_t, HIGHEST>` 将 second 的 64 个有效 BF16 紧排到目标寄存器高半段。
- `Or` 合并两路互补结果，得到一行连续的 128 个 BF16。
- `DATA_BLOCK_COPY` 将整行拆成 8 个 32 B DataBlock，直接写入 NZ layout。

Ascend 950 的 UB 按 16 个 32 B bank 交织。自然 stride 96 满足 `96 % 16 = 0`，同一条跨步 Store 的 8 个目标 block 会落入同一个 bank；将 UB stride 改为 97 后，8 个 block 依次分散到 8 个 bank。

`trace` case 的全局输入 shape 为 `[6144, 128]`，每个 AIV 处理一个 96 行 tile。每行读取 256 B BF16、256 B sin 和 256 B cos，共 768 B，并写出 256 B BF16。单个 AIV 的有效输入为 73728 B，有效输出为 24576 B。

## 目录说明

```text
bf16_rope_nd2nz/
├── 0_pack_or_fusion/        # Pack+Or 融合基线，UB stride=96
│   ├── CMakeLists.txt
│   └── pack_or_fusion.asc
├── 1_conflict_padding/      # 最终版本，UB stride=97
│   ├── CMakeLists.txt
│   └── conflict_padding.asc
├── CMakeLists.txt
└── README.md
```

两个阶段均包含完整的 VF、kernel、GM/UB 搬运、shape 约束和 host 验证逻辑，不引用其他阶段的实现，可以独立构建和运行。

## 构建与运行

在仓库根目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DNPU_ARCH=dav-3510 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vf_data_transform_bf16_rope_nd2nz_tutorial -j4
```

分别运行两个阶段：

```bash
./build/Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/bf16_rope_nd2nz/0_pack_or_fusion/vf_data_transform_bf16_rope_nd2nz_0_pack_or_fusion --case all
./build/Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/bf16_rope_nd2nz/1_conflict_padding/vf_data_transform_bf16_rope_nd2nz --case all
```

## 基准实现：0_pack_or_fusion

源码：[0_pack_or_fusion/pack_or_fusion.asc](./0_pack_or_fusion/pack_or_fusion.asc)，目标：`vf_data_transform_bf16_rope_nd2nz_0_pack_or_fusion`。

两路 FP32 RoPE 结果分别回落到 BF16 后，使用两个 Pack 提取有效的低 B16，并将它们放入目标寄存器的低、高半段：

```cpp
AscendC::Reg::Cast<bfloat16_t, float, CAST_B32_TO_B16>(outFirstB16, resultFirst, fullB32Mask);
AscendC::Reg::Cast<bfloat16_t, float, CAST_B32_TO_B16>(outSecondB16, resultSecond, fullB32Mask);
AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(packedLow, outFirstB32);
AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::HIGHEST>(packedHigh, outSecondB32);
AscendC::Reg::Or(packedB16, packedLow, packedHigh, fullPackedMask);
```

合并后的 B16 register 直接通过 `DATA_BLOCK_COPY` 写到 NZ。该路径每行保留 4 次必要 Load 和 1 次 NZ Store，不产生中间 ND Store/Load。

基线的 UB stride 为 96，同一条 Store 的 8 个目标 block 落入同一个 bank。CANNsim 测得 `RV_VSSTB` 平均 duration 为 56.93 cycles，VF 用时 993 cycles；计算与布局已经融合，跨步 Store 的 bank conflict 是当前可定位瓶颈。

## 优化阶段一：1_conflict_padding

源码：[1_conflict_padding/conflict_padding.asc](./1_conflict_padding/conflict_padding.asc)，目标：`vf_data_transform_bf16_rope_nd2nz`。

保持 RoPE 计算、Cast、Pack、Or 和 Store 指令完全不变，只在 UB 内部 NZ layout 中增加一行 padding：

```cpp
constexpr uint32_t kRowsPerTile = 96;
constexpr uint32_t kPaddedRowsPerTile = kRowsPerTile + 1U;  // 97
```

`97 % 16 = 1`，同一行的 8 个 NZ block 分布到 8 个不同 bank。每个 UB slot 仅增加 256 B，CopyOut 仍按标准紧凑 NZ layout 写入 GM。

优化后 Vector 指令数保持不变，`RV_VSSTB` 平均 duration 从 56.93 降至 10.05 cycles，Vector 执行 IPC 从 1.295 提升到 1.425，VF cycles 从 993 降到 905。

## 性能结果

以下结果使用 CANNsim Ascend950PR_9589 CAMODEL V100 采集，命令参数为 `--case trace`、`-g vf -n 0`。每个 AIV 处理一个 96 行 tile；`VF cycles` 取 core 0 业务 VF 的 `avg_cycles`。

| 阶段 | UB stride | Vector 执行 IPC | `RV_VSSTB` 平均 duration | VF cycles | 有效输入 B/cycle | 有效输出 B/cycle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0_pack_or_fusion` | 96 | 1.295 | 56.93 | 993 | 74.25 | 24.75 |
| `1_conflict_padding` | 97 | 1.425 | 10.05 | 905 | 81.47 | 27.16 |

padding 使 VF cycles 减少 `8.9%`，局部数据通路提速 `1.10x`。最终路径达到每行 4 次必要 Load 和 1 次 NZ Store 的搬运指令下界，但仍包含 BF16/FP32 Cast、四次 Mul、Add/Sub、两次 Pack 和一次 Or。后续优化应关注 FP32 运算流水、sin/cos 复用以及与相邻算子的融合。本教程没有语义完全相同的融合库接口，因此不列不等价的库性能数据。

## 验证与预期输出

golden 在 FP32 中执行相同的 RoPE 公式，按 round-to-nearest-even 回落到 BF16，再按标准 NZ 索引排列。`--case all` 会依次运行 full、tail 和 perf case，通过时关键日志如下：

```text
[HOST][full] status=PASS ... bin_match=true
[HOST][tail] status=PASS ... bin_match=true
[HOST][perf] status=PASS ... bin_match=true
```

## 总结与复用建议

`Pack+Or` 在寄存器中完成两路 BF16 结果的紧排和合并，使 RoPE 结果可以直接写入 NZ；padding 则消除融合 Store 的 UB bank conflict。更换 RoPE 配对方式、sin/cos 布局、dtype 或 tile 行数时，需要同时重新推导 Load lane、Pack 高低半段以及 `dataBlockStride % bank_count`。
