# BF16 ND2NZ：DATA_BLOCK_COPY 与 UB bank conflict

## 引言

ND2NZ 将连续矩阵 `ND[M, N]` 转换为分块布局 `NZ[N/16, M_align, 16]`，常用于矩阵乘前的数据整理。本教程固定 `N=128`、`N0=16`，使用 SIMD VF 完成 BF16 ND2NZ：先用一条 `DATA_BLOCK_COPY` Store 写出一行中的 8 个 NZ block，再通过 UB padding 消除跨步写出的 bank conflict。

最终数据流如下：

```text
BF16 ND[M, 128]
  -> LoadAlign(DIST_NORM)                     // 一次读取一整行
  -> StoreAlign(DATA_BLOCK_COPY, stride=193)  // 跨步写出 8 个 NZ block
```

## 支持范围

- 硬件：Ascend 950，`dav-3510`，64 个 AIV。
- 软件：CANN 9.2.0，Release 构建，ASC `-O3`。
- 数据类型：输入和输出均为 BF16，不执行 Cast。
- Shape：`N` 固定为 128，`M` 可变，输出的 `M` 维按 16 对齐。
- 布局：输入为 ND，输出逻辑 shape 为 `[8, M_align, 16]`。
- 验证：逐 bit 比较有效 NZ 输出，并检查 `M` 尾部 padding 和输出 guard。

## 硬件架构基础

一行 128 个 BF16 共 256 B，恰好占满一个 Vector 寄存器，也可视为 8 个连续的 32 B DataBlock。`StoreAlign<..., DataCopyMode::DATA_BLOCK_COPY>` 能在一条指令中取出这 8 个 DataBlock，并按 `dataBlockStride` 分散写入 UB。该 stride 的单位是 32 B DataBlock，不是 BF16 元素，也不是全局输出的 `M_align`。

Ascend 950 的 UB 按 16 个 32 B bank 交织，目标地址所在 bank 为：

```text
bank = (byte_offset / 32) % 16
```

如果 `dataBlockStride` 是 16 的整数倍，同一条 Store 的 8 个目标 DataBlock 会落入同一个 bank，引发写写冲突。解决方法是在 UB 内部 NZ layout 的行跨度上增加一个 DataBlock，使 stride 与 16 互质。padding 只存在于 UB，CopyOut 仍生成标准紧凑 NZ 输出。

`trace` case 的全局输入 shape 为 `[12288, 128]`，每个 AIV 处理一个 `192×128` tile。单个 AIV 的有效输入和输出均为 49152 B。每行至少需要一次整行 Load 和一次跨步 Store，两个阶段的指令数都已达到这个数据通路下界；性能差异来自 Store 的 bank 分布。

## 目录说明

```text
bf16_nd2nz/
├── 0_data_block_copy/       # DATA_BLOCK_COPY 基线，UB stride=192
│   ├── CMakeLists.txt
│   └── data_block_copy.asc
├── 1_conflict_padding/      # 最终版本，UB stride=193
│   ├── CMakeLists.txt
│   └── conflict_padding.asc
├── CMakeLists.txt
└── README.md
```

每个阶段的 `.asc` 都完整包含 VF、kernel、GM/UB 搬运、shape 约束和 host 验证逻辑，不引用其他阶段的实现，可以独立构建和运行。

## 构建与运行

在仓库根目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DNPU_ARCH=dav-3510 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vf_data_transform_bf16_nd2nz_tutorial -j4
```

分别运行两个阶段：

```bash
./build/Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/bf16_nd2nz/0_data_block_copy/vf_data_transform_bf16_nd2nz_0_data_block_copy --case all
./build/Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/bf16_nd2nz/1_conflict_padding/vf_data_transform_bf16_nd2nz --case all
```

## 基准实现：0_data_block_copy

源码：[0_data_block_copy/data_block_copy.asc](./0_data_block_copy/data_block_copy.asc)，目标：`vf_data_transform_bf16_nd2nz_0_data_block_copy`。

基线使用一次普通 Load 读取整行，再用一次 `DATA_BLOCK_COPY` Store 完成 ND 到 NZ 的寄存器级布局转换：

```cpp
AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_NORM>(
    rowReg, input + static_cast<uint32_t>(row) * kPhysicalColumns);
AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
    output + static_cast<uint32_t>(row) * kN0, rowReg, dataBlockStride, fullB16Mask);
```

此时 UB layout 使用自然行跨度 192。对同一行的第 `block` 个 NZ block，其目标 bank 为：

```text
bank(block) = (block * 192 + row) % 16 = row % 16
```

8 个 block 的 bank 完全相同。CANNsim 记录到 192 次 Vector Load 和 193 个 Vector Store pipe 事件，VF 用时 1672 cycles；Load/Store IPC 仅为 0.234。功能结果正确，但跨步 Store 的写写冲突成为主要瓶颈。

## 优化阶段一：1_conflict_padding

源码：[1_conflict_padding/conflict_padding.asc](./1_conflict_padding/conflict_padding.asc)，目标：`vf_data_transform_bf16_nd2nz`。

保持 Load 和 `DATA_BLOCK_COPY` Store 不变，只在 UB 内部输出 layout 中增加一行 padding：

```cpp
constexpr uint32_t kRowsPerTile = 192;
constexpr uint32_t kPaddedRowsPerTile = kRowsPerTile + 1U;  // 193
```

`193 % 16 = 1`。以 row 0 为例，8 个目标 DataBlock 的 offset 为 `0, 193, 386, ... 1351`，对应 bank `0, 1, 2, ... 7`，不再发生同一条 Store 内的 bank 重复。相邻行整体偏移一个 DataBlock，同样不会恢复为固定的同 bank 写入。

修改后指令数量不变，Load/Store IPC 从 0.234 提升到 1.199，VF cycles 从 1672 降到 349。该结果说明，本阶段的收益来自消除 bank conflict，而不是减少指令。

## 性能结果

以下结果使用 CANNsim Ascend950PR_9589 CAMODEL V100 采集，命令参数为 `--case trace`、`-g vf -n 0`。每个 AIV 处理一个 192 行 tile；`VF cycles` 取 core 0 业务 VF 的 `avg_cycles`，有效 B/cycle 按单方向 49152 B 计算。

| 阶段 | UB stride | VF cycles | Load/Store IPC | 单方向有效 B/cycle | 瓶颈 |
| --- | ---: | ---: | ---: | ---: | --- |
| `0_data_block_copy` | 192 | 1672 | 0.234 | 29.40 | 8 个 NZ block 写入同一 bank |
| `1_conflict_padding` | 193 | 349 | 1.199 | 140.84 | 每行必要的一次 Load 和一次 Store |

加入 padding 后，VF cycles 减少 `79.1%`，局部数据通路提速 `4.79x`。最终阶段达到每行一次 Load 和一次 Store 的指令数下界；继续优化需要从 tile 选择、前后算子融合或整体搬运流水入手。本教程没有语义完全相同的加速库接口，因此不列不等价的库性能数据。

## 验证与预期输出

程序按 `[8, M_align, 16]` 生成 golden，验证有效元素、尾部 padding 和输出 guard。`--case all` 会依次运行 full、tail 和 perf case，通过时关键日志如下：

```text
[HOST][full] status=PASS ... bin_match=true
[HOST][tail] status=PASS ... bin_match=true
[HOST][perf] status=PASS ... bin_match=true
```

## 总结与复用建议

`DATA_BLOCK_COPY` 负责把一行数据直接分散写入 NZ block，conflict padding 决定这条跨步 Store 能否获得预期吞吐。复用到其他 dtype、列数或 tile 行数时，需要重新计算每条 Store 覆盖的 DataBlock 数以及 `dataBlockStride % bank_count`，不能直接照搬常量 193。
