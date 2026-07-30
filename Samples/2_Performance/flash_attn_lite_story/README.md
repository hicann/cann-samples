# [WIP] Flash Attention Lite：Ascend 950 上的 CV 融合优化实践

## 概述

Flash Attention Lite（FALite）是面向 Ascend 950 的教学样例，以固定规格的 Flash Attention 前向计算说明 Cube 与 Vector 融合算子的实现和流水排布。代码从单槽串行基线开始，逐步加入 CV 核间双槽流水、AIC 核内双缓冲和 task 级 I/O 双缓冲。

本文覆盖 v0～v5。v0/v1 为 GM 通路基线（Cube/Vector 经 GM 交换数据），v2 起转为片内零拷贝通路。各版本使用相同的算法、输入规格和精度判据。

## 算子实现原理

### 功能与计算公式

标准 Attention 前向计算为：

$$
O=\operatorname{softmax}\left(QK^T\cdot scale\right)V,\qquad scale=\frac{1}{\sqrt{D}}
$$

序列长度按 `Br=Bc=128` 分块。对于第 $i$ 个 Q tile 和第 $j$ 个 KV tile，online softmax 的递推过程为：

$$
S_j=Q_iK_j^T\cdot scale
$$

$$
m_j=\max\left(m_{j-1},\operatorname{rowmax}(S_j)\right),\qquad
\alpha_j=e^{m_{j-1}-m_j}
$$

$$
P_j=e^{S_j-m_j},\qquad
l_j=\alpha_jl_{j-1}+\operatorname{rowsum}(P_j)
$$

$$
OAcc_j=\alpha_jOAcc_{j-1}+P_jV_j,\qquad
O_i=\frac{OAcc_{\mathrm{final}}}{l_{\mathrm{final}}}
$$

Kernel 的 C1 阶段实际计算 `K_j × Q_i^T`，以 `S_j^T` 的形式交给 AIV。这是片上布局和矩阵乘方向的选择，不改变上述数学结果。

### 支持规格

| 项目 | 当前支持范围 |
| --- | --- |
| NPU 架构 | `dav-3510`（Ascend 950） |
| Q/K/V/O 数据类型 | BF16 |
| 逻辑 shape | `(B,N=1,S,D=128)` |
| GM 排布 | 连续 ND，可等价看作 `(B,S,128)` |
| 分块 | `Br=Bc=128` |
| Batch | `B > 0` |
| 序列长度 | `S > 0` 且 `S % 128 == 0` |
| Softmax scale | Demo 固定为 `1 / sqrt(128)` |

当前不支持尾块、attention mask、causal、dropout、变长序列、多 head 和反向计算。

### 分块计算流程

一个 task 处理 128 行 Q。每个 task 遍历全部 KV tile，由同一 Mix 组的 1 个 AIC 和 2 个 AIV 协作；两个 AIV 各处理 64 行 Q。

| 阶段 | 执行核心 | 主要计算 |
| --- | --- | --- |
| C1 | AIC | `K × Q^T -> S^T` |
| V1 | AIV×2 | online softmax，更新 `m/l/alpha`，生成 BF16 NZ `P^T` |
| C2 | AIC | `P × V -> DeltaO` |
| V2 | AIV×2 | `OAcc = alpha × OAcc + DeltaO` |

中间矩阵 P 不写回 GM。AIV 在 UB 中完成 Softmax、BF16 Cast 和 DN→NZ 转换，再将 P 写入 Mix 组共享的 L1；AIC 从同一块 L1 读取 P 并送入 L0A：

```text
S UB（FP32）
  -> PWork UB（BF16 NZ）
  -> P L1
  -> L0A
  -> C2 Mmad
```

## 工程结构

```text
flash_attn_lite_story/
├── CMakeLists.txt
├── README.md
├── requirements.txt
├── images/                     # 文档图片
├── scripts/
│   ├── flash_attn_lite_gendata.py
│   ├── flash_attn_lite_verify.py
│   └── thread_limit.py
└── src/
    ├── flash_attn_lite.h        # 各版本共用的 Host 接口
    ├── flash_attn_lite_demo.cpp # 各版本共用的 Demo
    ├── v2/
    ├── v3/
    ├── v4/
    └── v5/
```

每个 `src/vN` 目录保留该版本独立的 TilingData、Host tiling、Kernel 入口和 AIC/AIV 实现。CMake 自动识别 `src/vN` 并生成 `falite_vN`。

## 环境准备与编译运行

### 安装 Python 依赖

```bash
python3 -m pip install \
    -r Samples/2_Performance/flash_attn_lite_story/requirements.txt
```

`numpy` 和 `ml_dtypes` 是必选依赖。CPU 版 `torch` 为可选依赖，可缩短 Golden 计算时间。

### 编译

在 cann-samples 根目录执行：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510

# 构建本 Story 的全部版本，两个聚合目标作用相同。
cmake --build build --target falite
cmake --build build --target flash_attn_lite_story

# 只构建一个版本。
cmake --build build --target falite_v0
cmake --build build --target falite_v1
cmake --build build --target falite_v2
cmake --build build --target falite_v3
cmake --build build --target falite_v4
cmake --build build --target falite_v5
```

二进制和共用 Python 脚本位于：

```text
build/Samples/2_Performance/flash_attn_lite_story/
```

真机默认使用 mode2 完成同一 Mix 组内 `1 AIC <-> 2 AIV` 的 CrossCore 同步。CANNsim 调试时应使用独立构建目录，并显式打开兼容路径：

```bash
cmake -S . -B build-sim \
    -DNPU_ARCH=dav-3510 \
    -DSIM_COMPATIBLE=ON
```

### 运行与校验

```bash
./build/Samples/2_Performance/flash_attn_lite_story/falite_v0 --size 1 32768
./build/Samples/2_Performance/flash_attn_lite_story/falite_v1 --size 1 32768
./build/Samples/2_Performance/flash_attn_lite_story/falite_v2 --size 1 32768
./build/Samples/2_Performance/flash_attn_lite_story/falite_v3 --size 1 32768
./build/Samples/2_Performance/flash_attn_lite_story/falite_v4 --size 1 32768
./build/Samples/2_Performance/flash_attn_lite_story/falite_v5 --size 1 32768
```

本次迁移使用 CANN 9.0.0 Community、`dav-3510` 和全部 32 个 AIC 验证上述规格，v2～v5 的 Golden 比对均为 0 个失败元素。

命令行参数：

- `--size B S`：设置 Batch 和序列长度。
- `--core-num n`：指定本次 launch 使用的 AIC 数量，主要用于仿真；真机通常不传。
- `--dry-run`：仍执行数据生成、Kernel、D2H 和 `npuout_o.bin` 落盘，只跳过 Golden 计算与精度比对。

程序在可执行文件所在目录重建 `data/`，生成 Q/K/V 并写出 NPU 结果。非 `--dry-run` 模式使用 FP32 计算 Golden，转为 BF16 后逐元素检查：

```text
abs(npu - golden) <= 2^-6 + 2^-6 * abs(golden)
```

全部元素通过时程序返回 0。

## 分阶段优化

### v0：分块 + online softmax / GM 块交换

v0 实现分块 FA 的 j-loop CV 流水。Cube 与 Vector 通过 GM 交换中间矩阵 S/P/ΔO，每轮 j 迭代完成 C1→V1→C2→V2 四个阶段的 CrossCore 同步。

```text
j-loop: C1[j] --S_READY--> V1[j] --P_READY--> C2[j] --O_READY--> V2[j] --DONE--> C1[j+1]
```

C1/C2 使用完整单 Mmad(128,128,128) + Fixpipe L0C→GM 将 S/ΔO 落盘，Q 每 task 搬入一次、K/V 每轮搬入。C1/C2 结果直接写 GM。AIV 做 online softmax 后直写 BF16 O。v0 与 v1 的 C1/C2 结构完全一致，仅 S/ΔO 搬运路径不同（v0: L0C→GM→UB, v1: L0C→UB）。

GM 中间 buffer 均 per-task 复用 (j 间由 CrossCore 保护)：S (Bc×Br FP32) + P (Bc×Br BF16) + ΔO (Br×D FP32)。

### v1：L0C→UB 优化

v1 在 v0 基础上将 C1 的 S 和 C2 的 ΔO 改为 Fixpipe L0C→UB，直接写入 AIV 的 UB。AIV 不再从 GM 搬运 S/ΔO，省去两路 GM 读写。P 仍通过 AIV→GM→AIC P L1 路径交换。

```text
C1: K×Q^T → S^T⸺FixpipeToVecUB→ AIV UB (dualDstCtl=2)
V1: UB 读 S → online softmax → P → GM
C2: GM 读 P × V → ΔO⸺FixpipeToVecUB→ AIV UB
V2: UB 读 ΔO → O_acc 更新
```

C1 使用单 Mmad (K[128,128]×Q^T[128,128])，不再做 D 维拆分。L0B/L0C buffer 相应增大至 16384 elem。

GM 中间 buffer：仅 P (BF16)，S/ΔO 走 UB。

### v2：单槽融合基线

v2 打通 1C2V 的四阶段数据链，并使 P 通过 UB→L1 直接交给 C2。Q/K/V/P、L0A/L0B/L0C 以及 AIV 工作区均为单槽。

```text
C1[j] -> V1[j] -> C2[j] -> V2[j] -> C1[j+1]
```

核间使用四个固定 flag：

```text
C1 --S_READY--> V1 --P_READY--> C2 --O_READY--> V2 --DONE--> 下一轮 C1
```

其中 `DONE` 门控下一轮 C1，保护单槽资源复用，不表示整个 task 已写回 GM。单槽地址和完整的阶段交接便于核对正确性，但 AIC 与 AIV 在不同阶段交替等待，tile 之间的主要工作基本串行。

参考性能：`113837.031250 us`。

### v3：双槽 CV 分组流水

v3 把相邻两个 KV tile 编为一组。K/V/P L1 和 S/DeltaO/alpha/PWork UB 改为双槽；Q、`m/l/OAcc` 和 L0A/L0B/L0C 仍为单槽。

```text
Loop1:  AIC C1[0]
Loop2:  AIC C1[1]      || AIV V1[0]
Loop3:  AIC C2[0]      || AIV V1[1]
Loop4:  AIC C2[1]      || AIV V2[0]
Loop5:  AIC C1[next,0] || AIV V2[1]
```

`||` 表示两个阶段允许重叠，不表示阶段时长相同。核间只保留按 slot 区分的三类就绪 flag：

```text
C1[s] --S_READY[s]--> V1[s] --P_READY[s]--> C2[s]
                                     C2[s] --O_READY[s]--> V2[s]
```

相较 v2，v3 删除了每个 tile 末尾的 `DONE`，AIC 和 AIV 可以错位处理相邻 tile。L0 仍是单槽，每个 Cube stage 末尾使用 `FIX_S` 等待 Fix 完成，因此相邻 slot 的 MTE1、Mmad 和 Fix 仍难以重叠。

参考性能：`75413.796875 us`，相较 v2 下降 `33.75%`，加速 `1.509x`。

### v4：AIC 核内双缓冲

v4 保持 v3 的 CV 双槽分组和 AIV 发射顺序，为 L0A、L0B、L0C 分配 slot 0/1，并删除 Cube stage 末尾的 `FIX_S`。

```text
Loop1:  Load[0]
Loop2:  Load[1]    || Mmad[0]
Loop3:                  Mmad[1]    || Fix[0]
Loop4:                                  Fix[1]
```

HardEvent 只连接真实的数据依赖和同槽复用：

```text
正向：MTE2 --MTE2_MTE1--> MTE1 --MTE1_M--> Mmad --M_FIX--> Fix
反向：C2.Mmad --M_MTE1--> 下一组 C1.Load
      C2.Fix  --FIX_M----> 下一组 C1.Mmad
```

C1 先发射 Q 的 L1→L0B，再发射 K 的 GM→L1；C2 先发射 P 的 L1→L0A，再发射 V 的 GM→L1。`P_READY_MTE2` 门控 V 的 MTE2，使 P 的 MTE1 与 V 的 MTE2 可以交叠；`MTE1_MTE2` 保护 V L1 slot 的复用。

相较 v3，v4 放开了不同 L0 slot 上 MTE1、Mmad 和 Fix 的错位执行。当前调度仍以两个 KV tile 为一组，V 也仍在 C2 阶段搬入。

参考性能：`51068.414062 us`，相较 v3 下降 `32.28%`，加速 `1.477x`。

### v5：KV 预取与外层 I/O 双缓冲

v5 保留 v4 的双 tile 分组和 L0 双缓冲，将 V 的 GM→L1 从 C2 前移到 C1，与同一 KV tile 的 K 一起预取。C2 只执行 P/V 的 L1→L0 搬运及矩阵乘。

```text
task start: CopyIn(Q: GM -> L1)
C1[j]:     Load(Q: L1 -> L0B) + CopyIn(K/V: GM -> L1)
           -> Load(K: L1 -> L0A) -> Mmad(K, Q) -> S
C2[j]:     Load(P/V: L1 -> L0A/L0B) -> Mmad(P, V) -> DeltaO
```

V 前移后，v4 的 `P_READY_MTE2` 不再需要，核间重新使用 `S_READY/P_READY/O_READY` 三类 flag。K/V L1 共用 slot 的 ready/free 事件，C2 读完 V 后才允许下一代 C1 覆写。

Q L1 和 OAcc UB 另外按 task 使用 `ioSlot=0/1`。最终 Div/Cast 在 OAcc 槽内原地完成，输出 MTE3 可以和另一个 I/O 槽上的 task 错位执行：

```text
task t:   Q[io] -> C1/V1/C2/V2 -> Div/Cast[io] -> MTE3 O[io]
task t+1:                 Q[io^1] -> C1/V1/C2/V2 -> ...
```

相较 v4，v5 去掉了 C2 中单独的 V MTE2，并允许 task 边界两侧的 Q 搬入、Vector 计算和输出 MTE3 形成重叠。当前仍按两个 KV tile 分组；PipeUtilization 中 AIV Vector 的活动区间最长，但各组件区间存在重叠，现有数据不足以认定单一 bound。

参考性能：`46974.507812 us`，相较 v4 下降 `8.02%`，加速 `1.087x`。

## 精度验证

v0/v1 使用与 v2～v5 相同的 BF16 算法和精度判据。以下为 CANN 9.2.0 + dav-3510 + 28 AIC 条件下的 Golden 比对结果：

| 版本 | S=128 | S=1024 | S=4096 | S=32768 | S=131072 |
|------|-------|--------|--------|---------|----------|
| v0 | 0 失败 | 0 失败 | 0 失败 | 0 失败 | dry-run 可运行 (未做 Golden 比对) |
| v1 | 0 失败 | 0 失败 | 0 失败 | 0 失败 | dry-run 可运行 (未做 Golden 比对) |

v0/v1 的 S/P/ΔO buffer 均 per-task 复用 (Bc×Br / Bc×Br / Br×D)，S=131072 时 v0 buffer 合计约 160 MiB，v1 仅需 P buffer 约 32 MiB。

## 性能参考

v0/v1 性能数据（CANN 9.2.0, dav-3510, msopprof Task Duration）：

| 版本 | S=32768 (28 AIC) | vs v0 | S=131072 (32 AIC) | vs v0 |
|------|-----------------|-------|-------------------|-------|
| v0 | 20,359 us | 1.00x | 269,403 us | 1.00x |
| v1 | 10,409 us | **1.96x** | 139,645 us | **1.93x** |

v1 通过 L0C→UB 省去 S/ΔO 两路 GM 读写，加速比稳定在约 2x。

以下为 v2～v5 性能参考。本次只调整版本目录编号，四份实现源码未改。下表按当前编号列出同一次干净构建和同一套采集条件下的结果：

- CANN 9.2.0
- `dav-3510`，正式 mode2
- 全部 32 个 AIC
- `--dry-run --size 1 131072`
- `msopprof --warm-up=5 --launch-count=1`
- 指标为 `Task Duration`

| 版本 | Task Duration | 相较前版 | 加速比 |
| --- | ---: | ---: | ---: |
| v2 | `113837.031250 us` | 当前四个实现的基线 | — |
| v3 | `75413.796875 us` | `-33.75%` | `1.509x` |
| v4 | `51068.414062 us` | `-32.28%` | `1.477x` |
| v5 | `46974.507812 us` | `-8.02%` | `1.087x` |

从 v2 到 v5，Task Duration 下降 `58.74%`，对应 `2.423x`。这些单次实板数据用于说明当前版本在同条件下的相对变化，不代表所有规格上的固定收益。

## 当前实现边界

- 所有版本使用相同的 BF16 算法和固定分块，不支持本 README“支持规格”之外的输入。
- v2 是单槽串行基线；v3～v5 均以两个 KV tile 为一组，尾组不足两个 tile 时只执行 slot 0。
- v5 已加入 AIC 核内双缓冲、KV 预取和 task 级 I/O 双缓冲，但 group 边界仍会留下流水空洞。
- 性能表只比较本次同源报告中的 `Task Duration`。组件 active time 可以重叠，不能直接相加，也不能只凭某一项 active time 判定稳定的性能 bound。
