# [WIP] Kimi Delta Attention Lite：Ascend 950 上的 Chunk KDA 优化实践

## 概述

Kimi Delta Attention Lite（KDALite）是面向 Ascend 950 的 Kimi Delta Attention Prefill 样例。代码以逐 token 的 Recurrent KDA 为数学定义，在设备端使用等价的 Chunk KDA。v0～v3 依次实现三 Kernel 基线、Kernel 融合、Cube/Vector 协作和状态流水。

四个版本使用相同的输入接口、零初始状态和精度判据。版本之间只调整 Chunk 中间量、Kernel 划分、片上缓冲和发射顺序。Q/K/V 投影、ShortConv、门参数生成、输出归一化与投影不在本样例范围内。

## 算子实现原理

### 支持规格与范围

| 项目 | 当前支持范围 |
| --- | --- |
| NPU 架构 | `dav-3510`（Ascend 950） |
| Head 与维度 | `N=1`，`Dk=Dv=128`；当前源码布局要求 `Dk==Dv` |
| Q/K/V/O | BF16，逻辑 shape `[B,N=1,S,128]` |
| `log_decay` | FP32，逻辑 shape `[B,N=1,S,128]`，值不大于 0 |
| beta | BF16，逻辑 shape `[B,N=1,S]`，值位于 `[0,1]` |
| `final_state` | FP32，逻辑 shape `[B,N=1,128,128]` |
| Batch 与序列 | `B>0`、`S>0`，支持尾 Chunk |
| 初始状态 | 全零，不提供非零 `initial_state` 输入 |
| ChunkSize | 默认 32；v0/v1 支持编译期 16、32、64，v2/v3 支持 16、32 |

固定为 1 的 Head 轴不写入数据文件，物理 shape 分别为 `Q/K/V/O=[B,S,128]`、`log_decay=[B,S,128]`、`beta=[B,S]` 和 `final_state=[B,128,128]`。

`Dk==Dv` 是本样例接口、任务切分和片上布局的支持边界，不是 KDA 数学公式对维度的要求；一般形式允许 `Dk` 与 `Dv` 不同。

Q/K 已完成 L2Norm，Q 已乘 `1/sqrt(Dk)`。`log_decay` 是门控后的逐 token、逐 Key 通道值，beta 是经过 sigmoid 的逐 token 标量。Host 只检查 B/S、任务数、指针、workspace 和 Mix 组数；接口不携带 dtype 或维度元数据，因此不能复核实际缓冲区布局，也不读取设备数据检查这些前置条件。

当前不支持多 Head、非零初态、连续 Prefill、Decode、packed/varlen 输入、训练反向，以及 KDA core 之外的投影、卷积、归一化和门控计算。v2/v3 已按 `log_decay∈[-5,0]` 验证 C16/C32；C64 未通过强衰减精度校验，因此不开放。v0/v1 的 C64 只通过较弱随机衰减数据，不能外推到完整门值范围。

### 功能与计算公式

#### 记号与形状

除非特别说明，本文中的向量均为行向量。先省略 Batch 轴和固定为 1 的 Head 轴，讨论一条序列。KDA core 接收

$$
\mathbf Q,\mathbf K,\mathbf g\in\mathbb R^{S\times D_k},\qquad
\mathbf V\in\mathbb R^{S\times D_v},\qquad
\boldsymbol\beta\in\mathbb R^{1\times S},
$$

输出

$$
\mathbf O\in\mathbb R^{S\times D_v},\qquad
\mathbf H_{\mathrm{final}}\in\mathbb R^{D_k\times D_v}.
$$

其中，$\mathbf g$ 对应输入 `log_decay`，指数运算按元素进行。第 $i$ 个 token 的变量和状态形状为

$$
\mathbf q_i,\mathbf k_i,\mathbf g_i\in\mathbb R^{1\times D_k},\qquad
\mathbf v_i,\mathbf o_i,\mathbf p_i,\mathbf r_i\in\mathbb R^{1\times D_v},\qquad
\mathbf H_i\in\mathbb R^{D_k\times D_v},\qquad
\beta_i\in\mathbb R.
$$

| 记号 | 含义 |
| --- | --- |
| $\mathbf x\mathbf Y$ | 矩阵乘；行向量读取状态时结果仍为行向量 |
| $\mathbf x^\top\mathbf y$ | 列向量与行向量的外积 |
| $\odot$ | 逐元素乘 |
| $\operatorname{Diag}(\mathbf x)$ | 以向量 $\mathbf x$ 为对角线构造方阵 |
| $\operatorname{StrictLower}(\mathbf X)$ | 只保留严格下三角元素 |
| $\operatorname{Lower}(\mathbf X)$ | 保留下三角和对角线 |

本文使用 $B$ 表示 Batch size，$N$ 表示 Head 数，$S$ 表示序列长度，$C$ 表示 ChunkSize，$T_c=\lceil S/C\rceil$ 表示 Chunk 数。本样例固定 $N=1$ 和 $D_k=D_v=128$，但下面的数学公式允许 $D_k\ne D_v$。

#### Recurrent KDA

Recurrent KDA 是本样例的正确性定义。令 $\mathbf H_{i-1}\in\mathbb R^{D_k\times D_v}$ 为处理第 $i$ 个 token 前的状态，初始状态为 $\mathbf H_{-1}=\mathbf 0$。一次递推依次执行

$$
\begin{aligned}
\boldsymbol\lambda_i &= \exp(\mathbf g_i) &&\in\mathbb R^{1\times D_k},\\
\overline{\mathbf H}_i &= \operatorname{Diag}(\boldsymbol\lambda_i)\mathbf H_{i-1} &&\in\mathbb R^{D_k\times D_v},\\
\mathbf p_i &= \mathbf k_i\overline{\mathbf H}_i &&\in\mathbb R^{1\times D_v},\\
\mathbf r_i &= \beta_i(\mathbf v_i-\mathbf p_i) &&\in\mathbb R^{1\times D_v},\\
\mathbf H_i &= \overline{\mathbf H}_i+\mathbf k_i^\top\mathbf r_i &&\in\mathbb R^{D_k\times D_v},\\
\mathbf o_i &= \mathbf q_i\mathbf H_i &&\in\mathbb R^{1\times D_v}.
\end{aligned}
$$

这里先按 Key 通道衰减旧状态，再用 $\mathbf k_i$ 读取预测值。残差 $\mathbf r_i$ 经 $\beta_i$ 缩放后，以外积 $\mathbf k_i^\top\mathbf r_i$ 写回状态。输出读取的是已经包含当前 token 更新的 $\mathbf H_i$，而不是 $\mathbf H_{i-1}$。Golden 在 FP32 中直接执行这组逐 token 递推，不复用设备端的 Chunk 变换。

#### 从 Recurrent 形式推到 Chunk 形式

考虑一个有效长度为 $L\le C$ 的 Chunk，并用局部下标 $i=0,\ldots,L-1$。补齐后的 Chunk 张量为

$$
\mathbf Q,\mathbf K,\mathbf g\in\mathbb R^{C\times D_k},\qquad
\mathbf V\in\mathbb R^{C\times D_v},\qquad
\boldsymbol\beta\in\mathbb R^{1\times C},\qquad
\mathbf H_{\mathrm{in}}\in\mathbb R^{D_k\times D_v}.
$$

先定义累计 log-decay

$$
\mathbf G_i=\sum_{t=0}^{i}\mathbf g_t\in\mathbb R^{1\times D_k},\qquad
\mathbf E_i=\operatorname{Diag}\!\left(\exp(\mathbf G_i)\right)\in\mathbb R^{D_k\times D_k},
$$

并约定 $\mathbf G_{-1}=\mathbf 0$、$\mathbf E_{-1}=\mathbf I_{D_k}$，因此 $\widetilde{\mathbf H}_{-1}=\mathbf H_{\mathrm{in}}$。由于相邻 token 的衰减满足 $\operatorname{Diag}(\exp(\mathbf g_i))=\mathbf E_i\mathbf E_{i-1}^{-1}$，可把状态归一化为

$$
\widetilde{\mathbf H}_i=\mathbf E_i^{-1}\mathbf H_i.
$$

代入 Recurrent 状态更新后，连续的衰减项被消去：

$$
\widetilde{\mathbf H}_i
=\widetilde{\mathbf H}_{i-1}
+\left(\mathbf k_i\odot\exp(-\mathbf G_i)\right)^\top\mathbf r_i.
$$

因此

$$
\widetilde{\mathbf H}_i
=\mathbf H_{\mathrm{in}}
+\sum_{j=0}^{i}\left(\mathbf k_j\odot\exp(-\mathbf G_j)\right)^\top\mathbf r_j.
$$

定义按行堆叠的变换矩阵

$$
\begin{aligned}
\mathbf Q^+_i &= \mathbf q_i\odot\exp(\mathbf G_i),\\
\mathbf K^+_i &= \mathbf k_i\odot\exp(\mathbf G_i),\\
\mathbf K^-_i &= \mathbf k_i\odot\exp(-\mathbf G_i),
\end{aligned}
\qquad
\mathbf Q^+,\mathbf K^+,\mathbf K^-\in\mathbb R^{C\times D_k}.
$$

第 $i$ 个 token 的预测值便可写成

$$
\mathbf p_i
=\mathbf K^+_i\mathbf H_{\mathrm{in}}
+\sum_{j=0}^{i-1}(\mathbf P_{\mathrm{raw}})_{ij}\mathbf r_j,
\qquad
\mathbf P_{\mathrm{raw}}=\mathbf K^+(\mathbf K^-)^\top\in\mathbb R^{C\times C}.
$$

其中

$$
(\mathbf P_{\mathrm{raw}})_{ij}
=\left\langle\mathbf k_i\odot\exp(\mathbf G_i),\;\mathbf k_j\odot\exp(-\mathbf G_j)\right\rangle
\in\mathbb R
$$

描述第 $j$ 个残差对第 $i$ 个预测的影响。因果关系只使用 $j<i$ 的严格下三角元素。

将所有残差按行堆叠成 $\mathbf R\in\mathbb R^{C\times D_v}$，令 $\mathbf B_\beta=\operatorname{Diag}(\boldsymbol\beta)\in\mathbb R^{C\times C}$，即可得到单位下三角方程

$$
\underbrace{\left[\mathbf I_C+\operatorname{StrictLower}(\mathbf B_\beta\mathbf P_{\mathrm{raw}})\right]}_{\mathbf T\in\mathbb R^{C\times C}}
\mathbf R
=\mathbf B_\beta\left(\mathbf V-\mathbf K^+\mathbf H_{\mathrm{in}}\right).
$$

这一步把 Chunk 内按 token 串行的残差依赖集中到了一个下三角系统中。$\mathbf T$ 的对角线恒为 1，因此可以按行前向代入，不需要计算通用矩阵逆。

#### Chunk KDA

定义

$$
\mathbf M=\mathbf T^{-1}\mathbf B_\beta\in\mathbb R^{C\times C}.
$$

实现通过方程 $\mathbf T\mathbf M=\mathbf B_\beta$ 求 $\mathbf M$。若 $\mathbf e_i\in\mathbb R^{1\times C}$ 是第 $i$ 个标准基行向量，则前向代入为

$$
\mathbf M_{i,:}
=\beta_i\mathbf e_i
-\beta_i\sum_{j=0}^{i-1}(\mathbf P_{\mathrm{raw}})_{ij}\mathbf M_{j,:}.
$$

于是残差矩阵可改写为

$$
\begin{aligned}
\mathbf W &= \mathbf M\mathbf K^+ &&\in\mathbb R^{C\times D_k},\\
\mathbf U &= \mathbf M\mathbf V &&\in\mathbb R^{C\times D_v},\\
\mathbf R &= \mathbf U-\mathbf W\mathbf H_{\mathrm{in}} &&\in\mathbb R^{C\times D_v}.
\end{aligned}
$$

输出端定义

$$
\mathbf A_{\mathrm{raw}}=\mathbf Q^+(\mathbf K^-)^\top\in\mathbb R^{C\times C},\qquad
\mathbf A=\operatorname{Lower}(\mathbf A_{\mathrm{raw}})\in\mathbb R^{C\times C}.
$$

其中 $\mathbf A$ 保留对角线，因为 $\mathbf o_i$ 读取的是更新后的 $\mathbf H_i$。整个 Chunk 的输出为

$$
\mathbf O
=\underbrace{\mathbf Q^+\mathbf H_{\mathrm{in}}}_{\text{history}\in\mathbb R^{C\times D_v}}
+\underbrace{\mathbf A\mathbf R}_{\text{local}\in\mathbb R^{C\times D_v}}
\in\mathbb R^{C\times D_v}.
$$

令 $\mathbf G_{\mathrm{tail}}=\mathbf G_{L-1}$，并定义

$$
\mathbf d=\exp(\mathbf G_{\mathrm{tail}})\in\mathbb R^{1\times D_k},\qquad
\mathbf K^{\mathrm{tail}}_i=\mathbf k_i\odot\exp(\mathbf G_{\mathrm{tail}}-\mathbf G_i),\qquad
\mathbf K^{\mathrm{tail}}\in\mathbb R^{C\times D_k}.
$$

Chunk 末状态为

$$
\mathbf H_{\mathrm{out}}\in\mathbb R^{D_k\times D_v},\qquad
\mathbf H_{\mathrm{out}}
=\operatorname{Diag}(\mathbf d)\mathbf H_{\mathrm{in}}
+(\mathbf K^{\mathrm{tail}})^\top\mathbf R.
$$

以上公式与 Recurrent KDA 在实数精确算术下等价。实现按下面三类工作组织：

| 阶段 | 主要计算 | 依赖关系 |
| --- | --- | --- |
| Chunk 准备 | 生成 $\mathbf M,\mathbf A,\mathbf Q^+,\mathbf K^+,\mathbf K^{\mathrm{tail}},\mathbf d$，以及版本需要的 $\mathbf W/\mathbf U$ | 每个 Chunk 独立 |
| 状态推进 | 根据 $\mathbf H_{\mathrm{in}}$ 计算 $\mathbf R$ 和 $\mathbf H_{\mathrm{out}}$ | 同一 Batch 的 Chunk 必须按序 |
| 输出计算 | 计算 history、local 和 $\mathbf O$ | 各 Chunk 的 $D_v$ 列块可并行 |

v1/v2 不保存 $\mathbf W$，而按结合律计算 $\mathbf M(\mathbf K^+\mathbf H_{\mathrm{in}})$；v0/v3 使用 $(\mathbf M\mathbf K^+)\mathbf H_{\mathrm{in}}$。两条路径在实数精确算术中相同，但 BF16 中间量的舍入位置不同，因此每个版本都单独对 Recurrent Golden 做精度校验。

$\mathbf K^-$ 只用于数学推导。v2/v3 为 Cube 构造带 anchor 的 BF16 因子，避免直接生成 $\exp(-\mathbf G_i)$ 带来的过大动态范围；anchor 在矩阵乘中相互抵消，不改变上述实数公式。

尾块将无效行补为 $\mathbf Q=\mathbf K=\mathbf V=\mathbf 0$、$\beta=0$、$\mathbf g=\mathbf 0$。这些行不衰减状态，也不写入新的状态增量。Kernel 仍按完整 $C$ 行计算，最终只回写有效的 $L$ 行输出。

## 公共接口与执行模型

### Host 接口

公共头文件 `src/kimi_delta_attn_lite.h` 在全局命名空间提供以下接口：

```cpp
bool GetKimiDeltaAttnLiteWorkspaceSize(
    uint32_t batchSize,
    uint32_t seqLen,
    uint64_t& workspaceBytes);

bool KimiDeltaAttnLiteNPU(
    uint8_t* dQ,
    uint8_t* dK,
    uint8_t* dV,
    uint8_t* dLogDecay,
    uint8_t* dBeta,
    uint8_t* dO,
    uint8_t* dFinalState,
    uint8_t* dWorkspace,
    uint64_t workspaceBytes,
    uint32_t batchSize,
    uint32_t seqLen,
    uint32_t requestedMixCoreNum,
    aclrtStream stream);
```

| 参数 | dtype 与物理布局 | 说明 |
| --- | --- | --- |
| `dQ/dK/dV` | BF16 `[B,S,128]` | 输入，逻辑 shape 为 `[B,N=1,S,D]` |
| `dLogDecay` | FP32 `[B,S,128]` | 输入，每个 token、每个 Key 通道一个 log-decay |
| `dBeta` | BF16 `[B,S]` | 输入，每个 token 一个标量 |
| `dO` | BF16 `[B,S,128]` | 输出，逻辑 shape 为 `[B,N=1,S,Dv]` |
| `dFinalState` | FP32 `[B,128,128]` | 输出，逻辑 shape 为 `[B,N=1,Dk,Dv]` |
| `dWorkspace` | 字节缓冲区 | 调用方分配，大小不得小于查询接口返回值 |
| `requestedMixCoreNum` | `uint32_t` | Mix 组数上限；0 表示使用本卡全部 Mix 组，正数不得超过设备 Mix 组数 |
| `stream` | `aclrtStream` | 两个或三个 Kernel 按版本顺序异步提交到该 stream |

Host 会校验 B/S、指针、任务数、workspace 和 Mix 组数，但不会验证调用方传入缓冲区的实际 dtype 与布局，也不会读取设备数据检查 Q/K 是否归一化或 `log_decay`、beta 的值域。`KimiDeltaAttnLiteNPU` 返回 `true` 只表示参数校验通过且 Kernel 已提交；调用方需要让设备指针、workspace 和 stream 存活到 stream 执行结束。

workspace 是版本私有协议。调用方必须使用当前二进制对应的 `GetKimiDeltaAttnLiteWorkspaceSize` 查询大小，不能把某个版本生成的 workspace 交给另一个版本继续执行。

### Device Kernel 接口

Kernel 均接收按值传递的 `KimiDeltaAttnLiteTilingData`。v0 使用三个 Mix Kernel：

```cpp
__global__ __mix__(1, 2) void kimi_delta_attn_lite_prepare_k(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR logDecay, GM_ADDR beta,
    GM_ADDR workspace, KDALite::KimiDeltaAttnLiteTilingData data);

__global__ __mix__(1, 2) void kimi_delta_attn_lite_state_update_k(
    GM_ADDR finalState, GM_ADDR workspace, KDALite::KimiDeltaAttnLiteTilingData data);

__global__ __mix__(1, 2) void kimi_delta_attn_lite_local_output_k(
    GM_ADDR output, GM_ADDR workspace, KDALite::KimiDeltaAttnLiteTilingData data);
```

v1 使用一个纯 Vector Prepare 和一个 Mix StateOutput：

```cpp
__global__ __vector__ void kimi_delta_attn_lite_prepare_k(
    GM_ADDR q, GM_ADDR k, GM_ADDR logDecay, GM_ADDR beta,
    GM_ADDR workspace, KDALite::KimiDeltaAttnLiteTilingData data);

__global__ __mix__(1, 2) void kimi_delta_attn_lite_state_update_k(
    GM_ADDR value, GM_ADDR output, GM_ADDR finalState,
    GM_ADDR workspace, KDALite::KimiDeltaAttnLiteTilingData data);
```

v2/v3 的参数与 v1 相同，但 Prepare 改为 Mix Kernel：

```cpp
__global__ __mix__(1, 2) void kimi_delta_attn_lite_prepare_k(
    GM_ADDR q, GM_ADDR k, GM_ADDR logDecay, GM_ADDR beta,
    GM_ADDR workspace, KDALite::KimiDeltaAttnLiteTilingData data);

__global__ __mix__(1, 2) void kimi_delta_attn_lite_state_update_k(
    GM_ADDR value, GM_ADDR output, GM_ADDR finalState,
    GM_ADDR workspace, KDALite::KimiDeltaAttnLiteTilingData data);
```

第二个 Kernel 已同时完成状态更新和输出生成，但源码和 profiler 中的入口符号仍为 `kimi_delta_attn_lite_state_update_k`。

令 `M` 为 `requestedMixCoreNum` 解析后的 Mix 组上限：传 0 时取设备全部 Mix 组。纯 AIV Kernel 最多使用 `2*M` 路 AIV，纯 AIC 或 `__mix(1,2)` Kernel 最多使用 `M` 个核或 Mix 组；如果任务数更少，则按实际任务数缩减 launch。

| 版本 | 同一 stream 中的提交顺序 | 主要任务 | 各 Kernel 的 launch 上限 |
| --- | --- | --- | --- |
| v0 | `Prepare -> StateUpdate -> LocalOutput` | `B*Tc`、`B*4`、`B*Tc*4` | 三个 Mix Kernel 均为 `M` 个 Mix 组 |
| v1 | `Prepare -> StateOutput` | `B*Tc` 个 AIV task、`B*4` 个 Mix task | Prepare 为 `2*M` 路 AIV；StateOutput 为 `M` 个 Mix 组 |
| v2/v3 | `Prepare -> StateOutput` | `ceil(B*Tc/2)` 个 Prepare Mix task、`B*4` 个 State Mix task | 两个 Kernel 均为 `M` 个 Mix 组 |

Kernel 在同一 stream 中依次提交，前一个 Kernel 完成后，后一个 Kernel 才读取其 workspace。

### 任务和 1C2V 分工

文档中的 AIC 侧 DvTile 对应源码常量 `DV_TILE=32`；由于 `VALUE_DIM=128`，状态列块数为 `DV_TILE_COUNT=VALUE_DIM/DV_TILE=4`。一个 Mix 组包含两路 AIV，因此 AIV 侧 DvTile 为 `AIV_DV_TILE=DV_TILE/2=16`。这组 Dv 切分与序列方向的 `ChunkSize=C` 无关。

| 阶段 | AIV0/AIV1 如何分工 |
| --- | --- |
| v0 Prepare | 两路合作处理同一个 Chunk，各写 `Dk/2=64` 个通道；AIV0 额外计算 M 和 A |
| v1 Prepare | 纯 Vector Kernel，每个 AIV block 独立处理一个完整 Chunk |
| v2/v3 Prepare | 同组两路 AIV 各处理一个完整 Chunk，AIC 依次处理两份数据 |
| State | 两路合作处理同一个 `DV_TILE`，各维护 `AIV_DV_TILE` 列 state 和 R |
| Output | v0～v2 由两路 AIV 各写 `AIV_DV_TILE` 列；v3 由 AIC 一次写完整 `DV_TILE` |

状态推进中，两路 AIV 把各自的 state 或 R 列块写到共享 L1 的相邻地址，AIC 等两路完成后按一个完整 `DV_TILE` 读取。AIC 返回需要由 AIV 消费的 U、prediction、delta、history 或 local 时，Fixpipe 再沿 Dv 列把结果均分到两路 AIV。这个过程只是按地址组合和拆分，不做额外的数值拼接或求和。

组内 CrossCore 事件负责确认两路数据何时可读或可复用，核内 Mutex 管理 MTE、Cube、Fixpipe 和 Vector 对同一片上缓冲区的使用顺序。具体 flag、槽位和尾部排空方式见各版本 README。

## 分阶段优化

| 版本 | 上一阶段的问题 | 本版改动 | 默认 C32 workspace | 结果与限制 |
| --- | --- | --- | ---: | --- |
| [v0](src/v0/README.md) | 建立三阶段 Chunk KDA 基线 | 三个 Mix Kernel，单槽执行，阶段间经 GM workspace 交接 | 68096 B/Chunk | 支持可变 B/S 和尾块；三次 launch、workspace 较大，不排跨 task 流水 |
| [v1](src/v1/README.md) | v0 中间量多，LocalOutput 单独 launch | Prepare 改为纯 AIV；StateUpdate 与 LocalOutput 合并；删除 W/U/R/O_history workspace | 29184 B/Chunk | 相对 v0 加速 1.8604x；Pair/A 仍在 AIV，每个 Mix task 内只推进一条状态列链 |
| [v2](src/v2/README.md) | v1 Prepare 的点积和重复 Exp 占用 Vector | Pair/Araw 改由 Cube 计算；Prepare 双槽；StateOutput 根据任务量并行 1、2 或 4 条状态列链 | 29184 B/Chunk | 相对 v1 加速 1.5508x；同一 Batch 的 Chunk 仍保持递推顺序，不开放 C64 |
| [v3](src/v3/README.md) | v2 重复计算 `K_plus@state`，输出还经过 AIV 写回 | W 前移到 Prepare；每项 Mmad 从 6 次减为 5 次；history/local 在 L0C 累加并直接写 O | 29184 B/Chunk | 相对 v2 加速 1.2465x；R1/R2/R4、尾块和强/混合衰减回归通过 |

### v0：三 Kernel 正确性基线

v0 先固定 Chunk 公式、状态方向和中间量生命周期。三个 Kernel 顺序执行，前一阶段完成后下一阶段才能读取其结果：

```text
Prepare(batch,chunk)
  -> W、U、A、Q_plus、K_tail、G_last
  -> StateUpdate(batch,dvTile)
       -> R、O_history、state_out
       -> LocalOutput(batch,chunk,dvTile)
            -> O_history + A@R -> O
```

每个 Kernel 内使用单槽资源。StateUpdate 将 Dv 分成四个 `DV_TILE` task，使 `B=1` 时仍有四条独立的状态列链。

本版保留三次 launch 和较大的 GM workspace，W、U、R、O_history 和 state 的阶段边界清晰。它是后续版本的功能与性能基线，详细数据路径见 [v0 README](src/v0/README.md)。

### v1：合并状态与输出

v1 把 LocalOutput 合并进 StateOutput，并让 Prepare 只生成后续状态计算需要的只读 Chunk 数据：

```text
Prepare, __vector__
  -> K_plus、Q_plus、K_tail、M、A、state_decay
  -> StateOutput, __mix(1,2)
       -> O、final_state
```

StateOutput 使用 `M@(K_plus@state)` 计算 prediction，U、R、history 和 local 都只在片上存活。双槽让相邻 Chunk 的搬运和计算可以交错，L0C 四槽吸收 Mmad 与 Fixpipe 的局部速率差。

本版删除一次 Kernel launch，并将默认 C32 workspace 降到 29184 B/Chunk。Prepare 的 Pair/A 点积仍在单个 AIV 上执行。StateOutput 的不同 `(batch,dvTile)` task 可以分配到不同 Mix 组，但每个 Mix task 内只顺序推进一条状态列链，没有 v2 的多 lane 滚动调度；实现细节见 [v1 README](src/v1/README.md)。

### v2：Cube Prepare 与多条状态列链

v2 将 Prepare 的 Pair/Araw 点积改由 Cube 计算。两路 AIV 各准备一个完整 Chunk，AIC 依次计算两份 Pair/Araw，再把每份完整结果返回对应 AIV。

```text
Prepare:     VP(AIV) -> C(AIC) -> VS(AIV)
StateOutput: C1(AIC) -> V1(AIV) -> C2(AIC) -> V2(AIV)
```

Prepare 通过双槽错开发射 VP、Cube 和 VS。StateOutput 根据每个 AIC 分到的任务数，同时滚动 1、2 或 4 条独立的 `(batch,dvTile)` 状态链；同一 wave 内与 Dv 无关的 Chunk 数据只搬入一次。

本版降低了 Prepare 的 Vector 工作量，也减少了 StateOutput 的重复搬运。同一 Batch 内的 Chunk 仍严格按递推顺序处理，C64 强衰减路径不开放；完整调度见 [v2 README](src/v2/README.md)。

### v3：W 前移和输出直写

v3 在 Prepare 计算 `W=M@K_plus`，让同一 Batch/Chunk 的四个 `DV_TILE` 共用 W。StateOutput 直接执行 `prediction=W@state`，删除 KPlusState 的 Fixpipe、L1 暂存和回读。

输出也改由 AIC 完成：

```text
history -> output L0C
local   -> 同一 L0C 原地累加 -> Fixpipe -> O(GM)
```

多条状态列链按递推依赖重新排序：

```text
AIC: C1Pre(new) -> C2Core(old) -> C1Post(new) -> Output(old)
AIV: V2(old) -> V1(new)
```

Prepare 因新增 W 略有增长，StateOutput 每项 Mmad 从 6 次降到 5 次，并删除 AIV 的逐 Chunk 输出链路。性能对比合并统计两个 Kernel；完整同步与资源布局见 [v3 README](src/v3/README.md)。

## 工程结构

```text
kimi_delta_attn_lite_story/
├── CMakeLists.txt
├── README.md
├── requirements.txt
├── scripts/
│   ├── kimi_delta_attn_lite_gendata.py
│   ├── kimi_delta_attn_lite_verify.py
│   └── kdalite_thread_limit.py
└── src/
    ├── kimi_delta_attn_lite.h
    ├── kimi_delta_attn_lite_demo.cpp
    ├── v0/
    │   ├── README.md
    │   ├── host/
    │   └── kernel/
    ├── v1/
    ├── v2/
    └── v3/
```

`kimi_delta_attn_lite.h` 和 `kimi_delta_attn_lite_demo.cpp` 由所有版本共用。每个 `src/vN` 目录保留该版本的 Host tiling、workspace 布局、Kernel 入口和 AIC/AIV 实现。CMake 自动识别版本目录并生成 `kdalite_vN`。

## 环境准备与编译运行

### 安装依赖与编译

工程默认编译架构为 `dav-3510`。CANN Toolkit 不在默认安装目录时，先设置：

```bash
export ASCEND_HOME_PATH=/path/to/ascend-toolkit/latest
```

数据生成和 Golden 校验依赖 NumPy 与 `ml_dtypes`：

```bash
python3 -m pip install -r Samples/2_Performance/kimi_delta_attn_lite_story/requirements.txt
```

Golden 的 `auto` 模式在 `B>=32` 且 `B*S>=262144` 时优先使用 CPU 版 PyTorch，其余规格使用 NumPy；也可以通过 `KDA_VERIFY_BACKEND=torch|numpy` 显式选择后端。PyTorch 未安装时会回退到 NumPy，因此不是必选依赖。`KDA_PYTHON_THREADS` 用于限制 Golden 的 CPU 线程数。

在 cann-samples 根目录执行：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510

# 构建全部版本，两个聚合 target 作用相同。
cmake --build build --target kdalite -j
cmake --build build --target kimi_delta_attn_lite_story -j

# 也可以只构建指定版本。
cmake --build build --target kdalite_v0 -j
cmake --build build --target kdalite_v1 -j
cmake --build build --target kdalite_v2 -j
cmake --build build --target kdalite_v3 -j
```

可执行文件和共用 Python 脚本位于 `build/Samples/2_Performance/kimi_delta_attn_lite_story/`。

### 运行与校验

以下小规格会生成 Recurrent Golden，并校验 O 与 `final_state`：

```bash
./build/Samples/2_Performance/kimi_delta_attn_lite_story/kdalite_v0 --size 2 65
./build/Samples/2_Performance/kimi_delta_attn_lite_story/kdalite_v1 --size 2 65
./build/Samples/2_Performance/kimi_delta_attn_lite_story/kdalite_v2 --size 2 65
./build/Samples/2_Performance/kimi_delta_attn_lite_story/kdalite_v3 --size 2 65
```

快速执行示例使用 `--dry-run --size 32 4096`。该规格用于检查构建和执行，不是性能表的采集规格：

```bash
./build/Samples/2_Performance/kimi_delta_attn_lite_story/kdalite_v0 --dry-run --size 32 4096
./build/Samples/2_Performance/kimi_delta_attn_lite_story/kdalite_v1 --dry-run --size 32 4096
./build/Samples/2_Performance/kimi_delta_attn_lite_story/kdalite_v2 --dry-run --size 32 4096
./build/Samples/2_Performance/kimi_delta_attn_lite_story/kdalite_v3 --dry-run --size 32 4096
```

命令行参数：

- `--size B S`：设置 Batch 和序列长度，默认值为 `B=1,S=16`。
- `--core-num n`：设置 launch 使用的 Mix 组数上限，不能超过本卡 Mix 组数。纯 AIV Kernel 最多使用 `2*n` 路 AIV，纯 AIC 或 Mix Kernel 最多使用 `n` 个核或 Mix 组；任务不足时按实际任务数缩减。不传时使用设备全部 Mix 组；该参数主要用于仿真，真机性能测试通常不传。
- `--dry-run`：仍生成输入、执行 Kernel、D2H 并写出 O 与 final_state，只跳过 Golden 和比对。

不使用 `--dry-run` 时，程序从已经落盘的 BF16 Q/K/V/beta 和 FP32 `log_decay` 重新计算 Recurrent Golden。O 在量化为 BF16 后比较，final_state 按 FP32 比较：

```text
abs(npu-golden) <= 2^-6 + 2^-6 * abs(golden)
```

两项输出均要求全部元素通过，NaN 或 Inf 直接判定失败。各版本的数据目录为 `build/Samples/2_Performance/kimi_delta_attn_lite_story/data/kdalite_vN`，主要文件包括：

```text
q.bin / k.bin / v.bin
log_decay.bin / beta.bin
npuout_o.bin / npuout_final_state.bin
golden_o.bin / golden_final_state.bin
```

环境变量 `KDA_DATA_CASE` 可选择 `random`、`beta_zero`、`beta_one`、`no_decay`、`strong_decay` 或 `mixed_decay`：

```bash
KDA_DATA_CASE=strong_decay ./build/Samples/2_Performance/kimi_delta_attn_lite_story/kdalite_v3 --size 32 65
```

### 选择 ChunkSize

每个版本的默认 ChunkSize 只在自己的 `kimi_delta_attn_lite_common.h` 中定义。CMake 仅在显式传入版本选项时覆盖源码默认值：

```bash
cmake -S . -B build-kdalite-v3-c16 \
    -DNPU_ARCH=dav-3510 \
    -DKDALITE_V3_CHUNK_SIZE=16
cmake --build build-kdalite-v3-c16 --target kdalite_v3 -j
./build-kdalite-v3-c16/Samples/2_Performance/kimi_delta_attn_lite_story/kdalite_v3 --size 1 17
```

| 版本 | CMake 选项 | 合法值 |
| --- | --- | --- |
| v0 | `KDALITE_V0_CHUNK_SIZE` | 16、32、64 |
| v1 | `KDALITE_V1_CHUNK_SIZE` | 16、32、64 |
| v2 | `KDALITE_V2_CHUNK_SIZE` | 16、32 |
| v3 | `KDALITE_V3_CHUNK_SIZE` | 16、32 |

构建目录会在 `CMakeCache.txt` 中保留显式设置。恢复源码默认值时，可以使用新的构建目录，也可以在重新配置时删除对应 cache，例如 `cmake -S . -B build -U KDALITE_V3_CHUNK_SIZE`。

### CANNsim

CANNsim 对多轮组内 CrossCore 同步的支持有限。仿真时应使用独立构建目录并打开 `SIM_COMPATIBLE=ON`；该选项只把 CrossCore 切换为 mode4 兼容映射，不改变算法和数据切分：

```bash
cmake -S . -B build-kdalite-sim \
    -DNPU_ARCH=dav-3510 \
    -DSIM_COMPATIBLE=ON
cmake --build build-kdalite-sim --target kdalite_v3 -j

cannsim record -s Ascend950 -g \
    -o ./build-kdalite-sim/cannsim/kdalite_v3_1_512 \
    "./build-kdalite-sim/Samples/2_Performance/kimi_delta_attn_lite_story/kdalite_v3 --dry-run --core-num 1 --size 1 512"
```

仿真结果用于检查发射顺序、同步和组件重叠，性能结论以真机 profile 为准。

## 精度验证

四个版本都使用独立的 FP32 Recurrent KDA Golden。精度运行不带 `--dry-run`，同时比较 BF16 O 和 FP32 `final_state`：

```text
abs(npu-golden) <= 2^-6 + 2^-6 * abs(golden)
```

两项输出都要求全部元素通过，NaN 或 Inf 直接判定失败。尾块、`beta=0/1`、无衰减、强衰减和混合衰减的具体覆盖范围见各版本 README。

## 性能参考

四个版本使用同一口径采集：CANN 9.2、`dav-3510`、默认 C32、不传 `--core-num`（使用设备全部 32 个 Mix 组，即 32 AIC/64 AIV）、1650 MHz，运行参数为 `--dry-run --size 32 65536`。

每个版本不指定 `--kernel-name`，在一次应用运行中采集该版本的全部 Kernel。重复三次后，先对每个 Kernel 的 `Task Duration` 取中位数，再计算合计。

| 版本 | Prepare (us) | StateUpdate/StateOutput (us) | LocalOutput (us) | Kernel Task Duration 合计 (us) | 相对前版 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0 | 17472.701172 | 15906.120117 | 7940.161133 | 41318.982422 | 基线 |
| v1 | 8816.471680 | 13393.181641 | - | 22209.653321 | `-46.2483%`，`1.8604x` |
| v2 | 4767.770996 | 9553.227539 | - | 14320.998535 | `-35.5190%`，`1.5508x` |
| v3 | 4945.325195 | 6543.591797 | - | 11488.916992 | `-19.7757%`，`1.2465x` |

从 v0 到 v3，Kernel Task Duration 合计下降 72.1946%，加速 3.5964x。v3 把 W 前移到 Prepare，因此 Prepare 相对 v2 增长 3.7241%；StateOutput 删除一段 Mmad 和 AIV 输出链路，下降 31.5039%。

这些数据来自设备任务统计，合计不是 Host 端到端耗时，也不包含数据生成、H2D/D2H、Kernel launch、Golden 和比对。`--dry-run` 仍会执行 Kernel、回读并落盘输出，只跳过 Golden 和精度比对。

## 当前实现边界

- 当前接口固定 `N=1`、`Dk=Dv=128` 和零初始状态；`Dk==Dv` 是源码布局限制，不是 KDA 公式的限制。
- v0/v1 支持编译期 C16/C32/C64，v2/v3 只开放 C16/C32。v0/v1 的 C64 仅通过较弱随机衰减数据，v2/v3 的 C64 强衰减路径未达到精度要求。
- 当前不支持多 Head、非零初态、连续 Prefill、Decode、packed/varlen、训练反向，以及 KDA core 之外的投影、卷积、归一化和门控计算。
- 性能表比较同一采集口径下的 Kernel `Task Duration`。组件 active time 可以重叠，不能直接相加推导 Kernel 耗时。
