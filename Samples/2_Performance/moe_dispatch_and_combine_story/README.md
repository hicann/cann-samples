# moe dispatch 和 combine 通信算子性能优化实践与效果分析


## 背景介绍

在大模型训练与推理中，MoE（Mixture-of-Experts）架构凭借动态专家激活带来的计算稀疏性优势，以及在千亿参数规模下的高吞吐推理能力，已成为超大规模模型的关键技术路线。MoE 主要通过 `Dispatch` 与 `Combine` 两个核心过程，实现输入 token 的动态分发与多专家输出的高效聚合，从而在维持海量参数规模的同时获得较高计算效率。但随着专家并行（EP）规模持续扩大，专家之间更频繁的数据交换会带来显著的通信开销，逐步演化为影响端到端推理性能的主要瓶颈。

在整网推理过程中，各层的 TopK 路由结果会动态变化。若采用传统的 `alltoallv` 通信流程，通常需要先交换各 rank 的发送量信息，再通过排序/重排将发往同一专家的数据聚集，最后再进行一次 `alltoallv` 发送 token 数据。这一流程链路长、同步点多，整体效率不高。因此，我们考虑将 `Dispatch/Combine` 的数据收发过程进一步细粒度拆分：基于共享内存的写入机制按 token 逐个发送与接收，通过流水并行（pipeline）重叠通信与数据整理开销，以获得更好的整体性能。

## Dispatch 功能介绍

- 完成 token 分发：根据 token 选中的专家，将 token 发送到对应卡上。
- 为支持后续 FFN 计算，将各卡发送过来的 token 连续重排。
- 为支持后续 `Combine` 处理，统计接收 token 信息，支撑 `Combine` 将 FFN 计算后的 token 发送回源端。
- 将量化提前到通信之前，将量化后的 `int8 token` 与 `fp32 scale` 合并发送，并在接收端重排分离。

### 计算步骤

1. **循环发送数据（`写UB -> quant -> 写远端内存` 流水并行）**
   - 从 HBM 搬运待发送数据到 AIV 核内的 UBuffer。
   - 对 UBuffer 上的 token 进行量化，并将量化后的 `int8 token` 与 `fp32 scale` 拼接。
   - 将源端信息（`rank_id`、`bs_id`、`k_offset`，即三元组）拼接到量化后的 UBuffer。
   - 根据 `expert_ids` 查找对端 rank 地址，并通过 `AIV + UBmem` 执行 `写远端` 发送。

2. **发送 `flag` 标识与 `count` 到所有对端**
   - 根据 `expert_id` 统计发往每个 expert 的 token 数（分核处理）。
   - 统计后执行 `SyncAll` 多核同步；确认所有核发送完成后，将完成 `flag` 与 `count` 通过 `AIV + UBmem` 发送到对端。

3. **等待对端写入完成并计算偏移**
   - 分核读取状态区 `flag`，直到全部为 `1`（表示对应对端已完成发送）。
   - 读取状态区 `count`，计算各 rank 数据搬运到输出中的偏移。
   - 执行 `SyncAll` 多核同步。

4. **本地数据整理**
   - 分核并行将通信 Shared Memory 中的数据整理到最终输出（`data`、`scale`、`expert token num`）。
   - 输出各 expert 的 token 数量。

## Combine 功能介绍

- 将 FFN 计算后的 token 发送回源端。
- 将选中的 `K`（MoE 专家数）个 FFN 结果加权求和，完成 `combine` 计算。

### 计算步骤

1. **循环发送数据与发送 `flag`（按总 token 数分核）**
   - 从 `recv_count` 读取总 token 数，并平均分配到各核。
   - 每个 token 携带三元组信息（`rank_id`、`bs_id`、`k_offset`）；发送时根据三元组计算对端地址偏移。
   - 使用 `AIV + UBmem` 将数据发送到对端。
   - 每个 token 发送完成后，按 token 发送 `flag`。

2. **循环等待并执行 `combine`（按 batch size 分核）**
   - 按 batch 循环处理。
   - 等待对应 batch 状态（`K+1` 个状态位全部为 `1`）。
   - 状态齐备后，从通信 Shared Memory 搬运各 FFN 结果。
   - 从输入 HBM 搬运 `expert scale`。
   - 执行 `combine`：对各 MoE FFN 结果乘以 `expert scale` 后求和，再叠加 shared FFN 结果。


## 示例演示

本示例演示 **MoE（Mixture-of-Experts）场景下的分布式 Dispatch + Combine** 的端到端流程：

- **Dispatch**：按 `expert_ids`（TopK 路由结果）将 token 特征从各 rank “发往”目标 expert（跨 rank all-to-all），并输出用于 Combine 的辅助信息（如 `expand_idx`、`ep_recv_count`、`expert_token_nums` 等）。
- **Combine**：根据 Dispatch 生成的 `assist_info_for_combine` / `ep_recv_count` 等信息，将 expert 侧处理后的 `expand_x` 按 TopK 权重 `expert_scales` 汇聚回原 token（按 rank 分布式 all-to-all + 本地按 TopK sum）。

## 运行环境与约束

- **硬件**：NPU多卡环境。
- **芯片型号**：默认`--npu-arch=dav-3510`。
- **软件环境**：需要已安装 Ascend CANN Toolkit，并在构建和运行前加载 Toolkit 环境变量。

---

## 目录结构

```text
moe_dispatch_and_combine_story/
├─ CMakeLists.txt
├─ include/                             # device侧核函数
│  ├─ moe_distribute_dispatch.h
│  ├─ moe_distribute_dispatch_quant.h
│  └─ moe_distribute_combine.h
├─ src/
│  ├─ dispatch_and_combine_final.cpp    # 终极性能版本
│  └─ utils.h
└─ scripts/
   ├─ gen_data.py                       # 生成输入与 golden
   └─ verify_result.py                  # 比对输出与 golden
```

在本示例目录运行本示例，运行过程中会在本示例目录生成：

- `input/`：输入 bin（按 `chip_{rankId}` 分目录）
- `golden/`：golden bin（用于精度对比）
- `output/`：算子输出 bin（按 `chip_{rankId}` 分目录）

---

## 环境准备

构建和运行前，需要先加载 Ascend Toolkit 环境变量。若使用 root 用户按默认路径安装，可执行：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

如果 Toolkit 安装在自定义路径，请将上述路径替换为实际安装目录下的 `set_env.sh`。建议在同一个 shell 会话中完成后续的构建、数据生成、运行和校验，避免环境变量丢失。

---

## 构建

在本示例目录执行 CMake 构建，目标为 `moe_dispatch_and_combine_story`（见本示例目录下的 `CMakeLists.txt`）。构建完成后，可执行文件会生成到 `build/Samples/2_Performance/moe_dispatch_and_combine_story/` 目录下。以下命令中的 `${cann_samples_path}` 表示用户本地 `cann-samples` 仓库所在目录，请根据实际路径替换。

```bash
cd ${cann_samples_path}/Samples/2_Performance/moe_dispatch_and_combine_story
cmake -S ../../../ -B ../../../build
cmake --build ../../../build --target moe_dispatch_and_combine_story
```

---

## 生成测试数据（input + golden）及 output

构建完成后，在本示例目录生成一组用于运行和精度校验的测试数据（命令行未传入的参数使用默认值）：

```bash
python3 ./scripts/gen_data.py --chip-num-per-server 2 --bs 8
```

该脚本会在本示例目录生成 `input/` 和 `golden/`。其中 `input/` 作为算子运行输入，`golden/` 作为后续 `verify_result.py` 的精度比对基准。算子运行后会在本示例目录生成 `output/`，用于保存实际运行输出。

脚本参数均有默认值，只需要传入与默认值不同的配置。常用参数如下：

- **`--bs`**：算子入参 batch size 大小。
- **`--h`**：算子入参 hidden size 大小。
- **`--k`**：算子入参 topk 大小。
- **`--chip-num-per-server`**：生成的 rank 数。
- **`--token-dtype-choice`**：0 表示 bfloat16，1 表示 float16（默认 1）。
- **`--quant-mode`**：0 表示不量化，4 表示 MXFP8动态量化。

---

## 运行（Dispatch + Combine）

测试数据生成后，在本示例目录运行构建产物。命令行参数依次为 `rankNum` 和 `bs`，需要与生成数据时的 `--chip-num-per-server`、`--bs` 保持一致：

```bash
../../../build/Samples/2_Performance/moe_dispatch_and_combine_story/moe_dispatch_and_combine_story <rankNum> <bs>
```

例如前面生成的是 2 张卡、batch size 为 8 的数据，则执行：

```bash
../../../build/Samples/2_Performance/moe_dispatch_and_combine_story/moe_dispatch_and_combine_story 2 8
```

运行完成后，算子输出会写入本示例目录下的 `output/`，并按 `chip_{rankId}` 分目录保存。

---

## 精度验证（output vs golden）

算子运行结束后，在本示例目录执行精度校验脚本：

```bash
python3 ./scripts/verify_result.py
```

该脚本默认读取本示例目录下的 `golden/` 与 `output/`，逐个比对 `golden/**/*.bin` 与 `output/**/*.bin`（按相对路径对应），并打印每个 bin 的一致性结果。

---
