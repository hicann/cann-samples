# moe dispatch 和 combine 通信算子性能优化实践与效果分析

本示例演示 **MoE（Mixture-of-Experts）场景下的分布式 Dispatch + Combine** 的端到端流程：

- **Dispatch**：按 `expert_ids`（TopK 路由结果）将 token 特征从各 rank “发往”目标 expert（跨 rank all-to-all），并输出用于 Combine 的辅助信息（如 `expand_idx`、`ep_recv_count`、`expert_token_nums` 等）。
- **Combine**：根据 Dispatch 生成的 `assist_info_for_combine` / `ep_recv_count` 等信息，将 expert 侧处理后的 `expand_x` 按 TopK 权重 `expert_scales` 汇聚回原 token（按 rank 分布式 all-to-all + 本地按 TopK sum）。

## 运行环境与约束

- **硬件**：NPU多卡环境。
- **芯片型号**：默认`--npu-arch=dav-3510`。

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

运行过程中会在当前目录生成：

- `input/`：输入 bin（按 `chip_{rankId}` 分目录）
- `golden/`：golden bin（用于精度对比）
- `output/`：算子输出 bin（按 `chip_{rankId}` 分目录）

---

## 生成测试数据（input + golden）

在该目录下执行（默认值可以不填写）：

```bash
python3 scripts/gen_data.py \
  --chip-num-per-server 2 \
  --bs 8 \
  --h 7168 \
  --k 8 \
  --token-dtype-choice 1 \
  --quant-mode 4 \
  --expert-recv-info-type 1
```

参数均有默认值，只需要填写与默认值不一样的：
```bash
python3 scripts/gen_data.py --chip-num-per-server 2 --bs 8
```

- **`--bs`**：算子入参bs 大小。
- **`--h`**：0 算子入参hidden size大小。
- **`--k`**：0 算子入参topk 大小。
- **`--chip-num-per-server`**：生成的 rank 数。
- **`--token-dtype-choice`**：0 表示 bfloat16，1 表示 float16（默认 1）。
- **`--quant-mode`**：0 表示不量化，4 表示 MXFP8动态量化。

## 编译

该示例通过 CMake 构建目标 `moe_dispatch_and_combine_story`（见 `CMakeLists.txt`）。

```bash
cmake -S . -B build
cmake --build build --target moe_dispatch_and_combine_story
```

---

## 运行（Dispatch + Combine）

在包含 `input/` 的运行目录下执行（建议就在本目录运行）：

```bash
./moe_dispatch_and_combine_story <rankNum> <bs>
```

示例：

```bash
./moe_dispatch_and_combine_story 2 8
```

---

## 精度验证（output vs golden）

运行完成后，在本目录执行（路径默认，可以不填写）：

```bash
python3 scripts/verify_result.py --golden-dir ./golden --op-out-dir ./output
```

该脚本会逐个比对 `golden/**/*.bin` 与 `output/**/*.bin`（按相对路径对应），并打印每个 bin 的一致性结果。

---

