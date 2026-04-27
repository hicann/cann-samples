# MXFP4矩阵乘分步教程

## 概述

本目录提供从基础实现到高性能实现的分步教程，覆盖 `0_naive` 到 `7_fullload` 共 8 个 Step。  
教程可执行文件参数统一为：

```text
<program> m k n
```

- `m`：矩阵 A 的行数
- `k`：矩阵 A 的列数，同时也是矩阵 B 的行数（要求为偶数）
- `n`：矩阵 B 的列数

## Step 列表

- `matmul_tutorial_mxfp4_base`（Step 0）
- `matmul_tutorial_mxfp4_pingpong`（Step 1）
- `matmul_tutorial_mxfp4_swat`（Step 2）
- `matmul_tutorial_mxfp4_swat_balance`（Step 3）
- `matmul_tutorial_mxfp4_swat_unitflag`（Step 4）
- `matmul_tutorial_mxfp4_half1l1_ping_halfl1_pong`（Step 5）
- `matmul_tutorial_mxfp4_memery_access_coalescing`（Step 6）
- `matmul_tutorial_mxfp4_a_fullload`（Step 7）

## 一键运行（推荐）

仓库提供 `scripts/run.sh`，可一键串联 **构建 → 数据生成 → 算子执行 → 结果校验** 全流程。
建议先进入 `matmul_tutorials/` 目录再执行：

```bash
cd Samples/2_Performance/matmul_story/matmul_tutorials

# 指定 Step 运行
bash scripts/run.sh --target matmul_tutorial_mxfp4_swat 256 256 256

# 自动构建 + 未指定 target 时运行 Step 0（matmul_tutorial_mxfp4_base）
bash scripts/run.sh 256 256 256

# 跳过构建阶段
bash scripts/run.sh --target matmul_tutorial_mxfp4_swat_balance --skip-build 256 256 256

# 查看完整帮助
bash scripts/run.sh --help
```

### run.sh 参数说明

| 参数 | 说明 |
|------|------|
| `m k n` | 矩阵维度（必填）。`k` 须为偶数。 |
| `--target <name>` | 指定要运行的教程可执行文件名；省略时默认 Step 0（`matmul_tutorial_mxfp4_base`） |
| `--skip-build` | 跳过构建/安装阶段，复用已有 `build_out`。 |
| `-h, --help` | 显示帮助信息。 |

## 手动构建与运行

如需手动控制各步骤，在仓库根目录执行编译安装，并进入教程安装目录：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --parallel
cmake --install build --prefix ./build_out
cd build_out/2_Performance/matmul_story/matmul_tutorials
```

随后按以下流程执行：

```bash
# 1) 生成输入数据与 CPU golden
python3 gen_data.py 256 256 256

# 2) 运行某个 Step（示例：Step 2）
./matmul_tutorial_mxfp4_swat 256 256 256
```

## 相关文档

- 顶层说明：[`../README.md`](../README.md)
- 分步优化说明与流水图：[`../docs/quant_matmul_mxfp4_tutorials.md`](../docs/quant_matmul_mxfp4_tutorials.md)
