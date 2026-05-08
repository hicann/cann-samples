# MXFP4量化矩阵乘算子

## 概述

本示例展示了MXFP4量化矩阵乘算子在昇腾AI处理器上的完整实现，包含基于SWAT模板的高性能优化方案。MXFP4是一种4位浮点数量化格式，通过GroupSize=32的分组量化策略，在兼顾模型精度的同时显著降低访存开销和计算成本，适用于大语言模型推理等场景。

当前目录提供以下能力：

- `quant_matmul_mxfp4_swat`：基于SWAT模板、双L1缓冲（2-buffer）的实现。
- `quant_matmul_mxfp4_swat_4_buffer`：四L1缓冲（4-buffer）的实现。
- `quant_matmul_mxfp4_a_full_load`：A矩阵full load方案的实现。
- `gen_data.py`：生成输入数据和CPU golden结果。
- `verify_result.py`：校验NPU输出与CPU golden是否一致。
- `quant_matmul_mxfp4_algorithm_recommend.py`：对当前目录下可执行算法进行兼容性筛选和耗时排序。

## 使用约束

当前样例支持以下场景：

- 支持通过命令行参数`transA`/`transB`选择A/B矩阵转置。
- FP4打包沿输入矩阵内轴进行，内轴长度必须为偶数：
  - A：`transA=0`时内轴为`K`（要求`k`为偶数）；`transA=1`时内轴为`M`（要求`m`为偶数）。
  - B：`transB=1`时内轴为`K`（要求`k`为偶数）；`transB=0`时内轴为`N`（要求`n`为偶数）。

## 支持架构

NPU ARCH 3510

## 性能优化指南

关于算子涉及的模板实现及优化策略，请参考[MX量化矩阵乘算子性能优化指南](../../../docs/quant_matmul_mx_performance.md)

## API参考

[Ascend C API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0003.html)

## 参数说明

各可执行文件的命令行参数格式一致：

```text
<program> m k n [transA transB]
```

- `m`：矩阵A的行数。
- `k`：矩阵A的列数，同时也是矩阵B的归约维。
- `n`：矩阵B的行数，对应输出矩阵的列数。
- `transA`（可选）：A矩阵转置信息（`0/1/true/false/t/f`）。`0`/`false`/`f`表示非转置，shape为`[M, K]`；`1`/`true`/`t`表示转置，shape为`[K, M]`。默认为非转置。
- `transB`（可选）：B矩阵转置信息（`0/1/true/false/t/f`）。`0`/`false`/`f`表示非转置，shape为`[K, N]`；`1`/`true`/`t`表示转置，shape为`[N, K]`。默认为转置。

输出矩阵C的逻辑形状为`[M, N]`。

## 数据与校验

`gen_data.py`会在当前目录下生成以下文件：

- `input/input_a.bin`
- `input/input_b.bin`
- `input/input_scaleA.bin`
- `input/input_scaleB.bin`
- `output/cpu_output.bin`

样例执行完成后会额外生成：

- `output/npu_out.bin`

各可执行文件在运行结束后都会自动调用`verify_result.py`，将NPU输出与CPU golden进行一致性校验。

## 一键运行（推荐）

仓库提供`run.sh`（位于`matmul_recipes/examples/quant_matmul_mxfp4/scripts/`），可一键串联**构建 → 数据生成 → 算子执行 → 结果校验**全流程。
推荐先进入样例目录再执行，命令更短：

```bash
cd Samples/2_Performance/matmul_story/matmul_recipes/examples/quant_matmul_mxfp4

# 自动构建 + 自动推荐最优算法 + 运行
bash scripts/run.sh 16 128 16384 0 1

# 指定目标可执行文件，跳过重新构建
bash scripts/run.sh \
  --target quant_matmul_mxfp4_a_full_load --skip-build 16 128 16384 0 1

# 查看完整帮助
bash scripts/run.sh --help
```

### run.sh参数说明

| 参数 | 说明 |
|------|------|
| `m k n [transA transB]` | 矩阵维度与转置参数。`transA/transB`可选，支持`0/1/true/false/t/f`；省略时默认`transA=false(0)`、`transB=true(1)`。FP4打包约束：A/B的内轴长度须为偶数（由`transA/transB`决定是`m`、`k`或`n`）。 |
| `--target <name>` | 指定要运行的可执行文件名（如`quant_matmul_mxfp4_swat`、`quant_matmul_mxfp4_swat_4_buffer`、`quant_matmul_mxfp4_a_full_load`）。省略时自动调用推荐脚本选择最优目标。 |
| `--skip-build` | 跳过构建/安装阶段，复用已有`build_out`。 |
| `-h, --help` | 显示帮助信息。 |

如需查看完整算法推荐排名（含耗时表格），请在安装目录下直接运行`quant_matmul_mxfp4_algorithm_recommend.py`（见下文「手动构建与运行」）。

## 手动构建与运行

如需手动控制各步骤，可在仓库根目录下完成编译和安装后，进入当前样例目录：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --parallel
cmake --install build --prefix ./build_out
cd build_out/2_Performance/matmul_story/matmul_recipes/quant_matmul_mxfp4
```

### 1. 生成测试数据

```bash
python3 gen_data.py 16 128 16384 0 1
```

### 2. 运行单个算法样例

```bash
./quant_matmul_mxfp4_swat 16 128 16384 0 1
```

或：

```bash
./quant_matmul_mxfp4_swat_4_buffer 16 128 16384 0 1
```

或：

```bash
./quant_matmul_mxfp4_a_full_load 16 128 16384 0 1
```

### 3. 运行算法推荐脚本

```bash
python3 quant_matmul_mxfp4_algorithm_recommend.py 16 128 16384 0 1
```

下图为推荐脚本输出的**结构示意**（数值为虚构，仅说明版式）：

```text
[Profile Breakdown]
+------------------------------------+----------+---------+----------+---------+---------+------------+--------------+
| candidate                          |kernel(us)| mac(us) |scalar(us)| mte1(us)| mte2(us)|fixpipe(us) |icache_miss(%)|
+====================================+==========+=========+==========+=========+=========+============+==============+
| quant_matmul_mxfp4_swat            |    12.345|   1.234 |     0.567|   0.123 |   0.456 |     0.789 |        0.100 |
| quant_matmul_mxfp4_swat_4_buffer   |    11.900|   1.200 |     0.550|   0.110 |   0.440 |     0.770 |        0.095 |
| quant_matmul_mxfp4_a_full_load     |    15.678|   2.100 |     0.800|   0.200 |   0.300 |     0.500 |        0.250 |
+------------------------------------+----------+---------+----------+---------+---------+------------+--------------+
```