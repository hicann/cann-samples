# MXFP8量化矩阵乘算子

## 概述

本示例展示了MXFP8量化矩阵乘算子在昇腾AI处理器上的完整实现，包含基于SWAT模板的高性能优化方案。MXFP8是一种8位浮点数量化格式，可在保持较好精度的前提下显著降低带宽开销与访存压力，适用于大语言模型推理等场景。

当前目录提供以下能力：

**样例（可执行目标）**

权重`ND`排布（包括`NDExtLayout`/`DNExtLayout`）

- `quant_matmul_mxfp8_swat`：基于SWAT模板、双L1缓冲（2-buffer）的实现。
- `quant_matmul_mxfp8_swat_4_buffer`：基于SWAT模板、四L1缓冲（4-buffer）的实现。
- `quant_matmul_mxfp8_a_full_load`：A矩阵full load方案的实现。

权重`NZ`排布（权重矩阵B为`NZLayout`/`ZNLayout`，矩阵A数据排布与`ND`保持一致）

- `quant_matmul_mxfp8_swat_weight_nz`：基于SWAT模板、双L1缓冲（2-buffer）的实现。

**辅助脚本（`scripts/` 目录）**

- `gen_data.py`：生成`ND`权重输入数据和CPU golden结果。
- `gen_data_weight_nz.py`：生成`NZ`权重输入数据和CPU golden结果。
- `verify_result.py`：校验NPU输出与CPU golden是否一致。
- `quant_matmul_mxfp8_algorithm_recommend.py`：对当前目录下可执行算法进行兼容性筛选和耗时排序。
- `run.sh`：一键串联构建、数据生成、算子执行与校验（详见下文「一键运行」）。

## 使用约束

当前样例支持以下场景：

- 支持通过命令行参数`transA`/`transB`选择A/B矩阵转置。
- 支持权重矩阵B数据排布`ND`/`NZ`输入。


## 支持架构

NPU ARCH 3510

## 性能优化指南

关于算子涉及的模板实现及优化策略，请参考[MX量化矩阵乘算子性能优化指南](../../../docs/quant_matmul_mx_performance.md)

## API参考

[Ascend C API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0003.html)

## 参数说明

可执行文件的命令行参数格式一致：

```text
<program> m k n [transA transB]
```

- `m`：矩阵A的行数。
- `k`：矩阵A的列数，同时也是矩阵B的归约维。
- `n`：矩阵B的行数，对应输出矩阵的列数。
- `transA`（可选）：A矩阵转置信息（`0/1/true/false/t/f`）。`0`/`false`/`f`表示非转置，shape为`[M, K]`；`1`/`true`/`t`表示转置，shape为`[K, M]`。默认为非转置。
- `transB`（可选）：B矩阵转置信息（`0/1/true/false/t/f`）。`0`/`false`/`f`表示非转置，shape为`[K, N]（ND）`/`[N1, K1, K0, N0]（NZ）`；`1`/`true`/`t`表示转置，shape为`[N, K]（ND）`/`[K1, N1, N0, K0]（NZ）`。其中`K0=32`、`N0=16`，`K1=ceil(K/K0)`，`N1=ceil(N/N0)`。默认为转置。

输出矩阵C的逻辑形状为`[M, N]`。

## 数据与校验

`gen_data.py`与`gen_data_weight_nz.py`会在**运行时的当前工作目录**（通常为安装目录`build_out/.../quant_matmul_mxfp8`）下生成以下文件（二者输出文件名相同，仅B矩阵数据排布方式不同，请勿混用与可执行目标不匹配的数据）：

- `input/input_a.bin`
- `input/input_b.bin`
- `input/input_scaleA.bin`
- `input/input_scaleB.bin`
- `output/cpu_output.bin`

样例执行完成后会额外生成：

- `output/npu_out.bin`

各可执行文件在运行结束后都会自动调用`verify_result.py`，将NPU输出与CPU golden进行一致性校验。

## 一键运行（推荐）

仓库提供`run.sh`（位于`matmul_recipes/examples/quant_matmul_mxfp8/scripts/`），可一键串联**构建 → 数据生成 → 算子执行 → 结果校验**全流程。
推荐先进入样例目录再执行，命令更短：

```bash
cd Samples/2_Performance/matmul_story/matmul_recipes/examples/quant_matmul_mxfp8

# 自动构建 + 自动推荐最优算法 + 运行
bash scripts/run.sh 16 128 16384 0 1

# 指定目标可执行文件，跳过重新构建
bash scripts/run.sh \
  --target quant_matmul_mxfp8_a_full_load --skip-build 16 128 16384 0 1

# 查看完整帮助
bash scripts/run.sh --help
```

### run.sh参数说明

| 参数 | 说明 |
|------|------|
| `m k n [transA transB]` | 矩阵维度与转置参数。`transA/transB`可选，支持`0/1/true/false/t/f`；省略时默认`transA=false(0)`、`transB=true(1)`。 |
| `--target <name>` | 指定要运行的可执行文件名。省略时自动调用推荐脚本选择最优目标。 |
| `--skip-build` | 跳过构建/安装阶段，复用已有`build_out`。 |
| `-h, --help` | 显示帮助信息。 |

如需查看完整算法推荐排名（含耗时表格），请在安装目录下直接运行`quant_matmul_mxfp8_algorithm_recommend.py`（见下文「手动构建与运行」）。

## 手动构建与运行

如需手动控制各步骤，可在仓库根目录下完成编译和安装后，进入当前样例目录：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --parallel
cmake --install build --prefix ./build_out
cd build_out/2_Performance/matmul_story/matmul_recipes/quant_matmul_mxfp8
```

### 1. 生成测试数据

`ND`权重：

```bash
python3 gen_data.py 16 128 16384 0 1
```

`NZ`权重：

```bash
python3 gen_data_weight_nz.py 16 128 16384 0 1
```

### 2. 运行单个算法样例

`ND`权重：

```bash
# 2-buffer SWAT
./quant_matmul_mxfp8_swat 16 128 16384 0 1
# 或：4-buffer SWAT
./quant_matmul_mxfp8_swat_4_buffer 16 128 16384 0 1
# 或：A full load
./quant_matmul_mxfp8_a_full_load 16 128 16384 0 1
```

`NZ`权重：

```bash
# 2-buffer SWAT
./quant_matmul_mxfp8_swat_weight_nz 16 128 16384 0 1
```

### 3. 运行算法推荐脚本

```bash
python3 quant_matmul_mxfp8_algorithm_recommend.py 16 128 16384 0 1
```

下图为推荐脚本输出的**结构示意**（数值为虚构，仅说明版式）：

```text
[Profile Breakdown]
+------------------------------------+----------+---------+----------+---------+---------+------------+--------------+
| candidate                          |kernel(us)| mac(us) |scalar(us)| mte1(us)| mte2(us)|fixpipe(us) |icache_miss(%)|
+====================================+==========+=========+==========+=========+=========+============+==============+
| quant_matmul_mxfp8_swat_weight_nz  |    10.100|   1.150 |     0.520|   0.100 |   0.280 |     0.720 |        0.082  |
| quant_matmul_mxfp8_swat_4_buffer   |    11.900|   1.200 |     0.550|   0.110 |   0.440 |     0.770 |        0.095  |
| quant_matmul_mxfp8_swat            |    12.345|   1.234 |     0.567|   0.123 |   0.456 |     0.789 |        0.100  |
| quant_matmul_mxfp8_a_full_load     |    15.678|   2.100 |     0.800|   0.200 |   0.300 |     0.500 |        0.250  |
+------------------------------------+----------+---------+----------+---------+---------+------------+--------------+
```