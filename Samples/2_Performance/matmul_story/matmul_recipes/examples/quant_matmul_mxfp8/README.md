# MXFP8量化矩阵乘算子

## 概述

本示例展示了MXFP8量化矩阵乘算子在昇腾AI处理器上的完整实现，包含基于SWAT模板的高性能优化方案。MXFP8是一种8位浮点数量化格式，可在保持较好精度的前提下显著降低带宽开销与访存压力，适用于大语言模型推理等场景。

当前目录提供以下能力：

- `quant_matmul_mxfp8_swat`：基于SWAT模板的实现。
- `quant_matmul_mxfp8_a_full_load`：A full load 方案的实现。
- `gen_data.py`：生成输入数据和CPU golden结果。
- `verify_result.py`：校验NPU输出与CPU golden是否一致。
- `quant_matmul_mxfp8_algorithm_recommend.py`：对当前目录下可执行算法进行兼容性筛选和耗时排序。

## 使用约束

当前样例仅支持以下场景：

- A不转置、B转置。
- A的形状为`[M, K]`，B的形状为`[N, K]`。
- A和B的K轴都位于内轴。

## 支持架构

NPU ARCH 3510

## 性能优化指南

关于算子涉及的模板实现及优化策略，请参考[MX量化矩阵乘算子性能优化指南](../../../docs/quant_matmul_mx_performance.md)

## API参考

[Ascend C API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0003.html)

## 参数说明

两个可执行文件的命令行参数格式一致：

```text
<program> m k n
```

- `m`：矩阵A的行数。
- `k`：矩阵A的列数，同时也是矩阵B的归约维。
- `n`：矩阵B的行数，对应输出矩阵的列数。

在当前布局下：

- A按`[M, K]`组织。
- B按`[N, K]`组织。
- 输出矩阵C的形状为`[M, N]`。

## 数据与校验

`gen_data.py`会在当前目录下生成以下文件：

- `input/input_a.bin`
- `input/input_b.bin`
- `input/input_scaleA.bin`
- `input/input_scaleB.bin`
- `output/cpu_output.bin`

样例执行完成后会额外生成：

- `output/npu_out.bin`

两个可执行文件在运行结束后都会自动调用`verify_result.py`，将NPU输出与CPU golden进行一致性校验。

## 构建与运行

在仓库根目录下完成编译和安装后，进入当前样例目录：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --parallel
cmake --install build --prefix ./build_out
cd build_out/2_Performance/matmul_story/matmul_recipes/quant_matmul_mxfp8
```

### 1. 生成测试数据

```bash
python3 gen_data.py 16 2048 16384
```

### 2. 运行单个算法样例

```bash
./quant_matmul_mxfp8_swat 16 2048 16384
```

或：

```bash
./quant_matmul_mxfp8_a_full_load 16 2048 16384
```

### 3. 运行算法推荐脚本

```bash
python3 quant_matmul_mxfp8_algorithm_recommend.py 16 2048 16384
```