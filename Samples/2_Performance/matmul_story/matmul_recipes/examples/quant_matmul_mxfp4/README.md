# MXFP4量化矩阵乘算子

## 概述

本示例展示了 MXFP4 量化矩阵乘算子在昇腾 AI 处理器上的完整实现，包含基于 SWAT 模板的高性能优化方案。MXFP4 是一种 4 位浮点数量化格式，通过 GroupSize=32 的分组量化策略，在兼顾模型精度的同时显著降低访存开销和计算成本，适用于大语言模型推理等场景。

当前目录提供以下能力：

- `quant_matmul_mxfp4_swat`：基于 SWAT 模板的实现。
- `quant_matmul_mxfp4_a_full_load`：另一种量化矩阵乘实现。
- `gen_data.py`：生成输入数据和 CPU golden 结果。
- `verify_result.py`：校验 NPU 输出与 CPU golden 是否一致。
- `quant_matmul_mxfp4_algorithm_recommend.py`：对当前目录下可执行算法进行兼容性筛选和耗时排序。

## 使用约束

当前样例仅支持以下场景：

- A 不转置、B 转置。
- A 的形状为 `[M, K]`，B 的形状为 `[N, K]`。
- A 和 B 的 K 轴都位于内轴。
- 输入参数 `k` 必须为偶数，以满足两个 `B4` 打包成一个 `B8` 的存储约束。

## 支持架构

NPU ARCH 3510

## 性能优化指南

关于算子涉及的模板实现及优化策略，请参考[MXFP4量化矩阵乘算子性能优化指南](../../../docs/quant_matmul_mxfp4_performance.md)。

## API参考

[Ascend C API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0003.html)

## 参数说明

两个可执行文件的命令行参数格式一致：

```text
<program> m k n
```

- `m`：矩阵 A 的行数。
- `k`：矩阵 A 的列数，同时也是矩阵 B 的归约维。
- `n`：矩阵 B 的行数，对应输出矩阵的列数。

在当前布局下：

- A 按 `[M, K]` 组织。
- B 按 `[N, K]` 组织。
- 输出矩阵 C 的形状为 `[M, N]`。

## 数据与校验

`gen_data.py` 会在当前目录下生成以下文件：

- `input/input_a.bin`
- `input/input_b.bin`
- `input/input_scaleA.bin`
- `input/input_scaleB.bin`
- `output/cpu_output.bin`

样例执行完成后会额外生成：

- `output/npu_out.bin`

两个可执行文件在运行结束后都会自动调用 `verify_result.py`，将 NPU 输出与 CPU golden 进行一致性校验。

## 构建与运行

在仓库根目录下完成编译和安装后，进入当前样例目录：

```bash
cmake -S . -B build
cmake --build build --parallel
cmake --install build --prefix ./build_out
cd build_out/2_Performance/matmul_story/matmul_recipes/quant_matmul_mxfp4
```

### 1. 生成测试数据

```bash
python3 gen_data.py 16 2048 16384
```

### 2. 运行单个算法样例

```bash
./quant_matmul_mxfp4_swat 16 2048 16384
```

或：

```bash
./quant_matmul_mxfp4_a_full_load 16 2048 16384
```

### 3. 运行算法推荐脚本

```bash
python3 quant_matmul_mxfp4_algorithm_recommend.py 16 2048 16384
```
