# MXFP4量化矩阵乘算子

## 概述

本示例展示了MXFP4量化矩阵乘算子在昇腾AI处理器上的完整实现，包含基于SWAT模板的高性能优化方案。MXFP4是一种4位浮点数量化格式，通过GroupSize=32的分组量化策略，在保持模型精度的同时显著减少内存访问量和计算密度，特别适用于大语言模型推理等场景。

当前样例仅要求输入的`k`为偶数，以满足两个`B4`打包成一个`B8`的存储约束。

## 性能优化指南

关于算子涉及的模板实现及优化策略，请参考[MXFP4量化矩阵乘算子性能优化指南](../../../docs/quant_matmul_mxfp4_performance.md)

## 支持架构

NPU ARCH 3510

## API参考

[Ascend C API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0003.html)

## 输入参数

两个可执行文件的命令行参数格式一致：

```text
<program> m k n
```

- `m`：矩阵 A 的行数
- `k`：矩阵 A 的列数，同时也是矩阵 B 的行数
- `n`：矩阵 B 的列数

golden 输入数据由 `matmul_recipes/examples/quant_matmul_mxfp4/gen_data.py` 生成。

## 构建与运行

在仓库根目录下执行全量编译与安装，并进入安装目录：

```bash
cmake -S . -B build
cmake --build build --parallel
cmake --install build --prefix ./build_out
cd build_out/2_Performance/matmul_story/matmul_recipes/examples/quant_matmul_mxfp4
```

之后可按需执行以下命令：

```bash
# 先生成一组测试数据
python3 gen_data.py 256 256 256

# 运行其中一个可执行文件，如
./quant_matmul_mxfp4_swat 256 256 256

# 运行算法推荐脚本
python3 quant_matmul_mxfp4_algorithm_recommend.py 256 256 256
```
