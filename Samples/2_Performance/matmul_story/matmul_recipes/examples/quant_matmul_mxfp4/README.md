# MXFP4量化矩阵乘算子

## 概述

本示例展示了MXFP4量化矩阵乘算子在昇腾AI处理器上的完整实现，包含基于SWAT模板的高性能优化方案。MXFP4是一种4位浮点数量化格式，通过GroupSize=32的分组量化策略，在保持模型精度的同时显著减少内存访问量和计算密度，特别适用于大语言模型推理等场景。

## 性能优化指南

关于算子涉及的模板实现及优化策略，请参考[MXFP4量化矩阵乘算子性能优化指南](../../../docs/quant_matmul_mxfp4_performance.md)

## 支持架构

NPU ARCH 3510

## API参考

[Ascend C API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0003.html)

## 参数说明

待补充

## 编译与运行

待补充