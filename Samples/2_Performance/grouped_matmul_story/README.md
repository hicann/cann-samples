# 分组矩阵乘性能优化实践

## 目录结构

```
grouped_matmul_story/
├── CMakeLists.txt
├── README.md
├── grouped_matmul_recipes/                     # 算子实现与示例代码
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── common/                                 # 公共工具函数（host/kernel）
│   ├── include/                                # 头文件 (block, kernel, tile, tiling, utils)
│   └── examples/                               # 算子示例目录
│       ├── quant_grouped_matmul_mxfp4/         # MXFP4 分组量化矩阵乘示例
│       ├── quant_grouped_matmul_mxfp8/         # MXFP8 分组量化矩阵乘示例
│       └── weight_quant_grouped_matmul_mxfp8fp4/  # MXA8W4 权重量化分组矩阵乘示例
└── docs/                                       # 性能优化技术文档
```

## 概述

本仓库提供分组矩阵乘算子在昇腾AI处理器上的完整性能优化实践方案。分组矩阵乘面向MoE等多专家计算场景，可按专家分组执行多组矩阵乘计算，提升推理场景下的计算效率。

- **多数据类型支持**：涵盖MXFP8、MXFP4、MXA8W4等多种数据类型的实现示例，满足不同精度和性能需求
- **完整优化体系**：包含分组调度、Tiling切分、权重排布、数据传输优化等完整技术栈，从理论到实践全方位指导
- **分组矩阵乘样例**：提供MX量化与权重量化场景下的分组矩阵乘实现，帮助开发者快速掌握昇腾平台高性能编程技巧

## 算子示例

- [quant_grouped_matmul_mxfp4](./grouped_matmul_recipes/examples/quant_grouped_matmul_mxfp4/README.md)：MXFP4 分组量化矩阵乘算子优化实践
- [quant_grouped_matmul_mxfp8](./grouped_matmul_recipes/examples/quant_grouped_matmul_mxfp8/README.md)：MXFP8 分组量化矩阵乘算子优化实践
- [weight_quant_grouped_matmul_mxfp8fp4](./grouped_matmul_recipes/examples/weight_quant_grouped_matmul_mxfp8fp4/README.md)：MXA8W4 权重量化分组矩阵乘算子优化实践

## 优化指南

- [quant_grouped_matmul_mx_performance](./docs/quant_grouped_matmul_mx_performance.md)：MX量化场景分组矩阵乘算子性能优化指南

## grouped_matmul_recipes

样例列表、目录树与使用方式见 [grouped_matmul_recipes/README.md](./grouped_matmul_recipes/README.md)。
