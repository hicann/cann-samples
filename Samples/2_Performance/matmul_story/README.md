# 矩阵乘性能优化实践

## 目录结构

```
matmul_story/
├── common/                                     # 公共工具函数与验证脚本
├── docs/                                       # 性能优化技术文档
├── matmul_recipes/                               # 算子实现与示例代码
│   ├── include/                                # 头文件 (block, kernel, policy, utils)
│   └── examples/                               # 算子示例目录
│       ├── quant_matmul_mxfp4/                 # MXFP4 量化矩阵乘示例
│       │   ├── quant_matmul_mxfp4_swat.cpp     # SWAT模板Host侧实现
│       │   └── README.md                       # 示例说明文档
│       ├── ...                                 # 其他算子示例
│       └── matmul_a16w16/                      # A16W16 非量化矩阵乘示例
└── matmul_tutorials/                           # 分步教程
```

## 概述

本仓库提供矩阵乘算子在昇腾AI处理器上的完整性能优化实践方案。矩阵乘法是深度学习模型中最核心的计算操作之一，其性能直接影响模型的整体训练和推理效率。

- **多数据类型支持**：涵盖Float16、BFloat16、MXFP8、MXFP4等多种数据类型的实现示例，满足不同精度和性能需求
- **完整优化体系**：包含性能建模、数据传输优化、计算效率优化、指令并行度优化等完整技术栈，从理论到实践全方位指导
- **分步教程**：提供从零开始实现算子极致性能的详细指导，帮助开发者快速掌握昇腾平台高性能编程技巧

## 算子示例

- [matmul_a16w16](./matmul_recipes/examples/matmul_a16w16/README.md)：A16W16 非量化矩阵乘算子优化实践
- [quant_matmul_mxfp4](./matmul_recipes/examples/quant_matmul_mxfp4/README.md)：MXFP4 量化矩阵乘算子优化实践

## 分步教程

待补充