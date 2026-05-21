# 矩阵乘性能优化实践

## 目录结构

```
matmul_story/
├── CMakeLists.txt
├── README.md
├── matmul_recipes/                             # 算子实现与示例代码
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── common/                                 # 公共工具函数（host/kernel）
│   ├── include/                                # 头文件 (block, kernel, policy, tile, tiling, utils)
│   ├── docs/                                   # 性能优化技术文档
│   └── examples/                               # 算子示例目录
│       ├── quant_matmul_mxfp4/                 # MXFP4 量化矩阵乘示例
│       ├── quant_matmul_mxfp8/                 # MXFP8 量化矩阵乘示例
│       ├── quant_matmul_hifp8/                 # HiFloat8 量化矩阵乘示例
│       └── matmul_a16w16/                      # A16W16 非量化矩阵乘示例
└── matmul_tutorials/                           # 分步教程（细目见 matmul_tutorials/README.md）
    ├── CMakeLists.txt
    ├── README.md
    ├── common/                                 # 教程共享工具函数与 tile 头文件
    ├── images/                                 # 教程流水截图
    ├── scripts/                                # 数据生成与验证脚本
    ├── 0_naive/                                # Step 0 基准
    ├── 1_pingpong/                             # Step 1 打开 Ping-Pong
    ├── 2_block_swat/                           # Step 2 SWAT
    ├── 3_last_round_tile_balance/              # Step 3 尾轮负载均衡
    ├── 4_unit_flag/                            # Step 4 UnitFlag
    ├── 5_halfl1_ping_halfl1_pong/              # Step 5 Half-L1 Ping/Half-L1 Pong
    ├── 6_scale_memory_access_coalescing/       # Step 6 Scale 访存合并优化
    └── 7_fullload/                             # Step 7 A 全载
```

## 概述

本仓库提供矩阵乘算子在昇腾AI处理器上的完整性能优化实践方案。矩阵乘法是深度学习模型中最核心的计算操作之一，其性能直接影响模型的整体训练和推理效率。

- **多数据类型支持**：涵盖Float16、BFloat16、MXFP8、MXFP4等多种数据类型的实现示例，满足不同精度和性能需求
- **完整优化体系**：包含性能建模、数据传输优化、计算效率优化、指令并行度优化等完整技术栈，从理论到实践全方位指导
- **分步教程**：提供从零开始实现算子极致性能的详细指导，帮助开发者快速掌握昇腾平台高性能编程技巧

## 算子示例

- [matmul_a16w16](./matmul_recipes/examples/matmul_a16w16/README.md)：A16W16 非量化矩阵乘算子优化实践
- [quant_matmul_mxfp4](./matmul_recipes/examples/quant_matmul_mxfp4/README.md)：MXFP4 量化矩阵乘算子优化实践
- [quant_matmul_mxfp8](./matmul_recipes/examples/quant_matmul_mxfp8/README.md)：MXFP8 量化矩阵乘算子优化实践
- [quant_matmul_hifp8](./matmul_recipes/examples/quant_matmul_hifp8/README.md)：HiFloat8（hifp8）矩阵乘，`quant_matmul_hifp8_tt` / `quant_matmul_hifp8_tc` 双入口说明见目录 README

## 优化指南

- [quant_matmul_mx_performance](./docs/quant_matmul_mx_performance.md)：MX量化场景矩阵乘算子性能优化指南


## 分步教程

各 Step 说明、Case 与流水图见 [matmul_tutorials/README.md](./matmul_tutorials/README.md)；目录树见上文 `matmul_tutorials/`。
