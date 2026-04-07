# 矩阵乘算子样例

## 概述

本目录汇总了矩阵乘在昇腾 AI 处理器上的典型实现样例。每个样例提供完整的算子代码、运行脚本与说明文档，便于直接编译运行并做性能对比。

## 目录结构

```text
matmul_recipes/
├── CMakeLists.txt
├── README.md
├── common/                           # 公共工具（host/kernel）
├── include/                          # 共享头文件（block、kernel、policy、tile 等）
└── examples/
    ├── quant_matmul_mxfp4/           # MXFP4 量化矩阵乘样例
    │   ├── README.md
    │   ├── quant_matmul_mxfp4_swat.cpp
    │   ├── quant_matmul_mxfp4_a_full_load.cpp
    │   └── scripts/
    │       ├── gen_data.py
    │       ├── verify_result.py
    │       └── quant_matmul_mxfp4_algorithm_recommend.py
    └── matmul_a16w16/                  # A16W16 非量化矩阵乘样例
        └── README.md
```

## 样例列表

| 样例 | 数据类型 | 说明 |
|------|----------|------|
| [matmul_a16w16](examples/matmul_a16w16/README.md) | Float16 | A16W16 非量化矩阵乘 |
| [quant_matmul_mxfp4](examples/quant_matmul_mxfp4/README.md) | MXFP4 | 4 位浮点量化矩阵乘，包含 SWAT 与 A 全载两种实现 |
| [quant_matmul_mxfp8](examples/quant_matmul_mxfp8/README.md) | MXFP8 | 8 位浮点量化矩阵乘，包含 SWAT 与 A 全载两种实现 |

## 使用方式

- 查看对应样例目录下的 `README.md`，按说明完成构建、运行与结果校验。
- 推荐先从 `quant_matmul_mxfp4` 开始，便于快速验证脚本与可执行文件的配套流程。

## 性能优化指南

各样例涉及的模板实现及优化策略详见 [MX 量化矩阵乘算子性能优化指南](../docs/quant_matmul_mx_performance.md)。
