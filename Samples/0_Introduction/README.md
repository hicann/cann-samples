# Introduction

面向昇腾 NPU 算子开发的入门指引，帮助开发者建立基本概念，补全从入门到精通的知识空缺。

本目录提供三个层次的学习路径，建议按顺序阅读：

| 示例 | 学习目标 |
|------|---------|
| [npu_execution](./npu_execution) | 理解算子从 Runtime 下发到芯片执行的完整链路，建立对任务调度、硬件初始化、计算执行的全局认知 |
| [vector_add](./vector_add) | 掌握 AscendC Vector Core kernel 的基本编程模型：数据搬运、向量计算、流水并行（DoubleBuffer），以及编译运行流程 |
| [matmul](./matmul) | 掌握 AscendC Cube Core kernel 的基本编程模型：矩阵分块（Tiling）、Cube 指令调用，以及性能 profiling 方法 |

### [npu_execution](./npu_execution)

纯概念文档，无代码。拆解一个 NPU 算子从 PyTorch 调用到芯片执行所经历的完整链路。

### [vector_add](./vector_add)

第一个动手示例。通过实现向量逐元素加法，学习 AscendC Vector Core kernel 的基本编程模型与编译运行流程。

### [vector_add_c_api](./vector_add_c_api)
演示如何使用 Ascend C API（C语言风格接口）在 NPU 上编写简单的 Vector Core kernel，实现向量逐元素相加。入门级向量计算示例。支持架构：dav-2201。

### [matmul](./matmul)

Cube Core 入门示例。通过实现矩阵乘法，学习 AscendC Cube Core kernel 的基本编程模型与性能 profiling 方法。
