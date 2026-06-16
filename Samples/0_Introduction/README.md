# Introduction

面向昇腾 NPU 算子开发的入门指引，帮助开发者建立基本概念，补全从入门到精通的知识空缺。

### [npu_execution](./npu_execution)
纯概念文档，无代码。拆解一个 NPU 算子从 PyTorch 调用到芯片执行所经历的完整链路。

### [vector_add](./vector_add)
第一个动手示例。通过实现向量逐元素加法，学习 AscendC Vector Core kernel 的基本编程模型与编译运行流程。

### [vector_add_c_api](./vector_add_c_api)
演示如何使用 Ascend C API（C 语言风格接口）在 NPU 上编写简单的 Vector Core kernel，实现向量逐元素相加。入门级向量计算示例。支持架构：dav-2201。

### [matmul](./matmul)
Cube Core 入门示例。通过实现矩阵乘法，学习 AscendC Cube Core kernel 的基本编程模型与性能 profiling 方法。

### [vector_function_getting_started](./vector_function_getting_started)
RegBase 编程模型入门文档，无可执行代码。从 MemBase 写法的搬运开销切入，讲解 Vector Function 的编程模型（SIMD、Mask、Load/Store）与硬件执行机制（乱序、硬件循环、指令并行），并以 MulAdd 为例展示完整 VF 实现。
