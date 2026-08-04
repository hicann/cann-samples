# Introduction

面向昇腾 NPU 算子开发的入门指引，帮助开发者建立基本概念，补全从入门到精通的知识空缺。

### [npu_execution](./npu_execution)
纯概念文档，无代码。拆解一个 NPU 算子从 PyTorch 调用到芯片执行所经历的完整链路。

### [vector_add_c_api](./vector_add_c_api)
最简实现，适合新人入门。演示如何使用 Ascend C API（C 语言风格接口）在 NPU 上编写简单的 Vector Core kernel，实现向量逐元素相加。

### [vector_add](./vector_add)
使用 Ascend C TQue / TPipe 实现向量逐元素加法，学习 Vector Core kernel 的基本编程模型与编译运行流程。

### [vector_function_add](./vector_function_add)
RegBase 编程模型的可运行入门样例。使用 Vector Function（VF）在寄存器上实现与 vector_add 相同的向量加法，学习 `__simd_vf__`、`AscendC::Reg::*`、Mask 尾块自适应与 `asc_vf_call` 调用方式。仅支持 Ascend 950（dav-3510）。

### [matmul](./matmul)
使用 Ascend C Tensor API 实现矩阵乘法，学习 Cube Core kernel 的基本编程模型与性能 profiling 方法。

### [vector_function_getting_started](./vector_function_getting_started)
RegBase 编程模型入门文档，无可执行代码。从 MemBase 写法的搬运开销切入，讲解 Vector Function 的编程模型（SIMD、Mask、Load/Store）与硬件执行机制（乱序、硬件循环、指令并行），并以 MulAdd 为例展示完整 VF 实现。
### [custom_op](./custom_op)
算子框架适配层（Plugin）开发示例。演示如何将第三方框架（如 ONNX、TensorFlow）的算子映射为昇腾平台支持的算子，包含属性解析、图改写、算子映射等进阶用法，是打通自定义算子从前端框架到 Ascend 硬件执行的关键环节。
