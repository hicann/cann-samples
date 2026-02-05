# Features

关键特性，解耦大模型核心算子底层能力。

### [simt](./simt)
演示如何使用 SIMT（单指令多线程）编程模型在 NPU 上实现 Gather 算子。使用 `__simt_vf__` 和 `asc_vf_call` API 进行开发。

### [vector_function](./vector_function)
演示 Vector Function 编程概念，通过 GeLU 激活函数展示传统实现与 VF 优化实现的性能对比，揭示计算融合的优势。

### [hif8](./hif8)
演示 HiFloat8（HIF8）量化数据类型的应用，展示 Quantize 算子的实现，支持 8 位浮点格式以优化存储和计算效率。
