# Hardware Features

芯片特性相关样例。

### [simt](./simt)
演示如何使用 SIMT（单指令多线程）编程模型在 NPU 上实现 Gather 算子。使用 `__simt_vf__` 和 `asc_vf_call` API 进行开发。

### [vector_function](./vector_function)
演示 Vector Function 编程概念，通过 GeLU 激活函数展示传统实现与 VF 优化实现的性能对比，揭示计算融合的优势。

### [simd_vf_constraints](./simd_vf_constraints)
汇总 Ascend950 上 SIMD VF 的常见编程约束，涵盖函数定义与入参、`asc_vf_call` 调用、Hardware Loop、LocalMemBar 流水保序，以及寄存器、标量、地址与数据访问等硬件资源限制，每条约束均提供对应的正例与反例。

### [hif8](./hif8)
演示 HiFloat8（HIF8）量化数据类型的应用，展示 Quantize 算子的实现，支持 8 位浮点格式以优化存储和计算效率。

### [cv_datapath](./cv_datapath)
演示 Ascend 950 CV（Cube-Vector）数据通路特性：以 MatMul+ReLU 对比 Cube/Vector 分离、Mix Scenario1（L0C→GM→UB）与 Scenario2（L0C→UB Fixpipe 直通），帮助理解 AIC/AIV 协同与数据通路选型。

### [mem_bandwidth](./mem_bandwidth)
测量 NPU 的访存带宽，覆盖纯读、读写拷贝、读+计算+写三种数据流，统一采用 `TPipe + TQue` 多 buffer 流水范式，通过成对扫描 UB tile 大小与 buffer 数，观察带宽随搬运粒度与缓冲深度的变化。带宽由 `msprof` 采集 `Task Duration` 换算。
