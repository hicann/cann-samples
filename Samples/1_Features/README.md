# Features

关键特性，解耦大模型核心算子底层能力。

### [访存优化方法](./memory_optimization)
当前预留，后续补充访存优化方法相关样例。

### [指令优化方法](./instruction_optimization)
- [n_buffer](./instruction_optimization/n_buffer)：演示如何使用 nBuffer（多区块缓存）编程模型在 NPU 上实现搬运计算流水并行。
- [unit_flag](./instruction_optimization/unit_flag)：演示使用 `unit_flag` 开启计算（MMAD）与搬出（Fixpipe）流水并行，进一步提升流水并行度。

### [系统优化方法](./system_optimization)
当前预留，后续补充系统优化方法相关样例。

### [芯片特性](./hardware_features)
- [simt](./hardware_features/simt)：演示如何使用 SIMT（单指令多线程）编程模型在 NPU 上实现 Gather 算子。
- [vector_function](./hardware_features/vector_function)：演示 Vector Function 编程概念，通过 GeLU 对比展示 VF 能力。
- [hif8](./hardware_features/hif8)：演示 HiFloat8（HIF8）量化数据类型及相关样例实现。
