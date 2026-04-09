# Features

关键特性，解耦大模型核心算子底层能力。

### [访存优化方法](./memory_optimization)
- [full_load](./memory_optimization/full_load)：演示在 MTE2 带宽受限的场景下，当 A 矩阵与 B 矩阵规模较小时，通过采用全载（full load）操作减少 MTE2 的搬运次数，从而提升整体性能。
- [l1_bank_conflict](./memory_optimization/l1_bank_conflict)：演示在 MTE1 带宽受限的场景下，通过解决 L1 的 bank 冲突问题，减少 MTE1 的搬运时间，从而提升整体性能。
- [slide_window_adaptive_template](./memory_optimization/slide_window_adaptive_template)：演示在带宽受限的场景下，通过提高数据搬运效率来提升整体性能。

### [指令优化方法](./instruction_optimization)
- [n_buffer](./instruction_optimization/n_buffer)：演示如何使用 nBuffer（多区块缓存）编程模型在 NPU 上实现搬运计算流水并行。
- [unit_flag](./instruction_optimization/unit_flag)：演示使用 `unit_flag` 开启计算（MMAD）与搬出（Fixpipe）流水并行，进一步提升流水并行度。

### [系统优化方法](./system_optimization)
- [tail_rebalance](./system_optimization/tail_rebalance)：演示如何通过尾轮负载均衡策略提升尾轮计算效率，进而提高整体性能。

### [芯片特性](./hardware_features)
- [simt](./hardware_features/simt)：演示如何使用 SIMT（单指令多线程）编程模型在 NPU 上实现 Gather 算子。
- [vector_function](./hardware_features/vector_function)：演示 Vector Function 编程概念，通过 GeLU 对比展示 VF 能力。
- [hif8](./hardware_features/hif8)：演示 HiFloat8（HIF8）量化数据类型及相关样例实现。
