# Features

关键特性，解耦大模型核心算子底层能力。

### [访存优化方法](./memory_optimization)
- [full_load](./memory_optimization/full_load)：演示在 MTE2 带宽受限的场景下，当 A 矩阵与 B 矩阵规模较小时，通过采用全载（full load）操作减少 MTE2 的搬运次数，从而提升整体性能。
- [l1_bank_conflict](./memory_optimization/l1_bank_conflict)：演示在 MTE1 带宽受限的场景下，通过解决 L1 的 bank 冲突问题，减少 MTE1 的搬运时间，从而提升整体性能。
- [slide_window_adaptive_template](./memory_optimization/slide_window_adaptive_template)：演示在带宽受限的场景下，通过提高数据搬运效率来提升整体性能。
- [scale_cache](./memory_optimization/scale_cache)：演示当单次搬运数据量不足、无法充分发挥带宽性能时，如何利用L1缓存的剩余空间，提前加载并缓存后续所需的Scale数据。解决因数据量过小导致的带宽利用率下降问题。

### [指令优化方法](./instruction_optimization)
- [n_buffer](./instruction_optimization/n_buffer)：演示如何使用 nBuffer（多区块缓存）编程模型在 NPU 上实现搬运计算流水并行。
- [unit_flag](./instruction_optimization/unit_flag)：演示使用 `unit_flag` 开启计算（MMAD）与搬出（Fixpipe）流水并行，进一步提升流水并行度。
- [weightnz](./instruction_optimization/weightnz)：演示了在带宽成为瓶颈的情况下，如何通过将权重矩阵预先转换为 FRACTAL_NZ 格式，以减少随路带宽的损耗，从而提升数据搬运效率。
- [mte2_preload](./instruction_optimization/mte2_preload)：演示了在指令数量超出芯片预设指令队列深度的情况下，如何通过将矩阵预加载到L1缓存中，使得Ping-Pong指令实现强制同步，从而优化数据搬运效率。

### [系统优化方法](./system_optimization)
- [tail_rebalance](./system_optimization/tail_rebalance)：演示如何通过尾轮负载均衡策略提升尾轮计算效率，进而提高整体性能。
- [streamk](./system_optimization/streamk)：演示如何通过将计算任务均衡分配到多个核心以提升整体性能。其核心方法是将任务划分为k份，在各个核心上并行执行，从而有效提升尾轮计算效率，最终实现整体性能的提高。

### [芯片特性](./hardware_features)
- [simt](./hardware_features/simt)：演示如何使用 SIMT（单指令多线程）编程模型在 NPU 上实现 Gather 算子。
- [vector_function](./hardware_features/vector_function)：演示 Vector Function 编程概念，通过 GeLU 对比展示 VF 能力。
- [hif8](./hardware_features/hif8)：演示 HiFloat8（HIF8）量化数据类型及相关样例实现。
