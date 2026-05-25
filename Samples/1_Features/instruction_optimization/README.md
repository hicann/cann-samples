# Instruction Optimization
指令优化方法相关样例。

### [n_buffer](./n_buffer)
演示如何使用 nBuffer（多区块缓存）编程模型在 NPU 上实现搬运计算流水并行。

### [unit_flag](./unit_flag)
演示使用 `unit_flag` 来开启计算（MMAD）与搬出（Fixpipe）流水并行，进一步提升流水并行度。

### [mte2_preload](./mte2_preload)
演示指令数量超出队列深度时，通过矩阵预加载到L1缓存，利用Ping-Pong指令强制同步，优化搬运效率。

### [weightnz](./weightnz)
演示带宽瓶颈时，将权重矩阵预转为 FRACTAL_NZ 格式，减少带宽损耗，提升搬运效率。