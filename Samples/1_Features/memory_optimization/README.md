# Memory Optimization

访存优化方法相关样例。

### [full_load](./full_load)
针对 MTE2 带宽受限、A/B 矩阵规模较小的场景，演示如何采用全载（full load）操作来减少 MTE2 搬运次数，实现整体性能的提升。

### [l1_bank_conflict](./l1_bank_conflict)
针对 MTE1 带宽受限的场景，演示如何通过解决 L1 bank 冲突问题来缩短 MTE1 搬运时间，从而实现整体性能的提升。

### [slide_window_adaptive_template](./slide_window_adaptive_template)
演示如何利用 SWAT（Slide Window Adaptive Template）滑窗技术优化访存效率。

### [scale_cache](./scale_cache)
演示当单次搬运数据量不足时，如何利用L1缓存预取Scale数据，解决带宽利用率下降问题。

### [transpose_batch_matmul](./transpose_batch_matmul)
演示如何通过 Layout 抽象将转置语义内嵌到 GM→L1 搬运指令中，省去显式 Transpose 算子的全量 GM 读写搬运开销，并利用 batch 维参与多核调度提升硬件利用率。
