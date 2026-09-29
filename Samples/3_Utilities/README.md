# Utilities

> 使用 npusim 进行仿真需要 CANN 9.2.0 及以上版本。

面向 CANN 算子开发、调试与分析流程的工具类样例，帮助开发者借助仿真、Profiling 与辅助工具定位问题、理解性能瓶颈并指导优化。

### [simulation-based-vf-profiling](./simulation-based-vf-profiling)

基于 `npusim` 的 VF 性能分析样例。通过最小化 VF kernel、关闭 VF 自动融合并结合 Trace，演示如何观察 VF 执行周期、load / exec / store pipe 分布，定位瓶颈并推断优化方向。支持架构：`dav-3510`。
