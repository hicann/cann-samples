# Performance

最佳实践, 从Baseline到极致性能的调优实践。

### [matmul_story](./matmul_story)
以 Atlas A2 训练/推理系列产品为例，介绍 MatMul 算子的完整性能优化实践。包括任务调度与多核并行、内存带宽优化、核内流水线排布、硬件特性适配等优化策略，从理论分析到代码实践的端到端调优指南。

### [rms_norm_quant_story](./rms_norm_quant_story)
以 Ascend 950PR/950DT 训练/推理系列产品为例，介绍 RmsNormQuant 算子的完整性能优化实践。包括多核并行与数据预加载、内存带宽优化、核内流水线排布、硬件特性适配等优化策略，从理论分析到代码实践的端到端调优指南。