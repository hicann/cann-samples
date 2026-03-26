# Performance

最佳实践, 从Baseline到极致性能的调优实践。

### [matmul_story](./matmul_story)
矩阵乘性能优化专题，覆盖 MatMul 与量化 MatMul（如 MXFP4）两类实践，包含性能分析文档、分步教程（baseline→SWAT→尾轮负载均衡→UnitFlag）以及可运行的 recipe 示例（A16W16、quant_matmul_mxfp4）。

### [rms_norm_quant_story](./rms_norm_quant_story)
以 Ascend 950PR/950DT 训练/推理系列产品为例，介绍 RmsNormQuant 算子的完整性能优化实践。包括多核并行与数据预加载、内存带宽优化、核内流水线排布、硬件特性适配等优化策略，从理论分析到代码实践的端到端调优指南。

### [full_quant_fused_infer_attention_score_story](./full_quant_fused_infer_attention_score_story)
围绕 FIA（Fused Infer Attention Score）算子提供 per-block 全量化实现示例，包含输入数据生成、算子执行与结果校验流程。

### [moe_init_routing_story](./moe_init_routing_story)
以 Ascend950PR/DT 训练/推理系列产品为例，介绍 MoeInitRoutingV3 算子的完整性能优化实践。包括多核并行、内存带宽优化、核内流水线排布、SIMT编程、硬件特性适配等优化策略，从理论分析到代码实践的端到端调优指南。

### [moe_dispatch_and_combine_story](./moe_dispatch_and_combine_story)
围绕 moe dispatch/combine 通信算子给出性能优化实践，包含构建运行命令、测试数据生成与精度校验流程。
