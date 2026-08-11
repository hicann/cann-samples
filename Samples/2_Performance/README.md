# Performance

最佳实践, 从Baseline到极致性能的调优实践。

### [grouped_matmul_story](./grouped_matmul_story)
分组矩阵乘性能优化专题，覆盖 grouped matmul 的 tiling、数据搬运与 kernel 实现，并提供 MXFP4/MXFP8 可运行示例及数据校验流程。

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

### [kv_rms_norm_rope_cache_story](./kv_rms_norm_rope_cache_story)
围绕 Ascend 950 上的 KvRmsNormRopeCache full-load 路径给出 MemBase 与 RegBase 两个 BF16 直调样例，展示 RMSNorm、interleave RoPE 与 Norm cache 更新的融合实现，以及从 MemBase 到 RegBase 的寄存器化优化点。

### [simt_scatter_story](./simt_scatter_story)
以 Scatter 算子为例的 SIMT 递进教学样例（`dav-3510`）：演示 SIMT 直接访问 GM 完成不规则写，以及通过目标地址分组和单写者选择处理重复 index 带来的写冲突。

### [softmax_regbase_story](./softmax_regbase_story)
Softmax 算子的 RegBase 递进优化专题，演示 reduce + element-wise 混合算子从 MemBase 基线到 RegBase VF 融合、多行并行的优化实践。

### [simt_histogram_story](./simt_histogram_story)
以 Histogram 算子为例的 SIMT 递进教学样例（`dav-3510`）：从 MTE+Vector 串行基线出发，逐步引入 SIMT 并行计数（grid-stride）、float4 向量化、launch_bounds 寄存器分析与 GridDim 扫描，展示 A5 纯 SIMT 编程模型下的性能优化路径。

### [gelu_eltwise_regbase_story](./gelu_eltwise_regbase_story)
用 GELU + Element-wise 融合算子演示 RegBase 的改写和优化（`dav-3510`）：Case 0 是 MemBase 基线，Case 1~4 依次做 VF 融合、循环拆分、循环展开、常量外提，共 5 个独立可执行 Case。

### [flash_attn_lite_story](./flash_attn_lite_story)
Flash Attention Lite 教学样例（Ascend 950），以固定规格的 Flash Attention 前向计算说明 Cube 与 Vector 融合算子的实现和流水排布，从单槽串行基线逐步加入 CV 核间双槽流水、AIC 核内双缓冲和 task 级 I/O 双缓冲，覆盖 v0～v5 递进版本。

### [scalar_story](./scalar_story)
Scalar 单元性能优化专题，分析 Ascend 950 上 ScalarBound 问题的根因与诊断方法，涵盖 icache 预取、静态创建 LocalTensor、局部变量替代成员变量、消除多级指针解引用等优化手段，以 FusedInferAttentionScore 和 QuantBatchMatmul 为案例。

### [simd_vf_story](./simd_vf_story)
SIMD VF 编程范式实践，覆盖 Broadcast（尾轴/首轴/中间轴）、Elemwise 逐元素与 Reduce 归约算子的 SIMD VF 实现与优化分析，展示不同写法间的性能差异与优化原理。
