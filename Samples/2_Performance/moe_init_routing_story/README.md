# MoeInitRoutingV3算子性能优化实践与效果分析

## 构建与运行

在项目根目录启动构建，执行：

```bash
cmake -S . -B build
cmake --build build --target moe_init_routing_story
```
构建完成后会在./build/Samples/2_Performance/moe_init_routing_story/目录下生成MoeInitRoutingV3算子各个版本的可执行文件，如1_multi_core、2_double_buffer。

先在可执行文件的同目录下生成一组测试数据，其中n和c为输入Token的维度特征(n, c)，k表示为每个Token动态选择Top-K个专家：

```bash
python3 ./Samples/2_Performance/moe_init_routing_story/scripts/gen_data.py -n 2048 -k 8 -c 32
```

运行相应的可执行文件，三个参数分别对应上文的-n、-k和-c：

```bash
./build/Samples/2_Performance/moe_init_routing_story/1_multi_core 2048 8 32
```

最后执行精度校验：

```bash
python3 ./Samples/2_Performance/moe_init_routing_story/scripts/verify_result.py
```

执行结束后会在控制台输出精度比对结果，如：

```bash
ExpandedX Precision is 100.0000%
ExpandedRowIdx Precision is 100.0000%
TokenCount Precision is 100.0000%
```