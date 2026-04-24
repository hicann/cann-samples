# A16W16非量化矩阵乘算子

## 概述


本示例展示了 A16W16 非量化矩阵乘算子在昇腾 AI 处理器上的完整实现，包含基于 SWAT 模板的高性能优化方案。A16W16 支持 Float16、BFloat16 数据类型，是深度学习中最基础且最重要的计算操作，广泛应用于各种神经网络层，包括全连接层、注意力机制等。


当前目录提供以下能力：


- `matmul_a16w16_swat`：基于 SWAT 模板的实现。
- `matmul_a16w16_streamk`：基于 STREAMK 模板的实现。
- `gen_data.py`：生成输入数据和 CPU golden 结果。
- `verify_result.py`：校验 NPU 输出与 CPU golden 是否一致。

## 使用约束

当前样例支持以下场景：

- A 的形状为 `[M, K]`(非转置)或 `[K, M]`(转置)
- B 的形状为 `[K, N]`(非转置)或 `[N, K]`(转置)。
- 输出矩阵 C 的形状为 `[M, N]`。
- 支持数据类型：Float16、BFloat16，矩阵 A,B,C 数据类型需保持一致。
- 默认数据类型为 BFloat16，如果需要更改请同步修改以上代码文件。

## 支持架构

NPU ARCH 3510

## 性能优化指南

关于算子涉及的模板实现及优化策略，请参考[非量化矩阵乘算子性能优化指南](../../../docs/matmul_performance.md)

## API参考

[Ascend C API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0003.html)

## 参数说明


可执行文件的命令行参数格式：


```text
<program> m k n
<program> m k n transA transB
```


- `m`：矩阵 A 的行数。
- `k`：矩阵 A 的列数，同时也是矩阵 B 的行数（归约维）。
- `n`：矩阵 B 的列数，对应输出矩阵的列数。
- `transA`：矩阵 A 的转置信息。
- `transB`：矩阵 B 的转置信息。

在当前布局下, 如果不传入转置信息，默认A矩阵非转置，B矩阵转置：

- A 按 `[M, K]` 组织。
- B 按 `[N, K]` 组织。
- 输出矩阵 C 的形状为 `[M, N]`。

## 数据与校验

`gen_data.py` 会在当前目录下生成以下文件：

- `input/input_a.bin`
- `input/input_b.bin`
- `output/cpu_output.bin`

样例执行完成后会额外生成：

- `output/npu_out.bin`

可执行文件在运行结束后会自动调用 `verify_result.py`，将 NPU 输出与 CPU golden 进行一致性校验。

## 手动构建与运行

如需手动控制各步骤，可在仓库根目录下完成编译和安装后，进入当前样例目录：

```bash
cmake -S . -B build
cmake --build build --parallel
cmake --install build --prefix ./build_out
cd build_out/2_Performance/matmul_story/matmul_recipes/matmul_a16w16
```

### 1. 生成测试数据

```bash
python3 scripts/gen_data.py 100 50 200 false true
```

参数说明：
- `m`：矩阵 A 的行数
- `k`：矩阵 A 的列数，同时也是矩阵 B 的行数（归约维）
- `n`：矩阵 B 的列数
- `transA`：矩阵 A 是否转置，`false` 表示 A 形状为 `[M, K]`，`true` 表示 A 形状为 `[K, M]`
- `transB`：矩阵 B 是否转置，`false` 表示 B 形状为 `[K, N]`，`true` 表示 B 形状为 `[N, K]`

示例：
- `false false`：A 为 `[M, K]`，B 为 `[K, N]`
- `true false`：A 为 `[K, M]`，B 为 `[K, N]`
- `false true`：A 为 `[M, K]`，B 为 `[N, K]`
- `true true`：A 为 `[K, M]`，B 为 `[N, K]`

### 2. 运行算子样例

```bash
./matmul_a16w16_swat 100 50 200 
./matmul_a16w16_swat 100 50 200 false true
```

或：

```bash
./matmul_a16w16_streamk 32 4096 64
./matmul_a16w16_streamk 32 4096 64 false true
```
