# MXFP8FP4权重量化矩阵乘算子

## 概述

本示例展示了MXFP8输入、packed MXFP4 NZ权重、E8M0 scale、BF16输出的权重量化矩阵乘算子在昇腾AI处理器上的完整实现。

当前目录提供以下能力：

**样例（可执行目标）**

权重`NZ`排布（权重矩阵B为packed MXFP4 `NZLayout`，矩阵A为`ND`排布）

- `weight_quant_matmul_mxfp8fp4`：MXFP8输入、packed MXFP4 NZ权重、BF16输出的2-buffer实现。
- `weight_quant_matmul_mxfp8fp4_swat_4_buffer`：MXFP8输入、packed MXFP4 NZ权重、BF16输出的4-buffer实现。

**辅助脚本（`scripts/` 目录）**

- `gen_data.py`：生成输入数据和CPU golden结果。
- `verify_result.py`：校验NPU输出与CPU golden是否一致。

## 使用约束

当前样例支持以下场景：

- 命令行参数固定为`m k n`，不支持`transA`/`transB`参数。
- A矩阵固定为`ND`排布，权重矩阵B固定为packed MXFP4 `NZ`排布。
- `m`、`k`、`n`均需为正整数。
- `k`必须为64的倍数。
- `n`必须为16的倍数。
- 2-buffer/4-buffer由可执行target决定，不随输入shape自动切换。

## 支持架构

NPU ARCH 3510

## 性能优化指南

关于算子涉及的模板实现及优化策略，请参考[MX量化矩阵乘算子性能优化指南](../../../docs/quant_matmul_mx_performance.md)

## API参考

[Ascend C API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0003.html)

## 参数说明

可执行文件的命令行参数格式一致：

```text
<program> m k n
```

- `m`：矩阵A的行数。
- `k`：矩阵A的列数，同时也是矩阵B的归约维。
- `n`：矩阵B的列数，对应输出矩阵的列数。

输出矩阵C的逻辑形状为`[M, N]`。

## 数据与校验

`gen_data.py`会在**运行时的当前工作目录**（通常为安装目录`build_out/.../weight_quant_matmul_mxfp8fp4`）下生成以下文件：

- `input/input_a.bin`
- `input/input_b.bin`
- `input/input_scaleA.bin`
- `input/input_scaleB.bin`
- `output/cpu_output.bin`

样例执行完成后会额外生成：

- `output/npu_out.bin`

可执行文件在运行结束后会自动调用`verify_result.py`，将NPU输出与CPU golden进行一致性校验。

## 手动构建与运行

在仓库根目录下完成编译和安装后，进入当前样例目录：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --parallel
cmake --install build --prefix ./build_out
cd build_out/2_Performance/matmul_story/matmul_recipes/weight_quant_matmul_mxfp8fp4
```

### 1. 生成测试数据

```bash
python3 gen_data.py 256 1024 256
```

### 2. 运行单个算法样例

```bash
./weight_quant_matmul_mxfp8fp4 256 1024 256
./weight_quant_matmul_mxfp8fp4_swat_4_buffer 384 2048 1024
```
