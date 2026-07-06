# INT8 量化矩阵乘样例

## 概述

本示例演示 **INT8** 激活与权重、**INT32** 输出的矩阵乘路径，使用 `quant_matmul_int8` SWAT（非全载）tiling / block / kernel 头文件。输入矩阵 A 和 B 均为 `int8_t`，输出矩阵 C 为 `int32_t`，利用 Cube Core 的 INT8 MMA 指令在 L0C 中以 INT32 累加。

## 目录与脚本

- `quant_matmul_int8.asc`：宿主 + kernel 联合体。
- `scripts/gen_data.py`：生成 `input/`、`output/`（含 `cpu_output.bin`）。
- `scripts/verify_result.py`：校验 `./output/npu_out.bin` 与 `./output/cpu_output.bin`。
- `scripts/run.sh`：一键串联构建、数据生成、算子执行与校验（详见下文「一键运行」）。

## 支持架构

NPU ARCH 3510（Ascend 950）

## 使用约束

- 当前样例仅支持 `dav-3510` 架构，使用其它架构构建时本目标会被跳过。
- 支持通过命令行参数 `transA`/`transB` 选择 A/B 矩阵转置，二者须同时省略或同时给出；省略时等价于 `transA=false`、`transB=true`。
- `gen_data.py` 与可执行程序的转置参数须保持一致，否则 NPU 输出与 CPU golden 不匹配会导致校验失败。

## 构建

在 `matmul_recipes` 工程内随其他 recipe 一并构建；目标名为 `quant_matmul_int8`。

## 参数说明

```text
<program> m k n [transA transB]
```

- `m`：矩阵 A 的行数。
- `k`：矩阵 A 的列数，同时也是矩阵 B 的归约维。
- `n`：矩阵 B 的列数，对应输出矩阵的列数。
- `transA`（可选）：A 矩阵转置信息（`0/1/true/false/t/f`）。`0`/`false`/`f` 表示非转置，`1`/`true`/`t` 表示转置。默认为非转置。
- `transB`（可选）：B 矩阵转置信息（取值同上）。默认为转置。

输出矩阵 C 的逻辑形状为 `[M, N]`。

## 一键运行（推荐）

仓库提供 `run.sh`（位于 `scripts/`），可一键串联**构建 → 数据生成 → 算子执行 → 结果校验**全流程。推荐先进入样例目录再执行：

```bash
cd Samples/2_Performance/matmul_story/matmul_recipes/examples/quant_matmul_int8

# 默认 transA=false、transB=true
bash scripts/run.sh 1024 2048 4096

# 指定转置组合（须与 gen_data.py 保持一致，run.sh 已自动透传）
bash scripts/run.sh 1024 2048 4096 true true

# 跳过重新构建，复用已有 build_out
bash scripts/run.sh --skip-build 1024 2048 4096

# 查看完整帮助
bash scripts/run.sh --help
```

## 手动构建与运行

如需手动控制各步骤，可在仓库根目录下完成编译和安装后，进入当前样例安装目录：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target quant_matmul_int8
cmake --install build --prefix ./build_out
cd build_out/2_Performance/matmul_story/matmul_recipes/quant_matmul_int8
```

### 1. 生成测试数据

```bash
# 默认 transA=false、transB=true
python3 gen_data.py 1024 2048 4096
# 或指定转置组合
python3 gen_data.py 1024 2048 4096 true true
```

### 2. 运行样例

```bash
# 转置参数须与 gen_data.py 保持一致
./quant_matmul_int8 1024 2048 4096
./quant_matmul_int8 1024 2048 4096 true true
```

可执行程序在结束后会自动执行：

```bash
python3 verify_result.py <m> <n>
```

运行成功后，终端将打印如下类似信息：

```txt
[PASS] NPU results are consistent with CPU.
```
