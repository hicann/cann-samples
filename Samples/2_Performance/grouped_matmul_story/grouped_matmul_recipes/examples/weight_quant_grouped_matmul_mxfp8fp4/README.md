# Weight Quant Grouped Matmul MXFP8FP4量化矩阵乘算子

## 概述

本示例展示了 Weight Quant Grouped Matmul（`A: FP8(E4M3)`，`B: FP4(E2M1)`，输出 `BF16`）在昇腾 AI 处理器上的完整流程。算子以专家数进行分组计算：输入矩阵 `A` 在 `M` 维按组拼接，权重矩阵 `B` 按组独立存储，适用于 MoE 等多专家推理场景。

当前目录提供以下能力：

- `weight_quant_grouped_matmul_mxfp8fp4`：基于 `m` 轴分组的权重量化 grouped matmul 执行程序。
- `scripts/gen_data.py`：生成输入数据和 CPU golden 结果。
- `scripts/verify_result.py`：校验 NPU 输出与 CPU golden 是否一致。
- `scripts/batch_test_accuracy.py`：批量随机精度回归测试脚本。

## 使用约束

当前样例需要满足以下约束条件：

- 当前仅支持 `A` 不转置、`B` 转置场景。
- `A` 的形状为 `[M, K]`，`B` 的形状为 `[E, N, K]`。
- 当前仅支持 `m` 轴分组。
- `k` 需为 `64` 的正整数倍（数据生成与校验脚本按该约束实现）。
- `n` 需为 `32` 的正整数倍（精度回归脚本按该约束实现）。
- `m` 需满足 `m >= sum(group_m_list)`。

## 支持架构

NPU ARCH 3510

## API参考

[Ascend C API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0003.html)

## 输入参数

算子执行文件与结果校验脚本的命令行参数格式一致：

```text
<program> group_num m k n
```

- `group_num`：专家数，也就是分组数。
- `m`：总的 `M` 大小，要求满足 `m >= sum(group_m_list)`。
- `k`：矩阵 `A` 的列数，同时也是每组矩阵 `B` 的列数（要求为 `64` 的倍数）。
- `n`：每组矩阵 `B` 的行数，也是输出矩阵每组结果的列数（要求为 `32` 的倍数）。

其中实际参与计算的 `group_m_list` 由 `gen_data.py` 生成并写入 `input/input_groupList.bin`。该文件保存每个分组各自的 `M` 大小，允许某些组为 `0`。

golden 输入数据由 `scripts/gen_data.py` 生成。安装后该脚本与可执行文件位于同一目录，脚本名为 `gen_data.py`（无 `scripts/` 前缀）。

## 数据生成方式

`gen_data.py` 支持以下两种调用方式（源码树在示例目录下使用 `scripts/gen_data.py`，安装目录下直接使用 `gen_data.py`）：

### 方式一：显式指定 `group_m_list`

```bash
python3 gen_data.py group_list group_m_list m k n
```

示例：

```bash
python3 gen_data.py group_list 128,128,0 384 256 256
```

含义如下：

- `group_list`：显式分组模式，直接传入每个专家的分组大小。
- `group_m_list`：每个专家对应的分组大小，例如 `128,128,0`。
- `m`：总的 `M` 上限，要求 `m >= sum(group_m_list)`。
- `k`：矩阵乘的 `k` 维（需为 `64` 的倍数）。
- `n`：矩阵乘的 `n` 维（需为 `32` 的倍数）。

### 方式二：按专家数和期望平均值随机生成 `group_m_list`

```bash
python3 gen_data.py expect_m_per_group group_num expect_m_per_group m k n
```

示例：

```bash
python3 gen_data.py expect_m_per_group 3 128 384 256 256
```

含义如下：

- `expect_m_per_group`：随机分组模式，按每组期望分组大小随机生成分组。
- `group_num`：专家数 / 分组数。
- `expect_m_per_group`：每组期望平均分组大小。
- `m`：总的 `M` 上限，要求 `m >= sum(group_m_list)`。
- `k`：矩阵乘的 `k` 维（需为 `64` 的倍数）。
- `n`：矩阵乘的 `n` 维（需为 `32` 的倍数）。

在该模式下，脚本会随机生成长度为 `group_num` 的 `group_m_list`，并保证：

- 每个分组大小均在 `[floor(0.7 * expect_m_per_group), ceil(1.3 * expect_m_per_group)]` 范围内。
- `sum(group_m_list) <= m`。

`gen_data.py` 生成的文件：

```text
CPU golden: output/output_cpu.bin
input/input_a.bin
input/input_b.bin
input/input_scaleA.bin
input/input_scaleB.bin
input/input_groupList.bin
```

运行 `weight_quant_grouped_matmul_mxfp8fp4` 后会额外生成：

```text
NPU output: output/output_npu.bin
```

## 构建与运行

在仓库根目录下执行全量编译与安装，并进入安装目录：

```bash
cmake -S . -B build
cmake --build build --parallel
cmake --install build --prefix ./build_out
cd build_out/2_Performance/grouped_matmul_story/grouped_matmul_recipes/weight_quant_grouped_matmul_mxfp8fp4
```

之后可按需执行以下命令：

```bash
# 生成数据方式一：显式指定 group_list 生成一组测试数据
python3 gen_data.py group_list 128,128,0 384 256 256

# 生成数据方式二：按专家数和平均 M 随机生成 group_list
python3 gen_data.py expect_m_per_group 3 128 384 256 256

# 运行可执行文件（以上面的 group_list 示例为例）
# 程序会在执行完成后自动调用 verify_result.py 进行结果校验
./weight_quant_grouped_matmul_mxfp8fp4 3 384 256 256

# 可选：手动再次校验（用于调试/复核）
python3 verify_result.py 3 384 256 256
```
