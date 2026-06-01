# Grouped Matmul HiFloat8量化矩阵乘算子

## 概述

本示例展示了Grouped Matmul HiFloat8量化矩阵乘算子在昇腾AI处理器上的完整实现。算子以专家数进行分组，执行分组矩阵乘计算，适用于MoE等包含多专家分组计算的训推场景。

当前目录提供以下能力：

- `quant_grouped_matmul_hif8_split_m_tt`：基于M轴分组、采用T-T[量化模式](../../../../../../docs/zh/context/量化介绍.md)的分组量化矩阵乘示例。每个group一组FP32标量`pertoken_scale.bin` + `scale.bin`。
- `quant_grouped_matmul_hif8_split_m_tc`：基于M轴分组、采用T-C[量化模式](../../../../../../docs/zh/context/量化介绍.md)的分组量化矩阵乘示例。每个group的N维`uint64` per-channel `scale.bin`。
- `scripts/gen_data_tt.py`：生成T-T[量化模式](../../../../../../docs/zh/context/量化介绍.md)输入数据和CPU golden结果。
- `scripts/gen_data_tc.py`：生成T-C[量化模式](../../../../../../docs/zh/context/量化介绍.md)输入数据和CPU golden结果。
- `scripts/verify_result.py`：校验NPU输出与CPU golden是否一致。

## 使用约束

当前样例需要满足以下约束条件：

- 当前仅支持M轴分组（split-M路径），`transA`必须为`false`。
- 支持`transB=true`和`transB=false`两种场景：
  - 当`transB=true`时，A的形状为`[M, K]`，B的形状为`[E, N, K]`。
  - 当`transB=false`时，A的形状为`[M, K]`，B的形状为`[E, K, N]`。
- 输入数据类型为`hifloat8`，输出数据类型为`bfloat16`。

## 支持架构

NPU ARCH 3510

## API参考

[Ascend C API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0003.html)

## 输入参数

算子执行文件与结果校验脚本的命令行参数格式一致：

```text
<program> group_num m k n [transA transB]
```

- `program`：可执行文件，当前支持`quant_grouped_matmul_hif8_split_m_tt`和`quant_grouped_matmul_hif8_split_m_tc`
- `group_num`：专家数，也就是分组数
- `group_value_list`：表示每个专家对应的分组大小，例如`64,80,96`
- `m`：总的`M`大小，要求`m >= sum(group_value_list)`
- `k`：矩阵`A`的列数，同时也是每组矩阵`B`的行数（`transB=true`时）或列数（`transB=false`时）
- `n`：每组矩阵`B`的列数（`transB=true`时）或行数（`transB=false`时），也是输出矩阵每组结果的列数
- `transA`：可选参数，默认值为`false`；当前仅支持`false`
- `transB`：可选参数，默认值为`true`；`true`表示B以`[E, N, K]`组织，`false`表示B以`[E, K, N]`组织

`transA`和`transB`需要同时省略或同时指定，取值支持`0/1/true/false`。

其中实际参与计算的`group_value_list`由数据生成脚本（`gen_data_tt.py`或`gen_data_tc.py`）生成，并写入`input/input_groupList.bin`。当前文件中保存的是每个分组各自的分组值大小，允许某些组为`0`。

golden输入数据由对应的数据生成脚本生成。编译安装后请在`build_out`下的本示例目录中执行该脚本。

## 数据生成方式

`scripts/gen_data_tt.py`（T-T[量化模式](../../../../../../docs/zh/context/量化介绍.md)）与`scripts/gen_data_tc.py`（T-C[量化模式](../../../../../../docs/zh/context/量化介绍.md)）支持以下两种调用方式：

### 方式一：显式指定`group_value_list`

```bash
python3 scripts/<gen_script>.py group_list group_value_list m k n [transA transB]
```

示例：

```bash
# T-T量化模式
python3 scripts/gen_data_tt.py group_list 64,80,96 256 128 256
# T-C量化模式
python3 scripts/gen_data_tc.py group_list 64,80,96 256 128 256 false false
```

含义如下：

- `group_list`：显式分组模式，直接传入每个专家的分组大小。
- `group_value_list`：每个专家对应的分组大小，例如`64,80,96`
- `m`：矩阵乘的`m`维，要求`m >= sum(group_value_list)`
- `k`：矩阵乘的`k`维
- `n`：矩阵乘的`n`维

### 方式二：按专家数和期望平均值随机生成`group_value_list`

```bash
python3 scripts/<gen_script>.py expect_m_per_group group_num expect_m_per_group m k n [transA transB]
```

示例：

```bash
# T-T量化模式
python3 scripts/gen_data_tt.py expect_m_per_group 3 80 256 128 256
# T-C量化模式
python3 scripts/gen_data_tc.py expect_m_per_group 3 80 256 128 256 false false
```

含义如下：

- `expect_m_per_group`：随机分组模式，按每组期望分组大小随机生成分组
- `group_num`：专家数/分组数
- `expect_m_per_group`：每组期望平均分组大小
- `m`：矩阵乘的`m`维，要求`m >= sum(group_value_list)`
- `k`：矩阵乘的`k`维
- `n`：矩阵乘的`n`维

在该模式下，脚本会随机生成长度为`group_num`的`group_value_list`，并保证：

- 每个分组大小均在`[floor(0.7 * expect_m_per_group), ceil(1.3 * expect_m_per_group)]`范围内

## 构建与运行

在仓库根目录下执行全量编译与安装，并进入安装目录：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --parallel
cmake --install build --prefix ./build_out
cd build_out/2_Performance/grouped_matmul_story/grouped_matmul_recipes/quant_grouped_matmul_hif8
```

之后可按需执行以下命令：

T-T[量化模式](../../../../../../docs/zh/context/量化介绍.md)：

```bash
# 生成数据方式一：显式指定grouplist生成一组测试数据
python3 scripts/gen_data_tt.py group_list 64,80,96 256 128 256

# 生成数据方式二：按专家数和平均M随机生成grouplist
python3 scripts/gen_data_tt.py expect_m_per_group 3 80 256 128 256

# 运行可执行文件并校验结果（默认transA=false,transB=true）
./quant_grouped_matmul_hif8_split_m_tt 3 256 128 256

# 运行transA=false, transB=false场景（显式指定transA/transB）
python3 scripts/gen_data_tt.py group_list 64,80,96 256 128 256 false false
./quant_grouped_matmul_hif8_split_m_tt 3 256 128 256 false false
```

T-C[量化模式](../../../../../../docs/zh/context/量化介绍.md)：

```bash
# 生成数据方式一：显式指定grouplist生成一组测试数据
python3 scripts/gen_data_tc.py group_list 64,80,96 256 128 256

# 生成数据方式二：按专家数和平均M随机生成grouplist
python3 scripts/gen_data_tc.py expect_m_per_group 3 80 256 128 256

# 运行可执行文件并校验结果（默认transA=false,transB=true）
./quant_grouped_matmul_hif8_split_m_tc 3 256 128 256

# 运行transA=false, transB=false场景（显式指定transA/transB）
python3 scripts/gen_data_tc.py group_list 64,80,96 256 128 256 false false
./quant_grouped_matmul_hif8_split_m_tc 3 256 128 256 false false
```

```bash
# 可选：手动再次校验（用于调试/复核）
python3 scripts/verify_result.py 3 256 128 256
```