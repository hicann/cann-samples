# Grouped Matmul MXFP4量化矩阵乘算子

## 概述

本示例展示了Grouped Matmul MXFP4量化矩阵乘算子在昇腾AI处理器上的完整实现。算子以专家数进行分组，执行分组矩阵乘计算，输入矩阵`A`在`M`维按组拼接，权重矩阵`B`按组独立存储，适用于MoE等包含多专家分组计算的推理场景。

当前目录提供以下能力：

- `quant_grouped_matmul_mxfp4_split_m`：基于m轴分组、权重按ND（包括**NDExtLayout/DNExtLayout**，下面仅用ND统称）逻辑组织的分组量化矩阵乘示例。
- `quant_grouped_matmul_mxfp4_split_m_weight_nz`：基于m轴分组、权重按GM上NZ（包括**NZLayout/ZNLayout**，下面仅用NZ统称）存储的分组量化矩阵乘示例。
- `gen_data.py`：生成ND权重输入数据和CPU golden结果。
- `gen_data_weight_nz.py`：生成NZ权重输入数据和CPU golden结果。
- `verify_result.py`：校验NPU输出与CPU golden是否一致。

## 使用约束

当前样例需要满足以下约束条件：

- 当前仅支持`transA=false，transB=true`和`transA=false，transB=false`两种场景。
- 当`transB=true`时，A的形状为`[M, K]`，B ND/NZ的形状为`[E, N, K]`/`[E, K1, N1, N0, K0]`，N0=16，K0=64，N1=ceil(N/N0)，K1=ceil(K/K0)。
- 当`transB=false`时，A的形状为`[M, K]`，B ND/NZ的形状为`[E, K, N]`/`[E, N1, K1, K0, N0]`,K0=16,N0=64,K1=ceil(K/K0)，N1=ceil(N/N0)。
- 当前仅支持m轴分组。
- MXFP4内轴`K`必须为偶数；当`transB=false`时，`N`也必须为偶数。

## 支持架构

NPU ARCH 3510

## API参考

[Ascend C API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0003.html)

## 输入参数

算子执行文件与结果校验脚本的命令行参数格式一致：

```text
<program> group_num m k n [transA transB]
```

- `group_num`：专家数，也就是分组数
- `group_m_list`：表示每个专家对应的分组大小，例如`128,128,0`
- `m`：总的`M`大小，要求满足`m >= sum(group_m_list)`
- `k`：矩阵`A`的列数，同时也是每组矩阵`B`的列数
- `n`：每组矩阵`B`的行数，也是输出矩阵每组结果的列数
- `transA`：可选参数，当前仅支持`false`，默认值为`false`
- `transB`：可选参数，默认值为`true`；`true`表示B以`[E, N, K]`/`[E, K1, N1, N0, K0]`组织，`false`表示B以`[E, K, N]`/`[E, N1, K1, K0, N0]`组织

`transA`和`transB`需要同时省略或同时指定，取值支持`0/1/true/false`。

其中实际参与计算的`group_m_list`由数据生成脚本（`gen_data.py`或`gen_data_weight_nz.py`）生成，并写入`input/input_groupList.bin`。当前文件中保存的是每个分组各自的`M`大小，允许某些组为`0`。

golden输入数据由对应的数据生成脚本生成。编译安装后请在`build_out`下的本示例目录中执行该脚本（ND权重用`gen_data.py`，NZ权重用`gen_data_weight_nz.py`）。

## 数据生成方式

`gen_data.py`（ND权重）与`gen_data_weight_nz.py`（NZ权重）支持以下两种调用方式，仅将脚本名替换即可：

### 方式一：显式指定`group_m_list`

```bash
# ND权重
python3 gen_data.py group_list group_m_list m k n [transA transB]
# NZ权重
python3 gen_data_weight_nz.py group_list group_m_list m k n [transA transB]
```

示例：

```bash
# ND权重
python3 gen_data.py group_list 128,128,0 384 256 256 false false
# NZ权重
python3 gen_data_weight_nz.py group_list 128,128,0 384 256 256 false false
```

含义如下：

- `group_list`：显式分组模式，直接传入每个专家的分组大小。
- `group_m_list`：每个专家对应的分组大小，例如`128,128,0`
- `m`：总的`M`上限，要求`m >= sum(group_m_list)`
- `k`：矩阵乘的`k`维
- `n`：矩阵乘的`n`维

### 方式二：按专家数和期望平均值随机生成`group_m_list`

```bash
# ND权重
python3 gen_data.py expect_m_per_group group_num expect_m_per_group m k n [transA transB]
# NZ权重
python3 gen_data_weight_nz.py expect_m_per_group group_num expect_m_per_group m k n [transA transB]
```

示例：

```bash
# ND权重
python3 gen_data.py expect_m_per_group 3 128 384 256 256 false false
# NZ权重
python3 gen_data_weight_nz.py expect_m_per_group 3 128 384 256 256 false false
```

含义如下：

- `expect_m_per_group`：随机分组模式，按每组期望分组大小随机生成分组
- `group_num`：专家数 / 分组数
- `expect_m_per_group`：每组期望平均分组大小
- `m`：总的`M`上限，要求`m >= sum(group_m_list)`
- `k`：矩阵乘的`k`维
- `n`：矩阵乘的`n`维

在该模式下，脚本会随机生成长度为`group_num`的`group_m_list`，并保证：

- 每个分组大小均在`[floor(0.7 * expect_m_per_group), ceil(1.3 * expect_m_per_group)]`范围内
- `sum(group_m_list) <= m`

## 构建与运行

在仓库根目录下执行全量编译与安装，并进入安装目录：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --parallel
cmake --install build --prefix ./build_out
cd build_out/2_Performance/grouped_matmul_story/grouped_matmul_recipes/quant_grouped_matmul_mxfp4
```

之后可按需执行以下命令：

ND权重：
```bash
# 生成数据方式一：显式指定grouplist生成一组测试数据
python3 gen_data.py group_list 128,128,0 384 256 256

# 生成数据方式二：按专家数和平均M随机生成grouplist
python3 gen_data.py expect_m_per_group 3 128 384 256 256

# 运行可执行文件并校验结果（默认transA=false, transB=true）
./quant_grouped_matmul_mxfp4 3 384 256 256

# 运行transB=false场景（显式指定transA/transB）
python3 gen_data.py group_list 128,128,0 384 256 256 false false
./quant_grouped_matmul_mxfp4 3 384 256 256 false false
```

NZ权重：
```bash
# 生成数据方式一：显式指定grouplist生成一组测试数据
python3 gen_data_weight_nz.py group_list 128,128,0 384 256 256

# 生成数据方式二：按专家数和平均M随机生成grouplist
python3 gen_data_weight_nz.py expect_m_per_group 3 128 384 256 256

# 运行可执行文件并校验结果（默认transA=false, transB=true）
./quant_grouped_matmul_mxfp4_split_m_weight_nz 3 384 256 256

# 运行transB=false场景（显式指定transA/transB）
python3 gen_data_weight_nz.py group_list 128,128,0 384 256 256 false false
./quant_grouped_matmul_mxfp4_split_m_weight_nz 3 384 256 256 false false
```

```bash
# 可选：手动再次校验（用于调试/复核）
python3 verify_result.py 3 384 256 256
```
