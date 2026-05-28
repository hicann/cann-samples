# Grouped Matmul MXFP8量化矩阵乘算子

## 概述

本示例展示了Grouped Matmul MXFP8量化矩阵乘算子在昇腾AI处理器上的完整实现。算子以专家数进行分组，执行分组矩阵乘计算，适用于MoE等包含多专家分组计算的训推场景。

当前目录提供以下能力：

- `quant_grouped_matmul_mxfp8_split_m`：基于m轴分组、权重按ND（包括**NDExtLayout/DNExtLayout**，下面仅用ND统称）逻辑组织的分组量化矩阵乘示例。
- `quant_grouped_matmul_mxfp8_split_m_3buffer`：基于m轴分组、权重按ND逻辑组织的3buffer分组量化矩阵乘示例，其它示例均为2buffer。
- `quant_grouped_matmul_mxfp8_split_m_weight_nz`：基于m轴分组、权重按GM上NZ（包括**NZLayout/ZNLayout**，下面仅用NZ统称）存储的分组量化矩阵乘示例。
- `quant_grouped_matmul_mxfp8_split_k`：基于K轴分组、权重按ND逻辑组织的分组量化矩阵乘示例，仅支持`transA=true, transB=false`场景。
- `gen_data.py`：生成ND权重输入数据和CPU golden结果。
- `gen_data_weight_nz.py`：生成NZ权重输入数据和CPU golden结果。
- `verify_result.py`：校验NPU输出与CPU golden是否一致。

## 使用约束

当前样例需要满足以下约束条件：

- 当前支持`transA=false，transB=true`,`transA=false，transB=false`和`transA=true，transB=false`3种转置组合和M/K轴分组，其对应约束为:
  - 当M轴分组且`transA=false，transB=true`时，A的形状为`[M, K]`，B ND/NZ的形状为`[E, N, K]`/`[E, K1, N1, N0, K0]`，N0=16，K0=32，N1=ceil(N/N0)，K1=ceil(K/K0)。
  - 当M轴分组且`transA=false，transB=false`时，A的形状为`[M, K]`，B ND/NZ的形状为`[E, K, N]`/`[E, N1, K1, K0, N0]`，K0=16，N0=32，K1=ceil(K/K0)，N1=ceil(N/N0)。
  - 当K轴分组即`transA=true，transB=false`时，A的形状为`[K, M]`，B ND的形状为`[K, N]`。

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
- `group_value_list`：表示每个专家对应的分组大小，例如`128,128,0`
- `m`：总的`M`大小，要求M轴分组满足`m >= sum(group_value_list)`
- `k`：矩阵`A`的列数，同时也是每组矩阵`B`的列数，要求K轴分组满足`k >= sum(group_value_list)`
- `n`：每组矩阵`B`的行数，也是输出矩阵每组结果的列数
- `transA`：可选参数，默认值为`false`；`false`表示A以`[M, K]`组织,`true`表示A以`[K, M]`组织
- `transB`：可选参数，默认值为`true`；`true`表示B以`[E, N, K]`/`[E, K1, N1, N0, K0]`组织，`false`表示B以`[E, K, N]`/`[E, N1, K1, K0, N0]`组织

`transA`和`transB`需要同时省略或同时指定，取值支持`0/1/true/false`。

其中实际参与计算的`group_value_list`由数据生成脚本（`gen_data.py`或`gen_data_weight_nz.py`）生成，并写入`input/input_groupList.bin`。当前文件中保存的是每个分组各自的`分组值大小，允许某些组为`0`。

golden输入数据由对应的数据生成脚本生成。编译安装后请在`build_out`下的本示例目录中执行该脚本。

## 数据生成方式

`gen_data.py`（ND权重）与`gen_data_weight_nz.py`（NZ权重）支持以下两种调用方式，仅将脚本名替换即可：

### 方式一：显式指定`group_value_list`

```bash
python3 <gen_script>.py group_list group_value_list m k n [transA transB]
```

示例：

```bash
# ND权重
python3 gen_data.py group_list 128,128,0 384 256 256 false false
# NZ权重(仅支持M轴分组)
python3 gen_data_weight_nz.py group_list 128,128,0 384 256 256 false false
```

含义如下：

- `group_list`：显式分组模式，直接传入每个专家的分组大小。
- `group_value_list`：每个专家对应的分组大小，例如`128,128,0`
- `m`：矩阵乘的`m`维，M轴分组要求`m >= sum(group_value_list)`
- `k`：矩阵乘的`k`维，K轴分组要求`k >= sum(group_value_list)`
- `n`：矩阵乘的`n`维

### 方式二：按专家数和期望平均值随机生成`group_value_list`

```bash
python3 <gen_script>.py expect_m_per_group group_num expect_m_per_group m k n [transA transB]
```

示例：

```bash
# ND权重
python3 gen_data.py expect_m_per_group 3 128 384 256 256 false false
# NZ权重(仅支持M轴分组)
python3 gen_data_weight_nz.py expect_m_per_group 3 128 384 256 256 false false
```

含义如下：

- `expect_m_per_group`：随机分组模式，按每组期望分组大小随机生成分组
- `group_num`：专家数/分组数
- `expect_m_per_group`：每组期望平均分组大小
- `m`：矩阵乘的`m`维，M轴分组要求`m >= sum(group_value_list)`
- `k`：矩阵乘的`k`维，K轴分组要求`k >= sum(group_value_list)`
- `n`：矩阵乘的`n`维

在该模式下，脚本会随机生成长度为`group_num`的`group_value_list`，并保证：

- 每个分组大小均在`[floor(0.7 * expect_m_per_group),ceil(1.3 * expect_m_per_group)]`范围内

## 构建与运行

在仓库根目录下执行全量编译与安装，并进入安装目录：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --parallel
cmake --install build --prefix ./build_out
cd build_out/2_Performance/grouped_matmul_story/grouped_matmul_recipes/quant_grouped_matmul_mxfp8
```

之后可按需执行以下命令：

ND权重：

```bash
# 生成数据方式一：显式指定grouplist生成一组测试数据
python3 gen_data.py group_list 128,128,0 384 256 256

# 生成数据方式二：按专家数和平均M随机生成grouplist
python3 gen_data.py expect_m_per_group 3 128 384 256 256

# 运行可执行文件并校验结果（默认transA=false,transB=true）
./quant_grouped_matmul_mxfp8_split_m 3 384 256 256

# 运行M轴分组:transA=false, transB=true场景（显式指定transA/transB）
python3 gen_data.py group_list 128,128,0 384 256 256 false true
./quant_grouped_matmul_mxfp8_split_m 3 384 256 256 false true

# 运行M轴分组:transA=false, transB=false场景（显式指定transA/transB）
python3 gen_data.py group_list 128,128,0 384 256 256 false false
./quant_grouped_matmul_mxfp8_split_m 3 384 256 256 false false

# 运行M轴分组3buffer场景（显式指定transA/transB）
python3 gen_data.py group_list 128,128,0 384 256 256 false false
./quant_grouped_matmul_mxfp8_split_m_3buffer 3 384 256 256 false false

# 运行K轴分组:transA=true, transB=false场景（显式指定transA/transB）
python3 gen_data.py group_list 128,128,0 384 384 256 true false
./quant_grouped_matmul_mxfp8_split_k 3 384 384 256 true false
```

NZ权重(仅支持M轴分组)：

```bash
# 生成数据方式一：显式指定grouplist生成一组测试数据
python3 gen_data_weight_nz.py group_list 128,128,0 384 256 256

# 生成数据方式二：按专家数和平均M随机生成grouplist
python3 gen_data_weight_nz.py expect_m_per_group 3 128 384 256 256

# 运行可执行文件并校验结果（默认transA=false,transB=true）
./quant_grouped_matmul_mxfp8_split_m_weight_nz 3 384 256 256

# 运行transA=false, transB=false场景（显式指定transA/transB）
python3 gen_data_weight_nz.py group_list 128,128,0 384 256 256 false false
./quant_grouped_matmul_mxfp8_split_m_weight_nz 3 384 256 256 false false
```

```bash
# 可选：手动再次校验（用于调试/复核）
python3 verify_result.py 3 384 256 256
```
