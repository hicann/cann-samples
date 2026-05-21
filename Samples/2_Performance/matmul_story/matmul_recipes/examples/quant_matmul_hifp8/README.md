# HiFloat8 量化矩阵乘样例

## 概述

本示例演示 **HiFloat8** 激活与权重、`bfloat16` 输出的高性能矩阵乘路径，使用本目录同名的 **`quant_matmul_hifp8_*` tiling / block / kernel** 头文件（SWAT、非全载）。与仓库内 MX 系列样例相同，源码以 `.asc` 组织，CMake 生成的可执行文件安装在与本目录同名的 **`quant_matmul_hifp8`** 安装子目录下。

提供两个宿主入口，对应两种量化缩放语义（**数据目录均为** `input/`、`output/`，无 `TT`/`TC` 子目录；跑某个可执行前请用对应脚本重新生成数据，以免被另一模式覆盖）：

| 可执行文件名 | 说明 |
|--------------|------|
| `quant_matmul_hifp8_tt` | **双标量缩放**（tiling：`x1`/`x2` 均为 `KERNEL_QUANT_PERTENSOR`）：磁盘 `pertoken_scale.bin` + `scale.bin`（FP32） |
| `quant_matmul_hifp8_tc` | **按通道缩放**（tiling：`x1`=`KERNEL_QUANT_DEFAULT` / None，`x2`=`KERNEL_QUANT_PERCHANNEL`）：磁盘 `scale.bin` 为长度 **N** 的 **FP32（per-channel）**；宿主把每个标度转为 **uint64**（低 32b 为 DEQ 对齐后的 FP32 位型，`0xFFFFE000`；并 **`(uint64)1<<46`** 元数据位）再 H2D，与 golden 一致 |

## 目录与脚本

- `quant_matmul_hifp8_tt.asc` / `quant_matmul_hifp8_tc.asc`：宿主 + kernel 联合体（命名与仓库内 `quant_matmul_mxfp8_*.cpp` / `.asc` 风格一致：**quant_matmul\_** 前缀，模式后缀 **`_tt` / `_tc` 全小写**）。
- `scripts/gen_data_tt.py`：生成 TT 用的 `input/`、`output/`（含 `cpu_output.bin`）。
- `scripts/gen_data_tc.py`：生成 TC 用的 `input/`、`output/`（`scale.bin` 为 `N*sizeof(float)` 字节；会覆盖同名 bin，请先按需运行其一）。
- `scripts/verify_result.py`：校验 `./output/npu_out.bin` 与 `./output/cpu_output.bin`（与当前运行的 tt/tc 及刚生成的 golden 一致即可）。

## 构建

在 **`matmul_recipes`** 工程内随其他 recipe 一并构建；目标名为 **`quant_matmul_hifp8_tt`**、 **`quant_matmul_hifp8_tc`**。

## 参数说明

与 `matmul_a16w16` / MX 量子样例一致：

```text
<program> m k n [transA transB]
```

- `transA`、`transB` 须同时省略或同时给出；省略时等价于 **`transA=false`、`transB=true`**（与 `matmul_a16w16/scripts/gen_data.py` 默认一致）。

## 数据生成与校验

在**可执行文件所在目录**（安装后 `build_out/.../matmul_recipes/quant_matmul_hifp8/`，脚本与可执行文件同级；源码树中可在本示例目录下使用 `scripts/` 前缀调用）：

**运行 TT：**

```bash
python3 gen_data_tt.py <m> <k> <n>
./quant_matmul_hifp8_tt <m> <k> <n>
```

（源码树路径示例：`python3 scripts/gen_data_tt.py <m> <k> <n>`。）

**运行 TC：**

```bash
python3 gen_data_tc.py <m> <k> <n>
./quant_matmul_hifp8_tc <m> <k> <n>
```

可执行程序在结束后会执行：

```bash
python3 verify_result.py <m> <n>
```

也可在可执行目录下手动运行上述命令，对比 `output/cpu_output.bin` 与 `output/npu_out.bin`。

## 支持与参考

- 架构与矩阵乘教程一致时，请参考上级目录 **[MX量化矩阵乘算子性能优化指南](../../../docs/quant_matmul_mx_performance.md)** 中的通用流水线思路；HiFloat8 数据类型简介见 **`Samples/1_Features/hardware_features/hif8/`**。
