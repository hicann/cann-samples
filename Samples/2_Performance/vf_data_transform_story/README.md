# SIMD VF 数据变换性能优化实践

数据类型转换和布局转换常出现在量化、反量化、位置编码以及算子融合的边界。单次转换的计算量不大，但寄存器 lane 排布、UB 访存方式和写回布局会直接决定 Vector 流水的效率。

本专题面向 Ascend 950（`dav-3510`），提供 7 个可独立构建、运行和验证的 tutorial，共包含 7 个基线和 18 个后续阶段。每个 tutorial 固定 host、shape、tile 和双缓冲框架，从当前优化路径的第一个技巧开始，逐项引入后续技巧，并使用 CANNsim trace 解释性能变化。

## 目录结构

```text
vf_data_transform_story/
├── CMakeLists.txt
├── README.md
└── vf_data_transform_tutorials/
    ├── CMakeLists.txt
    ├── README.md
    ├── include/                         # ACL 运行时和二进制 I/O 公共辅助代码
    ├── scripts/                         # 数据生成、结果校验和性能采集脚本
    ├── half_to_int8/                    # HALF 窄化为 INT8
    ├── fp32_to_fp8/                     # FP32 窄化为 FP8 E4M3FN
    ├── int8_to_half/                    # INT8 扩展为 HALF
    ├── int4_to_bf16/                    # packed INT4 扩展为 BF16
    ├── fp4_to_fp8/                      # FP4 E2M1 raw-field 映射为 FP8 E4M3FN
    ├── bf16_nd2nz/                      # BF16 ND 转 NZ
    └── bf16_rope_nd2nz/                 # BF16 RoPE 与 ND2NZ 融合
```

每个功能目录包含一篇完整教程和连续编号的阶段目录。每个阶段目录内都有独立 `.asc` 源码与 `CMakeLists.txt`，源码完整包含该阶段的 VF、kernel、搬运流程、shape 约束和 host 调用，不引用其他阶段或功能级实现头文件，可以单独阅读、构建和运行。`vf_data_transform_tutorials/include/` 仅封装 ACL 初始化、设备内存和二进制 I/O 等公共辅助能力，不承载数据转换实现。

## Tutorial 列表

| Tutorial | 优化路径 | 主要问题 | 最终目标 |
| --- | --- | --- | --- |
| [HALF → INT8](./vf_data_transform_tutorials/half_to_int8) | `0_dintlv_load` → `1_complementary_cast` → `2_or_merge` → `3_contiguous_store` | 如何用互补 lane 减少窄化转换的寄存器整理与 Store | `vf_data_transform_half_to_int8` |
| [FP32 → FP8](./vf_data_transform_tutorials/fp32_to_fp8) | `0_dintlv_b32_load` → `1_complementary_cast` → `2_pack_b16_store` → `3_shared_b8_mask` | 如何用互补 lane 和 pack Store 完成 4:1 窄化 | `vf_data_transform_fp32_to_fp8` |
| [INT8 → HALF](./vf_data_transform_tutorials/int8_to_half) | `0_compact_b8_load` → `1_complementary_cast` → `2_intlv_b16_store` → `3_b8_cast_mask` | 紧凑读取和交织写出如何改变整体流水 | `vf_data_transform_int8_to_half` |
| [INT4 → BF16](./vf_data_transform_tutorials/int4_to_bf16) | `0_unpack4_load` → `1_single_cast` → `2_separate_views` → `3_b8_predicate` | 如何在 Load 阶段展开 nibble 并用单路 Cast 输出 | `vf_data_transform_int4_to_bf16` |
| [FP4 → FP8](./vf_data_transform_tutorials/fp4_to_fp8) | `0_raw_field_mapping` → `1_unpack_packed_byte` → `2_shift_select_and` → `3_hoist_constants` → `4_fold_scale` | 没有直接 Cast 时如何完成 raw-field 转换 | `vf_data_transform_fp4_to_fp8` |
| [BF16 ND2NZ](./vf_data_transform_tutorials/bf16_nd2nz) | `0_data_block_copy` → `1_conflict_padding` | 跨步 Store 如何规避 UB bank conflict | `vf_data_transform_bf16_nd2nz` |
| [BF16 RoPE + ND2NZ](./vf_data_transform_tutorials/bf16_rope_nd2nz) | `0_pack_or_fusion` → `1_conflict_padding` | 如何融合计算和 NZ 写出并规避 bank conflict | `vf_data_transform_bf16_rope_nd2nz` |

前四个转换中，Load 分布模式、`RegLayout`、寄存器合并和 Store 分布模式共同决定输出顺序。这些修改存在数据排布依赖：中间阶段使用显式 Pack、Interleave 或普通 Store 保持结果正确，后续阶段再逐项移除这些过渡操作。因此局部阶段可能出现性能回退，应结合指令变化、完整路径和最终结果判断方案。跨功能点的分析规则见 [通用优化方法](#通用优化方法)。

## 性能结果汇总

下表对比每条优化路径的起点和最终阶段。数据均来自各阶段保留的 Ascend950PR_9589 CANNsim `trace` 报告；不同 tutorial 的 tile 大小和有效字节数不同，不应直接横向比较 cycles。

| Tutorial | 起点 VF cycles | 最终 VF cycles | 路径总体结果 |
| --- | ---: | ---: | ---: |
| HALF → INT8 | 780 | 315 | 2.48x，减少 59.6% |
| FP32 → FP8 | 1250 | 379 | 3.30x，减少 69.7% |
| INT8 → HALF | 446 | 458 | 0.97x，增加 2.7% |
| INT4 → BF16 | 699 | 453 | 1.54x，减少 35.2% |
| FP4 → FP8 | 1301 | 698 | 1.86x，减少 46.3% |
| BF16 ND2NZ | 1672 | 349 | 4.79x，减少 79.1% |
| BF16 RoPE + ND2NZ | 993 | 905 | 1.10x，减少 8.9% |

INT8 → HALF 的互补 Cast 和 INTLV Store 是一组布局依赖优化。最终路径减少了 Load/Store 指令，但 INTLV Store 在当前 CAMODEL 上的时延抵消了指令数收益。该路径的价值在于为后续融合提供紧凑输入和两路 HALF 寄存器；若只执行独立转换，应在目标硬件和实际 shape 上与简单路径重新比较。

## 支持范围

- 硬件架构：Ascend 950，`dav-3510`，64 个 AIV。
- 软件版本：CANN 9.2.0；ASC 编译器使用 `-O3`。
- 编程模型：RegBase SIMD VF（`__simd_vf__`、`asc_vf_call`）和 tensor API。
- 输入布局：Cast 和 raw-field tutorial 使用连续二维 ND；ND2NZ tutorial 的 `N` 固定为 128。
- 尾块处理：输入按当前阶段的完整寄存器组补齐，写回时只复制有效区；ND2NZ 输出的 `M` 对齐到 16。
- 正确性：逐 bit 比较有效输出，同时检查输出区两侧 guard，验证越界写。

各 tutorial 的数据类型、shape、舍入语义、缩放约定和可复用边界见对应 README。

## 构建

在仓库根目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DNPU_ARCH=dav-3510 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vf_data_transform_story -j4
```

也可以只构建一个 tutorial：

```bash
cmake --build build --target vf_data_transform_half_to_int8_tutorial -j4
cmake --build build --target vf_data_transform_bf16_rope_nd2nz_tutorial -j4
```

## 运行与验证

先安装 Python 数据生成依赖：

```bash
python3 -m pip install -r requirements.txt
```

每个可执行文件支持 `full`、`tail`、`perf`、`trace` 和 `all`。以 HALF → INT8 最终阶段为例：

```bash
./build/Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/half_to_int8/3_contiguous_store/vf_data_transform_half_to_int8 --case all
```

程序会生成固定模式输入和 golden，执行 kernel 后进行 bitwise 校验。通过时输出包含以下关键字段，`...` 表示省略的 shape 和 tile 信息：

```text
[HOST][tail] status=PASS ... bin_match=true
```

数据默认保存在可执行文件旁的 `artifacts/<tutorial>/<case>/`。可通过 `VF_DATA_TRANSFORM_ARTIFACT_DIR` 指定其他目录。

## 性能采集

教程优先使用 CANNsim 分析单核 VF 流水。统一脚本会对指定 tutorial 的全部阶段运行同一个 `trace` case：

```bash
bash Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/scripts/profile_tutorial.sh \
  cannsim bf16_nd2nz build /tmp/vf_data_transform_cannsim trace
```

脚本在新版本工具包中优先调用 `npusim`，并兼容仍使用 `cannsim` 命令的版本。若当前 Python 环境没有 `plotly`，先安装报告渲染依赖：

```bash
python3 -m pip install plotly
```

本文数据的统一口径如下：

- Ascend950PR_9589 CAMODEL V100，Vector 频率 1.65 GHz。
- CANN 9.2.0，Release 构建，ASC `-O3`。
- `trace` case 启动 64 个 AIV，每个 AIV 处理一个完整 tile；表格读取 core 0 的业务 kernel。
- `VF cycles` 取业务 kernel VF 报告中的 `avg_cycles`；指令数按 VF execution 数归一化，避免把重复采样累计两次。
- `VF cycles` 只表示 `asc_vf_call` 内部的局部流水，不包含 GM 搬运、同步和 kernel 启动开销。
- 基线和优化阶段使用相同 shape、tile、缓冲数量和验证流程。

各阶段的指令数量、cycle、瓶颈变化和适用条件在对应 tutorial 中给出，性能结论直接使用同一测试口径下的数据进行比较。

需要观察完整 kernel 时延时，可在真机上使用同一脚本的 `msopprof` 模式。脚本对每个阶段采集三次，汇总脚本输出原始值和中位数：

```bash
bash Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/scripts/profile_tutorial.sh \
  msopprof bf16_nd2nz build /tmp/vf_data_transform_msopprof
python3 Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials/scripts/summarize_msopprof.py \
  /tmp/vf_data_transform_msopprof
```

CANNsim 的 VF 局部收益和真机端到端收益应分别解读；GM 搬运或 kernel 固定开销占主导时，两者不会按相同比例变化。

## 通用优化方法

> 本节总结 7 个数据变换样例共用的设计和验证方法。具体 API 及阶段代码以各功能点的教程为准。

### G1：先确定完整的 lane 数据流

数据类型转换不能只看 `Cast`。优化前应先确定一组输入在 Load、寄存器 lane、Cast layout 和 Store 之间的完整映射：

- Load 一次读取的物理字节数和逻辑元素数；
- `LoadDist` 产生的寄存器数量及各路 lane 顺序；
- `RegLayout` 在窄化或扩展后保留的有效 lane；
- 输出是由 `Or`、`Pack`、INTLV Store 还是 pack Store 恢复连续顺序。

当 Load 和 Store 的分布模式能直接完成 lane 重排时，应删除显式 `Pack`、`UnPack`、`Interleave` 和中间寄存器。每次改变 layout 后都要重新核对输出顺序，不能仅根据 API 名称推断结果。

### G2：mask 必须跟随 API 的物理 lane 视图

predicate 类型取决于当前 API 消费的寄存器 lane，不能机械地跟随 GM 或 UB Tensor 的声明类型。例如，compact INT8 的 widening Cast 需要 B8 mask，packed INT4 的 Cast 也使用 packed-byte 的 B8 predicate view。

固定 `CreateMask` 应在 VF 循环前创建一次，并在兼容的指令间复用。尾块由 kernel 补齐到完整寄存器组，回写 GM 时只拷贝有效区域，避免在 VF 热循环中反复更新 mask。

### G3：循环不变量统一外提

`Duplicate` 生成的移位量、bit mask 和其他常量寄存器在 VF 循环前创建一次。即使当前 `-O3` 能够得到相同指令序列，也应显式表达循环不变性，避免代码复用到其他上下文时引入重复指令。

### G4：VF 热循环保持无分支

VF 热循环只保留寄存器 API 和按组计算的地址偏移，不在循环中处理数据格式分支、标量尾块或动态 layout 切换。

- 循环变量和 `groupTimes` 使用 `uint16_t`，满足 SIMD VF 参数约束；
- 地址在使用点按 `base + group * stride` 表达；
- 数据格式和 layout 由独立入口或模板实例选择；
- VF 只接收 `__ubuf__` 基址和预计算组数，不按值传入复杂参数结构。

### G5：区分物理存储单位和逻辑元素单位

packed INT4/FP4 的输入索引按物理 byte 计算，输出和 shape 则按逻辑元素计算。两种视图应在 Tensor layout、tile 大小和指针解释处明确分开：进入 VF 后才将 packed UB 基址解释为相应的 packed 类型，避免用逻辑元素数直接推导物理拷贝量。

### G6：跨步 Store 要先分析 UB bank 映射

Ascend 950 的 UB 按 32 B DataBlock 在 16 个 bank 之间交织：

```text
bank = (byte_offset / 32) % 16
```

`DATA_BLOCK_COPY` 的 stride 以 32 B DataBlock 为单位。若多个目标 block 的 stride 是 16 的整数倍，它们会落入同一 bank。可在 UB 内部 layout 增加一个 padding DataBlock，使 `stride % 16 = 1`；从 UB 拷贝到 GM 时仍只拷贝标准紧凑布局，padding 不进入最终输出。

### G7：优先融合计算、布局转换和 scale

当计算结果随后需要 ND2NZ、量化或反量化时，优先在寄存器中完成 `Pack + Or` 或 raw-field 变换，直接写出目标布局，避免中间 ND 结果往返 UB/GM。可与下游等价合并的固定 scale 应写入接口契约，由下游已有 scale 统一补偿，不在当前 VF 中新增独立乘法。

### G8：用统一口径验证每个阶段

教程阶段只改变当前要分析的数据通路，host、shape、tile、核数、缓冲数和转换语义保持一致。每个阶段先通过 full、tail 和 guard 的逐 bit 校验，再比较性能。

CANNsim 统一运行 `trace` 用例，读取 core 0 业务 kernel 的 `avg_cycles`。指令数按 VF execution 数归一化，有效 B/cycle 只计算当前 tile 的业务字节。文档直接列出数据和瓶颈变化，不依赖报告截图。

相互依赖的 lane 和 Store 优化可能在中间阶段暂时增加整理指令。这类阶段应保留真实数据，说明回退来源和后续解决方式，不用不同 shape 或测量口径制造表面收益。
