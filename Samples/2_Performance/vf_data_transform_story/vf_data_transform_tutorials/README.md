# VF 数据变换 Tutorials

本目录按功能点组织 7 篇 SIMD VF 数据变换教程，共 25 个可执行阶段。每篇教程从当前优化路径的第一个技巧开始，逐步改变寄存器 lane 排布、Load/Store 分布模式或 UB 布局，并用相同输入和 CANNsim trace 对比修改前后的流水。

## 代码组织

```text
vf_data_transform_tutorials/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── sample_acl_runtime.h
│   └── sample_data_io.h
├── scripts/
│   ├── gen_data.py
│   ├── verify_result.py
│   ├── profile_tutorial.sh
│   └── summarize_msopprof.py
└── <tutorial>/
    ├── CMakeLists.txt
    ├── README.md
    ├── 0_<baseline>/
    │   ├── CMakeLists.txt
    │   └── <baseline>.asc
    ├── 1_<optimization>/
    │   ├── CMakeLists.txt
    │   └── <optimization>.asc
    └── N_<optimization>/
        ├── CMakeLists.txt
        └── <optimization>.asc
```

每个阶段的 `.asc` 都是完整实现，包含当前阶段的 VF、kernel、搬运流程、shape 约束和 host 调用，不包含其他阶段源码，也不通过功能级头文件选择数据通路。每个 CMake 目标都可独立生成并运行对应可执行文件。tutorials 上层公共头文件只提供以下 host 辅助能力：

- `sample_acl_runtime.h`：ACL 初始化、Stream、设备内存和错误检查。
- `sample_data_io.h`：调用数据脚本、读写二进制以及触发结果校验。

数据生成和校验脚本由 7 个 tutorial 共用；VF、kernel、数据搬运和 shape 约束均完整保留在各阶段源码中，公共头文件不承载任何转换步骤。

## 阶段与目标

| Tutorial | 基线目标 | 中间目标 | 最终目标 |
| --- | --- | --- | --- |
| `half_to_int8` | `vf_data_transform_half_to_int8_0_dintlv_load` | `1_complementary_cast` / `2_or_merge` | `vf_data_transform_half_to_int8` |
| `fp32_to_fp8` | `vf_data_transform_fp32_to_fp8_0_dintlv_b32_load` | `1_complementary_cast` / `2_pack_b16_store` | `vf_data_transform_fp32_to_fp8` |
| `int8_to_half` | `vf_data_transform_int8_to_half_0_compact_b8_load` | `1_complementary_cast` / `2_intlv_b16_store` | `vf_data_transform_int8_to_half` |
| `int4_to_bf16` | `vf_data_transform_int4_to_bf16_0_unpack4_load` | `1_single_cast` / `2_separate_views` | `vf_data_transform_int4_to_bf16` |
| `fp4_to_fp8` | `vf_data_transform_fp4_to_fp8_0_raw_field_mapping` | `1_unpack_packed_byte` / `2_shift_select_and` / `3_hoist_constants` | `vf_data_transform_fp4_to_fp8` |
| `bf16_nd2nz` | `vf_data_transform_bf16_nd2nz_0_data_block_copy` | — | `vf_data_transform_bf16_nd2nz` |
| `bf16_rope_nd2nz` | `vf_data_transform_bf16_rope_nd2nz_0_pack_or_fusion` | — | `vf_data_transform_bf16_rope_nd2nz` |

对应的聚合构建目标为 `vf_data_transform_<tutorial>_tutorial`。全部阶段由 `vf_data_transform_tutorials` 和上层 `vf_data_transform_story` 目标统一聚合。

## 统一用例

每个程序都支持以下 `--case`：

| Case | 用途 |
| --- | --- |
| `full` | 完整 tile 的多核正确性验证 |
| `tail` | 非整 tile、非整寄存器组和 guard 验证 |
| `perf` | 多 tile 的端到端性能采集 |
| `trace` | 每核一个 tile 的 CANNsim VF 流水分析 |
| `all` | 顺序运行 `full`、`tail` 和 `perf`；命令行缺省值 |

Cast 和 raw-field tutorial 只复制有效输出，补齐区不会写回 GM。ND2NZ tutorial 按标准 NZ `[N/16, M_align, 16]` 生成 golden，并额外检查 `M` 尾部对齐区。

## 阅读建议

窄化和扩展转换先阅读 `half_to_int8` 与 `int8_to_half`，理解寄存器宽度变化对 lane 的影响；随后阅读 `fp32_to_fp8`、`int4_to_bf16` 和 `fp4_to_fp8`，比较 pack、unpack 与 raw-field 映射。最后两篇从跨步 Store 入手，说明 bank conflict 以及计算和布局转换的融合方式。

统一构建、运行和性能采集命令见 [专题总览](../README.md)。
