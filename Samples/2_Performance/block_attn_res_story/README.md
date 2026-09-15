# BlockAttnRes 性能优化实践

## 目录结构

```
block_attn_res_story/
├── CMakeLists.txt
├── README.md
└── block_attn_res_recipes/                     # 算子实现与示例代码
    ├── CMakeLists.txt
    ├── README.md
    ├── include/
    │   ├── kernel/                            # Prepare Vector/Mixed、Update 的设备端流程
    │   ├── block/                             # Mixed 的调度、MMAD 与 epilogue
    │   ├── vf/                                # Prepare/Update 的向量计算
    │   └── tiling/                            # Host tiling 与 Host/Device 共享数据
    ├── scripts/                               # 数据生成、运行与验证脚本
    └── examples/                              # 算子示例目录
        └── block_attn_res/                    # Prepare、Update 与 E2E 示例
```

## 概述

本目录提供 BlockAttnRes 算子在 Ascend 950（`dav-3510`）上的性能优化实践。样例包含历史状态计算、残差更新与两阶段串联，支持独立运行和端到端结果校验。

- **多模板实现**：Prepare 提供 Vector 与 Cube+Vector Mixed 模板，支持自动选择和手动指定
- **流水优化**：Update 使用 D 全载模板并沿 T 维分块，包含 single-tile 和采用 UB 双缓冲的 multi-tile 两条路径
- **端到端示例**：提供 Prepare 与 Update 在同一 stream 内串联的实现，中间状态采用 FP32，Update 输出采用 BF16

## 算子示例

- [block_attn_res](./block_attn_res_recipes/examples/block_attn_res/README.md)：BlockAttnRes Prepare、Update 与两阶段串联优化实践
