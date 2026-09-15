# BlockAttnRes 算子样例

## 概述

本目录提供 BlockAttnRes 的 Prepare、Update 及两阶段串联样例，包含算子代码、动态 Tiling、运行脚本与结果校验，便于编译运行和性能测试。

## 目录结构

```text
block_attn_res_recipes/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── kernel/                     # Prepare Vector/Mixed、Update 的设备端流程
│   ├── block/                      # Mixed 的调度、MMAD 与 epilogue
│   ├── vf/                         # Prepare/Update 的向量计算
│   ├── tiling/                     # Prepare/Update 动态 Tiling 与共享数据
│   ├── sample_common.h             # Host 参数解析、ACL 与文件工具
│   └── sample_process.h            # Python 脚本启动
├── examples/
│   └── block_attn_res/              # Prepare、Update 与 E2E 入口
└── scripts/
    ├── gen_data.py                  # 输入数据与参考结果生成
    ├── verify_result.py             # 结果校验
    ├── test_verify_result.py        # 校验脚本单元测试
    └── run.sh                      # 样例运行脚本
```

## 样例列表

| 目标 | 张量类型 | 说明 |
| --- | --- | --- |
| `block_attn_res_prepare` | FP32：`block_res`、`pseudo_query`、输出；UINT64：`valid_blocks` | 计算历史状态，支持 Vector/Mixed 模板 |
| `block_attn_res_update` | FP32：`partial_block`、`pseudo_query`、历史状态；BF16：`delta`、`h` | 更新 partial_block，合并历史状态并输出 h |
| `block_attn_res_e2e` | 同 Prepare、Update | 在同一 ACL stream 内串联 Prepare 与指定 slot 的 Update |

三个目标使用相同的 `T N S D` 参数。Update 独立样例读取参考历史状态；E2E 样例使用 Prepare 的设备输出。

## 使用方式

- 输入输出说明见[样例 README](examples/block_attn_res/README.md)。
- 支持范围、构建步骤、运行参数和性能采集说明见[样例 README](examples/block_attn_res/README.md)。
- 运行样例时自动生成数据并校验结果，校验脚本输出精度统计及通过或失败状态。
