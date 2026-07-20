# Flash Attention Lite 性能优化实践

## 通用介绍

本样例以简化版 Flash Attention 前向为基础, 按版本展示 Ascend 950 上的实现和后续优化过程:

$$
O=\operatorname{softmax}(QK^T\cdot scale)V
$$

样例仅支持 `NPU_ARCH=dav-3510`. Q、K、V、O 的逻辑 shape 为 `(B,N,S,D)`, 当前固定 `N=1`、`D=128`, 数据类型为 BF16. 连续 GM 存储和 `.bin` 文件可等价看作 `(B,S,128)`.

各版本源码位于 `src/vN`, 每个版本均为独立可执行程序. story 层 CMake 自动识别版本目录并生成 `falite_vN` 构建目标.

### 目录结构

```text
flash_attn_lite_story/
├── CMakeLists.txt
├── README.md
├── scripts/                    # 各版本共用的数据生成与精度校验脚本
└── src/
    ├── v0/                     # falite_v0
    └── v1/                     # falite_v1
```

### 编译

在 cann-samples 仓库根目录执行:

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510

# 构建本目录全部版本, 两个目标作用相同.
cmake --build build --target falite
cmake --build build --target flash_attn_lite_story

# 只构建单个版本.
cmake --build build --target falite_v0
cmake --build build --target falite_v1
```

构建产物位于:

```text
build/Samples/2_Performance/flash_attn_lite_story/falite_v0
build/Samples/2_Performance/flash_attn_lite_story/falite_v1
```

构建目录中还会包含各版本共用的 Python 脚本. 安装必选依赖:

```bash
python3 -m pip install -r Samples/2_Performance/flash_attn_lite_story/requirements.txt
```

`numpy` 和 `ml_dtypes` 用于数据生成与精度校验; CPU 版 `torch` 为可选依赖, 可加速 Golden 计算.

## 版本实现

### v0: 单缓冲 baseline

v0 是 P 矩阵不落 GM 的单缓冲基础实现. 一个 task 处理 128 行 Q, 由 1 个 AIC 和 2 个 AIV 协作完成, 两个 AIV 各处理 64 行.

当前约束:

- `B > 0`, `S > 0`, 且 `S` 是 128 的整数倍.
- 固定 `N=1`、`D=128`、`Br=Bc=128`.
- 不支持尾块、causal mask、attention mask、dropout、变长序列和反向计算.

一次分块计算包括四个阶段:

1. C1: AIC 计算 `K×Q^T`, 直接得到分块 `S^T`.
2. V1: 两个 AIV 各处理一半行, 完成 online softmax、FP32 到 BF16 转换和 DN 到 NZ 转换.
3. C2: AIV 将 P 从 UB 直接写入共享 L1, AIC 随后计算 `P×V`; P 不写回 GM.
4. V2: AIV 在线更新输出累加值, 最后归一化并将 BF16 O 写回 GM.

Kernel 使用显式 `LocalTensor(pos, addr, elements)` 规划 L1、L0A、L0B、L0C 和 UB, 并通过 CrossCore 事件同步 1C2V 的四个阶段.

运行示例:

```bash
./build/Samples/2_Performance/flash_attn_lite_story/falite_v0
./build/Samples/2_Performance/flash_attn_lite_story/falite_v0 --size 1 1024
```

可选参数:

- `--size B S`: 设置 Batch 和序列长度.
- `--core-num n`: 限制本次 launch 使用的 AIC 数量, 主要用于仿真; 真机通常不指定.
- `--dry-run`: 仍执行 kernel、D2H 和 `npuout_o.bin` 落盘, 仅跳过 Golden 计算与精度比对.

程序在可执行文件同目录重建 `data/`, 生成 Q/K/V, 执行 NPU kernel 并写出 `npuout_o.bin`. 非 `--dry-run` 模式还会以 FP32 计算 Golden, 转为 BF16 后按以下条件逐元素比较:

```text
abs(npu - golden) <= 2^-6 + 2^-6 * abs(golden)
```

全部元素通过时程序返回 0.

### v1

待实现.
