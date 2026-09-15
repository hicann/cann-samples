# BlockAttnRes Recipe 示例

## Prepare

输入 `block_res[T,N,D]`、`valid_blocks[1]` 和 `pseudo_query[S,D]`，输出：

- `numerator[S,T,D]`：softmax 加权分子；
- `logit_max[S,T]`：历史 logit 最大值；
- `exp_sum[S,T]`：max-shift 后的指数和。

`valid_blocks=0` 时输出在线 Softmax 空状态；大于 N 时按 N 处理。

## Update

输入 `partial_block[T,D]`、BF16 `delta[T,D]`、一个 slot 的 `pseudo_query[D]`，以及同一 slot 的
`numerator/logit_max/exp_sum`。Kernel 原地写回 FP32 `partial_block`，并输出 BF16 `h[T,D]`。

## E2E

E2E 入口先启动 Prepare，再把 `--slot` 对应的状态地址直接传给 Update。两个 Kernel 位于同一个 ACL stream，
因此无需 Host 中间同步或回传。

## 支持范围

| 项目 | 范围 |
| --- | --- |
| 架构 | Ascend 950PR/950DT，`dav-3510` |
| `T` | 正整数，受设备内存和 uint32 tiling 范围限制 |
| `N` | `1~64` |
| `S` | 正整数 |
| `D` | `1~8192`，Update 要求一行 D 可双缓冲放入 UB |
| `valid_blocks` | 运行时值，Prepare 按 `min(valid_blocks,N)` 处理 |
| `eps` | 有限正数 |

Prepare 支持在线 Softmax 空状态，即 `valid_blocks=0`。Update 与 E2E 要求历史状态非空。

## 构建

在 ops-samples 仓库根目录执行：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target block_attn_res_recipes --parallel
cmake --install build --prefix ./build_out
cd build_out/2_Performance/block_attn_res_story/block_attn_res_recipes
```

## 运行

统一参数格式为：

```text
<program> T N S D [--valid-blocks n] [--slot n]
          [--template auto|vector|mix] [--eps value]
          [--warmup n] [--repeat n] [--compare isclose|stat_rel_err]
```

示例：

```bash
# Vector 路径与 D tail
bash run.sh block_attn_res_prepare 37 8 3 129 --template vector --valid-blocks 6

# 自动选择 Mixed 路径
bash run.sh block_attn_res_prepare 128 8 32 512 --template auto --repeat 10

# Update 使用 slot 1 的历史状态
bash run.sh block_attn_res_update 128 8 32 512 --slot 1 --repeat 10

# 两阶段同 stream 串联
bash run.sh block_attn_res_e2e 128 8 32 512 --slot 1 --template auto --repeat 10
```

使用 `SAMPLE_DEVICE_ID` 指定设备，默认设备 0。性能采集建议配合 `msprof` 查看 Kernel Task Duration；
`--warmup` 和 `--repeat` 用于产生稳定的重复 Kernel 任务。

结果校验默认使用 `isclose`。可通过 `--compare stat_rel_err` 选择平均及最大相对误差判据；
该判据对求和相消产生的近零参考值更敏感。`run.sh` 与可执行程序均接受 `--compare` 参数。
