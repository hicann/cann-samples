# NPU 访存带宽测试

用一组最小化的 kernel 测量昇腾 NPU 的 GM↔UB 访存带宽。三个可执行分别隔离出**纯读**、**读写拷贝**、**读+计算+写**三条数据流，通过成对扫描 UB tile 大小与 buffer 数，观察带宽随搬运粒度和缓冲深度的变化。

kernel 自身不计时，带宽由 `msprof` 采集 `Task Duration` 后换算。

| 可执行 | 数据流 | 流水阶段 | 计量 | 带宽 |
| --- | --- | --- | --- | --- |
| `bw_read` | 只搬入，随即回收 buffer，不计算不搬出 | CopyIn → Drain | 读 | `DATA_BYTES / 时间` |
| `bw_rw` | `y = x` 纯拷贝，无矢量计算 | CopyIn → CopyOut | 读+写 | `2 × DATA_BYTES / 时间` |
| `bw_rcw` | `y = \|x\|`，搬运与计算重叠 | CopyIn → Abs → CopyOut | 读+写 | `2 × DATA_BYTES / 时间` |

## 实现原理

### 数据流

```
         MTE2                        MTE3
 GM ──────────────> UB tile ──[Abs]──────────────> GM
     DataCopyPad              DataCopyPad

 bw_read : 只走到 UB，DeQue 后立即 FreeTensor 丢弃，不触发 MTE3
 bw_rw   : TQueBind 把 VECIN/VECOUT 绑成同一条队列，同块 buffer 搬入后直接搬出，
           省掉 UB→UB 拷贝（aiv_vec_time 应接近 0，可用来验证绑定确实生效）
 bw_rcw  : inQue_ / outQue_ 两条独立队列，中间插入 Abs
```

三者都用 `TPipe + TQue` 多 buffer 流水：`bufNum` 份 buffer 让相邻 tile 的搬运（与计算）首尾重叠，`bufNum = 2` 即 DoubleBuffer。

> **`queBytes` 是每个 buffer 的字节数，不是整条队列的。** 它对应 `pipe.InitBuffer(que, bufNum, len)` 的第三个参数 `len`，单条队列实际占用 `bufNum × queBytes`。`bw_rcw` 有两条队列，UB 总占用还要再乘 2 —— 这直接决定了它能跑哪些组合，见[参数说明](#参数说明)。

### 多核与 tiling

```
elems      = DATA_BYTES / sizeof(float)
blockFactor = elems / AIV核数          # 各核均分，要求整除
loops       = ceil(blockFactor / ubFactor)
ubFactor    = queBytes / sizeof(float)  # 每个 tile 的元素数
tailCount   = blockFactor - (loops-1) × ubFactor   # 核内最后一轮的尾块
```

每个核的 GM 偏移为 `blockFactor × GetBlockIdx()`，核内再按 tile 循环，最后一轮用 `tailCount` 收尾。

### 数据构造

host 端用固定种子 `mt19937(12345)`、`uniform_real_distribution(-1.0f, 1.0f)` 填充，因此**多次运行的数据完全一致、可复现**。用随机数而非全零，是为了避免全零内存被压缩后读出虚高带宽。整轮扫描共用同一份 host 数据，但**每组组合都重新 `aclrtMalloc` 并刷入 device**，避免上一轮的 cache 与数据残留影响测量。

## 目录结构

```
mem_bandwidth/
├── CMakeLists.txt          # 每个 src/*.asc 编一个独立可执行，新增 .asc 自动纳入
├── README.md
└── src/
    ├── bw_common.h         # CHECK_ACL 宏、DATA_BYTES、两张成对扫描表
    ├── bw_read.asc         # 纯读
    ├── bw_rw.asc           # 读写拷贝 y = x
    └── bw_rcw.asc          # 读 + abs + 写 y = |x|
```

## 环境要求

**架构**：仅支持 `NPU_ARCH=dav-3510`（Ascend 950）。仓库只接受 `dav-3510` 与 `dav-2201` 两个取值，其它值在配置阶段就以 `Unsupported NPU_ARCH` 报错中止。用 `dav-2201` 配置时本样例会被**静默跳过**（配置日志里有一行 `Skip sample ...`，但不报错），此时 `bw_read` / `bw_rw` / `bw_rcw` / `mem_bandwidth` 四个 target 都不存在，编译会以 `No rule to make target` 失败。

**内存**：`DATA_BYTES` 默认为 `56 × 64 MiB = 3 758 096 384 B ≈ 3.5 GiB`（939 524 096 个 float）。运行前请确认：

| 可执行 | device 内存 | host 内存 |
| --- | --- | --- |
| `bw_read` | ≈ 3.5 GiB（x） | ≈ 3.5 GiB（hData） |
| `bw_rw` / `bw_rcw` | ≈ 7 GiB（x + y 同时存在） | ≈ 7 GiB（hData + yHost） |

内存不足时会在 `aclrtMalloc` 处经 `CHECK_ACL` 打印 `ACL error: <码> at <文件>:<行>` 并以返回值 1 退出。要缩小规模，改 `src/bw_common.h` 的 `DATA_BYTES` 即可，但**必须保证 `DATA_BYTES / 4` 能被 AIV 核数整除**，否则程序会直接报错退出（默认值的 `56` 正是为整除 56 核而选）。

**CANN 环境变量**：`source ${install_path}/ascend-toolkit/set_env.sh`（路径按实际安装目录替换）。

## 编译与运行

在仓库根目录执行：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target mem_bandwidth   # 三个目标一起编，也可单独 --target bw_read

cd build/Samples/1_Features/hardware_features/mem_bandwidth
./bw_read              # 无参：按下标成对扫默认列表
./bw_read 16384 4      # 带参：只跑 16KB tile、buffer 数 4 这一组
./bw_rcw 16384 4
```

构建产物即上面 `cd` 的目录。样例也注册了安装规则，`cmake --install build` 后落在 `<prefix>/1_Features/hardware_features/mem_bandwidth/`。

## 参数说明

命令行为 `<exe> [queBytes bufNum]`，两个参数**要么都不给，要么同时给出**：

| 形式 | 行为 |
| --- | --- |
| 无参数 | 按下标逐对扫描 `DEFAULT_QUE_BYTES_LIST` 与 `DEFAULT_BUFFER_NUM_LIST` |
| `queBytes bufNum` | 只跑这一组 |
| 只给一个 / 超过两个 | 打印用法并以返回值 1 退出 |

### 默认扫描列表

两张表按下标一一配对，**不是笛卡尔积**，共 29 组：

| buffer 数 | tile 大小 | 组数 |
| --- | --- | --- |
| 2 | 1KB / 2KB / 4KB / 8KB / 16KB / 32KB / 64KB | 7 |
| 3 | 1KB ~ 32KB | 6 |
| 4 | 1KB ~ 32KB | 6 |
| 6 | 1KB ~ 16KB | 5 |
| 8 | 1KB ~ 16KB | 5 |

64KB tile 只在 `bufNum = 2` 下出现一次。两张表**必须保持等长**：循环次数只按 `DEFAULT_QUE_BYTES_LIST` 的长度计算，代码里没有长度校验，只改一张表会造成越界读。

### 约束

| 约束 | 说明 |
| --- | --- |
| `queBytes` 为 32B 正整数倍 | 否则跳过该组并置错误码 |
| `bufNum ≥ 1` | 否则跳过该组并置错误码 |
| `queBytes × bufNum × 队列数 ≤ UB 容量` | UB 容量由 `GetCoreMemSize` 运行期查得（dav-3510 为 248 KB）。**队列数：`bw_read` 与 `bw_rw` 为 1，`bw_rcw` 为 2** |

UB 预算这条对 `bw_rcw` 有实际影响：因为它开两条队列，默认列表里 `64KB×2`、`32KB×4`、`16KB×8` 三组各需 256 KB，超出 248 KB，**`./bw_rcw` 无参运行会自动跳过这 3 组、实际执行 26 组**。这属于硬件容量所限而非运行失败，不会置错误码；但若在命令行显式指定这类超预算组合，则会因该组跑不成而返回 1。

## 预期输出与指标含义

### stdout

> 以下为**格式示例**，非实测数据。带宽数值需自行采集，见下文。

每跑完一组打印一行。`bw_read`：

```
queBytes=1024 bufNum=2 读取完成（3758096384 bytes）
queBytes=2048 bufNum=2 读取完成（3758096384 bytes）
...
```

`bw_rw` / `bw_rcw` 额外带精度信息，`bw_rcw` 还输出最大绝对误差：

```
queBytes=1024 bufNum=2 精度通过率=100%            # bw_rw
queBytes=1024 bufNum=2 精度通过率=100% maxErr=0   # bw_rcw
```

无参运行时 `bw_read` / `bw_rw` 的 stdout 是 29 行；`bw_rcw` 的 stdout 只有 26 行，另外 3 组的跳过提示走 **stderr**（参数非法、超 UB 预算等提示一律在 stderr）。

两者精度判据不同：`bw_rw` 是纯拷贝，用严格相等 `!=` 比对，任何一 bit 差异都算错；`bw_rcw` 用绝对误差容差 `diff > 1e-6f`。

> **通过率的显示精度不足以判定成败。** `passRate` 用默认 `ostream` 精度输出，9.4 亿个元素里错几个也照样显示 `100%`。判断是否真的全对要看**进程返回码**，或 `bw_rcw` 的 `maxErr`。

### 返回码

返回 0 表示所有实际执行的组合都完成且精度通过。以下情形返回 1：参数用法错误、参数非法被跳过、显式指定的组合超 UB 预算、精度校验失败、任一 ACL 调用失败、元素数不能被核数整除。其中精度失败不中断扫描（跑完全部组合才返回），ACL 失败则立即中止。

### 采集带宽

kernel 不打印耗时，需用 `msprof`：

```bash
msprof --application="./bw_read 16384 2" --output=./prof
```

**建议逐组带参数采集。** 无参运行会连续发起 29 次 kernel launch，profiling 结果里是 29 条同名 task 记录，且不携带 `queBytes` / `bufNum`，无法把某条 `Task Duration` 对回具体组合。

从结果中取 `Task Duration` 换算：

- `bw_read`：`DATA_BYTES / Task Duration`
- `bw_rw` / `bw_rcw`：`2 × DATA_BYTES / Task Duration`

辅助指标（<!-- TODO -->字段名取自源码注释，未经 msprof 实际输出核对）：

| 指标 | 用途 |
| --- | --- |
| `aiv_mte2_time` | 搬入耗时，三个可执行都适用 |
| `aiv_mte3_time` | 搬出耗时，只有 `bw_rw` / `bw_rcw` 有 |
| `aiv_vec_time` | `bw_rw` 应接近 0，可据此确认 `TQueBind` 生效、没有多余的 UB→UB 拷贝 |

## 常见问题

- **`用法: ./bw_read [queBytes bufNum]...`** —— 只给了一个参数或超过两个。两个参数是一组，要么都不给。
- **`数据量 939524096 floats 不能被核数 N 整除`** —— 默认 `DATA_BYTES` 按 56 核选定。换到 AIV 核数不同的芯片需调整 `src/bw_common.h` 的 `DATA_BYTES`，使 `DATA_BYTES / 4` 能被核数整除。
- **`queBytes (N) 必须是 32B 的正整数倍，已跳过`** / **`bufNum (N) 必须 ≥ 1，已跳过`** —— 参数非法，该组跳过、其余照常执行，进程最终返回 1。
- **`... 超出 UB 容量 Y B，已跳过`** —— 该组合放不进 UB，`bw_rcw` 因双队列占用翻倍最容易触发。无参扫描时属预期行为，不影响返回码。
- **`ACL error: <码> at <文件>:<行>`** —— ACL 调用失败，最常见的是 `aclrtMalloc` 内存不足（见[环境要求](#环境要求)）。程序立即返回 1，已申请的 device 内存与 stream 不会释放。
- **编译报 `No rule to make target 'mem_bandwidth'`** —— 配置时 `NPU_ARCH` 不是 `dav-3510`，样例被跳过、target 未生成。

<!-- TODO: 以下内容需在真机上补充
     1. 各组合的实测带宽数值与带宽-粒度曲线
     2. 队列模板深度（三个 kernel 均硬编码为 1）与 InitBuffer 的 bufNum 在实际流水重叠深度上的关系，
        bufNum 增大到 6/8 时能否真的加深重叠，需 profiling 验证
     3. 默认 29 组扫描跑完的实际耗时
-->
