# FALite v00：一个 Mix 核组顺序完成全部任务

## 本版内容

v00 固定发射一个 Mix 核组，由 1 个 AIC 和 2 个 AIV 顺序完成全部 Query 分块任务。两次矩阵乘之间的 S、P、DeltaO 都通过 GM（全局内存）交接，片上工作区只有一套。

本文按 Host、AIC 和 AIV 实现解释任务归属、数据布局与同步。共同数学推导见[总 README 的算法基础](../../README.md#算法基础)，术语和硬件空间见[Ascend 950 上的 FALite 实现](../../README.md#ascend-950-上的-falite-实现)。

输入输出为 BF16 `[B,N,S,128]`，`Br=Bc=D=128`；`B/N/S` 均为正整数，`S` 无需按 128 对齐。入口固定调用同起点、等长度的方阵 causal 实例，功能保证范围为 `B*N*S<=131072`；该范围不是 Host 的主动拒绝条件。完整接口约束和未支持能力见[样例定位](../../README.md#falite-样例定位)。

## task 归属与两路 AIV 分工

一个 task 完成某个 `(b,n)` 的一个 Query tile，最多输出 128 行。一个 item 是该 task 与一个 K/V tile 的组合。Host 和 Kernel 使用以下编号：

```text
tr = ceil(S / 128)
numTasks = B * N * tr
batchHeadIdx = taskId / tr
qTileIdx = taskId % tr
batchIdx = batchHeadIdx / N
headIdx = batchHeadIdx % N
```

Host 不保存分核参数，入口固定 `<<<1,0,stream>>>`。公共接口虽然接收 `--core-num`，本版会忽略它。唯一的 AIC 和同组两路 AIV 都按 `taskId=0,1,...,numTasks-1` 遍历。

两路 AIV 用 `subAivIdx=GetSubBlockIdx()` 区分：AIV0 负责 Query 行 0～63，AIV1 负责 64～127。它们分别维护 64 行的 `m/l/alpha` 和 `OAcc`，无需交换或合并这些状态。

| 阶段 | 核心 | 本版职责 |
| --- | --- | --- |
| C1 | AIC | `K_j × Q_i^T → S^T`，BF16 输入、FP32 累加 |
| V1 | 两路 AIV | 计算分块 Softmax，更新 `m/l/alpha`，生成尚未除以完整分母的 BF16 P |
| C2 | AIC | `P_j × V_j → DeltaO`，BF16 输入、FP32 累加 |
| V2 | 两路 AIV | `OAcc = alpha × OAcc + DeltaO`，保持 FP32；task 末尾才除以 l、转 BF16 并写回 |

## GM 与片上数据布局

Q/K/V/O 在 GM 中紧凑排列，单个 token 连续保存 128 个 BF16 通道。某个 tile 的首元素偏移为 `(batchHeadIdx*S + tileIdx*128)*128`；此处偏移以元素为单位，TilingData 中 `Addr` 字段则以字节为单位。

S 和 P 的 GM 物理形状均为 `[numTasks,128,128]`，每个 task 保存一个 `S^T/P^T[Key,Query]`，不是整个 `S×S` 矩阵。其元素偏移为 `taskId*128*128 + keyLocal*128 + queryLocal`。同 task 的各 item 重复使用这一块地址。

两路 AIV 按行间隔读 S、写 P：每个 Key 行只取属于自己的 64 个 Query 列。DeltaO 的物理形状为 `[numTasks,Query,128]`，每路取连续的 64 个 Query 行。

| Host 申请的 GM workspace | 每 task 大小 | 使用范围 |
| --- | ---: | --- |
| FP32 S | 64 KiB | C1 写入，V1 读入 |
| BF16 P | 32 KiB | V1 写入，C2 读入 |
| FP32 DeltaO | 64 KiB | C2 写入，V2 读入 |
| 合计 | 160 KiB | 按 numTasks 分配，不按 item 数累加 |

Host 按全部 task 分配 workspace，并在释放前调用 `aclrtSynchronizeStream`。

### AIC 缓冲与生命周期

`Q/K/V/P` 的 L1 数据为 BF16 NZ，即 Cube 读取所需的分块排列。以下均为单槽，不存在 slot 翻转：

| 物理空间 | 起始地址 | 大小 | 生命周期 |
| --- | ---: | ---: | --- |
| P L1 | 0 KiB | 32 KiB | 每 item 生成一次，C2 搬到 L0A 后不再读取 |
| Q L1 | 32 KiB | 32 KiB | 每 task 搬一次，供全部 C1 复用 |
| K L1 | 64 KiB | 32 KiB | C1 搬入，MTE1 读完后可复用 |
| V L1 | 96 KiB | 32 KiB | C2 搬入，MTE1 读完后可复用 |
| L0A | 本空间 0 | 32 KiB | C1 装 K，C2 装 P，Mmad 读完后归还 |
| L0B | 本空间 0 | 32 KiB | C1 装 Q，C2 装 V，Mmad 读完后归还 |
| L0C | 本空间 0 | 64 KiB | C1/C2 交替写入，Fixpipe 读完后归还 |

L1 合计 128 KiB；L0A/B/C 属于不同物理空间，不能把三个起始地址 0 当成同一块内存。

### 单路 AIV 缓冲与生命周期

| UB 区域 | 起始字节地址 | 大小 | 用途 |
| --- | ---: | ---: | --- |
| S | 0 | 32 KiB | `[128,64]` FP32 分数；V1 原地改写为 FP32 未归一化权重 |
| DeltaO | 32768 | 32 KiB | `[64,128]` FP32，当轮 V2 消费 |
| OAcc | 65536 | 32 KiB | `[64,128]` FP32，整个 task 累计 |
| P/Output | 98304 | 16 KiB | V1 的 BF16 P；task 末尾复用为 BF16 输出 |
| m / l / alpha | 114688 / 114944 / 115200 | 各 256 B | 每个 Query 行一个 FP32 值 |

每路 UB 共占 115456 B（112.75 KiB），两路 AIV 各有一套。S、DeltaO、P 和 alpha 按 item 复用；m、l、OAcc 按 task 复用。

![v00 SRAM 分配示意图](../../images/sram/falite_v00_sram.png)

## AIC 核内流水与同步

C1 按 `K GM→L1→L0A` 和 `Q L1→L0B` 准备输入，再执行 Mmad 和 Fixpipe。Q 的 MTE2 写入发生在 task 开始，MTE1 在 `j==0` 取得 Q 槽，到 `j+1==kvTileCount` 时归还；这里的末次由 causal 裁剪后的实际 item 数判断。

C2 先把 P 从 GM 搬入 L1，再将 P 转置到 L0A；随后发射 V 的 GM→L1 和 L1→L0B 搬运。两次 Mmad 都初始化 L0C，而不是在 L0C 内跨 item 累加；跨 item 的输出累计由 AIV 的 OAcc 完成。

| Mutex ID | 资源 | Pipe 交接 |
| ---: | --- | --- |
| 0 | Q L1 | MTE2 写 → MTE1 读 → 下一 task 的 MTE2 |
| 1 | K L1 | MTE2 写 → MTE1 读 → 下一 C1 的 MTE2 |
| 2 | V L1 | MTE2 写 → MTE1 读 → 下一 C2 的 MTE2 |
| 3 | P L1 | MTE2 写 → MTE1 读 → 下一 C2 的 MTE2 |
| 4 | L0A/L0B 共用所有权 | MTE1 写 → Mmad 读 → 下一阶段 MTE1 |
| 5 | L0C | Mmad 写 → Fixpipe 读 → 下一阶段 Mmad |

每个 Pipe 用 `Mutex::Lock/Unlock` 取得和归还对应资源。L0A/B 在 Mmad 之后即可归还，L0C 则必须等 Fixpipe 读取；不能把二者的释放合并为一个笼统的“矩阵乘结束”。本版没有手写 HardEvent 初始化或排空循环；核内交接由上述 Mutex 表达，Kernel 入口先执行 `InitSocState()`。

Scalar 按函数顺序发射指令，不等于所有 Pipe 都已完成。C2 中“先发射 P、再发射 V”不能单凭代码行次序推断两个 Pipe 的完成先后；数据完成与槽位复用仍以 Mutex、CrossCore 和同一 Pipe 内的指令顺序为准。

## AIV 核内流水与同步

以下只描述一路 AIV；另一路执行相同控制流。

V1 等 S_READY 后由 MTE2 把 GM 的半块 S 读入 UB，再由 Vector 处理。V2 同样在 O_READY 后将半块 DeltaO 从 GM 搬入 UB。`MUTEX_S_UB=1` 和 `MUTEX_DELTA_O_UB=2` 分别在 MTE2 写入与 Vector 读取之间交接。

V1 的 `OnlineColwiseSoftmaxVF` 将 FP32 权重写回 S，随后 `Cast` 转成普通二维 BF16 P。`MUTEX_P_UB=0` 的 Vector Lock/Unlock 包住 Cast，MTE3 用同一 Mutex 读取 P 并写入 GM。P 保留 `[Key,本路Query]` 排列，L1 的 NZ 转换由 AIC MTE2 完成。

Softmax 每次加载连续的 64 个 Query 列，一个 Vector 寄存器的 64 个 FP32 lane 各对应一个 Query 行；沿 Key 维循环即可分别维护 64 份最大值和分母。V2 用 `OnlineUpdateVF` 对每行广播 alpha，顺序更新 OAcc。m/l/OAcc 不设多槽，也不在两路 AIV 之间做归约。

task 结束时，`FusedDivCastVF` 将 `OAcc/l` 转为连续 BF16 输出，仍写进 P UB。Vector→MTE3 的 Mutex 交接保证结果生成后才能写回，下一次 Vector 取得该槽又要等 MTE3 读完。OAcc 的下一 task 清零排在本核归一化之后；无需等待 GM 写回才清零，因为 MTE3 读取的是 BF16 工作区，不是 OAcc。

## CV 核间同步

单槽的主要数据消费依赖如下；CV 指 AIC 与 AIV 的交接：

```text
C1 --S_READY--> V1 --P_READY--> C2 --O_READY--> V2
 ^                                               |
 +------------------ DONE -----------------------+
```

| flag 名称 | flag ID | Set 端 | Wait 端 | 数据含义 |
| --- | ---: | --- | --- | --- |
| S_READY | 0 | AIC FIX | AIV MTE2 | S 已写入 GM，可读入 UB |
| O_READY | 1 | AIC FIX | AIV MTE2 | DeltaO 已写入 GM，可读入 UB |
| DONE | 2 | AIV V | AIC MTE1 | 本 item 的 V2 已消费完成 |
| P_READY | 4 | AIV MTE3 | AIC MTE2 | 两路 P 已写入 GM |

表中的 V、FIX 等对应 `PIPE_V`、`PIPE_FIX`。真机 mode2 下，一次 AIC Set 通知同组两路 AIV；AIC 等待 P_READY 或 DONE 时，必须等两路 AIV 都发出对应信号。`SIM_COMPATIBLE=ON` 的仿真封装改用 mode4 分别处理两路，逻辑 flag 和阶段不变。

DONE 不是 task 输出完成信号。它在每次 V2 后发出，最后一次 DONE 仍早于归一化与 O 写回；其消费者是下一 item 的 AIC MTE1，跨 task 时也照常消费。AIC 的首个 item 跳过 DONE 等待，全部 task 结束后再消费最后一次 DONE。没有预置的 DONE，也不能在每个 task 开头都跳过它。

DONE 绑定 MTE1，不是 Scalar 全局屏障。下一轮 K 或下一 task Q 的 MTE2 可以先进入自己的队列，真正使用单槽结果的 MTE1 及其下游仍受 DONE 限制。因此“单槽顺序”描述主要阶段依赖，不表示所有搬运都完全没有重叠。

P 不单独设置“已读完”信号：C2 读 P 后才能产生 O_READY，随后 V2 发 DONE、下一 C1 发 S_READY，AIV 才生成下一份 P。这条依赖链保护了 P 的跨 item 复用；核内 Mutex 则另行保护各 Pipe 对本地工作区的读写。

## 调度伪代码

下面保留两侧独立的 task/item 循环。`with mutex(资源, Pipe)` 表示该 Pipe 的 Lock/Unlock；C1/C2 的核内资源交接按前面的表执行，不额外添加全 Pipe 等待。

```python
# AIC
first = True
for taskId in range(numTasks):
    i = taskId % tr
    kvTileCount = i + 1
    with mutex(Q_L1, MTE2):
        copy_q_to_l1(valid_rows(i))       # 尾行补零

    for j in range(kvTileCount):
        if not first:
            WaitAivToAic(MTE1, DONE)
        first = False
        # CubeStage1：j==0 取得 Q 的 MTE1 所有权，
        # j+1==kvTileCount 时归还；内部按 Mutex 交接 L0。
        CubeStage1(j)                    # Fixpipe -> S GM
        SetAicToAiv(FIX, S_READY)
        WaitAivToAic(MTE2, P_READY)
        CubeStage2(j)                    # P GM -> L0A；Fixpipe -> DeltaO GM
        SetAicToAiv(FIX, O_READY)

WaitAivToAic(MTE1, DONE)                  # 只消费末 item 的 DONE
```

```python
# AIV0 / AIV1，各处理自己的 64 个 Query 行
for taskId in range(numTasks):
    i = taskId % tr
    init(m=FLOAT_LOWEST, l=0, OAcc=0)
    for j in range(i + 1):
        WaitAicToAiv(MTE2, S_READY)
        with mutex(S_UB, MTE2):
            copy_s_half_from_gm()
        with mutex(S_UB, V):
            VectorStage1(j, i)           # 内含 P_UB 的 V Lock/Cast/Unlock
        with mutex(P_UB, MTE3):
            copy_p_half_to_gm()
        SetAivToAic(MTE3, P_READY)

        WaitAicToAiv(MTE2, O_READY)
        with mutex(DeltaO_UB, MTE2):
            copy_delta_o_half_from_gm()
        with mutex(DeltaO_UB, V):
            VectorStage2()
        SetAivToAic(V, DONE)

    outputRows = clamp(valid_rows(i) - subAivIdx * 64, 0, 64)
    if outputRows > 0:
        with mutex(P_UB, V):
            FusedDivCastVF(outputRows)
        with mutex(P_UB, MTE3):
            copy_output_to_gm(outputRows * 128)
```

## 首轮、末轮与尾块

每个 task 都重新初始化 m、l、OAcc。首个 V1 直接建立 m/l，并写 `alpha=1`；OAcc 初值为 0，所以首个 V2 仍调用普通更新函数。只有本核首个 item 不等 DONE，和“每个 task 的首个 V1”不是同一个条件。

causal 下只发射 `j=0...i`；对角 item 的 V1 在求最大值与指数和的两次扫描中都屏蔽 `keyLocal>queryLocal`，也就是逻辑 Query 行、Key 列矩阵的上三角。两路 AIV 分别使用 Query 局部起点 0、64，不能都从 0 计算 mask。

末尾不足 128 行时，`CopyGmToL1` 只读取紧凑 GM 的有效行，在 L1 补零；Cube 仍执行固定 128 大小的矩阵乘。V1 只扫描有效 Key，将其余 Key 对应的 P 置零。GM 中间缓冲仍按完整 tile 保存，最终输出才按有效 Query 行裁剪。

即使 AIV1 的 `outputRows=0`，也必须完成全部 V1/V2 和 P_READY/DONE，只跳过最终归一化与 GM 输出。否则同组 AIC 的聚合等待无法完成。本版逐 item 循环，没有两个 item 一组的奇偶尾组分支。

## 流水示意与证据入口

![v00 流水示意图](../../images/pipeline/falite_v00_pipeline.png)

示意图说明本版单核组遍历 task 与 GM 交接的关系。色块宽度和图中空隙不代表实际耗时；跨 Pipe 的精确完成顺序应结合源码同步判断。

![v00 上板 PipeTimeline](../../images/pipe_trace/falite_v00_pipe.png)

截图来自 `B=1,N=1,S=2048,D=128` 的完整 PipeTimeline trace，窗口为 `[368.8,408.8] μs`。它展示各 Pipe 的忙区和空隙，不能把某个空隙直接解释为 DONE 或 P_READY 的精确等待时长。

![v00 CANNSIM 流水截图](../../images/cannsim_trace/falite_v00_cannsim.png)

此图来自 `B=1,N=1,S=512,D=128` 的完整 CANNSIM trace，窗口为 `[18,32] μs`，用于查看单 Mix 的组件与同步泳道；它与上方真机截图不是同一 shape。

本版在 `B=1,N=1,S=131072` 下的 Task Duration 中位数为 `2574789.750000 μs`（固定 1 个 AIC）。完整采集环境、核数差异和 MFU 口径统一见[总 README 的统一性能结果](../../README.md#统一性能结果)。本文不按示意图的宽度比较版本收益。

## 代码阅读入口与运行

| 阅读顺序 | 文件与函数 | 主要看什么 |
| --- | --- | --- |
| 1 | [Host](host/flash_attn_lite_host.cpp)：`ComputeFlashAttnLiteTilingData`、`FlashAttnLiteNPU` | task 数、固定单核入口参数、地址规划、workspace 申请与同步释放 |
| 2 | [TilingData](flash_attn_lite_common.h) | `Addr`/`Elems` 单位，Host 与 Kernel 共用布局 |
| 3 | [Kernel 入口](kernel/flash_attn_lite_kernel.asc) | `InitSocState`、1 AIC + 2 AIV、causal 模板实例 |
| 4 | [AIC](kernel/falite_kernel_aic.h)：`KernelProcessForAIC`、`CubeStage1/2` | task/item 循环，Q 生命周期，GM→L1→L0，单槽复用 |
| 5 | [AIV](kernel/falite_kernel_aiv.h)：`KernelProcessForAIV`、`VectorStage1/2` | 两路切片，S 原地改写和 Cast，输出工作区复用 |
| 6 | [核间封装](kernel/falite_kernel_common.h) | flag ID、mode2/mode4 分支、`GetKvTileCount` 与有效行数 |

在 cann-samples 根目录执行以下命令即可构建并运行本版：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510 -DSIM_COMPATIBLE=OFF
cmake --build build --target falite_v00 -j
./build/Samples/2_Performance/flash_attn_lite_story/falite_v00 --core-num 32 --size 2 3 257
```

v00 会提示忽略 `--core-num 32`，实际仍为一个 Mix 核组。精度标准和更多运行选项见[总 README 的编译、运行与复现](../../README.md#编译运行与复现)。

## 与相邻版本的区别

v00 没有 task 分核。所有 task 由唯一的 Mix 核组顺序遍历，GM workspace 和片上单槽都是显式规划的。[v01](../v01/README.md) 保持四阶段和内存通路，只增加按固定步长分配 task。
