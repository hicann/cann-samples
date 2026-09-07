# FALite v05：L0 双槽与分开的搬运等待

## 本版内容

v05 保留两个 item 一组的 CV 调度，将 L0A/L0B/L0C 也改为双槽。C1 调整 Q/K 的装载顺序；C2 让 P 的 L1→L0A 和 V 的 GM→L1 分别等待 P 就绪，解除 v04 的“装完 P 再搬 V”依赖。

本版固定使用 BF16 `Q/K/V/O [B,N,S,128]`、`Br=Bc=D=128`，计算同起点、等长度的方阵 causal Attention；`B/N/S` 为正整数，`S` 无需对齐，功能保证范围为 `B*N*S<=131072`。完整能力边界见[总文档的样例定位](../../README.md#falite-样例定位)，共同公式见[算法基础](../../README.md#算法基础)。

Host 令 `tr=ceil(S/128)`、`numTasks=B*N*tr`，启动 `useAicNum=min(numTasks,可用核数)` 个 Mix 组。第 `aicIdx` 组处理 `taskId=aicIdx+k*useAicNum`；`batchHeadIdx=taskId/tr`，`i=taskId%tr`。一个 task 负责一个 Query tile 的全部输出，同一 task 的 K/V item 编号为 `j=0...i`。相邻至多两个 item 组成一组，`slot=j%2`。

AIV0 负责 Query 行 0～63，AIV1 负责 64～127。两路 AIV 独立维护对应行的状态，不交换或归并 Softmax 结果。

`--core-num=0` 时使用设备可用 AIC 数；非零时采用请求值，再按 task 数收缩启动规模。请求超过设备核数会报错，不启动 Kernel。

## CV 核间流水与数据布局

满组仍按以下本核顺序发射，没有跨组自由滚动：

```text
AIC: C1(0) → C1(1) → C2(0) → C2(1) | 下一组
AIV: V1(0) → V1(1) → V2(0) → V2(1) | 下一组
```

C1 生成分数，V1 更新 Softmax 并交付 P，C2 生成 DeltaO，V2 更新 `OAcc=alpha*OAcc+DeltaO`。L0 的编号也是 `j%2`；同一 item 的 C1/C2 复用同号 L0，不是给两个阶段各分一套。

C1 用 `K[128,128] × Qᵀ[128,128]` 计算转置分数。Fixpipe 的 `dualDstCtl=2` 沿 Query 维切分，每路 AIV 接收 FP32 `Sᵀ[128 Key,64 Query]`，每个连续的 64 元素对应不同 Query。V1 在这个布局上沿 Key 维扫描，并原地把 S 改为未归一化权重。

`FusedDNToNZCastVF` 将权重转为 BF16 NZ 分块布局。每路 PWork 有四个 16 Query 列分组，每组含 128 个 32 B 数据块和一个 32 B 填充块；`CopyPWorkToL1` 跳过填充，写入共享 P L1 的半块。以 BF16 元素计，目标偏移是 `slot*16384+subAivIdx*8192`。两半按 Query 维拼接，不求和。

C2 通过转置 LoadData 把 L1 的 Pᵀ 装成 L0A 的 P，计算 `P×V`。Fixpipe 默认 `dualDstCtl=1` 按 Query 行切分，每路得到普通行布局的 FP32 `DeltaO[64,128]`。S、P、DeltaO 都不经过 GM。

Host 不申请中间 workspace，提交 Kernel 后即可返回；调用方读取输出或释放输入输出前仍需同步 stream。

| flag | flag ID（slot 0/1） | Set → Wait | 作用 |
| --- | --- | --- | --- |
| S_READY | 0/1 | AIC PIPE_FIX → AIV PIPE_V | S 就绪 |
| P_READY | 2/3 | AIV PIPE_MTE3 → AIC PIPE_MTE1 | 允许读取 P |
| O_READY | 4/5 | AIC PIPE_FIX → AIV PIPE_V | DeltaO 就绪 |
| P_READY_MTE2 | 6/7 | AIV PIPE_MTE3 → AIC PIPE_MTE2 | 允许启动 V 搬入 |

两个 P 通知来自同一次 P 写入完成点，分别由 MTE1/MTE2 消费，不能合并成一个全核 Wait。真机 mode2 下每个 P Wait 都要等两路 AIV 到达；AIC 的 S/O Set 则通知两路 AIV。仿真 mode4 封装分别处理两路，AIV1 的 flag ID 加 16。

没有独立的 slot-free 或 DONE 通知。复用依赖要连起来读：

- S：旧 V1 读完 S 后才能交付 P；旧 C2 的 MTE1 等到 P，后续 C1 的 MTE1 又排在旧 C2 之后，才会沿 Mmad→Fixpipe 写入下一代 S。
- P：下一组 V1 排在本组全部 V2 之后；旧 V2 等待旧 C2 写好 DeltaO，因此下一代 P 不会覆盖尚未被 C2 读取的旧 P。
- DeltaO：下一代 C2 要等下一代 V1 交付 P，而该 V1 排在旧 V2 之后，因此下一次 Fixpipe 写入不会覆盖尚未消费的 DeltaO。
- alpha：两次 V1 分别保存 alpha[0/1]，两次 V2 再按相同顺序读取；下一组 V1 排在这两次 V2 之后。

这些是 CrossCore、Mutex 和同一 Pipe 队列顺序组成的依赖链，不是“函数调用结束就代表所有硬件工作完成”。

## AIC：L0 双槽怎样交接

固定规格下 L1 共 224 KiB，L0A/L0B 各 64 KiB，L0C 128 KiB。

| 资源 | 槽数 × 单槽容量 | Mutex ID | 生命周期 |
| --- | --- | --- | --- |
| K L1 | 2 × 32 KiB | 0/1 | C1 MTE2 写入 → MTE1 读 K |
| V L1 | 2 × 32 KiB | 2/3 | C2 MTE2 写入 → MTE1 读 V |
| Q L1 | 1 × 32 KiB | 4 | task 搬入；首个 C1 的 MTE1 取得，末个 C1 归还 |
| P L1 | 2 × 32 KiB | 无本地 Mutex | AIV 写入，P_READY 保护读取 |
| L0A、L0B | 各 2 × 32 KiB | 5/6，每号联合管理一对 A/B | MTE1 写 → Mmad 读 → 下一次 MTE1 写 |
| L0C | 2 × 64 KiB | 7/8 | Mmad 写 → Fixpipe 读 → 下一次 Mmad 写 |

C1 先向 MTE2 发射 K 的 GM→L1，再在 MTE1 上装 Q→L0B，随后才取得 K Mutex、装 K→L0A。Q 的装载无需等 K，所以这两条搬运 Pipe 可以重叠。末个 C1 在 MTE1 装载结束后归还 Q，后续 Mmad 读取的是 L0B，不再占用 Q L1。

C2 的源码顺序值得单独展开：

```python
WaitAivToAic(MTE1, P_READY[s])       # 位于 KernelProcessForAIC
Lock(L0AB[s], MTE1)                 # 以下位于 CubeStage2
CopyL1ToL0A(P[s], transpose=True)
WaitAivToAic(MTE2, P_READY_MTE2[s])
with mutex(V_L1[s], MTE2):
    CopyGmToL1(V[s])
with mutex(V_L1[s], MTE1):
    CopyL1ToL0B(V[s], transpose=True)
Unlock(L0AB[s], MTE1)
```

先发射 P 装载，不代表要等它完成才发射 V 搬运；两条 Pipe 各自等自己的 P 通知。v04 的 `MUTEX_C2_LOAD_ORDER` 在本版已删除，V 的就绪由 V L1 Mutex 交给 MTE1。

C1/C2 的后半段相同：PIPE_M 取得 L0AB 和 L0C，执行 Mmad，先后归还两者；PIPE_FIX 取得 L0C，Fixpipe 写出后归还。返回主循环后才发出 S/O_READY。下一次 MTE1 只需要等同槽 L0AB，下一次 Mmad 才需要等同槽 L0C，所以另一槽的装载、计算与写出可以错位。所有权等待位于具体 Pipe，不是 Scalar 全核屏障。

## AIV：保留双槽 PWork 与单份递推状态

每路 UB 共 193.25 KiB：

| 资源 | 槽数 × 单槽容量 | 使用方式 |
| --- | --- | --- |
| S、DeltaO | 各 2 × 32 KiB | CrossCore 就绪后供 V1/V2 使用 |
| PWork | 2 × 16.125 KiB | Mutex 0/1：Vector 写入 → MTE3 读取 → 下次 Vector 覆盖 |
| alpha | 2 × 256 B | V1 保存到对应 V2 |
| m、l | 各 1 × 256 B | 同一 Vector 流水按 j 顺序推进 |
| OAcc | 1 × 32 KiB | 两次 V2 依次累加，不按 item 分槽 |

V1 仍分开调用 `OnlineColwiseSoftmaxVF` 和 `FusedDNToNZCastVF`，再由 `CopyPWorkToL1` 写入共享 L1。MTE3 归还 PWork 后依次发出 P_READY、P_READY_MTE2。V2 仍调用普通乘加的 `OnlineUpdateVF`。

最终 `FusedDivCastVF` 读取 OAcc/l，将 BF16 输出写入 PWork[0]，再由 MTE3 写 GM。最终输出沿用 Mutex 0，因此下一 task 重用 PWork[0] 前要等旧输出读完；m/l/OAcc 自身不和输出缓冲别名。两路 AIV 的 Mutex 编号相同，但各自只管理本核 UB。

## 调度伪代码

这里的 Lock/Wait 均绑定所标 Pipe。C1/C2 内的 L0 Lock/Unlock 见上一节，不将函数返回解释成所有 Pipe 已完成。

```python
# AIC
for taskId in range(aicIdx, numTasks, useAicNum):
    i = taskId % tr
    with mutex(Q, MTE2):
        CopyGmToL1(Q, taskId)
    for begin in range(0, i + 1, 2):
        end = min(begin + 2, i + 1)
        for j in range(begin, end):
            s = j % 2
            CubeStage1(j, s)            # K搬入；Q先装L0B，再等K；首/末C1持有/归还Q
            SetAicToAiv(FIX, S_READY[s])
        for j in range(begin, end):
            s = j % 2
            WaitAivToAic(MTE1, P_READY[s])
            CubeStage2(j, s)            # 先发射P的MTE1，再发射MTE2上的P_READY_MTE2等待
            SetAicToAiv(FIX, O_READY[s])

# 每路 AIV
for taskId in range(aicIdx, numTasks, useAicNum):
    i = taskId % tr
    init(OAcc=0, m=FLOAT_LOWEST, l=0)
    for begin in range(0, i + 1, 2):
        end = min(begin + 2, i + 1)
        for j in range(begin, end):
            s = j % 2
            WaitAicToAiv(V, S_READY[s])
            with mutex(PWork[s], V):
                VectorStage1(j, i, subAivIdx)
            with mutex(PWork[s], MTE3):
                CopyPWorkToL1(s, subAivIdx)
            SetAivToAic(MTE3, P_READY[s])
            SetAivToAic(MTE3, P_READY_MTE2[s])
        for j in range(begin, end):
            s = j % 2
            WaitAicToAiv(V, O_READY[s])
            VectorStage2(OAcc, DeltaO[s], alpha[s])
    if outputRows > 0:
        with mutex(PWork[0], V):
            FusedDivCastVF(PWork[0], OAcc, l, outputRows)
        with mutex(PWork[0], MTE3):
            DataCopy(O_GM, PWork[0], outputRows * 128)
```

## 首轮、奇数尾组与序列尾块

每个 task 先初始化 `m=FLOAT_LOWEST`、`l=0`、`OAcc=0`。首个 V1 使用 `OnlineColwiseSoftmaxVF<true,...>`，直接建立 m/l 并令 alpha=1；首个 V2 仍执行普通乘加。只有 `j==i` 的对角 item 做块内 causal mask：最大值和指数求和两遍都屏蔽 `keyLocal>queryLocal`，即逻辑 Query 行、Key 列矩阵的上三角。

组尾取 `groupEnd=min(groupStart+2,i+1)`。例如 `i=2` 时执行 `[0,1]`、`[2]`，尾组只用 slot 0，不产生 slot 1 的通知或等待。Q 的释放条件是 `j+1==kvTileCount`，不能用整条序列的 `tc` 代替。

`CopyGmToL1` 只读 Q/K/V 的有效行，并在 L1 的 NZ 布局中补零。V1 只扫描有效 Key 行，其余 P 行写零，Cube 仍计算完整 128×128 tile。每路输出行数为 `clamp(qValidRows-subAivIdx*64,0,64)`；即使 AIV1 没有有效输出行，也要完成所有 V1/V2 和 P 就绪通知，只跳过最终归一化和 GM 写回。

入口调用 `InitSocState()`，源码没有额外的手工初始 SetFlag 或循环末尾 DONE 排空。各缓冲的最后一次消费通过对应 Mutex 释放；最终输出在 AIV task 尾部发射，不能把 AIC 循环结束当成输出写回完成。

## 从 v04 到 v05

L0A/L0B/L0C 从各一槽增为两槽，Mutex 随物理槽编号；C1 改为先装 Q 再等 K；C2 删除加载顺序 Mutex，改由两个 P_READY 分别通知 MTE1/MTE2。CV 分组、Q/P/K/V L1 槽数、AIV 数值计算和最终输出位置不变。[v06](../v06/README.md) 再把 V 搬运移到 C1，并增加 task 级输入输出双槽。

## 流水示意与性能参考

统一口径下，本版 Task Duration 为 26319.761719 μs，因果有效 Cube MFU 为 38.6810%；条件和完整对比见[总文档性能表](../../README.md#统一性能结果)。

![v05 CV 与 L0 双槽流水示意图](../../images/pipeline/falite_v05_pipeline.png)

示意图展示连续两组与同 item 就绪依赖，省略反向复用及核内 Mutex。图中的 P 就绪边概括了两条 Pipe 的通知；精确位置见 C2 伪代码。色块宽度不是实测耗时。

![v05 上板流水截图](../../images/pipe_trace/falite_v05_pipe.png)

PipeTimeline 截图使用 `B=N=1,S=2048`、单 Mix 核组，窗口为 `[77.589,117.589]` μs，用于观察 MTE1、CUBE 与 FIXP 的重叠；上述长序列性能来自独立计时。

## 代码阅读入口

| 入口 | 重点 |
| --- | --- |
| [Host tiling](host/flash_attn_lite_host.cpp) 的 `ComputeFlashAttnLiteTilingData` | task 数、实际核数、各 SRAM 地址与容量检查 |
| [公共结构](flash_attn_lite_common.h) | 槽数常量、`SRAMLayoutAIC/AIV`；Addr 是字节，多槽缓冲的 Elems 包含全部槽；`rowStatsUBElems=64` 例外，表示单份行统计长度，alpha 总长度还需乘槽数 |
| [Kernel 入口](kernel/flash_attn_lite_kernel.asc) | `__mix__(1,2)`、`InitSocState`、固定 causal 模板 |
| [AIC 实现](kernel/falite_kernel_aic.h) | 先读 `KernelProcessForAIC` 的分组循环，再读 `CubeStage1/2` 和 Mutex |
| [AIV 实现](kernel/falite_kernel_aiv.h) | `KernelProcessForAIV`、`VectorStage1/2`、`CopyPWorkToL1` 和最终输出 VF |
| [同步封装](kernel/falite_kernel_common.h) | flag ID、Set/Wait 所在 Pipe、`GetKvTileCount` 与 `GetTileValidRows` |

从仓库根目录可构建 `cmake --build build --target falite_v05 -j`，运行 `./build/Samples/2_Performance/flash_attn_lite_story/falite_v05 --core-num 1 --size 1 1 385` 阅读满组、奇数尾组和非整块输出路径；完整配置与精度标准见[编译、运行与复现](../../README.md#编译运行与复现)。
