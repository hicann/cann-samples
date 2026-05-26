# FIA算子FP8 per-block全量化场景最佳优化实践

## 概述

本文档系统阐述**FIA**算子的实现原理、性能建模方法与优化实践，覆盖FP8 per-block全量化场景。

## 算子实现原理

### 算子功能说明

- **算子功能**  
适配decode & prefill场景的FlashAttention算子，既可以支持prefill计算场景（PromptFlashAttention），也可支持decode计算场景（IncreFlashAttention）。

- **计算公式**  
self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。
self-attention的计算公式一般定义如下，其中$query、key、value$为输入样本的重要属性元素，是输入样本经过空间变换得到的矩阵，且可以统一到一个特征空间中。$dequantScaleQuery、dequantScaleKey、dequantScaleValue$代表对应的反量化矩阵。公式及算子名称中的"Attention"为"self-attention"的简写。

$$
Attention(query， key， value， dequantScaleQuery， dequantScaleKey， dequantScaleValue) = 
$$
$$
Softmax(\frac{(query * dequantScaleQuery) * (key * dequantScaleKey)^T}{\sqrt{d}})(value * dequantScaleValue)
$$

其中$(query * dequantScaleQuery)$和$(key * dequantScaleKey)^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$value * dequantScaleValue$相乘后得到一个$n*d$的矩阵。

- **FIA per-block全量化介绍**  
为了满足训练场景的低bit量化诉求，对标A3的FP8量化能力，设计了per-block全量化，兼顾进度与性能，实现业界通用量化形式对FP8/HIFP8的支持。
相比于per-tensor全量化，对于$query、key、value$矩阵采用同一个量化参数进行计算，per-block全量化要求根据block_size对输入分开进行量化，通常要求block_size的值，与算子tiling块一致。
per-block全量化主要计算流程：

1、fp8格式的$query、key$矩阵经过matmul矩阵乘运算，输出为fp32类型。

2、经过$dequantScaleQuery、dequantScaleKey$的反量化计算，输出fp32类型。

3、经过softmax计算，cast为fp8类型的P矩阵。

4、fp8格式的P矩阵与V矩阵进行matmul矩阵乘，输出fp32类型。

5、经过$dequantScaleValue$反量化计算，输出fp32类型。

6、经过V2计算，输出bf16类型。


**说明**：
<blockquote>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
<br>Q_S表示query shape中的S，KV_S表示key和value shape中的S，Q_N表示numQueryHeads，KV_N表示numKeyValueHeads。dequantScaleKey表示key的per-block反量化参数；dequantScaleValue表示value的per-block反量化参数；dequantScaleQuery表示query的per-block反量化参数。P表示Softmax(<span>(QK<sup class="superscript">T</sup>) / <span class="sqrt">d</span></span>)的计算结果。</blockquote>

### FIA per-block全量化参数说明

| **变量名** | **描述**          | **Dtype**     | **Layout** | **Shape**                                      |
| ------- | --------------- | ------------- | ---------- | -------------------------------------------------- |
| query       | 公式中的输入Q | `FLOAT8_E4M3FN` | ND         | `(B,N1,S1,D)`                           |
| key       | 公式中的输入K | `FLOAT8_E4M3FN` | ND         | `(B,N2,S2,D)`                                     |
| value  | 公式中的输入V | `FLOAT8_E4M3FN` | ND         | `(B,N2,S2,D)` |
| dequantScaleKey  | 表示key的反量化因子 | `FLOAT32` | ND         | `(B,N2,Ceil(S2/256),1)`                            |
| dequantScaleValue       | 表示value的反量化因子              | `FLOAT32`    | ND         | `(B,N2,Ceil(S2/256),1)`                                      |
| dequantScaleQuery       | 表示query的反量化因子 | `FLOAT32` | ND         |   `(B,N1,Ceil(S1/128),1)`                        |
| numQueryHeads       | query的head个数 | `INT64` | NA         | NA                           |
| softmaxScale       | 公式中d开根号的倒数 | `DOUBLE` | NA         | NA                           |
| inputLayout       | 标识输入query、key、value的数据排布格式 | `STRING` | NA         | NA                           |
| numKeyValueHeads       | key、value中head个数 | `INT64` | NA         | NA                           |
| queryQuantMode       | query反量化的模式 | `INT64` | NA         | NA                        |
| keyQuantMode       | key反量化的模式 | `INT64` | NA         | NA                          |
| valueQuantMode       | value反量化的模式 | `INT64` | NA         | NA                          |
| innerPrecise       | 表示高精度或者高性能选择 | `BOOL` | NA         | NA                     |
| returnSoftmaxLse       | 是否输出softmax_lse | `BOOL` | NA         | NA                         |
| queryDtype       | 用于在PTA接口中指定query的dtype | FP8 per-block全量化场景支持`FLOAT8_E4M3FN` | NA       | NA                          |
| keyDtype       | 用于在PTA接口中指定key的dtype | FP8 per-block全量化场景支持`FLOAT8_E4M3FN` | NA         | NA                           |
| valueDtype       | 用于在PTA接口中指定value的dtype | FP8 per-block全量化场景支持`FLOAT8_E4M3FN` | NA         | NA                          |
| attentionOut       | 公式中的输出 | `FLOAT16` | ND         | `(B,N1,S1,D)`                          |


### 算子实现流程说明
FIA per-block全量化实现流程图如下。
  <div align="center">
    <img src="images/image1-FIA-process-fullquant.png" width="700" />
  </div>


## 算子性能建模

### 性能瓶颈分析

FIA算子的性能瓶颈主要分为以下三类。

1. **CUBE Bound**：算子性能受限于硬件的Cube算力规格，本身已经实现连续的MMAD计算。在prefill场景意味着算子性能已经最优，但需要重点关注**多核计算负载是否均衡**，避免出现单核Cube Bound，但整体Cube利用率偏低的情况。

2. **Memory Bound**：算子性能受限于数据搬运能力，主要的性能优化手段是减少搬运量、提高带宽利用率或者将低带宽的搬运转换成高带宽的搬运，进而发挥算子极致性能。因Bound在不同的流水上而区分出**MTE2 Bound**、**MTE1 Bound**以及**FIXPIPE Bound**。在decode场景Memory Bound意味着算子性能已经接近最优，但是但需要重点关注带宽是否打满。

3. **Vector Bound**：算子性能受限于硬件的vector算力规格，本身已经实现连续的vector计算。这种场景需要通过DN方案提高vetor指令并行度，或者调大FIA算子的基本块的N轴从而减少update次数。

### FIA算子性能建模公式

**1. MMAD计算时间**

$$
T_{cube} = \frac{M \times N \times D + M \times D \times N}{16 \times C0 \times 16 \times 核数 \times 频率}
$$

其中`16 × C0 × 16`表示MX量化在Cube核上每拍的计算量,`M * N * D`表示Bmm1环节计算量，`M * D * N`表示Bmm2环节计算量，M 、 N为基本块的宽高，D为query的shape中的尾轴size。

**2. MTE2搬运时间**

MTE2的搬运量包含了因切分带来的的重复搬运，暂不考虑Scale的搬运：

$$
T_{mte2} = \frac{(M \times D  + 2 \times N \times D ) \times (1 \times sizeof(dtype))}{BandWidth_{mte2}}
$$

其中`M × D`表示query从GM到L1的搬运量，`2 x M × D`表示key、value从GM到L1的搬运量。由于$dequantScaleQuery、dequantScaleKey、dequantScaleValue$的size为$query、key、value$的128 * D分之1或者256 * D分之1，故忽略其从GM到L1的耗时。
MTE2的综合带宽包含DDR带宽和L2带宽的共同作用，可简化为：

$$
T_{mte2} \approx \frac{Size_{DDR}}{BandWidth_{DDR}} + \frac{Size_{L2}}{BandWidth_{L2}}
$$

由于左矩阵复用，在单基本块视角，可以忽略query的搬运耗时，仅考虑key、value的搬运耗时（即仅考虑N * D相关耗时，忽略M * D相关耗时）。

**3. FIXPIPE搬出时间**

$$
T_{fixp} = \frac{M \times N \times sizeof(dtype)}{BandWidth_{fixp}} + \frac{M \times D \times sizeof(dtype)}{BandWidth_{fixp}}
$$

其中`M × N`表示Bmm1的结果从L0到UB的搬运量，`M × D`表示Bmm2的结果从L0到UB的搬运量。

**4. VECTOR计算时间**
采用实测方式统计vector耗时。

### 优化目标

使VECTOR、MMAD、MTE2、MTE1、FIXPIPE尽量匹配，避免单一流水过长；prefill场景实现VECTOR bound, decode场景实现MTE2 bound并打满带宽；不同核之间耗时尽可能接近，提高多核利用率。

## 算子优化实践

本章介绍FIA算子中应用的优化措施，包括通用的指令并行度优化和多核利用率优化，以及针对不同的Bound类型场景分别提供搬运效率优化、计算效率优化方法。

### Step 0 — Naive 基线

直接按数学公式及流程图编写FIA全量化算子，不做任何优化，先拿一个能跑、能测、能暴露瓶颈的基线。

### Step 1 — 多核负载均衡
KV Cache在不同Batch上缓存的历史Key/Value的数据量可能不相同，FA算子批量进行Flash-Attention计算时，不同Batch的计算耗时也不相同。FA根据实际Key/Value数据量大小（actual-sequence），对每个Batch上的计算量（耗时）进行评估。之后按Tiling块粒度，将每个Tiling块的计算均摊到多个核上并行计算，充分发挥算力。
  <div align="center">
    <img src="images/image2-FIA-balance.png" width="900" />
  </div>
该方案称为负载均衡，部分Batch的计算会分到不同核上（例如图中的batch2）。该场景下，FA算子会利用Flash-Decoding算法，对分配在不同核上的计算进行Flash-Decoding归约处理。

负载均衡方案的关键步骤如下：

针对每个输入Batch，先执行数据Tiling切分（将单Batch拆解为若干连续的Tiling块），再对每个Tiling块的计算耗时进行预估值计算。 预估规则：Tiling块的计算耗时与该块的Query（Q）数据量、Key/Value（KV）数据量呈正相关，基于计算数据的拟合模型输出每个 Tiling块的耗时预估值（记为Tᵢ，i为Tiling块序号）。

将所有Tiling块分配至N个计算核上，使得各核累计计算耗时尽可能接近全局平均值，同时保持Tiling块的连续分配。具体分配流程如下：

step1：计算所有Tiling块的总预估耗时（T_total = ΣTᵢ），初始化0核上的初始期望负载：

T_target(0) = T_total / N

step2：对当前待分配的核，依次将Tiling块加入其任务队列。

若某个Batch内所有Tiling块的总预估耗时T_batch ≤ T_target，则将整个Batch分配至当前核，跳过逐块遍历，提升分配效率。

在每个核最后一块的判断上，设T_accumulated为该核已分配的Tiling块预估耗时之和，若第j个Tiling块为最后一块，满足：

T_accumulated + (Tj / 2) ≤ T_target
此条件引入了冗余度，确保在实现负载均衡的同时，避免因单个Tiling块耗时略超预期而未被分配，导致后续核的任务越来越重。

step3：每当一个核完成分配后，剩余未分配Tiling块的总耗时（T_remain）与剩余核数（N_remain）用于更新下一个核（k）的目标负载：

T_target(k) = T_remain / N_remain

此动态调整机制可有效应对早期分配偏差，提升整体均衡性。

step4：重复step2、step3操作，直到最后一个核放入所有Tiling块，结束分配。

负载均衡方案的核心流程如下：
  <div align="center">
    <img src="images/image3-FIA-balance.png" width="400" />
  </div>

负载均衡的伪代码实现如下：
 ```cpp
MaxCost = Inf
for CoreUsePlan ← MinCore.MaxCore do
    UnDistributedCost ← CostAll
    LoadIdx = 0
    CurMaxCost = 0
    for i ← 0, CoreUsePlan - 1 do:
        CostLimit = UnDistributedLoad/(CoreUsePlan-i)
        CoreLoad(i) = 0
        CoreCost(i) = 0
        for j ← LoadIdx, TotalLoad - 1 do:
            LoadCost(j) = BasicAveCost(M(j), basicS2(j))
            if CoreCost(i) + LoadCost(j) * 0.5 < CostLimit:
                CoreLoad(i) = CoreLoad(i) + 1
                CoreCost(i) = CoreCost(i) + LoadCost(j)                
                CurMaxCost = max(CurMaxCost, CoreCost(i))
                UnDistributedLoad = UnDistributedLoad - CoreLoad(i)
            else:
                LoadIdx = j
                break
    if CurMaxCost + CoreUsePlan < MaxCost:
        Update Plan: MaxCost = CurMaxCost + CoreUsePlan

```

理论最大收益的计算：

假设63个核分配了n个最小块（minBasic），1个核分配n个最大块（maxBasic），此时的计算耗时为A：

A = maxBasic ∗ n

因受慢核影响，如果负载均匀分配，即(maxBasic * n - minBasic * n)的负载量均匀分配给64个核，此时的计算耗时为B：

B = minBasic ∗ n + (maxBasic ∗ n − minBasic ∗ n) / 64

那么节省的时间为C = A - B

C = maxBasic ∗ n − minBasic ∗ n−(maxBasic ∗ n − minBasic ∗ n) / 64 = ((maxBasic − minBasic) ∗ 23n) / 64

均衡后的收益率为D = C / A

D = ((maxBasic − minBasic) ∗ 63n) / 64 ÷ (maxBasic ∗ n) × 100%

根据实测，maxBasic可为13.9us、minBasic可为2.4us，此时性能收益D约为81%。

**性能数据：**

| 指标 | Step 0 | Step 1 | 变化 |
|------|--------|--------|------|
| **耗时** | maxBasic ∗ n | minBasic ∗ n + (maxBasic ∗ n − minBasic ∗ n) / 64 | **-81%，5.26x** |


### Step 2 — 流水优化

#### Cube/Vector核间流水并行
FA算子涉及Cube和Vector计算，通过合理排布Cube、Vector核间流水，实现核并行工作。具体来说，调整切块后mm1、vec1、mm2、vec2的执行顺序，将没有数据依赖的mm计算提前执行，从而使得没有数据依赖关系的Cube计算和Vector计算能够并行执行。

下图为FA算子未排布Cueb、Vector核间流水时，Cube与Vector核上的执行示意图。
  <div align="center">
    <img src="images/image4-FIA-balance.png" width="1100" />
  </div>
可以发现FA的四阶段计算（mm1、vec1、mm2、vec2）为串行执行，这样会导致Cube核在执行时Vector核空闲，而Vector核执行时Cube核空闲，从而导致Cube或Vector核利用率非常低下。

根据FA四阶段计算中执行依赖关系：

① 相同轮次的mm1、vec1、mm2、vec2必须顺序执行；

② 不同轮次之间的vec1顺序执行；

③ 不同轮次之间的vec2顺序执行。

可以通过错位计算轮次，使得非同一轮次的mm1与vec1并行执行，非同一轮次的mm1与vec2并行执行，非同一轮次的mm2与vec1并行执行，非同一轮次的mm2与vec2并行执行。
  <div align="center">
    <img src="images/image5-FIA-balance.png" width="800" />
  </div>
流水代码执行如下图所示：
  <div align="center">
    <img src="images/image6-FIA-balance.png" width="800" />
  </div>

 ```cpp
Loop1：触发第一轮的mm1
Loop2：触发第二轮的mm1、第一轮的vec1、第一轮的mm2
Loop3：触发第三轮的mm1、第二轮的vec1、第二轮的mm2、第一轮的vec2
Loop4：               第三轮的vec1、第三轮的mm2、第二轮的vec2
Loop5：                                       第三轮的vec2

```
核间流水排布的伪代码如下：
 ```cpp
static constexpr uint32_t FIA_PRELOAD_TASK_CACHE_SIZE = 3;
for (B、N、S1、S2):
    RunInfo &extraInfo0 = extraInfo[loop % FIA_PRELOAD_TASK_CACHE_SIZE];       // 本轮任务
    RunInfo &extraInfo2 = extraInfo[(loop + 2) % FIA_PRELOAD_TASK_CACHE_SIZE]; // 上一轮任务
    RunInfo &extraInfo1 = extraInfo[(loop + 1) % FIA_PRELOAD_TASK_CACHE_SIZE]; // 上两轮任务
    if (extraInfo0.isValid) {
        if ASCEND_IS_AIC {
            ComputeMm1(extraInfo0);
        }
    }
    if (extraInfo2.isValid) {
        if ASCEND_IS_AIV {
            ComputeVec1(extraInfo2);
        }
        if ASCEND_IS_AIC {
            ComputeMm2(extraInfo2);
        }
    }
    if (extraInfo1.isValid) {
        if ASCEND_IS_AIV {
            ComputeVec2(extraInfo1);
        }
        extraInfo1.isValid = false;
    }

```

#### Cube核内流水并行（GQA）
FA-GQA场景下，构建Cube核内pipeline流水，提高访存和计算并行效率，具体流程如下。

1、在N轴、M轴循环遍历时：

Query常驻L1，只在nLoop为0时，将Query从GM拷入L1，减少Query的访存；

Query在L1上的buffer设计为双buffer，使得n、m循环当前轮的MTE1（QueryL1 copy to L0A）能与下一轮的MTE2（QueryGm copy to KeyL1）并行执行；

Key在L1上的buffer设计为乒乓buffer，使得n、m循环当前轮的MTE1（KeyL1 copy to L0B）能与下一轮的MTE2（KeyGm copy to KeyL1）并行执行；
  <div align="center">
    <img src="images/image7-FIA-balance.png" width="1100" />
  </div>

2、在K轴循环遍历时：
L0A、L0B的设计为乒乓buffer，使得k循环当前轮的MMAD（L0A*L0B）能与下一轮的MTE1（QueryL1 copy to L0A 和 KeyL1 copy to L0B）并行执行；
  <div align="center">
    <img src="images/image8-FIA-balance.png" width="500" />
  </div>

mm1计算的伪代码如下：
```cpp
// Tiling块大小
SingleN=256
SingleM=128
SingleK=128

for nLoop:SingleN/N_BASE:
    WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId);
    CopyKeyToL1(L1KeyBufferDb[kvBufId % 2], GmKey) 
    SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId);
    WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId);

    for mLoop:SingleM/M_BASE:
        if (nloop is first) {
            WaitFlag<HardEvent::MTE1_MTE2>(QP_EVENT0 + mLoop);            
            CopyQGmToL1(L1Query[mLoop], GmQuery)  
            SetFlag<HardEvent::MTE2_MTE1>(QP_EVENT0 + mLoop);
            WaitFlag<HardEvent::MTE2_MTE1>(QP_EVENT0 + mLoop);
        }

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId);
        for kLoop:SingleK/K_BASE:
             WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + l0abBufId);   
             LoadDataToL0A(L0ABufferDb[l0abBufId], L1Query[m])               // Query L1 to L0A
             LoadDataToLoB(L0BBufferDb[l0abBufId], L1KeyBufferDb[kvBufId])   // Key L1 to L0B
             SetFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + l0abBufId);
             WaitFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + l0abBufId);
             Mmad(L0CBufferDb[l0cBufId % 2], L0ABufferDb[l0abBufId], L0BBufferDb[l0abBufId])                     
             SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + l0abBufId);
             l0abBufId = (l0abBufId + 1) % 2;
        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId);
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId);
        Fixpipe(GM, L0CBufferDb[l0cBufId])   // L0C to GM   
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId); 
        l0cBufId = (l0cBufId + 1) % 2;         
       
        if (nloop is last) {
            SetFlag<HardEvent::MTE1_MTE2>(QP_EVENT0 + mLoop);
        }

    SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId);
    kvBufId = (kvBufId + 1) % 2;                                     第三轮的vec2

```

基于Ascend950平台，典型case(B = 64, N1 = 32, N2 = 32, S1 = 8192, S2 = 38048, D = 96)流水性能收益实测如下：
| 指标 | Step 1 | Step 2 | 变化 |
|------|--------|--------|------|
| **耗时** | 2934847.77us | 1641769.82us | **-44%，1.78x** |


### Step 3 —  使用DN方案提升指令并行度
1、背景：在进行vector计算时，应该尽量使用双发VF指令（理论IPC=2）,此时一个circle可以发射2条指令，代码并行度高。但是在ND方案中，softmax环节，需要采用单发的vcmax指令和vcadd指令来求解行最大值、行和，单发指令的理论IPC为1，指令发射效率较低。而DN方案可以使用双发的vmax指令和vadd指令，替代单发的上述指令，提升性能。

2、原理：
基本块M * N = 128 * 256时, ND方案时，需要获取每行的Max值，故需要对每一行使用vcmax指令，单发vcmax指令需要循环128次，影响VF的整体效率。流程图如下：
  <div align="center">
    <img src="images/image16-FIA-DN.png" width="400" />
  </div>

DN方案时，Bmm1的结果所在基本块转置，第i行是原版ND基本块第i列元素，此时相邻行之间可以不断循环进行双发的vmax指令，即可将原本ND方案中每一行的最大值，保存在一个寄存器中，示意图如下。
  <div align="center">
    <img src="images/image17-FIA-DN.png" width="500" />
  </div>

3、主要修改点：

1）Bmm1从原本的quer * key变为key * query ^ T,输出结果shape从[M, N]变为[N, M]。

2）vector1环节需要使用双发的vmax指令替代原本的单发的vcmax指令，来获取最大值。

3）Bmm2需要对vector1输出的shape为[N, M]的矩阵进行转置。


**性能数据：**

在单基本块维度(M = 128，N = 256)，基于Ascend950平台，DN方案的VF环节性能收益实测如下：
| 指标 | Step 2 | Step 3 | 变化 |
|------|--------|--------|------|
| **耗时** | 3849 circle | 2498 circle | **-35%，1.53x** |


### Step 4 —  搬运效率优化 : Double Buffer（双缓冲）

  Double Buffer使用两个缓冲区交替工作：一个缓冲区用于当前计算，另一个并行准备下一轮数据。通过计算与数据加载/准备的重叠，隐藏内存访问延迟，减少流水线停顿，提高算子吞吐量。
  详细介绍及效果可以查看专题解说[N-Buffer特性介绍](../../1_Features/instruction_optimization/n_buffer/README.md)。



## 优化策略选择指南

### 根据 Bound 类型选择优化策略


| **Bound类型**   | **推荐优化策略**       | **优先级**                                |
| ------------- | ----------------------- | ----------------------------------------- |
| CUBE Bound    | 多核负载均衡、Cube核内流水并行      | 高                 |
| VECTOR Bound    | Cube/Vector核间流水并行、DN方案            | 高                 |
| MTE2 Bound    | 基本块调整                    | 高   |
| FIXPIPE Bound | 基本块N轴调大                        | 中                                 |
| 流水停顿       | Double Buffer                  | 高 |

## 性能调优实践步骤

1. **性能分析**：使用Profiling工具结合性能建模分析当前算子的Bound类型
2. **瓶颈识别**：确定主要的性能瓶颈所在
3. **策略选择**：根据Bound类型选择合适的优化策略
4. **参数调优**：调整tiling参数，优化缓冲区使用
5. **效果验证**：对比优化前后的性能数据
6. **迭代优化**：根据结果进一步调整优化策略

## 总结

FIA算子的性能优化不仅要考虑优化核内流水的性能，还要考虑核间的负载均衡，同时根据性能建模及bound类型选择合适的优化策略。