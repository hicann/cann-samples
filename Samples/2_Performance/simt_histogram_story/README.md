# 以 Histogram 算子为例：串行 → SIMT 并行改写与调优指南

本样例面向 Ascend 950 系列上的 SIMT 编程教学，围绕 Histogram（直方图）算子，**核心教学目标是 SIMT 并行计数（grid-stride 跨步循环）**：将串行 `for (i=0; i<N; i++)` 替换为 `for (i=globalTid; i<N; i+=totalThreads)`，让每个线程只处理自己"负责"的那部分数据。在此基础上逐步引入 GM 直接访问、`asc_atomic_add` 在 UB 处理线程冲突、float4 向量化访存优化，以及 launch bounds + GridDim 调优，完整展示 A5 纯 SIMT 编程模型下的算子开发与性能优化路径。

## 支持范围

| 项目 | 说明 |
|:---|:---|
| **硬件 / 架构** | Ascend 950PR / 950DT，`NPU_ARCH=dav-3510` |
| **特性线** | SIMT |
| **编程模型** | 纯 SIMT（`__global__` 无 `__aicore__`，`--enable-simt` 编译） |
| **算子** | Histogram（直方图统计） |
| **数据类型** | 输入 `float32`，输出 `int32` |
| **输入 / 输出** | `x[1,000,000]`，`min[1]`，`max[1]` → `y[100]`（100 bins） |
| **语义** | 统计 `x` 中落在 `[min, max]` 区间内的元素在各等宽 bin 中的计数 |

已知限制：

- 本样例仅覆盖 `float32` 输入的直方图，不覆盖 `int32 / int8 / fp16` 等多类型分支。
- 本样例聚焦纯 SIMT 模型（`__global__` + `--enable-simt`），不使用 `__simt_vf__` / `VF_CALL` 混合模型。
- 纯 SIMT 下 `kernel_operator.h`（MTE / DataCopyPad / TPipe 等）不可用——`--enable-simt` 编译环境不包含 MTE 头文件，数据必须通过 GM 直接访问。
- 性能数据与硬件、CANN 版本、数据分布强相关，请以本地 msprof 为准。

## 目录说明

```text
simt_histogram_story/
├── CMakeLists.txt
├── README.md
├── include/
│   └── sample_common.h          # ACL 初始化、数据加载、golden 校验、RunSample 入口
├── scripts/
│   └── gen_data.py              # 生成输入 x、min、max 和 golden 直方图
└── src/
    ├── 0_serial_scalar.asc      # Case 0：串行/标量基线（MTE+Vector）
    ├── 1_basic_simt.asc         # Case 1：基础 SIMT 并行
    ├── 2_mem_optimized.asc      # Case 2：访存优化（float4 向量化）
    └── 3_advanced_tuning.asc    # Case 3：进阶调优（launch_bounds + GridDim）
```

## Histogram 语义

```text
y = [0] * bins
for each val in x:
    if min_val <= val <= max_val:
        idx = floor((val - min_val) * bins / (max_val - min_val))
        if idx == bins: idx = bins - 1
        y[idx] += 1
```

当多个元素落在同一 bin 时，多个 SIMT 线程可能同时写同一 bin 地址，需要原子操作或分区归约保证正确性。

## Case 0：串行/标量基线（MTE+Vector）

**源文件**：`src/0_serial_scalar.asc`

**目标**：展示传统 Ascend C MTE+Vector 实现，作为 SIMT 改写的性能与精度基线。

**核心实现**：

- `__global__ __aicore__` kernel，通过 `GetBlockIdx/GetBlockNum` 做多核切分
- MTE (`DataCopyPad`) 将输入分 tile 搬运到 UB（双缓冲流水）
- Scalar 逐元素计算 bin 索引，`GetValue/SetValue` 更新 UB 上的直方图
- 多核通过 `SetAtomicAdd + DataCopyPad` 归约到 GM
- `SetFlag/WaitFlag` 硬件事件同步

**关键代码**：

```cpp
// MTE 双缓冲流水：CopyIn（MTE 搬运）与 Compute（Scalar 计算）重叠
for (int64_t t = 0; t < tileNum; t++) {
    // --- CopyIn: MTE 搬运第 t 个 tile 到 UB ---
    LocalTensor<XType> xLocal = xQue.AllocTensor<XType>();
    DataCopyPad(xLocal, xGm[coreDataStart + t * TILE_DATA_LENGTH], cp, pad);
    xQue.EnQue(xLocal);

    // --- Compute: DeQue → 串行逐元素计算 → FreeTensor ---
    xLocal = xQue.DeQue<XType>();
    for (int32_t i = 0; i < TILE_DATA_LENGTH; i++) {
        XType val = xLocal.GetValue(i);          // 从 UB 读
        if (val >= minVal && val <= maxVal) {
            int32_t idx = static_cast<int32_t>(
                static_cast<float>(val - minVal) * bins / minMaxRange);
            if (idx == bins) idx = bins - 1;
            yLocal.SetValue(idx, yLocal.GetValue(idx) + 1);  // UB 上累加
        }
    }
    xQue.FreeTensor(xLocal);
}
```

**样例配置**：

- 核数：4 cores，通过 `GetBlockIdx/GetBlockNum` 多核切分
- 输入数据量：1,000,000 个 float（4MB）
- 每个 core 处理 ~250,000 个元素，分 tile 用双缓冲搬运

**性能数据**：

| Task Duration(μs) | aiv_time(μs) | aiv_total_cycles | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | L2 hit rate |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 11,840 | 11,839 | 78,124,711 | 0.3 | 0.000 | 11,823 | 0.999 | 14.2 | 0.001 | 0.76% |

> Case 0 的 `mte2_time` / `mte2_ratio` 源自 MTE 双缓冲 DMA 搬运（GM↔UB），Case 1-3 为纯 SIMT 模型，无 MTE 硬件单元参与，该项为 0，后续表格不再列出。

**优化效果分析**：

- 端到端耗时：**11,840μs**，主要瓶颈为 Scalar 串行逐元素计算，占比 **99.9%**
- 每个 core 串行处理 250K 元素，Scalar 引擎的吞吐构成性能上限
- MTE 双缓冲流水有效隐藏了部分 GM 访存延迟，但无法突破 Scalar 串行执行的根本约束

**运行**：

```bash
cmake --build build --target simt_histogram_0_serial_scalar
./build/Samples/2_Performance/simt_histogram_story/simt_histogram_0_serial_scalar
```

预期输出：

```text
[0_serial_scalar] step 0 PASSED
```

---

## Case 1：基础 SIMT 并行

**源文件**：`src/1_basic_simt.asc`

**目标**：将 Case 0 的串行逻辑首次改写为纯 SIMT 并行。核心是 **SIMT 并行计数**，在此基础上附带展示 CUDA 风格线程模型、GM 直接访问、`asc_atomic_add` 在 UB 上处理多线程写同一 bin 的冲突。

**为什么做这个优化**：Case 0 的 `aiv_scalar_ratio=0.999` 表明 99.9% 的执行时间消耗在 Scalar 串行计算上。单个 core 串行处理 250K 元素，Scalar 引擎的吞吐直接决定了端到端耗时。SIMT 并行化的核心思路是将这 250K 元素拆分为多个线程并行处理：4,096 个线程同时从 GM 读取各自负责的元素、各自更新直方图，将原本由 1 条 Scalar 流水线串行执行的工作分配到 4,096 条 Vector 流水线上——理论上能把 Scalar 瓶颈完全消除，将执行主力从 Scalar 引擎切换到 Vector 引擎。

**核心实现**：

- `__global__` kernel（无 `__aicore__`），`--enable-simt` 编译，`<<<4, 1024, 0, stream>>>` 启动
- `threadIdx.x / blockDim.x / blockIdx.x / gridDim.x` 替代 `GetBlockIdx/GetBlockNum`
- 线程以 grid-stride 跨步方式直接从 GM 读取 `x[i]`（标量）
- `asc_atomic_add` 在 UB 上做原子累加，处理多线程写同一 bin 的冲突
- `asc_atomic_add` 归约到 GM，多 Block 间原子累加得到最终结果

**与 Case 0 的范式差异**：

Case 0 和 Case 1 的差异本质是两种编程模型——SIMD 与 SIMT——的切换，其影响贯穿 kernel 声明、线程抽象、内存管理、数据获取、冲突处理和同步等各个层面。

- **kernel 声明与编译**：Case 0 使用 `__global__ __aicore__`，以默认标志编译，内核在 AI Core 的 SIMD 流水线上执行。Case 1 使用 `__global__`（无 `__aicore__`），以 `--enable-simt` 编译，内核按 SIMT 线程模型调度。
- **线程抽象与索引**：Case 0 通过 `GetBlockIdx()/GetBlockNum()` 获取物理 core ID，手动计算每个 core 的数据段 `[coreDataStart, coreDataLength)`。Case 1 通过 `blockIdx.x*blockDim.x+threadIdx.x` 计算全局线程 ID，配合 grid-stride 跨步循环自动分配数据——线程索引是逻辑的而非物理的。
- **UB 内存管理**：Case 0 使用 `TPipe/TQue/AllocTensor` 动态管理 UB buffer，buffer 的分配、入队、出队和释放均由程序员显式控制以匹配 MTE 流水线。Case 1 使用 `__ubuf__[]` 静态数组，编译期确定大小，无需运行时 buffer 管理。
- **数据获取路径**：Case 0 的数据经 GM→UB→寄存器三级流转——MTE 通过 `DataCopyPad` 将数据从 GM DMA 搬运至 UB，Scalar 引擎再通过 `GetValue` 从 UB 读取。Case 1 直接从 GM 加载至寄存器（`float val = x[i]`），省去了 UB 中转环节，代价是丧失 MTE 的批量 DMA 效率。
- **冲突处理**：Case 0 单核内串行执行，不存在对同一地址的并发写。Case 1 多个 SIMT 线程并发更新直方图，同一 bin 可能被多个线程同时写入，必须通过 `asc_atomic_add` 在 UB 上完成原子累加以保证结果正确。
- **同步机制**：Case 0 通过 `SetFlag/WaitFlag` 实现 MTE 与 Scalar 之间的流水线事件同步，确保 DMA 完成后再开始计算。Case 1 通过 `asc_syncthreads()` 实现线程栅栏——阻塞当前线程直至 Block 内所有线程到达同一点。

| 维度 | Case 0 | Case 1 |
|:---|:---|:---|
| kernel 声明 | `__global__ __aicore__` | `__global__` |
| 编译标志 | 默认 | `--enable-simt` |
| 线程索引 | `GetBlockIdx/GetBlockNum`（物理 core） | `blockIdx.x / gridDim.x`（逻辑线程） |
| UB 内存 | `TPipe/TQue/AllocTensor`（动态管理） | `__ubuf__[]` 静态数组 |
| 数据搬运 | MTE `DataCopyPad`（双缓冲 DMA） | `x[i]` 直接读 GM |
| 冲突处理 | 无需（单核串行） | `asc_atomic_add` 在 UB 处理 |
| 同步 | `SetFlag/WaitFlag`（流水线事件） | `asc_syncthreads()`（线程栅栏） |

**关键代码**：

```cpp
// 核心范式：SIMT 并行计数（grid-stride 跨步循环）
// 对比 Case 0 的串行 for (i=0; i<N; i++)
//           → SIMT 的 for (i=globalTid; i<N; i+=totalThreads)
int32_t totalThreads = gridDim.x * blockDim.x;          // 总线程数，即并行度
int32_t globalTid = blockIdx.x * blockDim.x + threadIdx.x; // 全局线程 ID

for (int32_t i = globalTid; i < totalLength; i += totalThreads) {  // grid-stride
    float val = x[i];                              // GM 直接读（指针解引用）
    if (val >= minVal && val <= maxVal) {
        int32_t idx = static_cast<int32_t>(
            (val - minVal) * binsF / range);
        if (idx == bins) idx = bins - 1;
        asc_atomic_add(warpHistos[warpId] + idx, 1); // UB 原子累加
    }
}
asc_syncthreads();

// 跨 Warp 求和后原子归约到 GM
for (int32_t i = tid; i < bins; i += blockThreads) {
    OutType sum = 0;
    for (int32_t w = 0; w < WARPS_PER_BLOCK; w++) {
        sum += warpHistos[w][i];
    }
    asc_atomic_add(y + i, sum);                    // GM 原子归约
}
```

> **grid-stride 跨步循环是本 Story 的核心教学点。** 后续 Case 2/3 的所有优化（float4 向量化、launch bounds、GridDim）都是在此并行计数框架上叠加，计数范式本身不变。

**样例配置**：

- Block 数 × 每 Block 线程数：4 × 1,024 = 4,096 线程
- 每线程处理 ~244 个标量 load（1M ÷ 4,096），grid-stride 跨步
- UB 直方图：`__ubuf__ int32_t[32][108]`（每 Warp 一份独立直方图），~13.5KB

**性能数据**：

| Task Duration(μs) | aiv_time(μs) | aiv_total_cycles | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | L2 hit rate |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 255.8 | 254.8 | 1,678,793 | 252.2 | 0.990 | 1.1 | 0.004 | 0.94% |

**优化效果分析**：

- 端到端耗时：**255.8μs**，相比 Case 0 加速 **46.3 倍**
- 主 Bound 从 Scalar (99.9%) 切换为 Vector (99%)——SIMT 并行化成功将串行瓶颈完全消除
- `asc_atomic_add` 在 UB 上以极低延迟处理多线程写同一 bin 的冲突
- GM 直接访问（标量 `x[i]`）替代 MTE 双缓冲 DMA，简化编程模型的同时消除了 MTE 的 `<uint16_t` 限制

**运行**：

```bash
cmake --build build --target simt_histogram_1_basic_simt
./build/Samples/2_Performance/simt_histogram_story/simt_histogram_1_basic_simt
```

预期输出：

```text
[1_basic_simt] step 1 PASSED
```

---

## Case 2：访存优化（float4 向量化）

**源文件**：`src/2_mem_optimized.asc`

**目标**：在 Case 1 基础上引入 float4 向量化读取，将 load 指令数降为 1/4，探索能否进一步降低耗时。

**为什么做这个优化**：Case 1 的 L2 命中率仅 0.94%。虽然默认路径下 DCache 可能拦截了部分请求导致 L2 计数器无法反映全貌，但该数值仍提示 GM 侧的单次 load 延迟可能处于较高水平。结合 `vec_ratio=0.990`——Vector 单元近乎完全忙碌，产生两种可能的假设：

- **假设 A（GM 带宽瓶颈）**：Vector 的忙碌主要消耗在等待 GM load 返回。若 GM 带宽已饱和，数据供给速率将低于 Vector 的消费速率，实际执行时间由访存延迟主导。
- **假设 B（计算瓶颈）**：Vector 的忙碌源自 `asc_atomic_add` 等计算指令占据执行流水线。GM load 的延迟被计算开销完全掩盖，即使降低访存代价也不会改变端到端耗时。

为区分这两种可能，Case 2 引入 float4 向量化，将 load 指令数降至 1/4。若耗时随之下降，表明 GM 侧为瓶颈（假设 A 成立）；若耗时不变，表明瓶颈在计算侧（假设 B 成立），减少访存代价无效。该优化因此同时承担两个角色：性能尝试与瓶颈假设验证。

**前提依赖**：UB 直方图分配、`asc_atomic_add` 冲突处理、归约逻辑全部继承自 Case 1。

**新增优化**：

- `float4 v = *reinterpret_cast<float4*>(x + i)`：一次 load 读 4 个 float，load 指令数降为 1/4

**关键代码**（仅列出与 Case 1 不同的部分）：

```cpp
// Case 1: float val = x[i];                     // 标量逐元素读 GM
// Case 2:
int32_t vecEnd = (totalLength / 4) * 4;           // float4 边界检测
for (int32_t i = globalTid * 4; i < vecEnd; i += totalThreads * 4) {
    float4 v = *reinterpret_cast<float4*>(x + i);  // 一次读 4 个 float
    // 处理 v.x, v.y, v.z, v.w ...
}
// 标量收尾：处理末尾不足 4 个元素
for (int32_t i = vecEnd + globalTid; i < totalLength; i += totalThreads) {
    float val = x[i];
}
```

**样例配置**：

- 同 Case 1（4 Blocks × 1,024 线程），仅修改 GM 读取方式。

**性能数据**：

| Task Duration(μs) | aiv_time(μs) | aiv_total_cycles | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | L2 hit rate |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 258.7 | 257.0 | 1,691,553 | 254.2 | 0.989 | 1.1 | 0.004 | 0.87% |

**优化效果分析**：

- 端到端耗时：**258.7μs**，相比 Case 1 几乎无变化（255.8 → 258.7 us）
- float4 将 load 指令数降为 1/4，但在当前 4,096 线程配置下 GM 带宽并非瓶颈，减少指令数未带来可见性能提升
- L2 命中率与 Case 1 相当（0.87% vs 0.94%），验证了两者 GM 访问路径相同（均走默认路径经 DCache）
- **教学意义**：float4 向量化是 GM 访存优化的基本手段，但它的收益取决于实际瓶颈所在。Case 2 展示了使用 msprof 数据定量验证优化有效性的完整流程

**运行**：

```bash
cmake --build build --target simt_histogram_2_mem_optimized
./build/Samples/2_Performance/simt_histogram_story/simt_histogram_2_mem_optimized
```

预期输出：

```text
[2_mem_optimized] step 2 PASSED
```

---

## Case 3：进阶调优（launch_bounds + GridDim）

**源文件**：`src/3_advanced_tuning.asc`

**目标**：通过 `__launch_bounds__` 寄存器分析和 GridDim 扫描，找到最优的线程/Block 配置。

**为什么做这个优化**：Case 2 的 float4 将 GM load 指令数降为 1/4，但 `aiv_total_cycles` 从 1,678,793 变为 1,691,553——几乎不变。这说明**瓶颈不在 GM 访存侧，而在 Vector 计算侧**（假设 B 成立）。确认计算瓶颈后，优化方向有两种选择：减少每条计算指令的代价，或减少每个线程的计算指令总数。`asc_atomic_add` 的延迟由硬件决定，无法通过软件降低单条指令的代价，因此唯一有效的路径是减少每个线程的执行指令数——即提升总线程数。当前 4,096 线程每线程处理 ~244 个元素，若将 GridDim 从 4 提升到 32（总线程数至 16,384），每线程只需处理 ~61 个元素，Vector 的计算负载相应降低。这需要两步：通过 `--cce-res-usage` 确定合适的每 Block 线程数（避免寄存器溢出），再通过 GridDim 扫描找到收益拐点（避免调度开销反超并行收益）。

**前提依赖**：UB 直方图分配、float4 向量化、归约逻辑全部继承自 Case 2。

**调优过程**：

Step 1 — `__launch_bounds__` 寄存器分析（`--cce-res-usage` 编译结果）：

| `__launch_bounds__` | 寄存器上限 | 实际使用 | Stack | 判定 |
|:---:|:---:|:---:|:---:|:---:|
| 512 | 64 | 34 | 0 | 最优（零溢出） |
| 1024 | 32 | 32 | 8 | 达上限 |
| 2048 | 16 | 16 | 128 | 严重溢出 |

选择 512 线程：每线程 64 寄存器，实际仅用 34 个，零栈溢出。

Step 2 — GridDim 扫描（固定 512 线程/Block）：

| GridDim | 总线程 | Task Duration | 判定 |
|:---:|:---:|:---:|:---:|
| 2 | 1,024 | 481.9 us | |
| 4 | 2,048 | 245.9 us | |
| 8 | 4,096 | 128.1 us | |
| 16 | 8,192 | 69.9 us | |
| **32** | **16,384** | **40.5 us** | 最优（拐点） |
| 64 | 32,768 | 46.1 us | 调度开销反超 |

选择 GridDim=32：每线程仅 ~15 次 float4 load。32→64 时额外 Block 的调度开销超过并行度提升。

**新增优化**：

- `__launch_bounds__(512)`：锁定每线程 64 寄存器，零溢出
- GridDim 4→32：总线程 4096→16384，提升并行度

**关键代码**（仅列出与 Case 2 不同的部分）：

```cpp
// Case 1/2: __global__ void kernel(...)           // 无 launch_bounds
// Case 3:
__global__ __launch_bounds__(512) void kernel(...)  // 64 寄存器/线程，零溢出

// Case 1/2: kernel<<<4, 1024, 0, stream>>>(...); // 4 Blocks × 1024 线程
// Case 3:
kernel<<<32, 512, 0, stream>>>(...);               // 32 Blocks × 512 线程
```

**样例配置**：

- Block 数 × 每 Block 线程数：32 × 512 = 16,384 线程
- 每线程处理 ~15 次 float4 load（1M ÷ 16,384 ÷ 4），相比 Case 2 的 244 次标量 load 大幅减少

**性能数据**：

| Task Duration(μs) | aiv_time(μs) | aiv_total_cycles | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | L2 hit rate |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **40.5** | 39.0 | 1,985,459 | 36.5 | 0.935 | 0.8 | 0.020 | 8.6% |

**优化效果分析**：

- 端到端耗时：**40.5μs**，相比 Case 2 加速 **6.4 倍**，相比 Case 0 加速 **292 倍**
- 通过 `--cce-res-usage` 定量选择 `__launch_bounds__(512)`，确保零寄存器溢出
- GridDim=32 是收益拐点——继续增加到 64 时调度开销反超并行收益
- float4 向量化减少了 75% 的 load 指令数，与 GridDim=32 的并行度提升叠加，共同贡献 6.4 倍加速

**运行**：

```bash
cmake --build build --target simt_histogram_3_advanced_tuning
./build/Samples/2_Performance/simt_histogram_story/simt_histogram_3_advanced_tuning
```

预期输出：

```text
[3_advanced_tuning] step 3 PASSED
```

---

## 性能对比

以下数据在 Ascend 950PR 上采集（CANN 9.1.0-beta3，warm-up=5）。

| Case | 优化组合（逐 Case 累加） | Task Duration | vs Case 0 | 主 Bound | L2 hit rate | 说明 |
|:---:|:---|:---:|:---:|:---|:---:|:---|
| 0 | MTE + Vector 串行基线 | 11,840 us | 1x | Scalar (99.9%) | 0.76% | 串行逐元素，Scalar 引擎为绝对瓶颈 |
| 1 | Case 0 + SIMT 并行 | 255.8 us | 46.3x | Vector (99%) | 0.94% | `asc_atomic_add` 在 UB 处理冲突，GM 直接访问 |
| 2 | Case 1 + float4 向量化 | 258.7 us | 45.8x | Vector (99%) | 0.87% | float4 减少 load 指令数，但 GM 非瓶颈，实际性能未变 |
| 3 | Case 2 + launch_bounds(512) + GridDim=32 | **40.5 us** | **292x** | Vector (98%) | 8.6% | GridDim 32 并行度提升是核心加速来源 |

## 实验运行

以下命令在 `cann-samples` 仓库根目录执行。

### 环境准备

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
# 或：source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh
```

Python 侧需要 `numpy`：

```bash
pip install numpy
```

### 构建

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target simt_histogram_story
```

也可以只构建单个 case：

```bash
cmake --build build --target simt_histogram_0_serial_scalar
cmake --build build --target simt_histogram_1_basic_simt
cmake --build build --target simt_histogram_2_mem_optimized
cmake --build build --target simt_histogram_3_advanced_tuning
```

### 运行

```bash
./build/Samples/2_Performance/simt_histogram_story/simt_histogram_0_serial_scalar
./build/Samples/2_Performance/simt_histogram_story/simt_histogram_1_basic_simt
./build/Samples/2_Performance/simt_histogram_story/simt_histogram_2_mem_optimized
./build/Samples/2_Performance/simt_histogram_story/simt_histogram_3_advanced_tuning
```

每个可执行文件启动后会自动调用 `gen_data.py --output <exe_dir>` 生成输入和 golden，再执行 kernel 并校验输出。

## 什么时候使用这种模式

适合：

- 需要对大量独立元素做离散统计（直方图、计数、频率分布等），统计目标数量适中（UB 可容纳）
- 数据量较大，串行处理成为瓶颈，且输入元素之间无依赖关系，天然可并行
- 希望从传统 Ascend C（MTE+Vector）迁移到纯 SIMT 模型，获得简洁的 CUDA 风格编程体验

不适合：

- bins 数量远超 UB 容量（纯 SIMT 无法使用 MTE 批量搬运，频繁的 GM 原子操作代价极高）
- 数据规模极小，线程调度和线程间协调的开销可能超过计算收益
- 需要精细控制流水线（双缓冲、MTE/Scalar 重叠），此时 `__aicore__` + `__simt_vf__` 混合模型更合适

## 总结

| Case | 继承关系 | 新增优化 | 核心能力点 | 源文件 |
|:---:|:---|:---|:---|:---|
| 0 | — | MTE+Vector 串行基线 | 双缓冲流水、硬件事件同步、多核归约 | `src/0_serial_scalar.asc` |
| 1 | Case 0 | SIMT 并行 | `__global__` kernel、`__ubuf__[]`、GM 直接访问、`asc_atomic_add` UB 原子累加 | `src/1_basic_simt.asc` |
| 2 | Case 1 | float4 向量化 | 向量化读 GM、边界检测、msprof 定量验证 | `src/2_mem_optimized.asc` |
| 3 | Case 2 | launch_bounds(512) + GridDim=32 | `--cce-res-usage` 寄存器分析、GridDim 扫描、参数空间搜索 | `src/3_advanced_tuning.asc` |

处理 SIMT Histogram 的推荐思路：**先理解串行逻辑（Case 0）→ 掌握 SIMT 并行计数范式 `for (i=globalTid; i<N; i+=totalThreads)` 及 GM 直接访问 + `asc_atomic_add` 在 UB 处理冲突（Case 1）→ 用 msprof 数据判断 GM 访存优化是否有效（Case 2）→ 寄存器分析 + GridDim 扫描找到最优配置（Case 3）**。其中 Case 1 的 grid-stride 跨步循环是所有后续优化的基础框架，是迁移到 SIMT 最重要的一步。
