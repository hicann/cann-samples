# Softmax 算子的 RegBase 递进优化

这个样例用 Softmax（reduce + element-wise 混合算子）演示怎么从 MemBase 基线改到 RegBase VF，再沿"拆 reduce 成 2 条独立链"→ 大 tile 攻搬运 → 合并 VF 的主线逐步优化。

当前 8 个 Case（0–7）：Case 0 是 MemBase 基线，Case 1 改成 RegBase 三趟 VF，Case 2 做单行 binary fold（chunk 轴），Case 3 做多行并行（row 轴），Case 4 试 MTE2↔V 流水（未生效），Case 5 做大 tile 攻搬运，Case 6/7 试合并 VF（Case 7 去掉 asc_vf_call 直接调用，综合最优）。每个 Case 都是独立的可执行文件，可以单独编译、运行、对比。

## 支持范围

| 项目 | 说明 |
|:---|:---|
| 硬件 / 架构 | Ascend 950PR / 950DT，`NPU_ARCH=dav-3510`（A5 RegBase） |
| CANN 版本 | 社区版 CANN Toolkit 9.1.0 及以上 |
| 数据类型 | `float32` |
| 输入 / 输出 shape | `x[256, 2048]` → `y[256, 2048]`（对应 `include/sample_common.h` 里的常量） |
| 并行 | 32 AI Core，每核处理 `8×2048` |

几点限制：

- 不支持 `dav-2201`（910B），CMake 配置阶段会自动跳过这个样例。
- 只验证了 `256×2048` 这一个 shape，换 shape 需要自己改 `TOTAL_M/N`、`SINGLE_CORE_*`、`TILE_LEN` 再验证。
- 固定 `axis=-1`（沿最后一维归约）。ops-nn 仓的 `softmax_v2` 算子支持任意 axis，通过 tiling 参数指定，本样例为简化只覆盖最后一维的情况。
- 性能数据和测试环境绑定，换硬件 / CANN 版本 / shape 都要重新测。

## 目录结构

```text
softmax_regbase_story/
├── CMakeLists.txt          # 构建 8 个可执行目标；Case 0 关掉 VF 融合，其余打开
├── README.md
├── include/
│   └── sample_common.h     # Host 侧：ACL 初始化、调用 gen_data、golden 比对、RunSample 入口
├── scripts/
│   └── gen_data.py         # 生成 input/input_x.bin 和 output/golden.bin
└── src/
    ├── 0_membase.asc       # Case 0：MemBase 基线
    ├── 1_vf_fused.asc      # Case 1：RegBase 三趟 VF
    ├── 2_binary_fold.asc   # Case 2：binary fold
    ├── 3_multi_row.asc     # Case 3：多行并行
    ├── 4_pipeline.asc      # Case 4：流水 prefetch
    ├── 5_bigtile.asc       # Case 5：大 tile
    ├── 6_merged_vf.asc     # Case 6：合并 VF
    └── 7_merged_vf_direct.asc # Case 7：直接调用合并 VF  ← 综合最优
```

每个 `src/*.asc` 都是完整的 Kernel + `Process` + VF 逻辑，可以按 Case 单独看；Host 侧公共代码都在 `sample_common.h`。

## Cases

计算的是：

$$
y_i = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}
$$

| Case | 源文件 | 可执行目标 | 做了什么 |
|:---:|:---|:---|:---|
| 0 | `src/0_membase.asc` | `softmax_regbase_0_membase` | MemBase 基线，`Compute` API + 每步 `PipeBarrier` |
| 1 | `src/1_vf_fused.asc` | `softmax_regbase_1_vf_fused` | 改成 RegBase，整行 Softmax 收进 VF，三趟计算 |
| 2 | `src/2_binary_fold.asc` | `softmax_regbase_2_binary_fold` | 单行 + binary fold |
| 3 | `src/3_multi_row.asc` | `softmax_regbase_3_multi_row` | 2 行并行 |
| 4 | `src/4_pipeline.asc` | `softmax_regbase_4_pipeline` | MTE2↔V 流水 prefetch（实验性，未生效）|
| 5 | `src/5_bigtile.asc` | `softmax_regbase_5_bigtile` | 大 tile |
| 6 | `src/6_merged_vf.asc` | `softmax_regbase_6_merged_vf` | 合并 VF |
| 7 | `src/7_merged_vf_direct.asc` | `softmax_regbase_7_merged_vf_direct` | 直接调用合并 VF — **综合最优** |

Case 1 → 2 → 3 是"把 reduce 拆成 2 条独立链"思想的演进（Case 2 chunk 轴 → Case 3 row 轴，两轴不可叠加）。
Case 3 之后搬运（mte2）成瓶颈：Case 5 用大 tile 切块搬运打破瓶颈；Case 4/6 试"打破 VF 屏障让 MTE2↔V 重叠"均未兑现性能耗时收益；Case 7 去掉 `asc_vf_call` 直接调用，优化 vector 耗时到最低(1.483)，dur 也降到最低(7.284)，vec 省下的计算确实兑现成了 dur 收益。能力关系：

| 能力 | Case 1 | Case 2 | Case 3 | Case 5 | Case 6 | Case 7 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| VF 融合 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Binary fold |  | ✓ |  |  |  |  |
| 多行并行 |  |  | ✓ | ✓ | ✓ | ✓ |
| 大 tile |  |  |  | ✓ | ✓ | ✓ |
| 合并 VF |  |  |  |  | ✓ | ✓ |
| 寄存器内广播 |  |  |  |  | ✓ | ✓ |
| 直接调用 |  |  |  |  |  | ✓ |

## 怎么验证

可执行文件启动后会自己生成数据、跑 Kernel、和 golden 比对，不用再单独跑 `verify_result.py`。

| 项目 | 说明 |
|:---|:---|
| 数据生成 | `scripts/gen_data.py`，`seed=42`，输入 `uniform(-3, 3)`，shape `[256, 2048]` |
| 比对容差 | 绝对误差 `1e-3`（`sample_common.h` 里的 `COMPARE_TOL`） |
| 失败时 | 打印前 20 个误差元素，进程返回非零 |
| 通过标志 | 终端打印 `[<case_name>] step <N> PASSED` |

数据默认写在可执行文件所在目录的 `input/`、`output/` 下（`RunSample` 会调 `gen_data.py --output <exe_dir>`）。想单独生成数据的话：

```bash
python3 Samples/2_Performance/softmax_regbase_story/scripts/gen_data.py \
  --output /tmp/softmax_regbase_data --seed 42
```

## 性能

下面的数据在这个环境下测的：

| 项目 | 值 |
|:---|:---|
| 硬件 / 架构 | Ascend 950PR，`dav-3510` |
| CANN | 9.2.0 |
| 输入 | `float32 [256, 2048]`，`seed=42` |
| 测量 | msprof，看 `task_time`（3 轮取均值） |

功能上，Case 0~7 在这个环境里精度都是 `PASSED`。各步实测 Task Time：

| Case | 可执行目标 | Task Time (µs) | vs 上一 Case |
|:---:|:---|---:|---:|
| 0 | `softmax_regbase_0_membase` | 10.637 | — |
| 1 | `softmax_regbase_1_vf_fused` | 8.584 | -19.3% ✓ |
| 2 | `softmax_regbase_2_binary_fold` | 8.266 | -3.7% ✓（单行，方差大）|
| 3 | `softmax_regbase_3_multi_row` | 7.747 | -6.3% ✓ |
| 4 | `softmax_regbase_4_pipeline` | 7.689 | -0.8% ✗ 流水未生效（`asc_vf_call` 屏障）|
| 5 | `softmax_regbase_5_bigtile` | 7.476 | -2.8% ✓ 大 tile |
| 6 | `softmax_regbase_6_merged_vf` | 7.492 | +0.2% 持平 |
| 7 | `softmax_regbase_7_merged_vf_direct` | **7.284** | -2.8% ✓ **vec 最低且 dur 最低** |

各步优化的预期与实测对照：

| 阶段 | 优化 | 预期 | 实测 |
|:---:|:---|:---|:---|
| 0 → 1 | VF 融合 | UB 往返变少 | -19.3% ✓ |
| 1 → 2 | Binary fold（单行，chunk 轴） | 串行步数 8→5 | -3.7% ✓（方差大）|
| 2 → 3 | 2 行并行（row 轴） | 砍 merge + 摊销 VF 调用 | -6.3% ✓ |
| 2 + 3 | 2 行之上叠 binary fold | 4 链进一步藏延迟 | +5% ✗ 回退（双发已饱和） |
| 3 → 4 | MTE2↔V 流水（prefetch） | 搬运与计算重叠 | -0.8% ✗ 未生效（`asc_vf_call` 屏障）|
| 3 → 5 | 大 tile（tileRow 2→4） | 大 DMA 吃满带宽 | -2.8% ✓ |
| 5 → 6 | 合并 VF（asc_vf_call×1，重算） | 少屏障 + 重叠 | dur 持平（+0.2%），vec +0.08（重算代价），重叠未发生 |
| 5 → 7 | 直接调用合并 VF（去 asc_vf_call） | LocalMemBar 生效→vec 归零 | vec -0.15（全场最低 1.483），dur -2.8%（7.476→7.284）|

Case 5（大 tile）压下搬运后，Case 6/7 继续优化计算侧：Case 6 合并 VF 但屏障时间守恒、dur 持平；
Case 7 去掉 `asc_vf_call` 直接调用，vec 压到全场最低 1.483（比 Case 5 的 1.635 还低 0.15），dur
继续下降到 7.284（**全场最低**）。Binary fold（Case 2）只在**单行**场景成立：N=2048（8 chunk）
下串行步数 8→5；但它**不能**叠在 Case 3（2 行）之上——2 行已把双发打满，再 even/odd 拆 4 链
无延迟可藏，实测 +5% 回退。同一思想搬到 row 轴就是 Case 3（砍 merge + 摊销）。Case 3 之后搬运
（mte2）成瓶颈，Case 5 用大 tile 切块搬运（mte2 −33%）。Case 6 合并 3 个 VF 为 1 个（经
`asc_vf_call`），但屏障时间守恒（1 长 VF ≈ 3 短 VF 总阻塞），重叠未发生，且 LocalMemBar 在
`asc_vf_call` 内死锁、被迫重算 Sub/Exp，多付 +0.08 vec；Case 7 去掉 `asc_vf_call` 直接调用
（CANN 官方模式），LocalMemBar 生效、无需重算，vec 降到 1.483，dur 也降到 7.284——vec 省下的
0.15 确实兑现成了 dur 收益。

### 各引擎耗时拆解（msprof `op_summary`，3 轮均值）

把每个 Case 的 kernel task 按执行引擎拆开（AIV = vector core；vec = 矢量计算，scalar = 标量/控制，
mte2 = GM→UB 搬入，mte3 = UB→GM 搬出）：

| Case | Task Duration (µs) | total_cycles | vec (占比) | scalar (占比) | mte2 搬入 (占比) | mte3 搬出 (占比) |
|:---:|---:|---:|---:|---:|---:|---:|
| 0 membase | 10.637 | 482380 | 3.286 (30%) | 2.138 (20%) | 2.930 (27%) | 1.091 (10%) |
| 1 vf_fused | 8.584 | 373029 | 2.823 (32%) | 1.221 (14%) | 3.002 (34%) | 1.145 (13%) |
| 2 binary_fold | 8.266 | 356495 | 2.338 (28%) | 1.275 (15%) | 2.903 (35%) | 1.134 (13%) |
| 3 multi_row | 7.747 | 331278 | 1.673 (21%) | 1.079 (13%) | 3.015 (38%) | 1.146 (14%) |
| 4 pipeline | 7.689 | 326414 | 1.672 (21%) | 0.988 (12%) | 2.963 (38%) | 1.163 (15%) |
| 5 bigtile | 7.476 | 321490 | 1.635 (21%) | 1.024 (13%) | 2.003 (26%) | 0.822 (10%) |
| 6 merged_vf | 7.492 | 326718 | 1.717 (22%) | 0.968 (12%) | 2.045 (27%) | 0.777 (10%) |
| 7 merged_vf_direct | 7.284 | 312953 | 1.483 (20%) | 1.000 (13%) | 2.041 (28%) | 0.800 (10%) |

> Case 0–4 里 mte2/mte3（GM↔UB 搬运）基本不变（≈4.1–4.3 µs）——看着像"不可压缩的搬运底座"，
> 其实是小 DMA（16KB）的 ramp-up 损耗。Case 5 加大 DMA（32KB）后 mte2 3.00→2.00（−33%）、
> mte3 1.15→0.82（−29%），才真正砍到搬运。计算（vec+scalar）随优化 5.42→4.04→3.61→2.76→2.66→2.66→2.69→2.48 µs。
> **Case 7 的 vec 1.483 是全场最低**（去 `asc_vf_call` 后 3 趟计算连续执行无屏障打断），且 dur 7.284
> 也是全场最低——vec 省下的 0.15 确实兑现成了 dur 收益，而非全变 idle。

### 瓶颈 → 优化 → 落地验证

**Case 0（MemBase 基线）— 计算瓶颈 + 标量开销高**
- 现状：vec+scalar 5.42 µs（50%）> 搬运 4.02 µs；其中 **scalar 占 20%**。MemBase 模式每步 softmax
  子算子都回写 UB、下一步再读，每步夹 `PipeBarrier` + 地址计算 → 标量开销 + UB 往返。
- 瓶颈：标量开销 + 中间 UB 往返（不是算不动，是被搬运/控制拖住）。

**0 → 1（VF 融合）— 干掉标量开销**
- 手段：整行 Softmax 收进一个 `__simd_vf__`，`Load` 一次 → 寄存器内连续算（Sub/Exp/Div）→ `Store`
  一次，中间结果不回 UB，去掉每步 `PipeBarrier`。
- 验证：scalar 2.138 → 1.221（**−43%**），total_cycles 482380 → 373029（**−23%**），duration
  10.637 → 8.584（**−19%**）。搬运不变。此时计算 4.04 ≈ 搬运 4.15，两者平衡。

**1 → 2（binary fold，单行）— 砍 reduce 串行延迟**
- 现状（Case 1）：vec 升到 33%，成为最大计算块——reduce 的 `Max(maxReg, maxReg, x)` 8 步串行
  依赖，latency-bound，单行没有第二条链藏延迟。
- 手段：一行内按 even/odd 拆 2 条独立累加链，关键路径 8→5，双发把 reduce 延迟叠掉。
- 验证：vec 2.823 → 2.338（**−17%**），duration 8.584 → 8.266（**−3.7%**，方差大）。此时
  搬运 4.04 > 计算 3.61，**瓶颈开始转到搬运**。

**2 → 3（多行并行，row 轴）— 砍 merge + 摊销 VF 调用**
- 现状（Case 2）：vec 仍有 27%，且每行 3 次 `asc_vf_call`（单行只摊 1 行）。
- 手段：把 2 链思想从 chunk 轴搬到 row 轴（row0/row1），两行本就独立 → 砍掉 even/odd 合并那拍，且
  3 次 `asc_vf_call` 一次处理 2 行（摊销翻倍）。
- 验证：vec 2.338 → 1.673（**−28%**），scalar 1.275 → 1.079，duration 8.266 → 7.747（**−6.3%**）。
  此时 **搬运（mte2 38%）已 > 计算（2.75）**，kernel 转为搬运瓶颈，继续优化计算收益递减。

**3 → 5（大 tile）— 攻搬运：加大 DMA 吃满带宽**
- 现状（Case 3）：mte2 占 39% 是最大头。诊断发现 Case 0–3 的 mte2/mte3 一直 ≈4.1–4.3µs 不降，不是
  "搬运不可压缩"，而是 16KB 小 DMA 的 ramp-up 损耗没吃满 HBM 带宽。
- 手段：`tileLen` 翻倍（tileRow 2→4，ping-pong 不变），每次 DMA 16KB→32KB。
- 验证：mte2 3.02→2.00（**−34%**），mte3 1.15→0.82（**−29%**），duration 7.75→7.48（**−3.5%**），
  搬运瓶颈被打破。但四引擎占比和从 0.89 掉到 0.73——DMA 省下的时间变 idle（serial，没法填进 compute）。
- **tileRow 调参**：在 Case 5 框架内测了 tileRow 4/6/7/8。**tileRow 4 最优**——ping-pong 重叠（需 ≥2
  tile）比 DMA 尺寸更重要：tileRow 8 单缓冲有最大 DMA(64KB) 却最慢（丢了重叠）；tileRow 7 尾 1 行摊销差。
  加大 tile 非新方向，4 行已触顶。

**Case 4（MTE2↔V 流水 prefetch）— 想填 idle 没成**：把 load(t+1) 提前到 compute(t) 之前发，试图让
搬运与计算重叠填掉那 27% idle。但 `asc_vf_call` 是屏障，VF 调用期间 MTE2 被阻塞，load 还是没法跟
compute 重叠，实测 dur 不动，还多付 prologue 开销。

**Case 6（合并 VF，经 asc_vf_call）— 屏障时间守恒**：把 3 次 `asc_vf_call` 合成 1 个大 VF，max/sum
标量用 `Reg::Duplicate(dst,src,mask)` 寄存器内广播绕开 UB。但 **(a)** `LocalMemBar` 在 `asc_vf_call`
内死锁，被迫重算 Sub/Exp（+0.08 vec 代价）；**(b)** 屏障时间守恒——1 长 VF ≈ 3 短 VF 总阻塞，MTE2
仍被挡整段，重叠未发生。dur +0.2% 与 Case 5 持平，vec 反升。

**Case 7（直接调用合并 VF，去 asc_vf_call）— vec 最低，dur 最低**：照搬 CANN
`SoftMaxGenericNDVFImpl` 模式，`__no_simd_vf_fusion__` + 直接调用（不经 `asc_vf_call`），`LocalMemBar`
生效、无需重算。vec 压到**全场最低 1.483**（比 Case 5 的 1.635 还低 0.15——3 趟计算连续执行无屏障
打断）。dur 继续下降到 **7.284**（全场最低）：省下的 0.15 vec 确实兑现成了 dur 收益，而非全变 idle
（四引擎占比和 0.73→0.73 基本不变）。从 Case 0 的 vec 3.3 一路压到 1.483，dur 从 10.6 降到 7.3。

**小结**：优化主线是瓶颈迁移 scalar → vec(reduce) → mte2(搬运) → idle。到 Case 5 搬运压下、27% idle
成新焦点；Case 4/6 试"打破 VF 屏障让 MTE2↔V 重叠"未兑现 dur 收益——Case 7 去掉 `asc_vf_call` 直接调用，
把 vec 砍到最低 1.483，dur 也降到最低 7.284，vec 省下的计算确实兑现成了 dur 收益。剩余的 ~27% idle
要靠 MTE2 与 VF 真并行（VF 期间发 MTE2）才能填，属于 `asc_vf_call` 模型之外的 API 能力，不是
tiling/重组/合并能解的。

---

以下几节按 Case 顺序讲原理和改法，可以配合源码一起看。

## 第一节　RegBase 为什么更快

MemBase 模式下，矢量指令的操作数都在 UB 上，一步计算就是「从 UB 读 → 算 → 写回 UB」。Softmax 由多步串起来时，每一步的中间结果都要写回 UB、下一步再读出来：

```
读x → ReduceMax → 写UB → 读UB → Sub → 写UB → 读UB → Exp → 写UB → …
```

这样有几个问题：中间结果反复占 UB 读写带宽；UB Bank 冲突变多；单步计算很短时，整体耗时被访存拖住，算子退化成搬运瓶颈。

RegBase 换个做法：把一段连续计算放到寄存器里做，只在进出这段计算时各碰一次 UB，中间结果一直留在寄存器里传：

```
UB → Load → [ ReduceMax → Sub → Exp → ReduceSum → Div → … ] → Store → UB
```

|  | MemBase | RegBase |
|:---|:---|:---|
| 数据放哪 | UB | 寄存器 |
| 中间结果 | 每步回写 UB | 寄存器里连续传递 |
| 接口 | `AscendC::ReduceMax/Exp/Muls` | `AscendC::Reg::ReduceMax/Sub/Exp/Div` |
| 好处 | — | 省掉中间 Load/Store，还能做 VF 融合和双发 |

Softmax 是典型的 reduce + element-wise 混合算子，中间结果（max、exp 值）反复读写 UB，非常适合改 RegBase。由于每行需独立求 max/sum 再归一化，**必须逐行处理**。

对应代码看 `src/0_membase.asc` 的 `ComputeMemBase` 和 `Process`。先把基线跑起来：

```bash
cmake --build build --target softmax_regbase_0_membase
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_0_membase
```

通过时会打印 `[0_membase] step 0 PASSED`，后面所有优化都拿它当对照。

**ComputeMemBase（单行）**：
```
ReduceMax(max, x, work, n) → 读标量 max → Adds(x, x, -max, n) → Exp(y, x, n) → ReduceSum(sum, y, work, n) → 读标量 sum → Muls(y, y, 1/sum, n)
```

## 第二节　RegBase 的代码骨架

RegBase 代码分两层，几乎所有 RegBase 算子都是这个结构：

- `__aicore__` 侧：做数据切分、GM↔UB 搬运，通过 `asc_vf_call` 调 VF；
- `__simd_vf__` 侧（Vector Function，简称 VF）：做 `Load → 计算 → Store`，计算用 `AscendC::Reg::*`，数据落在 `RegTensor` 这类寄存器上。

```cpp
// __aicore__ 侧：准备地址和循环参数，调 VF
__aicore__ inline void ComputeRegBase(LocalTensor<float>& x, LocalTensor<float>& y, uint32_t n) {
    __ubuf__ float* xAddr = (__ubuf__ float*)x.GetPhyAddr();
    __ubuf__ float* yAddr = (__ubuf__ float*)y.GetPhyAddr();
    uint16_t oneRepeat = AscendC::GetVecLen() / sizeof(float);
    uint16_t loopNum   = AscendC::CeilDivision(n, oneRepeat);
    asc_vf_call<YourVf>(xAddr, yAddr, n, loopNum, oneRepeat);
}

// __simd_vf__ 侧：Load → 计算 → Store
__simd_vf__ inline static void YourVf(__ubuf__ float* x, __ubuf__ float* y,
    uint32_t n, uint16_t loopNum, uint16_t oneRepeat) {
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::RegTensor<float> xReg, yReg;
    for (uint16_t i = 0; i < loopNum; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(n);
        AscendC::Reg::LoadAlign(xReg, x + i * oneRepeat);
        // … 用 Reg:: 接口写计算链 …
        AscendC::Reg::StoreAlign(y + i * oneRepeat, yReg, mask);
    }
}
```

几个必须认识的类型：`RegTensor`（矢量寄存器）、`MaskReg`（尾块掩码）、`LoadAlign / StoreAlign`（UB 和寄存器之间搬数据）。架构编成 `dav-3510` 就行，CMake 里已经配好架构和 VF 融合开关。

这一节只看骨架，不单独编一个 target，具体代码结合 `src/1_vf_fused.asc` 的 `ComputeRegBase` 和 `asc_vf_call` 看，下一节就会把 Case 1 跑起来。

## 第三节　RegBase 三趟 VF（Case 1）

把整行 Softmax 收进 `__simd_vf__` 函数，element-wise 部分（Sub、Exp、Div）全在寄存器里做，reduce 部分（Max、Sum）也在寄存器内完成。

**三趟计算**（每行 3 个 `asc_vf_call`）：

```
VfReduceMax:  Duplicate(maxReg, -INF) → Load+Max 跨 chunk 累积 → ReduceMax → StoreUnAlign 标量落 UB
VfSubExpSum:  LoadAlign<DIST_BRC_B32> 广播 max → Load+Sub+Exp+Store → Add 累积 sum → ReduceSum → StoreUnAlign 标量落 UB
VfDiv:        LoadAlign<DIST_BRC_B32> 广播 sum → Load+Div+Store
```

**关键点**：reduce 产生的标量须经 UB 做一次 `DIST_BRC_B32` 广播才能被所有 lane 使用。store 和 broadcast load **必须分属不同的 `asc_vf_call`**——同一 VF 函数内 store 再 load 同一 UB 地址无可见性保证，标量会读到旧值（实测输出全 inf）。

这是 reduce + element-wise 混合算子在 RegBase 下的固有特征：element-wise 部分受益于 VF 融合（省掉中间 UB 往返），但 reduce 标量必须经 UB 广播回去。

代码看 `src/1_vf_fused.asc`。跑一下，和 Case 0 对比：

```bash
cmake --build build --target softmax_regbase_0_membase
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_0_membase
cmake --build build --target softmax_regbase_1_vf_fused
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_1_vf_fused
```

两个都应该 `PASSED`。

## 第四节　Binary fold（Case 2，chunk 轴）—— 2 链思想起源

这一节是"把 reduce 拆成 2 条独立链"思想的**起源**（chunk 轴）。它本身不是最优——第五节
的 row 轴（Case 3）是它的演进版、且更优；放这里是为了讲清血缘，避免把两条轴叠错。

**问题**：Case 1 单行时 reduce 累积是串行链，每一步依赖上一步（latency-bound），没有
第二条链帮忙藏延迟：
```
maxReg = Max(maxReg, chunk0) → Max(maxReg, chunk1) → ... → Max(maxReg, chunk7)
                                  8 步串行依赖
```

**优化（chunk 轴）**：把一行的 8 个 chunk 按 even/odd 拆成 2 条独立累加链，各自累积一半，
末尾再 even/odd 合并 1 次。两条链互不依赖，硬件双发把它们叠掉，关键路径 8→5：
```
maxEven = Max(chunk0) → Max(chunk2) → Max(chunk4) → Max(chunk6)   ← 4 步，独立于 odd
maxOdd  = Max(chunk1) → Max(chunk3) → Max(chunk5) → Max(chunk7)   ← 4 步，独立于 even
maxReg  = Max(maxEven, maxOdd)                                    ← 1 步合并
```

sum 同理（pairwise summation，精度更高）。max 满足结合/交换律，精度无损。

**收益与边界（实测）**：单行场景下 vs Case 1 收益 **-2.1%**（串行步数 8→5，方差大）。但有几个边界：
- 它**不能**叠在第五节 Case 3（2 行）之上。2 行已经提供了 2 条独立链、把双发打满；
  再每行 even/odd 拆成 4 链，第 3/4 条链无延迟可藏，反而多付合并那拍 → 实测 +5% 回退。
- 因此本 Case 现为**单行 + binary fold**（每行 3 次 `asc_vf_call`），不是 2 行。
- 同一思想搬到 row 轴（直接拿 row0/row1 当 2 链）就能砍掉合并那拍 + 把 VF 调用摊到
  2 行，这就是第五节 Case 3，严格强于本 Case。

参考 `simd_vf_story/reduce/reduce_sum_ar_binary` 的二分折叠写法。

代码看 `src/2_binary_fold.asc`（单行版）。跑一下：

```bash
cmake --build build --target softmax_regbase_2_binary_fold
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_2_binary_fold
```

输出 `[2_binary_fold] step 2 PASSED`。**它不是最优**，计算侧最优是第五节 Case 3；本节意在讲清
"2 条独立链"思想及其 chunk 轴 / row 轴两条实现。

## 第五节　多行并行（Case 3，row 轴）—— 计算侧最优

**问题**：Case 1 单行逐行处理，reduce 的 `Max(maxReg, maxReg, xReg)` 每轮依赖上一轮的
`maxReg`，必须串行等。Case 2（binary fold）用 chunk 轴 even/odd 拆 2 链藏住了一部分延迟，
但末尾还要 even/odd 合并 1 拍，且每行仍要 3 次 `asc_vf_call`。

**优化（row 轴）**：2 组独立累加器（`maxReg0/maxReg1`、`sumReg0/sumReg1`）并行折叠 2 行。
row0 的 `Max` 和 row1 的 `Max` 互不依赖，硬件可以双发，把 reduce 延迟藏在另一行的 Load/Max
后面。相比 Case 2 的 chunk 轴：两行本就独立，**砍掉 even/odd 合并那拍**；且 3 次
`asc_vf_call` 一次处理 2 行（binary fold 只摊 1 行），**VF 调用摊销翻倍**。

```
Case 1（逐行）:                   Case 3（2 行并行）:
VF: row0 ReduceMax                VF: row0+1 ReduceMax
VF: row0 SubExpSum                  Load x0, Load x1        ← 独立
VF: row0 Div                        Max(max0, x0)           ← 依赖 max0
VF: row1 ReduceMax                   Max(max1, x1)           ← 依赖 max1, 与上面独立
VF: row1 SubExpSum                   ↑ 两行互不依赖 → 双发
VF: row1 Div
6 次 asc_vf_call                  3 次 asc_vf_call
```

参考了 `simd_vf_story/reduce/reduce_max_ar_unroll` 的多累加器展开模式。

代码看 `src/3_multi_row.asc`。跑一下，和 Case 2 对比：

```bash
cmake --build build --target softmax_regbase_3_multi_row
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_3_multi_row
```

输出 `[3_multi_row] step 3 PASSED`。这是 2 链思想的计算侧最优；综合最优是 Case 7（去掉 `asc_vf_call` 直接调用）。

本 Case 是第四节 binary fold（chunk 轴）的 **row 轴演进**：把"拆 reduce 成 2 条独立链"的
思想从 chunk 轴（even/odd）搬到 row 轴（row0/row1），因两行本就独立可砍掉 even/odd 合并那拍，
且 3 次 `asc_vf_call` 一次处理 2 行（binary fold 只摊 1 行），VF 调用摊销翻倍，故严格强于
binary fold。两轴不可叠加（见第四节边界）。

## 第六节　MTE2↔V 流水 prefetch（Case 4）—— 实验性，未生效

**问题**：Case 3 之后搬运（mte2）占 39% 成最大头，且四引擎占比和≈0.88（基本串行、
无重叠），有约 12% idle。原因是 `Process` 里每个 tile 按 load→compute→store 串行，
load(t+1) 的指令要等 store(t) 之后才发，MTE2 没机会与 V 重叠。

**优化（prefetch）**：prologue 先 load(0)，循环里先 prefetch load(t+1) 再 compute(t)，
让 load(t+1) 与 compute(t) 在时间上重叠。依赖关系不变（event-flag 照搬 Case 3），
只是重排指令发射顺序：

```
Case 3（串行）:              Case 4（prefetch）:
  load(t)                       load(0)              ← prologue
  compute(t)                    load(t+1)            ← prefetch，试图与下面的 compute 重叠
  store(t)                      compute(t)
  load(t+1)  ← 等 store 完      store(t)
  compute(t+1)                  ...
```

**为什么没成**：`asc_vf_call` 是屏障——VF 调用期间 MTE2 引擎被阻塞，load(t+1) 的
`DataCopyPad` 虽然提前发了指令，但实际执行仍要等 VF 结束。实测 dur 不动（-0.2%），
还多付 prologue 开销。这条路的启示是：只要 VF 经 `asc_vf_call` 调用，MTE2 与 V 就
没法真并行——后面的 Case 6/7 会再从"合并/去掉 `asc_vf_call`"的角度试一次。

代码看 `src/4_pipeline.asc` 的 `Process`（关注 prologue load 和循环里的 prefetch 顺序）。
跑一下，和 Case 3 对比：

```bash
cmake --build build --target softmax_regbase_4_pipeline
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_4_pipeline
```

输出 `[4_pipeline] step 4 PASSED`。流水未生效——这里不攻搬运本身，而是想藏搬运延迟，
但被 `asc_vf_call` 屏障挡住了。下一步（Case 5）换思路：不藏延迟，直接加大 tile 切块搬运。

## 第七节　大 tile（Case 5）—— 打破搬运瓶颈

**问题**：Case 0–3 的 mte2/mte3（GM↔UB 搬运）一直 ≈4.1–4.3 µs 不降，看着像"不可压缩
的搬运底座"。诊断发现不是不可压缩，而是 16KB 小 DMA 的 ramp-up 损耗没吃满 HBM 带宽。
Case 4 试流水藏延迟没成，这里换思路——不藏延迟，直接攻搬运本身。

**优化（大 tile）**：`tileLen` 翻倍（tileRow 2→4，ping-pong 不变），每次 DMA 16KB→32KB。
代码改动极小——VF 逻辑完全照搬 Case 3，只是模板参数从 `TILE_LEN` 改成 `2 * TILE_LEN`：

```cpp
// Case 3
using KernelOp = KernelSoftmaxMultiRow<..., SoftmaxRegbaseSample::TILE_LEN>;
// Case 5
using KernelOp = KernelSoftmaxBigTile<..., 2 * SoftmaxRegbaseSample::TILE_LEN>;
```

**验证**：mte2 3.02→2.00（−34%），mte3 1.15→0.82（−29%），duration 7.75→7.48（−3.5%），
打破搬运瓶颈。但四引擎占比和从 0.89 掉到 0.73——DMA 省下的时间变 idle（serial，没法
填进 compute），成为新的焦点（后面 Case 6/7 试图打破 VF 屏障填这个 idle）。

**tileRow 调参**：在 Case 5 框架内测了 tileRow 4/6/7/8。**tileRow 4 最优**——ping-pong
重叠（需 ≥2 tile）比 DMA 尺寸更重要：tileRow 8 单缓冲有最大 DMA(64KB) 却最慢（丢了重叠）；
tileRow 7 尾 1 行摊销差。加大 tile 非新方向，4 行已触顶。

代码看 `src/5_bigtile.asc`。跑一下，和 Case 3 对比：

```bash
cmake --build build --target softmax_regbase_5_bigtile
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_5_bigtile
```

输出 `[5_bigtile] step 5 PASSED`。大 tile 压下了搬运瓶颈（mte2 −34%）。

## 第八节　合并 VF（Case 6，经 asc_vf_call）—— 屏障时间守恒

**问题**：Case 5 搬运压下了但 27% idle 仍在。idle 的根源是 `asc_vf_call` 屏障——VF 期间
MTE2 被阻塞。Case 5 拆 3 次 `asc_vf_call`（每行 3 趟 = 3 个屏障同步点），想合并成 1 个
大 VF 少屏障，让 MTE2 有更长窗口与 V 重叠。

**优化（寄存器内广播 + 合并 VF）**：Case 5 拆 3 次 VF 是因为 max/sum 标量要经 UB 广播
给下一趟（可见性约束，见第三节"关键点"）。CANN 官方 `SoftMaxGenericNDVFImpl` 证明可用
`Reg::Duplicate(dst, src, mask)` 做寄存器内广播绕开 UB——reduce 标量不落 UB、不经
`DIST_BRC_B32` 广播，直接在寄存器内复制给所有 lane。这样 3 趟计算收进 1 个 `__simd_vf__`，
经 1 次 `asc_vf_call` 调用：

```
Case 5（3 次 asc_vf_call）:     Case 6（1 次 asc_vf_call）:
  asc_vf_call<VfReduceMax2Row>    asc_vf_call<VfSoftmax2RowMerged>
  asc_vf_call<VfSubExpSum2Row>      ↳ ReduceMax → Duplicate 广播（不经 UB）
  asc_vf_call<VfDiv2Row>            ↳ Sub+Exp+Store+Add → ReduceSum → Duplicate 广播
                                     ↳ Sub+Exp+Div+Store（重算，见下）
3 次屏障                          1 次屏障
```

标量不再落 UB，`KernelSoftmaxMergedVf` 的 UB 布局也去掉了 `scalarBuf`。

**两个问题**：

**(a) LocalMemBar 死锁 → 重算**：合并后第 2 趟把 exp 结果存到 `yBuf`，第 3 趟需 reload
做 Div。但 `Reg::LocalMemBar`（保证 store→load 可见性）在 `asc_vf_call` 内死锁，被迫改用
**重算 Sub+Exp** 绕开——第 3 趟不 reload yBuf，而是重新从 `xAddr` Load+Sub+Exp，多付
+0.08 vec 代价。

**(b) 屏障时间守恒 → 重叠未发生**：3 次短 `asc_vf_call` 合成 1 次长 VF，但总阻塞时间不变
（1 长 VF ≈ 3 短 VF 总阻塞）。MTE2 仍被挡整段，重叠没发生。dur +0.2% 与 Case 5 持平，
vec 反升（重算代价）。Case 7 会从"去掉 `asc_vf_call`"的角度解这两个问题。

代码看 `src/6_merged_vf.asc` 的 `VfSoftmax2RowMerged`（关注 `Duplicate` 广播和第 3 趟重算）。
跑一下，和 Case 5 对比：

```bash
cmake --build build --target softmax_regbase_6_merged_vf
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_6_merged_vf
```

输出 `[6_merged_vf] step 6 PASSED`。

## 第九节　直接调用合并 VF（Case 7，去 asc_vf_call）—— vec 最低，dur 最低

**问题**：Case 6 的两个问题都源于 `asc_vf_call`：(a) `LocalMemBar` 在 `asc_vf_call` 内
死锁被迫重算；(b) 屏障时间守恒导致重叠不发生。

**优化（直接调用）**：照搬 CANN `SoftMaxGenericNDVFImpl` 模式——`__no_simd_vf_fusion__`
+ **直接调用**（不经 `asc_vf_call`）。去掉 `asc_vf_call` 后：

- `LocalMemBar` 生效——第 2 趟 exp 结果 store 到 `yBuf` 后，`LocalMemBar` 保证第 3 趟
  reload 可见，**无需重算**，vec 代价归零。
- 3 趟计算连续执行无屏障打断——vec 压到**全场最低 1.483**（比 Case 5 的 1.635 还低 0.15）。

```
Case 6（经 asc_vf_call）:         Case 7（直接调用）:
  asc_vf_call<VfSoftmax2RowMerged>  VfSoftmax2RowDirect(...)
    ↳ 第 3 趟：重算 Sub+Exp           ↳ 第 3 趟：LocalMemBar → reload yBuf → Div
    （+0.08 vec 代价）                （无重算，vec 代价归零）
```

**dur 继续下降**（7.284 < 7.476，Case 5）：省下的 0.15 vec 确实兑现成了 dur 收益，而非全变 idle
（四引擎占比和 0.73→0.73 基本不变）。从 Case 0 的 vec 3.3 一路压到 1.483，dur 从 10.6 降到 7.3。
剩余的 ~27% idle 要靠 MTE2 与 VF 真并行（VF 期间发 MTE2）才能填，属于 `asc_vf_call` 模型之外的
API 能力，不是 tiling/重组/合并能解的。

代码看 `src/7_merged_vf_direct.asc` 的 `VfSoftmax2RowDirect`（关注 `LocalMemBar` 和直接调用）。
跑一下，和 Case 5/6 对比：

```bash
cmake --build build --target softmax_regbase_7_merged_vf_direct
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_7_merged_vf_direct
```

输出 `[7_merged_vf_direct] step 7 PASSED`。

## 第十节　小结：写一个 reduce + element-wise 混合算子 RegBase 的步骤

1. 建立基线（Case 0）：先写一个功能正确的 MemBase 版本，用 `TPipe::InitBuffer` 分配 UB。
2. VF 融合（Case 1）：抽出 `__simd_vf__`，按 reduce + element-wise 的三趟结构拆成多个 `asc_vf_call`，标量经 UB 广播。
3. 2 链思想（Case 2，chunk 轴）：reduce 累积按 even/odd 拆 2 条独立链，串行步数 8→5——
   单行场景成立，是后续 row 轴演进的起源。
4. 多行并行（Case 3，row 轴）：把 2 链思想搬到 row 轴（row0/row1），砍掉合并 + 摊销 VF 调用，
   是计算侧最优。注意两轴不可叠加：2 行之上再 even/odd = 4 链，双发已饱和，实测回退。
5. 大 tile（Case 5）：搬运成瓶颈后，tileLen 翻倍用大 tile 切块搬运（mte2 −34%），打破搬运瓶颈。
   tileRow 4 即触顶——ping-pong 重叠比 DMA 尺寸更重要。
6. 去掉 asc_vf_call（Case 7）：Case 4（流水）/ Case 6（合并 VF）试让 MTE2↔V 重叠均未兑现 dur
   收益。Case 7 去掉 `asc_vf_call` 直接调用，vec 压到全场最低(1.483)，dur 也降到最低(7.284)——
   vec 省下的计算确实兑现成了 dur 收益。剩余 ~27% idle 要靠 MTE2 与 VF 真并行才能填，属
   `asc_vf_call` 模型外的 API 能力。

两点经验：每一步都用 msprof 确认收益；优化不是越多越好，遇到负优化就退回上一步。reduce +
element-wise 混合算子的最优并行度取决于 shape 和架构。

---

## 完整构建与运行

下面的命令都在 `cann-samples` 仓库根目录执行。

先配好 CANN 环境（Toolkit 路径按实际改）：

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
# 或 source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

生成数据要用到 numpy，装一下依赖：`pip install -r requirements.txt`。

构建：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target softmax_regbase_story        # 8 个 Case 一起编
cmake --build build --target softmax_regbase_0_membase    # 或只编一个
```

可执行文件在 `build/Samples/2_Performance/softmax_regbase_story/`，每个跑起来会自动生成数据、执行、比对。
默认用 device 7，可用环境变量 `SAMPLE_DEVICE_ID` 换卡（如 `SAMPLE_DEVICE_ID=5 ./...`）：

```bash
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_0_membase
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_1_vf_fused
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_2_binary_fold
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_3_multi_row
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_4_pipeline
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_5_bigtile
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_6_merged_vf
./build/Samples/2_Performance/softmax_regbase_story/softmax_regbase_7_merged_vf_direct
```

通过时大概是这样：

```text
output precision 100%, errors 0, max diff ...
[0_membase] step 0 PASSED
```

建议先把 Case 0 跑通，再按 Case 1~3、5、7 顺序对比 msprof（Case 7 是综合最优）。Case 4/6
是实验性分支（见性能节"瓶颈→优化→落地验证"及第六~八节），Case 7 虽是综合最优但原理较复杂，
建议先看 Case 5 再看 Case 7。

## 参考文档

- [VF融合优化](https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00026.html)
- [VF循环优化](https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00023.html)
- [指令双发优化](https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00024.html)
