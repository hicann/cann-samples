# GELU + Element-wise：RegBase 递进优化

本样例用一条 GELU + Element-wise 融合计算链，演示如何把 MemBase 写法改成 RegBase，再按阶段做性能优化。每个 Case 是独立可执行目标，可单独编译、运行和对照源码。

计算公式：

$$
y = \exp\big(-0.5 \cdot (\text{GELU}(x) + 1)^2 + 2\big)
$$

递进关系：

| Case | 源文件 | 可执行目标 | 内容 |
|:---:|:---|:---|:---|
| 0 | `src/0_membase.asc` | `gelu_eltwise_regbase_0_membase` | MemBase 基线 |
| 1 | `src/1_vf_fused.asc` | `gelu_eltwise_regbase_1_vf_fused` | RegBase + VF 融合 |
| 2 | `src/2_loop_split.asc` | `gelu_eltwise_regbase_2_loop_split` | 按依赖拆成两个 VF 循环 |
| 3 | `src/3_unroll.asc` | `gelu_eltwise_regbase_3_unroll` | `#pragma unroll 6` |
| 4 | `src/4_scalar_tuning.asc` | `gelu_eltwise_regbase_4_scalar_tuning` | 循环外 `Duplicate` 常量 |

后一个 Case 在前一个基础上叠加：Case 1 做 VF 融合，Case 2 加循环拆分，Case 3 加展开，Case 4 加常量外提。

## 支持范围

| 项目 | 说明 |
|:---|:---|
| 硬件 / 架构 | Ascend 950PR / 950DT，`NPU_ARCH=dav-3510` |
| CANN 版本 | 社区版 CANN Toolkit 9.1.0 及以上 |
| 数据类型 | `float32` |
| 输入 / 输出 | `x[8192, 8192]` → `y[8192, 8192]` |
| 并行 | 32 AI Core，每核 `256×4096` |

## 目录结构

```text
gelu_eltwise_regbase_story/
├── CMakeLists.txt
├── README.md
├── include/sample_common.h   # Host：ACL 初始化、gen_data、golden 比对
├── scripts/gen_data.py
└── src/
    ├── 0_membase.asc
    ├── 1_vf_fused.asc
    ├── 2_loop_split.asc
    ├── 3_unroll.asc
    └── 4_scalar_tuning.asc
```

每个 `src/*.asc` 含完整 Kernel 与 VF 逻辑；Host 公共代码在 `sample_common.h`。

## 编译与运行

在 `cann-samples` 仓库根目录执行：

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
# 或：source /usr/local/Ascend/ascend-toolkit/set_env.sh

pip install -r requirements.txt

cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target gelu_eltwise_regbase_story
```

也可只编单个 Case，例如：

```bash
cmake --build build --target gelu_eltwise_regbase_0_membase
```

可执行文件在 `build/Samples/2_Performance/gelu_eltwise_regbase_story/`。启动后会自动生成数据、执行 Kernel 并与 golden 比对（容差 `1e-3`），通过时输出类似：

```text
output precision 100%, errors 0, max diff ...
[0_membase] step 0 PASSED
```

建议先跑通 Case 0，再按 Case 1~4 对照源码和 msprof。

```bash
./build/Samples/2_Performance/gelu_eltwise_regbase_story/gelu_eltwise_regbase_0_membase
./build/Samples/2_Performance/gelu_eltwise_regbase_story/gelu_eltwise_regbase_1_vf_fused
./build/Samples/2_Performance/gelu_eltwise_regbase_story/gelu_eltwise_regbase_2_loop_split
./build/Samples/2_Performance/gelu_eltwise_regbase_story/gelu_eltwise_regbase_3_unroll
./build/Samples/2_Performance/gelu_eltwise_regbase_story/gelu_eltwise_regbase_4_scalar_tuning
```

---

## Case 0：MemBase 基线

源文件：`src/0_membase.asc`

MemBase 里，矢量指令直接在 UB 的 `LocalTensor` 上算：每条指令读 UB、算完再写回 UB。多步串起来时，上一步的结果先写回 UB，下一步再读出来：

```text
读x → 算 → 写UB → 读UB → 算 → 写UB → …
```

中间结果在 UB 上翻来覆去读写，带宽和 Bank 冲突都会变差；如果单步本身很短，整段计算很容易被访存拖慢。

代码看 `ComputeMemBase`：一步步 `Mul` / `Exp` / `Div`，中间用 `PipeBarrier<PIPE_V>()` 同步。

```bash
cmake --build build --target gelu_eltwise_regbase_0_membase
./build/Samples/2_Performance/gelu_eltwise_regbase_story/gelu_eltwise_regbase_0_membase
```

预期输出 `[0_membase] step 0 PASSED`。后面几个 Case 都拿它当对照。

## Case 1：RegBase + VF 融合

源文件：`src/1_vf_fused.asc`

RegBase 换一种做法：把一段连续计算放到寄存器里做。VF 开头从 UB `Load` 一次输入，中间结果在寄存器里往下传，最后再 `Store` 回 UB：

```text
UB → Load → [算 → 算 → …] → Store → UB
```

代码分两层：

- `__aicore__`：做切分和 GM↔UB 搬运，用 `asc_vf_call` 调 VF
- `__simd_vf__`：`LoadAlign` → `Reg::*` 计算 → `StoreAlign`

把 MemBase 的 `Compute` 逐行改成 `Reg::` 接口，整条计算收进一个 VF。中间结果不用每步写回 UB，逐步 `PipeBarrier` 也可以去掉：

```cpp
AscendC::Reg::LoadAlign(xReg, xAddr + i * oneRepeat);
AscendC::Reg::Mul(yReg, xReg, xReg, mask);
// … 中间结果留在寄存器 …
AscendC::Reg::StoreAlign(yAddr + i * oneRepeat, yReg, mask);
```

重点看 `ComputeRegBase`、`asc_vf_call` 和 `VfSingleFused`。

```bash
cmake --build build --target gelu_eltwise_regbase_1_vf_fused
./build/Samples/2_Performance/gelu_eltwise_regbase_story/gelu_eltwise_regbase_1_vf_fused
```

预期输出 `[1_vf_fused] step 1 PASSED`。和 Case 0 比，msprof 里的 `aiv_vec_ratio` 一般会更高。

## Case 2：拆分 VF 循环

源文件：`src/2_loop_split.asc`

硬件一个周期可以发两条互不依赖的指令。如果整条计算都塞在一个循环里，依赖链太长，双发用不上。按依赖拆成两段循环：第一段算完写回 UB，第二段再从 UB 读出来继续算：

```text
循环 A（GELU，8 条）：x → GELU(x) → 写回 UB
循环 B（Elt，5 条）：从 UB 读入 → exp(...) → 写回 UB
```

看 `VfLoopSplit` 里的双循环。

```bash
cmake --build build --target gelu_eltwise_regbase_2_loop_split
./build/Samples/2_Performance/gelu_eltwise_regbase_story/gelu_eltwise_regbase_2_loop_split
```

预期输出 `[2_loop_split] step 2 PASSED`。有没有收益，用 msprof 和 Case 1 对比即可。

## Case 3：循环展开

源文件：`src/3_unroll.asc`

在 Case 2 的两个 `for` 前加上 `#pragma unroll 6`，让编译器把循环体展开，一次迭代里多放出几组互不依赖的指令，方便双发。循环要满足 Hardware Loop：迭代变量用 `uint16_t`、从 0 递增、循环里没有 `if/else`。展开份数可以试 4 / 6 / 8，太大可能把寄存器撑爆。

看 `VfLoopSplitUnroll`。

```bash
cmake --build build --target gelu_eltwise_regbase_3_unroll
./build/Samples/2_Performance/gelu_eltwise_regbase_story/gelu_eltwise_regbase_3_unroll
```

预期输出 `[3_unroll] step 3 PASSED`。

## Case 4：常量外提

源文件：`src/4_scalar_tuning.asc`

向量计算已经比较满时，循环里反复广播常量会多出标量开销。在 Case 3 基础上，把常量提到循环外，用一次 `Duplicate`；循环里直接复用。地址继续写成 `base + i * oneRepeatSize`，这样 `#pragma unroll` 还能生效。

看 `VfConstHoist`。

```bash
cmake --build build --target gelu_eltwise_regbase_4_scalar_tuning
./build/Samples/2_Performance/gelu_eltwise_regbase_story/gelu_eltwise_regbase_4_scalar_tuning
```

预期输出 `[4_scalar_tuning] step 4 PASSED`。可以用 msprof 看 `aiv_scalar_time`。

## 参考文档

- [Reg矢量计算编程](https://gitcode.com/Ascend/asc-devkit/blob/master/docs/guide/编程指南/编程模型/AI-Core-SIMD编程/基于指针的C语言编程/Reg矢量计算编程.md)
- [VF融合优化](https://gitcode.com/Ascend/asc-devkit/blob/master/docs/guide/算子实践参考/SIMD算子性能优化/矢量计算/VF性能优化/VF融合优化.md)
- [VF循环优化](https://gitcode.com/Ascend/asc-devkit/blob/master/docs/guide/算子实践参考/SIMD算子性能优化/矢量计算/VF性能优化/VF循环优化.md)
- [指令双发优化](https://gitcode.com/Ascend/asc-devkit/blob/master/docs/guide/算子实践参考/SIMD算子性能优化/矢量计算/VF性能优化/指令双发优化.md)
