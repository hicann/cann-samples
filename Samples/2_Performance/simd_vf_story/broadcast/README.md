# Broadcast 逻辑模式的 SIMD VF 实现

本文介绍算子中常见的逻辑 broadcast 模式在 SIMD VF 内的实现方式。

Broadcast 是指把 shape 中长度为 1 的轴按逻辑重复到目标长度。例如 `a[M, 1]` 与 `x[M, N]` 做逐元素计算时，可以把 `a` 看成逻辑上的 `a[M, N]`，但实际实现不需要把 `a` 物理展开。SIMD VF 中更常见的写法是：在 VF 内按合适的循环顺序读取原始 `a`，在寄存器中完成复用或广播，再和 `x/y` 的数据流配合。

复杂 broadcast 通常可以先合轴，再规约到 3 种模式：

| 模式 | 规约前 | 规约后 | 本文样例 |
| --- | --- | --- | --- |
| 尾轴 broadcast | `(A1, ..., Ak, 1) -> (A1, ..., Ak, B)` | `(M, 1) -> (M, N)` | `y[m, n] = x[m, n] + a[m]` |
| 首轴 broadcast | `(1, A1, ..., Ak) -> (B, A1, ..., Ak)` | `(1, N) -> (M, N)` | `y[m, n] = x[m, n] + a[n]` |
| 中间轴 broadcast | `(A..., 1, C...) -> (A..., B, C...)` | `(M, 1, N) -> (M, B, N)` | `y[m, b, n] = x[m, b, n] + a[m, n]` |

中间轴 broadcast 固定一个外层 `m` 后，可以看成首轴 broadcast：`x[m, :, :]` 是 `(B, N)`，`a[m, :]` 是 `(1, N)`，计算 `y_m[b, n] = x_m[b, n] + a_m[n]`。

## 尾轴 broadcast（tail_axis）

### 用例公式

```cpp
y[m, n] = x[m, n] + a[m]
```

这里 `a[m]` 在逻辑上沿尾轴 `n` 重复，等价于把 `a[M]` 看成 `a[M, 1] -> a[M, N]`。

### naive实现：逐行发起 VF

```cpp
__simd_vf__ inline void ComputeVF(
    __ubuf__ float* xAddr, __ubuf__ float* yAddr, uint32_t n, uint16_t vfLoopN, float aVal)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> yReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    uint32_t remain = n;
    for (uint16_t i = 0; i < vfLoopN; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(remain);
        AscendC::Reg::LoadAlign<float>(xReg, xAddr + i * VL_B32);
        AscendC::Reg::Adds<float>(yReg, xReg, aVal, mask);
        AscendC::Reg::StoreAlign<float>(yAddr + i * VL_B32, yReg, mask);
    }
}

auto* xUb = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
auto* yUb = reinterpret_cast<__ubuf__ float*>(yLocal.GetPhyAddr());
for (uint32_t j = 0; j < tiling.m; ++j) {
    asc_vf_call<ComputeVF>(
        xUb + j * tiling.nElemsAligned, yUb + j * tiling.nElemsAligned, tiling.n,
        VFLoopNumForB32(tiling.n), aLocal.GetValue(j));
}
```

这个实现按行发起 VF，每次处理一段连续 `n`，代码最直观。但每行都要启动一次 VF，还需要在 Main Scalar 侧读取 `aLocal.GetValue(j)` 后再通过 PB 传给 Vector 单元，VF 粒度偏碎。

源码：`src/tail_axis_1_naive_getvalue.asc`

![tail_axis_1_naive](./images/tail_axis_1_naive.png)

![tail_axis_1_naive_ld](./images/tail_axis_1_naive_ld.png)

### 优化实现：VF 内广播行标量

```cpp
__simd_vf__ inline void ComputeVF(
    __ubuf__ float* xAddr, __ubuf__ float* aAddr, __ubuf__ float* yAddr, uint32_t n, uint32_t nElemsAligned,
    uint16_t vfLoopM, uint16_t vfLoopN)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> aReg;
    AscendC::Reg::RegTensor<float> yReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    uint32_t remain;
    for (uint16_t j = 0; j < vfLoopM; ++j) {
        remain = n;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(aReg, aAddr + j);
        for (uint16_t i = 0; i < vfLoopN; ++i) {
            mask = AscendC::Reg::UpdateMask<float>(remain);
            AscendC::Reg::LoadAlign<float>(xReg, xAddr + j * nElemsAligned + i * VL_B32);
            AscendC::Reg::Add<float>(yReg, xReg, aReg, mask);
            AscendC::Reg::StoreAlign<float>(yAddr + j * nElemsAligned + i * VL_B32, yReg, mask);
        }
    }
}

auto* xUb = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
auto* aUb = reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr());
auto* yUb = reinterpret_cast<__ubuf__ float*>(yLocal.GetPhyAddr());
asc_vf_call<ComputeVF>(
    xUb, aUb, yUb, tiling.n, tiling.nElemsAligned, static_cast<uint16_t>(tiling.m), VFLoopNumForB32(tiling.n));
```

优化实现把 `m` 循环也收进一个 VF。每行开始时用广播读把 `a[j]` 写入 `aReg`，内层仍沿连续 `n` 处理 `x/y`。这样可以减少 VF 发起和 PB 传参，同时让核心计算留在 VF 硬循环内执行。

源码：`src/tail_axis_2_distbrc.asc`

![tail_axis_2_distbrc](./images/tail_axis_2_distbrc.png)

### 性能对比

| 样例 | VF count | PUSHQ VF sum_cycles | RVECSU sum_cycles | PUSH_PB sum_cycles | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `tail_axis_1_naive_getvalue` | 17 | 1247 | 34 | 68 | naive实现；VF 粒度过碎，还有额外的 Main Scalar 开销。 |
| `tail_axis_2_distbrc` | 1 | 356 | 35 | 8 | 优化实现；单 VF 内广播行标量。 |

尾轴 broadcast 的优化重点是把行循环放进 VF，在 VF 内完成行标量广播。主要收益来自 VF 发起次数和 PB 传参次数减少。

## 首轴 broadcast（head_axis）

### 用例公式

```cpp
y[m, n] = x[m, n] + a[n]
```

这里 `a[n]` 在逻辑上沿首轴 `m` 重复，等价于把 `a[N]` 看成 `a[1, N] -> a[M, N]`。

### naive实现：逐行发起 VF

```cpp
__simd_vf__ inline void ComputeVF(
    __ubuf__ float* xAddr, __ubuf__ float* aAddr, __ubuf__ float* yAddr, uint32_t n, uint16_t vfLoopN)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> aReg;
    AscendC::Reg::RegTensor<float> yReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    uint32_t remain = n;
    for (uint16_t i = 0; i < vfLoopN; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(remain);
        AscendC::Reg::LoadAlign<float>(xReg, xAddr + i * VL_B32);
        AscendC::Reg::LoadAlign<float>(aReg, aAddr + i * VL_B32);
        AscendC::Reg::Add<float>(yReg, xReg, aReg, mask);
        AscendC::Reg::StoreAlign<float>(yAddr + i * VL_B32, yReg, mask);
    }
}

auto* aUb = reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr());
for (uint32_t j = 0; j < tiling.m; ++j) {
    auto* xUbCurrentRow = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr()) + j * tiling.nElemsAligned;
    auto* yUbCurrentRow = reinterpret_cast<__ubuf__ float*>(yLocal.GetPhyAddr()) + j * tiling.nElemsAligned;
    asc_vf_call<ComputeVF>(xUbCurrentRow, aUb, yUbCurrentRow, tiling.n, VFLoopNumForB32(tiling.n));
}
```

naive实现按行发起 VF，每次计算一行 `x[j, :] + a[:]`。它保持了单行内连续访问，但会发起 17 次 VF，启动和 PB 开销占比高，也无法让跨行复用 `a[n]` 的机会留在 VF 内。

源码：`src/head_axis_1_naive.asc`

![head_axis_1_naive](./images/head_axis_1_naive.png)

### 优化实现：把 `m` 轴放进 VF

首轴 broadcast 的 `a[n]` 可以复用到多行。把 `m` 循环放进 VF 后，可以选择先遍历 `n` 以复用 `aReg`，也可以先遍历 `m` 以保持每行内连续访问。

**`n -> m`：广播数据复用优先**

```cpp
__simd_vf__ inline void ComputeVF(
    __ubuf__ float* xAddr, __ubuf__ float* aAddr, __ubuf__ float* yAddr, uint32_t n, uint32_t nElemsAligned,
    uint16_t vfLoopM, uint16_t vfLoopN)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> aReg;
    AscendC::Reg::RegTensor<float> yReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    uint32_t remain = n;
    for (uint16_t i = 0; i < vfLoopN; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(remain);
        AscendC::Reg::LoadAlign<float>(aReg, aAddr + i * VL_B32);
        for (uint16_t j = 0; j < vfLoopM; ++j) {
            AscendC::Reg::LoadAlign<float>(xReg, xAddr + i * VL_B32 + j * nElemsAligned);
            AscendC::Reg::Add<float>(yReg, xReg, aReg, mask);
            AscendC::Reg::StoreAlign<float>(yAddr + i * VL_B32 + j * nElemsAligned, yReg, mask);
        }
    }
}

auto* xUb = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
auto* aUb = reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr());
auto* yUb = reinterpret_cast<__ubuf__ float*>(yLocal.GetPhyAddr());
asc_vf_call<ComputeVF>(
    xUb, aUb, yUb, tiling.n, tiling.nElemsAligned, static_cast<uint16_t>(tiling.m), VFLoopNumForB32(tiling.n));
```

`n -> m` 写法每个 `n` 块只加载一次 `aReg`，再复用到所有 `m` 行。当前形状下，减少广播数据重复处理带来的收益更明显，因此它是本组最优的优化实现。

源码：`src/head_axis_2_vfloop_nm.asc`

![head_axis_2_vfloop_nm](./images/head_axis_2_vfloop_nm.png)

![head_axis_2_vfloop_nm_plt](./images/head_axis_2_vfloop_nm_plt.png)

**`m -> n`：主数据连续优先**

```cpp
__simd_vf__ inline void ComputeVF(
    __ubuf__ float* xAddr, __ubuf__ float* aAddr, __ubuf__ float* yAddr, uint32_t n, uint32_t nElemsAligned,
    uint16_t vfLoopM, uint16_t vfLoopN)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> aReg;
    AscendC::Reg::RegTensor<float> yReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    uint32_t remain;
    for (uint16_t j = 0; j < vfLoopM; ++j) {
        remain = n;
        for (uint16_t i = 0; i < vfLoopN; ++i) {
            mask = AscendC::Reg::UpdateMask<float>(remain);
            AscendC::Reg::LoadAlign<float>(aReg, aAddr + i * VL_B32);
            AscendC::Reg::LoadAlign<float>(xReg, xAddr + j * nElemsAligned + i * VL_B32);
            AscendC::Reg::Add<float>(yReg, xReg, aReg, mask);
            AscendC::Reg::StoreAlign<float>(yAddr + j * nElemsAligned + i * VL_B32, yReg, mask);
        }
    }
}

auto* xUb = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
auto* aUb = reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr());
auto* yUb = reinterpret_cast<__ubuf__ float*>(yLocal.GetPhyAddr());
asc_vf_call<ComputeVF>(
    xUb, aUb, yUb, tiling.n, tiling.nElemsAligned, static_cast<uint16_t>(tiling.m), VFLoopNumForB32(tiling.n));
```

`m -> n` 写法保持每行内连续访问，也把 17 次 VF 合并成 1 次，因此明显优于 naive实现。但它会在每行内重复读取 `aReg`，最内层也重复更新 `mask`，当前形状下不如 `n -> m`。

源码：`src/head_axis_3_vfloop_mn.asc`

![head_axis_3_vfloop_mn](./images/head_axis_3_vfloop_mn.png)

![head_axis_3_vfloop_mn_plt](./images/head_axis_3_vfloop_mn_plt.png)

### 性能对比

| 样例 | VF count | PUSHQ VF sum_cycles | RVECEX sum_cycles | RVECLD sum_cycles | RVECSU sum_cycles | PUSH_PB sum_cycles | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `head_axis_1_naive` | 17 | 839 | 2431 | 3552 | 34 | 68 | naive实现；VF 粒度过碎。 |
| `head_axis_2_vfloop_nm` | 1 | 275 | 1375 | 1782 | 13 | 4 | 优化实现；广播数据复用充分。 |
| `head_axis_3_vfloop_mn` | 1 | 442 | 2431 | 3546 | 35 | 4 | 优化实现；连续访问更直接，但重复处理 `a[n]` 和更新 `mask`。 |

首轴 broadcast 的优化重点是把逐行 VF 合并为单个 VF，并在 VF 内利用 `a[n]` 可跨行复用的特点。当前形状下，`n -> m` 复用 `aReg` 带来的事件规模下降更关键，因此优于 `m -> n`。

## 中间轴 broadcast（mid_axis）

### 用例公式

```cpp
y[m, b, n] = x[m, b, n] + a[m, n]
```

这里 `a[m, n]` 在逻辑上沿中间轴 `b` 重复，等价于 `a[M, 1, N] -> a[M, B, N]`。固定一个 `m` 后，它可以看成首轴 broadcast：`a_m[n]` 作为 `(1, N)`，沿 `b` 重复到 `(B, N)`。

### naive实现：每个 `(m, b)` 发起一次 VF

```cpp
__simd_vf__ inline void ComputeVF(
    __ubuf__ float* xAddr, __ubuf__ float* aAddr, __ubuf__ float* yAddr, uint32_t n, uint16_t vfLoopN)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> aReg;
    AscendC::Reg::RegTensor<float> yReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    uint32_t remain = n;
    for (uint16_t i = 0; i < vfLoopN; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(remain);
        AscendC::Reg::LoadAlign<float>(xReg, xAddr + i * VL_B32);
        AscendC::Reg::LoadAlign<float>(aReg, aAddr + i * VL_B32);
        AscendC::Reg::Add<float>(yReg, xReg, aReg, mask);
        AscendC::Reg::StoreAlign<float>(yAddr + i * VL_B32, yReg, mask);
    }
}

auto* xUb = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
auto* aUb = reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr());
auto* yUb = reinterpret_cast<__ubuf__ float*>(yLocal.GetPhyAddr());
for (uint32_t m = 0; m < tiling.m; ++m) {
    for (uint32_t b = 0; b < tiling.b; ++b) {
        uint32_t rowOffset = (m * tiling.b + b) * tiling.nElemsAligned;
        asc_vf_call<ComputeVF>(
            xUb + rowOffset, aUb + m * tiling.nElemsAligned, yUb + rowOffset, tiling.n,
            VFLoopNumForB32(tiling.n));
    }
}
```

naive实现的单个 VF 最简单，每次只处理一段连续 `n`。问题是 VF 粒度过碎，会发起 24 次 VF，PB 传参次数最高，整体性能最差。

源码：`src/mid_axis_1_naive.asc`

![mid_axis_1_naive](./images/mid_axis_1_naive.png)

### 第一步优化：把 broadcast 轴放进 VF

固定一个 `m` 后，`x[m, :, :]` 是 `(B, N)`，`a[m, :]` 是 `(1, N)`。第一步可以先让 kernel 每个 `m` 只发起一次 VF，在 VF 内处理 broadcast 轴 `b`。

**`n -> b`：复用广播数据**

```cpp
__simd_vf__ inline void ComputeVF(
    __ubuf__ float* xAddr, __ubuf__ float* aAddr, __ubuf__ float* yAddr, uint32_t n, uint32_t nElemsAligned,
    uint16_t vfLoopB, uint16_t vfLoopN)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> aReg;
    AscendC::Reg::RegTensor<float> yReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    uint32_t remain = n;
    for (uint16_t i = 0; i < vfLoopN; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(remain);
        AscendC::Reg::LoadAlign<float>(aReg, aAddr + i * VL_B32);
        for (uint16_t b = 0; b < vfLoopB; ++b) {
            AscendC::Reg::LoadAlign<float>(xReg, xAddr + b * nElemsAligned + i * VL_B32);
            AscendC::Reg::Add<float>(yReg, xReg, aReg, mask);
            AscendC::Reg::StoreAlign<float>(yAddr + b * nElemsAligned + i * VL_B32, yReg, mask);
        }
    }
}

auto* xUb = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
auto* aUb = reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr());
auto* yUb = reinterpret_cast<__ubuf__ float*>(yLocal.GetPhyAddr());
for (uint32_t m = 0; m < tiling.m; ++m) {
    asc_vf_call<ComputeVF>(
        xUb + m * tiling.b * tiling.nElemsAligned, aUb + m * tiling.nElemsAligned,
        yUb + m * tiling.b * tiling.nElemsAligned, tiling.n, tiling.nElemsAligned,
        static_cast<uint16_t>(tiling.b), VFLoopNumForB32(tiling.n));
}
```

`n -> b` 写法每个 `n` 块只加载一次 `aReg`，再用于所有 `b` 分块。它减少了广播数据的重复读取，但每个 `m` 仍要单独发起 VF，且短 `b` 循环在内层。当前样例里 `b = 3`，内层循环太短，展开和乱序发射后仍然不容易把向量加载、计算、写回流水充分铺开。

源码：`src/mid_axis_2_vfloop_nb.asc`

![mid_axis_2_vfloop_nb](./images/mid_axis_2_vfloop_nb.png)

**`b -> n`：保持主数据连续**

```cpp
__simd_vf__ inline void ComputeVF(
    __ubuf__ float* xAddr, __ubuf__ float* aAddr, __ubuf__ float* yAddr, uint32_t n, uint32_t nElemsAligned,
    uint16_t vfLoopB, uint16_t vfLoopN)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> aReg;
    AscendC::Reg::RegTensor<float> yReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    for (uint16_t b = 0; b < vfLoopB; ++b) {
        uint32_t remain = n;
        for (uint16_t i = 0; i < vfLoopN; ++i) {
            mask = AscendC::Reg::UpdateMask<float>(remain);
            AscendC::Reg::LoadAlign<float>(aReg, aAddr + i * VL_B32);
            AscendC::Reg::LoadAlign<float>(xReg, xAddr + b * nElemsAligned + i * VL_B32);
            AscendC::Reg::Add<float>(yReg, xReg, aReg, mask);
            AscendC::Reg::StoreAlign<float>(yAddr + b * nElemsAligned + i * VL_B32, yReg, mask);
        }
    }
}

auto* xUb = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
auto* aUb = reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr());
auto* yUb = reinterpret_cast<__ubuf__ float*>(yLocal.GetPhyAddr());
for (uint32_t m = 0; m < tiling.m; ++m) {
    asc_vf_call<ComputeVF>(
        xUb + m * tiling.b * tiling.nElemsAligned, aUb + m * tiling.nElemsAligned,
        yUb + m * tiling.b * tiling.nElemsAligned, tiling.n, tiling.nElemsAligned,
        static_cast<uint16_t>(tiling.b), VFLoopNumForB32(tiling.n));
}
```

`b -> n` 写法会在每个 `b` 下重新读取 `aReg`，但 `x/y` 沿连续 `n` 方向推进，内层循环长度由 `b = 3` 变成 `n` 方向的多个向量片段，流水并行度更容易展开。流水中也能看到，`n -> b` 的向量事件数量更少，但 RVECLD/RVECEX/RVECST 的执行跨度反而更长，RVECLP/RVECSU 事件也更多；说明它省下的 `aReg` 读取没有抵消短内层循环带来的发射效率损失。因此当前形状下，`b -> n` 比 `n -> b` 更好。

源码：`src/mid_axis_3_vfloop_bn.asc`

![mid_axis_3_vfloop_bn](./images/mid_axis_3_vfloop_bn.png)

这和尾轴 broadcast 的直觉不同。尾轴场景中 `a[m]` 是行标量，VF 内广播一次后内层自然沿连续 `n` 处理；而中间轴这里如果写成 `n -> b`，虽然复用了 `aReg`，但最内层只有 3 个 `b` 分块，无法形成足够长的连续指令流。当前形状下，优先保持主数据连续访问和足够长的内层循环更重要。

### 优化实现：把 `m` 也放进 VF

第一步优化已经说明：只看单个 `m` 时，`b -> n` 优于 `n -> b`。进一步优化时，可以把外层 `m` 也收入 VF，继续减少 VF 发起和 PB 传参。`m -> b -> n` 延续连续访问优先的思路，当前样例最快；`m -> n -> b` 也把三层循环收入单个 VF，虽然当前形状下略慢，但仍属于单 VF 优化实现。

**`m -> b -> n`：连续访问优先**

```cpp
__simd_vf__ inline void ComputeVF(__ubuf__ float* xAddr, __ubuf__ float* aAddr, __ubuf__ float* yAddr,
    uint16_t vfLoopM, uint16_t vfLoopB, uint32_t n, uint32_t nElemsAligned, uint16_t vfLoopN)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> aReg;
    AscendC::Reg::RegTensor<float> yReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    for (uint16_t m = 0; m < vfLoopM; ++m) {
        for (uint16_t b = 0; b < vfLoopB; ++b) {
            uint32_t remain = n;
            for (uint16_t i = 0; i < vfLoopN; ++i) {
                mask = AscendC::Reg::UpdateMask<float>(remain);
                AscendC::Reg::LoadAlign<float>(aReg, aAddr + m * nElemsAligned + i * VL_B32);
                AscendC::Reg::LoadAlign<float>(
                    xReg, xAddr + (m * vfLoopB + b) * nElemsAligned + i * VL_B32);
                AscendC::Reg::Add<float>(yReg, xReg, aReg, mask);
                AscendC::Reg::StoreAlign<float>(
                    yAddr + (m * vfLoopB + b) * nElemsAligned + i * VL_B32, yReg, mask);
            }
        }
    }
}

auto* xUb = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
auto* aUb = reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr());
auto* yUb = reinterpret_cast<__ubuf__ float*>(yLocal.GetPhyAddr());
asc_vf_call<ComputeVF>(xUb, aUb, yUb, static_cast<uint16_t>(tiling.m), static_cast<uint16_t>(tiling.b),
    tiling.n, tiling.nElemsAligned, VFLoopNumForB32(tiling.n));
```

源码：`src/mid_axis_4_vfloop_mbn.asc`

![mid_axis_4_vfloop_mbn](./images/mid_axis_4_vfloop_mbn.png)

**`m -> n -> b`：广播数据复用优先**

```cpp
__simd_vf__ inline void ComputeVF(__ubuf__ float* xAddr, __ubuf__ float* aAddr, __ubuf__ float* yAddr,
    uint16_t vfLoopM, uint16_t vfLoopB, uint32_t n, uint32_t nElemsAligned, uint16_t vfLoopN)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> aReg;
    AscendC::Reg::RegTensor<float> yReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    for (uint16_t m = 0; m < vfLoopM; ++m) {
        uint32_t remain = n;
        for (uint16_t i = 0; i < vfLoopN; ++i) {
            mask = AscendC::Reg::UpdateMask<float>(remain);
            AscendC::Reg::LoadAlign<float>(aReg, aAddr + m * nElemsAligned + i * VL_B32);
            for (uint16_t b = 0; b < vfLoopB; ++b) {
                AscendC::Reg::LoadAlign<float>(
                    xReg, xAddr + (m * vfLoopB + b) * nElemsAligned + i * VL_B32);
                AscendC::Reg::Add<float>(yReg, xReg, aReg, mask);
                AscendC::Reg::StoreAlign<float>(
                    yAddr + (m * vfLoopB + b) * nElemsAligned + i * VL_B32, yReg, mask);
            }
        }
    }
}

auto* xUb = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
auto* aUb = reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr());
auto* yUb = reinterpret_cast<__ubuf__ float*>(yLocal.GetPhyAddr());
asc_vf_call<ComputeVF>(xUb, aUb, yUb, static_cast<uint16_t>(tiling.m), static_cast<uint16_t>(tiling.b),
    tiling.n, tiling.nElemsAligned, VFLoopNumForB32(tiling.n));
```

源码：`src/mid_axis_5_vfloop_mnb.asc`

![mid_axis_5_vfloop_mnb](./images/mid_axis_5_vfloop_mnb.png)

### 性能对比

| 样例 | VF count | PUSHQ VF sum_cycles | RVECLP sum_cycles | RVECSU sum_cycles | PUSH_PB sum_cycles | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mid_axis_1_naive` | 24 | 1168 | 24 | 48 | 96 | naive实现；VF 调用粒度过碎。 |
| `mid_axis_2_vfloop_nb` | 8 | 884 | 96 | 104 | 32 | 第一步优化；每个 `m` 一次 VF，按 `n -> b` 复用 `aReg`。 |
| `mid_axis_3_vfloop_bn` | 8 | 751 | 32 | 56 | 32 | 第一步优化；每个 `m` 一次 VF，按 `b -> n` 连续处理。 |
| `mid_axis_4_vfloop_mbn` | 1 | 593 | 33 | 49 | 8 | 优化实现；单 VF，按 `m -> b -> n` 连续处理主数据。 |
| `mid_axis_5_vfloop_mnb` | 1 | 710 | 97 | 97 | 8 | 优化实现；单 VF，按 `m -> n -> b` 复用 `aReg`。 |

中间轴 broadcast 的优化顺序可以分两步看：先把 `b` 轴放进 VF，减少 `(m, b)` 粒度的重复发起；再把 `m` 轴也放进 VF，进一步减少 VF 和 PB 开销。当前形状下，`m -> b -> n` 最好；`m -> n -> b` 也属于优化实现，但短 `b` 内层带来的开销更明显。

## 总结

这几个 broadcast 样例的共同优化路径是：先写最直观的 naive实现，再逐步把外层循环收入 VF，让更多计算停留在 VF 硬循环里执行。这样通常可以减少 `asc_vf_call` 次数和 PB 传参次数，也更容易让 Vector 单元形成连续流水。

这三种 broadcast 模式的优化经验可以总结如下：

1. 如果 naive实现按行或按分块多次发起 VF，通常先尝试把外层循环放进 VF，减少 VF 启动和 PB 传参。
2. 对尾轴 broadcast，行标量适合在 VF 内广播后复用，内层继续沿连续 `n` 方向处理主数据。
3. 对首轴 broadcast，广播向量可以跨行复用；通常对于一个不太小的 `m` 来说，先遍历 `n` 再遍历 `m` 通常能拿到每一列复用 `a` 的收益。换句话说，如果 `m` 过于小导致内层循环展开短，也有可能拿不到收益，需要具体分析。
4. 对中间轴 broadcast，通常可以规约为首轴 broadcast 的模式，只需要把外层循环包入 VF 即可获得对应的提升。和 3. 类似，要判断 broadcast 轴 `b` 的大小来取舍。以本例来说，`b = 3` 的小场景有导致 `m->n->b` 的列循环实际上不如 `m->b->n` 的行循环。

实际开发中，建议先按最自然的循环顺序写出功能正确的 VF，再通过仿真流水验证直觉。若结果和直觉不一致，需要回到源码和流水一起看：总 VF 耗时、VF 调用次数、PB 传参、主数据访问是否连续、最内层循环是否足够长，以及 RVECLP/RVECSU 是否被放大。本文的中间轴 broadcast 就是一个典型例子：`n -> b` 看起来减少了 `aReg` 读取，但由于 `b = 3` 内层太短，最终反而慢于 `b -> n`。
