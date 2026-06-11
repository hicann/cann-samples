# Reg 数据搬运场景选型指南

> 适用硬件：Ascend 950PR / 950DT（NpuArch DAV_3510）。
> 频次统计自 `ops-cv` / `ops-math` / `ops-nn` / `ops-transformer` 四个算子仓全量源码。`AscendC::Reg` 与 `AscendC::MicroAPI` 是同一命名空间（`kernel_macros.h` 中 `namespace MicroAPI = Reg`），下文按各处真实代码写法混用。
> 文中实测输出均来自 `src/` 各示例在 950 实机上的运行结果（CANN 9.1.0，`cmake -DNPU_ARCH=dav-3510` 构建后逐个执行）；为便于逐列对照，引用时部分行内空白经排版对齐，数值未作任何改动。

## 问题引入

整条矢量计算链路是 `GM → UB → 寄存器 → UB → GM`，本指南只覆盖中间 `UB ↔ 寄存器` 这一跳（GM ↔ UB 的 DMA 不在此列）。唯有寄存器能做矢量计算，所有形变也都集中在进出寄存器这一步：搬入侧 `LoadDist`、搬出侧 `StoreDist` 各有一组 `dist` 枚举决定数据排布。

![数据搬运在矢量计算流程中的位置](images/sel-architecture.svg)

选错 `dist` 的代价几乎从不是「慢」，而是结果静默出错——不报错、不崩溃：mask 按错误类型生成会去开不存在的 lane，stride 按字节填会跳到错误的块，漏一条收尾指令会丢掉尾部数据。唯一立即暴露的故障是地址对齐违例：对齐搬入接口收到非 32 B 对齐的 UB 地址会直接触发设备异常（错误码 507035，编写 `src/scene10_unalign.asc` 时实测复现）。所以选型先于编码，每个模式都要先确认适用边界。

所有示例 VF 函数遵循相同上下文：`__simd_vf__ inline void SceneNVf(__ubuf__ T* addr, ..., uint32_t n, uint16_t loopNum)`。`__simd_vf__` 是 MicroAPI / Reg 指令的必需标记；`__ubuf__ T*` 由桥接层从 `LocalTensor` 经 `GetPhyAddr()` 提取；`n` 是元素总数，`loopNum = ceil(n / VL_B32)`。每个模式先给 `src/` 里可直接编译运行的最简函数与实测输出，再指向算子仓中的真实出处。

## 选型决策维度

UB ↔ 寄存器搬运的选型由四个相互独立的问题决定，依次回答即可锁定模式：

1. **排布是否变化。** UB 里的排布和寄存器里的排布一致（逐元素 1:1）就走常规搬运；需要紧排↔间隔（位宽变化）、标量↔整批（广播/归约）、奇偶交织拆合时，分别对应升降位、归约广播、交织三类。
2. **位宽是否变化。** 半精度存储、单精度计算的混合精度路线必须在搬入时升位（`UNPACK`）、搬出时降位（`PACK`），位宽不变则不涉及。
3. **UB 访问是否连续。** 连续地址走常规；固定步长跳写用 `DATA_BLOCK_COPY`；任意下标取数用 `Gather`（索引需先 `E2B` 扩块）。能连续搬就不要离散访问。
4. **对齐与粒度。** 对齐通路（`LoadAlign` / `StoreAlign`）要求 UB 地址 32 B 对齐，违反直接设备异常；非对齐或每轮变长写出用 `StoreUnAlign` 三件套。

寻址模式（地址如何推进）和 mask（哪些 lane 生效）是与上述选择正交的两条维度，任何模式都要配置，见后文。

## 方案对比

五种模式与典型场景的对应关系（详细用法见后文同名小节）：

| 模式 | 典型场景 | 搬入 dist | 搬出 dist | 频次量级 | 关键约束 |
|---|---|---|---|---|---|
| **常规** | 逐元素算子 | `DIST_NORM`（缺省） | `DIST_NORM` | 数千（最高） | 搬出必带 mask |
| **升降位** | 混合精度 | `DIST_UNPACK_B16` | `DIST_PACK_B32` | 千级 | mask 按宽类型；降位 Cast 必须显式 SatMode |
| **归约与广播** | norm / 方差 / softmax | `DIST_BRC_B32`（放） | `DIST_FIRST_ELEMENT_B32`（收） | 千级 | 归约写出偏移 `+i` 不是 `+i*VL` |
| **非连续** | 提列 / gather / 索引取数 | `DIST_E2B_B32` | `DATA_BLOCK_COPY` | 百级 | stride 单位是 32B 块；E2B 单轮仅 8 索引 |
| **特殊** | 复数 / 掩码 / 非对齐 | `DINTLV` / `MaskDist` | `INTLV` / `StoreUnAlign` | 百级及以下 | 漏 `Post` 丢尾；同地址存读 mask 要插屏障 |

按真实频次，前五名（NORM、UNPACK、BRC、PACK、FIRST_ELEMENT）覆盖了绝大多数代码，应优先掌握；非连续与特殊场景是按需查阅的长尾。寻址模式里的 `POST_MODE_UPDATE` 用量 2551，比任何单个 `dist` 都高，在生产代码里基本是默认做法。

b64 类型路径特殊：枚举层面存在 `StoreDist::DIST_PACK_B64`，b64 搬入在实现里按双 32 位寄存器（`DIST_DINTLV_B32`）处理，缺省 NORM 落到 `DIST_NORM_B32`。「b64 仅可用常规搬运」在接口实现中无静态断言依据，各 dist 对 b64 的实际可用性需实测确认（本批示例未覆盖 b64）。

---

## 常规搬运 `DIST_NORM`

> **用于**逐元素算子——向量加减乘、激活、比较这类输入输出形状一致、进出寄存器不做任何形变的运算，占了全部搬运的约九成。UB 中的排布原样进入寄存器，逐元素 1:1 对应，是最快、约束最少的通路；`dist` 模板参数缺省即为它。需要变位宽、广播、归约、跳写或交织时，再选用后面四种。

VF 内搬运推荐使用 `LoadAlign`（搬入）/ `StoreAlign`（搬出），方向由 API 名显式表达，模板参数需写明类型与 `dist`。`DataCopy` 是同一实现的兼容别名（`kernel_reg_compute_datacopy_intf_impl.h` 中两者调用同一 Impl，别名模板带 `DIST_NORM` 缺省值），存量算子代码大量使用，阅读时按方向规则区分：第一个参数是 `RegTensor` 为搬入、是 `__ubuf__ T*` 为搬出。

接口签名本身就规定了一条不对称规则：**搬入不带 mask，搬出必须带 mask**（`LoadAlign` dist 类重载无 mask 形参，`StoreAlign` 末参数为 `MaskReg&`）。UB 整块分配、尾部留有 padding，搬入时多读几个 lane 不会越界，残留值后续被 mask 屏蔽；搬出的 mask 决定哪些 lane 落盘，尾轮不开满 mask 就能避免垃圾值覆盖输出 buffer。例外是 `DATA_BLOCK_COPY` 模式，其搬入重载也带 mask。

![常规搬运](images/sel-a-norm.svg)

`src/scene1_dist_norm.asc` 的向量加法（N=150 取非 VL=64 整数倍以覆盖尾块路径，输出 buffer 预填校验值 -1）：

```cpp
static constexpr uint32_t VL_B32 = 256 / sizeof(float);  // float → 64 lane

template <typename T>
__simd_vf__ inline void Scene1Vf(__ubuf__ T* xAddr, __ubuf__ T* yAddr,
                                 __ubuf__ T* zAddr, uint32_t n, uint16_t loopNum)
{
    AscendC::Reg::RegTensor<T> vregA, vregB, vregC;
    AscendC::Reg::MaskReg mask;
    uint32_t count = n;

    for (uint16_t i = 0; i < loopNum; i++) {
        mask = AscendC::Reg::UpdateMask<T>(count);   // 开前 count 个 lane，count 原地递减

        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(
            vregA, xAddr + i * VL_B32);                              // 搬入，无 mask
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(
            vregB, yAddr + i * VL_B32);

        AscendC::Reg::Add(vregC, vregA, vregB, mask);

        AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM>(
            zAddr + i * VL_B32, vregC, mask);                        // 搬出，带 mask
    }
}
```

实测输出（x[i]=i·0.5，y[i]=i·0.25）：

```text
[scene1] in : x[0..3]=0.000000 0.500000 1.000000 1.500000  y[0..3]=0.000000 0.250000 0.500000 0.750000
[scene1] out: z[0..3]=0.000000 0.750000 1.500000 2.250000
[scene1] tail: z[148]=111.000000 z[149]=111.750000 | z[150]= -1.000000 z[151]= -1.000000 (sentinel -1)
```

尾轮 mask 只开 22 个 lane：z[149]=149×0.75 有效，z[150] 起预填值原样保留——这两个值直接证明了搬出 mask 对尾部的屏蔽作用。

`VL_B32` 是寄存器 lane 数，float 为 64；偏移 `i * VL_B32` 按元素跳（`T*` 指针加法自动换算 `sizeof(T)`），且每轮搬入地址保持 32 B 对齐。生产代码偏好显式风格，会写出 `MicroAPI::LoadDist::DIST_NORM` / `StoreDist::DIST_NORM` 并用框架维护的 `sreg` 做每轮递减，结构与上面完全一致——见 `ops-math/.../add_n_regbase.h:198`。

mask 是某条指令的逐 lane 开关，由两个要素决定：开几个 lane、开哪几个。**开几个**由传入的类型决定——寄存器固定 256 B，lane 数 = 256 / sizeof(类型)，b32 是 64 lane、b16 是 128、b8 是 256、b64 是 32。**开哪几个**有两个入口：`UpdateMask<T>(count)` 用于收尾，开前 `count` 个 lane（`count` ≥ lane 数时全开），底层 `plt` 指令带 `POST_UPDATE` 原地递减 `count`；`CreateMask<T, MaskPattern::X>()` 给固定式样，常用 `ALL`（全开）和 `VL1`（只开第 0 lane，配 `FIRST_ELEMENT` 写出）。

混合精度封装常把常规搬运与升位往返合在同一个编译期分支里。`ops-nn/.../ada_layer_norm_common.h:61` 的 `LoadTensor` 就是 fp32 走常规搬运、半精度走 UNPACK 升位：

```cpp
__simd_callee__ inline void LoadTensor(RegTensor<float>& dst, __ubuf__ T* srcAddr,
                                        MaskReg& pregLoop) {
    if constexpr (std::is_same_v<T, float>) {
        DataCopy(dst, srcAddr);                              // fp32：缺省 DIST_NORM
    } else {
        RegTensor<T> tmpFp16;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(tmpFp16, srcAddr); // 半精度 → UNPACK 升位
        Cast<float, T, castTraitB16ToB32>(dst, tmpFp16, pregLoop);
    }
}
```

> 完整源码：[`src/scene1_dist_norm.asc`](src/scene1_dist_norm.asc)

---

## 升降位搬运 `UNPACK` ↔ `PACK`

> **用于**混合精度算子：输入存成 fp16/bf16 能省一半带宽（缓解内存墙），但半精度动态范围小、直接算容易溢出或掉精度，于是搬入时升到 fp32 算、算完再降回半精度写出。`UNPACK` 升位（1293 次）、`PACK` 降位（847 次）互为逆操作、通常成对出现；纯量化场景里 `PACK` 也单独用，只在落盘时降精度。若全程同精度（如纯 fp32）则不必往返，走常规搬运。

fp16/bf16 在 UB 里是紧排的（每 16 位一个数、地址相邻），但 fp32 运算要求每个数独占一个 32 位槽。直接拿紧排 b16 当 fp32 寄存器用，相邻两个数的高、低 16 位会被当成同一个浮点数的符号、指数、尾数，算出来毫无意义。所以搬入必须 `UNPACK`：把紧排 b16 逐个放进 32 位槽；算完再用 `PACK` 压回——取每个 32 位槽的低 16 位、紧排写出。两条指令互为逆操作，几乎总是成对出现。整条 UNPACK→Cast→计算→Cast→PACK 链路的数值正确性已实测（见下），32 位槽内的具体填充方式是接口内部行为，未单独实测。

![UNPACK 升位排布](images/sel-b-unpack.svg)

搬入用 `LoadDist::DIST_UNPACK_B16`，搬出用 `StoreDist::DIST_PACK_B32`；u32→u8 窄化用 `DIST_PACK4_B32`，4 位数据用 `DIST_UNPACK4_B8`。有三条规则必须守住。

**① mask 跟「计算时的类型」走，不是存储类型。** 这是 lane 数规则的直接推论：bf16/half 按 b16 算本是 128 lane，但 UNPACK 把每个数升进 32 位后，寄存器是按 fp32、64 lane 在算。所以 mask 要用 `UpdateMask<float>`（64 lane），而不是 `UpdateMask<half>`（128 lane）。写成窄类型会多出一倍 lane、去开根本不存在的第 65–128 个 lane，编译器不报错、结果静默出错。整条 UNPACK→计算→PACK 全程都按宽类型生成 mask。

**② 降位 Cast 的 SatMode 必须显式。** f32→f16 的 `CastTrait` 写 `SatMode::UNKNOWN` 直接编译失败（vcvt 内建静态断言要求 NO_SAT/SAT，编写 scene2 时实测触发）。算子仓惯例是 `NO_SAT` + `CAST_RINT`。

**③ PACK 静默截断。** 它只取低 16 位、不做饱和检查〔截断行为未实测，需实测确认〕，用前先确认数值范围不会溢出半精度。

`src/scene2_unpack_b16.asc` 的完整往返——half → UNPACK → 升 fp32 → exp → 降 half → PACK：

```cpp
// CastTrait 等 Reg 类型仅 device 侧可见，常量需置于 #ifdef __NPU_ARCH__ 内
#ifdef __NPU_ARCH__
static constexpr AscendC::Reg::CastTrait CAST_B16_TO_B32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait CAST_B32_TO_B16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,   // 降位必须显式饱和模式
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
#endif

template <typename T>
__simd_vf__ inline void Scene2Vf(__ubuf__ T* xAddr, __ubuf__ T* yAddr,
                                 uint32_t n, uint16_t loopNum)
{
    AscendC::Reg::RegTensor<T> vreg16In, vreg16Out;       // 128 lane × 16b
    AscendC::Reg::RegTensor<float> vreg32In, vreg32Out;   // 64 lane × 32b
    AscendC::Reg::MaskReg mask;
    uint32_t count = n;

    for (uint16_t i = 0; i < loopNum; i++) {
        mask = AscendC::Reg::UpdateMask<float>(count);     // 按宽类型生成 mask

        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
            vreg16In, xAddr + i * VL_B32);                 // 紧排 b16 → 32 位槽

        AscendC::Reg::Cast<float, T, CAST_B16_TO_B32>(vreg32In, vreg16In, mask);
        AscendC::Reg::Exp(vreg32Out, vreg32In, mask);
        AscendC::Reg::Cast<T, float, CAST_B32_TO_B16>(vreg16Out, vreg32Out, mask);

        AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(
            yAddr + i * VL_B32, vreg16Out, mask);          // 32 位槽低 16 位 → 紧排
    }
}
```

实测输出（x[i]=i·0.01，host 按 half 容差校验全部通过）：

```text
[scene2] in : x[0..3]=0.000000 0.010002 0.020004 0.029999
[scene2] out: y[0..3]=1.000000 1.009766 1.020508 1.030273
[scene2] tail: y[149]= 4.437500 | y[150]=-1.000000 (sentinel -1)
```

y[149]=exp(1.49)≈4.4371 的 half 表示 4.4375——升降位往返的精度损失只剩 half 量化误差。同样的五步流水在 `ops-math/.../exp_dag.h:80` 完整出现。量化场景里 PACK 还常和 `Reg::Pack` 搭配：`Pack<LOWEST>` 在寄存器内压成 16 位，`DIST_PACK_B16` 只负责紧排落盘，见 `ops-nn/.../dynamic_mx_quant_tail_axis_fp8.h:520`。`src/scene3_pack_b32.asc` 是 PACK 单独使用的最简示范：fp32 算完 ×0.5、Cast 成 half、`PACK_B32` 落盘，输出由 4 B/元素紧排成 2 B/元素（实测 y[0..3]=0/0.25/0.5/0.75，尾部预填值完好）。

> 完整源码：[`src/scene2_unpack_b16.asc`](src/scene2_unpack_b16.asc)　[`src/scene3_pack_b32.asc`](src/scene3_pack_b32.asc)

---

## 归约与广播 `FIRST_ELEMENT` ↔ `BRC`

> **用于**「先把一批数归约成一个标量、再把这个标量作用回整批」的算子——LayerNorm、方差、softmax 均属此类（归约出 mean / max / var → 广播回去做 normalize 或减均值）。`FIRST_ELEMENT` 负责「收」（归约结果逐个写出，846 次）、`BRC` 负责「放」（标量广播铺满整批，993 次）。只做逐元素、不涉及批内归约或广播的运算走常规搬运。

![归约与广播](images/sel-c-reduce-bcast.svg)

**收**：`ReduceSum` / `ReduceMax` 把 64 个 lane 归约成一个标量、放在第 0 lane，其余 63 个 lane 是无意义的中间值；`StoreDist::DIST_FIRST_ELEMENT_B32` 只取这一个 lane 写到 `output[i]`。`src/scene4_first_element.asc` 按 VL 分组归约求和、每组只落一个标量：

```cpp
__simd_vf__ inline void Scene4Vf(__ubuf__ float* xAddr, __ubuf__ float* yAddr,
                                 uint32_t n, uint16_t loopNum)
{
    AscendC::Reg::RegTensor<float> vregIn, vregSum;
    AscendC::Reg::MaskReg maskVL, maskVL1;
    uint32_t count = n;
    maskVL1 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();

    for (uint16_t i = 0; i < loopNum; i++) {
        maskVL = AscendC::Reg::UpdateMask<float>(count);

        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
            vregIn, xAddr + i * VL_B32);

        AscendC::Reg::ReduceSum(vregSum, vregIn, maskVL);         // 归约 → 首 lane

        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            yAddr + i, vregSum, maskVL1);                          // 偏移 +i，逐元素推进
    }
}
```

实测输出（三组 64 lane 分别全 1、全 2、全 3）：

```text
[scene4] in : group0=1.000000 group1=2.000000 group2=3.000000 (per-lane const)
[scene4] out: y[0]= 64.000000 y[1]=128.000000 y[2]=192.000000 | y[3]= -1.000000 (sentinel -1)
```

三个组和紧排在 y[0..2]，y[3] 起预填值未被改写。**易错点在输出偏移——`yAddr + i`，每轮只前进 1 个元素。** 写成 `+ i * VL_B32` 输出就变稀疏（每 64 个位置才一个有效值），后续二分归并读到大片零，方差永远算成 0——这是 `FIRST_ELEMENT` 最常见的 bug。`MaskPattern::VL1` 只开第 0 lane，与 FIRST_ELEMENT 语义吻合。两遍法求方差的真实代码见 `ops-math/.../reduce_var_twopass.h:365`。

**放**：`LoadDist::DIST_BRC_B32` 从 UB 读 1 个值、复制铺满 64 个 lane，之后一条向量指令就能做整批运算（位宽版本 `DIST_BRC_B16` 173 次、`DIST_BRC_B8` 46 次）。`src/scene5_brc_b32.asc` 做数组加标量 bias：

```cpp
__simd_vf__ inline void Scene5Vf(__ubuf__ float* xAddr, __ubuf__ float* biasAddr,
                                 __ubuf__ float* yAddr, uint32_t n, uint16_t loopNum)
{
    AscendC::Reg::RegTensor<float> vregData, vregBias, vregOut;
    AscendC::Reg::MaskReg mask;
    uint32_t count = n;

    for (uint16_t i = 0; i < loopNum; i++) {
        mask = AscendC::Reg::UpdateMask<float>(count);

        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
            vregData, xAddr + i * VL_B32);

        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
            vregBias, biasAddr);                          // 1 个标量 → 64 lane

        AscendC::Reg::Add(vregOut, vregData, vregBias, mask);

        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(
            yAddr + i * VL_B32, vregOut, mask);
    }
}
```

实测输出（x[i]=i·0.5，UB 中仅 bias[0]=100 有效）：

```text
[scene5] in : x[0..3]=  0.000000   0.500000   1.000000   1.500000  bias[0]=100.000000
[scene5] out: y[0..3]=100.000000 100.500000 101.000000 101.500000
[scene5] tail: y[149]=174.500000 | y[150]= -1.000000 (sentinel -1)
```

`biasAddr` 循环内不变（每轮广播同一标量），配 `POST_MODE_NORMAL`——被广播的地址不随 Repeat 推进。收和放在生产代码中常衔接成「均值落 UB → BRC 读回整批运算」的两步：`ops-math/.../reduce_var_welford.h:1481` 先用高阶 `ReduceSum` 把均值落入 UB，再 `DIST_BRC_B32` 广播回去做整批减法；同文件其他段则以 `FIRST_ELEMENT` 落归约值。`ops-cv/.../nms_with_mask_regbase_base.h:51` 用 `if constexpr (isBroadcast)` 在编译期零开销切换广播/非广播。

> 完整源码：[`src/scene4_first_element.asc`](src/scene4_first_element.asc)　[`src/scene5_brc_b32.asc`](src/scene5_brc_b32.asc)

---

## 非连续存取 `DATA_BLOCK_COPY` + `Gather` / `E2B`

> **用于**数据在寄存器里连续、但在 UB 里读 / 写位置不连续的场景，分两类：**固定步长跳写**（提矩阵列、隔行落盘）用 `DATA_BLOCK_COPY`（502 次）；**按索引取数**用 `Gather` + `DIST_E2B_B32`（325 / 113 次，按别名 `DataCopyGather` 统计）。**凡可连续搬运的场景一律不应使用**——离散读写远慢于连续搬运。

### 跳跃搬运 `DATA_BLOCK_COPY`

以 DataBlock（32 B）为单位，每写一块就按 `dataBlockStride` 跳一段。**这里的单位是「块」：`dataBlockStride` 数的是 32 B 块，不是字节、也不是元素**（实现按 `stride * ONE_BLOCK_SIZE / sizeof(T)` 换算地址，`ONE_BLOCK_SIZE = 32`）——要跳 64 字节就写 `stride = 2`。按字节或元素计算会静默写到错误位置，不崩溃，且难以定位。mask 则是**逐元素粒度**，与普通搬出一致，半块也能精确截断（实测见下）。

![DATA_BLOCK_COPY 跳写](images/sel-d-blockcopy.svg)

`src/scene6_block_copy.asc` 连续读 64 元素、按 stride=2 跳写，另用 count=20 验证 mask 粒度：

```cpp
__simd_vf__ inline void Scene6Vf(__ubuf__ float* xAddr, __ubuf__ float* yAddr,
                                 __ubuf__ float* y2Addr)
{
    AscendC::Reg::RegTensor<float> vreg;
    AscendC::Reg::MaskReg maskAll, maskPart;
    uint32_t cntAll = VL_B32, cntPart = 20;

    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
        vreg, xAddr);                                        // 连续搬入 64 元素

    maskAll = AscendC::Reg::UpdateMask<float>(cntAll);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
        yAddr, vreg, STRIDE, maskAll);                       // 8 块全写，块间跳 32 B

    maskPart = AscendC::Reg::UpdateMask<float>(cntPart);     // 20 lane = 2.5 块
    AscendC::Reg::StoreAlign<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
        y2Addr, vreg, 1, maskPart);                          // stride=1：验证 mask 粒度
}
```

实测输出（x[i]=i，输出区预填校验值 -1）：

```text
[scene6] y[0]=0.000000 y[7]=7.000000 | y[8]=-1.000000 y[15]=-1.000000 (sentinel) | y[16]=8.000000 y[23]=15.000000
[scene6] mask20: y2[15]=15.000000 | y2[16..19]=16.000000 17.000000 18.000000 19.000000 | y2[20..23]=-1.000000 -1.000000 -1.000000 -1.000000
```

stride=2 时块 k 落在 y[16k..16k+7]、块间 8 个预填值原样保留；count=20 精确写到 y2[19] 为止、半块截断——证实 mask 按元素而非按块。`ops-nn/.../gather_nd_full_load_vgather.h:269` 就是用它把 gather 结果非连续落盘。

### 按索引取数 `Gather` + `DIST_E2B_B32`

Gather 要求索引在寄存器中按 DataBlock 粒度排布，但索引在 UB 里是紧排的（每个 `uint32_t` 占 4 字节）。`DIST_E2B_B32`（Element-to-Block）完成该转换，代价是单轮覆盖量显著受限：**单轮只展开前 8 个 b32 索引**（256 B / 32 B = 8 块），且块内 8 个 lane 重复同一索引值；Gather 后输出按块重复——`out[8k..8k+7]` 全部等于 `data[idx[k]]`（实测见下）。逐索引独立收集需以 8 索引为步长多轮处理，大批量索引代价显著。

`E2B →（加基址）→ Gather →（BLOCK_COPY 写出）` 是 gather 类算子的标准四步流水。`src/scene7_gather_e2b.asc` 是省略加基址的三步示范：

```cpp
__simd_vf__ inline void Scene7Vf(__ubuf__ uint32_t* idxAddr, __ubuf__ float* dataAddr,
                                 __ubuf__ float* outAddr)
{
    AscendC::Reg::RegTensor<uint32_t> idxReg;
    AscendC::Reg::RegTensor<float> vregOut;
    AscendC::Reg::MaskReg preg;
    uint32_t count = VL_B32;

    preg = AscendC::Reg::UpdateMask<uint32_t>(count);

    AscendC::Reg::LoadAlign<uint32_t, AscendC::Reg::LoadDist::DIST_E2B_B32>(
        idxReg, idxAddr);                                  // 前 8 个紧排索引 → 每索引一块

    AscendC::Reg::Gather<float, float, uint32_t>(vregOut, dataAddr, idxReg, preg);   // 离散收集

    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(
        outAddr, vregOut, preg);                           // 连续写出
}
```

实测输出（data[i]=i，倒序索引 idx[j]=63-j）：

```text
[scene7] idx[0..3]=63 62 61 60  data[0..3]=0.000000 1.000000 2.000000 3.000000
[scene7] out[0..3]=63.000000 63.000000 63.000000 63.000000 | out[7]=63.000000 out[8]=62.000000 | out[63]=56.000000
```

64 lane 输出只消费了 idx[0..7]=63..56，每索引重复 8 次，直接呈现了 E2B 的「扩块」语义。

![Gather 四步流水](images/sel-gather.svg)

索引位宽随数据类型走：b32/b64 数据用 u32 索引，b8/b16 用 u16（实现层有类型约束断言）。三条要点：**可连续搬运时不使用 gather**；索引位宽按数据类型选择，大张量避免 16 位索引（会溢出）；mask 按索引类型生成。完整四步流水（`E2B → Add 加行基址 → Gather → BLOCK_COPY`，一段代码同时用到三种搬运）见 `ops-nn/.../gather_nd_full_load_vgather.h:265`。

> 完整源码：[`src/scene6_block_copy.asc`](src/scene6_block_copy.asc)　[`src/scene7_gather_e2b.asc`](src/scene7_gather_e2b.asc)

---

## 特殊场景：交织 / 掩码 / 非对齐

三种边角情况各有各的坑，频次都不高，按需查阅即可。

### 交织拆合 `DINTLV` / `INTLV`

> **用于**复数 `[r0,i0,r1,i1,…]` 或双通道这类「两路交织」数据。普通搬入会把实部、虚部混进同一个寄存器，没法分别做 Abs、共轭、相位旋转；`DINTLV` 在搬入时就按奇偶拆成两个寄存器，各自算完再用 `INTLV` 交错合并写回。

`DINTLV` 一次搬两个寄存器：偶数位 0,2,4… 进 reg0，奇数位 1,3,5… 进 reg1，拆开后各自独立计算，算完再用 `INTLV` 交错合并回去。两条指令总是成对出现（拆 → 处理 → 合）。双寄存器重载的 `dist` 模板参数没有缺省值，必须显式写出。双寄存器搬运的数据量是单搬运的 2 倍：步长按 `VL * 2` 推进，mask 按 `T` 生成、一份同时作用于两个寄存器。

![DINTLV 拆/合](images/sel-e-interleave.svg)

`src/scene8_dintlv_intlv.asc` 把交织数据拆成实 / 虚两路、各自取绝对值后再合并：

```cpp
__simd_vf__ inline void Scene8Vf(__ubuf__ float* xAddr, __ubuf__ float* yAddr)
{
    AscendC::Reg::RegTensor<float> vregEven, vregOdd, vregAbsE, vregAbsO;
    AscendC::Reg::MaskReg preg;
    uint32_t count = PAIRS;                                // 64 对 (r,i)

    preg = AscendC::Reg::UpdateMask<float>(count);

    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(
        vregEven, vregOdd, xAddr);                         // 偶 lane → even，奇 lane → odd

    AscendC::Reg::Abs(vregAbsE, vregEven, preg);
    AscendC::Reg::Abs(vregAbsO, vregOdd, preg);

    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_INTLV_B32>(
        yAddr, vregAbsE, vregAbsO, preg);                  // 交错合并写回
}
```

实测输出（实部 k+0.25、虚部 -(k+0.5) 交织）：

```text
[scene8] in : x[0..5]= 0.250000 -0.500000  1.250000 -1.500000  2.250000 -2.500000 (r,i interleaved)
[scene8] out: y[0..5]= 0.250000  0.500000  1.250000  1.500000  2.250000  2.500000
```

虚部去负号、实部不变、交织次序保持。真实代码 `ops-math/.../abs_complex_dag.h:115` 用的是 `Reg::LoadAlign` 的等价写法。

### 掩码存取 `MaskDist`

> **用于**搬运 `MaskReg` 本身——把比较 / 判定得到的逐 lane 布尔结果存进 UB 暂存，之后（跨 Repeat 或跨阶段）再读回来复用：读回的 mask 常接 `MaskAnd` 做位与、再 `Select` 置零或二选一。NMS 保存 / 恢复 IoU 比较结果、attention mask 都是这个用法。

mask 在 UB 中是紧凑位流，与数据存储格式完全不同。实测布局：**MaskReg 落盘固定 256 bit（32 B），按 b8 lane 每 lane 1 bit；b32 数据的有效位是每 4 bit 取 1**——比较结果 `x>=32`（b32 lanes 32..63）落盘为后 4 个字 `0x11111111`。接口类型参数允许 1/2/4/8 字节整型（底层按 32 位访问），算子仓惯用 `int32_t`。存读参数顺序相反，统一规则为目标在前、源在后。

```cpp
// MaskReg → UB（存储）：UB 地址在前
AscendC::Reg::StoreAlign<int32_t, AscendC::Reg::MaskDist::DIST_NORM>(maskAddr, cmpMask);
// 同地址存→读必须显式插存读屏障，否则读回结果不定（实测概率性全 0）
AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
// UB → MaskReg（读取）：MaskReg 在前
AscendC::Reg::LoadAlign<int32_t, AscendC::Reg::MaskDist::DIST_NORM>(cmpMask, maskAddr);

AscendC::Reg::MaskAnd(cmpMask, cmpMask, evenMask, pregAll);
AscendC::Reg::Select(vregData, vregData, vregZeros, cmpMask);
```

实测输出（x[i]=i，`x>=32` 与偶 lane 位流求与后 Select）：

```text
[scene9] mask in UB: w0=0x0 w1=0x0 w2=0x0 w3=0x0 w4=0x11111111 w5=0x11111111 w6=0x11111111 w7=0x11111111
[scene9] out: y[30..35]= 0.000000  0.000000 32.000000  0.000000 34.000000  0.000000
```

**`LocalMemBar` 不可省**：VF 内编译器不追踪经 UB 指针的存读依赖，scene9 不加屏障时同一可执行文件连续运行交替出现全 0 与正确结果。`MaskDist` 取值上读侧允许 `NORM/US/DS`、写侧允许 `NORM/PACK`；数据经过 US/DS 采样时 mask 必须同步采样，否则 lane 数对不上。真实流水（存 IoU 比较结果 → 读回复用 → `MaskAnd` → `Select` 抑制重叠框）见 `ops-cv/.../nms_with_mask_regbase_multiprocess.h:375`。

![MaskDist 存取](images/sel-e-mask.svg)

### 非对齐搬运 `StoreUnAlign` + `Post`

> **用于**写出地址不是 32 B 对齐、或每轮写出元素数不定的场景（归约后变长写回、镜像 / pad）。普通 `StoreAlign` 要求地址对齐且一次写满整数个 DataBlock；`StoreUnAlign` 用一个内部 `UnalignReg` 暂存器，把不满一块的残余攒起来、够 32 B 再刷出。

**三件套必须完整，漏一步就丢数据**（`src/scene10_unalign.asc`，每轮写 50 个、共三轮）：

```cpp
AscendC::Reg::UnalignReg ureg;                            // ① 暂存器

for (uint16_t i = 0; i < ROUNDS; i++) {
    uint32_t cnt = PER_ROUND;                             // 50，非块整数倍
    mask = AscendC::Reg::UpdateMask<float>(cnt);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
        vreg, xAddr + i * VL_B32);                           // 读侧保持 32 B 对齐

    AscendC::Reg::StoreUnAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
        yAddr, vreg, ureg, PER_ROUND);                    // ② 累积，yAddr 自动推进
}
AscendC::Reg::StoreUnAlignPost(yAddr, ureg, 0);           // ③ 刷尾，漏掉则尾部丢失
```

![非对齐搬运三件套](images/sel-unalign.svg)

实测输出（x[i]=i，每轮取所在 VL 的前 50 个 lane）：

```text
[scene10] out: y[0]=  0.000000 y[49]= 49.000000 y[50]= 64.000000 y[99]=113.000000 y[100]=128.000000
[scene10] tail: y[144]=172.000000 y[149]=177.000000 | y[150]= -1.000000 (sentinel -1)
```

三轮 50 个无空洞拼接（y[50] 直接衔接第二轮首元素 64），尾段 y[144..149] 由 `StoreUnAlignPost` 刷出——三件套缺 ③ 时该 6 个元素滞留在暂存器中，不崩溃、不报错。`POST_MODE_UPDATE` 自动推进 `yAddr`，循环内不能再手动加偏移；`count` 每轮可变正是 UnAlign 的典型适用场景。读侧仍走对齐通路，本例编写时把搬入偏移写成 `+ i * 50`（200 B，非 32 B 对齐）当场触发设备异常 507035——**非对齐能力仅限写侧，读侧的对齐约束依然生效**。`StoreUnAlign` 有三种重载（`uint32_t` stride / `AddrReg` 偏移 / 无后处理），各配同形参的 `StoreUnAlignPost`，不可混用。真实代码 `ops-nn/.../dynamic_mx_quant_tail_axis_fp8.h:337` 按 block 求最大 exp 后变长写回，三件套完整落地。

> 完整源码：[`src/scene8_dintlv_intlv.asc`](src/scene8_dintlv_intlv.asc)　[`src/scene9_mask_dist.asc`](src/scene9_mask_dist.asc)　[`src/scene10_unalign.asc`](src/scene10_unalign.asc)

---

## 寻址模式（与上面五个模式正交）

寻址模式管的是「地址怎么推进」，与选哪种 `dist` 完全独立，可任意组合：

| 寻址模式 | 用量 | 适用 | 代价 / 注意 |
|---|---:|---|---|
| `POST_MODE_NORMAL`（默认） | — | 偏移简单（`+ i*VL`） | 偏移在调用点手动算 |
| `POST_MODE_UPDATE` | **2551** | 循环内多个地址同步线性前进 | 地址按引用被改写，**不能再手动 `+offset`** |
| `AddrReg` | 483 | 多维 / 不连续偏移、gather | `CreateAddrReg` 在寄存器内算偏移，多占一个地址寄存器 |

`POST_MODE_UPDATE` 用量 2551，比任何单个 `dist` 都高——生产代码里基本是默认做法。最常见的坑是用了它之后又手动 `yAddr + i*VL`，地址被推进两次导致错位：**用了 UPDATE 就别再手动加偏移。**

---

## 注意事项与常见误区

各模式的坑已在对应小节展开，跨模式的共性规律汇总于此。除对齐违例当场抛 507035 外，其余条目的共同点是静默出错——编译通过、运行不崩，只有结果不对。标注「实测」的条目在 `src/` 对应示例中有复现：

| 误区 | 后果 | 正解 | 验证 |
|---|---|---|---|
| 对齐搬入地址非 32 B 对齐 | 设备异常 507035 | 偏移按 VL 推进，变长收尾交给 mask / UnAlign | 实测（scene10） |
| 搬出尾轮 mask 全开 | 垃圾 lane 覆盖有效数据 | `UpdateMask<T>(count)` 收尾 | 实测（scene1） |
| 升降位按存储类型生成 mask | 开出不存在的 lane | mask 跟计算类型走（`UpdateMask<float>`） | scene2 全程宽类型 |
| 降位 Cast `SatMode::UNKNOWN` | 编译失败（vcvt 静态断言） | 显式 `NO_SAT`/`SAT` | 实测（scene2） |
| `FIRST_ELEMENT` 输出偏移按 VL 推进 | 输出稀疏、归并读到零 | 偏移 `+i` 逐元素推进 | 实测（scene4） |
| `dataBlockStride` 按字节/元素填 | 写到错误位置 | 单位是 32 B 块 | 实测（scene6） |
| 当作 BLOCK_COPY 的 mask 按块算 | 半块按整块处理、多写或少写 | mask 是逐元素粒度 | 实测（scene6） |
| 按 64 索引规划单轮 E2B+Gather | 只消费前 8 个索引、输出块内重复 | 以 8 索引为步长多轮 | 实测（scene7） |
| `POST_MODE_UPDATE` 后再手动加偏移 | 地址双重推进、错位 | UPDATE 与手动偏移二选一 | — |
| `StoreUnAlign` 漏 `StoreUnAlignPost` | 尾部残余丢失 | 三件套配齐，重载配对 | scene10 尾段由 Post 刷出 |
| 同地址存 mask 后立即读回 | 概率性读到全 0 | 插 `LocalMemBar<VEC_STORE, VEC_LOAD>` | 实测（scene9） |
| mask 位流按「b32 每 lane 1 bit」解释 | 位偏移错 4 倍 | 固定 256 bit、b8 粒度，b32 每 4 bit 取 1 | 实测（scene9） |
| 大张量 gather 用 16 位索引 | 索引溢出 | b32/b64 数据配 u32 索引 | — |

---

## 附录 · API 反查

从代码里看到某个 `dist` 时反查它属于哪个模式、用在哪里。频次按存量代码中的 `DataCopy` / `DataCopyGather` 别名写法统计，新代码推荐等价的 `LoadAlign` / `StoreAlign` / `Gather`。频次按**具体变体**计（与正文模式标题处的族计数粒度不同）；其中「按索引取数」场景的 `DataCopyGather`（325）与 `DIST_E2B`（113）是同一条流水的两步，分列计数。枚举全集见 `include/basic_api/reg_compute/kernel_reg_compute_utils.h`（`LoadDist` 还含 `US/DS`、`SPLT2/4CHN` 等通道类成员，`StoreDist` 含 `NORM_B8/B16/B32`、`PACK_B64`、`MRG2/4CHN` 等，本指南未覆盖的成员按需查头文件）。

| API / `dist` | 模式 | 频次 | 代表出处 |
|---|---|---:|---|
| `DataCopy` + `DIST_NORM`（默认） | 常规·搬入 | 919+ | add_n_regbase.h:207 |
| `DataCopy` + `DIST_NORM*` | 常规·搬出 | 1117 | add_n_regbase.h:210 |
| `LoadDist::DIST_UNPACK_B16` | 升降位·搬入 | 1293 | exp_dag.h:80 |
| `LoadDist::DIST_UNPACK4_B8` | 升降位·搬入(4×) | 216 | anti_mx_quant_tail_axis.h:690 |
| `StoreDist::DIST_PACK_B32` | 升降位·搬出 | 439 | exp_dag.h:88 |
| `StoreDist::DIST_PACK4_B32` | 升降位·搬出(4×) | 348 | u32→u8 窄化 |
| `StoreDist::DIST_PACK_B16` | 升降位·搬出 | 58 | dynamic_mx_quant_tail_axis_fp8.h:522 |
| `StoreDist::DIST_FIRST_ELEMENT_*` | 归约·写出 | 846 | reduce_var_twopass.h:365 |
| `LoadDist::DIST_BRC_*` | 广播·铺满 | 993 | nms_with_mask_regbase_base.h:52 |
| `DataCopyMode::DATA_BLOCK_COPY` | 非连续·跳跃 | 502 | gather_nd_full_load_vgather.h:269 |
| `DataCopyGather` | 非连续·索引取数 | 325 | gather_nd_full_load_vgather.h:268 |
| `LoadDist::DIST_E2B_*` | 非连续·索引扩块 | 113 | gather_nd_full_load_vgather.h:265 |
| `LoadDist::DIST_DINTLV_*` | 特殊·交织拆 | 223 | abs_complex_dag.h:115 |
| `StoreDist::DIST_INTLV_*` | 特殊·交织合 | 63 | random_kernel_base.h:307 |
| `MaskDist::DIST_NORM/US/DS/PACK` | 特殊·掩码存取 | 269 | nms_with_mask_regbase_multiprocess.h:375 |
| `StoreUnAlign` + `Post` | 特殊·非对齐 | ~156 | dynamic_mx_quant_tail_axis_fp8.h:337 |
| `MaskGenWithRegTensor` | 特殊·寄存器→mask | 1 | vf_antiquant_w4.h:916 |
| `POST_MODE_UPDATE` | 寻址（正交） | 2551 | — |
| `AddrReg` | 寻址（正交） | 483 | — |

注：`MaskDist` 枚举中 `DIST_PACK` 与 `DIST_US` 取同一枚举值（1），读侧支持 `NORM/US/DS`、写侧支持 `NORM/PACK`。

## 示例工程构建

```bash
cd cann-samples-tpc
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --parallel        # 或 --target scene1_dist_norm
./build/Samples/1_Features/memory_optimization/reg_data_movement/scene1_dist_norm
```

10 个示例（scene1–scene10）一一对应正文场景，结构一致：`__simd_vf__` 函数承载搬运指令，kernel 桥接 GM ↔ UB 并以 `AscendC::PRINTF` 打印关键数据，host 侧用确定性输入与预填校验值做逐元素校验，输出 `PASSED` / `FAILED`。
