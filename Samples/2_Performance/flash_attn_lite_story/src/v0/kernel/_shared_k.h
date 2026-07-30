/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <basic_api/reg_compute/kernel_reg_compute_utils.h>
#include <kernel_operator.h>

namespace FALite {

// ── 常量 ──
constexpr uint32_t C0_BYTES = 32;
constexpr uint32_t VECTOR_REG_WIDTH = 256;
constexpr uint32_t VL_B32 = VECTOR_REG_WIDTH / sizeof(float);
constexpr uint32_t VL_B16 = VECTOR_REG_WIDTH / sizeof(bfloat16_t);
constexpr uint32_t B16_PER_DATABLOCK = C0_BYTES / sizeof(bfloat16_t);
constexpr float FLOAT_LOWEST = -3.402823466e+38F;
constexpr AscendC::TEventID STATIC_EVENT_ID0 = 0;
constexpr AscendC::TEventID STATIC_EVENT_ID1 = 1;
constexpr AscendC::TEventID STATIC_EVENT_ID2 = 2;
constexpr AscendC::FixpipeConfig PFA_CFG_UB = {AscendC::CO2Layout::ROW_MAJOR, true};
constexpr AscendC::FixpipeConfig PFA_CFG_GM = {AscendC::CO2Layout::ROW_MAJOR, false};
constexpr uint8_t GROUP_CROSS_MODE = 2;
#ifdef SIM_COMPATIBLE
constexpr uint8_t PAIR_CROSS_MODE = 4;
constexpr uint16_t AIV1_FLAG_OFFSET = 16;
#endif

static constexpr AscendC::Reg::CastTrait castTraitZero = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};
static constexpr AscendC::Reg::CastTrait castTraitOne = {
    AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};

// ── Helper ──
template <AscendC::HardEvent E>
__aicore__ inline void SetWaitFlag(const AscendC::TEventID eventId)
{
    AscendC::SetFlag<E>(eventId);
    AscendC::WaitFlag<E>(eventId);
}

template <typename R, typename T1, typename T2>
__aicore__ inline R CeilDiv(T1 x, T2 y)
{
    if (y == 0)
        return static_cast<R>(0);
    return static_cast<R>((x + y - 1) / y);
}

template <typename R, typename T1, typename T2>
__aicore__ inline R CeilAlign(T1 x, T2 base)
{
    return static_cast<R>(CeilDiv<R, T1, T2>(x, base) * base);
}

template <typename T>
__aicore__ inline uint32_t C0ElemNum()
{
    return C0_BYTES / sizeof(T);
}

// ── Cube GEMM helper ──
template <typename T>
__aicore__ inline void CopyGmToL1(
    const AscendC::LocalTensor<T>& dstL1Local, const AscendC::GlobalTensor<T>& srcGlobal, uint32_t tileRow,
    uint32_t tileCol, uint32_t gmColStride)
{
    using namespace AscendC;
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = tileRow;
    p.dValue = tileCol;
    p.srcNdMatrixStride = 1;
    p.srcDValue = gmColStride;
    p.dstNzC0Stride = CeilAlign<uint16_t>(tileRow, BLOCK_CUBE);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 1;
    DataCopy(dstL1Local, srcGlobal, p);
}

template <typename T>
__aicore__ inline void CopyL1ToL0A(
    const AscendC::LocalTensor<T>& dstL0ALocal, const AscendC::LocalTensor<T>& srcL1Local, uint32_t l1Row,
    uint32_t l1Col, uint32_t l0Row, uint32_t l0Col, bool transpose = false)
{
    using namespace AscendC;
    LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = CeilDiv<uint16_t>(l0Row, BLOCK_CUBE);
    p.kStep = CeilDiv<uint16_t>(l0Col, C0ElemNum<T>());
    p.srcStride = CeilDiv<int32_t>(l1Row, BLOCK_CUBE);
    p.dstStride = p.mStep;
    p.ifTranspose = transpose;
    LoadData(dstL0ALocal, srcL1Local, p);
}

template <typename T>
__aicore__ inline void CopyL1ToL0B(
    const AscendC::LocalTensor<T>& dstL0BLocal, const AscendC::LocalTensor<T>& srcL1Local, uint32_t l1Row,
    uint32_t l1Col, uint32_t l0Row, uint32_t l0Col, bool transpose)
{
    using namespace AscendC;
    LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = CeilDiv<uint16_t>(l0Row, BLOCK_CUBE);
    p.kStep = CeilDiv<uint16_t>(l0Col, BLOCK_CUBE);
    p.srcStride = CeilDiv<int32_t>(l1Row, BLOCK_CUBE);
    p.dstStride = CeilDiv<uint16_t>(l0Col, BLOCK_CUBE);
    p.ifTranspose = transpose;
    LoadData(dstL0BLocal, srcL1Local, p);
}

template <typename dstT, typename aT, typename bT>
__aicore__ inline void CubeMmad(
    const AscendC::LocalTensor<dstT>& dstL0CLocal, const AscendC::LocalTensor<aT>& aL0ALocal,
    const AscendC::LocalTensor<bT>& bL0BLocal, uint32_t m, uint32_t n, uint32_t k, bool initC)
{
    using namespace AscendC;
    MmadParams p;
    p.m = m;
    p.n = n;
    p.k = k;
    p.cmatrixSource = false;
    p.cmatrixInitVal = initC;
    p.unitFlag = 0;
    Mmad(dstL0CLocal, aL0ALocal, bL0BLocal, p);
}

template <typename dstT, typename srcT>
__aicore__ inline void FixpipeL0CToGM(
    const AscendC::GlobalTensor<dstT>& dstGlobal, const AscendC::LocalTensor<srcT>& srcL0CLocal, uint32_t m, uint32_t n,
    uint32_t gmStride = 0)
{
    using namespace AscendC;
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> p;
    constexpr uint32_t FA = 8, FMA = 2;
    p.nSize = CeilAlign<uint32_t>(n, FA);
    p.mSize = CeilAlign<uint32_t>(m, FMA);
    p.srcStride = CeilAlign<uint32_t>(p.mSize, BLOCK_CUBE);
    p.dstStride = gmStride != 0 ? gmStride : CeilAlign<uint32_t>(p.nSize, BLOCK_CUBE);
    p.dualDstCtl = 1;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    Fixpipe<dstT, srcT, PFA_CFG_GM>(dstGlobal, srcL0CLocal, p);
}

// ── Pᵀ→L0A 转置 ──
template <typename TilingData>
__aicore__ inline void LoadPTransToL0A(
    AscendC::LocalTensor<bfloat16_t>& pL1Local, AscendC::LocalTensor<bfloat16_t>& aL0ALocal,
    AscendC::GlobalTensor<bfloat16_t>& pGlobal, uint64_t pOff, const TilingData& data)
{
    using namespace AscendC;
    Nd2NzParams pp;
    pp.ndNum = 1;
    pp.nValue = data.bc;
    pp.dValue = data.br;
    pp.srcNdMatrixStride = 1;
    pp.srcDValue = data.br;
    pp.dstNzC0Stride = CeilAlign<uint16_t>(data.bc, BLOCK_CUBE);
    pp.dstNzNStride = 1;
    pp.dstNzMatrixStride = 1;
    DataCopy(pL1Local, pGlobal[pOff], pp);
    PipeBarrier<PIPE_ALL>();
    CopyL1ToL0A<bfloat16_t>(aL0ALocal, pL1Local, data.bc, data.br, data.br, data.bc, true);
    PipeBarrier<PIPE_ALL>();
}

// ── VF online softmax (v0/v1 共享) ──
template <bool IS_FIRST_ITER>
__simd_vf__ inline void OnlineColwiseSoftmaxVF(
    __ubuf__ float* sUBAddr, __ubuf__ float* mUBAddr, __ubuf__ float* lUBAddr, __ubuf__ float* alphaUBAddr,
    const uint16_t halfBr, const uint16_t bc, const float scale)
{
    using namespace AscendC;
    Reg::RegTensor<float> s, maxReg, mOld, alpha, expReg, sumReg, lOld;
    Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Duplicate(maxReg, FLOAT_LOWEST, all);
    for (uint16_t r = 0; r < bc; ++r) {
        Reg::LoadAlign(s, sUBAddr + r * halfBr);
        Reg::Muls(s, s, scale, all);
        Reg::Max(maxReg, maxReg, s, all);
    }
    if constexpr (!IS_FIRST_ITER) {
        Reg::LoadAlign(mOld, mUBAddr);
        Reg::Max(maxReg, maxReg, mOld, all);
        Reg::Sub(alpha, mOld, maxReg, all);
        Reg::Exp(alpha, alpha, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(alphaUBAddr, alpha, all);
    }
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(mUBAddr, maxReg, all);
    Reg::Duplicate(sumReg, 0.0f, all);
    for (uint16_t r = 0; r < bc; ++r) {
        Reg::LoadAlign(s, sUBAddr + r * halfBr);
        Reg::Muls(s, s, scale, all);
        Reg::FusedExpSub(expReg, s, maxReg, all);
        Reg::Add(sumReg, sumReg, expReg, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(sUBAddr + r * halfBr, expReg, all);
    }
    if constexpr (IS_FIRST_ITER) {
        Reg::Duplicate(alpha, 1.0f, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(alphaUBAddr, alpha, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(lUBAddr, sumReg, all);
    } else {
        Reg::LoadAlign(lOld, lUBAddr);
        Reg::Mul(lOld, lOld, alpha, all);
        Reg::Add(lOld, lOld, sumReg, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(lUBAddr, lOld, all);
    }
}

__aicore__ inline void RunOnlineSoftmax(
    AscendC::LocalTensor<float>& sUBLocal, AscendC::LocalTensor<float>& mUBLocal, AscendC::LocalTensor<float>& lUBLocal,
    AscendC::LocalTensor<float>& alphaUBLocal, const uint16_t halfBr, const uint16_t bc, const float scale,
    const bool isFirstIter)
{
    using namespace AscendC;
    __ubuf__ float* sUBAddr = reinterpret_cast<__ubuf__ float*>(sUBLocal.GetPhyAddr());
    __ubuf__ float* mUBAddr = reinterpret_cast<__ubuf__ float*>(mUBLocal.GetPhyAddr());
    __ubuf__ float* lUBAddr = reinterpret_cast<__ubuf__ float*>(lUBLocal.GetPhyAddr());
    __ubuf__ float* alphaUBAddr = reinterpret_cast<__ubuf__ float*>(alphaUBLocal.GetPhyAddr());
    if (isFirstIter)
        asc_vf_call<OnlineColwiseSoftmaxVF<true>>(sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, halfBr, bc, scale);
    else
        asc_vf_call<OnlineColwiseSoftmaxVF<false>>(sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, halfBr, bc, scale);
}

// ── AIC workspace ──
template <typename TilingData>
struct AicCubeWorkspace {
    AscendC::LocalTensor<bfloat16_t> qL1, kL1, vL1, pL1, aL0A, bL0B;
    AscendC::LocalTensor<float> l0C;
};

template <typename TilingData>
__aicore__ inline AicCubeWorkspace<TilingData> MakeAicCubeWorkspace(const TilingData& data)
{
    using namespace AscendC;
    const auto& aic = data.layoutAIC;
    AicCubeWorkspace<TilingData> ws;
    ws.qL1 = LocalTensor<bfloat16_t>(TPosition::A1, aic.qL1Addr, aic.qL1Elems);
    ws.kL1 = LocalTensor<bfloat16_t>(TPosition::A1, aic.kL1Addr, aic.kL1Elems);
    ws.vL1 = LocalTensor<bfloat16_t>(TPosition::A1, aic.vL1Addr, aic.vL1Elems);
    ws.pL1 = LocalTensor<bfloat16_t>(TPosition::A1, aic.pL1Addr, aic.pL1Elems);
    ws.aL0A = LocalTensor<bfloat16_t>(TPosition::A2, aic.aL0AAddr, aic.aL0AElems);
    ws.bL0B = LocalTensor<bfloat16_t>(TPosition::B2, aic.bL0BAddr, aic.bL0BElems);
    ws.l0C = LocalTensor<float>(TPosition::CO1, aic.cL0CAddr, aic.cL0CElems);
    return ws;
}

// v0/v1 共享: online softmax + Cast → P(BF16), 返回 sUB 被 exp 覆写后可复用.
__aicore__ inline void SoftmaxAndCastP(
    AscendC::LocalTensor<float>& sUBLocal, AscendC::LocalTensor<float>& mUBLocal, AscendC::LocalTensor<float>& lUBLocal,
    AscendC::LocalTensor<float>& alphaUBLocal, AscendC::LocalTensor<bfloat16_t>& pUBLocal, const uint16_t halfBr,
    const uint16_t bc, const float scale, const bool isFirst)
{
    RunOnlineSoftmax(sUBLocal, mUBLocal, lUBLocal, alphaUBLocal, halfBr, bc, scale, isFirst);
    Cast<bfloat16_t, float>(pUBLocal, sUBLocal, AscendC::RoundMode::CAST_RINT, bc * halfBr);
}

// v0/v1 共享: Cast 后的 P(BF16) → GM. 用于 SoftmaxAndWriteP 的公共尾段.
__aicore__ inline void WritePToGM(
    AscendC::LocalTensor<bfloat16_t>& pUBLocal, AscendC::GlobalTensor<bfloat16_t>& pGlobal, uint64_t pHead,
    const uint16_t bc, const uint16_t halfBr, const uint16_t br)
{
    using namespace AscendC;
    SetWaitFlag<HardEvent::V_MTE3>(STATIC_EVENT_ID1);
    DataCopy(
        pGlobal[pHead], pUBLocal,
        DataCopyParams(
            static_cast<uint16_t>(bc), static_cast<uint16_t>(halfBr * sizeof(bfloat16_t) / C0_BYTES), 0,
            static_cast<uint16_t>((br - halfBr) * sizeof(bfloat16_t) / C0_BYTES)));
    SetWaitFlag<HardEvent::MTE3_V>(STATIC_EVENT_ID1);
}

// v0/v1 共享: O_acc = alpha * O_acc + ΔO
__simd_vf__ inline void OnlineUpdateVF(
    __ubuf__ float* oAccUBAddr, __ubuf__ float* oDeltaUBAddr, __ubuf__ float* alphaUBAddr, const uint16_t halfBr,
    const uint16_t d)
{
    using namespace AscendC;
    Reg::RegTensor<float> alpha, pre, cur, mul, add;
    Reg::MaskReg pregAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    constexpr uint16_t floatRepSize = VL_B32;
    const uint16_t dLoops = d / floatRepSize;
    for (uint16_t i = 0; i < halfBr; ++i) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(alpha, alphaUBAddr + i);
        for (uint16_t j = 0; j < dLoops; ++j) {
            Reg::LoadAlign(pre, oAccUBAddr + i * d + j * floatRepSize);
            Reg::LoadAlign(cur, oDeltaUBAddr + i * d + j * floatRepSize);
            Reg::Mul(mul, alpha, pre, pregAll);
            Reg::Add(add, mul, cur, pregAll);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(oAccUBAddr + i * d + j * floatRepSize, add, pregAll);
        }
    }
}
// v0/v1 共享: O_acc 更新核心 (OnlineUpdateVF). 调用方须确保 ΔO 已在 oDeltaUBLocal.
__aicore__ inline void AccumulateDeltaOCore(
    AscendC::LocalTensor<float>& oAccUBLocal, AscendC::LocalTensor<float>& oDeltaUBLocal,
    AscendC::LocalTensor<float>& alphaUBLocal, const uint16_t halfBr, const uint16_t d)
{
    asc_vf_call<OnlineUpdateVF>(
        reinterpret_cast<__ubuf__ float*>(oAccUBLocal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(oDeltaUBLocal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(alphaUBLocal.GetPhyAddr()), halfBr, d);
}

// v0/v1 共享: ProcessOneTaskAIV 初始化 (m/l/O_acc 清零 + 偏移计算).
__aicore__ inline void InitTaskAIV(
    AscendC::LocalTensor<float>& oAccUBLocal, AscendC::LocalTensor<float>& mUBLocal,
    AscendC::LocalTensor<float>& lUBLocal, uint64_t& pHead, uint64_t& outHead, uint32_t taskId, uint32_t base,
    uint32_t halfBr, uint32_t d, const FlashAttnLiteTilingData& data)
{
    using namespace AscendC;
    Duplicate<float>(oAccUBLocal, 0.0f, halfBr * d);
    Duplicate<float>(mUBLocal, FLOAT_LOWEST, halfBr);
    Duplicate<float>(lUBLocal, 0.0f, halfBr);
    const uint32_t batchIdx = taskId / data.tr, tileIdx = taskId % data.tr;
    pHead = static_cast<uint64_t>(taskId) * data.bc * data.br + static_cast<uint64_t>(base);
    outHead = static_cast<uint64_t>(batchIdx) * data.seqLen * d + static_cast<uint64_t>(tileIdx) * data.br * d +
              static_cast<uint64_t>(base) * d;
}

// v0/v1 共享: O_acc/l → BF16 (DINTLV, D=128)
static constexpr AscendC::Reg::CastTrait _castTraitZero = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};
static constexpr AscendC::Reg::CastTrait _castTraitOne = {
    AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};

__simd_vf__ inline void FusedDivCastVF(
    __ubuf__ bfloat16_t* dstUBAddr, __ubuf__ float* srcUBAddr, __ubuf__ float* lUBAddr, const uint16_t halfBr,
    const uint16_t d)
{
    using namespace AscendC;
    Reg::RegTensor<float> lReg, srcEven, srcOdd, divEven, divOdd;
    Reg::RegTensor<bfloat16_t> castEven, castOdd, packed;
    Reg::MaskReg pregAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregAllB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    constexpr uint16_t dintlvLen = VL_B32 * 2;
    const uint16_t dLoops = d / dintlvLen;
    for (uint16_t i = 0; i < halfBr; ++i) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(lReg, lUBAddr + i);
        for (uint16_t c = 0; c < dLoops; ++c) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(srcEven, srcOdd, srcUBAddr + i * d + c * dintlvLen);
            Reg::Div(divEven, srcEven, lReg, pregAll);
            Reg::Div(divOdd, srcOdd, lReg, pregAll);
            Reg::Cast<bfloat16_t, float, _castTraitZero>(castEven, divEven, pregAllB16);
            Reg::Cast<bfloat16_t, float, _castTraitOne>(castOdd, divOdd, pregAllB16);
            Reg::Or(
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(packed),
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(castEven),
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(castOdd), pregAllB16);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_NORM_B32>(
                dstUBAddr + i * d + c * dintlvLen, packed, pregAllB16);
        }
    }
}

// v0/v1 共享: 最终 O_acc/l → BF16 → GM. pUB 复用为 O 输出缓冲.
__aicore__ inline void FinalOutput(
    AscendC::LocalTensor<float>& oAccUBLocal, AscendC::LocalTensor<float>& lUBLocal,
    AscendC::LocalTensor<bfloat16_t>& pUBLocal, AscendC::GlobalTensor<bfloat16_t>& outGlobal, uint64_t outOff,
    const uint16_t halfBr, const uint16_t d)
{
    using namespace AscendC;
    asc_vf_call<FusedDivCastVF>(
        reinterpret_cast<__ubuf__ bfloat16_t*>(pUBLocal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(oAccUBLocal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(lUBLocal.GetPhyAddr()), halfBr, d);
    SetWaitFlag<HardEvent::V_MTE3>(STATIC_EVENT_ID1);
    DataCopy(outGlobal[outOff], pUBLocal, halfBr * d);
    SetWaitFlag<HardEvent::MTE3_V>(STATIC_EVENT_ID1);
}

} // namespace FALite
