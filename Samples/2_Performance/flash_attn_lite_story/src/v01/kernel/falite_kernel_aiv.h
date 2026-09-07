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

#include "falite_kernel_common.h"

namespace FALite {

// S、DeltaO 分别由 MTE2 搬入 UB；P 与最终输出复用一块 UB。
constexpr MutexID MUTEX_P_UB = 0;
constexpr MutexID MUTEX_S_UB = 1;
constexpr MutexID MUTEX_DELTA_O_UB = 2;

// ZERO/ONE layout 分别写 BF16 寄存器的低半段和高半段, 随后用 Or 合并.
static constexpr AscendC::Reg::CastTrait castTraitZero = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};
static constexpr AscendC::Reg::CastTrait castTraitOne = {
    AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};

// 输入 Sᵀ[Bc, halfBr] 使用 DN 布局, 每行占 1 个 64-lane FP32 寄存器.
// 沿 Bc 维逐 lane 计算 max/sum; 每个 lane 对应本 AIV 的 1 个 Q 行.
template <bool IS_FIRST_ITER, bool APPLY_CAUSAL_MASK>
__simd_vf__ inline void OnlineColwiseSoftmaxVF(
    __ubuf__ float* sUBAddr, __ubuf__ float* mUBAddr, __ubuf__ float* lUBAddr, __ubuf__ float* alphaUBAddr,
    const uint16_t halfBr, const uint16_t bc, const uint16_t validBc, const float scale, const uint16_t queryColBegin)
{
    using namespace AscendC;
    Reg::RegTensor<float> s, maxReg, mOld, alpha, expReg, sumReg, lOld;
    Reg::RegTensor<int32_t> queryIndex;
    Reg::RegTensor<float> maskedScore;
    Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg invalid;
    if constexpr (APPLY_CAUSAL_MASK) {
        Reg::Arange(queryIndex, static_cast<int32_t>(queryColBegin));
        Reg::Duplicate(maskedScore, FLOAT_LOWEST, all);
    }
    Reg::Duplicate(maxReg, FLOAT_LOWEST, all);
    for (uint16_t r = 0; r < validBc; ++r) {
        Reg::LoadAlign(s, sUBAddr + r * halfBr);
        Reg::Muls(s, s, scale, all);
        if constexpr (APPLY_CAUSAL_MASK) {
            Reg::CompareScalar<int32_t, CMPMODE::LT>(invalid, queryIndex, static_cast<int32_t>(r), all);
            Reg::Select(s, maskedScore, s, invalid);
        }
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
    for (uint16_t r = 0; r < validBc; ++r) {
        Reg::LoadAlign(s, sUBAddr + r * halfBr);
        Reg::Muls(s, s, scale, all);
        if constexpr (APPLY_CAUSAL_MASK) {
            Reg::CompareScalar<int32_t, CMPMODE::LT>(invalid, queryIndex, static_cast<int32_t>(r), all);
            Reg::Select(s, maskedScore, s, invalid);
        }
        Reg::FusedExpSub(expReg, s, maxReg, all);
        Reg::Add(sumReg, sumReg, expReg, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(sUBAddr + r * halfBr, expReg, all);
    }
    // C2 仍按完整 Bc 做矩阵乘，尾块中不存在的 K/V 行对应权重必须为 0。
    Reg::Duplicate(expReg, 0.0f, all);
    for (uint16_t r = validBc; r < bc; ++r) {
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

// 计算 OAcc = alpha * OAcc + DeltaO. OAcc 初始为 0, 无需单独处理首轮.
// alpha 通过 BRC 按行广播.
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

// 计算 OAcc / l, 并转为 DataCopy 写回 GM 所需的 BF16 ND 布局.
// l 通过 BRC 按行广播; DINTLV_B32 每次读取 2 个 FP32 寄存器.
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
            Reg::Cast<bfloat16_t, float, castTraitZero>(castEven, divEven, pregAllB16);
            Reg::Cast<bfloat16_t, float, castTraitOne>(castOdd, divOdd, pregAllB16);
            Reg::Or(
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(packed),
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(castEven),
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(castOdd), pregAllB16);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_NORM_B32>(
                dstUBAddr + i * d + c * dintlvLen, packed, pregAllB16);
        }
    }
}

// V1：更新分块 Softmax 状态，并将未归一化权重 Pᵀ 从 FP32 DN 转为 BF16 DN。
template <bool CAUSAL_MASK>
__aicore__ inline void VectorStage1(
    AscendC::LocalTensor<float>& sUBLocal, AscendC::LocalTensor<float>& mUBLocal, AscendC::LocalTensor<float>& lUBLocal,
    AscendC::LocalTensor<float>& alphaUBLocal, AscendC::LocalTensor<bfloat16_t>& pUBLocal,
    const FlashAttnLiteTilingData& data, uint32_t j, uint32_t qTileIdx, uint32_t subAivIdx)
{
    using namespace AscendC;

    if ASCEND_IS_AIV {
        __ubuf__ float* sUBAddr = reinterpret_cast<__ubuf__ float*>(sUBLocal.GetPhyAddr());
        __ubuf__ float* mUBAddr = reinterpret_cast<__ubuf__ float*>(mUBLocal.GetPhyAddr());
        __ubuf__ float* lUBAddr = reinterpret_cast<__ubuf__ float*>(lUBLocal.GetPhyAddr());
        __ubuf__ float* alphaUBAddr = reinterpret_cast<__ubuf__ float*>(alphaUBLocal.GetPhyAddr());
        const uint16_t halfBr = static_cast<uint16_t>(data.br / 2);
        const uint16_t bc = static_cast<uint16_t>(data.bc);
        const uint16_t validBc = static_cast<uint16_t>(GetTileValidRows(data.seqLen, j, data.bc));
        const uint16_t queryColBegin = static_cast<uint16_t>(subAivIdx * halfBr);
        if constexpr (CAUSAL_MASK) {
            // 未来 K/V 整块已被裁掉；这里只屏蔽对角块内的上三角。
            const bool isDiagonalTile = (j == qTileIdx);
            if (j == 0) {
                if (isDiagonalTile) {
                    asc_vf_call<OnlineColwiseSoftmaxVF<true, true>>(
                        sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, halfBr, bc, validBc, data.scale, queryColBegin);
                } else {
                    asc_vf_call<OnlineColwiseSoftmaxVF<true, false>>(
                        sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, halfBr, bc, validBc, data.scale, queryColBegin);
                }
            } else if (isDiagonalTile) {
                asc_vf_call<OnlineColwiseSoftmaxVF<false, true>>(
                    sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, halfBr, bc, validBc, data.scale, queryColBegin);
            } else {
                asc_vf_call<OnlineColwiseSoftmaxVF<false, false>>(
                    sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, halfBr, bc, validBc, data.scale, queryColBegin);
            }
        } else if (j == 0) {
            asc_vf_call<OnlineColwiseSoftmaxVF<true, false>>(
                sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, halfBr, bc, validBc, data.scale, queryColBegin);
        } else {
            asc_vf_call<OnlineColwiseSoftmaxVF<false, false>>(
                sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, halfBr, bc, validBc, data.scale, queryColBegin);
        }
        Mutex::Lock<PIPE_V>(MUTEX_P_UB);
        Cast<bfloat16_t, float>(pUBLocal, sUBLocal, RoundMode::CAST_RINT, halfBr * bc);
        Mutex::Unlock<PIPE_V>(MUTEX_P_UB);
    }
}

// V2: 更新 OAcc = alpha * OAcc + DeltaO.
__aicore__ inline void VectorStage2(
    AscendC::LocalTensor<float>& oAccUBLocal, AscendC::LocalTensor<float>& oDeltaUBLocal,
    AscendC::LocalTensor<float>& alphaUBLocal, const FlashAttnLiteTilingData& data)
{
    using namespace AscendC;

    if ASCEND_IS_AIV {
        const uint32_t halfBr = data.br / 2;
        asc_vf_call<OnlineUpdateVF>(
            reinterpret_cast<__ubuf__ float*>(oAccUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(oDeltaUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(alphaUBLocal.GetPhyAddr()), static_cast<uint16_t>(halfBr),
            static_cast<uint16_t>(HEAD_DIM));
    }
}

template <bool CAUSAL_MASK>
__aicore__ inline void KernelProcessForAIV(
    __gm__ float* sGMAddr, __gm__ bfloat16_t* pGMAddr, __gm__ float* deltaOGMAddr, __gm__ bfloat16_t* outGMAddr,
    FlashAttnLiteTilingData data)
{
    using namespace AscendC;

    if ASCEND_IS_AIV {
        const uint32_t br = data.br, bc = data.bc, halfBr = br / 2;
        constexpr uint32_t d = HEAD_DIM;
        const uint32_t outputHalfElements = halfBr * d;
        GlobalTensor<float> sGlobal, deltaOGlobal;
        GlobalTensor<bfloat16_t> pGlobal, outGlobal;
        sGlobal.SetGlobalBuffer(sGMAddr);
        pGlobal.SetGlobalBuffer(pGMAddr);
        deltaOGlobal.SetGlobalBuffer(deltaOGMAddr);
        outGlobal.SetGlobalBuffer(outGMAddr);

        // 两路 AIV 分别处理 Q 块的前、后 Br/2 行，并共享同组 AIC。
        const uint32_t subAivIdx = GetSubBlockIdx();
        const uint32_t aicIdx = GetBlockIdx() / GetSubBlockNum();
        const auto& aiv = data.layoutAIV;
        LocalTensor<float> sUBLocal(TPosition::VECCALC, aiv.sUBAddr, aiv.sUBElems);
        LocalTensor<float> oDeltaUBLocal(TPosition::VECCALC, aiv.oDeltaUBAddr, aiv.oDeltaUBElems);
        LocalTensor<float> oAccUBLocal(TPosition::VECCALC, aiv.oAccUBAddr, aiv.oAccUBElems);
        LocalTensor<bfloat16_t> pUBLocal(TPosition::VECCALC, aiv.pUBAddr, aiv.pUBElems);
        LocalTensor<float> mUBLocal(TPosition::VECCALC, aiv.mUBAddr, aiv.rowStatsUBElems);
        LocalTensor<float> lUBLocal(TPosition::VECCALC, aiv.lUBAddr, aiv.rowStatsUBElems);
        LocalTensor<float> alphaUBLocal(TPosition::VECCALC, aiv.alphaUBAddr, aiv.rowStatsUBElems);

        const uint16_t sBlockLen = static_cast<uint16_t>(halfBr * sizeof(float) / C0_BYTES);
        const uint16_t sSrcStride = static_cast<uint16_t>((br - halfBr) * sizeof(float) / C0_BYTES);
        const uint16_t pBlockLen = static_cast<uint16_t>(halfBr * sizeof(bfloat16_t) / C0_BYTES);
        const uint16_t pDstStride = static_cast<uint16_t>((br - halfBr) * sizeof(bfloat16_t) / C0_BYTES);

        for (uint32_t taskId = aicIdx; taskId < data.numTasks; taskId += data.useAicNum) {
            const uint32_t batchHeadIdx = taskId / data.tr;
            const uint32_t qTileIdx = taskId % data.tr;
            const uint32_t kvTileCount = GetKvTileCount<CAUSAL_MASK>(data, qTileIdx);
            const uint32_t qValidRows = GetTileValidRows(data.seqLen, qTileIdx, br);
            const uint32_t subRowBegin = subAivIdx * halfBr;
            const uint32_t outputRows =
                qValidRows > subRowBegin ? (qValidRows - subRowBegin < halfBr ? qValidRows - subRowBegin : halfBr) : 0;
            const uint64_t intermediateBase =
                static_cast<uint64_t>(taskId) * bc * br + static_cast<uint64_t>(subAivIdx) * halfBr;
            const uint64_t deltaOBase =
                static_cast<uint64_t>(taskId) * br * d + static_cast<uint64_t>(subAivIdx) * outputHalfElements;
            const uint64_t outOffset = static_cast<uint64_t>(batchHeadIdx) * data.seqLen * d +
                                       static_cast<uint64_t>(qTileIdx) * br * d +
                                       static_cast<uint64_t>(subAivIdx) * outputHalfElements;

            Duplicate<float>(oAccUBLocal, 0.0f, outputHalfElements);
            Duplicate<float>(mUBLocal, FLOAT_LOWEST, halfBr);
            Duplicate<float>(lUBLocal, 0.0f, halfBr);

            for (uint32_t j = 0; j < kvTileCount; ++j) {
                WaitAicToAiv<PIPE_MTE2>(FLAG_S_READY);
                Mutex::Lock<PIPE_MTE2>(MUTEX_S_UB);
                DataCopy(
                    sUBLocal, sGlobal[intermediateBase],
                    DataCopyParams(static_cast<uint16_t>(bc), sBlockLen, sSrcStride, 0));
                Mutex::Unlock<PIPE_MTE2>(MUTEX_S_UB);

                Mutex::Lock<PIPE_V>(MUTEX_S_UB);
                VectorStage1<CAUSAL_MASK>(
                    sUBLocal, mUBLocal, lUBLocal, alphaUBLocal, pUBLocal, data, j, qTileIdx, subAivIdx);
                Mutex::Unlock<PIPE_V>(MUTEX_S_UB);

                Mutex::Lock<PIPE_MTE3>(MUTEX_P_UB);
                DataCopy(
                    pGlobal[intermediateBase], pUBLocal,
                    DataCopyParams(static_cast<uint16_t>(bc), pBlockLen, 0, pDstStride));
                Mutex::Unlock<PIPE_MTE3>(MUTEX_P_UB);
                SetAivToAic<PIPE_MTE3>(FLAG_P_READY);

                WaitAicToAiv<PIPE_MTE2>(FLAG_O_READY);
                Mutex::Lock<PIPE_MTE2>(MUTEX_DELTA_O_UB);
                DataCopy(oDeltaUBLocal, deltaOGlobal[deltaOBase], outputHalfElements);
                Mutex::Unlock<PIPE_MTE2>(MUTEX_DELTA_O_UB);
                Mutex::Lock<PIPE_V>(MUTEX_DELTA_O_UB);
                VectorStage2(oAccUBLocal, oDeltaUBLocal, alphaUBLocal, data);
                Mutex::Unlock<PIPE_V>(MUTEX_DELTA_O_UB);
                SetAivToAic<PIPE_V>(FLAG_DONE);
            }

            if (outputRows > 0) {
                Mutex::Lock<PIPE_V>(MUTEX_P_UB);
                asc_vf_call<FusedDivCastVF>(
                    reinterpret_cast<__ubuf__ bfloat16_t*>(pUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(oAccUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(lUBLocal.GetPhyAddr()), static_cast<uint16_t>(outputRows),
                    static_cast<uint16_t>(d));
                Mutex::Unlock<PIPE_V>(MUTEX_P_UB);
                Mutex::Lock<PIPE_MTE3>(MUTEX_P_UB);
                DataCopy(outGlobal[outOffset], pUBLocal, outputRows * d);
                Mutex::Unlock<PIPE_MTE3>(MUTEX_P_UB);
            }
        }
    }
}

} // namespace FALite
