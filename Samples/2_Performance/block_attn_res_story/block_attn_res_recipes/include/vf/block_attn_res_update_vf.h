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

#include "kernel_operator.h"
#include "tensor_api/tensor.h"

namespace BlockAttnResUpdateOps {

namespace Reg = AscendC::Reg;
namespace Te = AscendC::Te;

constexpr uint32_t BARU_FP32_BYTES = sizeof(float);
constexpr uint32_t BARU_BF16_BYTES = sizeof(bfloat16_t);
constexpr uint32_t BARU_FP32_ALIGN_ELEMENTS = 8;
constexpr uint32_t BARU_BF16_ALIGN_ELEMENTS = 16;
constexpr uint32_t BARU_FP32_ALIGN_MASK = BARU_FP32_ALIGN_ELEMENTS - 1U;
constexpr uint32_t BARU_BF16_ALIGN_MASK = BARU_BF16_ALIGN_ELEMENTS - 1U;
// One 256-byte vector register holds 64 FP32 elements; SHIFT and MASK implement division/modulo by 64.
constexpr uint32_t BARU_VREG_FP32_ELEMENTS = 64;
constexpr uint32_t BARU_VREG_FP32_SHIFT = 6;
constexpr uint32_t BARU_VREG_FP32_MASK = BARU_VREG_FP32_ELEMENTS - 1U;
// Each generic VF iteration processes two adjacent vector registers.
constexpr uint32_t BARU_VREG_PAIR_NUM = 2;
constexpr uint32_t BARU_VREG_PAIR_ELEMENTS = BARU_VREG_PAIR_NUM * BARU_VREG_FP32_ELEMENTS;
// Two disjoint UB regions are alternated so transfers for adjacent tiles do not reuse an in-flight buffer.
constexpr uint32_t BARU_BUFFER_NUM = 2;
// The stats UB tensor stores the online-softmax state and the Phase 1 score in separate T-major planes.
constexpr uint32_t BARU_LOGIT_MAX_PLANE_INDEX = 0;
constexpr uint32_t BARU_EXP_SUM_PLANE_INDEX = 1;
constexpr uint32_t BARU_SCORE_PLANE_INDEX = 2;
constexpr uint32_t BARU_STATS_PLANE_NUM = 3;

constexpr AscendC::Reg::CastTrait BARU_CAST_BF16_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait BARU_CAST_FP32_TO_BF16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

// Phase 1 updates partial_block and computes its normalized query score.

__simd_callee__ inline __attribute__((always_inline)) void StoreUpdateScore(
    Reg::RegTensor<float>& squareSumReg, Reg::RegTensor<float>& dotSumReg, __ubuf__ float* stats, uint32_t statsTStride,
    uint16_t tIdx, float invD, float eps, Reg::MaskReg& scalarMask)
{
    Reg::RegTensor<float> rmsReg;
    Reg::RegTensor<float> scoreReg;
    Reg::Muls<float>(squareSumReg, squareSumReg, invD, scalarMask);
    Reg::Adds<float>(squareSumReg, squareSumReg, eps, scalarMask);
    Reg::Sqrt<float>(rmsReg, squareSumReg, scalarMask);
    Reg::Div<float>(scoreReg, dotSumReg, rmsReg, scalarMask);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        stats + BARU_SCORE_PLANE_INDEX * statsTStride + tIdx, scoreReg, scalarMask);
}

__simd_callee__ inline __attribute__((always_inline)) void ReduceUpdateScore(
    Reg::RegTensor<float>& squareReg, Reg::RegTensor<float>& dotReg, __ubuf__ float* stats, uint32_t statsTStride,
    uint16_t tIdx, float invD, float eps, Reg::MaskReg& allMask)
{
    Reg::MaskReg scalarMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
    Reg::RegTensor<float> squareSumReg;
    Reg::RegTensor<float> dotSumReg;
    Reg::Reduce<Reg::ReduceType::SUM>(squareSumReg, squareReg, allMask);
    Reg::Reduce<Reg::ReduceType::SUM>(dotSumReg, dotReg, allMask);
    StoreUpdateScore(squareSumReg, dotSumReg, stats, statsTStride, tIdx, invD, eps, scalarMask);
}

__simd_callee__ inline __attribute__((always_inline)) void CombineUpdateScore(
    Reg::RegTensor<float>& squareAccReg0, Reg::RegTensor<float>& squareAccReg1, Reg::RegTensor<float>& dotAccReg0,
    Reg::RegTensor<float>& dotAccReg1, __ubuf__ float* stats, uint32_t statsTStride, uint16_t tIdx, float invD,
    float eps, Reg::MaskReg& allMask)
{
    Reg::Add<float>(squareAccReg0, squareAccReg0, squareAccReg1, allMask);
    Reg::Add<float>(dotAccReg0, dotAccReg0, dotAccReg1, allMask);
    ReduceUpdateScore(squareAccReg0, dotAccReg0, stats, statsTStride, tIdx, invD, eps, allMask);
}

// Phase 1 computes p = partial + delta and
// score = dot(p, pseudoQuery) / sqrt(mean(p * p) + eps), then stores p and one score per T row.
// One-VREG specialization: the query is invariant across T, so load and tail-clear it once.
__simd_vf__ inline void BlockAttnResUpdatePhase1OneVLVF(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* delta, __ubuf__ float* pseudoQuery, __ubuf__ float* stats,
    uint16_t tSize, uint32_t dSize, uint32_t dAlignFp32, uint32_t dAlignBf16, uint32_t statsTStride, float eps,
    float invD, uint16_t hasTail)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg scalarMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
    uint32_t remainingD = dSize;
    Reg::MaskReg dMask = Reg::UpdateMask<float>(remainingD);

    Reg::RegTensor<float> partialReg;
    Reg::RegTensor<float> deltaFp32Reg;
    Reg::RegTensor<float> queryReg;
    Reg::RegTensor<float> dotProductReg;
    Reg::RegTensor<float> squareSumReg;
    Reg::RegTensor<float> dotSumReg;
    Reg::RegTensor<bfloat16_t> deltaReg;

    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg, pseudoQuery);
    // Add below zeroes inactive partial lanes. Clear query padding once so the reductions can use allMask.
    for (uint16_t loop = 0; loop < hasTail; ++loop) {
        Reg::ShiftLefts(
            (Reg::RegTensor<uint32_t>&)queryReg, (Reg::RegTensor<uint32_t>&)queryReg, static_cast<int16_t>(0), dMask);
    }

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        const uint32_t partialTOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t deltaTOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;

        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg, partial + partialTOffset);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaReg, delta + deltaTOffset);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg, deltaReg, dMask);
        Reg::Add<float>(partialReg, partialReg, deltaFp32Reg, dMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialTOffset, partialReg, dMask);

        // Form the dot product before reusing partialReg for the square product.
        Reg::Mul<float>(dotProductReg, partialReg, queryReg, allMask);
        Reg::Mul<float>(partialReg, partialReg, partialReg, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(squareSumReg, partialReg, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(dotSumReg, dotProductReg, allMask);
        StoreUpdateScore(squareSumReg, dotSumReg, stats, statsTStride, tIdx, invD, eps, scalarMask);
    }
}

// Two-VREG Phase 1 specialization: both query vectors stay resident across the T loop, and the second is tail-cleared
// once.
__simd_vf__ inline void BlockAttnResUpdatePhase1TwoVLVF(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* delta, __ubuf__ float* pseudoQuery, __ubuf__ float* stats,
    uint16_t tSize, uint32_t dSize, uint32_t dAlignFp32, uint32_t dAlignBf16, uint32_t statsTStride, float eps,
    float invD, uint16_t hasTail)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    uint32_t tailD = dSize - BARU_VREG_FP32_ELEMENTS;
    Reg::MaskReg tailMask = Reg::UpdateMask<float>(tailD);

    Reg::RegTensor<float> partialReg0;
    Reg::RegTensor<float> partialReg1;
    Reg::RegTensor<float> deltaFp32Reg0;
    Reg::RegTensor<float> deltaFp32Reg1;
    Reg::RegTensor<float> queryReg0;
    Reg::RegTensor<float> queryReg1;
    Reg::RegTensor<float> dotProductReg0;
    Reg::RegTensor<bfloat16_t> deltaReg0;
    Reg::RegTensor<bfloat16_t> deltaReg1;

    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg0, pseudoQuery);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg1, pseudoQuery + BARU_VREG_FP32_ELEMENTS);
    for (uint16_t loop = 0; loop < hasTail; ++loop) {
        Reg::ShiftLefts(
            (Reg::RegTensor<uint32_t>&)queryReg1, (Reg::RegTensor<uint32_t>&)queryReg1, static_cast<int16_t>(0),
            tailMask);
    }

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        const uint32_t partialTOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t deltaTOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;

        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + partialTOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
            partialReg1, partial + partialTOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaReg0, delta + deltaTOffset);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
            deltaReg1, delta + deltaTOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, allMask);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg1, deltaReg1, tailMask);
        Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, allMask);
        Reg::Add<float>(partialReg1, partialReg1, deltaFp32Reg1, tailMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialTOffset, partialReg0, allMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(
            partial + partialTOffset + BARU_VREG_FP32_ELEMENTS, partialReg1, tailMask);

        // Fold the second vector directly into the first product. queryReg1 padding was cleared above, and the
        // masked Add zeroes partialReg1 padding, so the ZEROING-only MulAddDst can safely use allMask.
        Reg::Mul<float>(dotProductReg0, partialReg0, queryReg0, allMask);
        Reg::Mul<float>(partialReg0, partialReg0, partialReg0, allMask);
        Reg::MulAddDst<float>(dotProductReg0, partialReg1, queryReg1, allMask);
        Reg::MulAddDst<float>(partialReg0, partialReg1, partialReg1, allMask);
        ReduceUpdateScore(partialReg0, dotProductReg0, stats, statsTStride, tIdx, invD, eps, allMask);
    }
}

// Generic Phase 1 path. D is the effective reduction length; aligned strides are used only for UB addressing.
template <bool CLEAR_TAIL>
__simd_callee__ inline __attribute__((always_inline)) void AccumulateUpdatePair(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* delta, __ubuf__ float* pseudoQuery, uint32_t partialTOffset,
    uint32_t deltaTOffset, uint32_t vectorOffset0, Reg::MaskReg& allMask, Reg::MaskReg& secondMask,
    Reg::RegTensor<float>& squareAccReg0, Reg::RegTensor<float>& squareAccReg1, Reg::RegTensor<float>& dotAccReg0,
    Reg::RegTensor<float>& dotAccReg1)
{
    Reg::RegTensor<float> partialReg0;
    Reg::RegTensor<float> partialReg1;
    Reg::RegTensor<float> deltaFp32Reg0;
    Reg::RegTensor<float> deltaFp32Reg1;
    Reg::RegTensor<float> loopQueryReg0;
    Reg::RegTensor<float> loopQueryReg1;
    Reg::RegTensor<bfloat16_t> deltaReg0;
    Reg::RegTensor<bfloat16_t> deltaReg1;
    const uint32_t vectorOffset1 = vectorOffset0 + BARU_VREG_FP32_ELEMENTS;
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + partialTOffset + vectorOffset0);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1, partial + partialTOffset + vectorOffset1);
    Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaReg0, delta + deltaTOffset + vectorOffset0);
    Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaReg1, delta + deltaTOffset + vectorOffset1);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(loopQueryReg0, pseudoQuery + vectorOffset0);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(loopQueryReg1, pseudoQuery + vectorOffset1);

    if constexpr (CLEAR_TAIL) {
        Reg::ShiftLefts(
            (Reg::RegTensor<uint32_t>&)loopQueryReg1, (Reg::RegTensor<uint32_t>&)loopQueryReg1, static_cast<int16_t>(0),
            secondMask);
    }
    Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, allMask);
    Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg1, deltaReg1, secondMask);
    Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, allMask);
    Reg::Add<float>(partialReg1, partialReg1, deltaFp32Reg1, secondMask);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialTOffset + vectorOffset0, partialReg0, allMask);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(
        partial + partialTOffset + vectorOffset1, partialReg1, secondMask);

    Reg::MulAddDst<float>(squareAccReg0, partialReg0, partialReg0, allMask);
    Reg::MulAddDst<float>(squareAccReg1, partialReg1, partialReg1, allMask);
    Reg::MulAddDst<float>(dotAccReg0, partialReg0, loopQueryReg0, allMask);
    Reg::MulAddDst<float>(dotAccReg1, partialReg1, loopQueryReg1, allMask);
}

__simd_callee__ inline __attribute__((always_inline)) void AccumulateUpdateRemainder(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* delta, __ubuf__ float* pseudoQuery, uint32_t partialTOffset,
    uint32_t deltaTOffset, uint32_t vectorOffset, Reg::MaskReg& allMask, Reg::MaskReg& tailMask,
    Reg::RegTensor<float>& squareAccReg0, Reg::RegTensor<float>& dotAccReg0)
{
    Reg::RegTensor<float> partialReg0;
    Reg::RegTensor<float> deltaFp32Reg0;
    Reg::RegTensor<float> loopQueryReg0;
    Reg::RegTensor<bfloat16_t> deltaReg0;
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + partialTOffset + vectorOffset);
    Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaReg0, delta + deltaTOffset + vectorOffset);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(loopQueryReg0, pseudoQuery + vectorOffset);
    Reg::ShiftLefts(
        (Reg::RegTensor<uint32_t>&)loopQueryReg0, (Reg::RegTensor<uint32_t>&)loopQueryReg0, static_cast<int16_t>(0),
        tailMask);
    Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, tailMask);
    Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, tailMask);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialTOffset + vectorOffset, partialReg0, tailMask);
    // Add and ShiftLefts clear inactive lanes before the all-mask accumulation.
    Reg::MulAddDst<float>(squareAccReg0, partialReg0, partialReg0, allMask);
    Reg::MulAddDst<float>(dotAccReg0, partialReg0, loopQueryReg0, allMask);
}

__simd_vf__ inline void BlockAttnResUpdatePhase1VF(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* delta, __ubuf__ float* pseudoQuery, __ubuf__ float* stats,
    uint16_t tSize, uint32_t dSize, uint32_t dAlignFp32, uint32_t dAlignBf16, uint32_t statsTStride, float eps,
    float invD, uint16_t hasMixedPair, uint16_t hasSingleRemainder, uint16_t hasRemainder, uint32_t remainderD)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg tailMask;

    Reg::RegTensor<float> squareAccReg0;
    Reg::RegTensor<float> squareAccReg1;
    Reg::RegTensor<float> dotAccReg0;
    Reg::RegTensor<float> dotAccReg1;

    const uint16_t fullLoops = static_cast<uint16_t>(dSize >> BARU_VREG_FP32_SHIFT);
    const uint16_t pairFullLoops = static_cast<uint16_t>(fullLoops >> 1U);
    uint32_t remainderActive = remainderD;
    for (uint16_t loop = 0; loop < hasRemainder; ++loop) {
        tailMask = Reg::UpdateMask<float>(remainderActive);
    }

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        const uint32_t partialTOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t deltaTOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;

        Reg::Duplicate(squareAccReg0, 0.0F, allMask);
        Reg::Duplicate(squareAccReg1, 0.0F, allMask);
        Reg::Duplicate(dotAccReg0, 0.0F, allMask);
        Reg::Duplicate(dotAccReg1, 0.0F, allMask);

        // Process every complete 2 * VL pair first. Query vectors are loaded near use instead of being kept
        // resident across the whole T loop.
        for (uint16_t loop = 0; loop < pairFullLoops; ++loop) {
            const uint32_t vectorOffset0 = static_cast<uint32_t>(loop) * BARU_VREG_PAIR_ELEMENTS;

            AccumulateUpdatePair<false>(
                partial, delta, pseudoQuery, partialTOffset, deltaTOffset, vectorOffset0, allMask, allMask,
                squareAccReg0, squareAccReg1, dotAccReg0, dotAccReg1);
        }

        // Pair an odd complete vector with the tail to retain two independent accumulation chains.
        for (uint16_t loop = 0; loop < hasMixedPair; ++loop) {
            const uint32_t vectorOffset0 = (static_cast<uint32_t>(pairFullLoops) * BARU_VREG_PAIR_NUM +
                                            static_cast<uint32_t>(loop) * BARU_VREG_PAIR_NUM) *
                                           BARU_VREG_FP32_ELEMENTS;
            AccumulateUpdatePair<true>(
                partial, delta, pseudoQuery, partialTOffset, deltaTOffset, vectorOffset0, allMask, tailMask,
                squareAccReg0, squareAccReg1, dotAccReg0, dotAccReg1);
        }

        // Odd-full-only and tail-only are mutually exclusive and share one single-VL remainder body.
        for (uint16_t loop = 0; loop < hasSingleRemainder; ++loop) {
            const uint32_t vectorOffset =
                (static_cast<uint32_t>(pairFullLoops) * BARU_VREG_PAIR_NUM + loop) * BARU_VREG_FP32_ELEMENTS;
            AccumulateUpdateRemainder(
                partial, delta, pseudoQuery, partialTOffset, deltaTOffset, vectorOffset, allMask, tailMask,
                squareAccReg0, dotAccReg0);
        }

        CombineUpdateScore(
            squareAccReg0, squareAccReg1, dotAccReg0, dotAccReg1, stats, statsTStride, tIdx, invD, eps, allMask);
    }
}

// Phase 2 merges the updated partial block with the Prepare history and emits BF16 h.

__simd_callee__ inline __attribute__((always_inline)) void UpdateMergeWeights(
    __ubuf__ float* stats, uint32_t statsTStride, uint16_t tIdx, Reg::MaskReg& allMask, Reg::RegTensor<float>& alphaReg,
    Reg::RegTensor<float>& betaReg, Reg::RegTensor<float>& historyEllReg)
{
    Reg::RegTensor<float> historyMaxReg;
    Reg::RegTensor<float> scoreReg;
    Reg::RegTensor<float> currentMaxReg;
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(
        historyMaxReg, stats + BARU_LOGIT_MAX_PLANE_INDEX * statsTStride + tIdx);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(
        historyEllReg, stats + BARU_EXP_SUM_PLANE_INDEX * statsTStride + tIdx);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(scoreReg, stats + BARU_SCORE_PLANE_INDEX * statsTStride + tIdx);
    Reg::Max<float>(currentMaxReg, historyMaxReg, scoreReg, allMask);
    Reg::ExpSub<float>(alphaReg, historyMaxReg, currentMaxReg, allMask);
    Reg::ExpSub<float>(betaReg, scoreReg, currentMaxReg, allMask);
    Reg::MulDstAdd<float>(historyEllReg, alphaReg, betaReg, allMask);
}

// Phase 2 performs a numerically stable online-softmax merge:
// newMax = max(logitMax, score), alpha = exp(logitMax - newMax), beta = exp(score - newMax),
// h = (numerator * alpha + partial * beta) / (expSum * alpha + beta).
// One-VREG specialization: the fixed D body removes the inner loop and builds its mask once per VF call.
__simd_vf__ inline void BlockAttnResUpdatePhase2OneVLVF(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* deltaH, __ubuf__ float* numerator, __ubuf__ float* stats,
    uint16_t tSize, uint32_t dSize, uint32_t dAlignFp32, uint32_t dAlignBf16, uint32_t statsTStride)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    uint32_t activeD = dSize;
    Reg::MaskReg dMask = Reg::UpdateMask<float>(activeD);

    Reg::RegTensor<float> partialReg;
    Reg::RegTensor<float> tmpReg;
    Reg::RegTensor<float> historyEllReg;
    Reg::RegTensor<float> alphaReg;
    Reg::RegTensor<float> betaReg;
    Reg::RegTensor<float> numeratorReg;
    Reg::RegTensor<bfloat16_t> outputBf16Reg;

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        UpdateMergeWeights(stats, statsTStride, tIdx, allMask, alphaReg, betaReg, historyEllReg);

        const uint32_t fp32TOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t bf16TOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg, partial + fp32TOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(numeratorReg, numerator + fp32TOffset);
        // Factor the shared denominator into the single output VREG to replace two coefficient divisions with one.
        Reg::Mul<float>(tmpReg, partialReg, betaReg, dMask);
        Reg::MulDstAdd<float>(numeratorReg, alphaReg, tmpReg, dMask);
        Reg::Div<float>(numeratorReg, numeratorReg, historyEllReg, dMask);
        Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg, numeratorReg, dMask);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(deltaH + bf16TOffset, outputBf16Reg, dMask);
    }
}

// Two-VREG Phase 2 specialization: two independent register chains expose the fixed pair to RVEC dual issue.
__simd_vf__ inline void BlockAttnResUpdatePhase2TwoVLVF(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* deltaH, __ubuf__ float* numerator, __ubuf__ float* stats,
    uint16_t tSize, uint32_t dSize, uint32_t dAlignFp32, uint32_t dAlignBf16, uint32_t statsTStride)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    uint32_t secondActiveD = dSize - BARU_VREG_FP32_ELEMENTS;
    Reg::MaskReg secondMask = Reg::UpdateMask<float>(secondActiveD);

    Reg::RegTensor<float> partialReg0;
    Reg::RegTensor<float> partialReg1;
    Reg::RegTensor<float> tmpReg0;
    Reg::RegTensor<float> tmpReg1;
    Reg::RegTensor<float> historyEllReg;
    Reg::RegTensor<float> alphaReg;
    Reg::RegTensor<float> betaReg;
    Reg::RegTensor<float> numeratorReg0;
    Reg::RegTensor<float> numeratorReg1;
    Reg::RegTensor<bfloat16_t> outputBf16Reg0;
    Reg::RegTensor<bfloat16_t> outputBf16Reg1;

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        UpdateMergeWeights(stats, statsTStride, tIdx, allMask, alphaReg, betaReg, historyEllReg);
        Reg::Div<float>(alphaReg, alphaReg, historyEllReg, allMask);
        Reg::Div<float>(betaReg, betaReg, historyEllReg, allMask);

        const uint32_t fp32TOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t bf16TOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + fp32TOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1, partial + fp32TOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(numeratorReg0, numerator + fp32TOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
            numeratorReg1, numerator + fp32TOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::Mul<float>(tmpReg0, partialReg0, betaReg, allMask);
        Reg::Mul<float>(tmpReg1, partialReg1, betaReg, secondMask);
        Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, allMask);
        Reg::MulDstAdd<float>(numeratorReg1, alphaReg, tmpReg1, secondMask);
        Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, allMask);
        Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg1, numeratorReg1, secondMask);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(deltaH + bf16TOffset, outputBf16Reg0, allMask);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
            deltaH + bf16TOffset + BARU_VREG_FP32_ELEMENTS, outputBf16Reg1, secondMask);
    }
}

// Generic Phase 2 path computes the online-softmax merge coefficients and forms BF16 h.
// Alpha and beta are normalized once per T row so that each D loop needs only Mul + MulDstAdd.
// Phase 1 and Phase 2 run in program order on V, so delta is fully consumed before its UB region is reused for h.
__simd_callee__ inline __attribute__((always_inline)) void StoreUpdatePair(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* deltaH, __ubuf__ float* numerator, uint32_t fp32TOffset,
    uint32_t bf16TOffset, uint32_t vectorOffset0, Reg::MaskReg& allMask, Reg::MaskReg& secondMask,
    Reg::RegTensor<float>& alphaReg, Reg::RegTensor<float>& betaReg)
{
    Reg::RegTensor<float> partialReg0;
    Reg::RegTensor<float> partialReg1;
    Reg::RegTensor<float> tmpReg0;
    Reg::RegTensor<float> tmpReg1;
    Reg::RegTensor<float> numeratorReg0;
    Reg::RegTensor<float> numeratorReg1;
    Reg::RegTensor<bfloat16_t> outputBf16Reg0;
    Reg::RegTensor<bfloat16_t> outputBf16Reg1;
    const uint32_t vectorOffset1 = vectorOffset0 + BARU_VREG_FP32_ELEMENTS;
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + fp32TOffset + vectorOffset0);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1, partial + fp32TOffset + vectorOffset1);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(numeratorReg0, numerator + fp32TOffset + vectorOffset0);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(numeratorReg1, numerator + fp32TOffset + vectorOffset1);
    Reg::Mul<float>(tmpReg0, partialReg0, betaReg, allMask);
    Reg::Mul<float>(tmpReg1, partialReg1, betaReg, secondMask);
    Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, allMask);
    Reg::MulDstAdd<float>(numeratorReg1, alphaReg, tmpReg1, secondMask);
    Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, allMask);
    Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg1, numeratorReg1, secondMask);
    Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
        deltaH + bf16TOffset + vectorOffset0, outputBf16Reg0, allMask);
    Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
        deltaH + bf16TOffset + vectorOffset1, outputBf16Reg1, secondMask);
}

__simd_callee__ inline __attribute__((always_inline)) void StoreUpdateRemainder(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* deltaH, __ubuf__ float* numerator, uint32_t fp32TOffset,
    uint32_t bf16TOffset, uint32_t vectorOffset, Reg::MaskReg& mask, Reg::RegTensor<float>& alphaReg,
    Reg::RegTensor<float>& betaReg)
{
    Reg::RegTensor<float> partialReg0;
    Reg::RegTensor<float> tmpReg0;
    Reg::RegTensor<float> numeratorReg0;
    Reg::RegTensor<bfloat16_t> outputBf16Reg0;
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + fp32TOffset + vectorOffset);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(numeratorReg0, numerator + fp32TOffset + vectorOffset);
    Reg::Mul<float>(tmpReg0, partialReg0, betaReg, mask);
    Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, mask);
    Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, mask);
    Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
        deltaH + bf16TOffset + vectorOffset, outputBf16Reg0, mask);
}

__simd_vf__ inline void BlockAttnResUpdatePhase2VF(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* deltaH, __ubuf__ float* numerator, __ubuf__ float* stats,
    uint16_t tSize, uint32_t dSize, uint32_t dAlignFp32, uint32_t dAlignBf16, uint32_t statsTStride, uint16_t fullLoops,
    uint16_t hasTail, uint16_t hasMixedPair, uint16_t hasOddFullOnly, uint16_t hasTailOnly)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg tailMask;
    Reg::RegTensor<float> historyEllReg;
    Reg::RegTensor<float> alphaReg;
    Reg::RegTensor<float> betaReg;

    const uint32_t tailD = dSize & BARU_VREG_FP32_MASK;
    const uint16_t pairFullLoops = static_cast<uint16_t>(fullLoops >> 1U);
    // Full vectors reuse allMask; only the true tail needs one dynamic mask for the whole VF call.
    uint32_t tailRemaining = tailD;
    for (uint16_t loop = 0; loop < hasTail; ++loop) {
        tailMask = Reg::UpdateMask<float>(tailRemaining);
    }

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        UpdateMergeWeights(stats, statsTStride, tIdx, allMask, alphaReg, betaReg, historyEllReg);
        // The two direct divisions are independent and remove the reciprocal-and-multiply dependency chain.
        Reg::Div<float>(alphaReg, alphaReg, historyEllReg, allMask);
        Reg::Div<float>(betaReg, betaReg, historyEllReg, allMask);
        const uint32_t fp32TOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t bf16TOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;
        // Separate register sets expose two adjacent D vectors to RVEC dual issue.
        for (uint16_t loop = 0; loop < pairFullLoops; ++loop) {
            const uint32_t vectorOffset0 = static_cast<uint32_t>(loop) * BARU_VREG_PAIR_ELEMENTS;
            StoreUpdatePair(
                partial, deltaH, numerator, fp32TOffset, bf16TOffset, vectorOffset0, allMask, allMask, alphaReg,
                betaReg);
        }

        // Keep an odd full vector paired with the tail so the original two-vector ILP is preserved.
        for (uint16_t loop = 0; loop < hasMixedPair; ++loop) {
            const uint32_t vectorOffset0 = (static_cast<uint32_t>(pairFullLoops) * BARU_VREG_PAIR_NUM +
                                            static_cast<uint32_t>(loop) * BARU_VREG_PAIR_NUM) *
                                           BARU_VREG_FP32_ELEMENTS;
            StoreUpdatePair(
                partial, deltaH, numerator, fp32TOffset, bf16TOffset, vectorOffset0, allMask, tailMask, alphaReg,
                betaReg);
        }

        for (uint16_t loop = 0; loop < hasOddFullOnly; ++loop) {
            const uint32_t vectorOffset =
                (static_cast<uint32_t>(pairFullLoops) * BARU_VREG_PAIR_NUM + loop) * BARU_VREG_FP32_ELEMENTS;
            StoreUpdateRemainder(
                partial, deltaH, numerator, fp32TOffset, bf16TOffset, vectorOffset, allMask, alphaReg, betaReg);
        }

        for (uint16_t loop = 0; loop < hasTailOnly; ++loop) {
            const uint32_t vectorOffset = (static_cast<uint32_t>(fullLoops) + loop) * BARU_VREG_FP32_ELEMENTS;
            StoreUpdateRemainder(
                partial, deltaH, numerator, fp32TOffset, bf16TOffset, vectorOffset, tailMask, alphaReg, betaReg);
        }
    }
}

} // namespace BlockAttnResUpdateOps
