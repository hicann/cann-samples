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

#include "vf/block_attn_res_update_vf.h"
#include "tiling/block_attn_res_update_tiling_data.h"

namespace BlockAttnResUpdateOps {

template <bool SINGLE_TILE>
class BlockAttnResUpdateFullD {
public:
    // Multi-tile mode seeds one MTE3-to-MTE2 reuse token for each ping-pong buffer. The destructor drains the final
    // tokens; single-tile mode never reuses a buffer.
    __aicore__ inline __attribute__((always_inline)) BlockAttnResUpdateFullD()
    {
        if constexpr (!SINGLE_TILE) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    __aicore__ inline __attribute__((always_inline)) ~BlockAttnResUpdateFullD()
    {
        if constexpr (!SINGLE_TILE) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void operator()(
        GM_ADDR partialBlock, GM_ADDR delta, GM_ADDR pseudoQuery, GM_ADDR numerator, GM_ADDR logitMax, GM_ADDR expSum,
        GM_ADDR h, const BlockAttnResUpdateTilingData* tilingData)
    {
        Init(partialBlock, delta, pseudoQuery, numerator, logitMax, expSum, h, tilingData);
        Process();
    }

private:
    __aicore__ inline __attribute__((always_inline)) void Init(
        GM_ADDR partialBlock, GM_ADDR delta, GM_ADDR pseudoQuery, GM_ADDR numerator, GM_ADDR logitMax, GM_ADDR expSum,
        GM_ADDR h, const BlockAttnResUpdateTilingData* tilingData)
    {
        tilingData_ = tilingData;

        partialBlockGm_ = reinterpret_cast<__gm__ float*>(partialBlock);
        deltaGm_ = reinterpret_cast<__gm__ bfloat16_t*>(delta);
        pseudoQueryGm_ = reinterpret_cast<__gm__ float*>(pseudoQuery);
        numeratorGm_ = reinterpret_cast<__gm__ float*>(numerator);
        logitMaxGm_ = reinterpret_cast<__gm__ float*>(logitMax);
        expSumGm_ = reinterpret_cast<__gm__ float*>(expSum);
        hGm_ = reinterpret_cast<__gm__ bfloat16_t*>(h);
        InitUbLayout();
    }

    __aicore__ inline __attribute__((always_inline)) void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t usedCoreNum = tilingData_->usedCoreNum;
        if (blockIdx >= usedCoreNum) {
            return;
        }

        const uint32_t tPerCore = tilingData_->tPerCore;
        const int64_t coreTStart = static_cast<int64_t>(blockIdx) * static_cast<int64_t>(tPerCore);
        const uint32_t coreTSize = (blockIdx + 1U == usedCoreNum) ? tilingData_->lastTPerCore : tPerCore;
        const uint32_t dSize = tilingData_->dSize;
        const uint32_t tileT = tilingData_->tileT;
        const uint32_t dAlignFp32 = (dSize + BARU_FP32_ALIGN_MASK) & ~BARU_FP32_ALIGN_MASK;
        const uint32_t dAlignBf16 = (dSize + BARU_BF16_ALIGN_MASK) & ~BARU_BF16_ALIGN_MASK;
        const uint32_t statsTStride = tilingData_->statsTStride;

        auto copyGmToUb = Te::MakeCopy(Te::CopyGM2UB{});
        auto queryGmLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, static_cast<int64_t>(dSize));
        auto pseudoQueryGm = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(pseudoQueryGm_), queryGmLayout);
        auto queryUbLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, static_cast<int64_t>(dAlignFp32));
        auto pseudoQueryUbMem = Te::MakeMemPtr<Te::Location::UB, float>(0);
        auto pseudoQueryUb = Te::MakeTensor(pseudoQueryUbMem, queryUbLayout);
        // Start the core-invariant Phase 1 input first, then prepare the remaining scalar descriptors while MTE2 runs.
        Te::Copy(copyGmToUb, pseudoQueryUb, pseudoQueryGm);

        // Keep runtime 0/1 loop bounds on the scalar side. Deriving them inside __simd_vf__ can trigger
        // HiIPUVectorLoopUnrollPass failures at -O2/-O3.
        const uint16_t fullDLoops = static_cast<uint16_t>(dSize >> BARU_VREG_FP32_SHIFT);
        const uint16_t hasDTail = static_cast<uint16_t>((dSize & BARU_VREG_FP32_MASK) != 0U);
        const uint16_t hasOddFullD = static_cast<uint16_t>(fullDLoops & 1U);
        const uint16_t hasMixedDPair = static_cast<uint16_t>(hasOddFullD & hasDTail);
        const uint16_t hasOddFullDOnly = static_cast<uint16_t>(hasOddFullD - hasMixedDPair);
        const uint16_t hasDTailOnly = static_cast<uint16_t>(hasDTail - hasMixedDPair);
        const uint16_t phase1HasSingleRemainder = static_cast<uint16_t>(hasOddFullDOnly + hasDTailOnly);
        const uint16_t phase1HasRemainder = static_cast<uint16_t>(hasMixedDPair + phase1HasSingleRemainder);
        const uint32_t phase1RemainderD = static_cast<uint32_t>(hasOddFullDOnly) * BARU_VREG_FP32_ELEMENTS +
                                          static_cast<uint32_t>(hasDTail) * (dSize & BARU_VREG_FP32_MASK);
        const float eps = tilingData_->eps;
        const float invD = tilingData_->invD;
        auto matrixGmLayout =
            Te::MakeFrameLayout<Te::NDExtLayoutPtn>(static_cast<int64_t>(coreTSize), static_cast<int64_t>(dSize));
        auto statsGmLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, static_cast<int64_t>(coreTSize));
        auto partialBlockGm = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(partialBlockGm_ + coreTStart * static_cast<int64_t>(dSize)),
            matrixGmLayout);
        auto deltaGm = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(deltaGm_ + coreTStart * static_cast<int64_t>(dSize)), matrixGmLayout);
        auto numeratorGm = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(numeratorGm_ + coreTStart * static_cast<int64_t>(dSize)), matrixGmLayout);
        auto logitMaxGm = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(logitMaxGm_ + coreTStart), statsGmLayout);
        auto expSumGm = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(expSumGm_ + coreTStart), statsGmLayout);
        auto hGm = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(hGm_ + coreTStart * static_cast<int64_t>(dSize)), matrixGmLayout);

        auto fp32UbLayout =
            Te::MakeFrameLayout<Te::NDExtLayoutPtn>(static_cast<int64_t>(tileT), static_cast<int64_t>(dAlignFp32));
        auto bf16UbLayout =
            Te::MakeFrameLayout<Te::NDExtLayoutPtn>(static_cast<int64_t>(tileT), static_cast<int64_t>(dAlignBf16));
        auto statsUbLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(
            static_cast<int64_t>(BARU_STATS_PLANE_NUM), static_cast<int64_t>(statsTStride));

        auto copyUbToGm = Te::MakeCopy(Te::CopyUB2GM{});

        // SINGLE_TILE executes one iteration; multi-tile mode alternates the two UB buffers until coreTSize is covered.
        for (uint32_t tileTStart = 0, bufferId = 0; SINGLE_TILE || tileTStart < coreTSize;) {
            const uint32_t remainingT = coreTSize - tileTStart;
            const uint32_t currentTSize = remainingT < tileT ? remainingT : tileT;
            const int64_t tileTStartDim = static_cast<int64_t>(tileTStart);
            const auto matrixTileShape = Te::MakeShape(static_cast<int64_t>(currentTSize), static_cast<int64_t>(dSize));

            auto partialBlockGmTile = partialBlockGm.Slice(Te::MakeCoord(tileTStartDim, 0L), matrixTileShape);
            auto deltaGmTile = deltaGm.Slice(Te::MakeCoord(tileTStartDim, 0L), matrixTileShape);

            const uint64_t partialUbOffset = queryUbBytes_ + static_cast<uint64_t>(bufferId) * bufferUbBytes_;
            const uint64_t deltaHUbOffset = partialUbOffset + partialUbBytes_;
            auto partialUbMem = Te::MakeMemPtr<Te::Location::UB, float>(partialUbOffset);
            auto deltaHUbMem = Te::MakeMemPtr<Te::Location::UB, bfloat16_t>(deltaHUbOffset);
            auto partialUbStorage = Te::MakeTensor(partialUbMem, fp32UbLayout);
            auto deltaHUbStorage = Te::MakeTensor(deltaHUbMem, bf16UbLayout);
            auto partialUb = partialUbStorage.Slice(Te::MakeCoord(0L, 0L), matrixTileShape);
            auto deltaHUb = deltaHUbStorage.Slice(Te::MakeCoord(0L, 0L), matrixTileShape);

            // Phase 1 copy-in. Prepare Phase 2 descriptors below while these MTE2 transfers are in flight.
            if constexpr (!SINGLE_TILE) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(bufferId);
            }
            Te::Copy(copyGmToUb, partialUb, partialBlockGmTile);
            Te::Copy(copyGmToUb, deltaHUb, deltaGmTile);

            const auto statsTileShape = Te::MakeShape(1L, static_cast<int64_t>(currentTSize));
            auto numeratorGmTile = numeratorGm.Slice(Te::MakeCoord(tileTStartDim, 0L), matrixTileShape);
            auto logitMaxGmTile = logitMaxGm.Slice(Te::MakeCoord(0L, tileTStartDim), statsTileShape);
            auto expSumGmTile = expSumGm.Slice(Te::MakeCoord(0L, tileTStartDim), statsTileShape);
            auto hGmTile = hGm.Slice(Te::MakeCoord(tileTStartDim, 0L), matrixTileShape);

            const uint64_t numeratorUbOffset = deltaHUbOffset + deltaHUbBytes_;
            const uint64_t statsUbOffset = numeratorUbOffset + partialUbBytes_;
            auto numeratorUbMem = Te::MakeMemPtr<Te::Location::UB, float>(numeratorUbOffset);
            auto statsUbMem = Te::MakeMemPtr<Te::Location::UB, float>(statsUbOffset);
            auto numeratorUbStorage = Te::MakeTensor(numeratorUbMem, fp32UbLayout);
            auto statsUbStorage = Te::MakeTensor(statsUbMem, statsUbLayout);
            auto numeratorUb = numeratorUbStorage.Slice(Te::MakeCoord(0L, 0L), matrixTileShape);
            auto logitMaxUb = statsUbStorage.Slice(
                Te::MakeCoord(static_cast<int64_t>(BARU_LOGIT_MAX_PLANE_INDEX), 0L), statsTileShape);
            auto expSumUb =
                statsUbStorage.Slice(Te::MakeCoord(static_cast<int64_t>(BARU_EXP_SUM_PLANE_INDEX), 0L), statsTileShape);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(bufferId);

            // Phase 1 compute.
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(bufferId);
            // TwoVLVF is specialized for D spanning at most two FP32 vector-register widths.
            if (dSize <= BARU_VREG_FP32_ELEMENTS) {
                asc_vf_call<BlockAttnResUpdatePhase1OneVLVF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), pseudoQueryUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentTSize), dSize, dAlignFp32, dAlignBf16, statsTStride, eps, invD,
                    hasDTail);
            } else if (dSize <= BARU_VREG_PAIR_ELEMENTS) {
                asc_vf_call<BlockAttnResUpdatePhase1TwoVLVF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), pseudoQueryUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentTSize), dSize, dAlignFp32, dAlignBf16, statsTStride, eps, invD,
                    hasDTail);
            } else {
                asc_vf_call<BlockAttnResUpdatePhase1VF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), pseudoQueryUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentTSize), dSize, dAlignFp32, dAlignBf16, statsTStride, eps, invD,
                    hasMixedDPair, phase1HasSingleRemainder, phase1HasRemainder, phase1RemainderD);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(bufferId);

            // Queue the partial copy-out as soon as its Phase 1 dependency is established.
            // MTE3 waits for Phase 1 while Scalar continues issuing the independent Phase 2 MTE2 copy-in.
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(bufferId);
            Te::Copy(copyUbToGm, partialBlockGmTile, partialUb);

            // Phase 2 copy-in can overlap Phase 1 compute because it writes disjoint UB regions.
            const uint32_t phase2EventId = bufferId + BARU_BUFFER_NUM;
            Te::Copy(copyGmToUb, numeratorUb, numeratorGmTile);
            Te::Copy(copyGmToUb, logitMaxUb, logitMaxGmTile);
            Te::Copy(copyGmToUb, expSumUb, expSumGmTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(phase2EventId);

            // The partial copy-out and Phase 2 only read partial, so MTE3 and V may overlap.
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(phase2EventId);
            // TwoVLVF is specialized for D spanning at most two FP32 vector-register widths.
            if (dSize <= BARU_VREG_FP32_ELEMENTS) {
                asc_vf_call<BlockAttnResUpdatePhase2OneVLVF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), numeratorUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentTSize), dSize, dAlignFp32, dAlignBf16, statsTStride);
            } else if (dSize <= BARU_VREG_PAIR_ELEMENTS) {
                asc_vf_call<BlockAttnResUpdatePhase2TwoVLVF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), numeratorUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentTSize), dSize, dAlignFp32, dAlignBf16, statsTStride);
            } else {
                asc_vf_call<BlockAttnResUpdatePhase2VF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), numeratorUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentTSize), dSize, dAlignFp32, dAlignBf16, statsTStride, fullDLoops,
                    hasDTail, hasMixedDPair, hasOddFullDOnly, hasDTailOnly);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(phase2EventId);

            // Phase 2 copy-out closes this buffer's reuse dependency chain.
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(phase2EventId);
            Te::Copy(copyUbToGm, hGmTile, deltaHUb);
            if constexpr (!SINGLE_TILE) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(bufferId);
            }
            if constexpr (SINGLE_TILE) {
                break;
            }
            tileTStart += currentTSize;
            bufferId ^= 1U;
        }
    }

    __aicore__ inline __attribute__((always_inline)) void InitUbLayout()
    {
        const uint32_t dSize = tilingData_->dSize;
        const uint32_t dAlignFp32 = (dSize + BARU_FP32_ALIGN_MASK) & ~BARU_FP32_ALIGN_MASK;
        const uint32_t dAlignBf16 = (dSize + BARU_BF16_ALIGN_MASK) & ~BARU_BF16_ALIGN_MASK;
        queryUbBytes_ = static_cast<uint64_t>(dAlignFp32) * BARU_FP32_BYTES;
        partialUbBytes_ = static_cast<uint64_t>(tilingData_->tileT) * queryUbBytes_;
        deltaHUbBytes_ = static_cast<uint64_t>(tilingData_->tileT) * dAlignBf16 * BARU_BF16_BYTES;
        const uint64_t statsBytes =
            static_cast<uint64_t>(BARU_STATS_PLANE_NUM) * tilingData_->statsTStride * BARU_FP32_BYTES;
        // UB layout: [query][buffer 0][buffer 1]. Each buffer contains partial, delta/h, numerator, and the logitMax,
        // expSum, and score stats planes in that order.
        const uint64_t numeratorUbBytes = partialUbBytes_;
        bufferUbBytes_ = partialUbBytes_ + numeratorUbBytes + deltaHUbBytes_ + statsBytes;
    }

    uint64_t queryUbBytes_ = 0;
    uint64_t partialUbBytes_ = 0;
    uint64_t deltaHUbBytes_ = 0;
    uint64_t bufferUbBytes_ = 0;

    const BlockAttnResUpdateTilingData* tilingData_ = nullptr;
    __gm__ float* partialBlockGm_ = nullptr;
    __gm__ bfloat16_t* deltaGm_ = nullptr;
    __gm__ float* pseudoQueryGm_ = nullptr;
    __gm__ float* numeratorGm_ = nullptr;
    __gm__ float* logitMaxGm_ = nullptr;
    __gm__ float* expSumGm_ = nullptr;
    __gm__ bfloat16_t* hGm_ = nullptr;
};

} // namespace BlockAttnResUpdateOps
