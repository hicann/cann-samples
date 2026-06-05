/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_matmul_mxfp8fp4_block_mmad_swat.h
 * \brief MXA8W4 AIC MMAD block consuming AIV-converted B tiles from L1.
 */
#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "include/tensor_api/tensor.h"
#include "kernel_utils/common_utils.h"
#include "../policy/dispatch_policy.h"
#include "../tile/copy_scale_l1_to_l0a.h"
#include "../tile/copy_scale_l1_to_l0b.h"
#include "../tile/tile_mmad_mx.h"
#include "../utils/constant.h"

namespace Block {

template <
    uint64_t L1_BUF_NUM_, class ATypeTuple_, class LayoutATuple_, class BTypeTuple_, class LayoutBTuple_, class CType_,
    class LayoutC_>
class BlockMmad<
    WeightQuantMatmulMxfp8Fp4DispatchPolicy<L1_BUF_NUM_>, ATypeTuple_, LayoutATuple_, BTypeTuple_, LayoutBTuple_,
    CType_, LayoutC_> {
public:
    using AType = typename AscendC::Std::tuple_element<0, ATypeTuple_>::type;
    using ScaleBType = typename AscendC::Std::tuple_element<1, BTypeTuple_>::type;
    using ScaleAType = typename AscendC::Std::tuple_element<1, ATypeTuple_>::type;
    using BType = typename AscendC::Std::tuple_element<0, BTypeTuple_>::type;
    using CType = CType_;
    using ConvertedBType = AType;

    using LayoutA = typename AscendC::Std::tuple_element<0, LayoutATuple_>::type;
    using LayoutScaleA = typename AscendC::Std::tuple_element<1, LayoutATuple_>::type;
    using LayoutB = typename AscendC::Std::tuple_element<0, LayoutBTuple_>::type;
    using LayoutScaleB = typename AscendC::Std::tuple_element<1, LayoutBTuple_>::type;
    using LayoutC = LayoutC_;
    using DispatchPolicy = WeightQuantMatmulMxfp8Fp4DispatchPolicy<L1_BUF_NUM_>;
    using L1TileShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using L0TileShape = AscendC::Shape<int64_t, int64_t, int64_t>;

    static constexpr int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    static constexpr uint64_t L1_BUFFER_NUM = DispatchPolicy::l1BufNum;
    static constexpr uint64_t L1_BUFFER_MASK = L1_BUFFER_NUM - 1U;
    static constexpr uint64_t SCALE_BUFFER_NUM = DOUBLE_BUFFER_COUNT;
    static constexpr bool USE_SPLIT_AIV = L1_BUFFER_NUM >= DOUBLE_BUFFER_COUNT;
    static constexpr bool USE_COMPACT_L1_LAYOUT = L1_BUFFER_NUM != L1_FOUR_BUFFER;
    static_assert(
        L1_BUFFER_NUM == DB_SIZE || L1_BUFFER_NUM == L1_FOUR_BUFFER,
        "MXFP8FP4 targets support only 2 or 4 L1 buffers.");

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        L1TileShape l1TileShape;
        L0TileShape l0TileShape;
        uint64_t kBubSize{0};
        uint64_t nBubSize{0};
    };

    __aicore__ inline explicit BlockMmad(const Params& params)
    {
        l1BaseM_ = static_cast<uint64_t>(AscendC::Std::get<0>(params.l1TileShape));
        l1BaseN_ = static_cast<uint64_t>(AscendC::Std::get<1>(params.l1TileShape));
        kL1Size_ = static_cast<uint64_t>(AscendC::Std::get<2>(params.l1TileShape));
        scaleKL1Size_ = static_cast<uint64_t>(AscendC::Std::get<3>(params.l1TileShape));
        kL0Size_ = static_cast<uint64_t>(AscendC::Std::get<2>(params.l0TileShape));
        InitEvents();
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmad()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_A_EVENT_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_A_EVENT_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_B_EVENT_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_B_EVENT_1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_EVENT_0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_EVENT_1);
        for (uint16_t index = 0; index < L1_BUFFER_NUM; ++index) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<AscendC::TEventID>(index));
        }
        AscendC::SetMMLayoutTransform(false);
    }

    template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void operator()(
        const TensorA& tensorA, const TensorScaleA& tensorScaleA, const TensorScaleB& tensorScaleB,
        const TensorC& tensorC)
    {
        InitBlockShape(tensorA, tensorScaleB, tensorC);
        ProcessBlock(tensorA, tensorScaleA, tensorScaleB, tensorC);
    }

private:
    template <typename TensorA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void InitBlockShape(const TensorA& tensorA, const TensorScaleB&, const TensorC& tensorC)
    {
        const auto& layoutA = tensorA.Layout();
        const auto& layoutC = tensorC.Layout();
        kSize_ = static_cast<uint64_t>(AscendC::Te::GetTotalColumnShape(layoutA));
        mL1Len_ = static_cast<uint64_t>(AscendC::Te::GetTotalRowShape(layoutC));
        nL1Len_ = static_cast<uint64_t>(AscendC::Te::GetTotalColumnShape(layoutC));
        // Tensor extents are real per-block M/N; buffer strides use baseM/baseN to stay fixed across tails.
        uint64_t kL1PadSize = AscendC::CeilAlign(kL1Size_, MXFP_DIVISOR_SIZE);
        uint64_t scaleKL1PadSize = AscendC::CeilAlign(scaleKL1Size_, MXFP_DIVISOR_SIZE);
        bL1Size_ = l1BaseN_ * kL1PadSize;
        aL1Size_ = l1BaseM_ * kL1PadSize;
        scaleAL1Size_ = l1BaseM_ * scaleKL1PadSize / MX_GROUP_SIZE;
        scaleBL1Size_ = l1BaseN_ * scaleKL1PadSize / MX_GROUP_SIZE;
        kTileCount_ = CeilDiv(kSize_, kL1Size_);
        InitL1BufferOffsets();
    }

    template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void ProcessBlock(
        const TensorA& tensorA, const TensorScaleA& tensorScaleA, const TensorScaleB& tensorScaleB,
        const TensorC& tensorC)
    {
        auto layoutL0C =
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<CUBE_BLOCK>>(mL1Len_, nL1Len_);
        auto tensorL0C =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(0), layoutL0C);

        uint64_t scaleWindowIter = 0;
        uint64_t scaleWindowCols = 0;
        for (uint64_t kLoopIdx = 0; kLoopIdx < kTileCount_; ++kLoopIdx) {
            uint64_t l1BufIdx = l1BufIdx_;
            uint64_t scaleABufIdx = scaleABufIdx_;
            uint64_t scaleBBufIdx = scaleBBufIdx_;
            uint64_t kL1Offset = kLoopIdx * kL1Size_;
            uint64_t kL1Len = Min(kSize_ - kL1Offset, kL1Size_);
            uint64_t scaleKOffset = kL1Offset / MX_GROUP_SIZE;
            uint64_t aL1Offset = l1BufferAOffset_[l1BufIdx];
            uint64_t bL1Offset = l1BufferBOffset_[l1BufIdx];
            uint64_t scaleAL1Offset = l1BufferScaleAOffset_[scaleABufIdx];
            uint64_t scaleBL1Offset = l1BufferScaleBOffset_[scaleBBufIdx];
            auto layoutAL1 =
                AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>(mL1Len_, kL1Len);
            auto tensorAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(aL1Offset), layoutAL1);
            auto layoutBL1 =
                AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>(kL1Len, nL1Len_);
            auto tensorBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ConvertedBType>(bL1Offset), layoutBL1);
            if (scaleWindowIter == 0) {
                // One scale window can cover multiple K-L1 tiles; reload only at the window boundary.
                scaleWindowCols = CalcScaleWindowCols(kL1Offset);
            }
            auto layoutScaleAL1 =
                AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(
                    mL1Len_, scaleWindowCols);
            auto tensorScaleAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ScaleAType>(scaleAL1Offset), layoutScaleAL1);
            auto layoutScaleBL1 =
                AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(
                    scaleWindowCols, nL1Len_);
            auto tensorScaleBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ScaleBType>(scaleBL1Offset), layoutScaleBL1);
            if (scaleWindowIter == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ScaleAEvent(scaleABufIdx));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ScaleBEvent(scaleBBufIdx));
                CopyMxScaleGmToL1(
                    tensorScaleA, tensorScaleB, tensorScaleAL1, tensorScaleBL1, scaleKOffset, scaleWindowCols);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<AscendC::TEventID>(l1BufIdx));
            CopyAGmToL1(tensorA, tensorAL1, kL1Offset, kL1Len);
            WaitConvertedWeightReady();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            IterateMatmul(
                kLoopIdx, tensorL0C, tensorAL1, tensorBL1, tensorScaleAL1, tensorScaleBL1, kL1Len,
                scaleWindowIter * kL1Size_);
            bool releaseScaleBuffer =
                ((scaleWindowIter + 1U) * kL1Size_ >= scaleKL1Size_) || (kLoopIdx + 1U == kTileCount_);
            PostProcess(releaseScaleBuffer, l1BufIdx, scaleABufIdx, scaleBBufIdx);
            scaleWindowIter = releaseScaleBuffer ? 0U : scaleWindowIter + 1U;
        }
        CopyCL0c2Gm(tensorC, tensorL0C);
    }

    __aicore__ inline void InitEvents()
    {
        for (uint16_t index = 0; index < L1_BUFFER_NUM; ++index) {
            // Seed one release token per L1 weight slot for the AIV converter.
            ReleaseWeightBufferToVector();
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<AscendC::TEventID>(index));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_A_EVENT_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_A_EVENT_1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_B_EVENT_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_B_EVENT_1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_EVENT_0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_EVENT_1);
    }

    __aicore__ inline void InitL1BufferOffsets()
    {
        // These offsets mirror the prologue's B-buffer layout; both sides must agree on fixed base strides.
        for (uint16_t bufIdx = 0; bufIdx < L1_BUFFER_NUM; ++bufIdx) {
            if constexpr (USE_COMPACT_L1_LAYOUT) {
                l1BufferBOffset_[bufIdx] = bufIdx * bL1Size_;
                l1BufferAOffset_[bufIdx] = L1_BUFFER_NUM * bL1Size_ + bufIdx * aL1Size_;
            } else {
                l1BufferBOffset_[bufIdx] = (bufIdx & 1U) * L1_HALF_SIZE + (bufIdx >> 1U) * bL1Size_;
                l1BufferAOffset_[bufIdx] =
                    (bufIdx & 1U) * L1_HALF_SIZE + DOUBLE_BUFFER_COUNT * bL1Size_ + (bufIdx >> 1U) * aL1Size_;
            }
        }
        for (uint16_t bufIdx = 0; bufIdx < SCALE_BUFFER_NUM; ++bufIdx) {
            if constexpr (USE_COMPACT_L1_LAYOUT) {
                l1BufferScaleBOffset_[bufIdx] = L1_BUFFER_NUM * (bL1Size_ + aL1Size_) + bufIdx * scaleBL1Size_;
                l1BufferScaleAOffset_[bufIdx] =
                    L1_BUFFER_NUM * (bL1Size_ + aL1Size_) + SCALE_BUFFER_NUM * scaleBL1Size_ + bufIdx * scaleAL1Size_;
            } else {
                l1BufferScaleBOffset_[bufIdx] =
                    (bufIdx & 1U) * L1_HALF_SIZE + DOUBLE_BUFFER_COUNT * bL1Size_ + DOUBLE_BUFFER_COUNT * aL1Size_;
                l1BufferScaleAOffset_[bufIdx] = l1BufferScaleBOffset_[bufIdx] + scaleBL1Size_;
            }
        }
    }

    __aicore__ inline void WaitConvertedWeightReady()
    {
        WaitWeightFlag<AIV_SYNC_AIC_FLAG>();
    }

    __aicore__ inline void ReleaseWeightBufferToVector()
    {
        SetWeightFlag<AIC_SYNC_AIV_FLAG>();
    }

    template <uint64_t FLAG>
    __aicore__ inline void WaitWeightFlag() const
    {
        if constexpr (USE_SPLIT_AIV) {
            // The second AIV subblock uses the paired flag range to avoid aliasing subblock 0.
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE1>(FLAG + FLAG_ID_MAX);
        }
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE1>(FLAG);
    }

    template <uint64_t FLAG>
    __aicore__ inline void SetWeightFlag() const
    {
        if constexpr (USE_SPLIT_AIV) {
            // Release both AIV subblocks after the AIC has consumed the converted weight slot.
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE1>(FLAG + FLAG_ID_MAX);
        }
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE1>(FLAG);
    }

    __aicore__ inline uint64_t CalcScaleWindowCols(uint64_t kL1Offset) const
    {
        uint64_t scaleWindowKLen = Min(kSize_ - kL1Offset, scaleKL1Size_);
        return AscendC::CeilAlign(scaleWindowKLen, MXFP_DIVISOR_SIZE) / MX_GROUP_SIZE;
    }

    template <typename TensorA, typename TensorAL1>
    __aicore__ inline void CopyAGmToL1(
        const TensorA& tensorA, const TensorAL1& tensorAL1, uint64_t kL1Offset, uint64_t kL1Len)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto gmBlockA = tensorA.Slice(
            AscendC::Te::MakeCoord(0, static_cast<int64_t>(kL1Offset)),
            AscendC::Te::MakeShape(static_cast<int64_t>(mL1Len_), static_cast<int64_t>(kL1Len)));
        AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);
    }

    template <typename TensorScaleA, typename TensorScaleB, typename TensorScaleAL1, typename TensorScaleBL1>
    __aicore__ inline void CopyMxScaleGmToL1(
        const TensorScaleA& tensorScaleA, const TensorScaleB& tensorScaleB, const TensorScaleAL1& tensorScaleAL1,
        const TensorScaleBL1& tensorScaleBL1, uint64_t scaleKOffset, uint64_t scaleWindowCols)
    {
        auto copyScaleGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto gmBlockScaleA = tensorScaleA.Slice(
            AscendC::Te::MakeCoord(0, static_cast<int64_t>(scaleKOffset)),
            AscendC::Te::MakeShape(static_cast<int64_t>(mL1Len_), static_cast<int64_t>(scaleWindowCols)));
        AscendC::Te::Copy(copyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

        auto gmBlockScaleB = tensorScaleB.Slice(
            AscendC::Te::MakeCoord(static_cast<int64_t>(scaleKOffset), 0),
            AscendC::Te::MakeShape(static_cast<int64_t>(scaleWindowCols), static_cast<int64_t>(nL1Len_)));
        AscendC::Te::Copy(copyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
    }

    template <
        typename TensorL0C, typename TensorAL1, typename TensorBL1, typename TensorScaleAL1, typename TensorScaleBL1>
    __aicore__ inline void IterateMatmul(
        uint64_t kLoopIdx, const TensorL0C& tensorL0C, const TensorAL1& tensorAL1, const TensorBL1& tensorBL1,
        const TensorScaleAL1& tensorScaleAL1, const TensorScaleBL1& tensorScaleBL1, uint64_t kL1Len,
        uint64_t kOffsetInScaleWindow)
    {
        uint64_t kL0Iter = CeilDiv(kL1Len, kL0Size_);
        for (uint64_t kL0Idx = 0; kL0Idx < kL0Iter; ++kL0Idx) {
            uint64_t kL0Offset = kL0Idx * kL0Size_;
            uint64_t realL0K = Min(kL1Len - kL0Offset, kL0Size_);
            uint64_t l0BufIdx = l0BufIdx_ & 1U;
            uint64_t l0Offset = l0BufIdx * HALF_L0_SIZE;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<AscendC::TEventID>(l0BufIdx));

            auto layoutAL0 =
                AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>(mL1Len_, realL0K);
            auto tensorAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset), layoutAL0);
            auto layoutBL0 =
                AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>(realL0K, nL1Len_);
            auto tensorBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, ConvertedBType>(l0Offset), layoutBL0);
            CopyAL1ToL0(tensorAL0, tensorAL1, kL0Offset, realL0K);
            CopyBL1ToL0(tensorBL0, tensorBL1, kL0Offset, realL0K);
            CopyScaleL1ToL0(tensorScaleAL1, tensorScaleBL1, kOffsetInScaleWindow + kL0Offset, realL0K, l0Offset);

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<AscendC::TEventID>(l0BufIdx));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<AscendC::TEventID>(l0BufIdx));

            AscendC::Te::MmadParams mmadParams;
            mmadParams.m = static_cast<uint16_t>(mL1Len_);
            mmadParams.k = static_cast<uint16_t>(AscendC::CeilAlign(realL0K, MXFP_DIVISOR_SIZE));
            mmadParams.n = static_cast<uint16_t>(nL1Len_);
            mmadParams.unitFlag =
                (kLoopIdx + 1U == kTileCount_ && kL0Idx + 1U == kL0Iter) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
            mmadParams.cmatrixInitVal = (kLoopIdx == 0U && kL0Idx == 0U);
            AscendC::Te::Mmad(
                AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, AscendC::Te::MmadTraitMX>>{}
                    .with(mmadParams),
                tensorL0C, tensorAL0, tensorBL0);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<AscendC::TEventID>(l0BufIdx));
            l0BufIdx_ ^= 1U;
        }
    }

    template <typename TensorAL0, typename TensorAL1>
    __aicore__ inline void CopyAL1ToL0(
        const TensorAL0& tensorAL0, const TensorAL1& tensorAL1, uint64_t kL0Offset, uint64_t realL0K)
    {
        auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        auto tensorBlockAL1 = tensorAL1.Slice(
            AscendC::Te::MakeCoord(0, static_cast<int64_t>(kL0Offset)),
            AscendC::Te::MakeShape(static_cast<int64_t>(mL1Len_), static_cast<int64_t>(realL0K)));
        AscendC::Te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1);
    }

    template <typename TensorBL0, typename TensorBL1>
    __aicore__ inline void CopyBL1ToL0(
        const TensorBL0& tensorBL0, const TensorBL1& tensorBL1, uint64_t kL0Offset, uint64_t realL0K)
    {
        auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        auto tensorBlockBL1 = tensorBL1.Slice(
            AscendC::Te::MakeCoord(static_cast<int64_t>(kL0Offset), 0),
            AscendC::Te::MakeShape(static_cast<int64_t>(realL0K), static_cast<int64_t>(nL1Len_)));
        AscendC::Te::Copy(copyL12L0B, tensorBL0, tensorBlockBL1);
    }

    template <typename TensorScaleAL1, typename TensorScaleBL1>
    __aicore__ inline void CopyScaleL1ToL0(
        const TensorScaleAL1& tensorScaleAL1, const TensorScaleBL1& tensorScaleBL1, uint64_t kOffsetInScaleWindow,
        uint64_t realL0K, uint64_t l0Offset)
    {
        // MX scale is stored per 32 K values but copied by 64-aligned MMAD K chunks.
        uint64_t realL0ScaleCols = CeilDiv(realL0K, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

        auto layoutScaleAL0 =
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(
                mL1Len_, realL0ScaleCols);
        auto tensorScaleAL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, ScaleAType>(l0Offset), layoutScaleAL0);
        auto copyScaleA = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleA3510{});
        AscendC::Te::Copy(
            copyScaleA, tensorScaleAL0, tensorScaleAL1,
            AscendC::Te::MakeCoord(0, static_cast<int64_t>(kOffsetInScaleWindow)));

        auto layoutScaleBL0 =
            AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(
                realL0ScaleCols, nL1Len_);
        auto tensorScaleBL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, ScaleBType>(l0Offset), layoutScaleBL0);
        auto copyScaleB = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleB3510{});
        AscendC::Te::Copy(
            copyScaleB, tensorScaleBL0, tensorScaleBL1,
            AscendC::Te::MakeCoord(static_cast<int64_t>(kOffsetInScaleWindow), 0));
    }

    template <typename TensorC, typename TensorL0C>
    __aicore__ inline void CopyCL0c2Gm(const TensorC& tensorC, const TensorL0C& tensorL0C)
    {
        constexpr uint64_t FP32_64_AS_UINT64 = 0x42800000;
        auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(
            copyL0C2GM, tensorC, tensorL0C, FP32_64_AS_UINT64, AscendC::Te::FixpipeParams{FINAL_ACCUMULATION});
    }

    __aicore__ inline void PostProcess(
        bool releaseScaleBuffer, uint64_t l1BufIdx, uint64_t scaleABufIdx, uint64_t scaleBBufIdx)
    {
        // Weight L1 slots are released every K-L1 tile; scale slots are released only after their window ends.
        ReleaseWeightBufferToVector();
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<AscendC::TEventID>(l1BufIdx));
        l1BufIdx_ = (l1BufIdx + 1U) & L1_BUFFER_MASK;
        if (releaseScaleBuffer) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ScaleAEvent(scaleABufIdx));
            scaleABufIdx_ = scaleABufIdx ^ 1U;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ScaleBEvent(scaleBBufIdx));
            scaleBBufIdx_ = scaleBBufIdx ^ 1U;
        }
    }

    __aicore__ inline AscendC::TEventID ScaleAEvent(uint64_t bufIdx) const
    {
        return static_cast<AscendC::TEventID>(SCALE_A_EVENT_BASE + bufIdx);
    }

    __aicore__ inline AscendC::TEventID ScaleBEvent(uint64_t bufIdx) const
    {
        return static_cast<AscendC::TEventID>(SCALE_B_EVENT_BASE + bufIdx);
    }

private:
    uint64_t kSize_{0};
    uint64_t aL1Size_{0};
    uint64_t bL1Size_{0};
    uint64_t scaleAL1Size_{0};
    uint64_t scaleBL1Size_{0};
    uint64_t l1BaseM_{0};
    uint64_t l1BaseN_{0};
    uint64_t kL1Size_{0};
    uint64_t scaleKL1Size_{0};
    uint64_t kL0Size_{0};
    uint64_t mL1Len_{0};
    uint64_t nL1Len_{0};
    uint64_t kTileCount_{0};
    uint64_t l1BufIdx_{0};
    uint64_t scaleABufIdx_{0};
    uint64_t scaleBBufIdx_{0};
    uint64_t l0BufIdx_{0};
    uint64_t l1BufferAOffset_[L1_BUFFER_NUM] = {0UL};
    uint64_t l1BufferBOffset_[L1_BUFFER_NUM] = {0UL};
    uint64_t l1BufferScaleAOffset_[SCALE_BUFFER_NUM] = {0UL};
    uint64_t l1BufferScaleBOffset_[SCALE_BUFFER_NUM] = {0UL};
    static constexpr uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    static constexpr uint64_t SCALE_A_EVENT_BASE = 4;
    static constexpr uint64_t SCALE_B_EVENT_BASE = 6;
    static constexpr AscendC::TEventID SCALE_A_EVENT_0 = 4;
    static constexpr AscendC::TEventID SCALE_A_EVENT_1 = 5;
    static constexpr AscendC::TEventID SCALE_B_EVENT_0 = 6;
    static constexpr AscendC::TEventID SCALE_B_EVENT_1 = 7;
    static constexpr AscendC::TEventID L0_EVENT_0 = 0;
    static constexpr AscendC::TEventID L0_EVENT_1 = 1;
};

} // namespace Block
