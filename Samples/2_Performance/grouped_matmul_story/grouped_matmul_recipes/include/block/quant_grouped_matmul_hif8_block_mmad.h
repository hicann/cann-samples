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
 * \file quant_grouped_matmul_hif8_block_mmad.h
 * \brief Block MMAD for grouped HiFloat8 quantized matmul.
 */
#pragma once

#include "kernel_utils/common_utils.h"
#include "include/tensor_api/tensor.h"

#include "../policy/dispatch_policy.h"
#include "../utils/grouped_matmul_constant.h"

namespace {
constexpr uint16_t HIF8_SCALE_BUFFER_NUM = 2;
constexpr uint16_t HIF8_AB_L1_TWO_BUFFER = 2;
constexpr uint16_t HIF8_INPUT_BUFFER_FLAG_0 = 0;
constexpr uint16_t HIF8_INPUT_BUFFER_FLAG_1 = 1;
constexpr uint16_t HIF8_INPUT_BUFFER_FLAG_2 = 2;
constexpr uint16_t HIF8_INPUT_BUFFER_FLAG_3 = 3;
constexpr uint16_t HIF8_X2_SCALE_BUFFER_FLAG_0 = 0;
constexpr uint16_t HIF8_X2_SCALE_BUFFER_FLAG_1 = 1;
} // namespace

namespace Block {

template <
    class DispatchPolicy_, class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_,
    class BiasType_>
class BlockMmad<
    DispatchPolicy_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_,
    AscendC::Std::enable_if_t<AscendC::Std::is_base_of_v<QuantGroupedMatmulBlockMmad, DispatchPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using L0CType = float;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool transA =
        AscendC::IsSameType<LayoutA, AscendC::Te::FrameLayoutFormat<AscendC::Te::DNExtLayoutPtn>>::value;
    static constexpr bool transB =
        AscendC::IsSameType<LayoutB, AscendC::Te::FrameLayoutFormat<AscendC::Te::DNExtLayoutPtn>>::value;
    static constexpr uint16_t L0C_C0 = 16;

    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
    };

    __aicore__ inline BlockMmad()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(HIF8_X2_SCALE_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(HIF8_X2_SCALE_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmad()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(HIF8_X2_SCALE_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(HIF8_X2_SCALE_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::SetMMLayoutTransform(false);
    }

    __aicore__ inline void Init(
        const TupleShape& problemShape, const BlockShape& l0TileShape, uint64_t kAL1, uint64_t kBL1,
        uint64_t l1BufferNum, GroupedMatmulRecipe::QuantMode quantMode, bool dbL0C)
    {
        m_ = AscendC::Te::Get<MNK_M>(problemShape);
        n_ = AscendC::Te::Get<MNK_N>(problemShape);
        k_ = AscendC::Te::Get<MNK_K>(problemShape);
        baseM_ = AscendC::Te::Get<MNK_M>(l0TileShape);
        baseN_ = AscendC::Te::Get<MNK_N>(l0TileShape);
        baseK_ = AscendC::Te::Get<MNK_K>(l0TileShape);
        l1BufNum_ = l1BufferNum;
        enableL0cPingPong_ = dbL0C;
        if (quantMode == GroupedMatmulRecipe::QuantMode::PERCHANNEL_MODE) {
            x2ScaleL1OneBuffer_ = baseN_ * sizeof(uint64_t);
        }
        if (l1BufferNum == HIF8_AB_L1_TWO_BUFFER) {
            kAL1_ = kAL1;
            kBL1_ = kBL1;
            aL1OneBuffer_ = baseM_ * kAL1_;
            bL1OneBuffer_ = baseN_ * kBL1_;
            kAL1Iter_ = CeilDiv(k_, kAL1_);
            kBL1Iter_ = CeilDiv(k_, kBL1_);
            if (kAL1 == kBL1) {
                kL1_ = kAL1;
                kL1Iter_ = CeilDiv(k_, kL1_);
            }
        } else {
            kL1_ = Min(kAL1, kBL1);
            aL1OneBuffer_ = baseM_ * kL1_;
            bL1OneBuffer_ = baseN_ * kL1_;
            kL1Iter_ = CeilDiv(k_, kL1_);
            kAL1_ = kL1_;
            kBL1_ = kL1_;
        }
        GetL1BufferOffset();
    }

    template <typename TensorA, typename TensorB, typename TScale, typename TensorC>
    __aicore__ inline void operator()(TensorA gmA, TensorB gmB, TScale scaleGlobal, TensorC gmC, BlockShape singleShape)
    {
        uint64_t curML1 = AscendC::Te::Get<MNK_M>(singleShape);
        uint64_t curNL1 = AscendC::Te::Get<MNK_N>(singleShape);
        AscendC::Te::MmadParams params;
        params.m = curML1;
        params.n = curNL1;
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        uint16_t scaleL1BufId = scaleLoopCnt_ & 1;

        auto layoutL0C =
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<L0C_C0>>(curML1, curNL1);
        auto c1Local = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, L0CType>(l0cOffset), layoutL0C);

        constexpr bool isScalarScale = AscendC::IsSameType<TScale, uint64_t>::value;
        auto layoutX2L1 = AscendC::Te::MakeFrameLayout<
            AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<uint64_t>>(1, curNL1);
        auto tensorX2L1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, uint64_t, uint64_t>(
                l1BufferX2ScaleOffset_[scaleL1BufId]),
            layoutX2L1);
        if constexpr (isScalarScale) {
            scalarScale_ = scaleGlobal;
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(scaleL1BufId);
            auto copyGM2L1Scale = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            AscendC::Te::Copy(copyGM2L1Scale, tensorX2L1, scaleGlobal);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(scaleL1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(scaleL1BufId);
        }

        if (kAL1_ == kBL1_) {
            IterateABL1(gmA, gmB, params, c1Local, curML1, curNL1);
        } else if (kAL1_ > kBL1_) {
            IterateAL1BL1(gmA, gmB, params, c1Local, curML1, curNL1);
        } else {
            IterateBL1AL1(gmA, gmB, params, c1Local, curML1, curNL1);
        }

        auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        if constexpr (isScalarScale) {
            AscendC::Te::Copy(
                copyL0C2GM, gmC, c1Local, scalarScale_,
                AscendC::Te::FixpipeParams{GroupedMatmulRecipe::FINAL_ACCUMULATION});
        } else {
            AscendC::Te::Copy(
                copyL0C2GM, gmC, c1Local, tensorX2L1,
                AscendC::Te::FixpipeParams{GroupedMatmulRecipe::FINAL_ACCUMULATION});
        }

        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
        if constexpr (!isScalarScale) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(scaleL1BufId);
            scaleLoopCnt_++;
        }
    }

private:
    static constexpr uint32_t L0_INIT_MODE_ABL1 = 0U;
    static constexpr uint32_t L0_INIT_MODE_SPLIT = 1U;

    template <typename TensorAL1, typename TensorBL1, typename C1Tensor>
    __aicore__ inline void Iterate(
        AscendC::Te::MmadParams& params, TensorAL1 tensorAL1, TensorBL1 tensorBL1, C1Tensor& c1Local,
        uint64_t curML1, uint64_t curNL1, uint64_t curInnerKL1, uint64_t aKPrefix, uint64_t bKPrefix,
        bool isL1LastRound, uint32_t initMode, uint64_t initQ0, uint64_t initQ1)
    {
        auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        uint64_t kL0Iter = CeilDiv(curInnerKL1, baseK_);
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            uint64_t curKL0 = (iter1 == kL0Iter - 1) ? (curInnerKL1 - iter1 * baseK_) : baseK_;
            uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);

            auto layoutAL0 = AscendC::Te::MakeFrameLayout<
                AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>(curML1, curKL0);
            auto layoutBL0 = AscendC::Te::MakeFrameLayout<
                AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>(curKL0, curNL1);
            auto l0aLocal = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset), layoutAL0);
            auto l0bLocal = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset), layoutBL0);

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            auto aL1Sub = tensorAL1.Slice(
                AscendC::Te::MakeCoord(0, aKPrefix + iter1 * baseK_), AscendC::Te::MakeShape(curML1, curKL0));
            AscendC::Te::Copy(copyL12L0A, l0aLocal, aL1Sub);
            auto bL1Sub = tensorBL1.Slice(
                AscendC::Te::MakeCoord(bKPrefix + iter1 * baseK_, 0), AscendC::Te::MakeShape(curKL0, curNL1));
            AscendC::Te::Copy(copyL12L0B, l0bLocal, bL1Sub);

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);

            params.k = curKL0;
            params.unitFlag = (isL1LastRound && iter1 + 1 == kL0Iter)
                                  ? GroupedMatmulRecipe::FINAL_ACCUMULATION
                                  : GroupedMatmulRecipe::NON_FINAL_ACCUMULATION;
            params.cmatrixInitVal =
                (initMode == L0_INIT_MODE_ABL1) ? (initQ0 == 0 && iter1 == 0)
                                                : (initQ0 == 0 && initQ1 == 0 && iter1 == 0);
            AscendC::Te::Mmad(
                AscendC::Te::MmadAtom<
                    AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, AscendC::Te::MmadTraitDefault>>{},
                c1Local, l0aLocal, l0bLocal, params);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            l0PingPong_++;
        }
    }

    template <typename TensorA, typename TensorB, typename C1Tensor>
    __aicore__ inline void IterateABL1(
        TensorA gmA, TensorB gmB, AscendC::Te::MmadParams& params, C1Tensor& c1Local, uint64_t curML1,
        uint64_t curNL1)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t curKL1 = (iter0 == kL1Iter_ - 1) ? (k_ - iter0 * kL1_) : kL1_;
            uint16_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);

            auto tensorAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1BufferAOffset_[l1BufId]),
                MakeLayoutAL1{}(curML1, curKL1));
            auto gmTileA = gmA.Slice(
                AscendC::Te::MakeCoord(0, iter0 * kAL1_), AscendC::Te::MakeShape(curML1, curKL1));
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);

            auto tensorBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[l1BufId]),
                MakeLayoutBL1{}(curKL1, curNL1));
            auto gmTileB = gmB.Slice(
                AscendC::Te::MakeCoord(iter0 * kBL1_, 0), AscendC::Te::MakeShape(curKL1, curNL1));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmTileB);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            Iterate(params, tensorAL1, tensorBL1, c1Local, curML1, curNL1, curKL1, 0UL, 0UL,
                iter0 == kL1Iter_ - 1, L0_INIT_MODE_ABL1, iter0, 0UL);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt_++;
        }
    }

    template <typename TensorA, typename TensorB, typename C1Tensor>
    __aicore__ inline void IterateAL1BL1(
        TensorA gmA, TensorB gmB, AscendC::Te::MmadParams& params, C1Tensor& c1Local, uint64_t curML1,
        uint64_t curNL1)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        for (uint64_t kAIter = 0; kAIter < kAL1Iter_; ++kAIter) {
            uint64_t curKAL1 = (kAIter == kAL1Iter_ - 1) ? (k_ - kAIter * kAL1_) : kAL1_;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_0 + aPingPongID_);
            auto tensorAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1BufferAOffset_[aPingPongID_]),
                MakeLayoutAL1{}(curML1, curKAL1));
            auto gmTileA = gmA.Slice(
                AscendC::Te::MakeCoord(0, kAIter * kAL1_), AscendC::Te::MakeShape(curML1, curKAL1));
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(HIF8_INPUT_BUFFER_FLAG_0 + aPingPongID_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(HIF8_INPUT_BUFFER_FLAG_0 + aPingPongID_);

            uint64_t kBL1Iter = CeilDiv(curKAL1, kBL1_);
            for (uint64_t kBIter = 0; kBIter < kBL1Iter; ++kBIter) {
                uint64_t curKBL1 = (kBIter == kBL1Iter - 1) ? (curKAL1 - kBIter * kBL1_) : kBL1_;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_2 + bPingPongID_);
                auto tensorBL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[bPingPongID_]),
                    MakeLayoutBL1{}(curKBL1, curNL1));
                uint64_t bKStart = kAIter * kAL1_ + kBIter * kBL1_;
                auto gmTileB = gmB.Slice(
                    AscendC::Te::MakeCoord(bKStart, 0), AscendC::Te::MakeShape(curKBL1, curNL1));
                AscendC::Te::Copy(copyGM2L1, tensorBL1, gmTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(HIF8_INPUT_BUFFER_FLAG_2 + bPingPongID_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(HIF8_INPUT_BUFFER_FLAG_2 + bPingPongID_);
                Iterate(params, tensorAL1, tensorBL1, c1Local, curML1, curNL1, curKBL1, kBIter * kBL1_, 0UL,
                    (kAIter == kAL1Iter_ - 1) && (kBIter == kBL1Iter - 1), L0_INIT_MODE_SPLIT, kAIter, kBIter);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_2 + bPingPongID_);
                bPingPongID_ ^= 1;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_0 + aPingPongID_);
            aPingPongID_ ^= 1;
        }
    }

    template <typename TensorA, typename TensorB, typename C1Tensor>
    __aicore__ inline void IterateBL1AL1(
        TensorA gmA, TensorB gmB, AscendC::Te::MmadParams& params, C1Tensor& c1Local, uint64_t curML1,
        uint64_t curNL1)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        for (uint64_t kBIter = 0; kBIter < kBL1Iter_; ++kBIter) {
            uint64_t curKBL1 = (kBIter == kBL1Iter_ - 1) ? (k_ - kBIter * kBL1_) : kBL1_;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_0 + bPingPongID_);
            auto tensorBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[bPingPongID_]),
                MakeLayoutBL1{}(curKBL1, curNL1));
            auto gmTileB = gmB.Slice(
                AscendC::Te::MakeCoord(kBIter * kBL1_, 0), AscendC::Te::MakeShape(curKBL1, curNL1));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(HIF8_INPUT_BUFFER_FLAG_0 + bPingPongID_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(HIF8_INPUT_BUFFER_FLAG_0 + bPingPongID_);

            uint64_t kAL1Iter = CeilDiv(curKBL1, kAL1_);
            for (uint64_t kAIter = 0; kAIter < kAL1Iter; ++kAIter) {
                uint64_t curKAL1 = (kAIter == kAL1Iter - 1) ? (curKBL1 - kAIter * kAL1_) : kAL1_;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_2 + aPingPongID_);
                auto tensorAL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1BufferAOffset_[aPingPongID_]),
                    MakeLayoutAL1{}(curML1, curKAL1));
                uint64_t aKStart = kBIter * kBL1_ + kAIter * kAL1_;
                auto gmTileA = gmA.Slice(
                    AscendC::Te::MakeCoord(0, aKStart), AscendC::Te::MakeShape(curML1, curKAL1));
                AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(HIF8_INPUT_BUFFER_FLAG_2 + aPingPongID_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(HIF8_INPUT_BUFFER_FLAG_2 + aPingPongID_);
                Iterate(params, tensorAL1, tensorBL1, c1Local, curML1, curNL1, curKAL1, 0UL, kAIter * kAL1_,
                    (kBIter == kBL1Iter_ - 1) && (kAIter == kAL1Iter - 1), L0_INIT_MODE_SPLIT, kBIter, kAIter);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_2 + aPingPongID_);
                aPingPongID_ ^= 1;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(HIF8_INPUT_BUFFER_FLAG_0 + bPingPongID_);
            bPingPongID_ ^= 1;
        }
    }

    __aicore__ inline void GetL1BufferOffset()
    {
        for (uint16_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            uint64_t l1Offset = (L1_SIZE >> 1) * (bufferId & 1);
            l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (bufferId >> 1);
            l1BufferBOffset_[bufferId] =
                l1Offset + aL1OneBuffer_ * (l1BufNum_ >> 1) + bL1OneBuffer_ * (bufferId >> 1);
        }
        for (uint16_t bufferId = 0; bufferId < HIF8_SCALE_BUFFER_NUM; bufferId++) {
            l1BufferX2ScaleOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
        }
    }

private:
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / GroupedMatmulRecipe::DOUBLE_BUFFER;
    constexpr static uint64_t HALF_L0C_SIZE = L0C_SIZE / GroupedMatmulRecipe::DOUBLE_BUFFER;

    uint64_t m_{0UL};
    uint64_t n_{0UL};
    uint64_t k_{0UL};
    uint64_t l1BufNum_{1UL};
    uint64_t kL1Iter_{0UL};
    uint64_t kAL1Iter_{0UL};
    uint64_t kBL1Iter_{0UL};
    uint64_t kL1_{1UL};
    uint64_t kAL1_{1UL};
    uint64_t kBL1_{1UL};
    uint64_t baseM_{16UL};
    uint64_t baseN_{16UL};
    uint64_t baseK_{16UL};
    uint16_t aPingPongID_{0U};
    uint16_t bPingPongID_{0U};
    uint64_t abL1LoopCnt_{0UL};
    uint64_t scaleLoopCnt_{0UL};
    uint64_t l0PingPong_{0UL};
    uint64_t l0cPingPong_{0UL};
    bool enableL0cPingPong_{false};
    uint64_t aL1OneBuffer_{0UL};
    uint64_t bL1OneBuffer_{0UL};
    uint64_t l1BufferAOffset_[4] = {0UL};
    uint64_t l1BufferBOffset_[4] = {0UL};
    uint64_t x2ScaleL1OneBuffer_{0UL};
    uint64_t l1BufferX2ScaleOffset_[2] = {0UL};
    uint64_t scalarScale_{0UL};
};

} // namespace Block
