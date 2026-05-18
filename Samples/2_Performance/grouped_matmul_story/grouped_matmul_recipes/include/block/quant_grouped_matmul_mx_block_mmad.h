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
 * \file quant_grouped_matmul_mx_block_mmad.h
 * \brief Block-level grouped MX MMAD wrapper.
 */
#pragma once

#include "kernel_utils/common_utils.h"
#include "kernel_utils/layout_utils.h"
#include "include/tensor_api/tensor.h"

#include "../policy/dispatch_policy.h"
#include "../tile/copy_scale_l1_to_l0a.h"
#include "../tile/copy_scale_l1_to_l0b.h"
#include "../tile/pad_mx_kl1.h"
#include "../tile/tile_mmad_mx.h"

namespace Block {

namespace {
// Half of one L0C bank (float workspace) when double-buffering the accumulator tile.
static constexpr uint64_t HALF_L0C_SIZE = L0C_SIZE / GroupedMatmulRecipe::DOUBLE_BUFFER / sizeof(float);
static constexpr uint64_t SCALE_BUFFER_NUM = 2UL;
// MTE1_MTE2 event ids: input ring (0-1) then scale ping-pong (4-5); cross-pipe sync uses base 2 + buffer id.
static constexpr uint16_t INPUT_BUFFER_MTE1_MTE2_BASE = 2;
// Scale ping-pong uses ids 4 + scaleL1BufId (0 or 1).
static constexpr uint16_t SCALE_BUFFER_MTE1_MTE2_BASE = 4;

} // namespace

template <
    class DispatchPolicy_, class ATypeTuple_, class LayoutATuple_, class BTypeTuple_, class LayoutBTuple_, class CType_,
    class LayoutC_, class BiasType_>
class BlockMmad<
    DispatchPolicy_, ATypeTuple_, LayoutATuple_, BTypeTuple_, LayoutBTuple_, CType_, LayoutC_, BiasType_,
    AscendC::Std::enable_if_t<AscendC::Std::is_base_of_v<QuantMatmulMxMultiBlockMmad, DispatchPolicy_>>> {
public:
    template <typename T>
    struct TypeUnpack {
        using Data = T;
        using Scale = void;
    };

    template <typename T0, typename T1>
    struct TypeUnpack<AscendC::Std::tuple<T0, T1>> {
        using Data = T0;
        using Scale = T1;
    };

    template <typename T>
    struct LayoutUnpack {
        using Data = T;
        using Scale = void;
    };

    template <typename T0, typename T1>
    struct LayoutUnpack<AscendC::Std::tuple<T0, T1>> {
        using Data = T0;
        using Scale = T1;
    };

    using AType = typename TypeUnpack<ATypeTuple_>::Data;
    using ScaleAType = typename TypeUnpack<ATypeTuple_>::Scale;
    using BType = typename TypeUnpack<BTypeTuple_>::Data;
    using ScaleBType = typename TypeUnpack<BTypeTuple_>::Scale;
    using CType = CType_;
    using BiasType = BiasType_;
    using LayoutA = typename LayoutUnpack<LayoutATuple_>::Data;
    using LayoutScaleA = typename LayoutUnpack<LayoutATuple_>::Scale;
    using LayoutB = typename LayoutUnpack<LayoutBTuple_>::Data;
    using LayoutScaleB = typename LayoutUnpack<LayoutBTuple_>::Scale;
    using MxL0AType = typename AscendC::GetL0DataType<AType, true>::Type;
    using MxL0BType = typename AscendC::GetL0DataType<BType, true>::Type;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    static constexpr uint64_t HALF_L0_SIZE = L0A_SIZE / GroupedMatmulRecipe::DOUBLE_BUFFER / sizeof(AType);
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    constexpr static int32_t SCALE_C0 = 2;
    constexpr static int32_t L0C_C0 = 16;
    static constexpr bool transA =
        AscendC::IsSameType<LayoutA, AscendC::Te::FrameLayoutFormat<AscendC::Te::DNExtLayoutPtn>>::value;
    using LayoutBPattern = AscendC::Te::GetLayoutPattern<decltype(LayoutB{}(0L, 0L))>;
    static constexpr bool transB = AscendC::Std::is_same_v<LayoutBPattern, AscendC::Te::DNExtLayoutPtn> ||
                                   AscendC::Std::is_same_v<LayoutBPattern, AscendC::Te::ZNLayoutPtn>;
    // MXFP8: zero-pad L1 K tail to match NZ layout / ND2NZ path (see Tile::PadMxK*L1);
    // MXFP4 keeps unpadded L1 views.
    static constexpr bool needASetL1KZero = AscendC::Std::is_one_of_v<AType, fp8_e5m2_t, fp8_e4m3fn_t>;
    static constexpr bool needBSetL1KZero = AscendC::Std::is_one_of_v<AType, fp8_e5m2_t, fp8_e4m3fn_t> ||
                                            (AscendC::Std::is_one_of_v<AType, fp4x2_e2m1_t, fp4x2_e1m2_t> && !transB);
    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
    };

    // K tiling on L1: separate strips for A and B, plus the K span at which block scales are reloaded.
    struct L1Params {
        uint64_t kAL1;
        uint64_t kBL1;
        uint64_t scaleKL1;
    };

    __aicore__ inline BlockMmad()
    {
        // Set all MTE1/MTE2 handshake flags so the first K stage can enter the copy/compute loop uniformly.
#pragma unroll
        for (uint8_t i = 0; i < GroupedMatmulRecipe::MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmad()
    {
        // Wait for every in-flight transfer and cube sync before disabling MM layout transform.
#pragma unroll
        for (uint8_t i = 0; i < GroupedMatmulRecipe::MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::SetMMLayoutTransform(false);
    }

    __aicore__ inline void Init(
        const TupleShape& problemShape, const BlockShape& l0TileShape, const L1Params& l1Params, bool enableL0cPingPong)
    {
        // Force double-buffering on L1.
        constexpr uint64_t l1BufNum = GroupedMatmulRecipe::DOUBLE_BUFFER;

        m_ = AscendC::Te::Get<GroupedMatmulRecipe::MNK_M>(problemShape);
        n_ = AscendC::Te::Get<GroupedMatmulRecipe::MNK_N>(problemShape);
        k_ = AscendC::Te::Get<GroupedMatmulRecipe::MNK_K>(problemShape);
        kAL1_ = l1Params.kAL1;
        kBL1_ = l1Params.kBL1;
        scaleKL1_ = l1Params.scaleKL1;
        baseM_ = AscendC::Te::Get<GroupedMatmulRecipe::MNK_M>(l0TileShape);
        baseN_ = AscendC::Te::Get<GroupedMatmulRecipe::MNK_N>(l0TileShape);
        baseK_ = AscendC::Te::Get<GroupedMatmulRecipe::MNK_K>(l0TileShape);
        // Prefer outer-K loops on the operand with the larger L1 K-tile to hide memory latency.
        orderAL1BL1_ = l1Params.kAL1 >= l1Params.kBL1;
        enableL0cPingPong_ = enableL0cPingPong;
        constexpr bool isFp4Type = AscendC::Std::is_one_of_v<AType, fp4x2_e2m1_t, fp4x2_e1m2_t>;
        constexpr uint64_t sizeShift = isFp4Type ? 1UL : 0UL;
        bL1OneBuffer_ = baseN_ * Align(kBL1_, GroupedMatmulRecipe::MX_DIVISOR_SIZE) >> sizeShift;
        auto mxScaleKL1B16 = CeilDiv(scaleKL1_, GroupedMatmulRecipe::MX_DIVISOR_SIZE);
        auto mxScaleKL1 = mxScaleKL1B16 * GroupedMatmulRecipe::MX_MULTI_SIZE;
        aL1OneBuffer_ = baseM_ * Align(kAL1_, GroupedMatmulRecipe::MX_DIVISOR_SIZE) >> sizeShift;
        scaleAL1OneBuffer_ = baseM_ * mxScaleKL1;
        // Lay out L1: ping-pong halves, then A/B tile slots, then scale A/B following each B buffer.
        for (int32_t bufferId = 0; bufferId < static_cast<int32_t>(l1BufNum); bufferId++) {
            uint64_t l1Offset = (L1_SIZE >> 1) * (bufferId & 1);
            l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (bufferId >> 1);
            l1BufferBOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (l1BufNum >> 1) + bL1OneBuffer_ * (bufferId >> 1);
        }
        for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum >> 1);
            l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
        }
    }

    // Update global (m,n,k) for next group
    __aicore__ inline void UpdateParamsForNextProblem(const TupleShape& problemShape)
    {
        m_ = AscendC::Te::Get<GroupedMatmulRecipe::MNK_M>(problemShape);
        n_ = AscendC::Te::Get<GroupedMatmulRecipe::MNK_N>(problemShape);
        k_ = AscendC::Te::Get<GroupedMatmulRecipe::MNK_K>(problemShape);
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void operator()(
        TensorA gmA, TensorB gmB, TensorScaleA gmScaleA, TensorScaleB gmScaleB, TensorC gmC, BlockShape singleShape)
    {
        Run(gmA, gmB, gmScaleA, gmScaleB, gmC, singleShape);
    }

private:
    // Byte column/row offset within the L1 scale tile for a given K position
    __aicore__ inline uint64_t GetScaleOffset(uint64_t kOffset) const
    {
        return (kOffset / GroupedMatmulRecipe::MX_DIVISOR_SIZE) * GroupedMatmulRecipe::MX_MULTI_SIZE;
    }

    // Span of scale elements in L1 for a K extent, in MX packing units.
    __aicore__ inline uint64_t GetScaleSpan(uint64_t kSpan) const
    {
        return CeilDiv(kSpan, GroupedMatmulRecipe::MX_DIVISOR_SIZE) * GroupedMatmulRecipe::MX_MULTI_SIZE;
    }

    template <typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline void CopyScalesInL1(
        TensorScaleA gmScaleA, TensorScaleB gmScaleB, uint64_t curM, uint64_t curN, uint64_t kL1Offset,
        uint64_t scaleL1BufId)
    {
        if (kL1Offset % scaleKL1_ != 0) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_MTE1_MTE2_BASE + scaleL1BufId);

        uint64_t curScaleKL1 = scaleKL1_;
        if (kL1Offset + curScaleKL1 > k_) {
            curScaleKL1 = k_ - kL1Offset;
        }

        auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto layoutScaleAL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            curM, CeilDiv(scaleKL1_, GroupedMatmulRecipe::MX_GROUP_SIZE));
        auto tensorScaleAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleAOffset_[scaleL1BufId]),
            layoutScaleAL1);
        auto gmBlockScaleA = gmScaleA.Slice(
            AscendC::Te::MakeCoord(0, kL1Offset / GroupedMatmulRecipe::MX_GROUP_SIZE),
            AscendC::Te::MakeShape(curM, GetScaleSpan(curScaleKL1)));
        AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

        auto layoutScaleBL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            CeilDiv(scaleKL1_, GroupedMatmulRecipe::MX_GROUP_SIZE), curN);
        auto tensorScaleBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
            layoutScaleBL1);
        auto gmBlockScaleB = gmScaleB.Slice(
            AscendC::Te::MakeCoord(kL1Offset / GroupedMatmulRecipe::MX_GROUP_SIZE, 0),
            AscendC::Te::MakeShape(GetScaleSpan(curScaleKL1), curN));
        AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
    }

    template <typename TensorA>
    __aicore__ inline void CopyAInL1(
        TensorA gmA, uint64_t curM, uint64_t curGmAKL1, uint64_t aL1BufId, uint64_t kL1Offset)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        uint64_t l1K = curGmAKL1;
        if constexpr (needASetL1KZero) {
            l1K = CeilAlign(curGmAKL1, GroupedMatmulRecipe::MX_DIVISOR_SIZE);
        }
        auto layoutAL1 = MakeLayoutAL1{}(curM, l1K);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1BufferAOffset_[aL1BufId]), layoutAL1);
        auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(0, kL1Offset), AscendC::Te::MakeShape(curM, curGmAKL1));
        if constexpr (needASetL1KZero) {
            ::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
        }
        AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);
    }

    template <typename TensorB>
    __aicore__ inline void CopyBInL1(
        TensorB gmB, uint64_t curN, uint64_t curGmBKL1, uint64_t bL1BufId, uint64_t kL1Offset)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        uint64_t l1K = curGmBKL1;
        if constexpr (needBSetL1KZero) {
            l1K = CeilAlign(curGmBKL1, GroupedMatmulRecipe::MX_DIVISOR_SIZE);
        }
        auto layoutBL1 = MakeLayoutBL1{}(l1K, curN);
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[bL1BufId]), layoutBL1);
        auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(kL1Offset, 0), AscendC::Te::MakeShape(curGmBKL1, curN));
        if constexpr (needBSetL1KZero) {
            ::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
        }
        AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);
    }

    template <typename TensorL0C>
    __aicore__ inline void Iterate(
        TensorL0C tensorL0C, uint64_t curM, uint64_t curN, uint64_t curGmAKL1, uint64_t curGmBKL1, uint64_t aL1BufId,
        uint64_t bL1BufId, uint64_t scaleL1BufId, uint64_t absKStartA, uint64_t absKStartB, uint64_t kaL1Offset,
        uint64_t kbL1Offset)
    {
        // Inner K loop: slice A/B and matching scale windows from L1, stage to L0, then Mad into L0C.
        uint64_t l1Ka = CeilAlign(curGmAKL1, GroupedMatmulRecipe::MX_DIVISOR_SIZE);
        uint64_t l1Kb = CeilAlign(curGmBKL1, GroupedMatmulRecipe::MX_DIVISOR_SIZE);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1BufferAOffset_[aL1BufId]),
            MakeLayoutAL1{}(curM, l1Ka));
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[bL1BufId]),
            MakeLayoutBL1{}(l1Kb, curN));
        auto layoutScaleAL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            curM, GetScaleSpan(scaleKL1_));
        auto tensorScaleAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleAOffset_[scaleL1BufId]),
            layoutScaleAL1);
        auto layoutScaleBL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            GetScaleSpan(scaleKL1_), curN);
        auto tensorScaleBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
            layoutScaleBL1);

        uint64_t minPadKL1 =
            Min(CeilAlign(curGmAKL1, GroupedMatmulRecipe::MX_DIVISOR_SIZE),
                CeilAlign(curGmBKL1, GroupedMatmulRecipe::MX_DIVISOR_SIZE));
        uint64_t minGmKL1 = Min(curGmAKL1, curGmBKL1);
        uint64_t scaleBaseA = GetScaleOffset(absKStartA % scaleKL1_);
        uint64_t scaleBaseB = GetScaleOffset(absKStartB % scaleKL1_);
        auto tensorBlockScaleAL1 = tensorScaleAL1.Slice(
            AscendC::Te::MakeCoord(0, scaleBaseA), AscendC::Te::MakeShape(curM, GetScaleSpan(curGmAKL1)));
        auto tensorBlockScaleBL1 = tensorScaleBL1.Slice(
            AscendC::Te::MakeCoord(scaleBaseB, 0), AscendC::Te::MakeShape(GetScaleSpan(curGmBKL1), curN));

        auto CopyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        auto CopyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        auto CopyL12L0MxScaleA3510 = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleA3510{});
        auto CopyL12L0MxScaleB3510 = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleB3510{});

        for (uint64_t kL0Offset = 0; kL0Offset < minGmKL1; kL0Offset += baseK_) {
            // Tail K in L0 uses the padded minimum of A/B L1 K strips for correct MX alignment.
            uint64_t curKL0 = (kL0Offset + baseK_ > minPadKL1) ? (minPadKL1 - kL0Offset) : baseK_;
            uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);

            auto tensorAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset),
                AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>(curM, curKL0));
            auto tensorBlockAL1 = tensorAL1.Slice(
                AscendC::Te::MakeCoord(0, kaL1Offset + kL0Offset), AscendC::Te::MakeShape(curM, curKL0));
            AscendC::Te::Copy(CopyL12L0A, tensorAL0, tensorBlockAL1);

            auto tensorBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset),
                AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>(curKL0, curN));
            auto tensorBlockBL1 = tensorBL1.Slice(
                AscendC::Te::MakeCoord(kbL1Offset + kL0Offset, 0), AscendC::Te::MakeShape(curKL0, curN));
            AscendC::Te::Copy(CopyL12L0B, tensorBL0, tensorBlockBL1);

            auto tensorScaleAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, fp8_e8m0_t>(l0Offset),
                AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                    curM, CeilDiv(curKL0, GroupedMatmulRecipe::MX_DIVISOR_SIZE) * GroupedMatmulRecipe::MX_MULTI_SIZE));
            AscendC::Te::Copy(
                CopyL12L0MxScaleA3510, tensorScaleAL0, tensorBlockScaleAL1,
                AscendC::Te::MakeCoord(0, kaL1Offset + kL0Offset));

            auto tensorScaleBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, fp8_e8m0_t>(l0Offset),
                AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                    CeilDiv(curKL0, GroupedMatmulRecipe::MX_DIVISOR_SIZE) * GroupedMatmulRecipe::MX_MULTI_SIZE, curN));
            AscendC::Te::Copy(
                CopyL12L0MxScaleB3510, tensorScaleBL0, tensorBlockScaleBL1,
                AscendC::Te::MakeCoord(kbL1Offset + kL0Offset, 0));

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);

            // Last K micro-tile uses final accumulation; first micro-tile initializes C in the Mad unit.
            uint8_t mmadUnitFlag = (absKStartA + kaL1Offset + kL0Offset + baseK_ >= k_) ?
                                       GroupedMatmulRecipe::FINAL_ACCUMULATION :
                                       GroupedMatmulRecipe::NON_FINAL_ACCUMULATION;
            bool mmadCmatrixInitVal = (absKStartA + kaL1Offset + kL0Offset == 0);
            AscendC::Te::MmadParams mmadParams;
            mmadParams.m = static_cast<uint16_t>(curM);
            mmadParams.k = static_cast<uint16_t>(CeilAlign(curKL0, GroupedMatmulRecipe::MX_DIVISOR_SIZE));
            mmadParams.n = static_cast<uint16_t>(curN);
            mmadParams.unitFlag = mmadUnitFlag;
            mmadParams.cmatrixInitVal = mmadCmatrixInitVal;
            AscendC::Te::Mmad(
                AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, AscendC::Te::MmadTraitMX>>{}
                    .with(mmadParams),
                tensorL0C, tensorAL0, tensorBL0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            l0PingPong_++;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void Run(
        TensorA gmA, TensorB gmB, TensorScaleA gmScaleA, TensorScaleB gmScaleB, TensorC gmC,
        const BlockShape& singleShape)
    {
        // current tile shape
        uint64_t curM = AscendC::Te::Get<GroupedMatmulRecipe::MNK_M>(singleShape);
        uint64_t curN = AscendC::Te::Get<GroupedMatmulRecipe::MNK_N>(singleShape);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto tensorL0C = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffset),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<L0C_C0>>(curM, curN));

        if (orderAL1BL1_) {
            // A-major: kAl1 >= kBl1, A outer loop, B inner loop
            for (uint64_t kOuter = 0; kOuter < k_; kOuter += kAL1_) {
                uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
                uint64_t aL1BufId = aL1LoopCnt_ & 1;
                uint64_t curGmAKL1 = (kOuter + kAL1_ > k_) ? (k_ - kOuter) : kAL1_;
                // copy scales to L1, scaleKL1_ is multiple of kAl1_
                CopyScalesInL1(gmScaleA, gmScaleB, curM, curN, kOuter, scaleL1BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                // copy A to L1
                CopyAInL1(gmA, curM, curGmAKL1, aL1BufId, kOuter);
                // B inner loop
                for (uint64_t kInner = kOuter; kInner < Min(kOuter + kAL1_, k_); kInner += kBL1_) {
                    uint64_t bL1BufId = bL1LoopCnt_ & 1;
                    uint64_t curGmBKL1 = (kInner + kBL1_ > k_) ? (k_ - kInner) : kBL1_;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_MTE1_MTE2_BASE + bL1BufId);
                    // copy B to L1
                    CopyBInL1(gmB, curN, curGmBKL1, bL1BufId, kInner);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                    // stage A/B and scale to L0, then Mad into L0C
                    Iterate(
                        tensorL0C, curM, curN, curGmAKL1, curGmBKL1, aL1BufId, bL1BufId, scaleL1BufId, kOuter, kInner,
                        kInner - kOuter, 0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_MTE1_MTE2_BASE + bL1BufId);
                    bL1LoopCnt_++;
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                if ((kOuter + kAL1_) % scaleKL1_ == 0 || kOuter + kAL1_ >= k_) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_MTE1_MTE2_BASE + scaleL1BufId);
                    scaleLoopCnt_++;
                }
                aL1LoopCnt_++;
            }
        } else {
            // B-major: kBl1 > kAl1, B outer loop, A inner loop
            for (uint64_t kOuter = 0; kOuter < k_; kOuter += kBL1_) {
                uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
                uint64_t bL1BufId = bL1LoopCnt_ & 1;
                uint64_t curGmBKL1 = (kOuter + kBL1_ > k_) ? (k_ - kOuter) : kBL1_;
                // copy scales to L1, scaleKL1_ is multiple of kBl1_
                CopyScalesInL1(gmScaleA, gmScaleB, curM, curN, kOuter, scaleL1BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufId);
                // copy B to L1
                CopyBInL1(gmB, curN, curGmBKL1, bL1BufId, kOuter);
                // A inner loop
                for (uint64_t kInner = kOuter; kInner < Min(kOuter + kBL1_, k_); kInner += kAL1_) {
                    uint64_t aL1BufId = aL1LoopCnt_ & 1;
                    uint64_t curGmAKL1 = (kInner + kAL1_ > k_) ? (k_ - kInner) : kAL1_;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_MTE1_MTE2_BASE + aL1BufId);
                    // copy A to L1
                    CopyAInL1(gmA, curM, curGmAKL1, aL1BufId, kInner);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                    // stage A/B and scale to L0, then Mad into L0C
                    Iterate(
                        tensorL0C, curM, curN, curGmAKL1, curGmBKL1, aL1BufId, bL1BufId, scaleL1BufId, kInner, kOuter,
                        0, kInner - kOuter);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_MTE1_MTE2_BASE + aL1BufId);
                    aL1LoopCnt_++;
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufId);
                if ((kOuter + kBL1_) % scaleKL1_ == 0 || kOuter + kBL1_ >= k_) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_MTE1_MTE2_BASE + scaleL1BufId);
                    scaleLoopCnt_++;
                }
                bL1LoopCnt_++;
            }
        }

        // Single fixpipe for this block's C tile (bf16 out); L0C ping-pong only when enabled by tiling.
        auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(
            CopyL0C2GM, gmC, tensorL0C, AscendC::Te::FixpipeParams{GroupedMatmulRecipe::FINAL_ACCUMULATION});
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

    // L1 sub-allocation sizes (bytes) and base offsets for ping-pong A/B and scale buffers.
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t scaleAL1OneBuffer_ = 0UL;
    uint64_t l1BufferAOffset_[2] = {0UL};
    uint64_t l1BufferBOffset_[2] = {0UL};
    uint64_t l1BufferScaleAOffset_[2] = {0UL};
    uint64_t l1BufferScaleBOffset_[2] = {0UL};
    // Current problem and tiling (K outer tiles may differ for A vs B strips).
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t kAL1_{1};
    uint64_t kBL1_{1};
    uint64_t scaleKL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint64_t aL1LoopCnt_{0};
    uint64_t bL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool orderAL1BL1_{false};
    bool enableL0cPingPong_{false};
    // Reserve on-chip tensor slots (A2/B2/CO1/A1) in cube pipeline
    AscendC::LocalTensor<MxL0AType> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<MxL0BType> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> c1Local_{AscendC::TPosition::CO1, 0, L0C_SIZE};
    AscendC::LocalTensor<AType> aL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<BType> bL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<fp8_e8m0_t> scaleAL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<fp8_e8m0_t> scaleBL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
};
} // namespace Block
