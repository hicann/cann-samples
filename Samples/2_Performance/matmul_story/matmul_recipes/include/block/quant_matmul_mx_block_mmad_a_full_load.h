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
 * \file quant_matmul_mx_block_mmad_a_full_load.h
 * \brief Block-level MX MMAD pipeline for the SWAT A-full-load path.
 */

#ifndef QUANT_MATMUL_MX_BLOCK_MMAD_A_FULL_LOAD_H
#define QUANT_MATMUL_MX_BLOCK_MMAD_A_FULL_LOAD_H

#include "kernel_utils/common_utils.h"
#include "kernel_utils/layout_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "include/tensor.h"
#include "../policy/dispatch_policy.h"
#include "../utils/quant_matmul_constant.h"
#include "../tile/tile_mmad_mx.h"
#include "../tile/copy_scale_l1_to_l0a.h"
#include "../tile/copy_scale_l1_to_l0b.h"
#include "../tile/copy_scale_gm_to_l1.h"
#include "../tile/pad_mx_kl1.h"

namespace Block {
using namespace AscendC;

template <
    class DispatchPolicy_, 
    class AType_, 
    class LayoutA_, 
    class BType_, 
    class LayoutB_, 
    class CType_, 
    class LayoutC_,
    class Enable = void>
class BlockMmad {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy_>, "Should not be here!");
};

template <
    class DispatchPolicy_, 
    class AType_, 
    class LayoutA_, 
    class BType_, 
    class LayoutB_, 
    class CType_, 
    class LayoutC_>
class BlockMmad<
    DispatchPolicy_, 
    AType_, 
    LayoutA_, 
    BType_, 
    LayoutB_, 
    CType_, 
    LayoutC_,
    AscendC::Std::enable_if_t<DispatchPolicy_::fullLoadMode == SWAT_A_FULL_LOAD_MODE>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using MxL0AType = typename GetL0DataType<AType, true>::Type;
    using MxL0BType = typename GetL0DataType<BType, true>::Type;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t l1BufNum_;
    uint64_t kL1Iter_;
    uint64_t kL1_;
    uint64_t scaleKL1_;
    uint64_t baseM_;
    uint64_t baseN_;
    uint64_t baseK_;
    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t HALF_L0C_SIZE = L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(float);
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    constexpr static uint64_t BLOCK_CUBE = 16UL;
    constexpr static uint64_t MXFP_GROUP_SIZE = 32UL;
    constexpr static uint64_t MXFP_DIVISOR_SIZE = 64UL;
    constexpr static uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
    constexpr static uint64_t SCALE_BUFFER_NUM = 2;
    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool enableL0cPingPong_{false};

    using MakeLayoutAL1 = AscendC::Te::NzLayoutFormat<AType>;
    using MakeLayoutBL1 = AscendC::Te::ZnLayoutFormat<BType>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t l1BufNum;
    };

    __aicore__ inline BlockMmad()
    {
        // Prime all producer/consumer events so the first iteration can enter
        // the pipelined copy-and-compute loop without special-case branches.
        #pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmad()
    {
        // Drain every in-flight transfer before leaving so later blocks do not
        // observe stale event state from the previous pipeline instance.
        #pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        AscendC::SetMMLayoutTransform(false);
    }

public:
    __aicore__ inline void Init(
        const TupleShape& problemShape, const BlockShape& l0TileShape, const L1Params& l1Params, bool dbL0c)
    {
        // In the A-full-load path, the A tile and its scale stay resident in
        // L1. The offset plan therefore reserves a dedicated resident region
        // and packs the rolling B/scaleB buffers around it.
        m_ = Get<IDX_M_IDX>(problemShape);
        n_ = Get<IDX_N_IDX>(problemShape);
        k_ = Get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        scaleKL1_ = l1Params.scaleKL1;
        baseM_ = Get<IDX_M_IDX>(l0TileShape);
        baseN_ = Get<IDX_N_IDX>(l0TileShape);
        baseK_ = Get<IDX_K_IDX>(l0TileShape);
        l1BufNum_ = l1Params.l1BufNum;
        enableL0cPingPong_ = dbL0c;

        constexpr bool isFp4Type = AscendC::IsSameType<AType, fp4x2_e2m1_t>::value;
        constexpr uint64_t sizeShift = isFp4Type ? 1UL : 0UL;
        bL1OneBuffer_ = (baseN_ * kL1_) >> sizeShift;
        scaleBL1OneBuffer_ = baseN_ * CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        uint64_t mAlign = Align(baseM_, BLOCK_CUBE);
        uint64_t kAlign = Align(k_, MXFP_DIVISOR_SIZE);
        aL1OneBuffer_ = (mAlign * kAlign) >> sizeShift;
        scaleAL1OneBuffer_ = baseM_ * CeilDiv(k_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        // 2 buffer: L1 space is : B0|BScale0|A|AScale|...|B1|BScale1|
        // 4 buffer: L1 space is : B0B2|BScale0|A|AScale|...|B1B3|BScale1|...
        l1BufferAOffset_[0] = bL1OneBuffer_ * (l1BufNum_ >> 1) + scaleBL1OneBuffer_;
        l1BufferScaleAOffset_[0] = l1BufferAOffset_[0] + aL1OneBuffer_;
        uint64_t b1Offset = l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_ >= (L1_SIZE >> 1)
            ? l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_
            : (L1_SIZE >> 1);
        for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            l1BufferBOffset_[bufferId] = b1Offset * (bufferId & 1) + bL1OneBuffer_ * (bufferId >> 1);
        }
        for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            l1BufferScaleBOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void operator()(
        TensorA gmA, TensorB gmB, TensorScaleA gmScaleA, TensorScaleB gmScaleB, TensorC gmC, BlockShape singleShape)
    {
        // This path keeps A-side data resident across the N tiles handled by
        // the same block. The first tile populates the resident A buffers and
        // loads scaleA once; later tiles mainly stream the B side through K.
        auto curM = Get<IDX_M_TILEIDX>(singleShape);
        auto curN = Get<IDX_N_TILEIDX>(singleShape);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C = AscendC::Te::MakeL0CLayout(curM, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeL0CmemPtr<float>(l0cOffset), layoutL0C);

        auto layoutAL1 = MakeLayoutAL1{}(curM, k_);
        auto tensorAL1 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<AType>(l1BufferAOffset_[0]), layoutAL1);
        auto layoutScaleAL1 = AscendC::Te::MakeZzLayout<fp8_e8m0_t>(
            curM, CeilDiv(k_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
        auto tensorScaleAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(l1BufferScaleAOffset_[0]), layoutScaleAL1);

        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
            uint64_t kL1Offset = iter0 * kL1_;
            uint64_t curGmBKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1Offset) : kL1_;
            uint64_t curPadKL1 = CeilAlign(curGmBKL1, MXFP_DIVISOR_SIZE);
            auto curGmAKL1 = curGmBKL1;

            if (iter0 % (scaleKL1_ / kL1_) == 0) {
                // scaleB follows the reuse window of `scaleKL1_`, so it is
                // refreshed only when the current K chunk enters a new window.
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
                uint64_t curScaleKL1 = scaleKL1_;
                if (kL1Offset + curScaleKL1 > k_) {
                    curScaleKL1 = k_ - kL1Offset;
                }
                auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(::Tile::CopyScaleGM2L1{});
                auto layoutScaleBL1 = AscendC::Te::MakeNnLayout<fp8_e8m0_t>(
                    CeilDiv(scaleKL1_, MXFP_GROUP_SIZE), curN);
                auto tensorScaleBL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]), layoutScaleBL1);
                auto gmBlockScaleB = gmScaleB(
                    AscendC::Te::MakeCoord(kL1Offset / MXFP_GROUP_SIZE, 0),
                    AscendC::Te::MakeShape(CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN));
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
            }

            if (abL1LoopCnt_ == 0) {
                // The resident scaleA tensor is loaded only once for the whole
                // block because the A-full-load path never moves the A tile.
                auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(::Tile::CopyScaleGM2L1{});
                auto gmBlockScaleA = gmScaleA(
                    AscendC::Te::MakeCoord(0, 0),
                    AscendC::Te::MakeShape(curM, CeilDiv(k_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto tensorBlockAL1 =
                tensorAL1(AscendC::Te::MakeCoord(0, kL1Offset), 
                          AscendC::Te::MakeShape(curM, curPadKL1));

            // slice 
            if (abL1LoopCnt_ < kL1Iter_) {
                auto gmBlockA = gmA(
                    AscendC::Te::MakeCoord(0, kL1Offset), 
                    AscendC::Te::MakeShape(curM, curGmAKL1));
                // auto tensorTotalAL1 =
                //     AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<AType>(l1BufferAOffset_[0]), layoutAL1);
                // tensorAL1 = tensorTotalAL1(
                //     AscendC::Te::MakeCoord(0, kL1Offset),
                //     AscendC::Te::MakeShape(curM, curPadKL1));
                ::Tile::PadMxKAL1::PadZero(tensorBlockAL1, gmBlockA);
                AscendC::Te::Copy(copyGM2L1, tensorBlockAL1, gmBlockA);
            }

            auto layoutBL1 = MakeLayoutBL1{}(curGmBKL1, curN);
            auto tensorBL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
            auto gmBlockB = gmB(AscendC::Te::MakeCoord(kL1Offset, 0), 
                                AscendC::Te::MakeShape(curGmBKL1, curN));
            ::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);

            // Both A/B copies for the current L1 slot must complete before the
            // slot is sliced into L0 tiles.
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            auto tensorBlockScaleAL1 = tensorScaleAL1(
                AscendC::Te::MakeCoord(0, iter0 * CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE),
                AscendC::Te::MakeShape(curM, CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
            auto layoutScaleBL1 = AscendC::Te::MakeNnLayout<fp8_e8m0_t>(
                CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN);
            auto tensorScaleBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]), layoutScaleBL1);
            auto coordScaleKL1 =
                (iter0 % (scaleKL1_ / kL1_)) * CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
            auto tensorBlockScaleBL1 = tensorScaleBL1(
                AscendC::Te::MakeCoord(coordScaleKL1, 0),
                AscendC::Te::MakeShape(CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN));

            uint64_t kL0Iter = CeilDiv(curGmBKL1, baseK_);
            for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                // Split the current B-side L1 chunk into MMAD-sized slices while
                // reusing the resident A-side tensors from the same block.
                auto kL0Offset = iter1 * baseK_;
                auto curKL0 = (kL0Offset + baseK_ > curPadKL1) ? (curPadKL1 - kL0Offset) : baseK_;
                auto curScaleKL0 = CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);

                auto CopyL12L0 = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0{});
                auto layoutAL0 = AscendC::Te::MakeNzLayout<AType>(curM, curKL0);
                auto tensorAL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<AType>(l0Offset), layoutAL0);
                auto tensorBlockAL1L1 = tensorBlockAL1(
                    AscendC::Te::MakeCoord(0, kL0Offset), AscendC::Te::MakeShape(curM, curKL0));
                AscendC::Te::Copy(CopyL12L0, tensorAL0, tensorBlockAL1L1);

                auto layoutScaleAL0 = AscendC::Te::MakeZzLayout<fp8_e8m0_t>(curM, curScaleKL0);
                auto tensorScaleAL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL0AmemPtr<fp8_e8m0_t>(l0Offset), layoutScaleAL0);
                auto CopyL12L0MxScaleA3510 = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleA3510{});
                AscendC::Te::Copy(
                    CopyL12L0MxScaleA3510, tensorScaleAL0, tensorBlockScaleAL1,
                    AscendC::Te::MakeCoord(0, kL0Offset));

                auto layoutBL0 = AscendC::Te::MakeZnLayout<BType>(curKL0, curN);
                auto tensorBL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<BType>(l0Offset), layoutBL0);
                auto tensorBlockBL1 = tensorBL1(
                    AscendC::Te::MakeCoord(kL0Offset, 0), AscendC::Te::MakeShape(curKL0, curN));
                AscendC::Te::Copy(CopyL12L0, tensorBL0, tensorBlockBL1);

                auto layoutScaleBL0 = AscendC::Te::MakeNnLayout<fp8_e8m0_t>(curScaleKL0, curN);
                auto tensorScaleBL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL0BmemPtr<fp8_e8m0_t>(l0Offset), layoutScaleBL0);
                auto CopyL12L0MxScaleB3510 = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleB3510{});
                AscendC::Te::Copy(
                    CopyL12L0MxScaleB3510, tensorScaleBL0, tensorBlockScaleBL1,
                    AscendC::Te::MakeCoord(kL0Offset, 0));

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);

                uint8_t mmadUnitFlag =
                    (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
                bool mmadCmatrixInitVal = (iter0 == 0 && iter1 == 0);
                AscendC::Te::Mad(
                    AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<::Tile::MmadMx>>{}.with(
                        static_cast<uint16_t>(curM), static_cast<uint16_t>(CeilAlign(curKL0, MXFP_DIVISOR_SIZE)),
                        static_cast<uint16_t>(curN),
                        mmadUnitFlag, false, mmadCmatrixInitVal),
                    tensorL0C, tensorAL0, tensorBL0);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                l0PingPong_++;
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            if ((iter0 + 1) % (scaleKL1_ / kL1_) == 0 || iter0 == kL1Iter_ - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
                scaleLoopCnt_++;
            }
            abL1LoopCnt_++;
        }

        auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        // The whole block accumulates into one L0C tile, which is flushed once
        // after all K chunks have contributed.
        AscendC::Te::Copy(CopyL0C2GM, gmC, tensorL0C, AscendC::Te::FixpipeParams{FINAL_ACCUMULATION});
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t scaleAL1OneBuffer_ = 0UL;
    uint64_t scaleBL1OneBuffer_ = 0UL;
    uint64_t l1BufferAOffset_[4] = {0UL};
    uint64_t l1BufferBOffset_[4] = {0UL};
    uint64_t l1BufferScaleAOffset_[2] = {0UL};
    uint64_t l1BufferScaleBOffset_[2] = {0UL};
};
} // namespace Block

#endif
