/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_MATMUL_MXFP4_BLOCK_MMAD_A_FULL_LOAD_H
#define QUANT_MATMUL_MXFP4_BLOCK_MMAD_A_FULL_LOAD_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_utils/common_utils.h"
#include "kernel_utils/layout_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "include/tensor.h"
#include "../utils/quant_matmul_constant.h"
#include "../tile/tile_mmad_mx.h"
#include "../tile/copy_scale_l1_to_l0a.h"
#include "../tile/copy_scale_l1_to_l0b.h"
#include "../tile/copy_scale_gm_to_l1.h"

namespace Block {
using namespace AscendC;

template <class AType_, class BType_, class CType_>
class BlockMmadMxAFullLoad {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool transA = false;
    static constexpr bool transB = true;

    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t HALF_L0C_SIZE = L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(float);
    constexpr static uint64_t BLOCK_CUBE = 16UL;
    constexpr static uint64_t MXFP_GROUP_SIZE_LOCAL = 32UL;
    constexpr static uint64_t MXFP_DIVISOR_SIZE_LOCAL = 64UL;
    constexpr static uint64_t MXFP_MULTI_BASE_SIZE_LOCAL = 2UL;
    constexpr static uint64_t SCALE_BUFFER_NUM = 2;

    using MakeLayoutAL1 = AscendC::Te::NzLayoutFormat<AType>;
    using MakeLayoutBL1 = AscendC::Te::ZnLayoutFormat<BType>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1{0};
        uint64_t scaleKL1{0};
        uint64_t l1BufNum{2};
    };

    __aicore__ inline BlockMmadMxAFullLoad()
    {
        #pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmadMxAFullLoad()
    {
        #pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        AscendC::SetMMLayoutTransform(false);
    }

    __aicore__ inline void Init(
        const TupleShape& problemShape, const BlockShape& l0TileShape, const L1Params& l1Params)
    {
        m_ = Get<IDX_M_IDX>(problemShape);
        n_ = Get<IDX_N_IDX>(problemShape);
        k_ = Get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        scaleKL1_ = l1Params.scaleKL1;
        baseM_ = Get<IDX_M_IDX>(l0TileShape);
        baseN_ = Get<IDX_N_IDX>(l0TileShape);
        baseK_ = Get<IDX_K_IDX>(l0TileShape);
        l1BufNum_ = l1Params.l1BufNum;

        bL1OneBuffer_ = (baseN_ * kL1_) >> 1;
        scaleBL1OneBuffer_ = baseN_ * CeilDiv(kL1_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL;

        uint64_t mAlign = Align(baseM_, BLOCK_CUBE);
        uint64_t kAlign = Align(k_, MXFP_DIVISOR_SIZE_LOCAL);
        aL1OneBuffer_ = (mAlign * kAlign) >> 1;
        scaleAL1OneBuffer_ = baseM_ * CeilDiv(k_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL;
        l1BufferAOffset_[0] = bL1OneBuffer_ * (l1BufNum_ >> 1) + scaleBL1OneBuffer_;
        l1BufferScaleAOffset_[0] = l1BufferAOffset_[0] + aL1OneBuffer_;
        uint64_t b1Offset = l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_ >= (L1_SIZE >> 1)
            ? l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_
            : (L1_SIZE >> 1);
        for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            l1BufferBOffset_[bufferId] = b1Offset * (bufferId & 1) + bL1OneBuffer_ * (bufferId >> 1);
        }
        for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            l1BufferScaleBOffset_[bufferId] =
                l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
        }

        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    template <typename TensorAL1, typename TensorScaleAL1, typename TensorBL1, typename TensorScaleBL1, typename TensorL0C>
    __aicore__ inline void IterateL0(
        TensorAL1 tensorBlockAL1, TensorScaleAL1 tensorBlockScaleAL1,
        TensorBL1 tensorBL1, TensorScaleBL1 tensorBlockScaleBL1,
        uint64_t curM, uint64_t curN,
        uint64_t curGmBKL1, uint64_t curPadKL1, uint64_t iter0,
        TensorL0C& tensorL0C)
    {
        uint64_t kL0Iter = CeilDiv(curGmBKL1, baseK_);
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            auto kL0Offset = iter1 * baseK_;
            auto curKL0 = (kL0Offset + baseK_ > curPadKL1) ? (curPadKL1 - kL0Offset) : baseK_;
            auto curScaleKL0 = CeilDiv(curKL0, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL;
            uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);

            auto CopyL12L0 = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0{});
            auto layoutAL0 = AscendC::Te::MakeNzLayout<AType>(curM, curKL0);
            auto tensorAL0 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<AType>(l0Offset), layoutAL0);
            auto tensorBlockAL1Slice = tensorBlockAL1(
                AscendC::Te::MakeCoord(0, kL0Offset), AscendC::Te::MakeShape(curM, curKL0));
            AscendC::Te::Copy(CopyL12L0, tensorAL0, tensorBlockAL1Slice);

            auto layoutScaleAL0 = AscendC::Te::MakeZzLayout<fp8_e8m0_t>(curM, curScaleKL0);
            auto tensorScaleAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeL0AmemPtr<fp8_e8m0_t>(l0Offset), layoutScaleAL0);
            auto CopyL12L0MxScaleA = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleA3510{});
            AscendC::Te::Copy(
                CopyL12L0MxScaleA, tensorScaleAL0, tensorBlockScaleAL1,
                AscendC::Te::MakeCoord(0, kL0Offset));

            auto layoutBL0 = AscendC::Te::MakeZnLayout<BType>(curKL0, curN);
            auto tensorBL0 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<BType>(l0Offset), layoutBL0);
            auto tensorBlockBL1Slice = tensorBL1(
                AscendC::Te::MakeCoord(kL0Offset, 0), AscendC::Te::MakeShape(curKL0, curN));
            AscendC::Te::Copy(CopyL12L0, tensorBL0, tensorBlockBL1Slice);

            auto layoutScaleBL0 = AscendC::Te::MakeNnLayout<fp8_e8m0_t>(curScaleKL0, curN);
            auto tensorScaleBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeL0BmemPtr<fp8_e8m0_t>(l0Offset), layoutScaleBL0);
            auto CopyL12L0MxScaleB = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleB3510{});
            AscendC::Te::Copy(
                CopyL12L0MxScaleB, tensorScaleBL0, tensorBlockScaleBL1,
                AscendC::Te::MakeCoord(kL0Offset, 0));

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);

            uint8_t mmadUnitFlag =
                (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter)
                    ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
            bool mmadCmatrixInitVal = (iter0 == 0 && iter1 == 0);
            AscendC::Te::Mad(
                AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<::Tile::MmadMx>>{}.with(
                    static_cast<uint16_t>(curM),
                    static_cast<uint16_t>(CeilAlign(curKL0, MXFP_DIVISOR_SIZE_LOCAL)),
                    static_cast<uint16_t>(curN), mmadUnitFlag, false, mmadCmatrixInitVal),
                tensorL0C, tensorAL0, tensorBL0);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            l0PingPong_++;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void operator()(
        TensorA gmA, TensorB gmB, TensorScaleA gmScaleA, TensorScaleB gmScaleB, TensorC gmC,
        BlockShape singleShape)
    {
        auto curM = Get<IDX_M_TILEIDX>(singleShape);
        auto curN = Get<IDX_N_TILEIDX>(singleShape);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C = AscendC::Te::MakeL0CLayout(curM, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeL0CmemPtr<float>(l0cOffset), layoutL0C);

        auto layoutAL1 = MakeLayoutAL1{}(curM, k_);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeL1memPtr<AType>(l1BufferAOffset_[0]), layoutAL1);
        auto layoutScaleAL1 = AscendC::Te::MakeZzLayout<fp8_e8m0_t>(
            curM, CeilDiv(k_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL);
        auto tensorScaleAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(l1BufferScaleAOffset_[0]), layoutScaleAL1);

        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
            uint64_t kL1Offset = iter0 * kL1_;
            uint64_t curGmBKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1Offset) : kL1_;
            uint64_t curPadKL1 = CeilAlign(curGmBKL1, MXFP_DIVISOR_SIZE_LOCAL);
            auto curGmAKL1 = curGmBKL1;

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
            {
                auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(::Tile::CopyScaleGM2L1{});
                uint64_t curScaleKL1 = kL1_;
                if (kL1Offset + curScaleKL1 > k_) {
                    curScaleKL1 = k_ - kL1Offset;
                }
                auto layoutScaleBL1 = AscendC::Te::MakeNnLayout<fp8_e8m0_t>(
                    CeilDiv(kL1_, MXFP_GROUP_SIZE_LOCAL), curN);
                auto tensorScaleBL1Buf = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
                    layoutScaleBL1);
                auto gmBlockScaleB = gmScaleB(
                    AscendC::Te::MakeCoord(kL1Offset / MXFP_GROUP_SIZE_LOCAL, 0),
                    AscendC::Te::MakeShape(
                        CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL, curN));
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1Buf, gmBlockScaleB);
            }

            if (abL1LoopCnt_ == 0) {
                auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(::Tile::CopyScaleGM2L1{});
                auto gmBlockScaleA = gmScaleA(
                    AscendC::Te::MakeCoord(0, 0),
                    AscendC::Te::MakeShape(
                        curM, CeilDiv(k_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL));
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto tensorBlockAL1 = tensorAL1(
                AscendC::Te::MakeCoord(0, kL1Offset), AscendC::Te::MakeShape(curM, curGmAKL1));

            if (abL1LoopCnt_ < kL1Iter_) {
                auto gmBlockA = gmA(
                    AscendC::Te::MakeCoord(0, kL1Offset), AscendC::Te::MakeShape(curM, curGmAKL1));
                AscendC::Te::Copy(copyGM2L1, tensorBlockAL1, gmBlockA);
            }

            auto layoutBL1 = MakeLayoutBL1{}(curGmBKL1, curN);
            auto tensorBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeL1memPtr<BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
            auto gmBlockB = gmB(
                AscendC::Te::MakeCoord(kL1Offset, 0), AscendC::Te::MakeShape(curGmBKL1, curN));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            auto tensorBlockScaleAL1 = tensorScaleAL1(
                AscendC::Te::MakeCoord(
                    0, iter0 * CeilDiv(kL1_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL),
                AscendC::Te::MakeShape(
                    curM, CeilDiv(kL1_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL));

            auto layoutScaleBL1ForL0 = AscendC::Te::MakeNnLayout<fp8_e8m0_t>(
                CeilDiv(kL1_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL, curN);
            auto tensorBlockScaleBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
                layoutScaleBL1ForL0);

            IterateL0(tensorBlockAL1, tensorBlockScaleAL1,
                      tensorBL1, tensorBlockScaleBL1,
                      curM, curN, curGmBKL1, curPadKL1, iter0, tensorL0C);

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
            scaleLoopCnt_++;
            abL1LoopCnt_++;
        }

        auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(CopyL0C2GM, gmC, tensorL0C, AscendC::Te::FixpipeParams{FINAL_ACCUMULATION});
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t baseM_;
    uint64_t baseN_;
    uint64_t baseK_;
    uint64_t kL1_;
    uint64_t scaleKL1_;
    uint64_t l1BufNum_;
    uint64_t kL1Iter_;
    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool enableL0cPingPong_{false};
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
