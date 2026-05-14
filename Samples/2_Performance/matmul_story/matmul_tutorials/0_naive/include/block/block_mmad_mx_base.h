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

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../../common/kernel_utils/common_utils.h"
#include "include/tensor_api/tensor.h"
#include "../utils/quant_matmul_constant.h"
#include "../tile/tile_mmad_mx.h"
#include "../tile/copy_scale_l1_to_l0a.h"
#include "../tile/copy_scale_l1_to_l0b.h"

namespace Block {
template <class AType_, class BType_, class CType_>
class BlockMmadMx {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool transA = false;
    static constexpr bool transB = true;
    static constexpr int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    static constexpr int32_t SCALE_C0 = 2;
    static constexpr int32_t L0C_C0 = 16;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1{0};
    };

    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t baseM_{256};
    uint64_t baseN_{256};
    uint64_t baseK_{128};
    uint64_t kL1_{1024};
    uint64_t kL1TileNum;
    uint64_t tailKL1;

    __aicore__ inline void Init(const TupleShape& problemShape, const BlockShape& l0TileShape,
                                const L1Params& l1Params)
    {
        m_ = static_cast<uint64_t>(AscendC::Te::Get<IDX_M_IDX>(problemShape));
        n_ = static_cast<uint64_t>(AscendC::Te::Get<IDX_N_IDX>(problemShape));
        k_ = static_cast<uint64_t>(AscendC::Te::Get<IDX_K_IDX>(problemShape));
        baseM_ = static_cast<uint64_t>(AscendC::Te::Get<IDX_M_IDX>(l0TileShape));
        baseN_ = static_cast<uint64_t>(AscendC::Te::Get<IDX_N_IDX>(l0TileShape));
        baseK_ = static_cast<uint64_t>(AscendC::Te::Get<IDX_K_IDX>(l0TileShape));
        if (baseK_ == 0) {
            baseK_ = 128 / sizeof(fp4x2_e2m1_t);
        }
        kL1_ = (l1Params.kL1 > 0) ? l1Params.kL1 : (baseK_ * 4);
        kL1TileNum = CeilDiv(k_, kL1_);
        tailKL1 = k_ - ((kL1TileNum - 1) * kL1_);
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void operator()(
        TensorA aGlobal,
        TensorB bGlobal,
        TensorScaleA scaleAGlobal,
        TensorScaleB scaleBGlobal,
        TensorC cGlobal,
        const BlockShape& singleShape)
    {
        uint64_t curM = static_cast<uint64_t>(AscendC::Te::Get<IDX_M_TILEIDX>(singleShape));
        uint64_t curN = static_cast<uint64_t>(AscendC::Te::Get<IDX_N_TILEIDX>(singleShape));

        auto layoutL0C = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<L0C_C0>>(curM, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(0), layoutL0C);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);

        uint64_t offsetA = 0;
        uint64_t offsetB = 0;
        uint64_t offsetScaleA = 0;
        uint64_t offsetScaleB = 0;

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        for (uint64_t iter0 = 0; iter0 < kL1TileNum; ++iter0) {
            uint64_t curKL1 = (iter0 == (kL1TileNum - 1)) ? tailKL1 : kL1_;

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            uint64_t offsetAL1 = 0;
            uint64_t offsetBL1 = baseM_ * kL1_ / OFFSET_4_8;
            uint64_t offsetScaleAL1 = (baseM_ + baseN_) * kL1_ / OFFSET_4_8;
            uint64_t offsetScaleBL1 = (baseM_ + baseN_) * kL1_ / OFFSET_4_8 + baseM_ * CeilDiv(kL1_, MXFP_GROUP_SIZE);
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto layoutAL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>(curM, curKL1);
            auto tensorAL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(offsetAL1), layoutAL1);
            auto gmBlockA = aGlobal.Slice(AscendC::Te::MakeCoord(0, offsetA),
                AscendC::Te::MakeShape(curM, curKL1));
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);

            auto layoutBL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>(curKL1, curN);
            auto tensorBL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(offsetBL1), layoutBL1);
            auto gmBlockB = bGlobal.Slice(AscendC::Te::MakeCoord(offsetB, 0),
                AscendC::Te::MakeShape(curKL1, curN));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);
            auto gmBlockScaleA = scaleAGlobal.Slice(AscendC::Te::MakeCoord(0, offsetScaleA),
                AscendC::Te::MakeShape(
                    curM, CeilDiv(curKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
            auto gmBlockScaleB = scaleBGlobal.Slice(AscendC::Te::MakeCoord(offsetScaleB, 0),
                AscendC::Te::MakeShape(
                    CeilDiv(curKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN));
            auto copyScaleGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto layoutScaleAL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                curM, CeilDiv(curKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
            auto tensorScaleAL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(offsetScaleAL1), layoutScaleAL1);
            AscendC::Te::Copy(copyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

            auto layoutScaleBL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                CeilDiv(curKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN);
            auto tensorScaleBL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(offsetScaleBL1), layoutScaleBL1);
            AscendC::Te::Copy(copyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
            offsetA += curKL1;
            offsetScaleA += CeilDiv(curKL1, MXFP_GROUP_SIZE);
            offsetB += curKL1;
            offsetScaleB += CeilDiv(curKL1, MXFP_GROUP_SIZE);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);

            uint64_t kL0TileNum = CeilDiv(curKL1, baseK_);
            uint64_t tailKL0 = curKL1 - (kL0TileNum - 1) * baseK_;
            for (uint64_t iter1 = 0; iter1 < kL0TileNum; ++iter1) {
                uint64_t curKL0 = (iter1 == (kL0TileNum - 1)) ? tailKL0 : baseK_;

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
                uint64_t l0Offset = 0;
                uint64_t kL0Offset = iter1 * baseK_;
                auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
                auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
                auto layoutAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>(curM, curKL0);
                auto tensorAL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset), layoutAL0);
                auto tensorBlockAL1 = tensorAL1.Slice(AscendC::Te::MakeCoord(0, kL0Offset), AscendC::Te::MakeShape(curM, curKL0));
                AscendC::Te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1);

                auto layoutBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>(curKL0, curN);
                auto tensorBL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset), layoutBL0);
                auto tensorBlockBL1 = tensorBL1.Slice(AscendC::Te::MakeCoord(kL0Offset, 0), AscendC::Te::MakeShape(curKL0, curN));
                AscendC::Te::Copy(copyL12L0B, tensorBL0, tensorBlockBL1);

                auto layoutScaleAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                    curM, CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
                auto tensorScaleAL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, fp8_e8m0_t>(l0Offset), layoutScaleAL0);
                auto tensorBlockScaleAL1 = tensorScaleAL1.Slice(AscendC::Te::MakeCoord(0, 0),
                    AscendC::Te::MakeShape(curM, CeilDiv(curKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
                auto copyL12L0MxScaleA = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleA3510{});
                AscendC::Te::Copy(
                    copyL12L0MxScaleA, tensorScaleAL0, tensorBlockScaleAL1, AscendC::Te::MakeCoord(0, kL0Offset));

                auto layoutScaleBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                    CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN);
                auto tensorScaleBL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, fp8_e8m0_t>(l0Offset), layoutScaleBL0);
                auto tensorBlockScaleBL1 = tensorScaleBL1.Slice(AscendC::Te::MakeCoord(0, 0),
                    AscendC::Te::MakeShape(CeilDiv(curKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN));
                auto copyL12L0MxScaleB = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleB3510{});
                AscendC::Te::Copy(
                    copyL12L0MxScaleB, tensorScaleBL0, tensorBlockScaleBL1, AscendC::Te::MakeCoord(kL0Offset, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);

                uint8_t mmadUnitFlag = 0;
                bool mmadCmatrixInitVal = (iter0 == 0 && iter1 == 0);
AscendC::Te::MmadParams mmadParams;
mmadParams.m = static_cast<uint16_t>(curM);
mmadParams.k = static_cast<uint16_t>(CeilAlign(curKL0, MXFP_DIVISOR_SIZE));
mmadParams.n = static_cast<uint16_t>(curN);
mmadParams.unitFlag = mmadUnitFlag;
mmadParams.cmatrixInitVal = mmadCmatrixInitVal;
AscendC::Te::Mmad(
    AscendC::Te::MmadAtom<
        AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, AscendC::Te::MmadTraitMX>>{}
        .with(mmadParams),
                    tensorL0C, tensorAL0, tensorBL0);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);

        auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(copyL0C2GM, cGlobal, tensorL0C);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
    }
};
}
