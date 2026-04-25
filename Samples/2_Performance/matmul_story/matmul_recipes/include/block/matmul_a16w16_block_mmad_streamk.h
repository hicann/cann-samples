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
 * \file matmul_a16w16_block_mmad_streamk.h
 * \brief Block-level A16W16 MMAD pipeline for StreamK path.
 */

#pragma once

#include "kernel_utils/common_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "block_mmad.h"
#include "include/tensor.h"
#include "../policy/dispatch_policy.h"
#include "../utils/matmul_a16w16_constant.h"

namespace Block {
using namespace AscendC;

template <
    class DispatchPolicy_, class TypeA_, class LayoutA_, class TypeB_, class LayoutB_, class TypeC_, class LayoutC_>
class BlockMmad<
    DispatchPolicy_, TypeA_, LayoutA_, TypeB_, LayoutB_, TypeC_, LayoutC_,
    AscendC::Std::enable_if_t<AscendC::Std::is_base_of_v<MatmulA16W16MultiBlockWithStreamK, DispatchPolicy_>>> {
public:
    using TypeA = TypeA_;
    using TypeB = TypeB_;
    using TypeC = TypeC_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool transA = AscendC::IsSameType<LayoutA, AscendC::Te::DNLayoutFormat<TypeA>>::value;
 	static constexpr bool transB = AscendC::IsSameType<LayoutB, AscendC::Te::DNLayoutFormat<TypeB>>::value;
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    constexpr static uint16_t L1_EVENT_ID_OFFSET = 2;
    constexpr static uint64_t L1_BUFFER_NUM = 2;
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t bL1Init_{0};
    uint64_t aL1OneBuffer_{1};
    uint64_t bL1OneBuffer_{1};

    using MakeLayoutAL1 =
        AscendC::Std::conditional_t<transA, AscendC::Te::ZnLayoutFormat<TypeA>, AscendC::Te::NzLayoutFormat<TypeA>>;
    using MakeLayoutBL1 =
        AscendC::Std::conditional_t<transB, AscendC::Te::ZnLayoutFormat<TypeB>, AscendC::Te::NzLayoutFormat<TypeB>>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
    };

    // Custom mmadTrait internal value type
    constexpr static AscendC::Te::MmadTrait MMAD_TRAIT{0, false, false, true, AscendC::Te::MmadType::NORMAL};
    struct MmadMmTraitConfig { // Custom Trait class
        using TraitType = AscendC::Te::MmadTrait;
        constexpr static const TraitType value = MMAD_TRAIT;
    };
    __aicore__ inline BlockMmad(
        const TupleShape& problemShape, const TupleShape& tileL1Shape, const TupleShape& tileL0Shape)
    {
        m_ = Get<IDX_M_IDX>(problemShape);
        n_ = Get<IDX_N_IDX>(problemShape);
        k_ = Get<IDX_K_IDX>(problemShape);
        mL1_ = Get<IDX_M_IDX>(tileL1Shape);
        nL1_ = Get<IDX_N_IDX>(tileL1Shape);
        kL1_ = Get<IDX_K_IDX>(tileL1Shape);
        baseM_ = Get<IDX_M_IDX>(tileL0Shape);
        baseN_ = Get<IDX_N_IDX>(tileL0Shape);
        baseK_ = Get<IDX_K_IDX>(tileL0Shape);
        aL1OneBuffer_ = mL1_ * kL1_;
        bL1Init_ = aL1OneBuffer_ * L1_BUFFER_NUM;
        bL1OneBuffer_ = nL1_ * kL1_;
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
        // Set a synchronized variable inside a for loop
        #pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    }

    __aicore__ inline ~BlockMmad()
    {
        #pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    }

    template <typename TensorC, typename TensorA, typename TensorB, typename TensorWorkSpace>
    __aicore__ inline void operator()(
        TensorC gmC, TensorA gmA, TensorB gmB, TensorWorkSpace gmWorkSpace, const BlockShape& tileShape,
        int64_t kCntIndex, bool checkIsSkScene)
    {
        uint64_t curML1 = Get<MNK_M>(tileShape);
        uint64_t curNL1 = Get<MNK_N>(tileShape);
        uint64_t curSingleCoreK = Get<MNK_K>(tileShape);
        uint64_t curKL1Iter = (curSingleCoreK + kL1_ - 1) / kL1_;

        // LoC move out
        auto layoutL0C = AscendC::Te::MakeL0CLayout(curML1, curNL1);
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeL0CmemPtr<float>(0), layoutL0C);
        // Loop of k in L1
        for (uint64_t iter0 = 0; iter0 < curKL1Iter; ++iter0) {
            auto curKL1 = (iter0 + 1 == curKL1Iter) ? (curSingleCoreK - iter0 * kL1_) : kL1_;
            // Switch on pingpong, now only support double buffer in streamk
            uint64_t l1BufId = abL1LoopCnt_ & (L1_BUFFER_NUM - 1);
            uint64_t offsetAL1 = aL1OneBuffer_ * l1BufId * sizeof(TypeA);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            uint64_t offsetBL1 = (bL1Init_ + bL1OneBuffer_ * l1BufId) * sizeof(TypeB);
            // A GM->L1
            auto layoutAL1 = MakeLayoutAL1{}(static_cast<int64_t>(curML1), static_cast<int64_t>(curKL1));
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto tensorAL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<TypeA>(offsetAL1), layoutAL1);
            auto gmTileA = gmA(AscendC::Te::MakeCoord(0, iter0 * kL1_), AscendC::Te::MakeShape(curML1, curKL1));
            // Copy AL1
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId + L1_EVENT_ID_OFFSET);
            // B GM->L1
            auto layoutBL1 = MakeLayoutBL1{}(static_cast<int64_t>(curKL1), static_cast<int64_t>(curNL1));
            auto tensorBL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<TypeB>(offsetBL1), layoutBL1);
            auto gmTileB = gmB(AscendC::Te::MakeCoord(iter0 * kL1_, 0), AscendC::Te::MakeShape(curKL1, curNL1));
            // Copy BL1
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId + L1_EVENT_ID_OFFSET);

            // Loop of k in L0
            uint64_t kL0Iter = (curKL1 + baseK_ - 1) / baseK_;
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                // A L1->L0
                auto copyL12L0 = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0{});
                auto layoutAL0 = AscendC::Te::MakeNzLayout<TypeA>(curML1, curK0);
                auto tensorAL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<TypeA>(l0Offset), layoutAL0);
                auto tensorBlockAL1 =
                    tensorAL1(AscendC::Te::MakeCoord(0, iter1 * baseK_), AscendC::Te::MakeShape(curML1, curK0));
                AscendC::Te::Copy(copyL12L0, tensorAL0, tensorBlockAL1);

                if (iter1 == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId + L1_EVENT_ID_OFFSET);
                }
                // B L1->L0
                auto layoutBL0 = AscendC::Te::MakeZnLayout<TypeB>(curK0, curNL1);
                auto tensorBL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<TypeB>(l0Offset), layoutBL0);
                auto tensorBlockBL1 =
                    tensorBL1(AscendC::Te::MakeCoord(iter1 * baseK_, 0), AscendC::Te::MakeShape(curK0, curNL1));
                AscendC::Te::Copy(copyL12L0, tensorBL0, tensorBlockBL1);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);

                // Original mmad parameters
                uint8_t unitFlag =
                    (iter0 + 1 == curKL1Iter && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
                bool cmatrixInitVal = (iter0 == 0 && iter1 == 0);
                AscendC::Te::MmadParams mmadParams(curML1, curNL1, curK0, unitFlag, cmatrixInitVal);
                // Pass custom Trait type in mmad
                AscendC::Te::Mad(
                    AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, MmadMmTraitConfig>>{},
                    tensorL0C, tensorAL0, tensorBL0, mmadParams);

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                l0PingPong_++;
            }
            if (iter0 + 1 == curKL1Iter) {
                auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
                // Depending on checkIsSkScene, decide to move out to GM or WorkSpace
                if (checkIsSkScene) {
                    AscendC::Te::Copy(
                        CopyL0C2GM, gmWorkSpace, tensorL0C, AscendC::Te::FixpipeParams(FINAL_ACCUMULATION));
                } else {
                    AscendC::Te::Copy(CopyL0C2GM, gmC, tensorL0C, AscendC::Te::FixpipeParams(FINAL_ACCUMULATION));
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId + L1_EVENT_ID_OFFSET);
            abL1LoopCnt_++;
        }
    }
};
} // namespace Block
