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
 * \file quant_grouped_matmul_mx_kernel.h
 * \brief Sample-side grouped MX kernel wrapper.
 */
#pragma once

#include "kernel_basic_intf.h"
#include "include/tensor_api/tensor.h"
#include "../../common/kernel_utils/common_utils.h"
#include "../block/quant_grouped_matmul_mx_block_mmad.h"
#include "../block/quant_grouped_matmul_mx_block_scheduler_split_m.h"
#include "../block/block_scheduler_utils.h"
#include "../policy/dispatch_policy.h"
#include "../tiling/quant_grouped_matmul_mx_tiling_data.h"
#include "../utils/grouped_matmul_constant.h"

namespace Kernel {

template <class ProblemShape, class BlockMmad, class BlockScheduler>
class KernelQGmmMx {
    static_assert(
        AscendC::Std::is_same_v<BlockScheduler, Block::BlockSchedulerGmmAswtWithTailSplit>,
        "Only BlockSchedulerGmmAswtWithTailSplit is supported");

public:
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutScaleA = typename BlockMmad::LayoutScaleA;
    using LayoutScaleB = typename BlockMmad::LayoutScaleB;
    using LayoutAPattern = AscendC::Te::GetLayoutPattern<decltype(LayoutA{}(0L, 0L))>;
    using LayoutBPattern = AscendC::Te::GetLayoutPattern<decltype(LayoutB{}(0L, 0L))>;
    static constexpr bool transA = AscendC::Std::is_same_v<LayoutAPattern, AscendC::Te::DNExtLayoutPtn> ||
                                   AscendC::Std::is_same_v<LayoutAPattern, AscendC::Te::ZNLayoutPtn>;
    static constexpr bool transB = AscendC::Std::is_same_v<LayoutBPattern, AscendC::Te::DNExtLayoutPtn> ||
                                   AscendC::Std::is_same_v<LayoutBPattern, AscendC::Te::ZNLayoutPtn>;
    static constexpr bool kWeightNzGm_ = AscendC::Std::is_same_v<LayoutBPattern, AscendC::Te::ZNLayoutPtn> ||
                                         AscendC::Std::is_same_v<LayoutBPattern, AscendC::Te::NZLayoutPtn>;
    static constexpr int32_t SCALE_C0 = 2;
    using ScaleBNzLayout = AscendC::Te::FrameLayoutFormat<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
    static constexpr bool kBscaleNzGm_ = AscendC::Std::is_same_v<LayoutScaleB, ScaleBNzLayout>;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockMmadShape = typename BlockMmad::BlockShape;
    using ScaleType = fp8_e8m0_t;

    struct TilingParams {
        uint32_t groupNum{0U};
        uint32_t m{0U};
        uint32_t n{0U};
        uint32_t k{0U};
        uint32_t baseM{0U};
        uint32_t baseN{0U};
        uint32_t baseK{0U};
        uint32_t kAL1{0U};
        uint32_t kBL1{0U};
        uint32_t scaleKAL1{0U};
        uint8_t dbL0C{0U};
    };

    struct Params {
        ProblemShape problemShape;
        typename BlockMmad::Params mmadParams;
        typename BlockScheduler::Params schedulerParams;
        TilingParams kernelParams;
        GM_ADDR groupListGmAddr{nullptr};
        Params() = default;
    };

    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }

private:
    __aicore__ inline void ResetGmAddr(const Params& params)
    {
        xGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        wGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        x1ScaleGmAddr_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.x1ScaleGmAddr);
        x2ScaleGmAddr_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.x2ScaleGmAddr);
        yGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    }

    __aicore__ inline int64_t GetScaleK(int64_t k) const
    {
        return CeilDiv(k, static_cast<int64_t>(GroupedMatmulRecipe::MX_DIVISOR_SIZE)) *
               static_cast<int64_t>(GroupedMatmulRecipe::MX_MULTI_SIZE);
    }

    __aicore__ inline int64_t GetWeightNzSize(int64_t n, int64_t k) const
    {
        constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
        constexpr int64_t nzN0 = transB ? AscendC::BLOCK_CUBE : C0_SIZE;
        constexpr int64_t nzK0 = transB ? C0_SIZE : AscendC::BLOCK_CUBE;
        return CeilDiv(n, nzN0) * nzN0 * CeilDiv(k, nzK0) * nzK0;
    }

    __aicore__ inline int64_t GetScaleBNzSize(int64_t n, int64_t k) const
    {
        int64_t n1 = CeilDiv(n, static_cast<int64_t>(AscendC::BLOCK_CUBE));
        int64_t kg = CeilDiv(k, static_cast<int64_t>(GroupedMatmulRecipe::MX_DIVISOR_SIZE));
        return n1 * kg * GroupedMatmulRecipe::CUBE_BLOCK * GroupedMatmulRecipe::MX_MULTI_SIZE;
    }

    __aicore__ static inline void SetSchedulerTailAlign(BlockScheduler& bs)
    {
        constexpr uint32_t mTailAlign =
            (AscendC::Std::is_same_v<LayoutAPattern, AscendC::Te::NDExtLayoutPtn> ||
             AscendC::Std::is_same_v<LayoutAPattern, AscendC::Te::NZLayoutPtn>) ?
                static_cast<uint32_t>(AscendC::BLOCK_CUBE) // (m, k) -> cube tile alignment
                :
                GroupedMatmulRecipe::INNER_AXIS_ALIGN; // (k, m) -> ND2NZ cacheline alignment
        constexpr uint32_t nTailAlign = (AscendC::Std::is_same_v<LayoutBPattern, AscendC::Te::NDExtLayoutPtn> ||
                                         AscendC::Std::is_same_v<LayoutBPattern, AscendC::Te::NZLayoutPtn>) ?
                                            GroupedMatmulRecipe::INNER_AXIS_ALIGN // (k, n) -> ND2NZ cacheline alignment
                                            :
                                            static_cast<uint32_t>(AscendC::BLOCK_CUBE); // (n, k) -> cube tile alignment
        bs.SetTailAlign(mTailAlign, nTailAlign);
    }

    __aicore__ inline void InitZeroOutputForEmptyGroups(const Params& params)
    {
        if ASCEND_IS_AIC {
            return;
        }
        // Only needed in splitK mode (transA=true) when some groups have k=0
        if constexpr (!transA) {
            return;
        }

        const int64_t m = static_cast<int64_t>(params.kernelParams.m);
        const int64_t n = static_cast<int64_t>(params.kernelParams.n);
        if (m <= 0 || n <= 0) {
            return;
        }

        // UB buffer for zero init (256B aligned)
        constexpr uint32_t UB_ALIGN_SIZE = 32;
        constexpr uint32_t MAX_REPEAT_TIMES = 255;
        constexpr uint32_t ONE_BLK_SIZE = 32;
        uint64_t usedBlockNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();

        // Calculate init buffer size in elements
        uint64_t ySize = static_cast<uint64_t>(m) * n;
        uint64_t initSize = (MAX_REPEAT_TIMES * ONE_BLK_SIZE) / sizeof(CType);
        uint64_t perCoreSize = CeilDiv(ySize * sizeof(CType), usedBlockNum);
        perCoreSize = Align(perCoreSize, UB_ALIGN_SIZE) / sizeof(CType);
        initSize = Min(initSize, perCoreSize);

        // Local tensor for zero init
        AscendC::LocalTensor<CType> initLocal = AscendC::LocalTensor<CType>(
            AscendC::TPosition::VECCALC, 0, MAX_REPEAT_TIMES * UB_ALIGN_SIZE / sizeof(CType));

        uint64_t yBaseOffset = 0;
        bool isKZeroInit = false;
        uint32_t blockIdx = AscendC::GetBlockIdx();

        // Local groupList tensor from params
        AscendC::GlobalTensor<int64_t> groupListGlobal;
        groupListGlobal.SetGlobalBuffer((__gm__ int64_t*)params.groupListGmAddr);
        auto cGm = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);

        for (uint32_t groupIdx = 0; groupIdx < params.kernelParams.groupNum; ++groupIdx) {
            int32_t groupK = static_cast<int32_t>(groupListGlobal.GetValue(groupIdx));
            if (groupK == 0) {
                // Calculate this core's portion
                uint64_t realCoreNum = Min(CeilDiv(ySize, initSize), static_cast<uint64_t>(usedBlockNum));
                if (blockIdx < realCoreNum) {
                    if (!isKZeroInit) {
                        // Initialize UB buffer with zeros (only once)
                        AscendC::Duplicate<CType>(initLocal, 0, initSize);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                        isKZeroInit = true;
                    }

                    // Calculate offset and size for this core
                    uint64_t yOffset = perCoreSize * blockIdx;
                    uint64_t outCurSize = (blockIdx == realCoreNum - 1) ? (ySize - yOffset) : perCoreSize;
                    uint64_t movRound = outCurSize / initSize;
                    uint64_t movTail = outCurSize - movRound * initSize;

                    // GM address for this group's output (set to group base, use relative offset in DataCopyPad)
                    AscendC::GlobalTensor<CType> yInitGlobal;
                    yInitGlobal.SetGlobalBuffer(cGm + yBaseOffset);

                    // DataCopy from UB to GM
                    AscendC::DataCopyExtParams ub2GmParams{1, static_cast<uint32_t>(initSize * sizeof(CType)), 0, 0, 0};
                    for (uint64_t i = 0; i < movRound; ++i) {
                        AscendC::DataCopyPad(yInitGlobal[yOffset + i * initSize], initLocal, ub2GmParams);
                    }
                    if (movTail != 0) {
                        ub2GmParams.blockLen = static_cast<uint32_t>(movTail * sizeof(CType));
                        AscendC::DataCopyPad(yInitGlobal[yOffset + movRound * initSize], initLocal, ub2GmParams);
                    }
                }
            }
            yBaseOffset += ySize;
        }
    }

    __aicore__ inline void Run(const Params& params)
    {
        if ASCEND_IS_AIV { // init zero in aiv when splitK
            InitZeroOutputForEmptyGroups(params);
        }
        if ASCEND_IS_AIC {
            ResetGmAddr(params);
            Init(params);
            using SchedulerOp = BlockScheduler;
            SchedulerOp bs(params.schedulerParams);
            SetSchedulerTailAlign(bs);
            for (uint32_t groupIdx = 0; groupIdx < groupNum_ - 1; ++groupIdx) {
                SetMNK(groupIdx);
                if (AscendC::Std::get<MNK_M>(problemShape_) <= 0 || AscendC::Std::get<MNK_K>(problemShape_) <= 0) {
                    continue;
                }
                BaseMBalance(bs, AscendC::Std::get<MNK_M>(problemShape_), params.kernelParams.baseM);
                bs.UpdateNextProblem(problemShape_);
                ProcessSingleGroup<false>(params, bs, groupIdx);
            }

            // Process the last group (groupNum_ must be greater than 0)
            uint32_t groupIdx = groupNum_ - 1;
            SetMNK(groupIdx);
            if (AscendC::Std::get<MNK_M>(problemShape_) > 0 && AscendC::Std::get<MNK_K>(problemShape_) > 0) {
                BaseMBalance(bs, AscendC::Std::get<MNK_M>(problemShape_), params.kernelParams.baseM);
                bs.UpdateNextProblem(problemShape_);
                if (IfNeedSplit(bs)) {
                    bs.UpdateTailTile();
                    ProcessSingleGroup<true>(params, bs, groupIdx);
                } else {
                    ProcessSingleGroup<false>(params, bs, groupIdx);
                }
            }
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        groupListPtr_ = params.groupListGmAddr;
        groupNum_ = params.kernelParams.groupNum;
        curBaseM_ = params.kernelParams.baseM;
        AscendC::Std::get<MNK_M>(problemShape_) = params.kernelParams.m;
        AscendC::Std::get<MNK_N>(problemShape_) = params.kernelParams.n;
        AscendC::Std::get<MNK_K>(problemShape_) = params.kernelParams.k;
        if constexpr (!transA) {
            constexpr bool isFp4Type = AscendC::Std::is_one_of_v<AType, fp4x2_e2m1_t, fp4x2_e1m2_t>;
            int64_t n = static_cast<int64_t>(params.kernelParams.n);
            int64_t k = static_cast<int64_t>(params.kernelParams.k);
            if constexpr (!kWeightNzGm_) {
                perGroupBOffset_ = n * k;
            } else {
                if constexpr (transB) {
                    perGroupBOffset_ = Align16(n) * (isFp4Type ? Align64(k) : Align32(k));
                } else {
                    perGroupBOffset_ = (isFp4Type ? Align64(n) : Align32(n)) * Align16(k);
                }
            }
        }
        if (groupListPtr_ != nullptr) {
            groupListGlobal_.SetGlobalBuffer((__gm__ int64_t*)groupListPtr_);
        }
        TupleShape l0Shape{
            static_cast<int64_t>(params.kernelParams.baseM), static_cast<int64_t>(params.kernelParams.baseN),
            static_cast<int64_t>(params.kernelParams.baseK)};
        typename BlockMmad::L1Params l1Params{
            static_cast<uint64_t>(params.kernelParams.kAL1), static_cast<uint64_t>(params.kernelParams.kBL1),
            static_cast<uint64_t>(params.kernelParams.scaleKAL1)};
        mmadOp_.Init(problemShape_, l0Shape, l1Params, params.kernelParams.dbL0C == GroupedMatmulRecipe::DOUBLE_BUFFER);
    }

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2Cache(
        const TupleShape& problemShape, uint64_t curBaseM, uint64_t baseN, TensorB& gmB, TensorScaleB& gmScaleB)
    {
        if constexpr (kWeightNzGm_) {
            if (curBaseM >= AscendC::Std::get<MNK_M>(problemShape)) {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
            } else {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
            }
        } else {
            if constexpr (transB) {
                if (curBaseM >= AscendC::Std::get<MNK_M>(problemShape) &&
                    (AscendC::Std::get<MNK_K>(problemShape) & 0xff) == 0) {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                } else {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                }
            } else {
                if (curBaseM >= AscendC::Std::get<MNK_M>(problemShape) &&
                    (AscendC::Std::get<MNK_N>(problemShape) & 0xff) == 0 && (baseN & 0xff) == 0) {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                } else {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                }
            }
        }
    }

    template <class SchedulerOp>
    __aicore__ inline void BaseMBalance(SchedulerOp& bs, int64_t m, int64_t baseM)
    {
        if constexpr (!transA) {
            int64_t mCnt = CeilDiv(m, baseM);
            curBaseM_ = Align16(CeilDiv(m, mCnt));
            bs.UpdateBaseM(curBaseM_);
        }
    }

    __aicore__ inline bool IfNeedSplit(const BlockScheduler& bs) const
    {
        return (bs.GetEndBlockIdx() + 1) <= (AscendC::GetBlockNum() >> 1);
    }

    __aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx)
    {
        int32_t splitValue = static_cast<int32_t>(groupListGlobal_.GetValue(groupIdx));
        preGroupListSum_ += splitValue;
        return splitValue;
    }

    __aicore__ inline void SetMNK(uint32_t groupIdx)
    {
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx);
        if constexpr (transA) {
            AscendC::Std::get<MNK_K>(problemShape_) = splitValue;
        } else {
            AscendC::Std::get<MNK_M>(problemShape_) = splitValue;
        }
    }

    __aicore__ inline void UpdateOffset(uint32_t groupIdx)
    {
        if (groupIdx == 0) {
            return;
        }
        int64_t m = AscendC::Std::get<MNK_M>(problemShape_);
        int64_t n = AscendC::Std::get<MNK_N>(problemShape_);
        int64_t k = AscendC::Std::get<MNK_K>(problemShape_);
        constexpr bool isFp4Type = AscendC::Std::is_one_of_v<AType, fp4x2_e2m1_t, fp4x2_e1m2_t>;
        constexpr uint64_t sizeShift = isFp4Type ? 1UL : 0UL;
        if constexpr (!transA) {
            AscendC::Std::get<0>(baseOffset_) = ((preGroupListSum_ - m) * k) >> sizeShift;
            AscendC::Std::get<1>(baseOffset_) = (perGroupBOffset_ * static_cast<int64_t>(groupIdx)) >> sizeShift;
            int64_t scaleK = CeilDiv(k, static_cast<int64_t>(GroupedMatmulRecipe::MX_DIVISOR_SIZE)) *
                             GroupedMatmulRecipe::MX_MULTI_SIZE;
            AscendC::Std::get<2>(baseOffset_) = (preGroupListSum_ - m) * scaleK;
            if constexpr (kBscaleNzGm_) {
                AscendC::Std::get<3>(baseOffset_) = static_cast<int64_t>(groupIdx) * GetScaleBNzSize(n, k);
            } else {
                AscendC::Std::get<3>(baseOffset_) = static_cast<int64_t>(groupIdx) * n * scaleK;
            }
            AscendC::Std::get<4>(baseOffset_) = (preGroupListSum_ - m) * n;
        } else {
            AscendC::Std::get<0>(baseOffset_) = (m * (preGroupListSum_ - k)) >> sizeShift;
            AscendC::Std::get<1>(baseOffset_) = (n * (preGroupListSum_ - k)) >> sizeShift;
            int64_t scaleStart =
                ((preGroupListSum_ - k) / static_cast<int64_t>(GroupedMatmulRecipe::MX_DIVISOR_SIZE) + groupIdx) *
                GroupedMatmulRecipe::MX_MULTI_SIZE;
            AscendC::Std::get<2>(baseOffset_) = m * scaleStart;
            AscendC::Std::get<3>(baseOffset_) = n * scaleStart;
            AscendC::Std::get<4>(baseOffset_) = static_cast<int64_t>(groupIdx) * m * n;
        }
    }

    template <bool isLastGroupAndNeedSplit, class SchedulerOp>
    __aicore__ inline void ProcessSingleGroup(const Params& params, SchedulerOp& bs, uint32_t groupIdx)
    {
        BlockCoord tileIdx;
        if (!bs.GetTileIdx(tileIdx)) {
            return;
        }
        UpdateOffset(groupIdx);
        int64_t groupM = AscendC::Std::get<MNK_M>(problemShape_);
        int64_t groupN = AscendC::Std::get<MNK_N>(problemShape_);
        int64_t groupK = AscendC::Std::get<MNK_K>(problemShape_);
        int64_t scaleK = GetScaleK(groupK);
        __gm__ AType* groupAPtr = xGmAddr_ + AscendC::Std::get<0>(baseOffset_);
        __gm__ BType* groupBPtr = wGmAddr_ + AscendC::Std::get<1>(baseOffset_);
        __gm__ ScaleType* groupScaleAPtr = x1ScaleGmAddr_ + AscendC::Std::get<2>(baseOffset_);
        __gm__ ScaleType* groupScaleBPtr = x2ScaleGmAddr_ + AscendC::Std::get<3>(baseOffset_);
        __gm__ CType* groupCPtr = yGmAddr_ + AscendC::Std::get<4>(baseOffset_);
        auto layoutA = LayoutA{}(groupM, groupK);
        auto layoutB = LayoutB{}(groupK, groupN);
        auto layoutScaleA = LayoutScaleA{}(groupM, scaleK);
        auto layoutScaleB = LayoutScaleB{}(scaleK, groupN);
        auto groupA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(groupAPtr), layoutA);
        auto groupB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(groupBPtr), layoutB);
        auto groupScaleA =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(groupScaleAPtr), layoutScaleA);
        auto groupScaleB =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(groupScaleBPtr), layoutScaleB);
        auto groupC = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(groupCPtr),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(groupM, groupN));
        if constexpr (!isLastGroupAndNeedSplit) {
            SetL2Cache(problemShape_, curBaseM_, params.kernelParams.baseN, groupB, groupScaleB);
        }
        mmadOp_.UpdateParamsForNextProblem(problemShape_);
        do {
            BlockShape singleShape = bs.GetBlockShape(tileIdx);
            if (AscendC::Std::get<MNK_M>(singleShape) <= 0 || AscendC::Std::get<MNK_N>(singleShape) <= 0) {
                return;
            }
            // block offset
            int64_t mOffset =
                AscendC::Std::get<0>(tileIdx) * static_cast<int64_t>(curBaseM_) + AscendC::Std::get<2>(singleShape);
            int64_t nOffset = AscendC::Std::get<1>(tileIdx) * static_cast<int64_t>(params.kernelParams.baseN) +
                              AscendC::Std::get<3>(singleShape);
            int64_t tileM = AscendC::Std::get<MNK_M>(singleShape);
            int64_t tileN = AscendC::Std::get<MNK_N>(singleShape);
            auto gmBlockA = groupA.Slice(AscendC::Te::MakeCoord(mOffset, 0), AscendC::Te::MakeShape(tileM, groupK));
            auto gmBlockB = groupB.Slice(AscendC::Te::MakeCoord(0, nOffset), AscendC::Te::MakeShape(groupK, tileN));
            auto gmBlockScaleA =
                groupScaleA.Slice(AscendC::Te::MakeCoord(mOffset, 0), AscendC::Te::MakeShape(tileM, scaleK));
            auto gmBlockScaleB =
                groupScaleB.Slice(AscendC::Te::MakeCoord(0, nOffset), AscendC::Te::MakeShape(scaleK, tileN));
            auto gmBlockC =
                groupC.Slice(AscendC::Te::MakeCoord(mOffset, nOffset), AscendC::Te::MakeShape(tileM, tileN));
            BlockMmadShape mmadShape{tileM, tileN, static_cast<int64_t>(groupK)};
            mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockC, mmadShape);
        } while (bs.GetTileIdx(tileIdx));
    }

private:
    BlockMmad mmadOp_;
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0};
    int64_t preGroupListSum_{0};
    int64_t perGroupBOffset_{0};
    AscendC::GlobalTensor<int64_t> groupListGlobal_;
    GM_ADDR groupListPtr_{nullptr};
    __gm__ AType* xGmAddr_{nullptr};
    __gm__ BType* wGmAddr_{nullptr};
    __gm__ ScaleType* x1ScaleGmAddr_{nullptr};
    __gm__ ScaleType* x2ScaleGmAddr_{nullptr};
    __gm__ CType* yGmAddr_{nullptr};
    uint32_t groupNum_{0};
    uint32_t curBaseM_{0};
};

template <class ProblemShape, class BlockMmad, class BlockScheduler>
using QuantGroupedMatmulMxKernelSplitM = KernelQGmmMx<ProblemShape, BlockMmad, BlockScheduler>;

template <class ProblemShape, class BlockMmad, class BlockScheduler>
using QuantGroupedMatmulMxfp4KernelSplitM = QuantGroupedMatmulMxKernelSplitM<ProblemShape, BlockMmad, BlockScheduler>;

template <class ProblemShape, class BlockMmad, class BlockScheduler>
using QuantGroupedMatmulMxfp4KernelWeightNz = QuantGroupedMatmulMxKernelSplitM<ProblemShape, BlockMmad, BlockScheduler>;

template <class ProblemShape, class BlockMmad, class BlockScheduler>
using QuantGroupedMatmulMxfp8KernelSplitM = QuantGroupedMatmulMxKernelSplitM<ProblemShape, BlockMmad, BlockScheduler>;

template <class ProblemShape, class BlockMmad, class BlockScheduler>
using QuantGroupedMatmulMxfp8KernelWeightNz = QuantGroupedMatmulMxKernelSplitM<ProblemShape, BlockMmad, BlockScheduler>;

template <class ProblemShape, class BlockMmad, class BlockScheduler>
using QuantGroupedMatmulMxfp8KernelBscaleNz = QuantGroupedMatmulMxKernelSplitM<ProblemShape, BlockMmad, BlockScheduler>;

} // namespace Kernel
