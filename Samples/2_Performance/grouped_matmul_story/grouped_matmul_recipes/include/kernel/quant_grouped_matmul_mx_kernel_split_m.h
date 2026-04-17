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
 * \file quant_grouped_matmul_mx_kernel_split_m.h
 * \brief Sample-side grouped MX split-M kernel wrapper.
 */
#pragma once

#include "kernel_basic_intf.h"
#include "include/tensor.h"
#include "../block/quant_grouped_matmul_mx_block_mmad_split_m.h"
#include "../block/quant_grouped_matmul_mx_block_scheduler_split_m.h"
#include "../block/block_scheduler_utils.h"
#include "../policy/dispatch_policy.h"
#include "../tiling/quant_grouped_matmul_mx_tiling_data.h"
#include "../utils/grouped_matmul_constant.h"

namespace Kernel {

template <class ProblemShape, class BlockMmad, class BlockScheduler>
class KernelQGmmMx {
public:
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using LayoutB = typename BlockMmad::LayoutB;
    static constexpr CubeFormat formatB = ::TagToFormat<LayoutB>::format;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockMmadShape = typename BlockMmad::BlockShape;
    using ScaleType = fp8_e8m0_t;
    using MakeLayoutA = AscendC::Te::NDLayoutFormat<AType>;
    using MakeLayoutB = AscendC::Te::DNLayoutFormat<BType>;
    using MakeLayoutScaleA = AscendC::Te::ScaleANDLayoutFormat<ScaleType>;
    using MakeLayoutScaleB = AscendC::Te::ScaleBDNLayoutFormat<ScaleType>;

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
               GroupedMatmulRecipe::MX_MULTI_SIZE;
    }

    __aicore__ inline void Run(const Params& params)
    {
        ResetGmAddr(params);
        Init(params);
        using SchedulerOp = BlockScheduler;
        SchedulerOp bs(params.schedulerParams);
        for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
            UpdateOffset(groupIdx);
            SetMNK(groupIdx);
            if (AscendC::Std::get<MNK_M>(problemShape_) <= 0 || AscendC::Std::get<MNK_K>(problemShape_) <= 0) {
                continue;
            }
            mmadOp_.UpdateParamsForNextProblem(problemShape_);
            BaseMBalance(bs, AscendC::Std::get<MNK_M>(problemShape_), params.kernelParams.baseM);
            typename SchedulerOp::TupleShape bsProblemShape{
                AscendC::Std::get<MNK_M>(problemShape_), AscendC::Std::get<MNK_N>(problemShape_),
                AscendC::Std::get<MNK_K>(problemShape_), 1L};
            bs.UpdateNextProblem(bsProblemShape);
            if (IsLastGroupAndNeedSplit(bs, groupIdx)) {
                bs.UpdateTailTile();
            }
            ProcessSingleGroup(params, bs);
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        groupListPtr_ = params.groupListGmAddr;
        groupNum_ = params.kernelParams.groupNum;
        curBaseM_ = params.kernelParams.baseM;
        baseOffset_ = {0, 0, 0, 0, 0};
        AscendC::Std::get<MNK_M>(problemShape_) = params.kernelParams.m;
        AscendC::Std::get<MNK_N>(problemShape_) = params.kernelParams.n;
        AscendC::Std::get<MNK_K>(problemShape_) = params.kernelParams.k;
        if (groupListPtr_ != nullptr) {
            groupListGlobal_.SetGlobalBuffer((__gm__ int64_t*)groupListPtr_);
        }
        TupleShape l0Shape{
            static_cast<int64_t>(params.kernelParams.baseM),
            static_cast<int64_t>(params.kernelParams.baseN),
            static_cast<int64_t>(params.kernelParams.baseK)};
        typename BlockMmad::L1Params l1Params{
            static_cast<uint64_t>(params.kernelParams.kAL1),
            static_cast<uint64_t>(params.kernelParams.kBL1),
            static_cast<uint64_t>(params.kernelParams.scaleKAL1)};
        mmadOp_.Init(
            problemShape_,
            l0Shape,
            l1Params,
            params.kernelParams.dbL0C == GroupedMatmulRecipe::DOUBLE_BUFFER);
    }

    template <class SchedulerOp>
    __aicore__ inline void BaseMBalance(SchedulerOp& bs, int64_t m, int64_t baseM)
    {
        int64_t mCnt = CeilDiv(m, baseM);
        curBaseM_ = CeilAlign(CeilDiv(m, mCnt), AscendC::BLOCK_CUBE);
        bs.UpdateBaseM(curBaseM_);
    }

    template <class SchedulerOp>
    __aicore__ inline bool IsLastGroupAndNeedSplit(const SchedulerOp& bs, uint32_t groupIdx) const
    {
        // Consider tail split only when at least half of the cores are still available.
        return groupIdx == groupNum_ - 1 && (bs.GetEndBlockIdx() + 1) <= AscendC::GetBlockNum() / 2;
    }

    __aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx)
    {
        return static_cast<int32_t>(groupListGlobal_.GetValue(groupIdx));
    }

    __aicore__ inline void SetMNK(uint32_t groupIdx)
    {
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx);
        AscendC::Std::get<MNK_M>(problemShape_) = splitValue;
    }

    __aicore__ inline void UpdateOffset(uint32_t groupIdx)
    {
        if (groupIdx == 0) {
            return;
        }
        int64_t m = AscendC::Std::get<MNK_M>(problemShape_);
        int64_t n = AscendC::Std::get<MNK_N>(problemShape_);
        int64_t k = AscendC::Std::get<MNK_K>(problemShape_);
        constexpr bool isFp4Type = AscendC::IsSameType<AType, fp4x2_e2m1_t>::value;
        constexpr uint64_t sizeShift = isFp4Type ? 1UL : 0UL;
        AscendC::Std::get<0>(baseOffset_) += (m * k) >> sizeShift;
        AscendC::Std::get<1>(baseOffset_) += (n * k) >> sizeShift;
        int64_t scaleK =
            CeilDiv(k, static_cast<int64_t>(GroupedMatmulRecipe::MX_DIVISOR_SIZE)) *
            GroupedMatmulRecipe::MX_MULTI_SIZE;
        AscendC::Std::get<2>(baseOffset_) += m * scaleK;
        AscendC::Std::get<3>(baseOffset_) += n * scaleK;
        AscendC::Std::get<4>(baseOffset_) += m * n;
    }

    template <class SchedulerOp>
    __aicore__ inline void ProcessSingleGroup(const Params& params, SchedulerOp& bs)
    {
        int64_t groupM = AscendC::Std::get<MNK_M>(problemShape_);
        int64_t groupN = AscendC::Std::get<MNK_N>(problemShape_);
        int64_t groupK = AscendC::Std::get<MNK_K>(problemShape_);
        int64_t scaleK = GetScaleK(groupK);
        __gm__ AType* groupAPtr = xGmAddr_ + AscendC::Std::get<0>(baseOffset_);
        __gm__ BType* groupBPtr = wGmAddr_ + AscendC::Std::get<1>(baseOffset_);
        __gm__ ScaleType* groupScaleAPtr = x1ScaleGmAddr_ + AscendC::Std::get<2>(baseOffset_);
        __gm__ ScaleType* groupScaleBPtr = x2ScaleGmAddr_ + AscendC::Std::get<3>(baseOffset_);
        __gm__ CType* groupCPtr = yGmAddr_ + AscendC::Std::get<4>(baseOffset_);
        auto layoutA = MakeLayoutA{}(groupM, groupK);
        auto layoutB = MakeLayoutB{}(groupK, groupN);
        auto layoutScaleA = MakeLayoutScaleA{}(groupM, scaleK);
        auto layoutScaleB = MakeLayoutScaleB{}(scaleK, groupN);
        auto groupA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(groupAPtr), layoutA);
        auto groupB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(groupBPtr), layoutB);
        auto groupScaleA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(groupScaleAPtr), layoutScaleA);
        auto groupScaleB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(groupScaleBPtr), layoutScaleB);
        auto groupC = AscendC::Te::MakeTensor(
            AscendC::Te::MakeGMmemPtr(groupCPtr),
            AscendC::Te::MakeNDLayout<CType>(groupM, groupN));
        BlockCoord tileIdx;
        while (bs.GetTileIdx(tileIdx)) {
            BlockShape singleShape = bs.GetBlockShape(tileIdx);
            if (AscendC::Std::get<MNK_M>(singleShape) <= 0 || AscendC::Std::get<MNK_N>(singleShape) <= 0) {
                return;
            }
            int64_t mOffset = AscendC::Std::get<0>(tileIdx) * static_cast<int64_t>(curBaseM_) +
                              AscendC::Std::get<2>(singleShape);
            int64_t nOffset = AscendC::Std::get<1>(tileIdx) * static_cast<int64_t>(params.kernelParams.baseN) +
                              AscendC::Std::get<3>(singleShape);
            int64_t tileM = AscendC::Std::get<MNK_M>(singleShape);
            int64_t tileN = AscendC::Std::get<MNK_N>(singleShape);
            auto gmBlockA = groupA(
                AscendC::Te::MakeCoord(mOffset, 0),
                AscendC::Te::MakeShape(tileM, groupK));
            auto gmBlockB = groupB(
                AscendC::Te::MakeCoord(0, nOffset),
                AscendC::Te::MakeShape(groupK, tileN));
            auto gmBlockScaleA = groupScaleA(
                AscendC::Te::MakeCoord(mOffset, 0),
                AscendC::Te::MakeShape(tileM, scaleK));
            auto gmBlockScaleB = groupScaleB(
                AscendC::Te::MakeCoord(0, nOffset),
                AscendC::Te::MakeShape(scaleK, tileN));
            auto gmBlockC = groupC(
                AscendC::Te::MakeCoord(mOffset, nOffset),
                AscendC::Te::MakeShape(tileM, tileN));
            BlockMmadShape mmadShape{
                tileM, tileN, static_cast<int64_t>(groupK)};
            mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockC, mmadShape);
        }
    }

private:
    BlockMmad mmadOp_;
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0};
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
using QuantGroupedMatmulMxfp8KernelSplitM = QuantGroupedMatmulMxKernelSplitM<ProblemShape, BlockMmad, BlockScheduler>;

} // namespace Kernel

