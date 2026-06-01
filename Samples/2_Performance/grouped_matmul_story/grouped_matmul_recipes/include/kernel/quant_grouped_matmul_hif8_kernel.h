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
 * \file quant_grouped_matmul_hif8_kernel.h
 * \brief Grouped HiFloat8 kernel wrapper.
 */
#pragma once

#include "kernel_basic_intf.h"
#include "include/tensor_api/tensor.h"

#include "../../common/kernel_utils/common_utils.h"
#include "../block/block_scheduler_utils.h"
#include "../block/quant_grouped_matmul_hif8_block_mmad.h"
#include "../block/quant_grouped_matmul_mx_block_scheduler_split_m.h"
#include "../policy/dispatch_policy.h"
#include "../tiling/quant_grouped_matmul_hif8_tiling_data.h"
#include "../utils/grouped_matmul_constant.h"

namespace Kernel {

namespace {
constexpr uint64_t HIF8_DEQ_SCALE_MUL = 0xFFFFE000;
} // namespace

template <class ProblemShape, class BlockMmad, class BlockScheduler>
class KernelQGmmHif8 {
public:
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using X2ScaleType = uint64_t;

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
        uint32_t nBufferNum{0U};
        uint32_t dbL0C{0U};
        uint32_t x1QuantMode{0U};
        uint32_t x2QuantMode{0U};
    };

    struct Params {
        ProblemShape problemShape;
        typename BlockMmad::Params mmParams;
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
    __aicore__ inline void Init(const Params& params)
    {
        groupNum_ = params.kernelParams.groupNum;
        curBaseM_ = params.kernelParams.baseM;
        n_ = params.kernelParams.n;
        k_ = params.kernelParams.k;
        preGroupMSum_ = 0;
        xGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmParams.aGmAddr);
        wGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmParams.bGmAddr);
        yGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmParams.cGmAddr);
        x1ScaleGmAddr_ = reinterpret_cast<__gm__ float*>(params.mmParams.x1ScaleGmAddr);
        x2ScaleFloatGmAddr_ = reinterpret_cast<__gm__ float*>(params.mmParams.x2ScaleGmAddr);
        x2ScaleU64GmAddr_ = reinterpret_cast<__gm__ X2ScaleType*>(params.mmParams.x2ScaleGmAddr);
        groupListGlobal_.SetGlobalBuffer((__gm__ int64_t*)params.groupListGmAddr);
        x1ScaleGlobal_.SetGlobalBuffer(x1ScaleGmAddr_);
        x2ScaleFloatGlobal_.SetGlobalBuffer(x2ScaleFloatGmAddr_);
    }

    template <class SchedulerOp>
    __aicore__ inline void BaseMBalance(SchedulerOp& bs, int64_t m, int64_t baseM)
    {
        int64_t mCnt = CeilDiv(m, baseM);
        curBaseM_ = CeilAlign(CeilDiv(m, mCnt), AscendC::BLOCK_CUBE);
        bs.UpdateBaseM(curBaseM_);
    }

    __aicore__ inline bool IfNeedSplit(const BlockScheduler& bs) const
    {
        return (bs.GetEndBlockIdx() + 1) <= (AscendC::GetBlockNum() >> 1);
    }

    __aicore__ inline uint64_t GetTensorOffset(uint32_t groupIdx, int64_t groupM) const
    {
        (void)groupIdx;
        return preGroupMSum_ * static_cast<int64_t>(k_);
    }

    __aicore__ inline uint64_t BuildScalarScale(uint32_t groupIdx)
    {
        float deqScale = x1ScaleGlobal_.GetValue(groupIdx) * x2ScaleFloatGlobal_.GetValue(groupIdx);
        uint32_t uint32Scale = *(reinterpret_cast<uint32_t*>(&deqScale));
        return static_cast<uint64_t>(uint32Scale & HIF8_DEQ_SCALE_MUL);
    }

    template <class SchedulerOp>
    __aicore__ inline void ProcessSingleGroup(
        const Params& params, SchedulerOp& bs, uint32_t groupIdx, int64_t groupM)
    {
        BlockCoord tileIdx;
        if (!bs.GetTileIdx(tileIdx)) {
            return;
        }
        int64_t groupN = static_cast<int64_t>(n_);
        int64_t groupK = static_cast<int64_t>(k_);
        TupleShape problemShape{groupM, groupN, groupK};
        BlockShape l0Shape{
            static_cast<int64_t>(curBaseM_), static_cast<int64_t>(params.kernelParams.baseN),
            static_cast<int64_t>(params.kernelParams.baseK), 0};
        bool enableL0CPingPong = params.kernelParams.dbL0C > 1U;
        mmOp_.Init(
            problemShape, l0Shape, params.kernelParams.kAL1, params.kernelParams.kBL1,
            params.kernelParams.nBufferNum, static_cast<GroupedMatmulRecipe::QuantMode>(params.kernelParams.x2QuantMode),
            enableL0CPingPong);

        __gm__ AType* groupAPtr = xGmAddr_ + GetTensorOffset(groupIdx, groupM);
        __gm__ BType* groupBPtr =
            wGmAddr_ + static_cast<int64_t>(groupIdx) * static_cast<int64_t>(n_) * static_cast<int64_t>(k_);
        __gm__ CType* groupCPtr = yGmAddr_ + preGroupMSum_ * static_cast<int64_t>(n_);
        auto groupA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(groupAPtr), LayoutA{}(groupM, groupK));
        auto groupB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(groupBPtr), LayoutB{}(groupK, groupN));
        auto groupC = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(groupCPtr), LayoutC{}(groupM, groupN));

        bool isPerChannel = static_cast<GroupedMatmulRecipe::QuantMode>(params.kernelParams.x2QuantMode) ==
                            GroupedMatmulRecipe::QuantMode::PERCHANNEL_MODE;
        do {
            BlockShape singleShape = bs.GetBlockShape(tileIdx);
            if (AscendC::Std::get<MNK_M>(singleShape) <= 0 || AscendC::Std::get<MNK_N>(singleShape) <= 0) {
                return;
            }
            int64_t mOffset = AscendC::Std::get<MNK_M>(tileIdx) * static_cast<int64_t>(curBaseM_) +
                              AscendC::Std::get<MNK_K>(singleShape);
            int64_t nOffset = AscendC::Std::get<MNK_N>(tileIdx) * static_cast<int64_t>(params.kernelParams.baseN) +
                              AscendC::Std::get<MNK_B>(singleShape);
            int64_t tileM = AscendC::Std::get<MNK_M>(singleShape);
            int64_t tileN = AscendC::Std::get<MNK_N>(singleShape);
            auto gmBlockA = groupA.Slice(
                AscendC::Te::MakeCoord(mOffset, 0), AscendC::Te::MakeShape(tileM, groupK));
            auto gmBlockB = groupB.Slice(
                AscendC::Te::MakeCoord(0, nOffset), AscendC::Te::MakeShape(groupK, tileN));
            auto gmBlockC = groupC.Slice(
                AscendC::Te::MakeCoord(mOffset, nOffset), AscendC::Te::MakeShape(tileM, tileN));
            if (isPerChannel) {
                auto layoutX2 = AscendC::Te::MakeFrameLayout<
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<X2ScaleType>>(1, groupN);
                auto gmX2Full = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                        x2ScaleU64GmAddr_ + static_cast<int64_t>(groupIdx) * static_cast<int64_t>(n_)),
                    layoutX2);
                auto gmBlockX2 = gmX2Full.Slice(
                    AscendC::Te::MakeCoord(0, nOffset), AscendC::Te::MakeShape(1, tileN));
                mmOp_(gmBlockA, gmBlockB, gmBlockX2, gmBlockC, singleShape);
            } else {
                mmOp_(gmBlockA, gmBlockB, BuildScalarScale(groupIdx), gmBlockC, singleShape);
            }
        } while (bs.GetTileIdx(tileIdx));
    }

    __aicore__ inline void Run(const Params& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        Init(params);
        BlockScheduler bs(params.schedulerParams);
        bs.SetTailAlign(AscendC::BLOCK_CUBE, AscendC::BLOCK_CUBE);
        for (uint32_t groupIdx = 0; groupIdx < groupNum_ - 1; ++groupIdx) {
            int64_t groupM = static_cast<int64_t>(groupListGlobal_.GetValue(groupIdx));
            if (groupM <= 0) {
                continue;
            }
            BaseMBalance(bs, groupM, params.kernelParams.baseM);
            typename BlockScheduler::TupleShape bsProblemShape{groupM, static_cast<int64_t>(n_),
                static_cast<int64_t>(k_)};
            bs.UpdateNextProblem(bsProblemShape);
            ProcessSingleGroup(params, bs, groupIdx, groupM);
            preGroupMSum_ += groupM;
        }
        // Process the last group (groupNum_ must be greater than 0)
        uint32_t groupIdx = groupNum_ - 1;
        int64_t groupM = static_cast<int64_t>(groupListGlobal_.GetValue(groupIdx));
        if (groupM > 0) {
            BaseMBalance(bs, groupM, params.kernelParams.baseM);
            typename BlockScheduler::TupleShape bsProblemShape{groupM, static_cast<int64_t>(n_),
                static_cast<int64_t>(k_)};
            bs.UpdateNextProblem(bsProblemShape);
            if (IfNeedSplit(bs)) {
                bs.UpdateTailTile();
                ProcessSingleGroup(params, bs, groupIdx, groupM);
            } else {
                ProcessSingleGroup(params, bs, groupIdx, groupM);
            }
            preGroupMSum_ += groupM;
        }
    }

private:
    BlockMmad mmOp_;
    AscendC::GlobalTensor<int64_t> groupListGlobal_;
    AscendC::GlobalTensor<float> x1ScaleGlobal_;
    AscendC::GlobalTensor<float> x2ScaleFloatGlobal_;
    __gm__ AType* xGmAddr_{nullptr};
    __gm__ BType* wGmAddr_{nullptr};
    __gm__ CType* yGmAddr_{nullptr};
    __gm__ float* x1ScaleGmAddr_{nullptr};
    __gm__ float* x2ScaleFloatGmAddr_{nullptr};
    __gm__ X2ScaleType* x2ScaleU64GmAddr_{nullptr};
    uint32_t groupNum_{0U};
    uint32_t curBaseM_{0U};
    uint32_t n_{0U};
    uint32_t k_{0U};
    int64_t preGroupMSum_{0};
};

template <class ProblemShape, class BlockMmad, class BlockScheduler>
using QuantGroupedMatmulHif8KernelSplitM = KernelQGmmHif8<ProblemShape, BlockMmad, BlockScheduler>;

} // namespace Kernel
