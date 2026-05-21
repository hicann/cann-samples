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
 * \file quant_matmul_hifp8_kernel_swat.h
 * \brief HiFP8 quantized batch matmul cube kernel (SWAT tiling, non–full-load).
 */

#pragma once
#include <cstring>
#include "kernel_operator_intf.h"
#include "../block/quant_matmul_hifp8_block_scheduler_swat.h"
#include "include/tensor_api/tensor.h"
#include "kernel_utils/common_utils.h"
#include "../utils/constant.h"

namespace Kernel {
#define QBMM_CUBE_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_CUBE_KERNEL_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler

using namespace AscendC;

namespace {
constexpr uint64_t DEQ_SCALE_MUL = 0xFFFFE000;
} // namespace

QBMM_CUBE_KERNEL_CLASS_TEM_PARAMS
class QuantMmBatchCube {
public:
    __aicore__ inline QuantMmBatchCube()
    {}
    __aicore__ inline ~QuantMmBatchCube()
    {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockMmadParams = typename BlockMmad::Params;
    using AType = typename BlockMmad::AType;
    using BlockSchedulerOp =
        Block::BlockSchedulerQuantHifp8Swat<ProblemShape, transA, transB>;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using X2ScaleType = uint64_t;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

    using MakeLayoutA = typename BlockMmad::LayoutA;
    using MakeLayoutB = typename BlockMmad::LayoutB;
    using MakeLayoutC = typename BlockMmad::LayoutC;

    struct QBMMTiling {
        uint32_t x1QuantMode;
        uint32_t x2QuantMode;
        uint32_t kAL1;
        uint32_t kBL1;
        uint32_t nBufferNum;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t dbL0C;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
    };

public:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Process(const Params& params);
    __aicore__ inline void operator()(const Params& params)
    {
        Process(params);
    }

private:
    __aicore__ inline void ProcessSingleBatch(const Params& params, BlockSchedulerOp& bs);
    __aicore__ inline TupleShape ToShapeTuple(const ProblemShape& problemShape)
    {
        return {problemShape.m, problemShape.n, problemShape.k};
    }

private:
    BlockMmad mmadOp_;
    TupleShape problemShape_{};
    __gm__ AType* aGmBase_{nullptr};
    __gm__ BType* bGmBase_{nullptr};
    __gm__ CType* cGmBase_{nullptr};
    __gm__ X2ScaleType* x2ScaleGmBase_{nullptr};
    uint64_t scaleScalar_{0};
    bool needUpdateTail_{false};
};

QBMM_CUBE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchCube<QBMM_CUBE_KERNEL_FUN_TEM_PARAMS>::Process(const Params& params)
{
    Init(params);
    BlockSchedulerOp bs(params.problemShape, params.schParams);
    problemShape_ = ToShapeTuple(params.problemShape);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    bool enableL0CPingPong = (params.qbmmParams.dbL0C > 1);
    mmadOp_.Init(
        problemShape_, l0TileShape, params.qbmmParams.kAL1, params.qbmmParams.kBL1, params.qbmmParams.nBufferNum,
        static_cast<QuantBatchMatmul::QuantMode>(params.qbmmParams.x2QuantMode), enableL0CPingPong);
    ProcessSingleBatch(params, bs);
}

QBMM_CUBE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchCube<QBMM_CUBE_KERNEL_FUN_TEM_PARAMS>::Init(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    aGmBase_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmBase_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmBase_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    if (static_cast<QuantBatchMatmul::QuantMode>(params.qbmmParams.x2QuantMode) ==
        QuantBatchMatmul::QuantMode::PERCHANNEL_MODE) {
        x2ScaleGmBase_ = reinterpret_cast<__gm__ X2ScaleType*>(params.mmadParams.x2ScaleGmAddr);
    } else if (
        static_cast<QuantBatchMatmul::QuantMode>(params.qbmmParams.x1QuantMode) ==
        QuantBatchMatmul::QuantMode::PERTENSOR_MODE) {
        auto x1Scale = AscendC::GlobalTensor<float>();
        auto x2Scale = AscendC::GlobalTensor<float>();
        x1Scale.SetGlobalBuffer((__gm__ float*)params.mmadParams.x1ScaleGmAddr);
        x2Scale.SetGlobalBuffer((__gm__ float*)params.mmadParams.x2ScaleGmAddr);
        float deqScale = x1Scale.GetValue(0) * x2Scale.GetValue(0);
        uint32_t uint32Scale = *(reinterpret_cast<uint32_t*>(&deqScale));
        scaleScalar_ = static_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
    }
}

QBMM_CUBE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchCube<QBMM_CUBE_KERNEL_FUN_TEM_PARAMS>::ProcessSingleBatch(
    const Params& params, BlockSchedulerOp& bs)
{
    const int64_t m = params.problemShape.m;
    const int64_t n = params.problemShape.n;
    const int64_t k = params.problemShape.k;
    constexpr int64_t kPos = 0;

    __gm__ AType* aBatch = aGmBase_;
    __gm__ BType* bBatch = bGmBase_;
    __gm__ CType* cBatch = cGmBase_;

    BlockCoord blockIdx;
    if (needUpdateTail_ || (bs.GetEndBlockIdx() + 1) * params.schParams.mTailTile * params.schParams.nTailTile <=
                                AscendC::GetBlockNum()) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(params.schParams.mTailTile, params.schParams.nTailTile);
    }

    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutC = MakeLayoutC{}(m, n);

    const bool isPerChannel = static_cast<QuantBatchMatmul::QuantMode>(params.qbmmParams.x2QuantMode) == QuantBatchMatmul::QuantMode::PERCHANNEL_MODE;

    auto gmAFull = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(reinterpret_cast<__gm__ AType*>(aBatch)), layoutA);
    auto gmBFull = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(reinterpret_cast<__gm__ BType*>(bBatch)), layoutB);
    auto gmCFull = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(reinterpret_cast<__gm__ CType*>(cBatch)), layoutC);

    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape = bs.GetBlockShape(blockIdx);
        if (AscendC::Std::get<MNK_M>(singleShape) <= 0 || AscendC::Std::get<MNK_N>(singleShape) <= 0) {
            return;
        }

        const int64_t mPos = AscendC::Std::get<MNK_M>(blockIdx);
        const int64_t nPos = AscendC::Std::get<MNK_N>(blockIdx);
        const int64_t curM = AscendC::Std::get<MNK_M>(singleShape);
        const int64_t curN = AscendC::Std::get<MNK_N>(singleShape);
        
        auto gmBlockA = 
            gmAFull.Slice(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(curM, k));
        auto gmBlockB = 
            gmBFull.Slice(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(k, curN));
        auto gmBlockC = 
            gmCFull.Slice(AscendC::Te::MakeCoord(mPos, nPos), AscendC::Te::MakeShape(curM, curN));

        if (isPerChannel) {
            auto layoutX2 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<X2ScaleType>>(1, n);
            auto gmX2Full = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(reinterpret_cast<__gm__ X2ScaleType*>(x2ScaleGmBase_)),
                layoutX2);
            auto gmBlockX2 = gmX2Full.Slice(AscendC::Te::MakeCoord(0, nPos), AscendC::Te::MakeShape(1, curN));
            mmadOp_(gmBlockA, gmBlockB, gmBlockX2, gmBlockC, singleShape);
        } else {
            mmadOp_(gmBlockA, gmBlockB, scaleScalar_, gmBlockC, singleShape);
        }
    }
}

} // namespace Kernel
