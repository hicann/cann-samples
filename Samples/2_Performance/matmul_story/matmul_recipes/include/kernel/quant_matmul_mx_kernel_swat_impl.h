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
 * \file quant_matmul_mx_kernel_swat_impl.h
 * \brief
 */

#ifndef QUANT_MATMUL_MX_KERNEL_SWAT_IMPL_H
#define QUANT_MATMUL_MX_KERNEL_SWAT_IMPL_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "kernel_utils/common_utils.h"
#include "kernel_utils/layout_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "include/tensor.h"
#include "../block/block_scheduler_mx.h"
#include "../block/block_mmad_mx.h"
#include "../utils/coord_utils.h"
#include "../utils/quant_matmul_constant.h"

namespace Kernel {
#define QBMM_MX_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockScheduler>
#define QBMM_MX_KERNEL_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler

using namespace AscendC;

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
class QuantMatmulMxKernelSwatImpl {
public:
    __aicore__ inline QuantMatmulMxKernelSwatImpl()
    {}
    __aicore__ inline ~QuantMatmulMxKernelSwatImpl()
    {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<
        ProblemShape, typename BlockMmad::L1TileShape, typename BlockMmad::L0TileShape, BlockScheduler, transA,
        transB>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using LayoutB = typename BlockMmad::LayoutB;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // x1, x2, x1Scale, x2Scale, y
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

    using MakeLayoutA =
        AscendC::Std::conditional_t<transA, AscendC::Te::DNLayoutFormat<AType>, AscendC::Te::NDLayoutFormat<AType>>;
    using MakeLayoutB =
        AscendC::Std::conditional_t<transB, AscendC::Te::DNLayoutFormat<BType>, AscendC::Te::NDLayoutFormat<BType>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        transA, AscendC::Te::ScaleADNLayoutFormat<fp8_e8m0_t>, AscendC::Te::ScaleANDLayoutFormat<fp8_e8m0_t>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        transB, AscendC::Te::ScaleBDNLayoutFormat<fp8_e8m0_t>, AscendC::Te::ScaleBNDLayoutFormat<fp8_e8m0_t>>;

    // Tile-level runtime knobs.
    //
    // These values describe how one logical block is computed:
    // - baseM/baseN/baseK define the block tile shape
    // - dbL0C enables ping-pong buffering on the L0C output tile
    struct QBMMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t dbL0C;
    };

    // Aggregate kernel parameters passed from host code.
    //
    // Keeping these grouped makes the kernel launch site compact while still
    // exposing the same conceptual layers used inside the implementation.
    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
    };

public:
    __aicore__ inline void operator()(const Params& params);

private:
    __aicore__ inline void ResetGmAddr(const Params& params);
    __aicore__ inline void Process(const Params& params, BlockSchedulerOp& bs);
    __aicore__ inline TupleShape ToShapeTuple(const ProblemShape& problemShape)
    {
        return {problemShape.m, problemShape.n, problemShape.k};
    }

private:
    BlockMmad mmadOp_;
    TupleShape problemShape_{};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};

    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ fp8_e8m0_t* scaleAGmAddr_;
    __gm__ fp8_e8m0_t* scaleBGmAddr_;

};

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwatImpl<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::operator()(const Params& params)
{
    // This path is implemented for AIC only. The example launches no AIV work.
    if ASCEND_IS_AIV {
        return;
    }

    // Bind GM tensors, construct the scheduler, initialize the MMAD pipeline,
    // then let `Process()` iterate over tiles assigned to the current hardware block.
    ResetGmAddr(params);
    BlockSchedulerOp bs(params.problemShape, params.schParams);
    problemShape_ = ToShapeTuple(params.problemShape);

    // `BlockMmad` expects the block tile shape in [M, N, K] form.
    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    bool enableL0CPingPong = (params.qbmmParams.dbL0C > 1);
    mmadOp_.Init(problemShape_, l0TileShape, params.l1Params, enableL0CPingPong);
    Process(params, bs);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwatImpl<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::ResetGmAddr(const Params& params)
{
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwatImpl<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::Process(
    const Params& params, BlockSchedulerOp& bs)
{
    auto layoutA = MakeLayoutA{}(params.problemShape.m, params.problemShape.k);
    auto layoutScaleA =
        MakeLayoutScaleA{}(params.problemShape.m, CeilDiv(params.problemShape.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
    auto layoutB = MakeLayoutB{}(params.problemShape.k, params.problemShape.n);
    auto layoutScaleB =
        MakeLayoutScaleB{}(CeilDiv(params.problemShape.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, params.problemShape.n);
    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(aGmAddr_), layoutA);
    auto gmScaleA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(scaleAGmAddr_), layoutScaleA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(bGmAddr_), layoutB);
    auto gmScaleB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(scaleBGmAddr_), layoutScaleB);
    auto layoutC = AscendC::Te::MakeNDLayout<CType>(params.problemShape.m, params.problemShape.n);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(cGmAddr_), layoutC);

    BlockCoord blockIdx;
    bs.UpdateTailTile(params.schParams.mTailTile, params.schParams.nTailTile);
    int64_t mPos = 0L;
    int64_t nPos = 0L;
    constexpr int64_t kPos = 0L;  // Do not slice K, so the coordinate is 0.
    // Each core processes the block sequentially.
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape = bs.GetBlockShape(blockIdx);
        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            // If an invalid shape is returned, stop processing for this core.
            // (Keep behavior unchanged; only comment is added/translated.)
            return;
        }

        bs.GetTileCoord(blockIdx, mPos, nPos);
        auto gmBlockA = gmA(
            AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), params.problemShape.k));
        auto gmBlockScaleA = gmScaleA(
            AscendC::Te::MakeCoord(mPos, kPos),
            AscendC::Te::MakeShape(Get<MNK_M>(singleShape), CeilDiv(params.problemShape.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
        auto gmBlockB = gmB(
            AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(params.problemShape.k, Get<MNK_N>(singleShape)));
        auto gmBlockScaleB = gmScaleB(
            AscendC::Te::MakeCoord(kPos, nPos),
            AscendC::Te::MakeShape(CeilDiv(params.problemShape.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, Get<MNK_N>(singleShape)));
        auto gmBlockC =
            gmC(AscendC::Te::MakeCoord(mPos, nPos),
                AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));
        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockC, singleShape);
    }
}

} // namespace Kernel

#endif
