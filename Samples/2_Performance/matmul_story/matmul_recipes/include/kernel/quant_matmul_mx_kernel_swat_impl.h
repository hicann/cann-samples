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
    using BiasType = typename BlockMmad::BiasType;
    using LayoutB = typename BlockMmad::LayoutB;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // x1, x2, x1Scale, x2Scale, bias, y
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using CoordClass = Coordinate<transA, transB, CubeFormat::ND, CubeFormat::ND, CubeFormat::ND>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

    // Tile-level runtime knobs.
    //
    // These values describe how one logical block is computed:
    // - baseM/baseN/baseK define the block tile shape
    // - isBias controls whether bias is loaded/accumulated
    // - dbL0C enables ping-pong buffering on the L0C output tile
    struct QBMMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
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
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void operator()(const Params& params);

private:
    __aicore__ inline void Process(const Params& params, BlockSchedulerOp& bs);
    __aicore__ inline TupleShape ToShapeTuple(const ProblemShape& problemShape)
    {
        return {problemShape.m, problemShape.n, problemShape.k};
    }

private:
    BlockMmad mmadOp_;
    TupleShape problemShape_{};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    AscendC::GlobalTensor<fp8_e8m0_t> scaleAGlobal_;
    AscendC::GlobalTensor<fp8_e8m0_t> scaleBGlobal_;
    bool isBias_{false};
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
    Init(params);
    BlockSchedulerOp bs(params.problemShape, params.schParams);
    problemShape_ = ToShapeTuple(params.problemShape);

    // `BlockMmad` expects the block tile shape in [M, N, K] form.
    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    bool enableL0CPingPong = (params.qbmmParams.dbL0C > 1);
    mmadOp_.Init(problemShape_, l0TileShape, params.l1Params, isBias_, enableL0CPingPong);
    Process(params, bs);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwatImpl<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::Init(const Params& params)
{
    // The host has already allocated GM buffers and passed raw addresses here.
    // The kernel-side wrapper converts those addresses into typed GlobalTensor views.
    aGlobal_.SetGlobalBuffer((__gm__ AType*)params.mmadParams.aGmAddr);
    bGlobal_.SetGlobalBuffer((__gm__ BType*)params.mmadParams.bGmAddr);
    cGlobal_.SetGlobalBuffer((__gm__ CType*)params.mmadParams.cGmAddr);
    if (params.qbmmParams.isBias == 1) {
        isBias_ = true;
        biasGlobal_.SetGlobalBuffer((__gm__ BiasType*)params.mmadParams.biasGmAddr);
    }
    scaleAGlobal_.SetGlobalBuffer((__gm__ fp8_e8m0_t*)params.mmadParams.scaleAGmAddr);
    scaleBGlobal_.SetGlobalBuffer((__gm__ fp8_e8m0_t*)params.mmadParams.scaleBGmAddr);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwatImpl<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::Process(
    const Params& params, BlockSchedulerOp& bs)
{
    // Addressing relies on `Coordinate` to translate tile indices + tail split offsets into GM offsets.
    CoordClass coord(
        params.problemShape.m, params.problemShape.n, params.problemShape.k, params.qbmmParams.baseM,
        params.qbmmParams.baseN, params.qbmmParams.baseK);
    BlockCoord blockIdx;
    const int64_t mTailTile = params.schParams.mTailTile;
    const int64_t nTailTile = params.schParams.nTailTile;
    // Tail-round load balance: split the last scheduled tiles into smaller pieces if needed.
    if ((bs.GetEndBlockIdx() + 1) * mTailTile * nTailTile <= AscendC::GetBlockNum()) {
        bs.UpdateTailTile(mTailTile, nTailTile);
    }
    // Each block (hardware core) processes a sequence of tiles.
    while (bs.GetTileIdx(blockIdx)) {
        // Get the current tile shape (with optional tail-split offsets).
        BlockShape singleShape = bs.GetBlockShape(blockIdx);
        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            // If an invalid shape is returned, stop processing for this core.
            // (Keep behavior unchanged; only comment is added/translated.)
            return;
        }
        AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> loadBalanceInfo = bs.GetLoadBalanceInfo();
        // Compute GM offsets for A/B/scales/bias/C based on tile indices and tail strategy.
        blockOffset_ = coord.template GetQuantOffset<true>(
            Get<IDX_M_TILEIDX>(blockIdx), Get<IDX_N_TILEIDX>(blockIdx),
            Get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape),
            Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape), loadBalanceInfo);

        // Execute one logical block:
        // 1. load the tile of A/B/scale/bias from the computed GM offsets
        // 2. iterate over K using the configured L1/L0 staging policy
        // 3. write the result tile back to the corresponding C location
        mmadOp_(
            aGlobal_[Get<IDX_A_OFFSET>(blockOffset_)],
            bGlobal_[Get<IDX_B_OFFSET>(blockOffset_)],
            scaleAGlobal_[Get<IDX_X1SCALE_OFFSET>(blockOffset_)],
            scaleBGlobal_[Get<IDX_X2SCALE_OFFSET>(blockOffset_)],
            biasGlobal_[Get<IDX_BIAS_OFFSET>(blockOffset_)],
            cGlobal_[Get<IDX_C_OFFSET>(blockOffset_)], singleShape);
    }
}

} // namespace Kernel

#endif
