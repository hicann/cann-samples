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
#include "../../../../common/kernel_utils/common_utils.h"
#include "../../../../common/kernel_utils/layout_utils.h"
#include "../../../../common/kernel_utils/tuple_utils.h"
#include "../block/block_scheduler_mx_swat.h"
#include "../block/block_mmad_mx_swat.h"
#include "../utils/coord_utils.h"

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

    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<ProblemShape, BlockScheduler>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = typename BlockSchedulerOp::BlockCoord;
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t>;  // a, b, scaleA, scaleB, c
    using CoordClass = Coordinate<transA, transB, CubeFormat::ND, CubeFormat::ND, CubeFormat::ND>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

    struct QBMMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
    };

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
    BlockOffset blockOffset_{0, 0, 0, 0, 0};
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    AscendC::GlobalTensor<fp8_e8m0_t> scaleAGlobal_;
    AscendC::GlobalTensor<fp8_e8m0_t> scaleBGlobal_;
};

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwatImpl<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::operator()(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }

    Init(params);
    BlockSchedulerOp bs(params.problemShape, params.schParams);
    problemShape_ = ToShapeTuple(params.problemShape);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    mmadOp_.Init(problemShape_, l0TileShape, params.l1Params);
    Process(params, bs);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwatImpl<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::Init(const Params& params)
{
    aGlobal_.SetGlobalBuffer((__gm__ AType*)params.mmadParams.aGmAddr);
    bGlobal_.SetGlobalBuffer((__gm__ BType*)params.mmadParams.bGmAddr);
    cGlobal_.SetGlobalBuffer((__gm__ CType*)params.mmadParams.cGmAddr);
    scaleAGlobal_.SetGlobalBuffer((__gm__ fp8_e8m0_t*)params.mmadParams.scaleAGmAddr);
    scaleBGlobal_.SetGlobalBuffer((__gm__ fp8_e8m0_t*)params.mmadParams.scaleBGmAddr);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwatImpl<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::Process(
    const Params& params, BlockSchedulerOp& bs)
{
    CoordClass coord(
        params.problemShape.m, params.problemShape.n, params.problemShape.k, params.qbmmParams.baseM,
        params.qbmmParams.baseN, params.qbmmParams.baseK);
    BlockCoord blockIdx;
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape = bs.GetBlockShape(blockIdx);
        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            return;
        }
        blockOffset_ = coord.template GetQuantOffset<false>(
            Get<IDX_M_TILEIDX>(blockIdx), Get<IDX_N_TILEIDX>(blockIdx),
            Get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape),
            Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape));

        mmadOp_(
            aGlobal_[Get<IDX_A_OFFSET>(blockOffset_)],
            bGlobal_[Get<IDX_B_OFFSET>(blockOffset_)],
            scaleAGlobal_[Get<IDX_X1SCALE_OFFSET>(blockOffset_)],
            scaleBGlobal_[Get<IDX_X2SCALE_OFFSET>(blockOffset_)],
            cGlobal_[Get<IDX_C_OFFSET>(blockOffset_)], singleShape);
    }
}

__global__ __aicore__ void QuantMatmulMxfp4SwatKernel(uint64_t m, uint64_t k, uint64_t n,
        GM_ADDR aGM, GM_ADDR bGM, GM_ADDR aScaleGM, GM_ADDR bScaleGM, GM_ADDR cGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    using AType = fp4x2_e2m1_t;
    using BType = fp4x2_e2m1_t;
    using CType = bfloat16_t;

    using BlockScheduler = Block::QuantMatmulMxSwatScheduler;
    using BlockMmad = Block::BlockMmadMx<AType, BType, CType>;
    using ProblemShape = MatmulShape;
    using QuantMatmulKernelImpl = Kernel::QuantMatmulMxKernelSwatImpl<ProblemShape, BlockMmad, BlockScheduler>;
    using Params = typename QuantMatmulKernelImpl::Params;

    constexpr uint32_t BASE_M = 256;
    constexpr uint32_t BASE_N = 256;
    constexpr uint32_t BASE_K = 256; // 128 / sizeof(fp4x2_e2m1_t)
    constexpr uint32_t PINGPONG_NUM = 2;

    Params params;
    params.problemShape.m = static_cast<int64_t>(m);
    params.problemShape.n = static_cast<int64_t>(n);
    params.problemShape.k = static_cast<int64_t>(k);
    params.mmadParams.aGmAddr = aGM;
    params.mmadParams.bGmAddr = bGM;
    params.mmadParams.scaleAGmAddr = aScaleGM;
    params.mmadParams.scaleBGmAddr = bScaleGM;
    params.mmadParams.cGmAddr = cGM;
    params.l1Params.kL1 = BASE_K;
    params.l1Params.scaleKL1 = BASE_K;
    params.l1Params.l1BufNum = PINGPONG_NUM;
    params.schParams.baseM = BASE_M;
    params.schParams.baseN = BASE_N;
    params.qbmmParams.baseM = BASE_M;
    params.qbmmParams.baseN = BASE_N;
    params.qbmmParams.baseK = BASE_K;

    QuantMatmulKernelImpl impl;
    impl(params);
}

} // namespace Kernel

#endif

