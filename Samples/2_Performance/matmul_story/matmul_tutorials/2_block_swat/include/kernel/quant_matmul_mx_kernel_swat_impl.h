/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_MATMUL_MX_KERNEL_SWAT_IMPL_H
#define QUANT_MATMUL_MX_KERNEL_SWAT_IMPL_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "../../../common/kernel_utils/common_utils.h"
#include "../../../common/kernel_utils/tuple_utils.h"
#include "include/tensor.h"
#include "../block/block_scheduler_mx_swat.h"
#include "../block/block_mmad_mx_swat.h"

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
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;
    using MakeLayoutA = AscendC::Te::NDLayoutFormat<AType>;
    using MakeLayoutB = AscendC::Te::DNLayoutFormat<BType>;
    using MakeLayoutScaleA = AscendC::Te::ScaleANDLayoutFormat<fp8_e8m0_t>;
    using MakeLayoutScaleB = AscendC::Te::ScaleBDNLayoutFormat<fp8_e8m0_t>;

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
    __aicore__ inline void operator()(const Params& params);

private:
    __aicore__ inline void Process(const Params& params, BlockSchedulerOp& bs);
    __aicore__ inline void ResetGmAddr(const Params& params);
    __aicore__ inline TupleShape ToShapeTuple(const ProblemShape& problemShape)
    {
        return {problemShape.m, problemShape.n, problemShape.k};
    }

private:
    BlockMmad mmadOp_;
    TupleShape problemShape_{};
    __gm__ AType* aGmAddr_{nullptr};
    __gm__ BType* bGmAddr_{nullptr};
    __gm__ CType* cGmAddr_{nullptr};
    __gm__ fp8_e8m0_t* scaleAGmAddr_{nullptr};
    __gm__ fp8_e8m0_t* scaleBGmAddr_{nullptr};
};

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwatImpl<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::operator()(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }

    ResetGmAddr(params);
    BlockSchedulerOp bs(params.problemShape, params.schParams);
    problemShape_ = ToShapeTuple(params.problemShape);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    mmadOp_.Init(problemShape_, l0TileShape, params.l1Params);
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
    auto layoutScaleA = MakeLayoutScaleA{}(
        params.problemShape.m, CeilDiv(params.problemShape.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
    auto layoutB = MakeLayoutB{}(params.problemShape.k, params.problemShape.n);
    auto layoutScaleB = MakeLayoutScaleB{}(
        CeilDiv(params.problemShape.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, params.problemShape.n);
    auto layoutC = AscendC::Te::MakeNDLayout<CType>(params.problemShape.m, params.problemShape.n);

    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(aGmAddr_), layoutA);
    auto gmScaleA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(scaleAGmAddr_), layoutScaleA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(bGmAddr_), layoutB);
    auto gmScaleB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(scaleBGmAddr_), layoutScaleB);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(cGmAddr_), layoutC);
    BlockCoord blockCoord;
    constexpr int64_t kPos = 0L;
    while (bs.GetTileIdx(blockCoord)) {
        int64_t mPos = Get<MNK_M>(blockCoord);
        int64_t nPos = Get<MNK_N>(blockCoord);
        BlockShape singleShape = bs.GetBlockShape(blockCoord);
        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            return;
        }
        auto gmBlockA = gmA(
            AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), params.problemShape.k));
        auto gmBlockScaleA = gmScaleA(
            AscendC::Te::MakeCoord(mPos, kPos),
            AscendC::Te::MakeShape(
                Get<MNK_M>(singleShape), CeilDiv(params.problemShape.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
        auto gmBlockB = gmB(
            AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(params.problemShape.k, Get<MNK_N>(singleShape)));
        auto gmBlockScaleB = gmScaleB(
            AscendC::Te::MakeCoord(kPos, nPos),
            AscendC::Te::MakeShape(
                CeilDiv(params.problemShape.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, Get<MNK_N>(singleShape)));
        auto gmBlockC = gmC(
            AscendC::Te::MakeCoord(mPos, nPos),
            AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));
        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockC, singleShape);
    }
}

__global__ __aicore__ __cube__ void QuantMatmulMxfp4SwatKernel(uint64_t m, uint64_t k, uint64_t n,
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
    constexpr uint32_t L1_BUFFER_NUM = 2;

    Params params;
    params.problemShape.m = static_cast<int64_t>(m);
    params.problemShape.n = static_cast<int64_t>(n);
    params.problemShape.k = static_cast<int64_t>(k);
    params.mmadParams.aGmAddr = aGM;
    params.mmadParams.bGmAddr = bGM;
    params.mmadParams.scaleAGmAddr = aScaleGM;
    params.mmadParams.scaleBGmAddr = bScaleGM;
    params.mmadParams.cGmAddr = cGM;
    params.l1Params.kL1 = BASE_K * L1_BUFFER_NUM;
    params.l1Params.scaleKL1 = BASE_K * L1_BUFFER_NUM;
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
