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
 * \file matmul_a16w16_kernel_swat.h
 * \brief Kernel-side SWAT A16W16 implementation for the non-full-load path.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "kernel_utils/common_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "include/tensor.h"

#include "../block/matmul_a16w16_block_mmad_swat.h"
#include "../block/matmul_a16w16_block_scheduler_swat.h"
#include "../utils/matmul_a16w16_constant.h"

namespace Kernel {

template <class ProblemShape, class BlockMmad, class BlockScheduler>
class MatmulA16W16KernelSwat {
public:
    __aicore__ inline MatmulA16W16KernelSwat()
    {}
    __aicore__ inline ~MatmulA16W16KernelSwat()
    {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, BlockScheduler, transA, transB>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using TypeA = typename BlockMmad::TypeA;
    using TypeB = typename BlockMmad::TypeB;
    using TypeC = typename BlockMmad::TypeC;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

    using MakeLayoutA = typename BlockMmad::LayoutA;
    using MakeLayoutB = typename BlockMmad::LayoutB;
    using MakeLayoutC = typename BlockMmad::LayoutC;

    struct KernelParams {
        uint32_t mL1;
        uint32_t nL1;
        uint32_t kL1;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint8_t dbL0C;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockSchedulerParams schParams;
        KernelParams kernelParams;
    };

public:
    __aicore__ inline void operator()(const Params& params);

private:
    __aicore__ inline TupleShape ToShapeTuple(const ProblemShape& problemShape)
    {
        return {problemShape.m, problemShape.n, problemShape.k};
    }

private:
    TupleShape problemShape_{};
    BlockMmadParams blockMmadParams_{};
};

template <class ProblemShape, class BlockMmad, class BlockScheduler>
__aicore__ inline void MatmulA16W16KernelSwat<ProblemShape, BlockMmad, BlockScheduler>::operator()(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    int64_t curBlockIdx = AscendC::GetBlockIdx();
    int64_t blockNum = AscendC::GetBlockNum();
    problemShape_ = ToShapeTuple(params.problemShape);
    blockMmadParams_ = params.mmadParams;
    BlockSchedulerOp bs(params.problemShape, curBlockIdx, blockNum, params.schParams);
    int64_t tileNum = bs.GetTileNum();
    TupleShape tileL1 = {params.kernelParams.mL1, params.kernelParams.nL1, params.kernelParams.kL1};
    TupleShape tileL0 = {params.kernelParams.baseM, params.kernelParams.baseN, params.kernelParams.baseK};
    int64_t realBlockNum = bs.GetBlockNum(params.problemShape, blockNum);
    if (curBlockIdx >= realBlockNum) {
        return;
    }
    AscendC::SetMMLayoutTransform(true);
    bool l0cDB = params.kernelParams.dbL0C > 1;
    // Instantiate mmadOp
    BlockMmad blockMmadOp(problemShape_, tileL1, tileL0, l0cDB);
    
    int64_t m = Get<MNK_M>(problemShape_);
    int64_t n = Get<MNK_N>(problemShape_);
    int64_t k = Get<MNK_K>(problemShape_);

    auto layoutA = MakeLayoutA{}(m, k); // ND layout for A
    auto layoutB = MakeLayoutB{}(k, n); // ND layout for B
    auto layoutC = MakeLayoutC{}(m, n); // ND layout for C

    // A,B,C Gm Tensor
    auto gmA =
        MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ TypeA*>(blockMmadParams_.aGmAddr)), layoutA);
    auto gmB =
        MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ TypeB*>(blockMmadParams_.bGmAddr)), layoutB);
    auto gmC =
        MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ TypeC*>(blockMmadParams_.cGmAddr)), layoutC);

    // Process tiles in ping-pong mode
    for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
        auto tileShape = bs.GetBlockShape(tileIdx); // (m, n, k, b)
        auto tileCoord = bs.GetBlockCoord(tileIdx); // (m, n, k, b)
        auto gmBlockA =
            gmA(AscendC::MakeCoord(Get<0>(tileCoord), 0), AscendC::MakeShape(Get<0>(tileShape), Get<2>(tileShape)));
        auto gmBlockB =
            gmB(AscendC::MakeCoord(0, Get<1>(tileCoord)), AscendC::MakeShape(Get<2>(tileShape), Get<1>(tileShape)));
        auto gmBlockC =
            gmC(AscendC::MakeCoord(Get<0>(tileCoord), Get<1>(tileCoord)),
                AscendC::MakeShape(Get<0>(tileShape), Get<1>(tileShape)));
        blockMmadOp(gmBlockC, gmBlockA, gmBlockB, tileShape);
    }

    AscendC::SetMMLayoutTransform(false);
}

} // namespace Kernel

