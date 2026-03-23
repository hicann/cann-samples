/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_SCHEDULER_MX_BASE_H
#define BLOCK_SCHEDULER_MX_BASE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../../../common/kernel_utils/common_utils.h"
#include "../../../../common/kernel_utils/tuple_utils.h"
#include "block_scheduler_utils.h"

namespace Block {

using namespace AscendC;

struct QuantMatmulMxBaseScheduler {};

/*!
 * \brief Column-major linear order of M–N plane tiles: tileIdx = nTileIdx * mCnt + mTileIdx.
 */
template <class ProblemShape_>
class BlockSchedulerColumnMajorMx {
public:
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t mCnt_{0};   // CeilDiv(m_, baseM_)
    int64_t nCnt_{0};   // CeilDiv(n_, baseN_)
    int64_t totalCnt_{0};
    int64_t tailM_{0};
    int64_t tailN_{0};
    int64_t blockIdx_{0};
    int64_t blockNum_{0};
    int64_t roundIdx_{0};
    int64_t round_{0};
    int64_t endBlockIdx_{0};

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        int64_t baseM;
        int64_t baseN;
    };

public:
    __aicore__ inline BlockSchedulerColumnMajorMx(const ProblemShape& shape, const Params& params)
    {
        m_ = shape.m;
        n_ = shape.n;
        k_ = shape.k;
        baseM_ = static_cast<int64_t>(params.baseM);
        baseN_ = static_cast<int64_t>(params.baseN);
        mCnt_ = CeilDiv(m_, baseM_);
        nCnt_ = CeilDiv(n_, baseN_);
        totalCnt_ = mCnt_ * nCnt_;
        tailM_ = m_ - (mCnt_ - 1) * baseM_;
        tailN_ = n_ - (nCnt_ - 1) * baseN_;

        blockIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
        blockNum_ = static_cast<int64_t>(AscendC::GetBlockNum());
        round_ = totalCnt_ > 0 ? CeilDiv(totalCnt_, blockNum_) : 0;
        endBlockIdx_ = totalCnt_ > 0 ? (totalCnt_ - 1) % blockNum_ : -1;
    }

    __aicore__ inline int64_t GetEndBlockIdx() const { return endBlockIdx_; }

    __aicore__ inline BlockShape GetBlockShape(BlockCoord blockCoord)
    {
        int64_t mTileIdx = Get<MNK_M>(blockCoord);
        int64_t nTileIdx = Get<MNK_N>(blockCoord);
        int64_t singleCoreM = (mTileIdx == mCnt_ - 1) ? tailM_ : baseM_;
        int64_t singleCoreN = (nTileIdx == nCnt_ - 1) ? tailN_ : baseN_;
        return {singleCoreM, singleCoreN, 0, 0};
    }

    __aicore__ inline bool GetTileIdx(BlockCoord& blockCoord)
    {
        if (roundIdx_ >= round_) {
            return false;
        }
        int64_t tileIdx = blockIdx_ + roundIdx_ * blockNum_;
        if (tileIdx >= totalCnt_) {
            return false;
        }
        // column-major: tileIdx = nTileIdx * mCnt_ + mTileIdx
        Get<MNK_M>(blockCoord) = tileIdx % mCnt_;
        Get<MNK_N>(blockCoord) = tileIdx / mCnt_;
        roundIdx_++;
        return true;
    }
};

template <class ProblemShape_>
struct BlockSchedulerSelector<ProblemShape_, QuantMatmulMxBaseScheduler> {
    using SchedulerOp = BlockSchedulerColumnMajorMx<ProblemShape_>;
};

}  // namespace Block
#endif
