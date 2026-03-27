/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Last-round tile balance scheduler (SWAT)
 * - Group by WINDOW_LEN M-rows, column-major, alternating N direction for S-shaped traversal
 * - In the last round, split tail blocks into mTailTile x nTailTile pieces for load balance
 */

#ifndef BLOCK_SCHEDULER_MX_BASE_H
#define BLOCK_SCHEDULER_MX_BASE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_utils/common_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "block_scheduler_utils.h"

namespace Block {

using namespace AscendC;

struct QuantMatmulMxBaseScheduler {};

template <class ProblemShape_>
class BlockSchedulerLastRoundTileBalance {
public:
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t mCnt_{0};
    int64_t nCnt_{0};
    int64_t totalCnt_{0};
    int64_t tailM_{0};
    int64_t tailN_{0};
    int64_t blockIdx_{0};
    int64_t blockNum_{0};
    int64_t roundIdx_{0};
    int64_t round_{0};
    int64_t endBlockIdx_{0};
    int64_t mCoreNum_{0};
    int64_t mTailCoreNum_{0};
    int64_t mainRow_{0};
    int64_t mTailTile_{1};
    int64_t nTailTile_{1};
    int64_t totalTailTile_{1};
    int64_t mSplitAddrOffset_{0};
    int64_t nSplitAddrOffset_{0};

    static constexpr int64_t WINDOW_LEN = 4;

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        int64_t baseM;
        int64_t baseN;
        uint32_t mTailTile{1};
        uint32_t nTailTile{1};
    };

public:
    __aicore__ inline BlockSchedulerLastRoundTileBalance(const ProblemShape& shape, const Params& params)
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
        if (blockIdx_ > endBlockIdx_) {
            round_ -= 1;
        }

        mCoreNum_ = (mCnt_ < WINDOW_LEN) ? mCnt_ : WINDOW_LEN;
        mainRow_ = (mCoreNum_ > 0) ? (mCnt_ / mCoreNum_ - 1) : 0;
        mTailCoreNum_ = (mCoreNum_ > 0) ? (mCnt_ - mCoreNum_ * mainRow_) : 0;

        mTailTile_ = static_cast<int64_t>(params.mTailTile);
        nTailTile_ = static_cast<int64_t>(params.nTailTile);
        if (totalCnt_ % blockNum_ != 0 && (totalCnt_ % blockNum_) * mTailTile_ * nTailTile_ <= blockNum_)
            UpdateTailTile(static_cast<uint32_t>(mTailTile_), static_cast<uint32_t>(nTailTile_));
    }

    __aicore__ inline void UpdateTailTile(uint32_t mTailTile, uint32_t nTailTile)
    {
        mTailTile_ = static_cast<int64_t>(mTailTile);
        nTailTile_ = static_cast<int64_t>(nTailTile);
        totalTailTile_ = mTailTile_ * nTailTile_;

        int64_t tailTileCnt = endBlockIdx_ + 1;
        int64_t newEndBlockIdx = tailTileCnt * totalTailTile_ - 1;

        if (blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx) {
            round_ += 1;
        }
        if (blockIdx_ > newEndBlockIdx) {
            mTailTile_ = 1;
            nTailTile_ = 1;
            totalTailTile_ = 1;
        }
        endBlockIdx_ = newEndBlockIdx;
    }

    __aicore__ inline int64_t GetEndBlockIdx() const { return endBlockIdx_; }

    __aicore__ inline BlockShape GetBlockShape(BlockCoord blockCoord)
    {
        int64_t mTileIdx = Get<MNK_M>(blockCoord);
        int64_t nTileIdx = Get<MNK_N>(blockCoord);
        int64_t singleCoreM = (mTileIdx == mCnt_ - 1) ? tailM_ : baseM_;
        int64_t singleCoreN = (nTileIdx == nCnt_ - 1) ? tailN_ : baseN_;

        if (totalTailTile_ == 1 || roundIdx_ < round_) {
            return {singleCoreM, singleCoreN, 0, 0};
        }
        int64_t singleCoreMSplit = CeilDiv(singleCoreM, mTailTile_);
        int64_t singleCoreNSplit = CeilDiv(singleCoreN, nTailTile_);
        int64_t mSplitIdx = (blockIdx_ % totalTailTile_) % mTailTile_;
        int64_t nSplitIdx = (blockIdx_ % totalTailTile_) / mTailTile_;
        mSplitAddrOffset_ = mSplitIdx * singleCoreMSplit;
        nSplitAddrOffset_ = nSplitIdx * singleCoreNSplit;
        if (mSplitAddrOffset_ >= singleCoreM || nSplitAddrOffset_ >= singleCoreN) {
            return {0, 0, 0, 0};
        }
        singleCoreM = (singleCoreM - mSplitAddrOffset_ < singleCoreMSplit)
                          ? (singleCoreM - mSplitAddrOffset_)
                          : singleCoreMSplit;
        singleCoreN = (singleCoreN - nSplitAddrOffset_ < singleCoreNSplit)
                          ? (singleCoreN - nSplitAddrOffset_)
                          : singleCoreNSplit;
        return {singleCoreM, singleCoreN, mSplitAddrOffset_, nSplitAddrOffset_};
    }

    __aicore__ inline bool GetTileIdx(BlockCoord& blockCoord)
    {
        if (roundIdx_ >= round_) {
            return false;
        }
        bool isLastRound = (roundIdx_ == round_ - 1);
        int64_t logicalBlockIdx = isLastRound ? (blockIdx_ / totalTailTile_) : blockIdx_;
        int64_t tileIdx = logicalBlockIdx + roundIdx_ * blockNum_;
        if (tileIdx >= totalCnt_) {
            return false;
        }
        int64_t rowIdx = (mCoreNum_ > 0) ? (tileIdx / (mCoreNum_ * nCnt_)) : 0;
        if (rowIdx < mainRow_) {
            int64_t localTileIdx = tileIdx - rowIdx * mCoreNum_ * nCnt_;
            Get<MNK_M>(blockCoord) = rowIdx * mCoreNum_ + localTileIdx % mCoreNum_;
            Get<MNK_N>(blockCoord) = (localTileIdx / mCoreNum_) % nCnt_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIdx = tileIdx - mainRow_ * mCoreNum_ * nCnt_;
            Get<MNK_M>(blockCoord) = mainRow_ * mCoreNum_ + tailIdx % mTailCoreNum_;
            Get<MNK_N>(blockCoord) = (tailIdx / mTailCoreNum_) % nCnt_;
        }
        if (rowIdx & 1) {
            Get<MNK_N>(blockCoord) = nCnt_ - 1 - Get<MNK_N>(blockCoord);
        }

        roundIdx_++;
        return true;
    }
};

template <class ProblemShape_>
struct BlockSchedulerSelector<ProblemShape_, QuantMatmulMxBaseScheduler> {
    using SchedulerOp = BlockSchedulerLastRoundTileBalance<ProblemShape_>;
};

}  // namespace Block
#endif
