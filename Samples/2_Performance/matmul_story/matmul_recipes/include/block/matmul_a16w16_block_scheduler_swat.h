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
 * \file matmul_a16w16_block_scheduler_swat.h
 * \brief SWAT block scheduler for *A16W16 non-full-load path.
 */

#ifndef MATMUL_A16W16_BLOCK_SCHEDULER_SWAT_H
#define MATMUL_A16W16_BLOCK_SCHEDULER_SWAT_H

#include "kernel_utils/common_utils.h"
#include "utils/matmul_a16w16_constant.h"
#include "tiling/matmul_a16w16_tiling_data.h"
#include "./block_scheduler_utils.h"

namespace Block {

template <class ProblemShape_>
class BlockSchedulerA16W16Swat {
public:
    int64_t mTileNum_{0};
    int64_t nTileNum_{0};
    int64_t blockIdx_{0};
    int64_t perCoreBlockNum_{0};
    int64_t blockNum_{0};
    int64_t batch_{0};
    int64_t k_{0};
    int64_t tailL1M_{0};
    int64_t tailL1N_{0};
    int64_t mTailCnt_{1};
    int64_t nTailCnt_{1};
    int64_t tailCnt_{1};
    int64_t tileNum_{1};
    int64_t mainWindow_{1};
    int64_t mainRow_{1};
    int64_t tailWindow_{1};
    int64_t mTileIdx_{1};
    int64_t nTileIdx_{1};
    int64_t lastTileIdx_{-1};
    int64_t nSplitOffset_{0};
    int64_t mSplitOffset_{0};
    int64_t mL1_{0};
    int64_t nL1_{0};
    int64_t kL1_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t baseK_{0};
    int64_t mL1NormCnt_{0};
    int64_t mL1TailSplitCnt_{1};
    int64_t mL1TailMain_{0};
    int64_t mL1TailLast_{0};
    int64_t nL1NormCnt_{0};
    int64_t nL1TailSplitCnt_{1};
    int64_t nL1TailMain_{0};
    int64_t nL1TailLast_{0};

    static constexpr uint64_t WINDOW_LEN = 4UL;
    static constexpr uint64_t BLOCK_SIZE_16 = 16UL;
    static constexpr uint64_t BLOCK_SIZE_32 = 32UL;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        int64_t mL1;
        int64_t nL1;
        int64_t kL1;
        int64_t baseM;
        int64_t baseN;
        int64_t baseK;
        int64_t mBaseTailSplitCnt;
        int64_t nBaseTailSplitCnt;
        int64_t mTailMain;
        int64_t nTailMain;
        int64_t mTailCnt;
        int64_t nTailCnt;
    };

public:
    __aicore__ inline BlockSchedulerA16W16Swat(
        const ProblemShape& shape, int64_t blockIdx, int64_t blockNum, const Params& params)
        : blockIdx_(blockIdx), blockNum_(blockNum)
    {
        k_ = shape.k;
        batch_ = AscendC::Std::max(shape.b, 1L);
        mL1_ = params.mL1;
        nL1_ = params.nL1;
        kL1_ = params.kL1;
        baseM_ = params.baseM;
        baseN_ = params.baseN;
        baseK_ = params.baseK;
        mTileNum_ = CeilDiv(shape.m, params.mL1);
        nTileNum_ = CeilDiv(shape.n, params.nL1);
        perCoreBlockNum_ = GetPerBlockNum(blockNum_, mTileNum_, nTileNum_, batch_);
        tileNum_ = mTileNum_ * nTileNum_;
        int64_t tailTileNum = tileNum_ % blockNum_;
        mL1TailSplitCnt_ = params.mBaseTailSplitCnt;
        nL1TailSplitCnt_ = params.nBaseTailSplitCnt;
        mL1NormCnt_ = mTileNum_ - mL1TailSplitCnt_;
        nL1NormCnt_ = nTileNum_ - nL1TailSplitCnt_;
        tailL1M_ = shape.m - mL1NormCnt_ * params.mL1;
        tailL1N_ = shape.n - nL1NormCnt_ * params.nL1;
        mL1TailMain_ = mL1TailSplitCnt_ == 1 ? tailL1M_ : params.mTailMain;
        mL1TailLast_ = tailL1M_ - (mL1TailSplitCnt_ - 1) * mL1TailMain_;
        nL1TailMain_ = nL1TailSplitCnt_ == 1 ? tailL1N_ : params.nTailMain;
        nL1TailLast_ = tailL1N_ - (nL1TailSplitCnt_ - 1) * nL1TailMain_;
        if (batch_ == 1) {
            mTailCnt_ = params.mTailCnt;
            nTailCnt_ = params.nTailCnt;
            int64_t mTailSplit = CeilDiv(mL1TailLast_, mTailCnt_);
            int64_t nTailSplit = CeilDiv(nL1TailLast_, nTailCnt_);
            mTailCnt_ = CeilDiv(mL1TailLast_, mTailSplit);
            nTailCnt_ = CeilDiv(nL1TailLast_, nTailSplit);
            tailCnt_ = mTailCnt_ * nTailCnt_;
            tileNum_ += (tailCnt_ - 1) * tailTileNum;
        }
        mainWindow_ = WINDOW_LEN < mTileNum_ ? WINDOW_LEN : mTileNum_;
        mainRow_ = mTileNum_ / mainWindow_ - 1;
        tailWindow_ = mTileNum_ - mainRow_ * mainWindow_;
    }

    __aicore__ inline int64_t GetPerBlockNum(int64_t coreNum, int64_t mTileNum, int64_t nTileNum, int64_t b = 1)
    {
        int64_t perCoreBlockNum = AscendC::CeilDiv(mTileNum * nTileNum * b, coreNum);
        return perCoreBlockNum;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return tileNum_ * batch_;
    }

    __aicore__ inline AscendC::Shape<int64_t, int64_t, int64_t, int64_t> GetTailParams()
    {
        return {mL1NormCnt_, mL1TailMain_, nL1NormCnt_, nL1TailMain_};
    }

    __aicore__ inline int64_t GetBlockNum(ProblemShape shape, int64_t blockNum)
    {
        int64_t tilingBlockNum = 0;
        if (tileNum_ * batch_ < blockNum) {
            tilingBlockNum = tileNum_ * batch_;
        } else {
            tilingBlockNum = blockNum;
        }
        return tilingBlockNum;
    }

    __aicore__ inline BlockShape GetBlockShape(int64_t tileIdx)
    {
        UpdateMNTileIdx(tileIdx);
        int64_t blkM = mL1_;
        int64_t blkN = nL1_;
        if (mTileIdx_ >= mL1NormCnt_) {
            blkM = mTileIdx_ == (mTileNum_ - 1) ? mL1TailLast_ : mL1TailMain_;
        }
        if (nTileIdx_ >= nL1NormCnt_) {
            blkN = nTileIdx_ == (nTileNum_ - 1) ? nL1TailLast_ : nL1TailMain_;
        }
        if (tileIdx / blockNum_ != (perCoreBlockNum_ - 1) || tailCnt_ == 1) {
            return {blkM, blkN, k_, batch_};
        }
        int64_t splitBlkM = CeilDiv(blkM, mTailCnt_);
        int64_t splitBlkN = CeilDiv(blkN, nTailCnt_);
        int64_t mSplitIdx = (blockIdx_ % tailCnt_) % mTailCnt_;
        int64_t nSplitIdx = (blockIdx_ % tailCnt_) / mTailCnt_;
        mSplitOffset_ = mSplitIdx * splitBlkM;
        nSplitOffset_ = nSplitIdx * splitBlkN;
        if (mSplitOffset_ >= blkM || nSplitOffset_ >= blkN) {
            return {0, 0, k_, batch_};
        }
        splitBlkM = AscendC::Std::min(blkM - mSplitOffset_, splitBlkM);
        splitBlkN = AscendC::Std::min(blkN - nSplitOffset_, splitBlkN);
        return {splitBlkM, splitBlkN, k_, batch_};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int tileIdx)
    {
        UpdateMNTileIdx(tileIdx);
        int64_t mOffset = mTileIdx_ * mL1_ + mSplitOffset_;
        int64_t nOffset = nTileIdx_ * nL1_ + nSplitOffset_;
        if (mTileIdx_ > mL1NormCnt_) {
            mOffset = mL1NormCnt_ * mL1_ + (mTileIdx_ - mL1NormCnt_) * mL1TailMain_ + mSplitOffset_;
        }
        if (nTileIdx_ > nL1NormCnt_) {
            nOffset = nL1NormCnt_ * nL1_ + (nTileIdx_ - nL1NormCnt_) * nL1TailMain_ + nSplitOffset_;
        }
        return {mOffset, nOffset, 0, 0};
    }

private:
    __aicore__ inline void UpdateMNTileIdx(int64_t tmpIdx)
    {
        if (lastTileIdx_ == tmpIdx) {
            return;
        }
        lastTileIdx_ = tmpIdx;

        int64_t tileIdx = tmpIdx % tileNum_;
        if (tileIdx / blockNum_ == (perCoreBlockNum_ - 1) && tailCnt_ > 1) {
            tileIdx = (perCoreBlockNum_ - 1) * blockNum_ + blockIdx_ / tailCnt_;
        }
        int64_t rowIdx = tileIdx / nTileNum_ / mainWindow_;
        if (rowIdx < mainRow_) {
            mTileIdx_ = rowIdx * mainWindow_ + tileIdx % mainWindow_;
            nTileIdx_ = (tileIdx / mainWindow_) % nTileNum_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIndex = tileIdx - mainRow_ * mainWindow_ * nTileNum_;
            mTileIdx_ = mainRow_ * mainWindow_ + tailIndex % tailWindow_;
            nTileIdx_ = (tailIndex / tailWindow_) % nTileNum_;
        }
        if (rowIdx % 2 != 0) {
            nTileIdx_ = nTileNum_ - 1 - nTileIdx_;
        }
    }
};

template <class ProblemShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, MatmulA16W16SwatScheduler<NO_FULL_LOAD_MODE>, TransA_, TransB_> {
    using SchedulerOp = BlockSchedulerA16W16Swat<ProblemShape_>;
};
} // namespace Block
#endif
