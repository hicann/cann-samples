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
 * \file quant_matmul_mxfp4_block_scheduler_a_full_load.h
 * \brief SWAT block scheduler for the MXFP4 A-full-load path.
 */

#ifndef QUANT_MATMUL_MXFP4_BLOCK_SCHEDULER_A_FULL_LOAD_H
#define QUANT_MATMUL_MXFP4_BLOCK_SCHEDULER_A_FULL_LOAD_H

#include "kernel_utils/common_utils.h"

#include "./block_scheduler_utils.h"

namespace Block {

template <class ProblemShape_, bool TransA_, bool TransB_>
class BlockSchedulerQuantMatmulMxAFullLoad {
public:
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t mCnt_{0};
    int64_t nCnt_{0};
    int64_t totalCnt_{0};
    int64_t mBaseNormCnt_{0};
    int64_t nBaseNormCnt_{0};
    int64_t mBaseTailMain_{0};
    int64_t nBaseTailMain_{0};
    int64_t mBaseTailLast_{0};
    int64_t nBaseTailLast_{0};
    int64_t blockIdx_{AscendC::GetBlockIdx() / AscendC::GetTaskRation()};
    int64_t blockNum_{AscendC::GetBlockNum()};
    int64_t endBlockIdx_{0};
    int64_t roundIdx_{0};
    int64_t round_{0};
    int64_t mTailTile_{1};
    int64_t nTailTile_{1};
    int64_t totalTailTile_{1};

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        // Host tiling passes the steady-state tile shape plus the merged-tail
        // description that the runtime scheduler must reconstruct on device.
        int64_t baseM;
        int64_t baseN;
        int64_t mTailTile;
        int64_t nTailTile;
        int64_t mBaseTailSplitCnt;
        int64_t nBaseTailSplitCnt;
        int64_t mTailMain;
        int64_t nTailMain;
    };

public:
    __aicore__ inline BlockSchedulerQuantMatmulMxAFullLoad(const ProblemShape& shape, const Params& params)
    {
        // Pre-compute the steady-state and tail tile geometry once so the hot
        // scheduling loop only needs light index arithmetic.
        m_ = shape.m;
        n_ = shape.n;
        k_ = shape.k;
        baseM_ = static_cast<int64_t>(params.baseM);
        baseN_ = static_cast<int64_t>(params.baseN);
        mCnt_ = CeilDiv(m_, baseM_);
        nCnt_ = CeilDiv(n_, baseN_);
        totalCnt_ = mCnt_ * nCnt_;
        endBlockIdx_ = (totalCnt_ - 1) % blockNum_;
        round_ = CeilDiv(totalCnt_, blockNum_);
        if (blockIdx_ > endBlockIdx_) {
            round_ -= 1;
        }
        // Apply host-selected tail splitting during construction so the
        // scheduler is fully initialized before the first tile query.
        if ((endBlockIdx_ + 1) * params.mTailTile * params.nTailTile <= AscendC::GetBlockNum()) {
            mTailTile_ = params.mTailTile;
            nTailTile_ = params.nTailTile;
            totalTailTile_ = params.mTailTile * params.nTailTile;

            uint64_t tailOriCnt = AscendC::Std::min(totalCnt_, endBlockIdx_ + 1);
            int64_t newEndBlockIdx = endBlockIdx_ + tailOriCnt * (totalTailTile_ - 1);
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

        mBaseNormCnt_ = mCnt_ - params.mBaseTailSplitCnt;
        int64_t mMergeSize = m_ - mBaseNormCnt_ * baseM_;
        mBaseTailMain_ = params.mBaseTailSplitCnt == 1 ? mMergeSize : params.mTailMain;
        mBaseTailLast_ = mMergeSize - (params.mBaseTailSplitCnt - 1) * mBaseTailMain_;

        nBaseNormCnt_ = nCnt_ - params.nBaseTailSplitCnt;
        int64_t nMergeSize = n_ - nBaseNormCnt_ * baseN_;
        nBaseTailMain_ = params.nBaseTailSplitCnt == 1 ? nMergeSize : params.nTailMain;
        nBaseTailLast_ = nMergeSize - (params.nBaseTailSplitCnt - 1) * nBaseTailMain_;
    }

    __aicore__ inline BlockShape GetBlockShape(BlockCoord blockCoord)
    {
        // `blockCoord` carries GM coordinates in M/N and keeps the logical
        // tile indices in K/B. Shape reconstruction must therefore read K/B.
        int64_t singleCoreM = baseM_;
        int64_t singleCoreN = baseN_;
        int64_t mTileIdx = Get<MNK_K>(blockCoord);
        int64_t nTileIdx = Get<MNK_B>(blockCoord);
        if (mTileIdx >= mBaseNormCnt_) {
            singleCoreM = mTileIdx < mCnt_ - 1 ? mBaseTailMain_ : mBaseTailLast_;
        }
        if (nTileIdx >= nBaseNormCnt_) {
            singleCoreN = nTileIdx < nCnt_ - 1 ? nBaseTailMain_ : nBaseTailLast_;
        }

        // `GetTileIdx` advances `roundIdx_` before the kernel asks for shape,
        // so equality here means "the tile that was just issued belongs to the
        // final split-tail round".
        bool isTailSplitRound = totalTailTile_ > 1 && roundIdx_ == round_;
        if (!isTailSplitRound) {
            return {singleCoreM, singleCoreN, 0, 0};
        }

        int64_t singleCoreMSplit = CeilDiv(singleCoreM, mTailTile_);
        int64_t singleCoreNSplit = CeilDiv(singleCoreN, nTailTile_);
        int64_t mSplitIdx = (blockIdx_ % totalTailTile_) % mTailTile_;
        int64_t nSplitIdx = blockIdx_ / mCnt_ % nTailTile_;
        int64_t mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        int64_t nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (mSplitAddrOffset >= singleCoreM || nSplitAddrOffset >= singleCoreN) {
            // Some synthetic tail slices may fall outside the valid range when
            // the edge tile is smaller than the requested split grid.
            return {0, 0, 0, 0};
        }
        singleCoreM = Min(singleCoreM - mSplitAddrOffset, singleCoreMSplit);
        singleCoreN = Min(singleCoreN - nSplitAddrOffset, singleCoreNSplit);
        return {singleCoreM, singleCoreN, mSplitAddrOffset, nSplitAddrOffset};
    }

    __aicore__ inline bool GetTileIdx(BlockCoord& blockCoord)
    {
        if (roundIdx_ >= round_) {
            return false;
        }

        // A-full-load keeps each core on a fixed M tile and advances N across
        // rounds so the resident A tile can be reused as long as possible.
        int64_t curRoundIdx = roundIdx_;
        int64_t mTileIdx = blockIdx_ % mCnt_;
        int64_t curNTailTile = (curRoundIdx == round_ - 1) ? nTailTile_ : 1;
        int64_t nTileIdx = curRoundIdx * blockNum_ / mCnt_ % nCnt_ + blockIdx_ / mCnt_ / curNTailTile;

        int64_t singleCoreM = mTileIdx >= mBaseNormCnt_ ? (mTileIdx < mCnt_ - 1 ? mBaseTailMain_ : mBaseTailLast_)
                                                        : baseM_;
        int64_t singleCoreN = nTileIdx >= nBaseNormCnt_ ? (nTileIdx < nCnt_ - 1 ? nBaseTailMain_ : nBaseTailLast_)
                                                        : baseN_;
        int64_t mSplitAddrOffset = 0;
        int64_t nSplitAddrOffset = 0;
        if (totalTailTile_ > 1 && curRoundIdx == round_ - 1) {
            int64_t singleCoreMSplit = CeilDiv(singleCoreM, mTailTile_);
            int64_t singleCoreNSplit = CeilDiv(singleCoreN, nTailTile_);
            int64_t mSplitIdx = (blockIdx_ % totalTailTile_) % mTailTile_;
            int64_t nSplitIdx = blockIdx_ / mCnt_ % nTailTile_;
            mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
            nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        }

        int64_t mPos = mTileIdx * baseM_ + mSplitAddrOffset;
        int64_t nPos = nTileIdx * baseN_ + nSplitAddrOffset;
        if (mTileIdx > mBaseNormCnt_) {
            mPos -= (mTileIdx - mBaseNormCnt_) * (baseM_ - mBaseTailMain_);
        }
        if (nTileIdx > nBaseNormCnt_) {
            nPos -= (nTileIdx - nBaseNormCnt_) * (baseN_ - nBaseTailMain_);
        }

        // Pack one scheduler result into `blockCoord`:
        // M/N hold GM origin, while K/B preserve logical tile indices for the
        // later `GetBlockShape` call.
        Get<MNK_M>(blockCoord) = mPos;
        Get<MNK_N>(blockCoord) = nPos;
        Get<MNK_K>(blockCoord) = mTileIdx;
        Get<MNK_B>(blockCoord) = nTileIdx;
        roundIdx_++;
        return true;
    }
};

template <class ProblemShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, QuantMatmulMxSwatScheduler<SWAT_A_FULL_LOAD_MODE>, TransA_, TransB_> {
    using SchedulerOp = BlockSchedulerQuantMatmulMxAFullLoad<ProblemShape_, TransA_, TransB_>;
};

} // namespace Block

#endif
