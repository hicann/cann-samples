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
 * \file quant_matmul_hifp8_block_scheduler_swat.h
 * \brief Non–full-load SWAT scheduler for HiFP8 quantized cube matmul.
 */

#pragma once

#include "kernel_utils/common_utils.h"

#include "./block_scheduler_policy.h"

namespace Block {

template <class ProblemShape_, bool TransA_, bool TransB_>
 class BlockSchedulerQuantHifp8Swat {
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
     int64_t mCoreNum_{0};
     int64_t mTailCoreNum_{0};
     int64_t blockIdx_{AscendC::GetBlockIdx() / AscendC::GetTaskRation()};
     int64_t blockNum_{AscendC::GetBlockNum()};
     int64_t startBlockIdx_{0};
     int64_t endBlockIdx_{0};
     int64_t roundIdx_{0};
     int64_t round_{0};
     int64_t mTailTile_{1}; // init value must be 1
     int64_t nTailTile_{1}; // init value must be 1
     int64_t totalTailTile_{1}; // init value must be 1
     int64_t mSplitAddrOffset_{0};
     int64_t nSplitAddrOffset_{0};
     int64_t mainRow_{0};
 
     using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
     using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
     using ProblemShape = ProblemShape_;
     struct Params {
         int64_t baseM;
         int64_t baseN;
         int64_t mTailTile;
         int64_t nTailTile;
         int64_t mBaseTailSplitCnt;
         int64_t nBaseTailSplitCnt;
         int64_t mTailMain;
         int64_t nTailMain;
     };
 
     const int64_t WINDOW_LEN = 4;
 
 public:
     __aicore__ inline BlockSchedulerQuantHifp8Swat(const ProblemShape &shape, const Params &params)
     {
         m_ = shape.m;
         n_ = shape.n;
         k_ = shape.k;
         baseM_ = static_cast<int64_t>(params.baseM);
         baseN_ = static_cast<int64_t>(params.baseN);
         mCnt_ = CeilDiv(m_, baseM_);
         nCnt_ = CeilDiv(n_, baseN_);
         totalCnt_ = mCnt_ * nCnt_;
         mCoreNum_ = Min(WINDOW_LEN, mCnt_);
         mainRow_ = mCnt_ / mCoreNum_ - 1;
         mTailCoreNum_ = mCnt_ - mCoreNum_ * mainRow_;
         endBlockIdx_ = (totalCnt_ - 1) % blockNum_;
         round_ = CeilDiv(totalCnt_, blockNum_);
         if (blockIdx_ > endBlockIdx_) {
             round_ -= 1;
         }
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
         if constexpr (!TransA_) {
             mBaseNormCnt_ = mCnt_ - params.mBaseTailSplitCnt;
             int64_t mMergeSize = m_ - mBaseNormCnt_ * baseM_;
             mBaseTailMain_ = params.mBaseTailSplitCnt == 1 ? mMergeSize : params.mTailMain;
             mBaseTailLast_ = mMergeSize - (params.mBaseTailSplitCnt - 1) * mBaseTailMain_;
         } else {
             mBaseTailMain_ = m_ - (mCnt_ - 1) * baseM_;
         }
         if constexpr (TransB_) {
             nBaseNormCnt_ = nCnt_ - params.nBaseTailSplitCnt;
             int64_t nMergeSize = n_ - nBaseNormCnt_ * baseN_;
             nBaseTailMain_ = params.nBaseTailSplitCnt == 1 ? nMergeSize : params.nTailMain;
             nBaseTailLast_ = nMergeSize - (params.nBaseTailSplitCnt - 1) * nBaseTailMain_;
         } else {
             nBaseTailMain_ = n_ - (nCnt_ - 1) * baseN_;
         }
     }
 
     __aicore__ inline void UpdateTailTile(uint32_t mTailTile, uint32_t nTailTile)
     {
         mTailTile_ = mTailTile;
         nTailTile_ = nTailTile;
         totalTailTile_ = mTailTile * nTailTile;
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
 
     __aicore__ inline int64_t GetTotalCnt() const
     {
         return totalCnt_;
     }
 
     __aicore__ inline int64_t GetEndBlockIdx() const
     {
         return endBlockIdx_;
     }
 
     // Logical M/N tile indices are packed in MNK_K / MNK_B; GM origins live in MNK_M / MNK_N.
     __aicore__ inline void CalSingleCoreShapeByCoord(
         int64_t& singleCoreM, int64_t& singleCoreN, const BlockCoord& blockCoord)
     {
         const int64_t mTileIdx = AscendC::Std::get<MNK_K>(blockCoord);
         const int64_t nTileIdx = AscendC::Std::get<MNK_B>(blockCoord);
         if constexpr (!TransA_) {
             if (mTileIdx >= mBaseNormCnt_) {
                 singleCoreM = mTileIdx < mCnt_ - 1 ? mBaseTailMain_ : mBaseTailLast_;
             }
         } else {
             if (mTileIdx == mCnt_ - 1) {
                 singleCoreM = mBaseTailMain_;
             }
         }
         if constexpr (TransB_) {
             if (nTileIdx >= nBaseNormCnt_) {
                 singleCoreN = nTileIdx < nCnt_ - 1 ? nBaseTailMain_ : nBaseTailLast_;
             }
         } else {
             if (nTileIdx == nCnt_ - 1) {
                 singleCoreN = nBaseTailMain_;
             }
         }
     }

     __aicore__ inline void PackTileIntoCoord(BlockCoord& blockCoord, int64_t mTileIdx, int64_t nTileIdx,
         int64_t mSplitAddrOffset, int64_t nSplitAddrOffset)
     {
         int64_t mPos = mTileIdx * baseM_ + mSplitAddrOffset;
         int64_t nPos = nTileIdx * baseN_ + nSplitAddrOffset;
         if constexpr (!TransA_) {
             if (mTileIdx > mBaseNormCnt_) {
                 mPos -= (mTileIdx - mBaseNormCnt_) * (baseM_ - mBaseTailMain_);
             }
         }
         if constexpr (TransB_) {
             if (nTileIdx > nBaseNormCnt_) {
                 nPos -= (nTileIdx - nBaseNormCnt_) * (baseN_ - nBaseTailMain_);
             }
         }
         AscendC::Std::get<MNK_M>(blockCoord) = mPos;
         AscendC::Std::get<MNK_N>(blockCoord) = nPos;
         AscendC::Std::get<MNK_K>(blockCoord) = mTileIdx;
         AscendC::Std::get<MNK_B>(blockCoord) = nTileIdx;
     }
 
     __aicore__ inline BlockShape GetBlockShape(BlockCoord blockCoord)
     {
         int64_t singleCoreM = baseM_;
         int64_t singleCoreN = baseN_;
         CalSingleCoreShapeByCoord(singleCoreM, singleCoreN, blockCoord);
 
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
         singleCoreM = Min(singleCoreM - mSplitAddrOffset_, singleCoreMSplit);
         singleCoreN = Min(singleCoreN - nSplitAddrOffset_, singleCoreNSplit);
         return {singleCoreM, singleCoreN, mSplitAddrOffset_, nSplitAddrOffset_};
     }
 
     __aicore__ inline AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetLoadBalanceInfo()
     {
         return {static_cast<uint32_t>(mBaseNormCnt_), static_cast<uint32_t>(mBaseTailMain_),
                 static_cast<uint32_t>(nBaseNormCnt_), static_cast<uint32_t>(nBaseTailMain_)};
     }
 
     __aicore__ inline void UpdateNextBatchBlockRoundParams()
     {
         startBlockIdx_ = endBlockIdx_ + 1 == blockNum_ ? 0 : (endBlockIdx_ + 1);
         endBlockIdx_ = (totalCnt_ + startBlockIdx_ - 1) % blockNum_;
 
         roundIdx_ = 0;
         round_ = CeilDiv(totalCnt_, blockNum_);
         if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
             round_ -= 1;
         } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
             round_ -= 1;
         }
     }
 
     __aicore__ inline bool GetTileIdx(BlockCoord &blockCoord)
     {
         if (roundIdx_ >= round_) {
             return false;
         }

        const int64_t curRoundIdx = roundIdx_;
         int64_t newBlockIdx = (curRoundIdx == round_ - 1) ? blockIdx_ / totalTailTile_ : blockIdx_;
         int64_t tileIdx = newBlockIdx + curRoundIdx * blockNum_;
         if (blockIdx_ < startBlockIdx_) {
             tileIdx += blockNum_ - startBlockIdx_;
         } else if (endBlockIdx_ + 1 >= totalTailTile_ * totalCnt_) {
             tileIdx -= startBlockIdx_ / totalTailTile_;
         } else {
             tileIdx -= startBlockIdx_;
         }
         int64_t rowIdx = tileIdx / nCnt_ / mCoreNum_;
         int64_t mTileIdx = 0;
         int64_t nTileIdx = 0;
         if (rowIdx < mainRow_) {
             mTileIdx = rowIdx * mCoreNum_ + tileIdx % mCoreNum_;
             nTileIdx = (tileIdx / mCoreNum_) % nCnt_;
         } else {
             rowIdx = mainRow_;
             int64_t tailIdx = tileIdx - mainRow_ * mCoreNum_ * nCnt_;
             mTileIdx = mainRow_ * mCoreNum_ + tailIdx % mTailCoreNum_;
             nTileIdx = (tailIdx / mTailCoreNum_) % nCnt_;
         }
         if (rowIdx & 1) {
             nTileIdx = nCnt_ - 1 - nTileIdx;
         }

         BlockCoord tileOnly{};
         AscendC::Std::get<MNK_K>(tileOnly) = mTileIdx;
         AscendC::Std::get<MNK_B>(tileOnly) = nTileIdx;
         int64_t singleCoreM = baseM_;
         int64_t singleCoreN = baseN_;
         CalSingleCoreShapeByCoord(singleCoreM, singleCoreN, tileOnly);
         int64_t mSplitAddrOffset = 0;
         int64_t nSplitAddrOffset = 0;
         if (totalTailTile_ > 1 && curRoundIdx == round_ - 1) {
             int64_t singleCoreMSplit = CeilDiv(singleCoreM, mTailTile_);
             int64_t singleCoreNSplit = CeilDiv(singleCoreN, nTailTile_);
             const int64_t mSplitIdx = (blockIdx_ % totalTailTile_) % mTailTile_;
             const int64_t nSplitIdx = (blockIdx_ % totalTailTile_) / mTailTile_;
             mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
             nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
         }
         PackTileIntoCoord(blockCoord, mTileIdx, nTileIdx, mSplitAddrOffset, nSplitAddrOffset);
         roundIdx_++;
         return true;
     }
 };

} // namespace Block
