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
 * \file block_scheduler_gmm_aswt_with_tail_split.h
 * \brief
 */

#pragma once
 #include "./block_scheduler_utils.h"
 
namespace Block {
 constexpr int64_t INNER_AXIS_MIN_SPLIT_VAL = 128LL; // ND2NZ cache line size is 128.
 constexpr int64_t WINDOW_LEN = 4LL; // Window length is 4.
 template <class ProblemShape_, bool TransA_, bool TransB_>
 class BlockSchedulerGmmAswtWithTailSplit {
 public:
     using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
     using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
     using ProblemShape = ProblemShape_;
     struct Params {
         int32_t baseM{0};
         int32_t baseN{0};
     };
 
 private:
     int64_t mCnt_;
     int64_t nCnt_;
     int64_t totalCnt_;
     int64_t m_{0};
     int64_t n_{0};
     int64_t k_;
     int64_t mTailCnt_{1};
     int64_t nTailCnt_{1};
     int64_t mTailAlign_{1};
     int64_t nTailAlign_{1};
     int64_t tailCnt_{1}; // Updated only for the last group.
     int64_t mainMWindow_;
     int64_t tailWindow_;
     int64_t mainRow_;
     int64_t round_;
     int64_t roundIdx_;
     int32_t baseM_;
     int32_t baseN_;
     int32_t mBaseTail_;
     int32_t nBaseTail_;
     uint32_t blockNum_ = AscendC::GetBlockNum();
     uint32_t blockIdx_ = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
     uint32_t startBlockIdx_;
     uint32_t endBlockIdx_{blockNum_ - 1};
 
 public:
     __aicore__ inline BlockSchedulerGmmAswtWithTailSplit(const Params &params)
         : baseM_(params.baseM), baseN_(params.baseN)
     {
     }
 
     __aicore__ inline BlockSchedulerGmmAswtWithTailSplit(int32_t baseM, int32_t baseN, int32_t /*baseK*/)
         : BlockSchedulerGmmAswtWithTailSplit(Params{baseM, baseN})
     {
     }
 
     __aicore__ inline void UpdateNextProblem(const TupleShape &problemShape)
     {
         k_ = Get<MNK_K>(problemShape);
         if (m_ != Get<MNK_M>(problemShape) || n_ != Get<MNK_N>(problemShape)) {
             m_ = Get<MNK_M>(problemShape);
             n_ = Get<MNK_N>(problemShape);
             mCnt_ = CeilDiv(m_, baseM_);
             nCnt_ = CeilDiv(n_, baseN_);
             mBaseTail_ = m_ - (mCnt_ - 1) * baseM_;
             nBaseTail_ = n_ - (nCnt_ - 1) * baseN_;
             totalCnt_ = mCnt_ * nCnt_;
             mainMWindow_ = WINDOW_LEN < mCnt_ ? WINDOW_LEN : mCnt_;
             mainRow_ = mCnt_ / mainMWindow_ - 1;
             tailWindow_ = mCnt_ - mainMWindow_ * mainRow_;
         }
         roundIdx_ = 0;
         round_ = CeilDiv(totalCnt_, blockNum_);
         // The first blockIdx of the new group.
         startBlockIdx_ = endBlockIdx_ == blockNum_ - 1 ? 0 : (endBlockIdx_ + 1);
         // The last blockIdx of the new group.
         endBlockIdx_ = (totalCnt_ + startBlockIdx_ - 1) % blockNum_;
         // Calculate the actual round count for the new group.
         if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
             round_ -= 1;
         } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
             round_ -= 1;
         }
     }
 
     __aicore__ inline void UpdateBaseM(uint32_t baseM)
     {
         baseM_ = baseM;
     }
 
     __aicore__ inline void SetTailAlign(uint32_t mTailAlign, uint32_t nTailAlign)
     {
         mTailAlign_ = mTailAlign;
         nTailAlign_ = nTailAlign;
     }
 
     __aicore__ inline int64_t GetTailTileCnt()
     {
         return Min(static_cast<int64_t>(endBlockIdx_ + 1), totalCnt_);
     }
 
     __aicore__ inline void UpdateTailTile(uint32_t mTailCnt, uint32_t nTailCnt)
     {
         mTailCnt_ = mTailCnt;
         nTailCnt_ = nTailCnt;
         tailCnt_ = mTailCnt_ * nTailCnt_;
         int64_t tailOriCnt = GetTailTileCnt();
         int64_t newEndBlockIdx = endBlockIdx_ + tailOriCnt * (tailCnt_ - 1);
         if (blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx) {
             round_ += 1;
         }
         if (blockIdx_ > newEndBlockIdx) { // No tail split is needed when blockIdx is not in the last round.
             mTailCnt_ = 1;
             nTailCnt_ = 1;
             tailCnt_ = 1;
         }
         endBlockIdx_ = newEndBlockIdx;
     }
 
     __aicore__ inline void UpdateTailTile()
     {
         // Calculate the splittable tile count; keep 1 when no split is needed.
         int64_t remainTile = (AscendC::GetBlockNum() - endBlockIdx_ - 1) / GetTailTileCnt() + 1;
         if (remainTile <= 1) {
             return;
         }
 
         // Initialize the minimum tile sizes.
         int64_t mMin = AscendC::BLOCK_CUBE;
         int64_t nMin = AscendC::BLOCK_CUBE;
 
         // Adjust the minimum tile sizes based on the A/B transpose layout.
         if constexpr (TransA_) {
             mMin = INNER_AXIS_MIN_SPLIT_VAL;
         }
         if constexpr (!TransB_) {
             nMin = INNER_AXIS_MIN_SPLIT_VAL;
         }
         int64_t mTile = Min(CeilDiv(mBaseTail_, mMin), remainTile);
         int64_t nTile = Min(CeilDiv(nBaseTail_, nMin), remainTile);
         while (mTile * nTile > remainTile) {
             if (mTile >= nTile) {
                 mTile -= 1;
             } else {
                 nTile -= 1;
             }
         }
         UpdateTailTile(mTile, nTile);
     }
 
     __aicore__ inline bool GetTileIdx(BlockCoord &blockCoord)
     {
         if (round_ == 0 || roundIdx_ > round_ - 1) {
             return false;
         }
         int64_t newBlockIdx = (roundIdx_ == round_ - 1) ? blockIdx_ / tailCnt_ : blockIdx_;
         int64_t index = newBlockIdx + roundIdx_ * blockNum_;
         // Apply the startBlockIdx offset.
         if (blockIdx_ < startBlockIdx_) {
             index += blockNum_ - startBlockIdx_;
         } else if (endBlockIdx_ + 1 >= tailCnt_ * totalCnt_) {
             index -= startBlockIdx_ / tailCnt_;
         } else {
             index -= startBlockIdx_;
         }
         int64_t rowIdx = index / nCnt_ / mainMWindow_;
         if (rowIdx < mainRow_) {
             Get<MNK_M>(blockCoord) = rowIdx * mainMWindow_ + index % mainMWindow_;
             Get<MNK_N>(blockCoord) = (index / mainMWindow_) % nCnt_;
         } else {
             rowIdx = mainRow_;
             int64_t tailIndex = index - mainRow_ * mainMWindow_ * nCnt_;
             Get<MNK_M>(blockCoord) = mainRow_ * mainMWindow_ + tailIndex % tailWindow_;
             Get<MNK_N>(blockCoord) = (tailIndex / tailWindow_) % nCnt_;
         }
 
         if (rowIdx & 1) {
             Get<MNK_N>(blockCoord) = nCnt_ - 1 - Get<MNK_N>(blockCoord);
         }
         roundIdx_++;
         return true;
     }
 
     __aicore__ inline TupleShape GetBlockShape(const BlockCoord &blockCoord)
     {
         int64_t singleCoreM = Get<MNK_M>(blockCoord) != (mCnt_ - 1) ? baseM_ : mBaseTail_;
         int64_t singleCoreN = Get<MNK_N>(blockCoord) != (nCnt_ - 1) ? baseN_ : nBaseTail_;
         if (tailCnt_ == 1 || roundIdx_ < round_) { // roundIdx++ in GetTileIdx
             return {singleCoreM, singleCoreN, 0, 0};
         }
 
         int64_t singleCoreMSplit = (singleCoreM + mTailCnt_ - 1) / mTailCnt_;
         int64_t singleCoreNSplit = (singleCoreN + nTailCnt_ - 1) / nTailCnt_;
         if constexpr (TransA_) { // (k, m)
             singleCoreMSplit = Align(singleCoreMSplit, INNER_AXIS_MIN_SPLIT_VAL);
         }
         singleCoreNSplit = Align(singleCoreNSplit, nTailAlign_);
         if constexpr (!TransB_) { // (k, n)
             singleCoreNSplit = Align(singleCoreNSplit, INNER_AXIS_MIN_SPLIT_VAL);
         }
         int64_t mSplitIdx = (blockIdx_ % tailCnt_) % mTailCnt_;
         int64_t nSplitIdx = (blockIdx_ % tailCnt_) / mTailCnt_;
         int64_t mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
         int64_t nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
         if (mSplitAddrOffset >= singleCoreM || nSplitAddrOffset >= singleCoreN) {
             return {0, 0, 0, 0};
         }
         singleCoreM = Min(singleCoreM - mSplitAddrOffset, singleCoreMSplit);
         singleCoreN = Min(singleCoreN - nSplitAddrOffset, singleCoreNSplit);
         return {singleCoreM, singleCoreN, mSplitAddrOffset, nSplitAddrOffset};
     }
 
     __aicore__ inline int64_t GetEndBlockIdx() const
     {
         return endBlockIdx_;
     }
 };

} // namespace Block
