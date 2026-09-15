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
 * \file block_scheduler_block_attn_res_prepare.h
 * \brief Block scheduler for the BlockAttnResPrepare Phase1 mixed kernel.
 */

#pragma once

#include "tensor_api/tensor.h"

namespace BlockAttnResPrepareMix {
namespace Attention {
namespace Block {

template <class ProblemShape_>
class BlockSchedulerBlockAttnResPrepare {
public:
    using ProblemShape = ProblemShape_;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    struct BlockInfo {
        BlockShape blockShape{};
        BlockCoord blockCoord{};
    };

    struct AivRowRange {
        uint32_t rowStart{0U};
        uint32_t rowCount{0U};
    };

    struct Params {
        uint32_t totalWorkUnits{0U};
        uint32_t usedCoreNum{0U};
        uint32_t baseT{0U};
        uint32_t baseS{0U};
        uint32_t sTileNum{0U};
        uint32_t mm1NAlign{0U};
    };

    __aicore__ inline BlockSchedulerBlockAttnResPrepare(
        const ProblemShape& problemShape, const Params& params, uint32_t validN, uint32_t coreIndex)
        : totalS_(static_cast<uint64_t>(AscendC::Te::Get<S_DIM_INDEX>(problemShape))),
          totalD_(static_cast<uint64_t>(AscendC::Te::Get<D_DIM_INDEX>(problemShape))),
          totalT_(static_cast<uint64_t>(AscendC::Te::Get<T_DIM_INDEX>(problemShape))),
          validN_(validN),
          totalBlockNums_(params.totalWorkUnits),
          usedCoreNum_(params.usedCoreNum),
          baseT_(params.baseT),
          baseS_(params.baseS),
          sTileNum_(params.sTileNum)
    {
        AdjustRuntimeTokenGroup(params.mm1NAlign);
        InitializeCoreBlockDistribution();
        InitializeCoreBlockRange(coreIndex);
    }

    __aicore__ inline uint32_t GetCoreNums() const
    {
        return usedCoreNum_;
    }

    __aicore__ inline uint32_t GetBlockNums() const
    {
        return totalBlockNums_;
    }

    // Return the next S/T block assigned to this logical AIC/AIV task group.
    __aicore__ inline bool GetNextBlock(BlockInfo& blockInfo)
    {
        if (sTileNum_ == 0U || currentBlockIdx_ >= endBlockIdx_) {
            return false;
        }
        const uint64_t blockIdx = currentBlockIdx_++;
        const uint64_t sBlockIdx = blockIdx % sTileNum_;
        const uint64_t tBlockIdx = blockIdx / sTileNum_;
        const uint64_t sOffset = sBlockIdx * baseS_;
        const uint64_t tOffset = tBlockIdx * baseT_;
        if (sOffset >= totalS_ || tOffset >= totalT_) {
            return false;
        }
        const uint64_t remainingS = totalS_ - sOffset;
        const uint64_t remainingT = totalT_ - tOffset;
        const int64_t blockS = static_cast<int64_t>(remainingS < baseS_ ? remainingS : baseS_);
        const int64_t blockT = static_cast<int64_t>(remainingT < baseT_ ? remainingT : baseT_);
        blockInfo.blockShape = BlockShape{blockS, static_cast<int64_t>(validN_), static_cast<int64_t>(totalD_), blockT};
        blockInfo.blockCoord = BlockCoord{static_cast<int64_t>(sOffset), 0, 0, static_cast<int64_t>(tOffset)};
        return true;
    }

    // Split the block's S rows evenly across the AIVs belonging to the same logical task group.
    __aicore__ inline AivRowRange GetAivRowRange(const BlockShape& blockShape) const
    {
        const uint32_t blockS = static_cast<uint32_t>(AscendC::Te::Get<S_DIM_INDEX>(blockShape));
        // The __mix__(1, 2) entry normally supplies 2; reject an invalid runtime ratio before division.
        const uint32_t taskRatio = AscendC::GetTaskRation();
        if (taskRatio == 0U) {
            return {0U, 0U};
        }
        const uint32_t rowsPerAiv = blockS / taskRatio + (blockS % taskRatio == 0U ? 0U : 1U);
        const uint32_t rowStart = AscendC::GetSubBlockIdx() * rowsPerAiv;
        const uint32_t remainingRows = rowStart < blockS ? blockS - rowStart : 0U;
        const uint32_t rowCount = rowsPerAiv < remainingRows ? rowsPerAiv : remainingRows;
        return {rowStart, rowCount};
    }

private:
    static constexpr int32_t S_DIM_INDEX = 0;
    static constexpr int32_t D_DIM_INDEX = 2;
    static constexpr int32_t T_DIM_INDEX = 3;
    static constexpr uint64_t SMALL_D_THRESHOLD = 512U;
    static constexpr uint32_t SMALL_D_MAX_RUNTIME_BASE_T = 16U;
    static constexpr uint32_t LARGE_D_MAX_RUNTIME_BASE_T = 2U;
    static constexpr uint32_t RUNTIME_BASE_T_CANDIDATES[] = {16U, 8U, 4U, 2U, 1U};

    __aicore__ inline void AdjustRuntimeTokenGroup(uint32_t mm1NAlign)
    {
        if (validN_ == 0U || validN_ > mm1NAlign) {
            return;
        }

        const uint32_t maxBaseTByWorkspace = mm1NAlign / validN_;
        const uint32_t maxBaseTByProblem =
            totalD_ < SMALL_D_THRESHOLD ? SMALL_D_MAX_RUNTIME_BASE_T : LARGE_D_MAX_RUNTIME_BASE_T;
        for (const uint32_t candidate : RUNTIME_BASE_T_CANDIDATES) {
            if (candidate <= baseT_) {
                break;
            }
            if (candidate > maxBaseTByProblem || candidate > maxBaseTByWorkspace || candidate > totalT_) {
                continue;
            }

            const uint64_t tTileNum = (totalT_ + candidate - 1U) / candidate;
            const uint64_t candidateBlockNums = tTileNum * sTileNum_;
            if (candidateBlockNums < usedCoreNum_) {
                continue;
            }

            baseT_ = candidate;
            totalBlockNums_ = static_cast<uint32_t>(candidateBlockNums);
            break;
        }
    }

    __aicore__ inline void InitializeCoreBlockDistribution()
    {
        if (usedCoreNum_ == 0U) {
            return;
        }
        blocksPerCore_ = totalBlockNums_ / usedCoreNum_;
        extraBlockCoreNum_ = totalBlockNums_ % usedCoreNum_;
    }

    __aicore__ inline void InitializeCoreBlockRange(uint32_t coreIndex)
    {
        if (coreIndex >= usedCoreNum_) {
            return;
        }
        currentBlockIdx_ = static_cast<uint64_t>(coreIndex) * blocksPerCore_ +
                           (coreIndex < extraBlockCoreNum_ ? coreIndex : extraBlockCoreNum_);
        const uint32_t blockCount = blocksPerCore_ + (coreIndex < extraBlockCoreNum_ ? 1U : 0U);
        endBlockIdx_ = currentBlockIdx_ + blockCount;
    }

    uint64_t totalS_{0U};
    uint64_t totalD_{0U};
    uint64_t totalT_{0U};
    uint32_t validN_{0U};
    uint32_t totalBlockNums_{0U};
    uint32_t usedCoreNum_{0U};
    uint32_t blocksPerCore_{0U};
    uint32_t extraBlockCoreNum_{0U};
    uint32_t baseT_{0U};
    uint32_t baseS_{0U};
    uint32_t sTileNum_{0U};
    uint64_t currentBlockIdx_{0U};
    uint64_t endBlockIdx_{0U};
};

} // namespace Block
} // namespace Attention
} // namespace BlockAttnResPrepareMix
