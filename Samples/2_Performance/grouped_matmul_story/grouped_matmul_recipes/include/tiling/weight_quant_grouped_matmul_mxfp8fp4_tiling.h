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
 * \file weight_quant_grouped_matmul_mxfp8fp4_tiling.h
 * \brief Host-side tiling helper for weight-quant grouped matmul.
 */
#pragma once

#include "weight_quant_grouped_matmul_mxfp8fp4_tiling_data.h"

#include <algorithm>
#include <cstdint>

#include "../utils/grouped_matmul_constant.h"
#include "host_utils/common_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace weight_quant_tiling_detail {
// BASIC_BLOCK_BASE_N / BASIC_BLOCK_BASE_N_MIN come from the original resplit logic comments:
// main block ~256, tail block ~128~256.
constexpr uint32_t BASIC_BLOCK_BASE_M = 256U;
constexpr uint64_t BASIC_BLOCK_BASE_N = 256UL;
constexpr uint64_t BASIC_BLOCK_BASE_N_MIN = 128UL;
} // namespace weight_quant_tiling_detail

class WeightQuantGroupedMatmulMxfp8fp4Tiling {
public:
    // Generate host-side tiling parameters from group count and problem sizes.
    void GetTilingData(uint32_t numOfGroups, uint64_t n, uint64_t k, WeightQuantGroupedMatmulTilingData& tilingData)
    {
        CHECK_COND(numOfGroups > 0U, "numOfGroups must be greater than zero.");
        CHECK_COND(n > 0U && k > 0U, "n and k must be greater than zero.");
        CHECK_COND((n % 32U) == 0U, "n must be a multiple of 32.");
        CHECK_COND((k % 64U) == 0U, "k must be a multiple of 64.");

        groupNum_ = numOfGroups;
        kSize_ = k;
        nSize_ = n;

        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
        CHECK_COND(ascendcPlatform != nullptr, "Get ascendcPlatform failed.");
        coreNum_ = static_cast<uint32_t>(ascendcPlatform->GetCoreNumAic());
        CHECK_COND(coreNum_ != 0U, "CoreNum should be greater than 0.");

        tilingData = {};
        CalcResplitTiling();
        SetBaseTiling(tilingData);
        tilingData.Print();
    }

private:
    struct TailBlockResplitParam {
        uint32_t mainBlockSize = 0;
        uint64_t mainBlockCount = 0;
        uint16_t firstTailBlockSize = 0;
        uint16_t secondTailBlockSize = 0;
        uint16_t firstTailBlockCount = 0;
        uint16_t secondTailBlockCount = 0;
    };

    void SetBaseTiling(WeightQuantGroupedMatmulTilingData& tilingData)
    {
        tilingData.groupNum = groupNum_;
        tilingData.coreNum = coreNum_;
        tilingData.kSize = kSize_;
        tilingData.nSize = nSize_;
        tilingData.cubeNumBlocksN = cubeNumBlocksN_;
        tilingData.mainBlockSize = resplitParam_.mainBlockSize;
        tilingData.mainBlockCount = resplitParam_.mainBlockCount * coreNum_;
        tilingData.firstTailBlockSize = resplitParam_.firstTailBlockSize;
        tilingData.secondTailBlockSize = resplitParam_.secondTailBlockSize;
        tilingData.firstTailBlockCount = resplitParam_.firstTailBlockCount;
        tilingData.secondTailBlockCount = resplitParam_.secondTailBlockCount;
        tilingData.baseM = weight_quant_tiling_detail::BASIC_BLOCK_BASE_M;
    }

    void CalcResplitTiling()
    {
        cubeNumBlocksN_ = static_cast<uint8_t>(coreNum_);
        if (nSize_ % (coreNum_ * static_cast<uint64_t>(weight_quant_tiling_detail::BASIC_BLOCK_BASE_N)) == 0UL) {
            resplitParam_.mainBlockSize = weight_quant_tiling_detail::BASIC_BLOCK_BASE_N;
            resplitParam_.mainBlockCount =
                nSize_ / (coreNum_ * static_cast<uint64_t>(weight_quant_tiling_detail::BASIC_BLOCK_BASE_N));
        } else if (nSize_ > coreNum_ * static_cast<uint64_t>(weight_quant_tiling_detail::BASIC_BLOCK_BASE_N_MIN)) {
            // Full-core split, keep tail block size in ~128~256.
            CalcFullNumBlocksResplitTiling();
        } else {
            // N <= coreNum * 128: prioritize tail size > 128.
            CalcNoFullNumBlocksResplitTiling();
        }
    }

    void CalcFullNumBlocksResplitTiling()
    {
        resplitParam_.mainBlockSize = weight_quant_tiling_detail::BASIC_BLOCK_BASE_N;
        resplitParam_.mainBlockCount = 0UL;
        if (nSize_ / (coreNum_ * static_cast<uint64_t>(weight_quant_tiling_detail::BASIC_BLOCK_BASE_N)) > 1UL) {
            resplitParam_.mainBlockCount =
                nSize_ / (coreNum_ * static_cast<uint64_t>(weight_quant_tiling_detail::BASIC_BLOCK_BASE_N)) - 1UL;
        }

        uint64_t tailSizeOri = nSize_ - resplitParam_.mainBlockCount * resplitParam_.mainBlockSize * coreNum_;
        uint64_t tailSize = tailSizeOri / C0_SIZE;

        if (tailSizeOri > coreNum_ * static_cast<uint64_t>(weight_quant_tiling_detail::BASIC_BLOCK_BASE_N)) {
            // When tail is too large, further resplit by factor=2.
            constexpr uint64_t resplitFactor = 2;
            resplitParam_.firstTailBlockSize = static_cast<uint16_t>(tailSize / (coreNum_ * resplitFactor));
            resplitParam_.secondTailBlockSize = static_cast<uint16_t>(resplitParam_.firstTailBlockSize + 1U);
            resplitParam_.secondTailBlockCount = static_cast<uint16_t>(tailSize % (coreNum_ * resplitFactor));
            resplitParam_.firstTailBlockCount =
                static_cast<uint16_t>(coreNum_ * resplitFactor - resplitParam_.secondTailBlockCount);
        } else {
            resplitParam_.firstTailBlockSize = static_cast<uint16_t>(tailSize / coreNum_);
            resplitParam_.secondTailBlockSize = static_cast<uint16_t>(resplitParam_.firstTailBlockSize + 1U);
            resplitParam_.secondTailBlockCount = static_cast<uint16_t>(tailSize % coreNum_);
            resplitParam_.firstTailBlockCount = static_cast<uint16_t>(coreNum_ - resplitParam_.secondTailBlockCount);
        }

        resplitParam_.firstTailBlockSize *= static_cast<uint16_t>(C0_SIZE);
        resplitParam_.secondTailBlockSize *= static_cast<uint16_t>(C0_SIZE);
    }

    void CalcNoFullNumBlocksResplitTiling()
    {
        resplitParam_.mainBlockSize = weight_quant_tiling_detail::BASIC_BLOCK_BASE_N;
        // This scenario cannot keep main blocks evenly split across cores.
        resplitParam_.mainBlockCount = 0UL;

        uint64_t mainBlkCount = CeilDiv<uint64_t>(nSize_, weight_quant_tiling_detail::BASIC_BLOCK_BASE_N);
        if (groupNum_ * mainBlkCount >= coreNum_) {
            // Accumulate enough basic blocks in group dimension, then split by mainBlkCount.
            resplitParam_.firstTailBlockSize = Align<uint64_t>(CeilDiv<uint64_t>(nSize_, mainBlkCount), C0_SIZE);
            resplitParam_.firstTailBlockCount = static_cast<uint16_t>(mainBlkCount);
            resplitParam_.secondTailBlockSize = 0;
            resplitParam_.secondTailBlockCount = 0;
            return;
        }

        uint64_t taskNum = std::max(1UL, nSize_ / weight_quant_tiling_detail::BASIC_BLOCK_BASE_N_MIN);
        cubeNumBlocksN_ = static_cast<uint8_t>(taskNum);
        resplitParam_.firstTailBlockSize = static_cast<uint16_t>(nSize_ / C0_SIZE / taskNum);
        resplitParam_.secondTailBlockSize = static_cast<uint16_t>(resplitParam_.firstTailBlockSize + 1U);
        resplitParam_.secondTailBlockCount = static_cast<uint16_t>(nSize_ / C0_SIZE % taskNum);
        resplitParam_.firstTailBlockCount = static_cast<uint16_t>(taskNum - resplitParam_.secondTailBlockCount);
        resplitParam_.firstTailBlockSize *= static_cast<uint16_t>(C0_SIZE);
        resplitParam_.secondTailBlockSize *= static_cast<uint16_t>(C0_SIZE);
    }

private:
    uint32_t groupNum_ = 0;
    uint64_t kSize_ = 0;
    uint64_t nSize_ = 0;

    uint8_t cubeNumBlocksN_ = 0;
    uint32_t coreNum_ = 1;
    TailBlockResplitParam resplitParam_ = {};

    static constexpr uint64_t C0_SIZE = 32;
};
