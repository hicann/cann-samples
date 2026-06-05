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
 * \file weight_quant_matmul_mxfp8fp4_tiling_swat.h
 * \brief SWAT-derived host tiling for the MXA8W4 sample.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "host_utils/common_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "utils/constant.h"
#include "weight_quant_matmul_mxfp8fp4_tiling_data.h"

class WeightQuantMatmulMxfp8Fp4SwatTiling {
public:
    explicit WeightQuantMatmulMxfp8Fp4SwatTiling(uint64_t targetL1BufferNum = DB_SIZE)
        : targetL1BufferNum_(targetL1BufferNum)
    {
        CHECK_COND(
            targetL1BufferNum_ == DB_SIZE || targetL1BufferNum_ == L1_FOUR_BUFFER,
            "MXFP8FP4 SWAT tiling target L1 buffer count must be 2 or 4.");
    }

    void GetTilingData(uint64_t m, uint64_t n, uint64_t k, WeightQuantMatmulMxfp8Fp4TilingData& tilingData)
    {
        CHECK_COND(m > 0U && n > 0U && k > 0U, "m, n, and k must be greater than zero.");
        CHECK_COND((k % MX_K_ALIGN_SIZE) == 0U, "k must be a multiple of 64.");
        CHECK_COND((n % BLOCK_CUBE) == 0U, "NZ packed weight path expects n to be a multiple of 16.");
        CHECK_COND(
            m <= std::numeric_limits<uint32_t>::max() && n <= std::numeric_limits<uint32_t>::max() &&
                k <= std::numeric_limits<uint32_t>::max(),
            "m, n, and k must not exceed UINT32_MAX.");

        InitPlatformInfo();
        shape_ = {m, n, k};
        runInfo_ = {};
        CalcBasicBlock();
        OptimizeEdgeBasicBlock();
        CalcTailBasicBlock();
        CalcPathSpecificL1();
        CHECK_COND(ValidateTilingResult(), "failed to find a valid MXFP8FP4 SWAT-derived tiling.");

        BuildTilingData(tilingData);
        Print(tilingData);
    }

private:
    struct PlatformParam {
        uint64_t aicNum{0UL};
        uint64_t ubSize{0UL};
        uint64_t l1Size{0UL};
        uint64_t l0aSize{0UL};
        uint64_t l0bSize{0UL};
        uint64_t l0cSize{0UL};
    };

    struct ShapeParam {
        uint64_t m{0UL};
        uint64_t n{0UL};
        uint64_t k{0UL};
    };

    struct RunInfo {
        uint64_t baseM{0UL};
        uint64_t baseN{0UL};
        uint64_t baseK{0UL};
        uint64_t tileShapeKL1{0UL};
        uint64_t tileShapeScaleKL1{0UL};
        uint64_t nBubSize{0UL};
        uint64_t kBubSize{0UL};
        uint64_t mBlockCnt{0UL};
        uint64_t nBlockCnt{0UL};
        uint64_t totalBlockCnt{0UL};
        uint64_t tailBlockCnt{0UL};
        uint64_t mTailSize{0UL};
        uint64_t nTailSize{0UL};
        uint64_t mTailTile{1UL};
        uint64_t nTailTile{1UL};
        uint64_t mBaseTailSplitCnt{1UL};
        uint64_t nBaseTailSplitCnt{1UL};
        uint64_t mTailMain{0UL};
        uint64_t nTailMain{0UL};
    };

    void InitPlatformInfo()
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
        platform_.aicNum = ascendcPlatform->GetCoreNumAic();
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, platform_.ubSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, platform_.l1Size);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platform_.l0aSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platform_.l0bSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platform_.l0cSize);

        platform_.ubSize = platform_.ubSize == 0UL ? DEFAULT_UB_SIZE : platform_.ubSize;
        platform_.l1Size = platform_.l1Size == 0UL ? DEFAULT_L1_SIZE : platform_.l1Size;
        platform_.l0aSize = platform_.l0aSize == 0UL ? DEFAULT_L0A_SIZE : platform_.l0aSize;
        platform_.l0bSize = platform_.l0bSize == 0UL ? DEFAULT_L0B_SIZE : platform_.l0bSize;
        platform_.l0cSize = platform_.l0cSize == 0UL ? DEFAULT_L0C_SIZE : platform_.l0cSize;
        CHECK_COND(platform_.aicNum > 0UL, "failed to query a valid AI Core count for SWAT tiling.");
    }

    void CalcBasicBlock()
    {
        runInfo_.baseM = Align(std::min(shape_.m, BASIC_BLOCK_SIZE_256), BLOCK_CUBE);
        runInfo_.baseN = Align(std::min(shape_.n, BASIC_BLOCK_SIZE_256), BLOCK_CUBE);
        runInfo_.baseK = Align(std::min(shape_.k, BASIC_BLOCK_SIZE_128), MX_K_ALIGN_SIZE);

        uint64_t blockNum = CeilDiv(shape_.m, runInfo_.baseM) * CeilDiv(shape_.n, runInfo_.baseN);
        if (blockNum < platform_.aicNum) {
            AdjustBasicBlock();
        }

        CHECK_COND(
            runInfo_.baseM != 0UL && runInfo_.baseN != 0UL && runInfo_.baseK != 0UL,
            "Failed to derive a valid SWAT base shape: baseM, baseN, and baseK must be non-zero.");
        CHECK_COND(
            IsL0Feasible(runInfo_.baseM, runInfo_.baseN, runInfo_.baseK),
            "Failed to derive a SWAT base shape that satisfies L0A/L0B/L0C capacity.");

        runInfo_.mBlockCnt = CeilDiv(shape_.m, runInfo_.baseM);
        runInfo_.nBlockCnt = CeilDiv(shape_.n, runInfo_.baseN);
        runInfo_.totalBlockCnt = runInfo_.mBlockCnt * runInfo_.nBlockCnt;
        runInfo_.tailBlockCnt = runInfo_.totalBlockCnt % platform_.aicNum;
        runInfo_.mTailSize = shape_.m - (runInfo_.mBlockCnt - 1UL) * runInfo_.baseM;
        runInfo_.nTailSize = shape_.n - (runInfo_.nBlockCnt - 1UL) * runInfo_.baseN;
    }

    void AdjustBasicBlock()
    {
        uint64_t mMaxTile = CeilDiv(shape_.m, BLOCK_CUBE);
        uint64_t nMaxTile = CeilDiv(shape_.n, BLOCK_CUBE);
        uint64_t tempBaseM = runInfo_.baseM;
        uint64_t tempBaseN = runInfo_.baseN;

        uint64_t mCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.m, runInfo_.baseM));
        uint64_t nCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.n, runInfo_.baseN));
        if (mMaxTile > nMaxTile) {
            tempBaseN = Align(CeilDiv(shape_.n, nCnt), BLOCK_CUBE);
            nCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.n, tempBaseN));
            mCnt = std::max<uint64_t>(1UL, platform_.aicNum / nCnt);
            tempBaseM = Align(CeilDiv(shape_.m, mCnt), BLOCK_CUBE);
        } else {
            tempBaseM = Align(CeilDiv(shape_.m, mCnt), BLOCK_CUBE);
            mCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.m, tempBaseM));
            nCnt = std::max<uint64_t>(1UL, platform_.aicNum / mCnt);
            tempBaseN = Align(CeilDiv(shape_.n, nCnt), BLOCK_CUBE);
        }

        mCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.m, tempBaseM));
        nCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.n, tempBaseN));
        while (tempBaseN > tempBaseM * BASEM_BASEN_RATIO && nCnt < platform_.aicNum / NUM_TWO &&
               tempBaseN != BLOCK_CUBE) {
            nCnt *= NUM_TWO;
            mCnt = std::max<uint64_t>(1UL, platform_.aicNum / nCnt);
            tempBaseM = Align(CeilDiv(shape_.m, mCnt), BLOCK_CUBE);
            tempBaseN = Align(CeilDiv(shape_.n, nCnt), BLOCK_CUBE);
            mCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.m, tempBaseM));
            nCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.n, tempBaseN));
        }
        while (tempBaseM >= tempBaseN * BASEM_BASEN_RATIO && mCnt < platform_.aicNum / NUM_TWO &&
               tempBaseM != BLOCK_CUBE) {
            mCnt *= NUM_TWO;
            nCnt = std::max<uint64_t>(1UL, platform_.aicNum / mCnt);
            tempBaseM = Align(CeilDiv(shape_.m, mCnt), BLOCK_CUBE);
            tempBaseN = Align(CeilDiv(shape_.n, nCnt), BLOCK_CUBE);
            mCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.m, tempBaseM));
            nCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.n, tempBaseN));
        }

        uint64_t kAlignValue = Align(shape_.k, BASIC_BLOCK_SIZE_128);
        uint64_t kMaxValue = (platform_.l0aSize / DB_SIZE) / std::max(tempBaseM, tempBaseN);
        kMaxValue = FloorAlign(kMaxValue, BASIC_BLOCK_SIZE_128);
        if (kMaxValue >= BASIC_BLOCK_SIZE_128 && IsL0Feasible(tempBaseM, tempBaseN, std::min(kAlignValue, kMaxValue))) {
            runInfo_.baseM = tempBaseM;
            runInfo_.baseN = tempBaseN;
            runInfo_.baseK = std::min(kAlignValue, kMaxValue);
            runInfo_.baseK =
                runInfo_.baseK > BASEK_LIMIT ? Align(runInfo_.baseK / NUM_TWO, BASIC_BLOCK_SIZE_256) : runInfo_.baseK;
        }
    }

    void OptimizeEdgeBasicBlock()
    {
        if (runInfo_.mBlockCnt == 1UL && runInfo_.nBlockCnt == 1UL) {
            return;
        }

        bool isInnerAxisAlign = (shape_.k * DATA_SIZE_UINT8) % MTE2_CACHELINE_SIZE == 0UL;
        uint64_t mTailSize = shape_.m % runInfo_.baseM;
        if (runInfo_.mBlockCnt > 1UL && mTailSize > 0UL && isInnerAxisAlign) {
            uint64_t baseTailCntMax = std::min((runInfo_.baseM - mTailSize) / BLOCK_CUBE, runInfo_.mBlockCnt);
            uint64_t windowSize = std::min(TAIL_WINDOW_LEN, runInfo_.mBlockCnt);
            uint64_t mainWindowNum = runInfo_.mBlockCnt / windowSize - 1UL;
            uint64_t tailWindowSize = runInfo_.mBlockCnt - mainWindowNum * windowSize;
            uint64_t perfRes = (mainWindowNum + 1UL) * runInfo_.baseM;
            uint64_t mergeWindowNum = 1UL;
            for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax;
                 mergeLen += windowSize, ++mergeWindowNum) {
                uint64_t newTailMain =
                    Align(CeilDiv(mergeLen * runInfo_.baseM + mTailSize, mergeLen + 1UL), BLOCK_CUBE);
                uint64_t curPerf =
                    (mainWindowNum + 1UL - mergeWindowNum) * runInfo_.baseM + mergeWindowNum * newTailMain;
                if (curPerf <= perfRes) {
                    perfRes = curPerf;
                    runInfo_.mTailMain = newTailMain;
                    runInfo_.mBaseTailSplitCnt = mergeLen + 1UL;
                }
            }
        }

        uint64_t nTailSize = shape_.n % runInfo_.baseN;
        if (runInfo_.nBlockCnt > 1UL && nTailSize > 0UL && isInnerAxisAlign) {
            uint64_t baseTailCntMax = std::min((runInfo_.baseN - nTailSize) / BLOCK_CUBE, runInfo_.nBlockCnt);
            uint64_t windowSize = std::min(TAIL_WINDOW_LEN, runInfo_.nBlockCnt);
            uint64_t mainWindowNum = runInfo_.nBlockCnt / windowSize - 1UL;
            uint64_t tailWindowSize = runInfo_.nBlockCnt - mainWindowNum * windowSize;
            uint64_t perfRes = (mainWindowNum + 1UL) * runInfo_.baseN;
            uint64_t mergeWindowNum = 1UL;
            for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax;
                 mergeLen += windowSize, ++mergeWindowNum) {
                uint64_t newTailMain =
                    Align(CeilDiv(mergeLen * runInfo_.baseN + nTailSize, mergeLen + 1UL), BLOCK_CUBE);
                uint64_t curPerf =
                    (mainWindowNum + 1UL - mergeWindowNum) * runInfo_.baseN + mergeWindowNum * newTailMain;
                if (curPerf <= perfRes) {
                    perfRes = curPerf;
                    runInfo_.nTailMain = newTailMain;
                    runInfo_.nBaseTailSplitCnt = mergeLen + 1UL;
                }
            }
        }
    }

    void CalcTailBasicBlock()
    {
        if (runInfo_.tailBlockCnt == 0UL) {
            return;
        }

        uint64_t mTile = 1UL;
        uint64_t nTile = 1UL;
        uint64_t preSplit = 1UL;
        uint64_t secSplit = 1UL;
        uint64_t& preSplitValid = runInfo_.mTailSize >= runInfo_.nTailSize ? mTile : nTile;
        uint64_t& secSplitValid = runInfo_.mTailSize >= runInfo_.nTailSize ? nTile : mTile;
        uint64_t mTileMax = CeilDiv(runInfo_.baseM, BLOCK_CUBE);
        uint64_t nTileMax = CeilDiv(runInfo_.baseN, BLOCK_CUBE);
        uint64_t preSplitMax = runInfo_.mTailSize >= runInfo_.nTailSize ? mTileMax : nTileMax;
        uint64_t secSplitMax = runInfo_.mTailSize >= runInfo_.nTailSize ? nTileMax : mTileMax;
        bool splitMFirst = runInfo_.mTailSize >= runInfo_.nTailSize;
        bool updated = true;
        while (updated) {
            updated = false;
            uint64_t currentUsedCoreNum = CalUsedCoreNum(mTile, nTile);
            uint64_t preCandidateM = splitMFirst ? preSplit + 1UL : secSplit;
            uint64_t preCandidateN = splitMFirst ? secSplit : preSplit + 1UL;
            uint64_t preCandidateUsedCoreNum = CalUsedCoreNum(preCandidateM, preCandidateN);
            if (preSplit < preSplitMax && preCandidateUsedCoreNum <= platform_.aicNum &&
                preCandidateUsedCoreNum > currentUsedCoreNum) {
                preSplitValid = ++preSplit;
                updated = true;
                currentUsedCoreNum = preCandidateUsedCoreNum;
            }
            uint64_t secCandidateM = splitMFirst ? preSplit : secSplit + 1UL;
            uint64_t secCandidateN = splitMFirst ? secSplit + 1UL : preSplit;
            uint64_t secCandidateUsedCoreNum = CalUsedCoreNum(secCandidateM, secCandidateN);
            if (secSplit < secSplitMax && secCandidateUsedCoreNum <= platform_.aicNum &&
                secCandidateUsedCoreNum > currentUsedCoreNum) {
                secSplitValid = ++secSplit;
                updated = true;
            }
        }

        runInfo_.mTailTile = mTile;
        runInfo_.nTailTile = nTile;
    }

    void CalcPathSpecificL1()
    {
        // A and converted B use symmetric K-L1 depth; target policy fixes the L1 buffer count.
        uint64_t maxStepK = std::min(STEPK_THRESHOLD, CeilDiv(shape_.k, runInfo_.baseK));
        for (uint64_t stepK = maxStepK; stepK >= 1UL; --stepK) {
            uint64_t kBl1Size = std::min(shape_.k, stepK * runInfo_.baseK);
            uint64_t nBubSize = std::min(shape_.n, runInfo_.baseN);
            // The device prologue keeps N whole and lets the two AIV subblocks split only the K range.
            uint64_t kBubSize = FindKOnlyBubSize(nBubSize, kBl1Size);
            if (!IsBubTilingValid(nBubSize, kBubSize)) {
                if (stepK == 1UL) {
                    break;
                }
                continue;
            }

            uint64_t maxScaleFactor = CalcMaxScaleFactor(stepK);
            for (uint64_t scaleFactor = maxScaleFactor; scaleFactor >= 1UL; --scaleFactor) {
                uint64_t tileShapeKL1 = stepK * runInfo_.baseK;
                uint64_t tileShapeScaleKL1 = tileShapeKL1 * scaleFactor;
                if (IsL1Feasible(tileShapeKL1, tileShapeScaleKL1)) {
                    runInfo_.tileShapeKL1 = tileShapeKL1;
                    runInfo_.tileShapeScaleKL1 = tileShapeScaleKL1;
                    runInfo_.nBubSize = nBubSize;
                    runInfo_.kBubSize = kBubSize;
                    return;
                }
                if (scaleFactor == 1UL) {
                    break;
                }
            }
            if (stepK == 1UL) {
                break;
            }
        }
        CHECK_COND(false, "MXFP8FP4 SWAT tiling cannot satisfy L1 and K-only UB capacity constraints.");
    }

    uint64_t CalcMaxScaleFactor(uint64_t stepK) const
    {
        uint64_t kL1Size = stepK * runInfo_.baseK;
        return std::max<uint64_t>(1UL, std::min(SCALER_FACTOR_MAX, CeilDiv(shape_.k, kL1Size)));
    }

    bool ValidateTilingResult() const
    {
        uint64_t kBl1Size = std::min(shape_.k, runInfo_.tileShapeKL1);
        uint64_t minK = Align(CeilDiv(kBl1Size, NUM_TWO), MX_K_ALIGN_SIZE);
        uint64_t maxK = FloorAlign(kBl1Size, MX_K_ALIGN_SIZE);
        return runInfo_.tileShapeKL1 > 0UL && runInfo_.tileShapeScaleKL1 > 0UL &&
               IsL0Feasible(runInfo_.baseM, runInfo_.baseN, runInfo_.baseK) &&
               IsL1Feasible(runInfo_.tileShapeKL1, runInfo_.tileShapeScaleKL1) && runInfo_.kBubSize == minK &&
               runInfo_.kBubSize <= maxK && IsBubTilingValid(runInfo_.nBubSize, runInfo_.kBubSize);
    }

    bool IsL0Feasible(uint64_t baseM, uint64_t baseN, uint64_t baseK) const
    {
        uint64_t a2Size = baseM * baseK * DB_SIZE;
        uint64_t b2Size = baseN * baseK * DB_SIZE;
        uint64_t cSize = baseM * baseN * DATA_SIZE_FP32;
        return a2Size <= platform_.l0aSize && b2Size <= platform_.l0bSize && cSize <= platform_.l0cSize;
    }

    bool IsL1Feasible(uint64_t tileShapeKL1, uint64_t tileShapeScaleKL1) const
    {
        uint64_t aL1Size = runInfo_.baseM * tileShapeKL1 * DATA_SIZE_UINT8;
        uint64_t bL1Size = runInfo_.baseN * tileShapeKL1 * DATA_SIZE_UINT8;
        uint64_t scaleAL1Size = runInfo_.baseM * tileShapeScaleKL1 * DATA_SIZE_UINT8 / MX_GROUP_SIZE;
        uint64_t scaleBL1Size = runInfo_.baseN * tileShapeScaleKL1 * DATA_SIZE_UINT8 / MX_GROUP_SIZE;
        if (targetL1BufferNum_ == L1_FOUR_BUFFER) {
            // Four-buffer mode is physically arranged as two independent half-L1 groups.
            uint64_t halfL1Limit = std::min(platform_.l1Size / NUM_TWO, L1_HALF_SIZE);
            uint64_t halfL1Use = DB_SIZE * (aL1Size + bL1Size) + scaleAL1Size + scaleBL1Size;
            return halfL1Use <= halfL1Limit && halfL1Use * NUM_TWO <= platform_.l1Size;
        }
        // Two-buffer mode packs all A/B/scale slots in one compact L1 address space.
        uint64_t compactL1Use =
            targetL1BufferNum_ * (aL1Size + bL1Size) + DOUBLE_BUFFER_COUNT * (scaleAL1Size + scaleBL1Size);
        return compactL1Use <= platform_.l1Size;
    }

    bool IsBubTilingValid(uint64_t nBubSize, uint64_t kBubSize) const
    {
        return nBubSize > 0UL && kBubSize > 0UL && kBubSize % MX_K_ALIGN_SIZE == 0UL &&
               GetBubSize(targetL1BufferNum_, nBubSize, kBubSize) <= platform_.ubSize;
    }

    void BuildTilingData(WeightQuantMatmulMxfp8Fp4TilingData& tilingData) const
    {
        tilingData = {};
        tilingData.m = static_cast<uint32_t>(shape_.m);
        tilingData.n = static_cast<uint32_t>(shape_.n);
        tilingData.k = static_cast<uint32_t>(shape_.k);
        tilingData.baseM = static_cast<uint32_t>(runInfo_.baseM);
        tilingData.baseN = static_cast<uint32_t>(runInfo_.baseN);
        tilingData.baseK = static_cast<uint32_t>(runInfo_.baseK);
        tilingData.tileShapeKL1 = static_cast<uint32_t>(runInfo_.tileShapeKL1);
        tilingData.tileShapeScaleKL1 = static_cast<uint32_t>(runInfo_.tileShapeScaleKL1);
        tilingData.usedCoreNum = static_cast<uint32_t>(
            (runInfo_.totalBlockCnt > 1UL || runInfo_.tailBlockCnt == 0UL) ?
                platform_.aicNum :
                CalUsedCoreNum(runInfo_.mTailTile, runInfo_.nTailTile));
        tilingData.cubeNumBlocksM = static_cast<uint32_t>(runInfo_.mBlockCnt);
        tilingData.cubeNumBlocksN = static_cast<uint32_t>(runInfo_.nBlockCnt);
        tilingData.iterateOrder = ORDER_N;
        tilingData.mTailTile = static_cast<uint32_t>(runInfo_.mTailTile);
        tilingData.nTailTile = static_cast<uint32_t>(runInfo_.nTailTile);
        tilingData.mBaseTailSplitCnt = static_cast<uint32_t>(runInfo_.mBaseTailSplitCnt);
        tilingData.nBaseTailSplitCnt = static_cast<uint32_t>(runInfo_.nBaseTailSplitCnt);
        tilingData.mTailMain = static_cast<uint32_t>(runInfo_.mTailMain);
        tilingData.nTailMain = static_cast<uint32_t>(runInfo_.nTailMain);
        tilingData.nBubSize = static_cast<uint32_t>(runInfo_.nBubSize);
        tilingData.kBubSize = static_cast<uint32_t>(runInfo_.kBubSize);
    }

    uint64_t FindKOnlyBubSize(uint64_t nBubSize, uint64_t kBl1Size) const
    {
        // AIC waits for both AIV subblocks; use the closest 64-aligned half split instead of the largest UB fit.
        uint64_t minK = Align(CeilDiv(kBl1Size, NUM_TWO), MX_K_ALIGN_SIZE);
        uint64_t maxK = FloorAlign(kBl1Size, MX_K_ALIGN_SIZE);
        if (maxK < minK) {
            return 0UL;
        }
        return IsBubTilingValid(nBubSize, minK) ? minK : 0UL;
    }

    static uint64_t GetBubSize(uint64_t bufferNum, uint64_t nDimSize, uint64_t kDimSize)
    {
        uint64_t nDimAlign = Align(nDimSize, BLOCK_CUBE);
        uint64_t kDimAlign = Align(kDimSize, MX_K_ALIGN_SIZE);
        uint64_t sizeWeightOut = bufferNum * DATA_SIZE_UINT8 * nDimAlign * kDimAlign;
        uint64_t sizeWeightIn = bufferNum * DATA_SIZE_UINT8 * kDimSize * nDimSize / INT4_PACK_NUM;
        return sizeWeightIn + sizeWeightOut;
    }

    uint64_t CalUsedCoreNum(uint64_t mTile, uint64_t nTile) const
    {
        uint64_t usedCoreNum = 0UL;
        uint64_t baseRoundTileNum = runInfo_.totalBlockCnt - runInfo_.tailBlockCnt;
        for (uint64_t tailIdx = 0UL; tailIdx < runInfo_.tailBlockCnt; ++tailIdx) {
            uint64_t mTileIdx = 0UL;
            uint64_t nTileIdx = 0UL;
            GetLogicalTileCoord(baseRoundTileNum + tailIdx, mTileIdx, nTileIdx);
            usedCoreNum += CalcValidSplitCount(GetSingleCoreM(mTileIdx), GetSingleCoreN(nTileIdx), mTile, nTile);
        }
        return usedCoreNum;
    }

    uint64_t CalcValidSplitCount(uint64_t singleCoreM, uint64_t singleCoreN, uint64_t mTile, uint64_t nTile) const
    {
        uint64_t singleCoreMSplit = Align(CeilDiv(singleCoreM, mTile), BLOCK_CUBE);
        uint64_t singleCoreNSplit = Align(CeilDiv(singleCoreN, nTile), BLOCK_CUBE);
        uint64_t validM = std::min(mTile, CeilDiv(singleCoreM, singleCoreMSplit));
        uint64_t validN = std::min(nTile, CeilDiv(singleCoreN, singleCoreNSplit));
        return validM * validN;
    }

    void GetLogicalTileCoord(uint64_t tileIdx, uint64_t& mTileIdx, uint64_t& nTileIdx) const
    {
        uint64_t mCoreNum = std::min(TAIL_WINDOW_LEN, runInfo_.mBlockCnt);
        uint64_t mainRow = runInfo_.mBlockCnt / mCoreNum - 1UL;
        uint64_t mTailCoreNum = runInfo_.mBlockCnt - mCoreNum * mainRow;
        uint64_t rowIdx = tileIdx / (mCoreNum * runInfo_.nBlockCnt);
        if (rowIdx < mainRow) {
            uint64_t localTileIdx = tileIdx - rowIdx * mCoreNum * runInfo_.nBlockCnt;
            mTileIdx = rowIdx * mCoreNum + localTileIdx % mCoreNum;
            nTileIdx = (localTileIdx / mCoreNum) % runInfo_.nBlockCnt;
        } else {
            rowIdx = mainRow;
            uint64_t tailIdx = tileIdx - mainRow * mCoreNum * runInfo_.nBlockCnt;
            mTileIdx = mainRow * mCoreNum + tailIdx % mTailCoreNum;
            nTileIdx = (tailIdx / mTailCoreNum) % runInfo_.nBlockCnt;
        }
        if (rowIdx & 1UL) {
            nTileIdx = runInfo_.nBlockCnt - 1UL - nTileIdx;
        }
    }

    uint64_t GetSingleCoreM(uint64_t mTileIdx) const
    {
        uint64_t mBaseNormCnt = runInfo_.mBlockCnt - runInfo_.mBaseTailSplitCnt;
        if (mTileIdx >= mBaseNormCnt) {
            uint64_t mMergeSize = shape_.m - mBaseNormCnt * runInfo_.baseM;
            uint64_t mBaseTailMain = runInfo_.mBaseTailSplitCnt == 1UL ? mMergeSize : runInfo_.mTailMain;
            uint64_t mBaseTailLast = mMergeSize - (runInfo_.mBaseTailSplitCnt - 1UL) * mBaseTailMain;
            return mTileIdx < runInfo_.mBlockCnt - 1UL ? mBaseTailMain : mBaseTailLast;
        }
        return runInfo_.baseM;
    }

    uint64_t GetSingleCoreN(uint64_t nTileIdx) const
    {
        uint64_t nBaseNormCnt = runInfo_.nBlockCnt - runInfo_.nBaseTailSplitCnt;
        if (nTileIdx >= nBaseNormCnt) {
            uint64_t nMergeSize = shape_.n - nBaseNormCnt * runInfo_.baseN;
            uint64_t nBaseTailMain = runInfo_.nBaseTailSplitCnt == 1UL ? nMergeSize : runInfo_.nTailMain;
            uint64_t nBaseTailLast = nMergeSize - (runInfo_.nBaseTailSplitCnt - 1UL) * nBaseTailMain;
            return nTileIdx < runInfo_.nBlockCnt - 1UL ? nBaseTailMain : nBaseTailLast;
        }
        return runInfo_.baseN;
    }

    static uint64_t CeilDiv(uint64_t lhs, uint64_t rhs)
    {
        return rhs == 0UL ? 0UL : (lhs + rhs - 1UL) / rhs;
    }

    static uint64_t Align(uint64_t value, uint64_t align)
    {
        return align == 0UL ? value : CeilDiv(value, align) * align;
    }

    static uint64_t FloorAlign(uint64_t value, uint64_t align)
    {
        return align == 0UL ? value : value / align * align;
    }

    void Print(const WeightQuantMatmulMxfp8Fp4TilingData& tilingData) const
    {
        printf("[WeightQuantMatmul MXA8W4 SWAT-Derived Tiling]\n");
        printf("  usedCoreNum        : %u\n", tilingData.usedCoreNum);
        printf("  m/n/k              : %u/%u/%u\n", tilingData.m, tilingData.n, tilingData.k);
        printf("  cubeNumBlocks M/N  : %u/%u\n", tilingData.cubeNumBlocksM, tilingData.cubeNumBlocksN);
        printf("  baseM/baseN/baseK  : %u/%u/%u\n", tilingData.baseM, tilingData.baseN, tilingData.baseK);
        printf("  tileShapeKL1       : %u\n", tilingData.tileShapeKL1);
        printf("  tileShapeScaleKL1  : %u\n", tilingData.tileShapeScaleKL1);
        printf("  iterateOrder       : %u\n", tilingData.iterateOrder);
        printf("  scheduler          : swat-derived\n");
        printf("  mTailTile          : %u\n", tilingData.mTailTile);
        printf("  nTailTile          : %u\n", tilingData.nTailTile);
        printf("  mBaseTailSplitCnt  : %u\n", tilingData.mBaseTailSplitCnt);
        printf("  nBaseTailSplitCnt  : %u\n", tilingData.nBaseTailSplitCnt);
        printf("  mTailMain          : %u\n", tilingData.mTailMain);
        printf("  nTailMain          : %u\n", tilingData.nTailMain);
        printf("  nBubSize/kBubSize  : %u/%u\n", tilingData.nBubSize, tilingData.kBubSize);
    }

private:
    ShapeParam shape_;
    PlatformParam platform_;
    RunInfo runInfo_;
    uint64_t targetL1BufferNum_{DB_SIZE};

    static constexpr uint64_t BLOCK_CUBE = 16UL;
    static constexpr uint64_t NUM_TWO = 2UL;
    static constexpr uint64_t STEPK_THRESHOLD = 4UL;
    static constexpr uint64_t BASEM_BASEN_RATIO = 2UL;
    static constexpr uint64_t BASEK_LIMIT = 4095UL;
    static constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
    static constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
    static constexpr uint64_t MX_K_ALIGN_SIZE = 64UL;
    static constexpr uint64_t MX_GROUP_SIZE = 32UL;
    static constexpr uint64_t MTE2_CACHELINE_SIZE = 128UL;
    static constexpr uint64_t TAIL_WINDOW_LEN = 4UL;
    static constexpr uint32_t ORDER_N = 1U;
    static constexpr uint64_t DATA_SIZE_UINT8 = 1UL;
    static constexpr uint64_t DATA_SIZE_FP32 = 4UL;
    static constexpr uint64_t INT4_PACK_NUM = 2UL;
    static constexpr uint64_t DEFAULT_UB_SIZE = 248UL * 1024UL;
    static constexpr uint64_t DEFAULT_L1_SIZE = 512UL * 1024UL;
    static constexpr uint64_t DEFAULT_L0A_SIZE = 64UL * 1024UL;
    static constexpr uint64_t DEFAULT_L0B_SIZE = 64UL * 1024UL;
    static constexpr uint64_t DEFAULT_L0C_SIZE = 256UL * 1024UL;
};
