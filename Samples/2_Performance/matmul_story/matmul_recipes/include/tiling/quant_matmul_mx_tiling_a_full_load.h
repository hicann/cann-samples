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
 * \file quant_matmul_mx_tiling_a_full_load.h
 * \brief SWAT tiling specialization for the MX A-full-load path.
 */

#ifndef QUANT_MATMUL_MX_TILING_A_FULL_LOAD_H
#define QUANT_MATMUL_MX_TILING_A_FULL_LOAD_H

#include <algorithm>

#include "quant_matmul_tiling_base.h"

template <DataType aDataType, DataType bDataType>
class QuantMatmulTilingAFullLoad : public QuantMatmulTilingBase<aDataType, bDataType> {
public:
    QuantMatmulTilingAFullLoad() = default;
    ~QuantMatmulTilingAFullLoad() override = default;

private:
    using Base = QuantMatmulTilingBase<aDataType, bDataType>;
    using Base::args_;
    using Base::platformInfo_;
    using Base::runInfo_;

protected:
    const char* TilingName() const override
    {
        return "a_full_load";
    }

    void DoOpTiling(QuantMatmulTilingData& tilingData) override
    {
        // The A-full-load path validates eligibility before computing its
        // L1 layout because later calculations assume A stays resident in L1.
        CalcBasicBlock();
        OptimizeEdgeBasicBlock();
        CanUseAFullLoad();
        CalcTailBasicBlock();
        CalcPathSpecificL1();

        uint32_t scaleKL1 = CalcScaleKL1();
        uint8_t nBufferNum = CalculateNBufferNum(scaleKL1);
        BuildTilingData(tilingData, scaleKL1, nBufferNum);
    }

private:
    uint32_t CalcScaleKL1() const
    {
        // Even in A-full-load mode, scale reuse is still capped by the smaller
        // reusable K window exposed by the A and B sides.
        return static_cast<uint32_t>(std::min(
            runInfo_.scaleFactorA * runInfo_.stepKa * runInfo_.baseK,
            runInfo_.scaleFactorB * runInfo_.stepKb * runInfo_.baseK));
    }

    void BuildTilingData(QuantMatmulTilingData& tilingData, uint32_t scaleKL1, uint8_t nBufferNum) const
    {
        // Flatten the host-side search result into the POD payload consumed by
        // the launcher and device kernel.
        tilingData = {};
        tilingData.m = static_cast<uint32_t>(args_.m);
        tilingData.n = static_cast<uint32_t>(args_.n);
        tilingData.k = static_cast<uint32_t>(args_.k);
        tilingData.baseM = static_cast<uint32_t>(runInfo_.baseM);
        tilingData.baseN = static_cast<uint32_t>(runInfo_.baseN);
        tilingData.baseK = static_cast<uint32_t>(runInfo_.baseK);
        tilingData.mTailTile = static_cast<uint32_t>(runInfo_.mTailTile);
        tilingData.nTailTile = static_cast<uint32_t>(runInfo_.nTailTile);
        tilingData.mBaseTailSplitCnt = static_cast<uint32_t>(runInfo_.mBaseTailSplitCnt);
        tilingData.nBaseTailSplitCnt = static_cast<uint32_t>(runInfo_.nBaseTailSplitCnt);
        tilingData.mTailMain = static_cast<uint32_t>(runInfo_.mTailMain);
        tilingData.nTailMain = static_cast<uint32_t>(runInfo_.nTailMain);
        tilingData.usedCoreNum =
            static_cast<uint32_t>((runInfo_.totalBlockCnt > 1UL || runInfo_.tailBlockCnt == 0UL)
                                      ? platformInfo_.aicNum
                                      : runInfo_.tailBlockCnt * runInfo_.mTailTile * runInfo_.nTailTile);
        tilingData.dbL0c = static_cast<uint8_t>(runInfo_.dbL0c);
        tilingData.scaleKL1 = scaleKL1;
        tilingData.stepK = static_cast<uint8_t>(std::min(runInfo_.stepKa, runInfo_.stepKb));
        tilingData.nBufferNum = nBufferNum;
    }

    void CalcTailBasicBlock()
    {
        // A-full-load keeps the M dimension unsplit and only fans out the N
        // edge tiles while extra AICs are still available.
        runInfo_.mTailTile = 1UL;
        uint64_t nTailTile = 1UL;
        if (runInfo_.tailBlockCnt != 0UL) {
            while (runInfo_.mTailTile * (nTailTile + 1UL) * runInfo_.tailBlockCnt <= platformInfo_.aicNum &&
                   runInfo_.baseN / (nTailTile + 1UL) >= BASIC_BLOCK_SIZE_16) {
                nTailTile += 1UL;
            }
        }
        runInfo_.nTailTile = nTailTile;
    }

    void CalcPathSpecificL1()
    {
        // Reserve resident space for the full A tile first, then use the
        // remaining L1 budget to determine B depth and scale reuse.
        runInfo_.stepKa = CeilDiv(args_.k, runInfo_.baseK);

        uint64_t aL1Size =
            GetSizeWithDataType<aDataType>(runInfo_.baseM * Align(args_.k, aDataType == DataType::FP4 ? FP4_C0_SIZE : FP8_C0_SIZE));
        runInfo_.scaleFactorA = 1U;
        uint64_t leftL1Size = platformInfo_.l1Size - aL1Size;
        uint64_t bL0Size = GetSizeWithDataType<bDataType>(runInfo_.baseN * runInfo_.baseK);
        uint64_t scaleAL1Size =
            runInfo_.baseM * Align(CeilDiv(args_.k, MX_GROUP_SIZE), TILING_MXFP_MULTI_BASE_SIZE);

        leftL1Size -= scaleAL1Size;
        runInfo_.stepKb = GetDepthB1AFullLoad(args_, runInfo_, leftL1Size);
        runInfo_.scaleFactorB =
            GetScaleFactorBAFullLoad(args_, runInfo_, leftL1Size - runInfo_.stepKb * DB_SIZE * bL0Size);
    }

    uint8_t CalculateNBufferNum(uint32_t scaleKL1) const
    {
        // In this path the full A tensor already occupies a fixed L1 region, so
        // only the remaining capacity can be traded between B ping-pong buffers.
        uint64_t stepK = std::min(runInfo_.stepKa, runInfo_.stepKb);
        uint64_t kL1 = stepK * runInfo_.baseK;
        uint64_t usedL1Size = GetSizeWithDataType<bDataType>(runInfo_.baseN * kL1) * L1_FOUR_BUFFER;
        usedL1Size += runInfo_.baseN * CeilDiv(static_cast<uint64_t>(scaleKL1), MX_GROUP_SIZE) * DB_SIZE;
        uint64_t scaleK = CeilDiv(args_.k, TILING_MXFP_DIVISOR_SIZE) * TILING_MXFP_MULTI_BASE_SIZE;
        uint64_t kAligned = Align(args_.k, TILING_MXFP_DIVISOR_SIZE);
        usedL1Size += GetSizeWithDataType<aDataType>(runInfo_.baseM * kAligned) + runInfo_.baseM * scaleK;
        return static_cast<uint8_t>(usedL1Size < platformInfo_.l1Size ? L1_FOUR_BUFFER : DB_SIZE);
    }

    void AdjustBasicBlock()
    {
        // Re-balance the initial M/N split when the first guess underutilizes
        // cores or produces a highly skewed tile.
        uint64_t mMaxtile = CeilDiv(args_.m, CUBE_BLOCK);
        uint64_t nMaxtile = CeilDiv(args_.n, CUBE_BLOCK);
        uint64_t tempBaseM = runInfo_.baseM;
        uint64_t tempBaseN = runInfo_.baseN;

        uint64_t mCnt = CeilDiv(args_.m, runInfo_.baseM);
        uint64_t nCnt = CeilDiv(args_.n, runInfo_.baseN);

        if (mMaxtile > nMaxtile) {
            tempBaseN = Align(CeilDiv(args_.n, nCnt), CUBE_BLOCK);
            nCnt = CeilDiv(args_.n, tempBaseN);
            mCnt = platformInfo_.aicNum / nCnt;
            tempBaseM = Align(CeilDiv(args_.m, mCnt), CUBE_BLOCK);
        } else {
            tempBaseM = Align(CeilDiv(args_.m, mCnt), CUBE_BLOCK);
            mCnt = CeilDiv(args_.m, tempBaseM);
            nCnt = platformInfo_.aicNum / mCnt;
            tempBaseN = Align(CeilDiv(args_.n, nCnt), CUBE_BLOCK);
        }

        while (tempBaseN > tempBaseM * BASEM_BASEN_RATIO && nCnt < platformInfo_.aicNum / NUM_TWO &&
               tempBaseN != CUBE_BLOCK) {
            nCnt = nCnt * NUM_TWO;
            mCnt = platformInfo_.aicNum / nCnt;
            tempBaseM = Align(CeilDiv(args_.m, mCnt), CUBE_BLOCK);
            tempBaseN = Align(CeilDiv(args_.n, nCnt), CUBE_BLOCK);
            mCnt = CeilDiv(args_.m, tempBaseM);
            nCnt = CeilDiv(args_.n, tempBaseN);
        }
        while (tempBaseM >= tempBaseN * BASEM_BASEN_RATIO && mCnt < platformInfo_.aicNum / NUM_TWO &&
               tempBaseM != CUBE_BLOCK) {
            mCnt = mCnt * NUM_TWO;
            nCnt = platformInfo_.aicNum / mCnt;
            tempBaseM = Align(CeilDiv(args_.m, mCnt), CUBE_BLOCK);
            tempBaseN = Align(CeilDiv(args_.n, nCnt), CUBE_BLOCK);
            mCnt = CeilDiv(args_.m, tempBaseM);
            nCnt = CeilDiv(args_.n, tempBaseN);
        }

        uint64_t kAlignValue = Align(args_.k, BASIC_BLOCK_SIZE_256);
        uint64_t kMaxValue = GetShapeWithDataType<aDataType>(platformInfo_.l0aSize / DB_SIZE) / std::max(tempBaseM, tempBaseN);
        kMaxValue = FloorAlign(kMaxValue, BASIC_BLOCK_SIZE_256);
        if (kMaxValue >= BASIC_BLOCK_SIZE_256) {
            runInfo_.baseM = tempBaseM;
            runInfo_.baseN = tempBaseN;
            runInfo_.baseK = std::min(kAlignValue, kMaxValue);
            runInfo_.baseK = runInfo_.baseK > BASEK_LIMIT ? Align(runInfo_.baseK / NUM_TWO, BASIC_BLOCK_SIZE_256)
                                                          : runInfo_.baseK;
        }
    }

    void CalcBasicBlock()
    {
        // Start from a 256-sized candidate tile, then refine it and capture
        // the tail statistics used by later scheduling decisions.
        runInfo_.baseM = Align(std::min(args_.m, BASIC_BLOCK_SIZE_256), CUBE_BLOCK);
        runInfo_.baseN = Align(std::min(args_.n, BASIC_BLOCK_SIZE_256), CUBE_BLOCK);
        runInfo_.baseK = Align(std::min(args_.k, aDataType == DataType::FP4 ? BASIC_BLOCK_SIZE_256 : BASIC_BLOCK_SIZE_128), TILING_MXFP_DIVISOR_SIZE);

        uint64_t blockNum = CeilDiv(args_.m, runInfo_.baseM) * CeilDiv(args_.n, runInfo_.baseN);
        if (blockNum < platformInfo_.aicNum) {
            AdjustBasicBlock();
        }
        CHECK_COND(
            runInfo_.baseM != 0UL && runInfo_.baseN != 0UL && runInfo_.baseK != 0UL,
            "Failed to derive a valid tiling base shape: baseM, baseN, and baseK must all be non-zero.");

        runInfo_.mBlockCnt = CeilDiv(args_.m, runInfo_.baseM);
        runInfo_.nBlockCnt = CeilDiv(args_.n, runInfo_.baseN);
        runInfo_.totalBlockCnt = runInfo_.mBlockCnt * runInfo_.nBlockCnt;
        runInfo_.tailBlockCnt = runInfo_.totalBlockCnt % platformInfo_.aicNum;
        runInfo_.mTailSize = args_.m - (runInfo_.mBlockCnt - 1UL) * runInfo_.baseM;
        runInfo_.nTailSize = args_.n - (runInfo_.nBlockCnt - 1UL) * runInfo_.baseN;
    }

    void OptimizeEdgeBasicBlock()
    {
        // Merge tiny M-edge tiles when K is cache-line aligned so the tail
        // block behaves more like the steady-state region.
        if (runInfo_.mBlockCnt == 1UL) {
            return;
        }
        uint64_t mTailSize = args_.m % runInfo_.baseM;
        bool isInnerAxisAlign = GetSizeWithDataType<aDataType>(args_.k) % MTE2_CACHELINE_SIZE == 0UL;
        if (mTailSize > 0UL && isInnerAxisAlign) {
            uint64_t baseTailCntMax = std::min((runInfo_.baseM - mTailSize) / BASIC_BLOCK_SIZE_16, runInfo_.mBlockCnt);
            uint64_t windowSize = std::min(WINDOW_LEN, runInfo_.mBlockCnt);
            uint64_t mainWindowNum = runInfo_.mBlockCnt / windowSize - 1UL;
            uint64_t tailWindowSize = runInfo_.mBlockCnt - mainWindowNum * windowSize;
            uint64_t perfRes = (mainWindowNum + 1UL) * runInfo_.baseM;
            uint64_t mergeWindowNum = 1UL;

            for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax; mergeLen += windowSize,
                          ++mergeWindowNum) {
                uint64_t newTailMain =
                    Align(CeilDiv((mergeLen * runInfo_.baseM + mTailSize), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
                uint64_t curPerf =
                    (mainWindowNum + 1UL - mergeWindowNum) * runInfo_.baseM + mergeWindowNum * newTailMain;
                if (curPerf <= perfRes) {
                    perfRes = curPerf;
                    runInfo_.mTailMain = newTailMain;
                    runInfo_.mBaseTailSplitCnt = mergeLen + 1UL;
                }
            }
        }
    }

    void CanUseAFullLoad()
    {
        // A-full-load is worthwhile only when one resident A tile fits in half
        // of L1 and can be reused across multiple N blocks.
        uint64_t maxBaseMSize = runInfo_.mBaseTailSplitCnt == 1UL ? runInfo_.baseM : runInfo_.mTailMain;
        runInfo_.isAFullLoad =
            runInfo_.mBlockCnt <= WINDOW_LEN && platformInfo_.aicNum % runInfo_.mBlockCnt == 0 &&
            GetSizeWithDataType<aDataType>(
                maxBaseMSize * Align(args_.k, aDataType == DataType::FP4 ? FP4_C0_SIZE : FP8_C0_SIZE)) <=
                platformInfo_.l1Size / NUM_TWO &&
            runInfo_.totalBlockCnt > platformInfo_.aicNum;
        CHECK_COND(runInfo_.isAFullLoad, "The requested A-full-load tiling does not support the current shape");

        // Fold the split tail into the steady-state baseM so the A tile can
        // stay resident in L1 across all K iterations.
        runInfo_.baseM = runInfo_.mBaseTailSplitCnt == 1UL ? runInfo_.baseM : runInfo_.mTailMain;
        runInfo_.dbL0c =
            runInfo_.baseM * runInfo_.baseN * DATA_SIZE_L0C * DB_SIZE <= platformInfo_.l0cSize ? DB_SIZE : 1U;
    }

    uint64_t GetDepthB1AFullLoad(const QuantMatmulArgs& args, const QuantMatmulRunInfo& runInfo,
                                        uint64_t leftSize)
    {
        // Build the B-side depth in multiples of the base K tile while
        // respecting both DMA granularity and the remaining L1 capacity.
        uint64_t baseStepK = 1UL;
        uint64_t baseKSize = GetSizeWithDataType<bDataType>(runInfo.baseK);
        if (baseKSize < BASIC_BLOCK_SIZE_128) {
            baseStepK = CeilDiv(BASIC_BLOCK_SIZE_128, baseKSize);
        }

        uint64_t scaleBaseK = Align(CeilDiv(runInfo.baseK, MX_GROUP_SIZE), TILING_MXFP_MULTI_BASE_SIZE);
        uint64_t baseSize = GetSizeWithDataType<bDataType>(runInfo.baseN * (runInfo.baseK + scaleBaseK) * baseStepK);
        uint64_t stepKBaseScale = 1UL;
        if (leftSize >= MTE2_MIN_LOAD_SIZE * DB_SIZE) {
            stepKBaseScale = CeilDiv(MTE2_MIN_LOAD_SIZE, baseSize);
        } else {
            // For small leftovers, still keep the depth proportional to the
            // remaining buffer budget instead of forcing a full minimum burst.
            stepKBaseScale = CeilDiv(leftSize / DB_SIZE, baseSize);
        }
        baseStepK = baseStepK * stepKBaseScale;

        constexpr uint64_t REFINED_STEP_K = 2UL;
        if (baseStepK == 1UL && args.k > runInfo.baseK && leftSize > baseSize * REFINED_STEP_K) {
            baseStepK = REFINED_STEP_K;
        }

        return baseStepK;
    }

    uint64_t GetScaleFactorBAFullLoad(const QuantMatmulArgs& args, const QuantMatmulRunInfo& runInfo,
                                             uint64_t leftSize)
    {
        // After B tiles are placed in L1, scaleB can reuse whatever capacity is
        // still left, capped by both K coverage and the path-specific max.
        uint64_t baseScaleBSize =
            runInfo.baseN * Align(CeilDiv(runInfo.baseK, MX_GROUP_SIZE), TILING_MXFP_MULTI_BASE_SIZE);

        uint64_t scaleFactorBBase = 1UL;
        uint64_t scaleBBaseKSize = Align(CeilDiv(runInfo.baseK, MX_GROUP_SIZE), TILING_MXFP_MULTI_BASE_SIZE);
        if (scaleBBaseKSize < BASIC_BLOCK_SIZE_128) {
            scaleFactorBBase = CeilDiv(BASIC_BLOCK_SIZE_128, scaleBBaseKSize);
        }

        uint64_t scaleFactorBMaxFromK = args.k / (runInfo.stepKb * runInfo.baseK);
        scaleFactorBMaxFromK = std::min(SCALER_FACTOR_MAX, scaleFactorBMaxFromK);
        scaleFactorBMaxFromK = std::max(SCALER_FACTOR_MIN, scaleFactorBMaxFromK);

        uint64_t scaleFactorB = 1UL;
        uint64_t scaleFactorBMax =
            std::min(MTE2_MIN_LOAD_SIZE * DB_SIZE, leftSize) / (baseScaleBSize * runInfo.stepKb * DB_SIZE);
        if (scaleFactorBMax != 0UL && scaleFactorBBase != 0UL) {
            if (scaleFactorBBase <= scaleFactorBMaxFromK && scaleFactorBMax >= scaleFactorBBase) {
                scaleFactorB = std::min(scaleFactorBMax / scaleFactorBBase * scaleFactorBBase, scaleFactorBMaxFromK);
            } else {
                scaleFactorB = std::min(scaleFactorBMax, scaleFactorBMaxFromK);
            }
        }

        return scaleFactorB;
    }
};

#endif // QUANT_MATMUL_MX_TILING_A_FULL_LOAD_H
