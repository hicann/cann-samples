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
 * \file quant_grouped_matmul_mx_tiling_split_m.h
 * \brief Host-side tiling helper for grouped MX split-M samples.
 */
#ifndef QUANT_GROUPED_MATMUL_MX_TILING_SPLIT_M_H
#define QUANT_GROUPED_MATMUL_MX_TILING_SPLIT_M_H

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "host_utils/common_utils.h"

#include "../utils/grouped_matmul_constant.h"
#include "quant_grouped_matmul_mx_tiling_data.h"
#include "quant_grouped_matmul_tiling_common.h"

template <gmm::DataType dataType>
class QuantGroupedMatmulMxTilingSplitM
{
public:
    void GetTilingData(
        uint32_t numOfGroups, uint32_t m, uint32_t n, uint32_t k, QuantGroupedMatmulMxTilingData& tilingData)
    {
        args_ = {};
        platformInfo_ = {};
        runInfo_ = {};

        InitCompileInfo();
        InitShapeArgs(numOfGroups, m, n, k);
        DoOpTiling(tilingData);
        PrintTilingData(tilingData);
    }

private:
    QuantGroupedMatmulTilingArgs args_{};
    QuantGroupedMatmulPlatformInfo platformInfo_{};
    QuantGroupedMatmulRunInfo runInfo_{};

    void InitCompileInfo()
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
        platformInfo_.aicNum = ascendcPlatform->GetCoreNumAic();
        platformInfo_.aivNum = ascendcPlatform->GetCoreNumAiv();
        platformInfo_.socVersion = ascendcPlatform->GetSocVersion();
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo_.ubSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo_.l2Size);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::BT, platformInfo_.btSize);
    }

    void InitShapeArgs(uint32_t numOfGroups, uint32_t m, uint32_t n, uint32_t k)
    {
        CHECK_COND(
            numOfGroups > 0U && m > 0U && n > 0U && k > 0U,
            "numOfGroups, m, n and k must be greater than zero.");
        args_.groupNum = numOfGroups;
        args_.m = m;
        args_.n = n;
        args_.k = k;
    }

    uint64_t GetTensorStorageBytes(uint64_t elemCount) const
    {
        return GetSizeWithDataType<dataType>(elemCount);
    }

    void CalBasicBlock()
    {
        runInfo_.baseM = Align(std::min<uint64_t>(args_.m, 256UL), GroupedMatmulRecipe::CUBE_BLOCK);
        runInfo_.baseN = Align(std::min<uint64_t>(args_.n, 256UL), GroupedMatmulRecipe::CUBE_BLOCK);
        runInfo_.baseK = Align(std::min<uint64_t>(args_.k, (dataType == gmm::DataType::DT_FLOAT4_E2M1) ? 256UL : 128UL), GroupedMatmulRecipe::MX_DIVISOR_SIZE);
        runInfo_.dbL0C =
            (runInfo_.baseM * runInfo_.baseN * QuantGroupedMatmulTilingConst::DATA_SIZE_L0C *
                    QuantGroupedMatmulTilingConst::DB_SIZE <=
                platformInfo_.l0cSize)
                ? static_cast<uint8_t>(QuantGroupedMatmulTilingConst::DB_SIZE)
                : 1U;
    }

    uint64_t GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit)
    {
        // `GroupedQmmTiling::GetDepthA1B1`
        if (depthInit > 1UL &&
            perDepthSize >
                QuantGroupedMatmulTilingConst::DB_SIZE *
                    QuantGroupedMatmulTilingConst::MTE2_MIN_LOAD_SIZE_V120) {
            return depthInit;
        }
        if (perDepthSize == 0UL) {
            return depthInit;
        }
        uint64_t depthScale = leftSize / perDepthSize;
        if (depthInit > 1UL) {
            uint64_t baseKSize = GetTensorStorageBytes(runInfo_.baseK);
            while ((depthScale * baseKSize) % QuantGroupedMatmulTilingConst::BASIC_BLOCK_SIZE_512 != 0 &&
                   (depthScale * baseKSize) > QuantGroupedMatmulTilingConst::BASIC_BLOCK_SIZE_512) {
                depthScale -= 1UL;
            }
            if ((depthScale * baseKSize) % QuantGroupedMatmulTilingConst::BASIC_BLOCK_SIZE_512 != 0 &&
                (depthScale * baseKSize) >= QuantGroupedMatmulTilingConst::BASIC_BLOCK_SIZE_256) {
                depthScale =
                    QuantGroupedMatmulTilingConst::BASIC_BLOCK_SIZE_256 / std::max<uint64_t>(1UL, baseKSize);
            }
            depthScale = std::max<uint64_t>(depthScale, 1UL);
        } else {
            constexpr uint64_t index = 2UL;
            depthScale = 1UL;
            while (depthScale * perDepthSize < leftSize) {
                depthScale *= index;
            }
            depthScale = (depthScale == 1UL) ? depthScale : (depthScale / index);
        }
        return depthInit * depthScale;
    }

    uint64_t GetDepthWithHighBW(uint64_t mnL1) const
    {
        // `GroupedQmmBasicApiTiling::GetDepthWithHighBW`
        uint64_t baseKSize = GetTensorStorageBytes(runInfo_.baseK);
        uint64_t depth =
            Align(
                CeilDiv<uint64_t>(
                    QuantGroupedMatmulTilingConst::MTE2_MIN_LOAD_SIZE_V120, std::max<uint64_t>(1UL, mnL1)),
                QuantGroupedMatmulTilingConst::BASIC_BLOCK_SIZE_256) /
            std::max<uint64_t>(1UL, baseKSize) * QuantGroupedMatmulTilingConst::DB_SIZE;
        uint64_t pow2Depth = QuantGroupedMatmulTilingConst::POWER_OF_TWO;
        while (pow2Depth < depth) {
            pow2Depth *= QuantGroupedMatmulTilingConst::POWER_OF_TWO;
        }
        return std::min<uint64_t>(
            pow2Depth,
            CeilDiv<uint64_t>(args_.k, runInfo_.baseK) * QuantGroupedMatmulTilingConst::DB_SIZE);
    }

    void ModifyDepthForUnalign(
        uint64_t leftL1Size, uint64_t baseASize, uint64_t baseBSize, uint64_t baseScaleABSize)
    {
        if (args_.k % QuantGroupedMatmulTilingConst::BASIC_BLOCK_SIZE_128 == 0) {
            return;
        }
        if (QuantGroupedMatmulTilingConst::TRANS_A &&
            (!QuantGroupedMatmulTilingConst::TRANS_B || QuantGroupedMatmulTilingConst::WEIGHT_FRACTAL_NZ)) {
            return;
        }
        if (!QuantGroupedMatmulTilingConst::TRANS_A) {
            if (runInfo_.depthA1 <= runInfo_.depthB1) {
                uint64_t leftASize =
                    leftL1Size - runInfo_.depthB1 * baseBSize - runInfo_.depthB1 * baseScaleABSize;
                while (runInfo_.depthA1 * QuantGroupedMatmulTilingConst::POWER_OF_TWO * baseASize <= leftASize) {
                    runInfo_.depthA1 *= QuantGroupedMatmulTilingConst::POWER_OF_TWO;
                }
                if (runInfo_.depthA1 * baseASize + runInfo_.depthB1 * baseBSize +
                        std::max(runInfo_.depthA1, runInfo_.depthB1) * baseScaleABSize >
                    leftL1Size) {
                    runInfo_.depthA1 = runInfo_.depthB1;
                }
            } else if (QuantGroupedMatmulTilingConst::TRANS_B &&
                       !QuantGroupedMatmulTilingConst::WEIGHT_FRACTAL_NZ) {
                uint64_t leftBSize =
                    leftL1Size - runInfo_.depthA1 * baseASize - runInfo_.depthA1 * baseScaleABSize;
                while (runInfo_.depthB1 * QuantGroupedMatmulTilingConst::POWER_OF_TWO * baseBSize <= leftBSize) {
                    runInfo_.depthB1 *= QuantGroupedMatmulTilingConst::POWER_OF_TWO;
                }
                if (runInfo_.depthA1 * baseASize + runInfo_.depthB1 * baseBSize +
                        std::max(runInfo_.depthA1, runInfo_.depthB1) * baseScaleABSize >
                    leftL1Size) {
                    runInfo_.depthB1 = runInfo_.depthA1;
                }
            }
        } else {
            while ((runInfo_.depthA1 * baseASize -
                       std::max(
                           runInfo_.depthA1,
                           runInfo_.depthB1 * QuantGroupedMatmulTilingConst::POWER_OF_TWO) *
                           baseScaleABSize) < leftL1Size) {
                runInfo_.depthB1 *= QuantGroupedMatmulTilingConst::POWER_OF_TWO;
            }
        }
    }

    void CalStepKs()
    {
        runInfo_.stepKa =
            runInfo_.depthA1 == 1UL ? 1UL : runInfo_.depthA1 / QuantGroupedMatmulTilingConst::DB_SIZE;
        runInfo_.stepKb =
            runInfo_.depthB1 == 1UL ? 1UL : runInfo_.depthB1 / QuantGroupedMatmulTilingConst::DB_SIZE;

        if (runInfo_.stepKa * runInfo_.baseK > args_.k) {
            runInfo_.stepKa = CeilDiv<uint64_t>(args_.k, runInfo_.baseK);
        }
        if (runInfo_.stepKb * runInfo_.baseK >= args_.k) {
            runInfo_.stepKb = CeilDiv<uint64_t>(args_.k, runInfo_.baseK);
        }
        if (runInfo_.stepKa >= runInfo_.stepKb && runInfo_.stepKa * runInfo_.baseK < args_.k) {
            runInfo_.stepKa = runInfo_.stepKa / runInfo_.stepKb * runInfo_.stepKb;
        }
        if (runInfo_.stepKb > runInfo_.stepKa && runInfo_.stepKb * runInfo_.baseK < args_.k) {
            runInfo_.stepKb = runInfo_.stepKb / runInfo_.stepKa * runInfo_.stepKa;
        }

        runInfo_.depthA1 = runInfo_.stepKa * QuantGroupedMatmulTilingConst::DB_SIZE;
        runInfo_.depthB1 = runInfo_.stepKb * QuantGroupedMatmulTilingConst::DB_SIZE;
    }

    bool CalScaleFactors()
    {
        uint64_t baseASize = GetTensorStorageBytes(runInfo_.baseM * runInfo_.baseK);
        uint64_t baseBSize = GetTensorStorageBytes(runInfo_.baseN * runInfo_.baseK);
        uint64_t baseScaleASize =
            CeilDiv<uint64_t>(runInfo_.baseK, GroupedMatmulRecipe::MX_GROUP_SIZE) * runInfo_.baseM;
        uint64_t baseScaleBSize =
            CeilDiv<uint64_t>(runInfo_.baseK, GroupedMatmulRecipe::MX_GROUP_SIZE) * runInfo_.baseN;
        uint64_t leftL1Size =
            platformInfo_.l1Size - (runInfo_.depthA1 * baseASize + runInfo_.depthB1 * baseBSize);
        uint32_t scaleInit = static_cast<uint32_t>(
            leftL1Size /
            (std::max(runInfo_.depthA1, runInfo_.depthB1) * (baseScaleASize + baseScaleBSize)));
        CHECK_COND(scaleInit > 0U, "MX per-group: scaleFactor init is 0 (check m/n/k and L1 tiling).");

        uint32_t scaleFactorAMax = std::min<uint32_t>(
            static_cast<uint32_t>(
                QuantGroupedMatmulTilingConst::MTE2_MIN_LOAD_SIZE_V120 /
                std::max<uint64_t>(1UL, baseScaleASize)),
            QuantGroupedMatmulTilingConst::SCALE_FACTOR_MAX);
        uint32_t scaleFactorBMax = std::min<uint32_t>(
            static_cast<uint32_t>(
                QuantGroupedMatmulTilingConst::MTE2_MIN_LOAD_SIZE_V120 /
                std::max<uint64_t>(1UL, baseScaleBSize)),
            QuantGroupedMatmulTilingConst::SCALE_FACTOR_MAX);
        uint32_t scaleFactorA =
            static_cast<uint32_t>(CeilDiv<uint64_t>(args_.k, runInfo_.stepKa * runInfo_.baseK));
        uint32_t scaleFactorB =
            static_cast<uint32_t>(CeilDiv<uint64_t>(args_.k, runInfo_.stepKb * runInfo_.baseK));
        runInfo_.scaleFactorA =
            std::max<uint32_t>(QuantGroupedMatmulTilingConst::SCALE_FACTOR_MIN, scaleFactorA);
        runInfo_.scaleFactorB =
            std::max<uint32_t>(QuantGroupedMatmulTilingConst::SCALE_FACTOR_MIN, scaleFactorB);
        runInfo_.scaleFactorA = std::min<uint32_t>(scaleFactorAMax, runInfo_.scaleFactorA);
        runInfo_.scaleFactorB = std::min<uint32_t>(scaleFactorBMax, runInfo_.scaleFactorB);

        if (runInfo_.scaleFactorA > scaleInit && runInfo_.scaleFactorB > scaleInit) {
            if (runInfo_.depthA1 >= runInfo_.depthB1) {
                runInfo_.scaleFactorB = static_cast<uint32_t>(
                    scaleInit * runInfo_.depthA1 / std::max<uint64_t>(1UL, runInfo_.depthB1));
                runInfo_.scaleFactorA = scaleInit;
            } else {
                runInfo_.scaleFactorA = static_cast<uint32_t>(
                    scaleInit * runInfo_.depthB1 / std::max<uint64_t>(1UL, runInfo_.depthA1));
                runInfo_.scaleFactorB = scaleInit;
            }
        }
        return true;
    }

    bool CalL1Tiling()
    {
        uint64_t leftL1Size = platformInfo_.l1Size;
        uint64_t baseASize = GetTensorStorageBytes(runInfo_.baseM * runInfo_.baseK);
        uint64_t baseBSize = GetTensorStorageBytes(runInfo_.baseN * runInfo_.baseK);
        uint64_t alignedGroups = Align(
            CeilDiv<uint64_t>(runInfo_.baseK, GroupedMatmulRecipe::MX_GROUP_SIZE),
            GroupedMatmulRecipe::MX_MULTI_SIZE);
        uint64_t baseScaleASize = alignedGroups * runInfo_.baseM;
        uint64_t baseScaleBSize = alignedGroups * runInfo_.baseN;
        uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
        CHECK_COND(leftL1Size >= baseL1Size, "L1 space overflow. Free L1 smaller than base tile.");

        uint64_t depthInit = GetDepthA1B1(leftL1Size, baseL1Size, 1UL);
        runInfo_.depthA1 = GetDepthWithHighBW(std::min<uint64_t>(args_.m, runInfo_.baseM));
        runInfo_.depthB1 = GetDepthWithHighBW(std::min<uint64_t>(args_.n, runInfo_.baseN));
        if (runInfo_.depthA1 * baseASize + runInfo_.depthB1 * baseBSize +
                std::max(runInfo_.depthA1, runInfo_.depthB1) * (baseScaleASize + baseScaleBSize) >
            leftL1Size) {
            runInfo_.depthA1 = depthInit;
            runInfo_.depthB1 = depthInit;
        }
        ModifyDepthForUnalign(leftL1Size, baseASize, baseBSize, baseScaleASize + baseScaleBSize);
        CalStepKs();
        return CalScaleFactors();
    }

    void DoOpTiling(QuantGroupedMatmulMxTilingData& tilingData)
    {
        CalBasicBlock();
        CHECK_COND(CalL1Tiling(), "CalL1Tiling failed.");

        runInfo_.kAL1 = std::min<uint64_t>(runInfo_.stepKa * runInfo_.baseK, args_.k);
        runInfo_.kBL1 = std::min<uint64_t>(runInfo_.stepKb * runInfo_.baseK, args_.k);
        runInfo_.kAL1 = Align(runInfo_.kAL1, GroupedMatmulRecipe::MX_DIVISOR_SIZE);
        runInfo_.kBL1 = Align(runInfo_.kBL1, GroupedMatmulRecipe::MX_DIVISOR_SIZE);

        uint64_t scaleK = std::min<uint64_t>(
            std::max<uint64_t>(
                static_cast<uint64_t>(runInfo_.scaleFactorA) * runInfo_.stepKa,
                static_cast<uint64_t>(runInfo_.scaleFactorB) * runInfo_.stepKb) *
                runInfo_.baseK,
            args_.k);
        scaleK = Align(scaleK, GroupedMatmulRecipe::MX_DIVISOR_SIZE);
        runInfo_.scaleKAL1 = scaleK;
        runInfo_.scaleKBL1 = scaleK;

        tilingData = {};
        tilingData.groupNum = static_cast<uint32_t>(args_.groupNum);
        tilingData.maxM = static_cast<uint32_t>(args_.m);
        tilingData.n = static_cast<uint32_t>(args_.n);
        tilingData.k = static_cast<uint32_t>(args_.k);
        tilingData.baseM = static_cast<uint32_t>(runInfo_.baseM);
        tilingData.baseN = static_cast<uint32_t>(runInfo_.baseN);
        tilingData.baseK = static_cast<uint32_t>(runInfo_.baseK);
        tilingData.kAL1 = static_cast<uint32_t>(runInfo_.kAL1);
        tilingData.kBL1 = static_cast<uint32_t>(runInfo_.kBL1);
        tilingData.scaleKAL1 = static_cast<uint32_t>(runInfo_.scaleKAL1);
        tilingData.scaleKBL1 = static_cast<uint32_t>(runInfo_.scaleKBL1);
        tilingData.usedCoreNum = static_cast<uint32_t>(std::max<uint64_t>(1, platformInfo_.aicNum));
        tilingData.dbL0C = runInfo_.dbL0C;
    }

    void PrintTilingData(const QuantGroupedMatmulMxTilingData& tilingData) const
    {
        printf("[GroupedMatmul Strategy]\n");
        printf("  strategy           : split_m\n");
        printf("[GroupedMatmul Tiling Data]\n");
        printf("  groupNum           : %u\n", tilingData.groupNum);
        printf("  maxM               : %u\n", tilingData.maxM);
        printf("  n                  : %u\n", tilingData.n);
        printf("  k                  : %u\n", tilingData.k);
        printf("  baseM              : %u\n", tilingData.baseM);
        printf("  baseN              : %u\n", tilingData.baseN);
        printf("  baseK              : %u\n", tilingData.baseK);
        printf("  kAL1               : %u\n", tilingData.kAL1);
        printf("  kBL1               : %u\n", tilingData.kBL1);
        printf("  scaleKAL1          : %u\n", tilingData.scaleKAL1);
        printf("  scaleKBL1          : %u\n", tilingData.scaleKBL1);
        printf("  usedCoreNum        : %u\n", tilingData.usedCoreNum);
        printf("  dbL0C              : %u\n", tilingData.dbL0C);
    }
};

using QuantGroupedMatmulMxfp4TilingSplitM = QuantGroupedMatmulMxTilingSplitM<gmm::DataType::DT_FLOAT4_E2M1>;
using QuantGroupedMatmulMxfp8TilingSplitM = QuantGroupedMatmulMxTilingSplitM<gmm::DataType::DT_FLOAT8_E4M3FN>;

#endif // QUANT_GROUPED_MATMUL_MX_TILING_SPLIT_M_H
