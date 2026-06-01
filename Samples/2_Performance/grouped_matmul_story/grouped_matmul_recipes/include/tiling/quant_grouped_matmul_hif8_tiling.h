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
 * \file quant_grouped_matmul_hif8_tiling.h
 * \brief Host-side tiling helper for grouped HiFloat8 samples.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "host_utils/common_utils.h"

#include "../utils/grouped_matmul_constant.h"
#include "quant_grouped_matmul_hif8_tiling_data.h"
#include "quant_grouped_matmul_tiling_common.h"

namespace hif8 {
constexpr uint64_t A_DTYPE_SIZE = 1UL;
constexpr uint64_t B_DTYPE_SIZE = 1UL;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32UL;
constexpr uint64_t L1_SHAPE_ALIGN = 32UL;
constexpr uint64_t L1_FOUR_BUFFER = 4UL;
constexpr uint32_t KERNEL_QUANT_DEFAULT = 0U;
constexpr uint32_t KERNEL_QUANT_PERCHANNEL = 2U;
constexpr uint32_t KERNEL_QUANT_PERTENSOR = 1U;
} // namespace hif8

struct QuantGroupedMatmulHif8Config {
    uint64_t scaleDtypeSize{sizeof(uint64_t)};
    uint32_t x1QuantMode{hif8::KERNEL_QUANT_DEFAULT};
    uint32_t x2QuantMode{hif8::KERNEL_QUANT_PERTENSOR};
};

class QuantGroupedMatmulHif8Tiling {
public:
    void SetQuantConfig(const QuantGroupedMatmulHif8Config& cfg)
    {
        config_ = cfg;
    }

    void GetTilingData(
        uint32_t numOfGroups, uint32_t m, uint32_t n, uint32_t k, bool transA, bool transB,
        QuantGroupedMatmulHif8TilingData& tilingData)
    {
        args_ = {};
        platformInfo_ = {};
        runInfo_ = {};
        transA_ = transA;
        transB_ = transB;

        InitCompileInfo();
        InitShapeArgs(numOfGroups, m, n, k);
        DoOpTiling(tilingData);
        PrintTilingData(tilingData);
    }

private:
    QuantGroupedMatmulTilingArgs args_{};
    QuantGroupedMatmulPlatformInfo platformInfo_{};
    QuantGroupedMatmulRunInfo runInfo_{};
    QuantGroupedMatmulHif8Config config_{};
    bool transA_{false};
    bool transB_{true};

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
        CHECK_COND(!transA_, "Grouped HiFloat8 sample currently supports split-M only: transA must be false.");
        args_.groupNum = numOfGroups;
        args_.m = m;
        args_.n = n;
        args_.k = k;
    }

    void CalBasicBlock()
    {
        runInfo_.baseM = Align(std::min<uint64_t>(args_.m, QuantGroupedMatmulTilingConst::BASIC_BLOCK_SIZE_256),
            GroupedMatmulRecipe::CUBE_BLOCK);
        runInfo_.baseN = std::min<uint64_t>(args_.n, QuantGroupedMatmulTilingConst::BASIC_BLOCK_SIZE_256);
        runInfo_.baseN = transB_ ? Align(runInfo_.baseN, GroupedMatmulRecipe::CUBE_BLOCK)
                                 : Align(runInfo_.baseN, hif8::L1_SHAPE_ALIGN);
        runInfo_.baseK = Align(
            std::min<uint64_t>(args_.k, QuantGroupedMatmulTilingConst::BASIC_BLOCK_SIZE_128),
            hif8::CUBE_REDUCE_BLOCK);
        CHECK_COND(
            runInfo_.baseM > 0UL && runInfo_.baseN > 0UL && runInfo_.baseK > 0UL,
            "Failed to derive a valid grouped HiFloat8 base tile.");
        runInfo_.dbL0C =
            (runInfo_.baseM * runInfo_.baseN * QuantGroupedMatmulTilingConst::DATA_SIZE_L0C *
                    QuantGroupedMatmulTilingConst::DB_SIZE <=
                platformInfo_.l0cSize)
                ? static_cast<uint8_t>(QuantGroupedMatmulTilingConst::DB_SIZE)
                : 1U;
    }

    uint64_t GetPerChannelScaleReservation() const
    {
        if (config_.x2QuantMode != hif8::KERNEL_QUANT_PERCHANNEL) {
            return 0UL;
        }
        return runInfo_.baseN * config_.scaleDtypeSize * QuantGroupedMatmulTilingConst::DB_SIZE;
    }

    void CalStepKs()
    {
        runInfo_.stepKa = std::max<uint64_t>(1UL, runInfo_.depthA1 / QuantGroupedMatmulTilingConst::DB_SIZE);
        runInfo_.stepKb = std::max<uint64_t>(1UL, runInfo_.depthB1 / QuantGroupedMatmulTilingConst::DB_SIZE);

        if (runInfo_.stepKa * runInfo_.baseK > args_.k) {
            runInfo_.stepKa = CeilDiv<uint64_t>(args_.k, runInfo_.baseK);
        }
        if (runInfo_.stepKb * runInfo_.baseK > args_.k) {
            runInfo_.stepKb = CeilDiv<uint64_t>(args_.k, runInfo_.baseK);
        }
        if (runInfo_.stepKa > runInfo_.stepKb && runInfo_.stepKb > 0UL) {
            runInfo_.stepKa = std::max<uint64_t>(1UL, runInfo_.stepKa / runInfo_.stepKb * runInfo_.stepKb);
        }
        if (runInfo_.stepKb > runInfo_.stepKa && runInfo_.stepKa > 0UL) {
            runInfo_.stepKb = std::max<uint64_t>(1UL, runInfo_.stepKb / runInfo_.stepKa * runInfo_.stepKa);
        }

        runInfo_.depthA1 = runInfo_.stepKa * QuantGroupedMatmulTilingConst::DB_SIZE;
        runInfo_.depthB1 = runInfo_.stepKb * QuantGroupedMatmulTilingConst::DB_SIZE;
    }

    void CalL1Tiling()
    {
        uint64_t scaleReserved = GetPerChannelScaleReservation();
        CHECK_COND(platformInfo_.l1Size > scaleReserved, "L1 budget is insufficient for per-channel scale.");
        uint64_t leftL1Size = platformInfo_.l1Size - scaleReserved;
        uint64_t baseASize = runInfo_.baseM * runInfo_.baseK * hif8::A_DTYPE_SIZE;
        uint64_t baseBSize = runInfo_.baseN * runInfo_.baseK * hif8::B_DTYPE_SIZE;
        CHECK_COND(baseASize > 0UL && baseBSize > 0UL, "Invalid base tile size for L1 tiling.");

        runInfo_.depthA1 = std::max<uint64_t>(
            QuantGroupedMatmulTilingConst::DB_SIZE, leftL1Size / 2UL / baseASize);
        runInfo_.depthB1 = std::max<uint64_t>(
            QuantGroupedMatmulTilingConst::DB_SIZE, leftL1Size / 2UL / baseBSize);
        CalStepKs();
    }

    uint8_t CalculateNBufferNum() const
    {
        uint64_t stepK = std::min(runInfo_.stepKa, runInfo_.stepKb);
        uint64_t kL1 = stepK * runInfo_.baseK;
        uint64_t usedL1Size =
            (runInfo_.baseM * kL1 * hif8::A_DTYPE_SIZE + runInfo_.baseN * kL1 * hif8::B_DTYPE_SIZE) *
            hif8::L1_FOUR_BUFFER;
        usedL1Size += GetPerChannelScaleReservation();
        return usedL1Size < platformInfo_.l1Size ? static_cast<uint8_t>(hif8::L1_FOUR_BUFFER)
                                                 : static_cast<uint8_t>(QuantGroupedMatmulTilingConst::DB_SIZE);
    }

    void DoOpTiling(QuantGroupedMatmulHif8TilingData& tilingData)
    {
        CalBasicBlock();
        CalL1Tiling();
        uint8_t nBufferNum = CalculateNBufferNum();
        uint64_t stepKaOut = runInfo_.stepKa;
        uint64_t stepKbOut = runInfo_.stepKb;
        if (nBufferNum == hif8::L1_FOUR_BUFFER) {
            uint64_t stepK = std::min(stepKaOut, stepKbOut);
            stepKaOut = stepK;
            stepKbOut = stepK;
        }

        tilingData = {};
        tilingData.groupNum = static_cast<uint32_t>(args_.groupNum);
        tilingData.m = static_cast<uint32_t>(args_.m);
        tilingData.n = static_cast<uint32_t>(args_.n);
        tilingData.k = static_cast<uint32_t>(args_.k);
        tilingData.usedCoreNum = static_cast<uint32_t>(std::max<uint64_t>(1UL, platformInfo_.aicNum));
        tilingData.baseM = static_cast<uint32_t>(runInfo_.baseM);
        tilingData.baseN = static_cast<uint32_t>(runInfo_.baseN);
        tilingData.baseK = static_cast<uint32_t>(runInfo_.baseK);
        tilingData.stepKa = static_cast<uint32_t>(stepKaOut);
        tilingData.stepKb = static_cast<uint32_t>(stepKbOut);
        tilingData.kAL1 = static_cast<uint32_t>(stepKaOut * runInfo_.baseK);
        tilingData.kBL1 = static_cast<uint32_t>(stepKbOut * runInfo_.baseK);
        tilingData.nBufferNum = static_cast<uint32_t>(nBufferNum);
        tilingData.dbL0C = runInfo_.dbL0C;
        tilingData.x1QuantMode = config_.x1QuantMode;
        tilingData.x2QuantMode = config_.x2QuantMode;
    }

    void PrintTilingData(const QuantGroupedMatmulHif8TilingData& tilingData) const
    {
        printf("[GroupedMatmul HiFloat8 Tiling Data]\n");
        printf("  transA             : %s\n", transA_ ? "true" : "false");
        printf("  transB             : %s\n", transB_ ? "true" : "false");
        printf("  groupNum           : %u\n", tilingData.groupNum);
        printf("  m                  : %u\n", tilingData.m);
        printf("  n                  : %u\n", tilingData.n);
        printf("  k                  : %u\n", tilingData.k);
        printf("  baseM              : %u\n", tilingData.baseM);
        printf("  baseN              : %u\n", tilingData.baseN);
        printf("  baseK              : %u\n", tilingData.baseK);
        printf("  kAL1               : %u\n", tilingData.kAL1);
        printf("  kBL1               : %u\n", tilingData.kBL1);
        printf("  nBufferNum         : %u\n", tilingData.nBufferNum);
        printf("  usedCoreNum        : %u\n", tilingData.usedCoreNum);
        printf("  dbL0C              : %u\n", tilingData.dbL0C);
        printf("  x1QuantMode        : %u\n", tilingData.x1QuantMode);
        printf("  x2QuantMode        : %u\n", tilingData.x2QuantMode);
    }
};
