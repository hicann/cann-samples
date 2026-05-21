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
 * \file quant_matmul_hifp8_tiling_swat.h
 * \brief SWAT tiling specialization for the HiFloat8 non-full-load path.
 *
 * Host configuration supplies kernel quant enums on x1/x2 (same encoding as
 * QuantBatchMatmul on device): DEFAULT means no separate scale tensor for that
 * operand (empty-scale path); both PERTENSOR for two scalar scales; DEFAULT +
 * PERCHANNEL for per-N channel scale on operand B only (L1 reservation follows x2).
 *
 * Compared with the MX SWAT template:
 *   - HiFloat8 operands use 1 byte per element on GM/L1/L0, so baseK is aligned
 *     to CUBE_REDUCE_BLOCK (32 elements) rather than to the MX group size.
 *   - The per-group scale footprint used by MX is dropped. Instead, L1 is first
 *     reduced by the per-channel scale reservations, then split evenly
 *     between A and B to derive depthA1/B1 and stepKa/stepKb.
 */

#ifndef QUANT_MATMUL_HIFP8_TILING_SWAT_H
#define QUANT_MATMUL_HIFP8_TILING_SWAT_H

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "host_utils/common_utils.h"
#include "quant_matmul_hifp8_tiling_data.h"
#include "quant_matmul_tiling_common.h"

namespace hifp8 {

// HiFloat8: one byte storage per activation/weight element in this recipe.
constexpr uint64_t A_DTYPE_SIZE = 1UL;
constexpr uint64_t B_DTYPE_SIZE = 1UL;

// Cube reduce-axis alignment for HiFloat8: 1 byte/element, 32B / 1B = 32 elements.
constexpr uint64_t CUBE_REDUCE_BLOCK = 32UL;
// Same L1/L2 byte-alignment rules as adaptive_sliding_window_tiling for 1-byte operands: L1 inner axis 32B, L2 128B.
constexpr uint64_t L1_SHAPE_ALIGN = 32UL;
constexpr uint64_t L2_SHAPE_ALIGN = 128UL;
constexpr uint64_t L1_TWO_BUFFER = 2UL;

// Values serialized to tilingData.x1QuantMode / x2QuantMode (QuantBatchMatmul::QuantMode on device).
// KERNEL_QUANT_DEFAULT (0): no scale GM tensor for that side — empty / unused scale slot.
constexpr uint32_t KERNEL_QUANT_DEFAULT = 0U;
constexpr uint32_t KERNEL_QUANT_PERCHANNEL = 1U;
constexpr uint32_t KERNEL_QUANT_PERTENSOR = 2U;

} // namespace hifp8

struct QuantMatmulHifp8Config {
    uint64_t scaleDtypeSize{sizeof(uint64_t)};
    uint32_t x1QuantMode{hifp8::KERNEL_QUANT_DEFAULT};
    uint32_t x2QuantMode{hifp8::KERNEL_QUANT_PERTENSOR};
};

class QuantMatmulHifp8TilingSwat {
public:
    QuantMatmulHifp8TilingSwat() = default;
    QuantMatmulHifp8TilingSwat(const QuantMatmulHifp8TilingSwat&) = delete;
    QuantMatmulHifp8TilingSwat& operator=(const QuantMatmulHifp8TilingSwat&) = delete;
    ~QuantMatmulHifp8TilingSwat() = default;

    void SetQuantConfig(const QuantMatmulHifp8Config& cfg)
    {
        config_ = cfg;
    }

    void GetTilingData(
        uint64_t m, uint64_t n, uint64_t k, bool transA, bool transB, QuantMatmulHifp8TilingData& tilingData)
    {
        args_ = {};
        runInfo_ = {};
        platformInfo_ = {};

        InitCompileInfo();
        args_.m = m;
        args_.n = n;
        args_.k = k;
        args_.transA = transA;
        args_.transB = transB;

        DoOpTiling(tilingData);
        PrintTilingData(tilingData);
    }

    // Shape defaults align with QuantMatmulArgs::transB == true unless the caller
    // overrides transposes in the six-argument overload.
    void GetTilingData(uint64_t m, uint64_t n, uint64_t k, QuantMatmulHifp8TilingData& tilingData)
    {
        GetTilingData(m, n, k, false, true, tilingData);
    }

private:
    QuantMatmulArgs args_{};
    QuantMatmulPlatformInfo platformInfo_{};
    QuantMatmulRunInfo runInfo_{};
    QuantMatmulHifp8Config config_{};

    static const char* TilingName()
    {
        return "hifp8_swat";
    }

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

    void DoOpTiling(QuantMatmulHifp8TilingData& tilingData)
    {
        // Stage the decisions the same way the SWAT template does: pick a base
        // block, smooth tail edges, size the tail tile, then figure out how
        // much L1 is left for A/B after reserving the per-channel scale.
        CalcBasicBlock();
        OptimizeEdgeBasicBlock();
        CalcTailBasicBlock();
        CalL1Tiling();

        uint8_t nBufferNum = CalculateDefaultNBufferNum();
        BuildTilingData(tilingData, nBufferNum);
    }

    // --------------------------- basic block ---------------------------------
    void CalcBasicBlock()
    {
        // Aligned to adaptive_sliding_window_tiling::CalcBasicBlock: baseM
        // uses CUBE (16) when A is not transposed, else L1 inner-axis
        // element alignment 32 (HiFloat8: one byte/element, so 32 elems = 32 B; same numeric rule as adaptive 8-bit tiling). baseN uses the symmetric rule for B
        // (CUBE when transB, L1 when not). baseK is capped to 128 K-elements
        // and reduced-axis aligned to 32.
        runInfo_.baseM = std::min(args_.m, BASIC_BLOCK_SIZE_256);
        runInfo_.baseM = !args_.transA ? Align(runInfo_.baseM, CUBE_BLOCK) : Align(runInfo_.baseM, hifp8::L1_SHAPE_ALIGN);
        runInfo_.baseN = std::min(args_.n, BASIC_BLOCK_SIZE_256);
        runInfo_.baseN = args_.transB ? Align(runInfo_.baseN, CUBE_BLOCK) : Align(runInfo_.baseN, hifp8::L1_SHAPE_ALIGN);
        runInfo_.baseK = Align(std::min(args_.k, BASIC_BLOCK_SIZE_128), hifp8::CUBE_REDUCE_BLOCK);

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
        runInfo_.dbL0c =
            runInfo_.baseM * runInfo_.baseN * DATA_SIZE_L0C * DB_SIZE <= platformInfo_.l0cSize ? DB_SIZE : 1UL;
    }

    void AdjustBasicBlock()
    {
        // Mirror adaptive_sliding_window_tiling::AdjustBasicBlock: per-axis
        // min granularity depends on (transA, transB) for HiFloat8 (1-byte) cube operand moves.
        uint64_t baseMAlignNum = args_.transA ? hifp8::L2_SHAPE_ALIGN : CUBE_BLOCK;
        uint64_t baseNAlignNum = args_.transB ? CUBE_BLOCK : hifp8::L2_SHAPE_ALIGN;
        uint64_t baseKAlignNum = (args_.transA && !args_.transB) ? 32UL : hifp8::L2_SHAPE_ALIGN;

        uint64_t mMaxtile = CeilDiv(args_.m, baseMAlignNum);
        uint64_t nMaxtile = CeilDiv(args_.n, baseNAlignNum);
        uint64_t tempBaseM = runInfo_.baseM;
        uint64_t tempBaseN = runInfo_.baseN;
        uint64_t coreNumMN = platformInfo_.aicNum;

        if (mMaxtile * nMaxtile >= coreNumMN || (!args_.transA && args_.transB)) {
            uint64_t mCore = CeilDiv(args_.m, runInfo_.baseM);
            uint64_t nCore = CeilDiv(args_.n, runInfo_.baseN);
            if (mMaxtile < nMaxtile || (mMaxtile == nMaxtile && baseNAlignNum == CUBE_BLOCK)) {
                tempBaseM = Align(CeilDiv(args_.m, mCore), baseMAlignNum);
                mCore = CeilDiv(args_.m, tempBaseM);
                nCore = coreNumMN / mCore;
                tempBaseN = Align(CeilDiv(args_.n, nCore), baseNAlignNum);
            } else {
                tempBaseN = Align(CeilDiv(args_.n, nCore), baseNAlignNum);
                nCore = CeilDiv(args_.n, tempBaseN);
                mCore = coreNumMN / nCore;
                tempBaseM = Align(CeilDiv(args_.m, mCore), baseMAlignNum);
            }

            while (tempBaseN >= tempBaseM * BASEM_BASEN_RATIO && nCore < coreNumMN / NUM_TWO &&
                   tempBaseN != baseNAlignNum) {
                nCore = nCore * NUM_TWO;
                mCore = coreNumMN / nCore;
                tempBaseM = Align(CeilDiv(args_.m, mCore), baseMAlignNum);
                tempBaseN = Align(CeilDiv(args_.n, nCore), baseNAlignNum);
                mCore = CeilDiv(args_.m, tempBaseM);
                nCore = CeilDiv(args_.n, tempBaseN);
            }
            while (tempBaseM >= tempBaseN * BASEM_BASEN_RATIO && mCore < coreNumMN / NUM_TWO &&
                   tempBaseM != baseMAlignNum) {
                mCore = mCore * NUM_TWO;
                nCore = coreNumMN / mCore;
                tempBaseM = Align(CeilDiv(args_.m, mCore), baseMAlignNum);
                tempBaseN = Align(CeilDiv(args_.n, nCore), baseNAlignNum);
                mCore = CeilDiv(args_.m, tempBaseM);
                nCore = CeilDiv(args_.n, tempBaseN);
            }

            uint64_t kValueAlign = Align(args_.k, baseKAlignNum);
            uint64_t kValueMax = (platformInfo_.l0aSize / DB_SIZE / hifp8::A_DTYPE_SIZE) / std::max(tempBaseM, tempBaseN);
            if (kValueMax >= baseKAlignNum) {
                kValueMax = FloorAlign(kValueMax, baseKAlignNum);
                runInfo_.baseM = tempBaseM;
                runInfo_.baseN = tempBaseN;
                runInfo_.baseK = std::min(kValueAlign, kValueMax);
                runInfo_.baseK = runInfo_.baseK > BASEK_LIMIT
                                     ? Align(runInfo_.baseK / NUM_TWO, baseKAlignNum)
                                     : runInfo_.baseK;
            }
        }
    }

    void OptimizeEdgeBasicBlock()
    {
        if (args_.transA && !args_.transB) {
            return;
        }
        // Merge tiny M-edge tiles when the K-axis is cache-line aligned so the
        // tail block behaves more like the steady-state region.
        if (runInfo_.mBlockCnt == 1UL) {
            return;
        }
        uint64_t mTailSize = args_.m % runInfo_.baseM;
        bool isInnerAxisAlign = (args_.k * hifp8::A_DTYPE_SIZE) % MTE2_CACHELINE_SIZE == 0UL;
        if (mTailSize > 0UL && isInnerAxisAlign) {
            uint64_t baseTailCntMax = std::min((runInfo_.baseM - mTailSize) / BASIC_BLOCK_SIZE_16, runInfo_.mBlockCnt);
            uint64_t windowSize = std::min(WINDOW_LEN, runInfo_.mBlockCnt);
            uint64_t mainWindowNum = runInfo_.mBlockCnt / windowSize - 1UL;
            uint64_t tailWindowSize = runInfo_.mBlockCnt - mainWindowNum * windowSize;
            uint64_t perfRes = (mainWindowNum + 1UL) * runInfo_.baseM;
            uint64_t mergeWindowNum = 1UL;

            for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax;
                 mergeLen += windowSize, ++mergeWindowNum) {
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

    void CalcTailBasicBlock()
    {
        if (runInfo_.tailBlockCnt == 0UL) {
            return;
        }

        // Non-full-load can split both M and N tail tiles. Grow the heavier
        // edge first, but keep the total tail work within the available cores.
        uint64_t mTile = 1UL;
        uint64_t nTile = 1UL;
        uint64_t preSplit = 1UL;
        uint64_t secSplit = 1UL;
        uint64_t& preSplitValid = runInfo_.mTailSize >= runInfo_.nTailSize ? mTile : nTile;
        uint64_t& secSplitValid = runInfo_.mTailSize >= runInfo_.nTailSize ? nTile : mTile;
        while ((CalUsedCoreNum(preSplit + 1UL, secSplit) <= platformInfo_.aicNum) ||
               (CalUsedCoreNum(preSplit, secSplit + 1UL) <= platformInfo_.aicNum)) {
            if (CalUsedCoreNum(preSplit + 1UL, secSplit) <= platformInfo_.aicNum) {
                preSplitValid = ++preSplit;
            }
            if (CalUsedCoreNum(preSplit, secSplit + 1UL) <= platformInfo_.aicNum) {
                secSplitValid = ++secSplit;
            }
        }

        runInfo_.mTailTile = mTile;
        runInfo_.nTailTile = nTile;
    }

    uint64_t CalUsedCoreNum(uint64_t mTile, uint64_t nTile) const
    {
        return mTile * nTile * runInfo_.tailBlockCnt;
    }

    // --------------------------- L1 tiling -----------------------------------
    //
    // The L1 budget is first reduced by the per-channel scale reservation when
    // needed. The remaining space is evenly split between A and B, following
    // CalL1TilingDepthNotfullload() in adaptive_sliding_window_tiling.h.
    void CalL1Tiling()
    {
        uint64_t scaleReserved = GetPerChannelScaleReservation();
        uint64_t totalL1 = platformInfo_.l1Size;
        CHECK_COND(totalL1 > scaleReserved, "L1 budget is insufficient to reserve per-channel scale.");
        uint64_t leftL1Size = totalL1 - scaleReserved;

        uint64_t baseASize = runInfo_.baseM * runInfo_.baseK * hifp8::A_DTYPE_SIZE;
        uint64_t baseBSize = runInfo_.baseN * runInfo_.baseK * hifp8::B_DTYPE_SIZE;
        CHECK_COND(baseASize > 0UL && baseBSize > 0UL, "Invalid base tile size for L1 tiling.");

        // Evenly split the remaining L1 between A and B; depthA1/B1 >= 2 is
        // guaranteed because baseM/N/K <= 256 keeps per-tile <= 16K bytes.
        runInfo_.depthA1 = leftL1Size / NUM_TWO / baseASize;
        runInfo_.depthB1 = leftL1Size / NUM_TWO / baseBSize;
        runInfo_.depthA1 = std::max(runInfo_.depthA1, DB_SIZE);
        runInfo_.depthB1 = std::max(runInfo_.depthB1, DB_SIZE);
        CalStepKs();
    }

    void CalStepKs()
    {
        // Convert L1 depth into step-K counts and keep A/B synchronized so
        // both sides advance through K with the same outer scheduling cadence.
        runInfo_.stepKa = runInfo_.depthA1 / DB_SIZE;
        runInfo_.stepKb = runInfo_.depthB1 / DB_SIZE;

        if (runInfo_.stepKa * runInfo_.baseK > args_.k) {
            runInfo_.stepKa = CeilDiv(args_.k, runInfo_.baseK);
        }
        if (runInfo_.stepKb * runInfo_.baseK > args_.k) {
            runInfo_.stepKb = CeilDiv(args_.k, runInfo_.baseK);
        }
        if (runInfo_.stepKa > runInfo_.stepKb && runInfo_.stepKb > 0UL) {
            runInfo_.stepKa = runInfo_.stepKa / runInfo_.stepKb * runInfo_.stepKb;
        }
        if (runInfo_.stepKb > runInfo_.stepKa && runInfo_.stepKa > 0UL) {
            runInfo_.stepKb = runInfo_.stepKb / runInfo_.stepKa * runInfo_.stepKa;
        }

        runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE;
        runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE;
    }

    uint64_t GetPerChannelScaleReservation() const
    {
        // Scalars stay in FB; only x2 PERCHANNEL reserves a baseN-sized slot in L1.
        if (config_.x2QuantMode != hifp8::KERNEL_QUANT_PERCHANNEL) {
            return 0UL;
        }
        return runInfo_.baseN * config_.scaleDtypeSize * hifp8::L1_TWO_BUFFER;
    }

    uint8_t CalculateDefaultNBufferNum() const
    {
        // Check whether the four-buffer A/B layout still fits once the
        // per-channel scale reservation is accounted for. Fall back to double buffering
        // otherwise, matching CalculateNBufferNum4Cube() in the sliding-window
        // basic-api tiling.
        uint64_t stepK = std::min(runInfo_.stepKa, runInfo_.stepKb);
        uint64_t kL1 = stepK * runInfo_.baseK;
        uint64_t usedL1Size = (runInfo_.baseN * kL1 * hifp8::B_DTYPE_SIZE) * L1_FOUR_BUFFER;
        usedL1Size += (runInfo_.baseM * kL1 * hifp8::A_DTYPE_SIZE) * L1_FOUR_BUFFER;
        usedL1Size += GetPerChannelScaleReservation();
        uint8_t nBufferNum = static_cast<uint8_t>(usedL1Size < platformInfo_.l1Size ? L1_FOUR_BUFFER : DB_SIZE);
        return nBufferNum;
    }

    void ResolveKernelQuantMode(uint32_t& x1Mode, uint32_t& x2Mode) const
    {
        x1Mode = config_.x1QuantMode;
        x2Mode = config_.x2QuantMode;
    }

    // --------------------------- output --------------------------------------
    void BuildTilingData(QuantMatmulHifp8TilingData& tilingData, uint8_t nBufferNum) const
    {
        uint64_t stepKaOut = runInfo_.stepKa;
        uint64_t stepKbOut = runInfo_.stepKb;
        if (nBufferNum == L1_FOUR_BUFFER) {
            uint64_t stepK = std::min(stepKaOut, stepKbOut);
            stepKaOut = stepK;
            stepKbOut = stepK;
        }
        tilingData = {};
        tilingData.m = static_cast<uint32_t>(args_.m);
        tilingData.n = static_cast<uint32_t>(args_.n);
        tilingData.k = static_cast<uint32_t>(args_.k);
        tilingData.baseM = static_cast<uint32_t>(runInfo_.baseM);
        tilingData.baseN = static_cast<uint32_t>(runInfo_.baseN);
        tilingData.baseK = static_cast<uint32_t>(runInfo_.baseK);
        tilingData.stepKa = static_cast<uint32_t>(stepKaOut);
        tilingData.stepKb = static_cast<uint32_t>(stepKbOut);
        tilingData.kAL1 = static_cast<uint32_t>(stepKaOut * runInfo_.baseK);
        tilingData.kBL1 = static_cast<uint32_t>(stepKbOut * runInfo_.baseK);
        tilingData.nBufferNum = static_cast<uint32_t>(nBufferNum);
        tilingData.dbL0c = static_cast<uint32_t>(runInfo_.dbL0c);
        tilingData.mTailTile = static_cast<uint32_t>(runInfo_.mTailTile);
        tilingData.nTailTile = static_cast<uint32_t>(runInfo_.nTailTile);
        tilingData.mBaseTailSplitCnt = static_cast<uint32_t>(runInfo_.mBaseTailSplitCnt);
        tilingData.nBaseTailSplitCnt = static_cast<uint32_t>(runInfo_.nBaseTailSplitCnt);
        tilingData.mTailMain = static_cast<uint32_t>(runInfo_.mTailMain);
        tilingData.nTailMain = static_cast<uint32_t>(runInfo_.nTailMain);
        tilingData.usedCoreNum = static_cast<uint32_t>(
            (runInfo_.totalBlockCnt > platformInfo_.aicNum || runInfo_.tailBlockCnt == 0UL)
                ? platformInfo_.aicNum
                : runInfo_.tailBlockCnt * runInfo_.mTailTile * runInfo_.nTailTile);
        ResolveKernelQuantMode(tilingData.x1QuantMode, tilingData.x2QuantMode);
    }

    void PrintTilingData(const QuantMatmulHifp8TilingData& tilingData) const
    {
        printf("[QuantMatmulHifp8 Strategy]\n");
        printf("  strategy           : %s\n", TilingName());
        printf("  config x1/x2       : %u / %u\n", config_.x1QuantMode, config_.x2QuantMode);
        printf("[QuantMatmulHifp8 Tiling Data]\n");
        printf("  usedCoreNum        : %u\n", tilingData.usedCoreNum);
        printf("  m                  : %u\n", tilingData.m);
        printf("  n                  : %u\n", tilingData.n);
        printf("  k                  : %u\n", tilingData.k);
        printf("  baseM              : %u\n", tilingData.baseM);
        printf("  baseN              : %u\n", tilingData.baseN);
        printf("  baseK              : %u\n", tilingData.baseK);
        printf("  stepKa             : %u\n", tilingData.stepKa);
        printf("  stepKb             : %u\n", tilingData.stepKb);
        printf("  kAL1               : %u\n", tilingData.kAL1);
        printf("  kBL1               : %u\n", tilingData.kBL1);
        printf("  mTailTile          : %u\n", tilingData.mTailTile);
        printf("  nTailTile          : %u\n", tilingData.nTailTile);
        printf("  mBaseTailSplitCnt  : %u\n", tilingData.mBaseTailSplitCnt);
        printf("  nBaseTailSplitCnt  : %u\n", tilingData.nBaseTailSplitCnt);
        printf("  mTailMain          : %u\n", tilingData.mTailMain);
        printf("  nTailMain          : %u\n", tilingData.nTailMain);
        printf("  nBufferNum         : %u\n", tilingData.nBufferNum);
        printf("  dbL0c              : %u\n", tilingData.dbL0c);
        printf("  x1QuantMode        : %u\n", tilingData.x1QuantMode);
        printf("  x2QuantMode        : %u\n", tilingData.x2QuantMode);
    }
};

#endif // QUANT_MATMUL_HIFP8_TILING_SWAT_H
