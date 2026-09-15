/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <array>

#include "block_attn_res_tiling_common.h"
#include "tiling/block_attn_res_prepare_tiling_data.h"

namespace BlockAttnResStory {

struct PreparePlan {
    bool useMix = false;
    uint32_t blockDim = 0;
    uint64_t workspaceBytes = 0;
    optiling::BlockAttnResPrepareTilingData vectorData{};
    optiling::BlockAttnResPrepareMixTilingData mixData{};
};

inline bool TrySetVectorTile(
    const ShapeConfig& shape, const PlatformInfo& platform, bool multipleRounds, uint64_t baseD,
    optiling::BlockAttnResPrepareTilingData& data)
{
    constexpr uint64_t RESERVED = 8UL * 1024UL;
    constexpr uint64_t FP32_BYTES = 4UL, REG_ELEMS = 64UL, STAT_ELEMS = 2UL * (64UL + 8UL);
    const uint64_t dAlign = AlignUp<uint64_t>(shape.d, REG_ELEMS);
    const uint64_t baseDAlign = AlignUp<uint64_t>(baseD, REG_ELEMS);
    const uint64_t dTileNum = CeilDiv<uint64_t>(shape.d, baseD);
    const uint64_t qBuffers = (dTileNum > 1 || multipleRounds) ? 2 : 1;
    const uint64_t oBuffers = qBuffers;
    const uint64_t vLoops = dTileNum * shape.n;
    const uint64_t vBuffers = (vLoops > 1 || multipleRounds) ? 2 : 1;
    const uint64_t fixedBytes = ((qBuffers + vBuffers + oBuffers) * baseDAlign + 2UL * STAT_ELEMS) * FP32_BYTES;
    if (fixedBytes > platform.ubSize - RESERVED) {
        return false;
    }
    const uint64_t cacheRows =
        std::min<uint64_t>(shape.n, (platform.ubSize - RESERVED - fixedBytes) / (dAlign * FP32_BYTES));
    data.baseD = static_cast<uint32_t>(baseD);
    data.statUbElems = static_cast<uint32_t>(STAT_ELEMS);
    data.vCacheRows = static_cast<uint8_t>(cacheRows);
    data.qBufferNum = static_cast<uint8_t>(qBuffers);
    data.vBufferNum = static_cast<uint8_t>(vBuffers);
    data.oBufferNum = static_cast<uint8_t>(oBuffers);
    return true;
}

inline PreparePlan BuildVectorPreparePlan(const ShapeConfig& shape, const PlatformInfo& platform)
{
    if (platform.ubSize <= 8UL * 1024UL || shape.d == 0) {
        throw std::runtime_error("Prepare Vector has no usable UB or D dimension");
    }
    const uint64_t totalWork = static_cast<uint64_t>(shape.t) * shape.s;
    const WorkDistribution distribution = Distribute(totalWork, platform.aivNum);
    const bool multipleRounds = distribution.tailBlockFactor > 1U;
    const std::array<uint64_t, 4> candidates = {
        shape.d, std::min<uint64_t>(shape.d, 1024UL), std::min<uint64_t>(shape.d, 512UL),
        std::min<uint64_t>(shape.d, 256UL)};
    PreparePlan plan;
    auto& data = plan.vectorData;
    data.totalT = shape.t;
    data.totalN = static_cast<uint8_t>(shape.n);
    data.totalS = shape.s;
    data.totalWorkUnits = static_cast<uint32_t>(totalWork);
    data.totalD = shape.d;
    data.usedCoreNum = static_cast<uint16_t>(distribution.usedCoreNum);
    data.bigCoreNum = static_cast<uint16_t>(distribution.bigCoreNum);
    data.blockFactor = distribution.blockFactor;
    data.tailBlockFactor = distribution.tailBlockFactor;
    data.eps = shape.eps;
    plan.blockDim = distribution.usedCoreNum;
    uint64_t previous = 0;
    for (uint64_t baseD : candidates) {
        if (baseD == 0 || baseD == previous)
            continue;
        previous = baseD;
        if (TrySetVectorTile(shape, platform, multipleRounds, baseD, data))
            return plan;
    }
    throw std::runtime_error("no Prepare Vector tile fits UB");
}

struct MixCandidate {
    uint64_t baseT = 0, baseS = 0, baseD = 0, baseDAlign = 0, dTileNum = 0, mm1NAlign = 0;
    uint64_t qBuffers = 0, vL1Buffers = 0, vUbBuffers = 2;
    uint64_t qElems = 0, vL1Elems = 0, eElems = 0, vUbElems = 0, dotElems = 0, reduceElems = 64, softmaxElems = 0;
    uint32_t sAlign = 0;
};

inline bool MixFits(const MixCandidate& c, uint32_t nAlign, const PlatformInfo& p)
{
    constexpr uint64_t BYTES = 4, RESERVED = 8UL * 1024UL;
    const uint64_t l1 = (c.qBuffers * c.qElems + c.vL1Buffers * c.vL1Elems + c.eElems) * BYTES;
    const uint64_t ub = (c.vUbBuffers * c.vUbElems + c.dotElems + c.reduceElems + c.softmaxElems) * BYTES;
    const uint64_t mm1K = std::min<uint64_t>(c.baseDAlign, 64);
    const uint64_t mm1L1Slot = (c.sAlign + c.mm1NAlign) * c.baseDAlign * BYTES;
    const uint64_t mm2L1Slot =
        (static_cast<uint64_t>(c.sAlign) * nAlign + static_cast<uint64_t>(nAlign) * c.baseDAlign) * BYTES;
    return p.ubSize > RESERVED && p.l1Size > RESERVED && l1 <= p.l1Size - RESERVED && ub <= p.ubSize - RESERVED &&
           mm1L1Slot <= p.l1Size / 4 && mm2L1Slot <= p.l1Size / 4 &&
           static_cast<uint64_t>(c.sAlign) * mm1K * BYTES <= p.l0aSize &&
           static_cast<uint64_t>(c.sAlign) * nAlign * BYTES <= p.l0aSize && c.mm1NAlign * mm1K * BYTES <= p.l0bSize &&
           static_cast<uint64_t>(nAlign) * c.baseDAlign * BYTES <= p.l0bSize &&
           static_cast<uint64_t>(c.sAlign) * c.mm1NAlign * BYTES <= p.l0cSize &&
           static_cast<uint64_t>(c.sAlign) * c.baseDAlign * BYTES <= p.l0cSize;
}

inline MixCandidate MakeMixCandidate(
    const ShapeConfig& shape, uint64_t baseT, uint64_t baseS, uint64_t baseD, uint32_t nAlign)
{
    MixCandidate c;
    c.baseT = baseT;
    c.baseS = baseS;
    c.baseD = baseD;
    c.baseDAlign = AlignUp<uint64_t>(baseD, 16);
    c.sAlign = AlignUp<uint32_t>(baseS, 16);
    c.dTileNum = CeilDiv<uint64_t>(shape.d, baseD);
    c.mm1NAlign = AlignUp<uint64_t>(baseT * shape.n, 16);
    c.qBuffers = c.dTileNum > 1 ? 2 : 1;
    c.vL1Buffers = c.qBuffers;
    c.qElems = static_cast<uint64_t>(c.sAlign) * c.baseDAlign;
    c.vL1Elems = c.mm1NAlign * c.baseDAlign;
    c.eElems = static_cast<uint64_t>(c.sAlign) * nAlign;
    c.vUbElems = static_cast<uint64_t>(nAlign) * c.baseDAlign;
    c.dotElems = static_cast<uint64_t>(c.sAlign) * nAlign + 64;
    c.softmaxElems = 2UL * c.sAlign;
    return c;
}

inline bool FindMixTile(
    const ShapeConfig& shape, const PlatformInfo& platform, uint64_t baseS, uint32_t mixCores, MixCandidate& selected)
{
    const uint32_t nAlign = AlignUp<uint32_t>(shape.n, 16U);
    const uint64_t sTiles = CeilDiv<uint64_t>(shape.s, baseS);
    const uint64_t baselineWork = static_cast<uint64_t>(shape.t) * sTiles;
    const uint64_t baselineTokenWork = CeilDiv<uint64_t>(baselineWork, std::min<uint64_t>(mixCores, baselineWork));
    const std::array<uint64_t, 4> tCandidates = {8, 4, 2, 1};
    const std::array<uint64_t, 3> dCandidates = {
        std::min<uint64_t>(shape.d, 512), std::min<uint64_t>(shape.d, 256), std::min<uint64_t>(shape.d, 128)};
    for (uint64_t baseT : tCandidates) {
        if (baseT > shape.t || baseT * shape.n > 16)
            continue;
        const uint64_t groupedWork = CeilDiv<uint64_t>(shape.t, baseT) * sTiles;
        const uint64_t groupedTokenWork =
            CeilDiv<uint64_t>(groupedWork, std::min<uint64_t>(mixCores, groupedWork)) * baseT;
        if (groupedTokenWork > baselineTokenWork)
            continue;
        uint64_t previousD = 0;
        for (uint64_t baseD : dCandidates) {
            if (baseD == 0 || baseD == previousD)
                continue;
            previousD = baseD;
            const MixCandidate candidate = MakeMixCandidate(shape, baseT, baseS, baseD, nAlign);
            if (MixFits(candidate, nAlign, platform)) {
                selected = candidate;
                return true;
            }
        }
    }
    return false;
}

inline void SetMixPreparePlan(
    const ShapeConfig& shape, const PlatformInfo& platform, const MixCandidate& selected, uint32_t mixCores,
    PreparePlan& plan)
{
    const uint32_t nAlign = AlignUp<uint32_t>(shape.n, 16U);
    const uint32_t dAlign = AlignUp<uint32_t>(shape.d, 16U);
    auto& data = plan.mixData;
    data.nAlign = nAlign;
    data.dAlign = dAlign;
    data.baseT = static_cast<uint32_t>(selected.baseT);
    data.baseS = static_cast<uint32_t>(selected.baseS);
    data.baseD = static_cast<uint32_t>(selected.baseD);
    data.baseDAlign = static_cast<uint32_t>(selected.baseDAlign);
    data.sAlign = selected.sAlign;
    data.dTileNum = static_cast<uint32_t>(selected.dTileNum);
    data.mm1NAlign = static_cast<uint32_t>(selected.mm1NAlign);
    data.qL1BufferNum = static_cast<uint8_t>(selected.qBuffers);
    data.vL1BufferNum = static_cast<uint8_t>(selected.vL1Buffers);
    data.vUbBufferNum = static_cast<uint8_t>(selected.vUbBuffers);
    data.qL1Elems = selected.qElems;
    data.vL1Elems = selected.vL1Elems;
    data.eL1Elems = selected.eElems;
    data.vUbElems = selected.vUbElems;
    data.dotUbElems = selected.dotElems;
    data.reduceUbElems = selected.reduceElems;
    data.softmaxUbElems = selected.softmaxElems;
    data.sTileNum = static_cast<uint32_t>(CeilDiv<uint64_t>(shape.s, selected.baseS));
    const uint64_t totalWork = CeilDiv<uint64_t>(shape.t, selected.baseT) * data.sTileNum;
    const WorkDistribution distribution = Distribute(totalWork, mixCores);
    data.totalT = shape.t;
    data.totalN = static_cast<uint8_t>(shape.n);
    data.totalS = shape.s;
    data.totalWorkUnits = static_cast<uint32_t>(totalWork);
    data.totalD = shape.d;
    data.usedCoreNum = static_cast<uint16_t>(distribution.usedCoreNum);
    data.aicCoreNum = static_cast<uint16_t>(platform.aicNum);
    data.aivCoreNum = static_cast<uint16_t>(platform.aivNum);
    data.eps = shape.eps;
    const uint64_t dotWorkspace = selected.baseS * selected.mm1NAlign;
    const uint64_t eWorkspace = static_cast<uint64_t>(selected.sAlign) * nAlign;
    const uint64_t eBufferCount = std::min<uint64_t>(shape.t, optiling::BLOCK_ATTN_RES_PREPARE_E_BUFFER_NUM);
    data.workspacePerCoreElems = dotWorkspace + eBufferCount * eWorkspace;
    plan.useMix = true;
    plan.blockDim = distribution.usedCoreNum;
    plan.workspaceBytes = data.workspacePerCoreElems * plan.blockDim * sizeof(float);
}

inline bool TryBuildMixPreparePlan(const ShapeConfig& shape, const PlatformInfo& platform, PreparePlan& plan)
{
    if (platform.aicNum == 0 || platform.aivNum != platform.aicNum * 2U || shape.t < 32 || shape.s < 16 ||
        shape.d < 256 || shape.t > std::numeric_limits<uint32_t>::max() / shape.d) {
        return false;
    }
    const uint32_t mixCores = std::min(platform.aicNum, platform.aivNum / 2U);
    const std::array<uint64_t, 5> sCandidates = {
        shape.s, std::min<uint64_t>(shape.s, 128), std::min<uint64_t>(shape.s, 64), std::min<uint64_t>(shape.s, 32),
        std::min<uint64_t>(shape.s, 16)};
    MixCandidate selected;
    uint64_t previousS = 0;
    for (uint64_t baseS : sCandidates) {
        if (baseS == 0 || baseS == previousS)
            continue;
        previousS = baseS;
        if (FindMixTile(shape, platform, baseS, mixCores, selected)) {
            SetMixPreparePlan(shape, platform, selected, mixCores, plan);
            return true;
        }
    }
    return false;
}

inline PreparePlan BuildPreparePlan(const ShapeConfig& shape, const PlatformInfo& platform)
{
    PreparePlan mixPlan;
    const bool mixAvailable = TryBuildMixPreparePlan(shape, platform, mixPlan);
    if (shape.prepareTemplate == PrepareTemplate::MIX) {
        if (!mixAvailable)
            throw std::runtime_error("requested Prepare Mix template is not capable");
        return mixPlan;
    }
    if (shape.prepareTemplate == PrepareTemplate::AUTO && mixAvailable)
        return mixPlan;
    return BuildVectorPreparePlan(shape, platform);
}

inline void ValidatePrepareShape(const ShapeConfig& shape)
{
    if (shape.t == 0 || shape.s == 0 || shape.n == 0 || shape.n > 64 || shape.d == 0 || shape.d > 8192 ||
        !(shape.eps > 0.0F)) {
        throw std::invalid_argument("Prepare requires T,S>0, 1<=N<=64, 1<=D<=8192 and eps>0");
    }
    if (shape.slot >= shape.s)
        throw std::invalid_argument("slot must be smaller than S");
}

} // namespace BlockAttnResStory
