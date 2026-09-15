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

#include "block_attn_res_tiling_common.h"
#include "tiling/block_attn_res_update_tiling_data.h"

namespace BlockAttnResStory {

struct UpdatePlan {
    uint32_t blockDim = 0;
    bool singleTile = false;
    BlockAttnResUpdateTilingData data{};
};

inline UpdatePlan BuildUpdatePlan(const ShapeConfig& shape, const PlatformInfo& platform)
{
    constexpr uint64_t RESERVED = 8UL * 1024UL;
    constexpr uint64_t MAX_COPY_BLOCKS = 4095;
    if (shape.t == 0 || shape.d == 0 || platform.aivNum == 0 || platform.ubSize <= RESERVED) {
        throw std::invalid_argument("invalid Update shape, core count or UB size");
    }
    const uint64_t tPerCore = CeilDiv<uint64_t>(shape.t, platform.aivNum);
    if (tPerCore == 0) {
        throw std::invalid_argument("Update rows per core must be positive");
    }
    const uint64_t usedCores = CeilDiv<uint64_t>(shape.t, tPerCore);
    const uint64_t remainder = shape.t % tPerCore;
    const uint32_t lastT = static_cast<uint32_t>(remainder == 0 ? tPerCore : remainder);
    const uint32_t dFp32 = AlignUp<uint32_t>(shape.d, 8U);
    const uint32_t dBf16 = AlignUp<uint32_t>(shape.d, 16U);
    const uint64_t maxTile = CeilDiv<uint64_t>(tPerCore, 2);
    const uint64_t bytesPerT = static_cast<uint64_t>(dFp32) * 4 * 2 + static_cast<uint64_t>(dBf16) * 2 + 3 * 4;
    const uint64_t queryBytes = static_cast<uint64_t>(dFp32) * 4;
    const uint64_t usable = platform.ubSize - RESERVED;
    if (usable <= queryBytes)
        throw std::runtime_error("Update query does not fit UB");
    uint64_t tileT =
        std::min<uint64_t>(std::min<uint64_t>(maxTile, MAX_COPY_BLOCKS), (usable - queryBytes) / (2 * bytesPerT));
    auto calcBytes = [&](uint32_t candidate, uint32_t& statsStride) {
        statsStride = AlignUp<uint32_t>(candidate, 8U);
        const uint64_t partial = static_cast<uint64_t>(candidate) * dFp32 * 4;
        const uint64_t deltaH = static_cast<uint64_t>(candidate) * dBf16 * 2;
        const uint64_t stats = static_cast<uint64_t>(3) * statsStride * 4;
        return queryBytes + 2 * (partial + deltaH + partial + stats);
    };
    uint32_t statsStride = 0;
    while (tileT > 0 && calcBytes(static_cast<uint32_t>(tileT), statsStride) > usable)
        --tileT;
    if (tileT == 0)
        throw std::runtime_error("one Update D row does not fit double-buffered UB");
    const uint64_t tileCount = CeilDiv<uint64_t>(tPerCore, tileT);
    tileT = CeilDiv<uint64_t>(tPerCore, tileCount);
    calcBytes(static_cast<uint32_t>(tileT), statsStride);

    UpdatePlan plan;
    plan.blockDim = static_cast<uint32_t>(usedCores);
    plan.singleTile = tPerCore <= tileT;
    plan.data.dSize = shape.d;
    plan.data.tPerCore = static_cast<uint32_t>(tPerCore);
    plan.data.lastTPerCore = lastT;
    plan.data.tileT = static_cast<uint32_t>(tileT);
    plan.data.statsTStride = statsStride;
    plan.data.eps = shape.eps;
    plan.data.invD = 1.0F / static_cast<float>(shape.d);
    plan.data.usedCoreNum = static_cast<uint16_t>(usedCores);
    return plan;
}

} // namespace BlockAttnResStory
