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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "tiling/platform/platform_ascendc.h"

namespace BlockAttnResStory {

template <typename T>
inline T CeilDiv(T value, T factor)
{
    if (factor == 0) {
        throw std::invalid_argument("CeilDiv factor must be positive");
    }
    return value / factor + static_cast<T>(value % factor != 0);
}

template <typename T>
inline T AlignUp(T value, T factor)
{
    return CeilDiv(value, factor) * factor;
}

enum class PrepareTemplate
{
    AUTO,
    VECTOR,
    MIX
};

struct ShapeConfig {
    uint32_t t = 0;
    uint32_t n = 0;
    uint32_t s = 0;
    uint32_t d = 0;
    uint64_t validBlocks = 0;
    uint32_t slot = 0;
    float eps = 1.0e-6F;
    PrepareTemplate prepareTemplate = PrepareTemplate::AUTO;
    uint32_t warmup = 1;
    uint32_t repeat = 1;
    std::string compareMethod = "isclose";
};

struct PlatformInfo {
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
};

inline PlatformInfo QueryPlatform()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        throw std::runtime_error("failed to query AscendC platform");
    }
    PlatformInfo info;
    info.aicNum = platform->GetCoreNumAic();
    info.aivNum = platform->GetCoreNumAiv();
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubSize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, info.l1Size);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, info.l0aSize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, info.l0bSize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, info.l0cSize);
    if (info.aicNum == 0 || info.aivNum == 0 || info.ubSize == 0) {
        throw std::runtime_error("invalid AscendC platform resources");
    }
    return info;
}

struct WorkDistribution {
    uint32_t usedCoreNum = 0;
    uint32_t blockFactor = 0;
    uint32_t bigCoreNum = 0;
    uint32_t tailBlockFactor = 0;
};

inline WorkDistribution Distribute(uint64_t totalWork, uint32_t maxCores)
{
    if (totalWork == 0 || maxCores == 0 || totalWork > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("invalid work distribution");
    }
    const uint64_t used = std::min<uint64_t>(totalWork, maxCores);
    WorkDistribution result;
    result.usedCoreNum = static_cast<uint32_t>(used);
    result.blockFactor = static_cast<uint32_t>(totalWork / used);
    result.bigCoreNum = static_cast<uint32_t>(totalWork % used);
    result.tailBlockFactor = result.blockFactor + static_cast<uint32_t>(result.bigCoreNum != 0);
    return result;
}

} // namespace BlockAttnResStory
