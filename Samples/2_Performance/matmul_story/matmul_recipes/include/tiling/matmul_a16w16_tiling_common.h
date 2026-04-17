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
 * \file matmul_tiling_common.h
 * \brief Shared constants and runtime state for MATMUL tiling generation.
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#include "tiling/platform/platform_ascendc.h"

struct MatmulA16W16PlatformInfo {
    uint64_t aicNum = 0;
    uint64_t aivNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l2Size = 0;
    uint64_t l0cSize = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t btSize = 0;
    platform_ascendc::SocVersion socVersion;
};

struct MatmulA16W16Args {
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t k = 0;
    bool hasBias = false;
};

struct MatmulA16W16RunInfo {
    uint64_t usedCoreNum = 1;
    uint64_t baseM = 256;
    uint64_t baseN = 256;
    uint64_t baseK = 256;
    uint64_t mL1 = 256;
    uint64_t nL1 = 256;
    uint64_t kL1 = 256;
    uint64_t mBlockCnt = 1;
    uint64_t nBlockCnt = 1;
    uint64_t totalBlockCnt = 1;
    uint64_t tailBlockCnt = 0;
    uint64_t mTailSize = 0;
    uint64_t nTailSize = 0;
    uint64_t mTailTile = 1;
    uint64_t nTailTile = 1;
    uint64_t mTailCnt = 1;
    uint64_t nTailCnt = 1;
    uint32_t mBaseTailSplitCnt = 1;
    uint32_t nBaseTailSplitCnt = 1;
    uint32_t mTailMain = 0;
    uint32_t nTailMain = 0;
    uint64_t depthA1 = 2;
    uint64_t depthB1 = 2;
    uint64_t stepKa = 1;
    uint64_t stepKb = 1;
    uint64_t l1BufferNum = 2;
    uint64_t dbL0c = 1;
};

