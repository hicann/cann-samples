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

#include <cstdint>

namespace FALite {

constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t BLOCK_K = 64;
constexpr uint32_t BR = 128;
constexpr uint32_t BC = 128;

struct SRAMLayoutAIC {
    uint32_t qL1Addr, qL1Elems;
    uint32_t kL1Addr, kL1Elems;
    uint32_t vL1Addr, vL1Elems;
    uint32_t pL1Addr, pL1Elems;
    uint32_t aL0AAddr, aL0AElems;
    uint32_t bL0BAddr, bL0BElems;
    uint32_t cL0CAddr, cL0CElems;
};

struct SRAMLayoutAIV {
    uint32_t sUBAddr, sUBElems;
    uint32_t oDeltaUBAddr, oDeltaUBElems;
    uint32_t oAccUBAddr, oAccUBElems;
    uint32_t pUBAddr, pUBElems;
    uint32_t mUBAddr, lUBAddr, alphaUBAddr;
    uint32_t rowStatsUBElems;
};

struct FlashAttnLiteTilingData {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t headDim;
    float scale;
    uint32_t br, bc, tr, tc;
    uint32_t useAicNum;
    uint32_t numTasks;
    SRAMLayoutAIC layoutAIC;
    SRAMLayoutAIV layoutAIV;
};

// internal: 7-GM-buffer kernel launch, S/P/ΔO 由 host 内部分配
void LaunchFlashAttnLiteKernel(
    uint8_t* dQ, uint8_t* dK, uint8_t* dV, uint8_t* dS, uint8_t* dP, uint8_t* dDO, uint8_t* dOut,
    const FlashAttnLiteTilingData& data, void* stream);

} // namespace FALite
