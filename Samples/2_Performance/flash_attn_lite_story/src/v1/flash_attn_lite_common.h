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
    uint32_t qL1Addr, qL1Elems, kL1Addr, kL1Elems, vL1Addr, vL1Elems, pL1Addr, pL1Elems;
    uint32_t aL0AAddr, aL0AElems, bL0BAddr, bL0BElems, cL0CAddr, cL0CElems;
};

struct SRAMLayoutAIV {
    uint32_t sUBAddr, sUBElems, oDeltaUBAddr, oDeltaUBElems, oAccUBAddr, oAccUBElems;
    uint32_t pUBAddr, pUBElems, mUBAddr, lUBAddr, alphaUBAddr, rowStatsUBElems;
};

struct FlashAttnLiteTilingData {
    uint32_t batchSize, seqLen, headDim;
    float scale;
    uint32_t br, bc, tr, tc, useAicNum, numTasks;
    SRAMLayoutAIC layoutAIC;
    SRAMLayoutAIV layoutAIV;
};

void LaunchFlashAttnLiteKernel(
    uint8_t* dQ, uint8_t* dK, uint8_t* dV, uint8_t* dP, uint8_t* dOut, const FlashAttnLiteTilingData& data,
    void* stream);

} // namespace FALite
