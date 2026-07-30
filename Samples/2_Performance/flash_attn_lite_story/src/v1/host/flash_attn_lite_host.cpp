/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../flash_attn_lite.h"
#include "../flash_attn_lite_common.h"
#include "_shared_h.h"

#include <cstdio>
#include <cmath>
#include <tiling/platform/platform_ascendc.h>

namespace {

void ComputeSRAMLayout(FALite::FlashAttnLiteTilingData& data)
{
    const uint32_t d = data.headDim, br = data.br, bc = data.bc, halfBr = br / 2;
    auto& aic = data.layoutAIC;
    // C1 单 Mmad: L0B=br*d(16384), L0C=br*bc(16384)
    aic.qL1Addr = 0;
    aic.qL1Elems = br * d;
    aic.kL1Addr = aic.qL1Addr + aic.qL1Elems * sizeof(uint16_t);
    aic.kL1Elems = bc * d;
    aic.vL1Addr = aic.kL1Addr + aic.kL1Elems * sizeof(uint16_t);
    aic.vL1Elems = bc * d;
    aic.pL1Addr = aic.vL1Addr + aic.vL1Elems * sizeof(uint16_t);
    aic.pL1Elems = br * bc;
    aic.aL0AAddr = 0;
    aic.aL0AElems = bc * d;
    aic.bL0BAddr = 0;
    aic.bL0BElems = d * br;
    aic.cL0CAddr = 0;
    aic.cL0CElems = br * bc;

    auto& aiv = data.layoutAIV;
    aiv.sUBAddr = 0;
    aiv.sUBElems = bc * halfBr;
    aiv.oDeltaUBAddr = aiv.sUBAddr + aiv.sUBElems * sizeof(float);
    aiv.oDeltaUBElems = halfBr * d;
    aiv.oAccUBAddr = aiv.oDeltaUBAddr + aiv.oDeltaUBElems * sizeof(float);
    aiv.oAccUBElems = halfBr * d;
    aiv.pUBAddr = aiv.oAccUBAddr + aiv.oAccUBElems * sizeof(float);
    aiv.pUBElems = bc * halfBr;
    aiv.rowStatsUBElems = halfBr;
    aiv.mUBAddr = aiv.pUBAddr + aiv.pUBElems * sizeof(uint16_t);
    aiv.lUBAddr = aiv.mUBAddr + aiv.rowStatsUBElems * sizeof(float);
    aiv.alphaUBAddr = aiv.lUBAddr + aiv.rowStatsUBElems * sizeof(float);
    if (aiv.pUBElems < halfBr * d)
        std::fprintf(stderr, "falite_v1: P workspace too small\n");
}

} // namespace

bool FlashAttnLiteNPU(
    uint8_t* dQ, uint8_t* dK, uint8_t* dV, uint8_t* dOut, uint32_t B, uint32_t S, float scale, uint32_t reqCores,
    aclrtStream stream)
{
    auto* plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aicNum = plat->GetCoreNumAic();
    if (reqCores > 0 && reqCores < aicNum)
        aicNum = reqCores;
    FALite::FlashAttnLiteTilingData data{};
    const char* err = FALite::InitAndCheckTiling(B, S, scale, aicNum, data, ComputeSRAMLayout);
    if (err) {
        std::fprintf(stderr, "falite_v1 tiling:%s\n", err);
        return false;
    }
    // S/ΔO 走 UB, 仅需 P 中间 buffer
    size_t pBytes = (size_t)data.numTasks * FALite::BC * FALite::BR * sizeof(uint16_t);
    void* dP = nullptr;
    if (aclrtMalloc(&dP, pBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::fprintf(stderr, "falite_v1: P alloc failed\n");
        return false;
    }
    std::printf("falite_v1: AIC=%u S=%u tr=%u tc=%u tasks=%u\n", aicNum, S, data.tr, data.tc, data.numTasks);
    FALite::LaunchFlashAttnLiteKernel(dQ, dK, dV, (uint8_t*)dP, dOut, data, stream);
    aclrtSynchronizeStream(stream);
    aclrtFree(dP);
    return true;
}
