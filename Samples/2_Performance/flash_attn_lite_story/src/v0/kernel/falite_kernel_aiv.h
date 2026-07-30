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

#include "falite_kernel_common.h"

namespace FALite {

// V1: S 从 GM 读到 UB, online softmax, P→GM. 核心逻辑复用共享层 SoftmaxAndCastP / WritePToGM.
__aicore__ inline void SoftmaxAndWriteP(
    uint32_t j, uint64_t sHead, uint64_t pHead, AscendC::LocalTensor<float>& sUBLocal,
    AscendC::LocalTensor<bfloat16_t>& pUBLocal, AscendC::LocalTensor<float>& mUBLocal,
    AscendC::LocalTensor<float>& lUBLocal, AscendC::LocalTensor<float>& alphaUBLocal,
    AscendC::GlobalTensor<float>& sGlobal, AscendC::GlobalTensor<bfloat16_t>& pGlobal,
    const FlashAttnLiteTilingData& data)
{
    using namespace AscendC;
    const uint32_t halfBr = data.br / 2, bc = data.bc;
    CrossCoreWaitFlag<GROUP_CROSS_MODE, PIPE_MTE2>(FLAG_S_READY);
    DataCopy(
        sUBLocal, sGlobal[sHead],
        DataCopyParams(
            static_cast<uint16_t>(bc), static_cast<uint16_t>(halfBr * sizeof(float) / C0_BYTES),
            static_cast<uint16_t>((data.br - halfBr) * sizeof(float) / C0_BYTES), 0));
    SetWaitFlag<HardEvent::MTE2_V>(STATIC_EVENT_ID0);
    SoftmaxAndCastP(
        sUBLocal, mUBLocal, lUBLocal, alphaUBLocal, pUBLocal, static_cast<uint16_t>(halfBr), static_cast<uint16_t>(bc),
        data.scale, j == 0);
    WritePToGM(
        pUBLocal, pGlobal, pHead, static_cast<uint16_t>(bc), static_cast<uint16_t>(halfBr),
        static_cast<uint16_t>(data.br));
    CrossCoreSetFlag<GROUP_CROSS_MODE, PIPE_MTE3>(FLAG_P_READY);
}

// V2: ΔO 从 GM 读到 UB, O_acc 更新. 核心逻辑复用共享层 AccumulateDeltaOCore.
__aicore__ inline void AccumulateDeltaO(
    uint64_t oHead, AscendC::LocalTensor<float>& oAccUBLocal, AscendC::LocalTensor<float>& oDeltaUBLocal,
    AscendC::LocalTensor<float>& alphaUBLocal, AscendC::GlobalTensor<float>& dOGlobal,
    const FlashAttnLiteTilingData& data)
{
    using namespace AscendC;
    const uint32_t halfBr = data.br / 2, d = data.headDim;
    CrossCoreWaitFlag<GROUP_CROSS_MODE, PIPE_MTE2>(FLAG_O_READY);
    DataCopy(oDeltaUBLocal, dOGlobal[oHead], halfBr * d);
    SetWaitFlag<HardEvent::MTE2_V>(STATIC_EVENT_ID0);
    AccumulateDeltaOCore(
        oAccUBLocal, oDeltaUBLocal, alphaUBLocal, static_cast<uint16_t>(halfBr), static_cast<uint16_t>(d));
    CrossCoreSetFlag<GROUP_CROSS_MODE, PIPE_V>(FLAG_DONE);
}

__aicore__ inline void ProcessOneTaskAIV(
    uint32_t taskId, uint32_t base, AscendC::LocalTensor<float>& sUBLocal, AscendC::LocalTensor<float>& oDeltaUBLocal,
    AscendC::LocalTensor<float>& oAccUBLocal, AscendC::LocalTensor<bfloat16_t>& pUBLocal,
    AscendC::LocalTensor<float>& mUBLocal, AscendC::LocalTensor<float>& lUBLocal,
    AscendC::LocalTensor<float>& alphaUBLocal, AscendC::GlobalTensor<float>& sGlobal,
    AscendC::GlobalTensor<bfloat16_t>& pGlobal, AscendC::GlobalTensor<float>& dOGlobal,
    AscendC::GlobalTensor<bfloat16_t>& outGlobal, const FlashAttnLiteTilingData& data)
{
    using namespace AscendC;
    const uint32_t halfBr = data.br / 2, bc = data.bc, d = data.headDim;
    uint64_t pHead, outHead;
    InitTaskAIV(oAccUBLocal, mUBLocal, lUBLocal, pHead, outHead, taskId, base, halfBr, d, data);
    const uint64_t sHead = pHead, oHead = static_cast<uint64_t>(taskId) * data.br * d + static_cast<uint64_t>(base) * d;
    for (uint32_t j = 0; j < data.tc; ++j) {
        SoftmaxAndWriteP(j, sHead, pHead, sUBLocal, pUBLocal, mUBLocal, lUBLocal, alphaUBLocal, sGlobal, pGlobal, data);
        AccumulateDeltaO(oHead, oAccUBLocal, oDeltaUBLocal, alphaUBLocal, dOGlobal, data);
    }
    FinalOutput(
        oAccUBLocal, lUBLocal, pUBLocal, outGlobal, outHead, static_cast<uint16_t>(halfBr), static_cast<uint16_t>(d));
}

__aicore__ inline void KernelProcessForAIV(
    __gm__ float* sGMAddr, __gm__ bfloat16_t* pGMAddr, __gm__ float* dOGMAddr, __gm__ bfloat16_t* outGMAddr,
    FlashAttnLiteTilingData data)
{
    using namespace AscendC;
    if ASCEND_IS_AIV {
        const uint32_t halfBr = data.br / 2, useAicNum = data.useAicNum;
        GlobalTensor<float> sGlobal;
        sGlobal.SetGlobalBuffer(sGMAddr);
        GlobalTensor<bfloat16_t> pGlobal, outGlobal;
        pGlobal.SetGlobalBuffer(pGMAddr);
        outGlobal.SetGlobalBuffer(outGMAddr);
        GlobalTensor<float> dOGlobal;
        dOGlobal.SetGlobalBuffer(dOGMAddr);
        const auto& aiv = data.layoutAIV;
        LocalTensor<float> sUBLocal(TPosition::VECCALC, aiv.sUBAddr, aiv.sUBElems);
        LocalTensor<float> oDeltaUBLocal(TPosition::VECCALC, aiv.oDeltaUBAddr, aiv.oDeltaUBElems);
        LocalTensor<float> oAccUBLocal(TPosition::VECCALC, aiv.oAccUBAddr, aiv.oAccUBElems);
        LocalTensor<bfloat16_t> pUBLocal(TPosition::VECCALC, aiv.pUBAddr, aiv.pUBElems);
        LocalTensor<float> mUBLocal(TPosition::VECCALC, aiv.mUBAddr, aiv.rowStatsUBElems);
        LocalTensor<float> lUBLocal(TPosition::VECCALC, aiv.lUBAddr, aiv.rowStatsUBElems);
        LocalTensor<float> alphaUBLocal(TPosition::VECCALC, aiv.alphaUBAddr, aiv.rowStatsUBElems);
        const uint32_t subAivIdx = GetSubBlockIdx();
        const uint32_t aicIdx = GetBlockIdx() / GetSubBlockNum();
        const uint32_t base = subAivIdx * halfBr;
        for (uint32_t taskId = aicIdx; taskId < data.numTasks; taskId += useAicNum)
            ProcessOneTaskAIV(
                taskId, base, sUBLocal, oDeltaUBLocal, oAccUBLocal, pUBLocal, mUBLocal, lUBLocal, alphaUBLocal, sGlobal,
                pGlobal, dOGlobal, outGlobal, data);
    }
}

} // namespace FALite
