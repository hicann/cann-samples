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

constexpr AscendC::TEventID EV_K_L1_READY = 1;
constexpr AscendC::TEventID EV_QK_L0_READY = 0;
constexpr AscendC::TEventID EV_S_L0C_READY = 0;
constexpr AscendC::TEventID EV_P_L1_DONE = 0;
constexpr AscendC::TEventID EV_V_L1_READY = 2;
constexpr AscendC::TEventID EV_PV_L0_READY = 1;
constexpr AscendC::TEventID EV_DO_L0C_READY = 1;
constexpr AscendC::TEventID EV_Q_L1_READY = 0;

// C1: K×Q^T → S^T→GM. Q 已在 L1.
__aicore__ inline void CubeStage1(
    AscendC::LocalTensor<bfloat16_t>& qL1Local, AscendC::LocalTensor<bfloat16_t>& kL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL0ALocal, AscendC::LocalTensor<bfloat16_t>& bL0BLocal,
    AscendC::LocalTensor<float>& mmadL0CLocal, AscendC::GlobalTensor<float>& sGlobal, uint64_t sOff,
    AscendC::GlobalTensor<bfloat16_t>& kGlobal, const FlashAttnLiteTilingData& data, uint32_t j, uint32_t batchIdx)
{
    using namespace AscendC;
    if ASCEND_IS_AIC {
        const uint32_t br = data.br, bc = data.bc, d = data.headDim;
        const uint64_t kOff = static_cast<uint64_t>(batchIdx) * data.seqLen * d + static_cast<uint64_t>(j) * bc * d;
        CopyGmToL1<bfloat16_t>(kL1Local, kGlobal[kOff], bc, d, d);
        SetWaitFlag<HardEvent::MTE2_MTE1>(EV_K_L1_READY);
        CopyL1ToL0A<bfloat16_t>(aL0ALocal, kL1Local, bc, d, bc, d);
        CopyL1ToL0B<bfloat16_t>(bL0BLocal, qL1Local, br, d, d, br, false);
        SetWaitFlag<HardEvent::MTE1_M>(EV_QK_L0_READY);
        CubeMmad<float, bfloat16_t, bfloat16_t>(mmadL0CLocal, aL0ALocal, bL0BLocal, bc, br, d, true);
        SetWaitFlag<HardEvent::M_FIX>(EV_S_L0C_READY);
        FixpipeL0CToGM<float, float>(sGlobal[sOff], mmadL0CLocal, bc, br, data.br);
        PipeBarrier<PIPE_FIX>();
    }
}

// C2: P×V → ΔO→GM. 单 Mmad.
__aicore__ inline void CubeStage2(
    AscendC::LocalTensor<bfloat16_t>& pL1Local, AscendC::LocalTensor<bfloat16_t>& vL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL0ALocal, AscendC::LocalTensor<bfloat16_t>& bL0BLocal,
    AscendC::LocalTensor<float>& mmadL0CLocal, AscendC::GlobalTensor<float>& dOGlobal, uint64_t dOOff,
    AscendC::GlobalTensor<bfloat16_t>& vGlobal, AscendC::GlobalTensor<bfloat16_t>& pGlobal,
    const FlashAttnLiteTilingData& data, uint32_t batchIdx, uint32_t taskId, uint32_t j)
{
    using namespace AscendC;
    if ASCEND_IS_AIC {
        const uint32_t br = data.br, bc = data.bc, d = data.headDim;
        const uint64_t pTaskBase = static_cast<uint64_t>(taskId) * bc * br;
        LoadPTransToL0A(pL1Local, aL0ALocal, pGlobal, pTaskBase, data);
        SetWaitFlag<HardEvent::MTE1_MTE2>(EV_P_L1_DONE);
        const uint64_t vOff = static_cast<uint64_t>(batchIdx) * data.seqLen * d + static_cast<uint64_t>(j) * bc * d;
        CopyGmToL1<bfloat16_t>(vL1Local, vGlobal[vOff], bc, d, d);
        SetWaitFlag<HardEvent::MTE2_MTE1>(EV_V_L1_READY);
        CopyL1ToL0B<bfloat16_t>(bL0BLocal, vL1Local, bc, d, bc, d, true);
        SetWaitFlag<HardEvent::MTE1_M>(EV_PV_L0_READY);
        CubeMmad<float, bfloat16_t, bfloat16_t>(mmadL0CLocal, aL0ALocal, bL0BLocal, br, d, bc, true);
        SetWaitFlag<HardEvent::M_FIX>(EV_DO_L0C_READY);
        FixpipeL0CToGM<float, float>(dOGlobal[dOOff], mmadL0CLocal, br, d, d);
        PipeBarrier<PIPE_FIX>();
    }
}

__aicore__ inline void ProcessOneTaskAIC(
    uint32_t taskId, AicCubeWorkspace<FlashAttnLiteTilingData>& ws,
    AscendC::GlobalTensor<float>& dOGlobal, AscendC::GlobalTensor<bfloat16_t>& qGlobal,
    AscendC::GlobalTensor<bfloat16_t>& kGlobal, AscendC::GlobalTensor<bfloat16_t>& vGlobal,
    AscendC::GlobalTensor<bfloat16_t>& pGlobal, AscendC::GlobalTensor<float>& sGlobal,
    const FlashAttnLiteTilingData& data)
{
    using namespace AscendC;
    const uint32_t batchIdx = taskId / data.tr, tileIdx = taskId % data.tr;
    const uint32_t br = data.br, d = data.headDim;
    const uint64_t qOff = static_cast<uint64_t>(batchIdx) * data.seqLen * d + static_cast<uint64_t>(tileIdx) * br * d;
    CopyGmToL1<bfloat16_t>(ws.qL1, qGlobal[qOff], br, d, d);
    SetWaitFlag<HardEvent::MTE2_MTE1>(EV_Q_L1_READY);

    for (uint32_t j = 0; j < data.tc; ++j) {
        if (j > 0) CrossCoreWaitFlag<GROUP_CROSS_MODE, PIPE_MTE2>(FLAG_DONE);
        const uint64_t sOff = static_cast<uint64_t>(taskId) * data.bc * data.br;
        CubeStage1(ws.qL1, ws.kL1, ws.aL0A, ws.bL0B, ws.l0C, sGlobal, sOff, kGlobal, data, j, batchIdx);
        CrossCoreSetFlag<GROUP_CROSS_MODE, PIPE_FIX>(FLAG_S_READY);
        CrossCoreWaitFlag<GROUP_CROSS_MODE, PIPE_MTE2>(FLAG_P_READY);
        const uint64_t dOOff = static_cast<uint64_t>(taskId) * data.br * d;
        CubeStage2(ws.pL1, ws.vL1, ws.aL0A, ws.bL0B, ws.l0C, dOGlobal, dOOff, vGlobal, pGlobal, data, batchIdx, taskId,
                   j);
        CrossCoreSetFlag<GROUP_CROSS_MODE, PIPE_FIX>(FLAG_O_READY);
    }
    CrossCoreWaitFlag<GROUP_CROSS_MODE, PIPE_MTE2>(FLAG_DONE);
}

__aicore__ inline void KernelProcessForAIC(
    __gm__ bfloat16_t* qGMAddr, __gm__ bfloat16_t* kGMAddr, __gm__ bfloat16_t* vGMAddr, __gm__ float* sGMAddr,
    __gm__ bfloat16_t* pGMAddr, __gm__ float* dOGMAddr, FlashAttnLiteTilingData data)
{
    using namespace AscendC;
    if ASCEND_IS_AIC {
        GlobalTensor<bfloat16_t> qGlobal, kGlobal, vGlobal, pGlobal;
        qGlobal.SetGlobalBuffer(qGMAddr);
        kGlobal.SetGlobalBuffer(kGMAddr);
        vGlobal.SetGlobalBuffer(vGMAddr);
        pGlobal.SetGlobalBuffer(pGMAddr);
        GlobalTensor<float> sGlobal, dOGlobal;
        sGlobal.SetGlobalBuffer(sGMAddr);
        dOGlobal.SetGlobalBuffer(dOGMAddr);
        auto ws = MakeAicCubeWorkspace(data);
        for (uint32_t taskId = GetBlockIdx(); taskId < data.numTasks; taskId += GetBlockNum())
            ProcessOneTaskAIC(taskId, ws, dOGlobal, qGlobal, kGlobal, vGlobal, pGlobal, sGlobal, data);
    }
}

} // namespace FALite
