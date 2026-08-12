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

#include "kdalite_kernel_common.h"

namespace KDALite {

// L0A/L0B 作为一组 Mmad 输入在 MTE1 和 M 之间交接; L0C 在 M 和 FIX 之间交接.
constexpr MutexId MUTEX_PREP_L0AB = 0;
constexpr MutexId MUTEX_PREP_L0C = 1;

__aicore__ inline void FixPrepareResultToAiv(
    const AscendC::LocalTensor<float>& dstUBLocal, const AscendC::LocalTensor<float>& srcL0CLocal, uint32_t subBlockIdx)
{
    using namespace AscendC;

    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> params;
    params.nSize = CHUNK_SIZE;
    params.mSize = CHUNK_SIZE;
    params.srcStride = CeilAlign<uint32_t>(CHUNK_SIZE, CUBE_BLOCK);
    params.dstStride = CHUNK_SIZE;
    // 两个 chunk 的结果分别写入 AIV0/AIV1 的同一 UB 地址, 不能使用双目标均分模式.
    params.dualDstCtl = 0;
    params.subBlockId = subBlockIdx != 0;
    params.params.ndNum = 1;
    params.params.srcNdStride = 0;
    params.params.dstNdStride = 0;
    Fixpipe<float, float, KDA_FIXPIPE_CFG_UB>(dstUBLocal, srcL0CLocal, params);
}

__aicore__ inline void PreparePairAndArawForSlot(
    const AscendC::LocalTensor<bfloat16_t>& kFactorL1Local, const AscendC::LocalTensor<bfloat16_t>& qFactorL1Local,
    const AscendC::LocalTensor<bfloat16_t>& kInvFactorL1Local, const AscendC::LocalTensor<bfloat16_t>& aL0ALocal,
    const AscendC::LocalTensor<bfloat16_t>& bL0BLocal, const AscendC::LocalTensor<float>& resultL0CLocal,
    const AscendC::LocalTensor<float>& pairUBLocal, const AscendC::LocalTensor<float>& aRawUBLocal,
    uint32_t subBlockIdx)
{
    using namespace AscendC;

    Mutex::Lock<PIPE_MTE1>(MUTEX_PREP_L0AB);
    CopyL1ToL0A<bfloat16_t>(aL0ALocal, kFactorL1Local, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, false);
    // KInvFactor 在 L1 中保存为 [C,D]. 非转置 LoadData 将其装入逻辑 [D,C] 的 L0B.
    CopyL1ToL0B<bfloat16_t>(bL0BLocal, kInvFactorL1Local, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, false);
    Mutex::Unlock<PIPE_MTE1>(MUTEX_PREP_L0AB);

    Mutex::Lock<PIPE_M>(MUTEX_PREP_L0AB);
    Mutex::Lock<PIPE_M>(MUTEX_PREP_L0C);
    CubeMmad<float, bfloat16_t, bfloat16_t>(resultL0CLocal, aL0ALocal, bL0BLocal, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM);
    Mutex::Unlock<PIPE_M>(MUTEX_PREP_L0AB);
    Mutex::Unlock<PIPE_M>(MUTEX_PREP_L0C);

    Mutex::Lock<PIPE_FIX>(MUTEX_PREP_L0C);
    FixPrepareResultToAiv(pairUBLocal, resultL0CLocal, subBlockIdx);
    Mutex::Unlock<PIPE_FIX>(MUTEX_PREP_L0C);

    // 第一次 Mmad 读完 L0A/L0B 后, 只需用 QFactor 覆写 L0A; KInvFactor 可继续留在 L0B.
    Mutex::Lock<PIPE_MTE1>(MUTEX_PREP_L0AB);
    CopyL1ToL0A<bfloat16_t>(aL0ALocal, qFactorL1Local, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, false);
    Mutex::Unlock<PIPE_MTE1>(MUTEX_PREP_L0AB);

    Mutex::Lock<PIPE_M>(MUTEX_PREP_L0AB);
    Mutex::Lock<PIPE_M>(MUTEX_PREP_L0C);
    CubeMmad<float, bfloat16_t, bfloat16_t>(resultL0CLocal, aL0ALocal, bL0BLocal, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM);
    Mutex::Unlock<PIPE_M>(MUTEX_PREP_L0AB);
    Mutex::Unlock<PIPE_M>(MUTEX_PREP_L0C);

    Mutex::Lock<PIPE_FIX>(MUTEX_PREP_L0C);
    FixPrepareResultToAiv(aRawUBLocal, resultL0CLocal, subBlockIdx);
    Mutex::Unlock<PIPE_FIX>(MUTEX_PREP_L0C);
}

__aicore__ inline void PrepareWForSlot(
    const AscendC::LocalTensor<bfloat16_t>& mL1Local, const AscendC::LocalTensor<bfloat16_t>& kPlusL1Local,
    const AscendC::LocalTensor<bfloat16_t>& aL0ALocal, const AscendC::LocalTensor<bfloat16_t>& bL0BLocal,
    const AscendC::LocalTensor<float>& resultL0CLocal, const AscendC::GlobalTensor<bfloat16_t>& wGlobal)
{
    using namespace AscendC;

    Mutex::Lock<PIPE_MTE1>(MUTEX_PREP_L0AB);
    CopyL1ToL0A<bfloat16_t>(aL0ALocal, mL1Local, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, false);
    CopyL1ToL0B<bfloat16_t>(bL0BLocal, kPlusL1Local, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, true);
    Mutex::Unlock<PIPE_MTE1>(MUTEX_PREP_L0AB);

    Mutex::Lock<PIPE_M>(MUTEX_PREP_L0AB);
    Mutex::Lock<PIPE_M>(MUTEX_PREP_L0C);
    CubeMmad<float, bfloat16_t, bfloat16_t>(resultL0CLocal, aL0ALocal, bL0BLocal, CHUNK_SIZE, HEAD_DIM, CHUNK_SIZE);
    Mutex::Unlock<PIPE_M>(MUTEX_PREP_L0AB);
    Mutex::Unlock<PIPE_M>(MUTEX_PREP_L0C);

    Mutex::Lock<PIPE_FIX>(MUTEX_PREP_L0C);
    FixpipeToGmRows<bfloat16_t, float>(wGlobal, resultL0CLocal, CHUNK_SIZE, HEAD_DIM, CHUNK_SIZE, HEAD_DIM);
    Mutex::Unlock<PIPE_FIX>(MUTEX_PREP_L0C);
}

__aicore__ inline void IssuePrepareCpairForAIC(
    const KimiDeltaAttnLiteTilingData& data, uint32_t aicIdx, uint32_t ordinal,
    const AscendC::LocalTensor<bfloat16_t>& aL0ALocal, const AscendC::LocalTensor<bfloat16_t>& bL0BLocal,
    const AscendC::LocalTensor<float>& resultL0CLocal)
{
    using namespace AscendC;
    const uint32_t cvSlot = ordinal % PREP_CV_SLOT_NUM;
    const uint32_t pairTaskId = aicIdx + ordinal * data.prepareUseAicNum;
    const uint16_t inputFlagId = SlotFlagId(FLAG_PREP_INPUT_HANDOFF_BASE, cvSlot);
    const uint16_t resultFlagId = SlotFlagId(FLAG_PREP_RESULT_HANDOFF_BASE, cvSlot);
    const uint16_t wFlagId = SlotFlagId(FLAG_PREP_W_HANDOFF_BASE, cvSlot);

    // 一次 mode2 Wait 等待两路 AIV 的 Set.
    WaitAivToAic<PIPE_MTE1>(inputFlagId);
    WaitAivToAic<PIPE_FIX>(resultFlagId);

    const uint32_t resultSlotAddr = cvSlot * PREP_SLOT_BYTES;
    LocalTensor<float> pairUBLocal(TPosition::VECCALC, resultSlotAddr + PREP_PAIR_FP32_UB_ADDR, CHUNK_C_ELEMS);
    LocalTensor<float> aRawUBLocal(TPosition::VECCALC, resultSlotAddr + PREP_A_RAW_FP32_UB_ADDR, CHUNK_C_ELEMS);
    for (uint32_t subBlockIdx = 0; subBlockIdx < PREP_SUB_AIV_NUM; ++subBlockIdx) {
        const uint32_t taskId = pairTaskId * PREP_SUB_AIV_NUM + subBlockIdx;
        if (taskId >= data.prepareNumTasks) {
            // prepare task 总数为奇数时, AIV1 仍参与组级握手, 但不计算尾部空 task.
            continue;
        }

        const uint32_t l1SlotAddr = cvSlot * PREP_L1_CV_SLOT_BYTES + subBlockIdx * PREP_L1_SLOT_BYTES;
        LocalTensor<bfloat16_t> kFactorL1Local(TPosition::A1, l1SlotAddr + PREP_K_FACTOR_L1_ADDR, CHUNK_D_ELEMS);
        LocalTensor<bfloat16_t> qFactorL1Local(TPosition::A1, l1SlotAddr + PREP_Q_FACTOR_L1_ADDR, CHUNK_D_ELEMS);
        LocalTensor<bfloat16_t> kInvFactorL1Local(TPosition::A1, l1SlotAddr + PREP_K_INV_FACTOR_L1_ADDR, CHUNK_D_ELEMS);
        PreparePairAndArawForSlot(
            kFactorL1Local, qFactorL1Local, kInvFactorL1Local, aL0ALocal, bL0BLocal, resultL0CLocal, pairUBLocal,
            aRawUBLocal, subBlockIdx);
    }

    // 两路 factor 均被 MTE1 读完后, AIV 才能在同一片 L1 写入 M/KPlus.
    SetAicToAiv<PIPE_MTE1>(wFlagId);
    SetAicToAiv<PIPE_FIX>(resultFlagId);
}

__aicore__ inline void IssuePrepareWForAIC(
    const AscendC::GlobalTensor<bfloat16_t>& wGlobal, const KimiDeltaAttnLiteTilingData& data, uint32_t aicIdx,
    uint32_t ordinal, const AscendC::LocalTensor<bfloat16_t>& aL0ALocal,
    const AscendC::LocalTensor<bfloat16_t>& bL0BLocal, const AscendC::LocalTensor<float>& resultL0CLocal)
{
    using namespace AscendC;
    const uint32_t cvSlot = ordinal % PREP_CV_SLOT_NUM;
    const uint32_t pairTaskId = aicIdx + ordinal * data.prepareUseAicNum;
    const uint16_t inputFlagId = SlotFlagId(FLAG_PREP_INPUT_HANDOFF_BASE, cvSlot);
    const uint16_t wFlagId = SlotFlagId(FLAG_PREP_W_HANDOFF_BASE, cvSlot);

    WaitAivToAic<PIPE_MTE1>(wFlagId);
    for (uint32_t subBlockIdx = 0; subBlockIdx < PREP_SUB_AIV_NUM; ++subBlockIdx) {
        const uint32_t taskId = pairTaskId * PREP_SUB_AIV_NUM + subBlockIdx;
        if (taskId >= data.prepareNumTasks) {
            continue;
        }

        const uint32_t l1SlotAddr = cvSlot * PREP_L1_CV_SLOT_BYTES + subBlockIdx * PREP_L1_SLOT_BYTES;
        LocalTensor<bfloat16_t> mL1Local(TPosition::A1, l1SlotAddr + PREP_W_M_L1_ADDR, CHUNK_C_ELEMS);
        LocalTensor<bfloat16_t> kPlusL1Local(TPosition::A1, l1SlotAddr + PREP_W_K_PLUS_L1_ADDR, CHUNK_D_ELEMS);
        PrepareWForSlot(
            mL1Local, kPlusL1Local, aL0ALocal, bL0BLocal, resultL0CLocal,
            wGlobal[static_cast<uint64_t>(taskId) * CHUNK_D_ELEMS]);
    }

    // 两路 W 输入均被 MTE1 读完后归还物理槽, 允许 VP(t+2) 覆写 factor.
    SetAicToAiv<PIPE_MTE1>(inputFlagId);
}

__aicore__ inline void KernelProcessPrepareForAIC(
    __gm__ uint8_t* workspaceGMAddr, const KimiDeltaAttnLiteTilingData& data)
{
    using namespace AscendC;

    if ASCEND_IS_AIC {
        GlobalTensor<bfloat16_t> wGlobal;
        // workspace 首段由 Prepare 写入 W, 不再保存 KPlus.
        wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.wOffset));

        LocalTensor<bfloat16_t> aL0ALocal(TPosition::A2, 0, CHUNK_D_ELEMS);
        LocalTensor<bfloat16_t> bL0BLocal(TPosition::B2, 0, CHUNK_D_ELEMS);
        LocalTensor<float> resultL0CLocal(TPosition::CO1, 0, CHUNK_D_ELEMS);

        const uint32_t aicIdx = GetBlockIdx();
        if (aicIdx >= data.preparePairNumTasks) {
            return;
        }
        const uint32_t pairTaskCount = CeilDiv<uint32_t>(data.preparePairNumTasks - aicIdx, data.prepareUseAicNum);

        // 初始时两个 L1 槽均归两路 AIV 写入.
        for (uint32_t cvSlot = 0; cvSlot < PREP_CV_SLOT_NUM; ++cvSlot) {
            SetAicToAiv<PIPE_MTE1>(SlotFlagId(FLAG_PREP_INPUT_HANDOFF_BASE, cvSlot));
        }

        const uint32_t preloadCount = pairTaskCount < PREP_CV_SLOT_NUM ? pairTaskCount : PREP_CV_SLOT_NUM;
        for (uint32_t ordinal = 0; ordinal < preloadCount; ++ordinal) {
            IssuePrepareCpairForAIC(data, aicIdx, ordinal, aL0ALocal, bL0BLocal, resultL0CLocal);
        }

        uint32_t ordinal = 0;
        // 稳态阶段在 Cw(t) 释放物理槽后, 在同槽发射 Cpair(t+2).
        for (; ordinal + PREP_CV_SLOT_NUM < pairTaskCount; ++ordinal) {
            IssuePrepareWForAIC(wGlobal, data, aicIdx, ordinal, aL0ALocal, bL0BLocal, resultL0CLocal);
            IssuePrepareCpairForAIC(data, aicIdx, ordinal + PREP_CV_SLOT_NUM, aL0ALocal, bL0BLocal, resultL0CLocal);
        }

        // 收尾阶段排空最后两个 Cw.
        for (; ordinal < pairTaskCount; ++ordinal) {
            IssuePrepareWForAIC(wGlobal, data, aicIdx, ordinal, aL0ALocal, bL0BLocal, resultL0CLocal);
        }

        // 循环前为每个槽发布空闲信号; 此处统一消费各槽最终归还的信号.
        for (uint32_t cvSlot = 0; cvSlot < PREP_CV_SLOT_NUM; ++cvSlot) {
            WaitAivToAic<PIPE_FIX>(SlotFlagId(FLAG_PREP_RESULT_HANDOFF_BASE, cvSlot));
        }
    }
}

} // namespace KDALite
