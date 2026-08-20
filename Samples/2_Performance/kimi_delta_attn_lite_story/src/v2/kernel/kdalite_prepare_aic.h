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

__aicore__ inline void FixPreparePairArawToAiv(
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
    const AscendC::LocalTensor<bfloat16_t>& bL0BLocal, const AscendC::LocalTensor<float>& mmadL0CLocal,
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
    CubeMmad<float, bfloat16_t, bfloat16_t>(mmadL0CLocal, aL0ALocal, bL0BLocal, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM);
    Mutex::Unlock<PIPE_M>(MUTEX_PREP_L0AB);
    Mutex::Unlock<PIPE_M>(MUTEX_PREP_L0C);

    Mutex::Lock<PIPE_FIX>(MUTEX_PREP_L0C);
    FixPreparePairArawToAiv(pairUBLocal, mmadL0CLocal, subBlockIdx);
    Mutex::Unlock<PIPE_FIX>(MUTEX_PREP_L0C);

    // 第一次 Mmad 读完 L0A/L0B 后, 只需用 QFactor 覆写 L0A; KInvFactor 可继续留在 L0B.
    Mutex::Lock<PIPE_MTE1>(MUTEX_PREP_L0AB);
    CopyL1ToL0A<bfloat16_t>(aL0ALocal, qFactorL1Local, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, false);
    Mutex::Unlock<PIPE_MTE1>(MUTEX_PREP_L0AB);

    Mutex::Lock<PIPE_M>(MUTEX_PREP_L0AB);
    Mutex::Lock<PIPE_M>(MUTEX_PREP_L0C);
    CubeMmad<float, bfloat16_t, bfloat16_t>(mmadL0CLocal, aL0ALocal, bL0BLocal, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM);
    Mutex::Unlock<PIPE_M>(MUTEX_PREP_L0AB);
    Mutex::Unlock<PIPE_M>(MUTEX_PREP_L0C);

    Mutex::Lock<PIPE_FIX>(MUTEX_PREP_L0C);
    FixPreparePairArawToAiv(aRawUBLocal, mmadL0CLocal, subBlockIdx);
    Mutex::Unlock<PIPE_FIX>(MUTEX_PREP_L0C);
}

__aicore__ inline void KernelProcessPrepareForAIC(const KimiDeltaAttnLiteTilingData& data)
{
    using namespace AscendC;

    if ASCEND_IS_AIC {
        LocalTensor<bfloat16_t> aL0ALocal(TPosition::A2, 0, CHUNK_D_ELEMS);
        LocalTensor<bfloat16_t> bL0BLocal(TPosition::B2, 0, CHUNK_D_ELEMS);
        // 单个 L0C 槽按顺序承载 Pair 和 Araw.
        LocalTensor<float> mmadL0CLocal(TPosition::CO1, 0, CHUNK_C_ELEMS);

        const uint32_t aicIdx = GetBlockIdx();
        if (aicIdx >= data.preparePairNumTasks) {
            return;
        }
        const uint32_t pairTaskCount = CeilDiv<uint32_t>(data.preparePairNumTasks - aicIdx, data.prepareUseAicNum);

        // 初始时两个 L1 槽均归两路 AIV 写入.
        for (uint32_t cvSlot = 0; cvSlot < PREP_CV_SLOT_NUM; ++cvSlot) {
            SetAicToAiv<PIPE_MTE1>(SlotFlagId(FLAG_PREP_L1_HANDOFF_BASE, cvSlot));
        }

        for (uint32_t ordinal = 0; ordinal < pairTaskCount; ++ordinal) {
            const uint32_t cvSlot = ordinal % PREP_CV_SLOT_NUM;
            const uint32_t pairTaskId = aicIdx + ordinal * data.prepareUseAicNum;
            const uint16_t l1HandoffFlagId = SlotFlagId(FLAG_PREP_L1_HANDOFF_BASE, cvSlot);
            const uint16_t pairArawFlagId = SlotFlagId(FLAG_PREP_PAIR_ARAW_HANDOFF_BASE, cvSlot);
            // 一次 mode2 Wait 等待两路 AIV 的 Set.
            WaitAivToAic<PIPE_MTE1>(l1HandoffFlagId);
            WaitAivToAic<PIPE_FIX>(pairArawFlagId);

            // subBlockId 决定 Fixpipe 的目的 AIV; 两路 AIV 使用相同的本地 UB 地址.
            const uint32_t pairArawSlotAddr = cvSlot * PREP_SLOT_BYTES;
            LocalTensor<float> pairUBLocal(
                TPosition::VECCALC, pairArawSlotAddr + PREP_PAIR_FP32_UB_ADDR, CHUNK_C_ELEMS);
            LocalTensor<float> aRawUBLocal(
                TPosition::VECCALC, pairArawSlotAddr + PREP_A_RAW_FP32_UB_ADDR, CHUNK_C_ELEMS);
            for (uint32_t subBlockIdx = 0; subBlockIdx < PREP_SUB_AIV_NUM; ++subBlockIdx) {
                const uint32_t taskId = pairTaskId * PREP_SUB_AIV_NUM + subBlockIdx;
                if (taskId >= data.prepareNumTasks) {
                    // prepare task 总数为奇数时, AIV1 仍参与组级握手, 但不计算尾部空 task.
                    continue;
                }

                const uint32_t l1SlotAddr = cvSlot * PREP_L1_CV_SLOT_BYTES + subBlockIdx * PREP_L1_SLOT_BYTES;
                LocalTensor<bfloat16_t> kFactorL1Local(
                    TPosition::A1, l1SlotAddr + PREP_K_FACTOR_L1_ADDR, CHUNK_D_ELEMS);
                LocalTensor<bfloat16_t> qFactorL1Local(
                    TPosition::A1, l1SlotAddr + PREP_Q_FACTOR_L1_ADDR, CHUNK_D_ELEMS);
                LocalTensor<bfloat16_t> kInvFactorL1Local(
                    TPosition::A1, l1SlotAddr + PREP_K_INV_FACTOR_L1_ADDR, CHUNK_D_ELEMS);
                PreparePairAndArawForSlot(
                    kFactorL1Local, qFactorL1Local, kInvFactorL1Local, aL0ALocal, bL0BLocal, mmadL0CLocal, pairUBLocal,
                    aRawUBLocal, subBlockIdx);
            }

            // MTE1 读完所有有效 factor 后归还 L1 槽; 全部 Fix 完成后发布 Pair/Araw ready.
            SetAicToAiv<PIPE_MTE1>(l1HandoffFlagId);
            SetAicToAiv<PIPE_FIX>(pairArawFlagId);
        }

        // 循环前为每个槽发布空闲信号; 此处统一消费各槽最终归还的信号.
        for (uint32_t cvSlot = 0; cvSlot < PREP_CV_SLOT_NUM; ++cvSlot) {
            WaitAivToAic<PIPE_FIX>(SlotFlagId(FLAG_PREP_PAIR_ARAW_HANDOFF_BASE, cvSlot));
        }
    }
}

} // namespace KDALite
