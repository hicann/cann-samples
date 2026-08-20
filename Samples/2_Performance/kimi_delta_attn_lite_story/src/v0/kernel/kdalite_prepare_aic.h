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

// L0A/L0B 作为一组输入共同在 MTE1 和 M 之间交接; L0C 在 M 和 FIX 之间交接.
// V 的 L1 区域由本核 MTE2 生产, MTE1 消费, 单独用一个 Mutex 管理所有权.
constexpr MutexId MUTEX_PREP_L0AB = 0;
constexpr MutexId MUTEX_PREP_L0C = 1;
constexpr MutexId MUTEX_PREP_V_L1 = 2;

template <bool RELEASE_SHARED_L1, bool PROTECT_V_L1, typename DstT>
__aicore__ inline void PrepareMmad(
    AscendC::LocalTensor<bfloat16_t>& lhsL1Local, AscendC::LocalTensor<bfloat16_t>& rhsL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL0ALocal, AscendC::LocalTensor<bfloat16_t>& bL0BLocal,
    AscendC::LocalTensor<float>& mmadL0CLocal, const AscendC::GlobalTensor<DstT>& outputGlobal)
{
    using namespace AscendC;

    if constexpr (PROTECT_V_L1) {
        Mutex::Lock<PIPE_MTE1>(MUTEX_PREP_V_L1);
    }
    Mutex::Lock<PIPE_MTE1>(MUTEX_PREP_L0AB);
    CopyL1ToL0A<bfloat16_t>(aL0ALocal, lhsL1Local, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, false);
    CopyL1ToL0B<bfloat16_t>(bL0BLocal, rhsL1Local, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, true);
    if constexpr (RELEASE_SHARED_L1) {
        // L1_FREE 排在第二次 L1->L0 的 MTE1 指令之后, AIV 收到后才可覆写共享 L1.
        SetAicToAiv<PIPE_MTE1>(FLAG_PREP_L1_FREE);
    }
    Mutex::Unlock<PIPE_MTE1>(MUTEX_PREP_L0AB);
    if constexpr (PROTECT_V_L1) {
        Mutex::Unlock<PIPE_MTE1>(MUTEX_PREP_V_L1);
    }

    Mutex::Lock<PIPE_M>(MUTEX_PREP_L0AB);
    Mutex::Lock<PIPE_M>(MUTEX_PREP_L0C);
    CubeMmad<float, bfloat16_t, bfloat16_t>(mmadL0CLocal, aL0ALocal, bL0BLocal, CHUNK_SIZE, HEAD_DIM, CHUNK_SIZE);
    Mutex::Unlock<PIPE_M>(MUTEX_PREP_L0AB);
    Mutex::Unlock<PIPE_M>(MUTEX_PREP_L0C);

    Mutex::Lock<PIPE_FIX>(MUTEX_PREP_L0C);
    FixpipeToGm<DstT, float>(outputGlobal, mmadL0CLocal, CHUNK_SIZE, HEAD_DIM);
    Mutex::Unlock<PIPE_FIX>(MUTEX_PREP_L0C);
}

__aicore__ inline void KernelProcessPrepareForAIC(
    __gm__ bfloat16_t* vGMAddr, __gm__ uint8_t* workspaceGMAddr, const KimiDeltaAttnLiteTilingData& data)
{
    using namespace AscendC;

    if ASCEND_IS_AIC {
        GlobalTensor<bfloat16_t> vGlobal, wGlobal;
        GlobalTensor<float> uGlobal;
        vGlobal.SetGlobalBuffer(vGMAddr);
        wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.wOffset));
        uGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGMAddr + data.uOffset));

        LocalTensor<bfloat16_t> mL1Local(TPosition::A1, PREP_M_L1_ADDR, PREP_M_L1_ELEMS);
        LocalTensor<bfloat16_t> kPlusL1Local(TPosition::A1, PREP_K_PLUS_L1_ADDR, PREP_K_PLUS_L1_ELEMS);
        LocalTensor<bfloat16_t> vL1Local(TPosition::A1, PREP_V_L1_ADDR, PREP_V_L1_ELEMS);
        LocalTensor<bfloat16_t> aL0ALocal(TPosition::A2, 0, PREP_L0A_ELEMS);
        LocalTensor<bfloat16_t> bL0BLocal(TPosition::B2, 0, PREP_L0B_ELEMS);
        // 单个 L0C 槽按顺序承载 W=M@KPlus 和 U=M@V.
        LocalTensor<float> mmadL0CLocal(TPosition::CO1, 0, PREP_L0C_ELEMS);

        for (uint32_t taskId = GetBlockIdx(); taskId < data.prepareNumTasks; taskId += data.prepareUseAicNum) {
            const uint32_t batchId = taskId / data.chunkCount;
            const uint32_t chunkId = taskId % data.chunkCount;
            const uint32_t firstToken = chunkId * CHUNK_SIZE;
            const uint32_t validLen = data.seqLen - firstToken < CHUNK_SIZE ? data.seqLen - firstToken : CHUNK_SIZE;
            const uint64_t tokenOffset = (static_cast<uint64_t>(batchId) * data.seqLen + firstToken) * HEAD_DIM;

            Mutex::Lock<PIPE_MTE2>(MUTEX_PREP_V_L1);
            if (validLen < CHUNK_SIZE) {
                Fill(vL1Local, {1, PREP_V_L1_ELEMS * sizeof(bfloat16_t) / C0_BYTES, 0, static_cast<bfloat16_t>(0)});
            }
            CopyGmToL1(vL1Local, vGlobal[tokenOffset], validLen, HEAD_DIM, HEAD_DIM, CHUNK_SIZE);
            Mutex::Unlock<PIPE_MTE2>(MUTEX_PREP_V_L1);

            // mode2 下两路 AIV 都完成 M/K_plus 的 NZ 写入后, MTE1 才能读取共享 L1.
            WaitAivToAic<PIPE_MTE1>(FLAG_PREP_INPUT_READY);

            const uint64_t chunkOffset = static_cast<uint64_t>(taskId) * CHUNK_D_ELEMS;
            PrepareMmad<false, false>(mL1Local, kPlusL1Local, aL0ALocal, bL0BLocal, mmadL0CLocal, wGlobal[chunkOffset]);
            PrepareMmad<true, true>(mL1Local, vL1Local, aL0ALocal, bL0BLocal, mmadL0CLocal, uGlobal[chunkOffset]);
        }
    }
}

} // namespace KDALite
