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

constexpr MutexId MUTEX_STATE_LHS_L1 = 0;
constexpr MutexId MUTEX_STATE_L0AB = 1;
constexpr MutexId MUTEX_STATE_L0C = 2;

__aicore__ inline void CopyStateLhsFromWorkspace(
    AscendC::LocalTensor<bfloat16_t>& lhsL1Local, const AscendC::GlobalTensor<bfloat16_t>& srcGlobal)
{
    using namespace AscendC;

    Mutex::Lock<PIPE_MTE2>(MUTEX_STATE_LHS_L1);
    CopyGmToL1<bfloat16_t>(lhsL1Local, srcGlobal, CHUNK_SIZE, HEAD_DIM, HEAD_DIM);
    Mutex::Unlock<PIPE_MTE2>(MUTEX_STATE_LHS_L1);
}

__aicore__ inline void StateMmadToL0C(
    AscendC::LocalTensor<bfloat16_t>& lhsL1Local, AscendC::LocalTensor<bfloat16_t>& rhsL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL0ALocal, AscendC::LocalTensor<bfloat16_t>& bL0BLocal,
    AscendC::LocalTensor<float>& mmadL0CLocal, uint32_t lhsL1Rows, uint32_t rhsL1Rows, uint32_t m, uint32_t n,
    uint32_t k, bool transposeLhs)
{
    using namespace AscendC;

    Mutex::Lock<PIPE_MTE1>(MUTEX_STATE_LHS_L1);
    Mutex::Lock<PIPE_MTE1>(MUTEX_STATE_L0AB);
    CopyL1ToL0A<bfloat16_t>(aL0ALocal, lhsL1Local, lhsL1Rows, m, k, transposeLhs);
    CopyL1ToL0B<bfloat16_t>(bL0BLocal, rhsL1Local, rhsL1Rows, k, n, true);
    Mutex::Unlock<PIPE_MTE1>(MUTEX_STATE_LHS_L1);
    Mutex::Unlock<PIPE_MTE1>(MUTEX_STATE_L0AB);

    Mutex::Lock<PIPE_M>(MUTEX_STATE_L0AB);
    Mutex::Lock<PIPE_M>(MUTEX_STATE_L0C);
    CubeMmad<float, bfloat16_t, bfloat16_t>(mmadL0CLocal, aL0ALocal, bL0BLocal, m, n, k);
    Mutex::Unlock<PIPE_M>(MUTEX_STATE_L0AB);
    Mutex::Unlock<PIPE_M>(MUTEX_STATE_L0C);
}

__aicore__ inline void StateMmadToVecUB(
    AscendC::LocalTensor<bfloat16_t>& lhsL1Local, AscendC::LocalTensor<bfloat16_t>& rhsL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL0ALocal, AscendC::LocalTensor<bfloat16_t>& bL0BLocal,
    AscendC::LocalTensor<float>& mmadL0CLocal, AscendC::LocalTensor<float>& predDeltaUBLocal, uint32_t lhsL1Rows,
    uint32_t rhsL1Rows, uint32_t m, uint32_t n, uint32_t k, bool transposeLhs, uint16_t readyFlag)
{
    using namespace AscendC;

    StateMmadToL0C(
        lhsL1Local, rhsL1Local, aL0ALocal, bL0BLocal, mmadL0CLocal, lhsL1Rows, rhsL1Rows, m, n, k, transposeLhs);

    Mutex::Lock<PIPE_FIX>(MUTEX_STATE_L0C);
    FixpipeToVecUB<float, float>(predDeltaUBLocal, mmadL0CLocal, m, n);
    SetAicToAiv<PIPE_FIX>(readyFlag);
    Mutex::Unlock<PIPE_FIX>(MUTEX_STATE_L0C);
}

__aicore__ inline void StateMmadToGm(
    AscendC::LocalTensor<bfloat16_t>& lhsL1Local, AscendC::LocalTensor<bfloat16_t>& rhsL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL0ALocal, AscendC::LocalTensor<bfloat16_t>& bL0BLocal,
    AscendC::LocalTensor<float>& mmadL0CLocal, const AscendC::GlobalTensor<float>& outputGlobal, uint32_t lhsL1Rows,
    uint32_t rhsL1Rows, uint32_t m, uint32_t n, uint32_t k, bool transposeLhs, uint32_t gmRowStride)
{
    using namespace AscendC;

    StateMmadToL0C(
        lhsL1Local, rhsL1Local, aL0ALocal, bL0BLocal, mmadL0CLocal, lhsL1Rows, rhsL1Rows, m, n, k, transposeLhs);

    Mutex::Lock<PIPE_FIX>(MUTEX_STATE_L0C);
    FixpipeToGm<float, float>(outputGlobal, mmadL0CLocal, m, n, gmRowStride);
    Mutex::Unlock<PIPE_FIX>(MUTEX_STATE_L0C);
}

__aicore__ inline void KernelProcessStateUpdateForAIC(
    __gm__ uint8_t* workspaceGMAddr, const KimiDeltaAttnLiteTilingData& data)
{
    using namespace AscendC;

    if ASCEND_IS_AIC {
        GlobalTensor<bfloat16_t> wGlobal;
        GlobalTensor<bfloat16_t> qPlusGlobal;
        GlobalTensor<bfloat16_t> kTailGlobal;
        GlobalTensor<float> oHistoryGlobal;
        wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.wOffset));
        qPlusGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.qPlusOffset));
        kTailGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.kTailOffset));
        oHistoryGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGMAddr + data.oHistoryOffset));

        LocalTensor<bfloat16_t> lhsL1Local(TPosition::A1, STATE_LHS_L1_ADDR, STATE_LHS_L1_ELEMS);
        LocalTensor<bfloat16_t> stateL1Local(TPosition::A1, STATE_STATE_L1_ADDR, STATE_STATE_L1_ELEMS);
        LocalTensor<bfloat16_t> rL1Local(TPosition::A1, STATE_R_L1_ADDR, STATE_R_L1_ELEMS);
        LocalTensor<bfloat16_t> aL0ALocal(TPosition::A2, 0, STATE_L0A_ELEMS);
        LocalTensor<bfloat16_t> bL0BLocal(TPosition::B2, 0, STATE_L0B_ELEMS);
        // L0C 依次承载 prediction、history 和 state delta; predDeltaUB 先后交接其中前后两项.
        LocalTensor<float> mmadL0CLocal(TPosition::CO1, 0, STATE_L0C_ELEMS);
        LocalTensor<float> predDeltaUBLocal(TPosition::VECCALC, STATE_PRED_DELTA_UB_ADDR, STATE_PRED_DELTA_UB_ELEMS);

        for (uint32_t taskId = GetBlockIdx(); taskId < data.stateNumTasks; taskId += data.stateUseAicNum) {
            const uint32_t batchId = taskId / DV_TILE_COUNT;
            const uint32_t dvTileId = taskId % DV_TILE_COUNT;

            for (uint32_t chunkId = 0; chunkId < data.chunkCount; ++chunkId) {
                const uint64_t chunkIndex = static_cast<uint64_t>(batchId) * data.chunkCount + chunkId;
                const uint64_t chunkOffset = chunkIndex * CHUNK_D_ELEMS;
                const uint64_t historyOffset = chunkOffset + static_cast<uint64_t>(dvTileId) * DV_TILE;

                // MM4: W[C,128] @ state[128,32]. 两个 AIV 按 N 维分片将 state 写入共享 L1.
                CopyStateLhsFromWorkspace(lhsL1Local, wGlobal[chunkOffset]);
                WaitAivToAic<PIPE_MTE1>(FLAG_STATE_INPUT_READY);
                StateMmadToVecUB(
                    lhsL1Local, stateL1Local, aL0ALocal, bL0BLocal, mmadL0CLocal, predDeltaUBLocal, CHUNK_SIZE,
                    HEAD_DIM, CHUNK_SIZE, DV_TILE, HEAD_DIM, false, FLAG_STATE_PRED_READY);

                // MM5: Q_plus[C,128] @ state[128,32]. 本阶段结束前始终读取旧 state 的副本.
                CopyStateLhsFromWorkspace(lhsL1Local, qPlusGlobal[chunkOffset]);
                StateMmadToGm(
                    lhsL1Local, stateL1Local, aL0ALocal, bL0BLocal, mmadL0CLocal, oHistoryGlobal[historyOffset],
                    CHUNK_SIZE, HEAD_DIM, CHUNK_SIZE, DV_TILE, HEAD_DIM, false, VALUE_DIM);

                // MM7: K_tail.T[128,C] @ R[C,32]. K_tail 先按原 ND [C,128]
                // 搬成 L1 NZ, 再在 L1->L0A 时转置.
                CopyStateLhsFromWorkspace(lhsL1Local, kTailGlobal[chunkOffset]);
                WaitAivToAic<PIPE_MTE1>(FLAG_STATE_R_READY);
                // MM5 直接写 GM, 不占用 predDeltaUB. MM7 覆写该槽前需等待 AIV 读完 MM4 结果.
                WaitAivToAic<PIPE_FIX>(FLAG_STATE_PRED_CONSUMED);
                StateMmadToVecUB(
                    lhsL1Local, rL1Local, aL0ALocal, bL0BLocal, mmadL0CLocal, predDeltaUBLocal, CHUNK_SIZE, CHUNK_SIZE,
                    HEAD_DIM, DV_TILE, CHUNK_SIZE, true, FLAG_STATE_DELTA_READY);
            }
        }
    }
}

} // namespace KDALite
