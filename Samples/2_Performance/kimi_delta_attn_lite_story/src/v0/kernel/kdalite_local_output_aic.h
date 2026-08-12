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

constexpr MutexId MUTEX_OUTPUT_A_L1 = 0;
constexpr MutexId MUTEX_OUTPUT_R_L1 = 1;
constexpr MutexId MUTEX_OUTPUT_L0AB = 2;
constexpr MutexId MUTEX_OUTPUT_L0C = 3;

__aicore__ inline void KernelProcessLocalOutputForAIC(
    __gm__ uint8_t* workspaceGMAddr, const KimiDeltaAttnLiteTilingData& data)
{
    using namespace AscendC;

    if ASCEND_IS_AIC {
        GlobalTensor<bfloat16_t> aGlobal;
        GlobalTensor<bfloat16_t> rGlobal;
        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.aOffset));
        rGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.rOffset));

        LocalTensor<bfloat16_t> aL1Local(TPosition::A1, OUTPUT_A_L1_ADDR, OUTPUT_A_L1_ELEMS);
        LocalTensor<bfloat16_t> rL1Local(TPosition::A1, OUTPUT_R_L1_ADDR, OUTPUT_R_L1_ELEMS);
        LocalTensor<bfloat16_t> aL0ALocal(TPosition::A2, 0, OUTPUT_L0A_ELEMS);
        LocalTensor<bfloat16_t> bL0BLocal(TPosition::B2, 0, OUTPUT_L0B_ELEMS);
        LocalTensor<float> resultL0CLocal(TPosition::CO1, 0, OUTPUT_L0C_ELEMS);
        LocalTensor<float> resultUBLocal(TPosition::VECCALC, OUTPUT_RESULT_UB_ADDR, OUTPUT_RESULT_UB_ELEMS);

        for (uint32_t taskId = GetBlockIdx(); taskId < data.outputNumTasks; taskId += data.outputUseAicNum) {
            const uint32_t dvTileId = taskId % DV_TILE_COUNT;
            const uint64_t chunkIndex = taskId / DV_TILE_COUNT;
            const uint64_t aOffset = chunkIndex * CHUNK_C_ELEMS;
            const uint64_t rOffset = chunkIndex * CHUNK_D_ELEMS + static_cast<uint64_t>(dvTileId) * DV_TILE;

            Mutex::Lock<PIPE_MTE2>(MUTEX_OUTPUT_A_L1);
            CopyGmToL1<bfloat16_t>(aL1Local, aGlobal[aOffset], CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);
            Mutex::Unlock<PIPE_MTE2>(MUTEX_OUTPUT_A_L1);

            Mutex::Lock<PIPE_MTE2>(MUTEX_OUTPUT_R_L1);
            CopyGmToL1<bfloat16_t>(rL1Local, rGlobal[rOffset], CHUNK_SIZE, DV_TILE, HEAD_DIM);
            Mutex::Unlock<PIPE_MTE2>(MUTEX_OUTPUT_R_L1);

            Mutex::Lock<PIPE_MTE1>(MUTEX_OUTPUT_A_L1);
            Mutex::Lock<PIPE_MTE1>(MUTEX_OUTPUT_R_L1);
            Mutex::Lock<PIPE_MTE1>(MUTEX_OUTPUT_L0AB);
            CopyL1ToL0A<bfloat16_t>(aL0ALocal, aL1Local, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, false);
            CopyL1ToL0B<bfloat16_t>(bL0BLocal, rL1Local, CHUNK_SIZE, CHUNK_SIZE, DV_TILE, true);
            Mutex::Unlock<PIPE_MTE1>(MUTEX_OUTPUT_A_L1);
            Mutex::Unlock<PIPE_MTE1>(MUTEX_OUTPUT_R_L1);
            Mutex::Unlock<PIPE_MTE1>(MUTEX_OUTPUT_L0AB);

            Mutex::Lock<PIPE_M>(MUTEX_OUTPUT_L0AB);
            Mutex::Lock<PIPE_M>(MUTEX_OUTPUT_L0C);
            CubeMmad<float, bfloat16_t, bfloat16_t>(
                resultL0CLocal, aL0ALocal, bL0BLocal, CHUNK_SIZE, DV_TILE, CHUNK_SIZE);
            Mutex::Unlock<PIPE_M>(MUTEX_OUTPUT_L0AB);
            Mutex::Unlock<PIPE_M>(MUTEX_OUTPUT_L0C);

            Mutex::Lock<PIPE_FIX>(MUTEX_OUTPUT_L0C);
            FixpipeToVecUB<float, float>(resultUBLocal, resultL0CLocal, CHUNK_SIZE, DV_TILE);
            SetAicToAiv<PIPE_FIX>(FLAG_OUTPUT_LOCAL_READY);
            Mutex::Unlock<PIPE_FIX>(MUTEX_OUTPUT_L0C);

            WaitAivToAic<PIPE_FIX>(FLAG_OUTPUT_DONE);
        }
    }
}

} // namespace KDALite
