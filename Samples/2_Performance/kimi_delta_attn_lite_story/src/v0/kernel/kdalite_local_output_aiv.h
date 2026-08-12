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

constexpr uint32_t OUTPUT_HISTORY_UB_ADDR = OUTPUT_RESULT_UB_ADDR + OUTPUT_RESULT_UB_ELEMS * sizeof(float);
constexpr uint32_t OUTPUT_BF16_UB_ADDR = OUTPUT_HISTORY_UB_ADDR + CHUNK_SIZE * AIV_DV_TILE * sizeof(float);

constexpr MutexId MUTEX_OUTPUT_HISTORY_UB = 0;

__aicore__ inline void KernelProcessLocalOutputForAIV(
    __gm__ bfloat16_t* outputGMAddr, __gm__ uint8_t* workspaceGMAddr, const KimiDeltaAttnLiteTilingData& data)
{
    using namespace AscendC;

    if ASCEND_IS_AIV {
        GlobalTensor<bfloat16_t> outputGlobal;
        GlobalTensor<float> oHistoryGlobal;
        outputGlobal.SetGlobalBuffer(outputGMAddr);
        oHistoryGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGMAddr + data.oHistoryOffset));

        LocalTensor<float> resultUBLocal(TPosition::VECCALC, OUTPUT_RESULT_UB_ADDR, OUTPUT_RESULT_UB_ELEMS);
        LocalTensor<float> historyUBLocal(TPosition::VECCALC, OUTPUT_HISTORY_UB_ADDR, CHUNK_SIZE * AIV_DV_TILE);
        LocalTensor<bfloat16_t> outputUBLocal(TPosition::VECCALC, OUTPUT_BF16_UB_ADDR, CHUNK_SIZE * AIV_DV_TILE);

        const uint32_t aivIdx = GetBlockIdx();
        const uint32_t subAivIdx = GetSubBlockIdx();
        const uint32_t aicIdx = aivIdx / GetSubBlockNum();

        for (uint32_t taskId = aicIdx; taskId < data.outputNumTasks; taskId += data.outputUseAicNum) {
            const uint32_t dvTileId = taskId % DV_TILE_COUNT;
            const uint64_t chunkIndex = taskId / DV_TILE_COUNT;
            const uint32_t batchId = chunkIndex / data.chunkCount;
            const uint32_t chunkId = chunkIndex % data.chunkCount;
            const uint32_t firstToken = chunkId * CHUNK_SIZE;
            const uint32_t validLen = data.seqLen - firstToken < CHUNK_SIZE ? data.seqLen - firstToken : CHUNK_SIZE;
            const uint32_t valueColumn = dvTileId * DV_TILE + subAivIdx * AIV_DV_TILE;
            const uint64_t historyOffset = chunkIndex * CHUNK_D_ELEMS + valueColumn;
            const uint64_t outputOffset =
                (static_cast<uint64_t>(batchId) * data.seqLen + firstToken) * VALUE_DIM + valueColumn;

            Mutex::Lock<PIPE_MTE2>(MUTEX_OUTPUT_HISTORY_UB);
            CopyGmToUbRows(historyUBLocal, oHistoryGlobal[historyOffset], CHUNK_SIZE, AIV_DV_TILE, VALUE_DIM);
            Mutex::Unlock<PIPE_MTE2>(MUTEX_OUTPUT_HISTORY_UB);

            WaitAicToAiv<PIPE_V>(FLAG_OUTPUT_LOCAL_READY);
            Mutex::Lock<PIPE_V>(MUTEX_OUTPUT_HISTORY_UB);
            Add(historyUBLocal, historyUBLocal, resultUBLocal, CHUNK_SIZE * AIV_DV_TILE);
            Cast(outputUBLocal, historyUBLocal, RoundMode::CAST_RINT, CHUNK_SIZE * AIV_DV_TILE);
            Mutex::Unlock<PIPE_V>(MUTEX_OUTPUT_HISTORY_UB);

            Mutex::Lock<PIPE_MTE3>(MUTEX_OUTPUT_HISTORY_UB);
            CopyUbToGmRows(outputGlobal[outputOffset], outputUBLocal, validLen, AIV_DV_TILE, AIV_DV_TILE, VALUE_DIM);
            Mutex::Unlock<PIPE_MTE3>(MUTEX_OUTPUT_HISTORY_UB);
            SetAivToAic<PIPE_MTE3>(FLAG_OUTPUT_DONE);
        }
    }
}

} // namespace KDALite
