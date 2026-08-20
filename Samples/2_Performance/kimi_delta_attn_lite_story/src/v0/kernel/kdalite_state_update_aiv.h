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

constexpr uint32_t STATE_VALUE_UB_ADDR = STATE_PRED_DELTA_UB_ADDR + STATE_PRED_DELTA_UB_ELEMS * sizeof(float);
constexpr uint32_t STATE_SHADOW_UB_ADDR = STATE_VALUE_UB_ADDR + HEAD_DIM * AIV_DV_TILE * sizeof(float);
constexpr uint32_t STATE_R_WORK_FP32_UB_ADDR = STATE_SHADOW_UB_ADDR + HEAD_DIM * AIV_DV_TILE * sizeof(bfloat16_t);
constexpr uint32_t STATE_R_BF16_UB_ADDR = STATE_R_WORK_FP32_UB_ADDR + CHUNK_SIZE * AIV_DV_TILE * sizeof(float);
constexpr uint32_t STATE_G_LAST_UB_ADDR = STATE_R_BF16_UB_ADDR + CHUNK_SIZE * AIV_DV_TILE * sizeof(bfloat16_t);

constexpr MutexId MUTEX_STATE_VALUE_UB = 0;
constexpr MutexId MUTEX_STATE_SHADOW_UB = 1;
constexpr MutexId MUTEX_STATE_R_WORK_UB = 2;
constexpr MutexId MUTEX_STATE_G_LAST_UB = 3;

static constexpr AscendC::Reg::CastTrait STATE_B32_TO_B16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

__simd_vf__ inline void UpdateStateAndShadowVF(
    __ubuf__ float* state, __ubuf__ bfloat16_t* stateShadow, __ubuf__ float* stateDelta, __ubuf__ float* gLast)
{
    using namespace AscendC;
    Reg::RegTensor<bfloat16_t> shadowB16Reg;
    Reg::RegTensor<float> stateReg, deltaReg, decayReg;
    Reg::MaskReg first16 = Reg::CreateMask<float, Reg::MaskPattern::VL16>();
    for (uint16_t row = 0; row < HEAD_DIM; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * AIV_DV_TILE;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(decayReg, gLast + row);
        Reg::Exp(decayReg, decayReg, first16);
        Reg::LoadAlign(stateReg, state + offset);
        Reg::LoadAlign(deltaReg, stateDelta + offset);
        Reg::MulDstAdd(stateReg, decayReg, deltaReg, first16);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(state + offset, stateReg, first16);
        Reg::Cast<bfloat16_t, float, STATE_B32_TO_B16>(shadowB16Reg, stateReg, first16);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(stateShadow + offset, shadowB16Reg, first16);
    }
}

__aicore__ inline void KernelProcessStateUpdateForAIV(
    __gm__ bfloat16_t* finalStateGMAddr, __gm__ uint8_t* workspaceGMAddr, const KimiDeltaAttnLiteTilingData& data)
{
    using namespace AscendC;

    if ASCEND_IS_AIV {
        GlobalTensor<float> uGlobal, gLastGlobal;
        GlobalTensor<bfloat16_t> rGlobal, finalStateGlobal;
        uGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGMAddr + data.uOffset));
        rGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.rOffset));
        gLastGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGMAddr + data.gLastOffset));
        finalStateGlobal.SetGlobalBuffer(finalStateGMAddr);

        LocalTensor<bfloat16_t> stateL1Local(TPosition::A1, STATE_STATE_L1_ADDR, STATE_STATE_L1_ELEMS);
        LocalTensor<bfloat16_t> rL1Local(TPosition::A1, STATE_R_L1_ADDR, STATE_R_L1_ELEMS);
        LocalTensor<float> predDeltaUBLocal(TPosition::VECCALC, STATE_PRED_DELTA_UB_ADDR, STATE_PRED_DELTA_UB_ELEMS);
        LocalTensor<float> stateUBLocal(TPosition::VECCALC, STATE_VALUE_UB_ADDR, HEAD_DIM * AIV_DV_TILE);
        LocalTensor<bfloat16_t> stateShadowUBLocal(TPosition::VECCALC, STATE_SHADOW_UB_ADDR, HEAD_DIM * AIV_DV_TILE);
        LocalTensor<float> rWorkFP32UBLocal(TPosition::VECCALC, STATE_R_WORK_FP32_UB_ADDR, CHUNK_SIZE * AIV_DV_TILE);
        LocalTensor<bfloat16_t> rUBLocal(TPosition::VECCALC, STATE_R_BF16_UB_ADDR, CHUNK_SIZE * AIV_DV_TILE);
        LocalTensor<float> gLastUBLocal(TPosition::VECCALC, STATE_G_LAST_UB_ADDR, HEAD_DIM);

        const uint32_t aivIdx = GetBlockIdx();
        const uint32_t subAivIdx = GetSubBlockIdx();
        const uint32_t aicIdx = aivIdx / GetSubBlockNum();

        for (uint32_t taskId = aicIdx; taskId < data.stateNumTasks; taskId += data.stateUseAicNum) {
            const uint32_t batchId = taskId / DV_TILE_COUNT;
            const uint32_t dvTileId = taskId % DV_TILE_COUNT;
            const uint32_t valueColumn = dvTileId * DV_TILE + subAivIdx * AIV_DV_TILE;

            // state 在整个 task 内由 V 持有; 最后一次更新完成后才交给 MTE3 写回.
            Mutex::Lock<PIPE_V>(MUTEX_STATE_VALUE_UB);
            Duplicate<float>(stateUBLocal, 0.0F, HEAD_DIM * AIV_DV_TILE);
            Mutex::Lock<PIPE_V>(MUTEX_STATE_SHADOW_UB);
            Duplicate<uint16_t>(stateShadowUBLocal.ReinterpretCast<uint16_t>(), 0, HEAD_DIM * AIV_DV_TILE);
            Mutex::Unlock<PIPE_V>(MUTEX_STATE_SHADOW_UB);

            Mutex::Lock<PIPE_MTE3>(MUTEX_STATE_SHADOW_UB);
            DataCopy(stateL1Local[subAivIdx * HEAD_DIM * AIV_DV_TILE], stateShadowUBLocal, HEAD_DIM * AIV_DV_TILE);
            Mutex::Unlock<PIPE_MTE3>(MUTEX_STATE_SHADOW_UB);
            SetAivToAic<PIPE_MTE3>(FLAG_STATE_INPUT_READY);

            for (uint32_t chunkId = 0; chunkId < data.chunkCount; ++chunkId) {
                const uint64_t chunkIndex = static_cast<uint64_t>(batchId) * data.chunkCount + chunkId;
                const uint64_t chunkOffset = chunkIndex * CHUNK_D_ELEMS + valueColumn;

                Mutex::Lock<PIPE_MTE2>(MUTEX_STATE_R_WORK_UB);
                CopyGmToUbRows(rWorkFP32UBLocal, uGlobal[chunkOffset], CHUNK_SIZE, AIV_DV_TILE, HEAD_DIM);
                Mutex::Unlock<PIPE_MTE2>(MUTEX_STATE_R_WORK_UB);
                Mutex::Lock<PIPE_MTE2>(MUTEX_STATE_G_LAST_UB);
                CopyGmToUbRows(gLastUBLocal, gLastGlobal[chunkIndex * HEAD_DIM], 1, HEAD_DIM, HEAD_DIM);
                Mutex::Unlock<PIPE_MTE2>(MUTEX_STATE_G_LAST_UB);

                WaitAicToAiv<PIPE_V>(FLAG_STATE_PRED_READY);
                Mutex::Lock<PIPE_V>(MUTEX_STATE_R_WORK_UB);
                Sub(rWorkFP32UBLocal, rWorkFP32UBLocal, predDeltaUBLocal, CHUNK_SIZE * AIV_DV_TILE);
                Cast(rUBLocal, rWorkFP32UBLocal, RoundMode::CAST_RINT, CHUNK_SIZE * AIV_DV_TILE);
                Mutex::Unlock<PIPE_V>(MUTEX_STATE_R_WORK_UB);
                SetAivToAic<PIPE_V>(FLAG_STATE_PRED_CONSUMED);

                Mutex::Lock<PIPE_MTE3>(MUTEX_STATE_R_WORK_UB);
                CopyUbToGmRows(rGlobal[chunkOffset], rUBLocal, CHUNK_SIZE, AIV_DV_TILE, AIV_DV_TILE, HEAD_DIM);
                DataCopy(rL1Local[subAivIdx * CHUNK_SIZE * AIV_DV_TILE], rUBLocal, CHUNK_SIZE * AIV_DV_TILE);
                Mutex::Unlock<PIPE_MTE3>(MUTEX_STATE_R_WORK_UB);
                SetAivToAic<PIPE_MTE3>(FLAG_STATE_R_READY);

                WaitAicToAiv<PIPE_V>(FLAG_STATE_DELTA_READY);
                Mutex::Lock<PIPE_V>(MUTEX_STATE_G_LAST_UB);
                Mutex::Lock<PIPE_V>(MUTEX_STATE_SHADOW_UB);
                asc_vf_call<UpdateStateAndShadowVF>(
                    reinterpret_cast<__ubuf__ float*>(stateUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ bfloat16_t*>(stateShadowUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(predDeltaUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(gLastUBLocal.GetPhyAddr()));
                Mutex::Unlock<PIPE_V>(MUTEX_STATE_SHADOW_UB);
                Mutex::Unlock<PIPE_V>(MUTEX_STATE_G_LAST_UB);

                if (chunkId + 1 < data.chunkCount) {
                    Mutex::Lock<PIPE_MTE3>(MUTEX_STATE_SHADOW_UB);
                    DataCopy(
                        stateL1Local[subAivIdx * HEAD_DIM * AIV_DV_TILE], stateShadowUBLocal, HEAD_DIM * AIV_DV_TILE);
                    Mutex::Unlock<PIPE_MTE3>(MUTEX_STATE_SHADOW_UB);
                    SetAivToAic<PIPE_MTE3>(FLAG_STATE_INPUT_READY);
                }
            }

            Mutex::Unlock<PIPE_V>(MUTEX_STATE_VALUE_UB);
            Mutex::Lock<PIPE_MTE3>(MUTEX_STATE_SHADOW_UB);
            const uint64_t finalStateOffset = static_cast<uint64_t>(batchId) * HEAD_DIM * VALUE_DIM + valueColumn;
            CopyUbToGmRows(
                finalStateGlobal[finalStateOffset], stateShadowUBLocal, HEAD_DIM, AIV_DV_TILE, AIV_DV_TILE, VALUE_DIM);
            Mutex::Unlock<PIPE_MTE3>(MUTEX_STATE_SHADOW_UB);
        }
    }
}

} // namespace KDALite
