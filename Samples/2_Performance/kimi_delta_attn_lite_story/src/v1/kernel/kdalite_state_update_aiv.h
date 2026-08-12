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

constexpr uint32_t STATE_VALUE_UB_ADDR = HANDOFF_UB_END_ADDR;
constexpr uint32_t STATE_SHADOW_UB_ADDR = STATE_VALUE_UB_ADDR + STATE_HALF_ELEMS * sizeof(float);
constexpr uint32_t R_BF16_UB_ADDR = STATE_SHADOW_UB_ADDR + STATE_HALF_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t STATE_DECAY_UB_ADDR = R_BF16_UB_ADDR + CHUNK_DV_HALF_ELEMS * sizeof(bfloat16_t) * DB_SLOT_NUM;
constexpr uint32_t OUTPUT_BF16_UB_ADDR = STATE_DECAY_UB_ADDR + HEAD_DIM * sizeof(float) * DB_SLOT_NUM;
constexpr uint32_t STATE_AIV_UB_END_ADDR = OUTPUT_BF16_UB_ADDR + CHUNK_DV_HALF_ELEMS * sizeof(bfloat16_t) * DB_SLOT_NUM;

static_assert(STATE_AIV_UB_END_ADDR <= AIV_USABLE_UB_BYTES, "FusedRecurrentOutput UB allocation exceeds 248 KiB");

constexpr MutexId MUTEX_STATE_VALUE_UB = 0;
constexpr MutexId MUTEX_STATE_SHADOW_UB = 1;
constexpr MutexId MUTEX_R_UB_BASE = 2;
constexpr MutexId MUTEX_DECAY_UB_BASE = MUTEX_R_UB_BASE + DB_SLOT_NUM;
constexpr MutexId MUTEX_OUTPUT_UB_BASE = MUTEX_DECAY_UB_BASE + DB_SLOT_NUM;

static constexpr AscendC::Reg::CastTrait STATE_B32_TO_B16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

__simd_vf__ inline void ComputeRAndCastVF(__ubuf__ bfloat16_t* r, __ubuf__ float* u, __ubuf__ float* prediction)
{
    using namespace AscendC;
    Reg::RegTensor<bfloat16_t> rB16Reg;
    Reg::RegTensor<float> uReg, predReg, rReg;
    Reg::MaskReg tileMask = Reg::CreateMask<float, Reg::MaskPattern::VL16>();
    for (uint16_t row = 0; row < CHUNK_SIZE; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * AIV_DV_TILE;
        Reg::LoadAlign(uReg, u + offset);
        Reg::LoadAlign(predReg, prediction + offset);
        Reg::Sub(rReg, uReg, predReg, tileMask);
        Reg::Cast<bfloat16_t, float, STATE_B32_TO_B16>(rB16Reg, rReg, tileMask);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(r + offset, rB16Reg, tileMask);
    }
}

__simd_vf__ inline void UpdateStateAndShadowVF(
    __ubuf__ float* state, __ubuf__ bfloat16_t* stateShadow, __ubuf__ float* stateDelta, __ubuf__ float* stateDecay)
{
    using namespace AscendC;
    Reg::RegTensor<bfloat16_t> shadowB16Reg;
    Reg::RegTensor<float> stateReg, deltaReg, decayReg;
    Reg::MaskReg tileMask = Reg::CreateMask<float, Reg::MaskPattern::VL16>();
    for (uint16_t row = 0; row < HEAD_DIM; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * AIV_DV_TILE;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(decayReg, stateDecay + row);
        Reg::LoadAlign(stateReg, state + offset);
        Reg::LoadAlign(deltaReg, stateDelta + offset);
        Reg::MulDstAdd(stateReg, decayReg, deltaReg, tileMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(state + offset, stateReg, tileMask);
        Reg::Cast<bfloat16_t, float, STATE_B32_TO_B16>(shadowB16Reg, stateReg, tileMask);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(stateShadow + offset, shadowB16Reg, tileMask);
    }
}

__simd_vf__ inline void OutputAddCastVF(__ubuf__ bfloat16_t* output, __ubuf__ float* history, __ubuf__ float* local)
{
    using namespace AscendC;
    Reg::RegTensor<bfloat16_t> outputB16Reg;
    Reg::RegTensor<float> historyReg, localReg;
    Reg::MaskReg tileMask = Reg::CreateMask<float, Reg::MaskPattern::VL16>();
    for (uint16_t row = 0; row < CHUNK_SIZE; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * AIV_DV_TILE;
        Reg::LoadAlign(historyReg, history + offset);
        Reg::LoadAlign(localReg, local + offset);
        Reg::Add(historyReg, historyReg, localReg, tileMask);
        Reg::Cast<bfloat16_t, float, STATE_B32_TO_B16>(outputB16Reg, historyReg, tileMask);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(output + offset, outputB16Reg, tileMask);
    }
}

__aicore__ inline void PreloadDecay(
    AscendC::LocalTensor<float>& stateDecayUBLocal, const AscendC::GlobalTensor<float>& stateDecayGlobal,
    uint64_t stateDecayOffset, MutexId decayMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_MTE2>(decayMutexId);
    DataCopy(stateDecayUBLocal, stateDecayGlobal[stateDecayOffset], HEAD_DIM);
    Mutex::Unlock<PIPE_MTE2>(decayMutexId);
}

__aicore__ inline void PublishState(
    AscendC::LocalTensor<bfloat16_t>& stateL1Local, AscendC::LocalTensor<bfloat16_t>& stateShadowUBLocal, uint32_t slot,
    uint32_t subAivIdx)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_MTE3>(MUTEX_STATE_SHADOW_UB);
    WaitAicToAiv<PIPE_MTE3>(SlotFlagId(FLAG_STATE_HANDOFF_BASE, slot));
    DataCopy(
        stateL1Local[slot * STATE_TILE_ELEMS + subAivIdx * STATE_HALF_ELEMS], stateShadowUBLocal, STATE_HALF_ELEMS);
    SetAivToAic<PIPE_MTE3>(SlotFlagId(FLAG_STATE_HANDOFF_BASE, slot));
    Mutex::Unlock<PIPE_MTE3>(MUTEX_STATE_SHADOW_UB);
}

__aicore__ inline void KernelProcessStateUpdateForAIV(
    __gm__ bfloat16_t* outputGMAddr, __gm__ float* finalStateGMAddr, __gm__ uint8_t* workspaceGMAddr,
    const KimiDeltaAttnLiteTilingData& data)
{
    using namespace AscendC;
    if ASCEND_IS_AIV {
        GlobalTensor<bfloat16_t> outputGlobal;
        GlobalTensor<float> finalStateGlobal, stateDecayGlobal;
        outputGlobal.SetGlobalBuffer(outputGMAddr);
        finalStateGlobal.SetGlobalBuffer(finalStateGMAddr);
        stateDecayGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGMAddr + data.stateDecayOffset));

        LocalTensor<bfloat16_t> stateL1Local(TPosition::A1, STATE_L1_ADDR, STATE_TILE_ELEMS * DB_SLOT_NUM);
        LocalTensor<bfloat16_t> rL1Local(TPosition::A1, R_L1_ADDR, CHUNK_DV_TILE_ELEMS);
        LocalTensor<float> predUBLocal(TPosition::VECCALC, PRED_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> historyUBLocal(TPosition::VECCALC, HISTORY_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> deltaUBLocal(TPosition::VECCALC, DELTA_UB_ADDR, STATE_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> localUBLocal(TPosition::VECCALC, LOCAL_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> uUBLocal(TPosition::VECCALC, U_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> stateUBLocal(TPosition::VECCALC, STATE_VALUE_UB_ADDR, STATE_HALF_ELEMS);
        LocalTensor<bfloat16_t> stateShadowUBLocal(TPosition::VECCALC, STATE_SHADOW_UB_ADDR, STATE_HALF_ELEMS);
        LocalTensor<bfloat16_t> rUBLocal(TPosition::VECCALC, R_BF16_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> stateDecayUBLocal(TPosition::VECCALC, STATE_DECAY_UB_ADDR, HEAD_DIM * DB_SLOT_NUM);
        LocalTensor<bfloat16_t> outputUBLocal(
            TPosition::VECCALC, OUTPUT_BF16_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);

        const uint32_t aivIdx = GetBlockIdx();
        const uint32_t subAivIdx = GetSubBlockIdx();
        const uint32_t aicIdx = aivIdx / GetSubBlockNum();

        for (uint32_t taskId = aicIdx; taskId < data.stateNumTasks; taskId += data.stateUseAicNum) {
            const uint32_t batchId = taskId / DV_TILE_COUNT;
            const uint32_t dvTileId = taskId % DV_TILE_COUNT;
            const uint32_t valueColumn = dvTileId * DV_TILE + subAivIdx * AIV_DV_TILE;
            // U/prediction, delta 和 history/local 由 AIC 写入. AIV 先发布空槽.
            for (uint32_t slot = 0; slot < DB_SLOT_NUM; ++slot) {
                SetAivToAic<PIPE_V>(SlotFlagId(FLAG_U_PRED_HANDOFF_BASE, slot));
                SetAivToAic<PIPE_V>(SlotFlagId(FLAG_DELTA_HANDOFF_BASE, slot));
                SetAivToAic<PIPE_V>(SlotFlagId(FLAG_HISTORY_LOCAL_HANDOFF_BASE, slot));
            }

            Mutex::Lock<PIPE_V>(MUTEX_STATE_VALUE_UB);
            Duplicate<float>(stateUBLocal, 0.0F, STATE_HALF_ELEMS);
            Mutex::Lock<PIPE_V>(MUTEX_STATE_SHADOW_UB);
            Duplicate<uint16_t>(stateShadowUBLocal.ReinterpretCast<uint16_t>(), 0, STATE_HALF_ELEMS);
            Mutex::Unlock<PIPE_V>(MUTEX_STATE_SHADOW_UB);
            PublishState(stateL1Local, stateShadowUBLocal, 0, subAivIdx);

            if (data.chunkCount > 0) {
                const uint64_t firstChunkIndex = static_cast<uint64_t>(batchId) * data.chunkCount;
                PreloadDecay(stateDecayUBLocal, stateDecayGlobal, firstChunkIndex * HEAD_DIM, MUTEX_DECAY_UB_BASE);
            }

            for (uint32_t chunkId = 0; chunkId < data.chunkCount; ++chunkId) {
                const uint32_t slot = chunkId % DB_SLOT_NUM;
                const uint32_t nextSlot = slot ^ 1;
                const uint32_t firstToken = chunkId * CHUNK_SIZE;
                const uint32_t validLen = data.seqLen - firstToken < CHUNK_SIZE ? data.seqLen - firstToken : CHUNK_SIZE;
                const uint64_t chunkIndex = static_cast<uint64_t>(batchId) * data.chunkCount + chunkId;
                auto decaySlotUBLocal = stateDecayUBLocal[slot * HEAD_DIM];
                auto rSlotUBLocal = rUBLocal[slot * CHUNK_DV_HALF_ELEMS];
                auto outputSlotUBLocal = outputUBLocal[slot * CHUNK_DV_HALF_ELEMS];
                const MutexId decayMutexId = MUTEX_DECAY_UB_BASE + static_cast<MutexId>(slot);
                const MutexId outputMutexId = MUTEX_OUTPUT_UB_BASE + static_cast<MutexId>(slot);
                const MutexId rMutexId = MUTEX_R_UB_BASE + static_cast<MutexId>(slot);

                if (chunkId + 1 < data.chunkCount) {
                    auto nextDecaySlotUBLocal = stateDecayUBLocal[nextSlot * HEAD_DIM];
                    PreloadDecay(
                        nextDecaySlotUBLocal, stateDecayGlobal, (chunkIndex + 1) * HEAD_DIM,
                        MUTEX_DECAY_UB_BASE + static_cast<MutexId>(nextSlot));
                }

                Mutex::Lock<PIPE_V>(rMutexId);
                WaitAicToAiv<PIPE_V>(SlotFlagId(FLAG_U_PRED_HANDOFF_BASE, slot));
                asc_vf_call<ComputeRAndCastVF>(
                    reinterpret_cast<__ubuf__ bfloat16_t*>(rSlotUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(uUBLocal[slot * CHUNK_DV_HALF_ELEMS].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(predUBLocal[slot * CHUNK_DV_HALF_ELEMS].GetPhyAddr()));
                SetAivToAic<PIPE_V>(SlotFlagId(FLAG_U_PRED_HANDOFF_BASE, slot));
                Mutex::Unlock<PIPE_V>(rMutexId);

                Mutex::Lock<PIPE_MTE3>(rMutexId);
                WaitAicToAiv<PIPE_MTE3>(FLAG_R_L1_HANDOFF);
                DataCopy(rL1Local[subAivIdx * CHUNK_DV_HALF_ELEMS], rSlotUBLocal, CHUNK_DV_HALF_ELEMS);
                SetAivToAic<PIPE_MTE3>(FLAG_R_L1_HANDOFF);
                Mutex::Unlock<PIPE_MTE3>(rMutexId);

                Mutex::Lock<PIPE_V>(decayMutexId);
                Mutex::Lock<PIPE_V>(MUTEX_STATE_SHADOW_UB);
                WaitAicToAiv<PIPE_V>(SlotFlagId(FLAG_DELTA_HANDOFF_BASE, slot));
                asc_vf_call<UpdateStateAndShadowVF>(
                    reinterpret_cast<__ubuf__ float*>(stateUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ bfloat16_t*>(stateShadowUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(deltaUBLocal[slot * STATE_HALF_ELEMS].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(decaySlotUBLocal.GetPhyAddr()));
                SetAivToAic<PIPE_V>(SlotFlagId(FLAG_DELTA_HANDOFF_BASE, slot));
                Mutex::Unlock<PIPE_V>(MUTEX_STATE_SHADOW_UB);
                Mutex::Unlock<PIPE_V>(decayMutexId);

                if (chunkId + 1 < data.chunkCount) {
                    PublishState(stateL1Local, stateShadowUBLocal, nextSlot, subAivIdx);
                }

                Mutex::Lock<PIPE_V>(outputMutexId);
                WaitAicToAiv<PIPE_V>(SlotFlagId(FLAG_HISTORY_LOCAL_HANDOFF_BASE, slot));
                asc_vf_call<OutputAddCastVF>(
                    reinterpret_cast<__ubuf__ bfloat16_t*>(outputSlotUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(historyUBLocal[slot * CHUNK_DV_HALF_ELEMS].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(localUBLocal[slot * CHUNK_DV_HALF_ELEMS].GetPhyAddr()));
                SetAivToAic<PIPE_V>(SlotFlagId(FLAG_HISTORY_LOCAL_HANDOFF_BASE, slot));
                Mutex::Unlock<PIPE_V>(outputMutexId);

                Mutex::Lock<PIPE_MTE3>(outputMutexId);
                const uint64_t outputOffset =
                    (static_cast<uint64_t>(batchId) * data.seqLen + firstToken) * VALUE_DIM + valueColumn;
                CopyUbToGmRows(
                    outputGlobal[outputOffset], outputSlotUBLocal, validLen, AIV_DV_TILE, AIV_DV_TILE, VALUE_DIM);
                Mutex::Unlock<PIPE_MTE3>(outputMutexId);
            }

            Mutex::Unlock<PIPE_V>(MUTEX_STATE_VALUE_UB);
            Mutex::Lock<PIPE_MTE3>(MUTEX_STATE_VALUE_UB);
            const uint64_t finalStateOffset = static_cast<uint64_t>(batchId) * HEAD_DIM * VALUE_DIM + valueColumn;
            CopyUbToGmRows(
                finalStateGlobal[finalStateOffset], stateUBLocal, HEAD_DIM, AIV_DV_TILE, AIV_DV_TILE, VALUE_DIM);
            Mutex::Unlock<PIPE_MTE3>(MUTEX_STATE_VALUE_UB);

            // 消费 AIC 归还的 state/R 空槽信号, 使本 task 的 CrossCore Set/Wait 成对.
            for (uint32_t slot = 0; slot < DB_SLOT_NUM; ++slot) {
                WaitAicToAiv<PIPE_MTE3>(SlotFlagId(FLAG_STATE_HANDOFF_BASE, slot));
            }
            WaitAicToAiv<PIPE_MTE3>(FLAG_R_L1_HANDOFF);
        }
    }
}

} // namespace KDALite
