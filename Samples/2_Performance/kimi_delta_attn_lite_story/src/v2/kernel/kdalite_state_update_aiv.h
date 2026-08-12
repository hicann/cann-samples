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
constexpr uint32_t STATE_SHADOW_UB_ADDR =
    STATE_VALUE_UB_ADDR + STATE_HALF_ELEMS * sizeof(float) * STATE_PIPELINE_MAX_LANE_NUM;
constexpr uint32_t R_BF16_UB_ADDR =
    STATE_SHADOW_UB_ADDR + STATE_HALF_ELEMS * sizeof(bfloat16_t) * STATE_PIPELINE_MAX_LANE_NUM;
constexpr uint32_t STATE_DECAY_UB_ADDR = R_BF16_UB_ADDR + CHUNK_DV_HALF_ELEMS * sizeof(bfloat16_t) * DB_SLOT_NUM;
constexpr uint32_t OUTPUT_BF16_UB_ADDR = STATE_DECAY_UB_ADDR + HEAD_DIM * sizeof(float) * STATE_PIPELINE_MAX_LANE_NUM;
constexpr uint32_t STATE_AIV_UB_END_ADDR = OUTPUT_BF16_UB_ADDR + CHUNK_DV_HALF_ELEMS * sizeof(bfloat16_t) * DB_SLOT_NUM;

static_assert(STATE_AIV_UB_END_ADDR <= AIV_USABLE_UB_BYTES, "FusedRecurrentOutput UB allocation exceeds 248 KiB");

constexpr MutexId MUTEX_STATE_UB_BASE = 0;
constexpr MutexId MUTEX_R_UB_BASE = MUTEX_STATE_UB_BASE + STATE_PIPELINE_MAX_LANE_NUM;
constexpr MutexId MUTEX_DECAY_UB_BASE = MUTEX_R_UB_BASE + DB_SLOT_NUM;
constexpr MutexId MUTEX_OUTPUT_UB_BASE = MUTEX_DECAY_UB_BASE + STATE_PIPELINE_MAX_LANE_NUM;
static_assert(MUTEX_OUTPUT_UB_BASE + DB_SLOT_NUM - 1 <= 27, "StateOutput AIV MutexID exceeds 27");

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
    AscendC::LocalTensor<bfloat16_t>& stateL1Local, AscendC::LocalTensor<bfloat16_t>& stateShadowUBLocal, uint32_t lane,
    uint32_t subAivIdx, MutexId stateMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_MTE3>(stateMutexId);
    WaitAicToAiv<PIPE_MTE3>(SlotFlagId(FLAG_STATE_HANDOFF_BASE, lane));
    DataCopy(
        stateL1Local[lane * STATE_TILE_ELEMS + subAivIdx * STATE_HALF_ELEMS], stateShadowUBLocal, STATE_HALF_ELEMS);
    SetAivToAic<PIPE_MTE3>(SlotFlagId(FLAG_STATE_HANDOFF_BASE, lane));
    Mutex::Unlock<PIPE_MTE3>(stateMutexId);
}

__aicore__ inline void IssueStateV1(
    AscendC::LocalTensor<bfloat16_t>& rL1Local, AscendC::LocalTensor<float>& predUBLocal,
    AscendC::LocalTensor<float>& uUBLocal, AscendC::LocalTensor<bfloat16_t>& rUBLocal, uint32_t itemId,
    uint32_t rQueueDepth, uint32_t subAivIdx)
{
    using namespace AscendC;
    const uint32_t uPredSlot = itemId % DB_SLOT_NUM;
    const uint32_t rStageSlot = itemId % DB_SLOT_NUM;
    const uint32_t rQueueSlot = itemId % rQueueDepth;
    const MutexId rMutexId = MUTEX_R_UB_BASE + static_cast<MutexId>(rStageSlot);
    auto rSlotUBLocal = rUBLocal[rStageSlot * CHUNK_DV_HALF_ELEMS];

    Mutex::Lock<PIPE_V>(rMutexId);
    WaitAicToAiv<PIPE_V>(SlotFlagId(FLAG_U_PRED_HANDOFF_BASE, uPredSlot));
    asc_vf_call<ComputeRAndCastVF>(
        reinterpret_cast<__ubuf__ bfloat16_t*>(rSlotUBLocal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(uUBLocal[uPredSlot * CHUNK_DV_HALF_ELEMS].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(predUBLocal[uPredSlot * CHUNK_DV_HALF_ELEMS].GetPhyAddr()));
    SetAivToAic<PIPE_V>(SlotFlagId(FLAG_U_PRED_HANDOFF_BASE, uPredSlot));
    Mutex::Unlock<PIPE_V>(rMutexId);

    Mutex::Lock<PIPE_MTE3>(rMutexId);
    WaitAicToAiv<PIPE_MTE3>(SlotFlagId(FLAG_R_HANDOFF_BASE, rQueueSlot));
    DataCopy(
        rL1Local[rQueueSlot * CHUNK_DV_TILE_ELEMS + subAivIdx * CHUNK_DV_HALF_ELEMS], rSlotUBLocal,
        CHUNK_DV_HALF_ELEMS);
    SetAivToAic<PIPE_MTE3>(SlotFlagId(FLAG_R_HANDOFF_BASE, rQueueSlot));
    Mutex::Unlock<PIPE_MTE3>(rMutexId);
}

__aicore__ inline void IssueStateV2(
    const AscendC::GlobalTensor<bfloat16_t>& outputGlobal, AscendC::LocalTensor<bfloat16_t>& stateL1Local,
    AscendC::LocalTensor<float>& historyUBLocal, AscendC::LocalTensor<float>& deltaUBLocal,
    AscendC::LocalTensor<float>& localUBLocal, AscendC::LocalTensor<float>& stateUBLocal,
    AscendC::LocalTensor<bfloat16_t>& stateShadowUBLocal, AscendC::LocalTensor<float>& stateDecayUBLocal,
    AscendC::LocalTensor<bfloat16_t>& outputUBLocal, const KimiDeltaAttnLiteTilingData& data, uint32_t taskId,
    uint32_t lane, uint32_t chunkId, uint32_t itemId, uint32_t subAivIdx)
{
    using namespace AscendC;
    const uint32_t batchId = taskId / DV_TILE_COUNT;
    const uint32_t dvTileId = taskId % DV_TILE_COUNT;
    const uint32_t valueColumn = dvTileId * DV_TILE + subAivIdx * AIV_DV_TILE;
    const uint32_t firstToken = chunkId * CHUNK_SIZE;
    const uint32_t validLen = data.seqLen - firstToken < CHUNK_SIZE ? data.seqLen - firstToken : CHUNK_SIZE;
    const uint32_t v2Slot = itemId % DB_SLOT_NUM;
    const MutexId stateMutexId = MUTEX_STATE_UB_BASE + static_cast<MutexId>(lane);
    const uint32_t decaySlot = lane;
    const MutexId decayMutexId = MUTEX_DECAY_UB_BASE + static_cast<MutexId>(decaySlot);
    const MutexId outputMutexId = MUTEX_OUTPUT_UB_BASE + static_cast<MutexId>(v2Slot);
    const uint16_t v2FlagId = SlotFlagId(FLAG_V2_PHASE_HANDOFF_BASE, v2Slot);
    auto stateSlotUBLocal = stateUBLocal[lane * STATE_HALF_ELEMS];
    auto stateShadowSlotUBLocal = stateShadowUBLocal[lane * STATE_HALF_ELEMS];
    auto decaySlotUBLocal = stateDecayUBLocal[decaySlot * HEAD_DIM];
    auto outputSlotUBLocal = outputUBLocal[v2Slot * CHUNK_DV_HALF_ELEMS];

    Mutex::Lock<PIPE_V>(decayMutexId);
    Mutex::Lock<PIPE_V>(stateMutexId);
    WaitAicToAiv<PIPE_V>(v2FlagId);
    asc_vf_call<UpdateStateAndShadowVF>(
        reinterpret_cast<__ubuf__ float*>(stateSlotUBLocal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t*>(stateShadowSlotUBLocal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(deltaUBLocal[v2Slot * STATE_HALF_ELEMS].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(decaySlotUBLocal.GetPhyAddr()));
    SetAivToAic<PIPE_V>(v2FlagId);
    Mutex::Unlock<PIPE_V>(stateMutexId);
    Mutex::Unlock<PIPE_V>(decayMutexId);

    if (chunkId + 1 < data.chunkCount) {
        PublishState(stateL1Local, stateShadowSlotUBLocal, lane, subAivIdx, stateMutexId);
    }

    Mutex::Lock<PIPE_V>(outputMutexId);
    WaitAicToAiv<PIPE_V>(v2FlagId);
    asc_vf_call<OutputAddCastVF>(
        reinterpret_cast<__ubuf__ bfloat16_t*>(outputSlotUBLocal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(historyUBLocal[v2Slot * CHUNK_DV_HALF_ELEMS].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(localUBLocal[v2Slot * CHUNK_DV_HALF_ELEMS].GetPhyAddr()));
    SetAivToAic<PIPE_V>(v2FlagId);
    Mutex::Unlock<PIPE_V>(outputMutexId);

    Mutex::Lock<PIPE_MTE3>(outputMutexId);
    const uint64_t outputOffset = (static_cast<uint64_t>(batchId) * data.seqLen + firstToken) * VALUE_DIM + valueColumn;
    CopyUbToGmRows(outputGlobal[outputOffset], outputSlotUBLocal, validLen, AIV_DV_TILE, AIV_DV_TILE, VALUE_DIM);
    Mutex::Unlock<PIPE_MTE3>(outputMutexId);
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

        LocalTensor<bfloat16_t> stateL1Local(
            TPosition::A1, STATE_L1_ADDR, STATE_TILE_ELEMS * STATE_PIPELINE_MAX_LANE_NUM);
        LocalTensor<bfloat16_t> rL1Local(TPosition::A1, R_L1_ADDR, CHUNK_DV_TILE_ELEMS * STATE_R_MAX_QUEUE_DEPTH);
        LocalTensor<float> predUBLocal(TPosition::VECCALC, PRED_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> historyUBLocal(TPosition::VECCALC, HISTORY_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> deltaUBLocal(TPosition::VECCALC, DELTA_UB_ADDR, STATE_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> localUBLocal(TPosition::VECCALC, LOCAL_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> uUBLocal(TPosition::VECCALC, U_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> stateUBLocal(
            TPosition::VECCALC, STATE_VALUE_UB_ADDR, STATE_HALF_ELEMS * STATE_PIPELINE_MAX_LANE_NUM);
        LocalTensor<bfloat16_t> stateShadowUBLocal(
            TPosition::VECCALC, STATE_SHADOW_UB_ADDR, STATE_HALF_ELEMS * STATE_PIPELINE_MAX_LANE_NUM);
        LocalTensor<bfloat16_t> rUBLocal(TPosition::VECCALC, R_BF16_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> stateDecayUBLocal(
            TPosition::VECCALC, STATE_DECAY_UB_ADDR, HEAD_DIM * STATE_PIPELINE_MAX_LANE_NUM);
        LocalTensor<bfloat16_t> outputUBLocal(
            TPosition::VECCALC, OUTPUT_BF16_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);

        const uint32_t aivIdx = GetBlockIdx();
        const uint32_t subAivIdx = GetSubBlockIdx();
        const uint32_t aicIdx = aivIdx / GetSubBlockNum();
        const uint32_t configuredLaneCount = data.statePipelineLaneCount;
        const uint64_t waveStride = static_cast<uint64_t>(data.stateUseAicNum) * configuredLaneCount;
        for (uint64_t waveTaskBase = static_cast<uint64_t>(aicIdx) * configuredLaneCount;
             waveTaskBase < data.stateNumTasks; waveTaskBase += waveStride) {
            const uint32_t remainingTasks = data.stateNumTasks - static_cast<uint32_t>(waveTaskBase);
            const uint32_t laneCount = remainingTasks < configuredLaneCount ? remainingTasks : configuredLaneCount;
            const uint32_t rQueueDepth = laneCount > 1 ? laneCount - 1 : 1;

            for (uint32_t slot = 0; slot < DB_SLOT_NUM; ++slot) {
                SetAivToAic<PIPE_V>(SlotFlagId(FLAG_U_PRED_HANDOFF_BASE, slot));
                SetAivToAic<PIPE_V>(SlotFlagId(FLAG_V2_PHASE_HANDOFF_BASE, slot));
            }

            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                const MutexId stateMutexId = MUTEX_STATE_UB_BASE + static_cast<MutexId>(lane);
                auto stateSlotUBLocal = stateUBLocal[lane * STATE_HALF_ELEMS];
                auto stateShadowSlotUBLocal = stateShadowUBLocal[lane * STATE_HALF_ELEMS];
                Mutex::Lock<PIPE_V>(stateMutexId);
                Duplicate<float>(stateSlotUBLocal, 0.0F, STATE_HALF_ELEMS);
                Duplicate<uint16_t>(stateShadowSlotUBLocal.ReinterpretCast<uint16_t>(), 0, STATE_HALF_ELEMS);
                Mutex::Unlock<PIPE_V>(stateMutexId);
                PublishState(stateL1Local, stateShadowSlotUBLocal, lane, subAivIdx, stateMutexId);
            }

            const uint32_t itemCount = data.chunkCount * laneCount;
            const uint32_t epochCount = itemCount + laneCount;
            for (uint32_t epoch = 0; epoch < epochCount; ++epoch) {
                if (epoch >= 1 && epoch < itemCount + 1) {
                    const uint32_t itemId = epoch - 1;
                    const uint32_t lane = itemId % laneCount;
                    const uint32_t chunkId = itemId / laneCount;
                    const uint32_t taskId = static_cast<uint32_t>(waveTaskBase) + lane;
                    const uint32_t batchId = taskId / DV_TILE_COUNT;
                    const uint64_t chunkIndex = static_cast<uint64_t>(batchId) * data.chunkCount + chunkId;
                    const uint32_t decaySlot = lane;
                    const MutexId decayMutexId = MUTEX_DECAY_UB_BASE + static_cast<MutexId>(decaySlot);
                    auto decaySlotUBLocal = stateDecayUBLocal[decaySlot * HEAD_DIM];
                    // stateDecay 在 V1 前预取到该 lane 的独占槽. 同 lane 的下一个 V1
                    // 排在当前 V2 之后, Mutex 保证 V 读完前 MTE2 不会覆写该槽.
                    PreloadDecay(decaySlotUBLocal, stateDecayGlobal, chunkIndex * HEAD_DIM, decayMutexId);
                    IssueStateV1(rL1Local, predUBLocal, uUBLocal, rUBLocal, itemId, rQueueDepth, subAivIdx);
                }
                if (epoch >= laneCount && epoch < itemCount + laneCount) {
                    const uint32_t itemId = epoch - laneCount;
                    const uint32_t lane = itemId % laneCount;
                    const uint32_t chunkId = itemId / laneCount;
                    const uint32_t taskId = static_cast<uint32_t>(waveTaskBase) + lane;
                    IssueStateV2(
                        outputGlobal, stateL1Local, historyUBLocal, deltaUBLocal, localUBLocal, stateUBLocal,
                        stateShadowUBLocal, stateDecayUBLocal, outputUBLocal, data, taskId, lane, chunkId, itemId,
                        subAivIdx);
                }
            }

            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                const uint32_t taskId = static_cast<uint32_t>(waveTaskBase) + lane;
                const uint32_t batchId = taskId / DV_TILE_COUNT;
                const uint32_t dvTileId = taskId % DV_TILE_COUNT;
                const uint32_t valueColumn = dvTileId * DV_TILE + subAivIdx * AIV_DV_TILE;
                const MutexId stateMutexId = MUTEX_STATE_UB_BASE + static_cast<MutexId>(lane);
                auto stateSlotUBLocal = stateUBLocal[lane * STATE_HALF_ELEMS];
                Mutex::Lock<PIPE_MTE3>(stateMutexId);
                const uint64_t finalStateOffset = static_cast<uint64_t>(batchId) * HEAD_DIM * VALUE_DIM + valueColumn;
                CopyUbToGmRows(
                    finalStateGlobal[finalStateOffset], stateSlotUBLocal, HEAD_DIM, AIV_DV_TILE, AIV_DV_TILE,
                    VALUE_DIM);
                Mutex::Unlock<PIPE_MTE3>(stateMutexId);
            }

            for (uint32_t lane = 0; lane < STATE_PIPELINE_MAX_LANE_NUM; ++lane) {
                WaitAicToAiv<PIPE_MTE3>(SlotFlagId(FLAG_STATE_HANDOFF_BASE, lane));
            }
            for (uint32_t slot = 0; slot < STATE_R_MAX_QUEUE_DEPTH; ++slot) {
                WaitAicToAiv<PIPE_MTE3>(SlotFlagId(FLAG_R_HANDOFF_BASE, slot));
            }
            // 固定的 state/R 物理槽均已排空. 此时复用 state lane 0 的 FlagID
            // 传递 wave 完成信号, 不会与该 lane 的双向交接混淆.
            SetAivToAic<PIPE_MTE3>(FLAG_STATE_WAVE_DONE);
        }
    }
}

} // namespace KDALite
