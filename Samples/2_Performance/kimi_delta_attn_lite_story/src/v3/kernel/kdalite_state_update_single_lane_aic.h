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

#include "kdalite_state_update_aic.h"

namespace KDALite::SingleLane {

// R=1 时按 chunk 顺序递推. Chunk 只读矩阵/Value 和 state 使用奇偶双槽,
// output 另设四槽 L0C 队列, 下一个 chunk 的输入和 U 在当前 chunk 的 C2 前预发射.

constexpr uint32_t CHUNK_MATRIX_W_SLOT_ADDR = 0;
constexpr uint32_t CHUNK_MATRIX_Q_PLUS_SLOT_ADDR = CHUNK_MATRIX_W_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t CHUNK_MATRIX_M_SLOT_ADDR = CHUNK_MATRIX_Q_PLUS_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t CHUNK_MATRIX_K_TAIL_SLOT_ADDR = CHUNK_MATRIX_M_SLOT_ADDR + CHUNK_C_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t CHUNK_MATRIX_A_SLOT_ADDR = CHUNK_MATRIX_K_TAIL_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t CHUNK_MATRIX_SLOT_BYTES = CHUNK_MATRIX_A_SLOT_ADDR + CHUNK_C_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t STATE_L1_ADDR = CHUNK_MATRIX_SLOT_BYTES * DB_SLOT_NUM;
constexpr uint32_t VALUE_L1_ADDR = STATE_L1_ADDR + STATE_TILE_ELEMS * sizeof(bfloat16_t) * DB_SLOT_NUM;
constexpr uint32_t R_L1_ADDR = VALUE_L1_ADDR + CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t) * DB_SLOT_NUM;
constexpr uint32_t AIC_L1_END_ADDR = R_L1_ADDR + CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t);

constexpr uint32_t L0A_SLOT_ELEMS = CHUNK_D_ELEMS;
constexpr uint32_t STATE_L0B_ADDR = 0;
constexpr uint32_t VALUE_L0B_ADDR = STATE_L0B_ADDR + STATE_TILE_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t R_L0B_ADDR = VALUE_L0B_ADDR + CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t L0C_SLOT_ELEMS = STATE_TILE_ELEMS;
constexpr uint32_t OUTPUT_L0C_ADDR = L0C_SLOT_ELEMS * sizeof(float) * L0C_QUEUE_DEPTH;
constexpr uint32_t OUTPUT_L0C_SLOT_ELEMS = CHUNK_DV_TILE_ELEMS;
constexpr uint32_t OUTPUT_L0C_QUEUE_DEPTH = L0C_QUEUE_DEPTH;

static_assert(AIC_L1_END_ADDR <= 512 * 1024, "StateOutput single-lane L1 allocation exceeds 512 KiB");
static_assert(L0A_SLOT_ELEMS * sizeof(bfloat16_t) * DB_SLOT_NUM <= 64 * 1024, "L0A allocation exceeds 64 KiB");
static_assert(
    R_L0B_ADDR + CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t) <= 64 * 1024,
    "StateOutput single-lane L0B allocation exceeds 64 KiB");
static_assert(
    OUTPUT_L0C_ADDR + OUTPUT_L0C_SLOT_ELEMS * sizeof(float) * OUTPUT_L0C_QUEUE_DEPTH <= 256 * 1024,
    "L0C allocation exceeds 256 KiB");

constexpr MutexId MUTEX_CHUNK_MATRIX_L1_BASE = 0;
constexpr MutexId MUTEX_L0A_BASE = MUTEX_CHUNK_MATRIX_L1_BASE + DB_SLOT_NUM;
constexpr MutexId MUTEX_STATE_L0B = MUTEX_L0A_BASE + DB_SLOT_NUM;
constexpr MutexId MUTEX_VALUE_L0B = MUTEX_STATE_L0B + 1;
constexpr MutexId MUTEX_R_L0B = MUTEX_VALUE_L0B + 1;
constexpr MutexId MUTEX_L0C_BASE = MUTEX_R_L0B + 1;
constexpr MutexId MUTEX_OUTPUT_L0C_BASE = MUTEX_L0C_BASE + L0C_QUEUE_DEPTH;
static_assert(
    MUTEX_OUTPUT_L0C_BASE + OUTPUT_L0C_QUEUE_DEPTH - 1 <= 27, "StateOutput single-lane AIC MutexID exceeds 27");

__aicore__ inline void PreloadChunkInputs(
    AscendC::LocalTensor<bfloat16_t>& wL1Local, AscendC::LocalTensor<bfloat16_t>& qPlusL1Local,
    AscendC::LocalTensor<bfloat16_t>& mL1Local, AscendC::LocalTensor<bfloat16_t>& kTailL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL1Local, AscendC::LocalTensor<bfloat16_t>& valueL1Local,
    const AscendC::GlobalTensor<bfloat16_t>& wGlobal, const AscendC::GlobalTensor<bfloat16_t>& qPlusGlobal,
    const AscendC::GlobalTensor<bfloat16_t>& mGlobal, const AscendC::GlobalTensor<bfloat16_t>& kTailGlobal,
    const AscendC::GlobalTensor<bfloat16_t>& aGlobal, const AscendC::GlobalTensor<bfloat16_t>& valueGlobal,
    uint64_t chunkDOffset, uint64_t chunkCOffset, uint64_t valueOffset, uint32_t validLen, MutexId slotMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_MTE2>(slotMutexId);
    CopyGmToL1(wL1Local, wGlobal[chunkDOffset], CHUNK_SIZE, HEAD_DIM, HEAD_DIM);
    CopyGmToL1(qPlusL1Local, qPlusGlobal[chunkDOffset], CHUNK_SIZE, HEAD_DIM, HEAD_DIM);
    CopyGmToL1(mL1Local, mGlobal[chunkCOffset], CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);
    CopyGmToL1(kTailL1Local, kTailGlobal[chunkDOffset], CHUNK_SIZE, HEAD_DIM, HEAD_DIM);
    CopyGmToL1(aL1Local, aGlobal[chunkCOffset], CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);
    if (validLen < CHUNK_SIZE) {
        Fill(valueL1Local, {1, CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t) / C0_BYTES, 0, static_cast<bfloat16_t>(0)});
    }
    CopyGmToL1(valueL1Local, valueGlobal[valueOffset], validLen, DV_TILE, VALUE_DIM, CHUNK_SIZE);
    Mutex::Unlock<PIPE_MTE2>(slotMutexId);
}

__aicore__ inline void IssueUForSlot(
    AscendC::LocalTensor<bfloat16_t>& mL1Local, AscendC::LocalTensor<bfloat16_t>& valueSlotL1Local,
    AscendC::LocalTensor<bfloat16_t>& valueL0BLocal, AscendC::LocalTensor<bfloat16_t>& uAL0Local,
    AscendC::LocalTensor<float>& uCL0Local, AscendC::LocalTensor<float>& uSlotUBLocal, uint32_t slot,
    MutexId chunkMatrixMutexId, MutexId l0aMutexId, MutexId l0cMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_MTE1>(chunkMatrixMutexId);
    Mutex::Lock<PIPE_MTE1>(MUTEX_VALUE_L0B);
    CopyL1ToL0B(valueL0BLocal, valueSlotL1Local, CHUNK_SIZE, CHUNK_SIZE, DV_TILE, true);
    Mutex::Unlock<PIPE_MTE1>(MUTEX_VALUE_L0B);
    KDALite::LoadA(uAL0Local, mL1Local, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, false, l0aMutexId);
    Mutex::Unlock<PIPE_MTE1>(chunkMatrixMutexId);

    KDALite::IssueMmad(
        uCL0Local, uAL0Local, valueL0BLocal, CHUNK_SIZE, DV_TILE, CHUNK_SIZE, l0aMutexId, MUTEX_VALUE_L0B, l0cMutexId);
    KDALite::FixToAivBegin(
        uSlotUBLocal, uCL0Local, CHUNK_SIZE, DV_TILE, SlotFlagId(FLAG_U_PRED_HANDOFF_BASE, slot), l0cMutexId);
}

__aicore__ inline void KernelProcessStateUpdateForAIC(
    __gm__ bfloat16_t* valueGMAddr, __gm__ bfloat16_t* outputGMAddr, __gm__ uint8_t* workspaceGMAddr,
    const KimiDeltaAttnLiteTilingData& data)
{
    using namespace AscendC;
    if ASCEND_IS_AIC {
        GlobalTensor<bfloat16_t> wGlobal, qPlusGlobal, kTailGlobal, mGlobal, aGlobal, valueGlobal, outputGlobal;
        wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.wOffset));
        qPlusGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.qPlusOffset));
        kTailGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.kTailOffset));
        mGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.mOffset));
        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.aOffset));
        valueGlobal.SetGlobalBuffer(valueGMAddr);
        outputGlobal.SetGlobalBuffer(outputGMAddr);

        // 每个槽打包当前 Chunk 的 W/QPlus/M/KTail/A; 两个槽按 Chunk 奇偶复用.
        LocalTensor<uint8_t> chunkMatrixL1Local(TPosition::A1, 0, CHUNK_MATRIX_SLOT_BYTES * DB_SLOT_NUM);
        LocalTensor<bfloat16_t> stateL1Local(TPosition::A1, STATE_L1_ADDR, STATE_TILE_ELEMS * DB_SLOT_NUM);
        LocalTensor<bfloat16_t> valueL1Local(TPosition::A1, VALUE_L1_ADDR, CHUNK_DV_TILE_ELEMS * DB_SLOT_NUM);
        LocalTensor<bfloat16_t> rL1Local(TPosition::A1, R_L1_ADDR, CHUNK_DV_TILE_ELEMS);
        LocalTensor<bfloat16_t> aL0ALocal(TPosition::A2, 0, L0A_SLOT_ELEMS * DB_SLOT_NUM);
        LocalTensor<bfloat16_t> stateL0BLocal(TPosition::B2, STATE_L0B_ADDR, STATE_TILE_ELEMS);
        LocalTensor<bfloat16_t> valueL0BLocal(TPosition::B2, VALUE_L0B_ADDR, CHUNK_DV_TILE_ELEMS);
        LocalTensor<bfloat16_t> rL0BLocal(TPosition::B2, R_L0B_ADDR, CHUNK_DV_TILE_ELEMS);
        // 通用 L0C 队列承载 U、prediction 和 delta; outputL0C 单独承载 history+local.
        LocalTensor<float> mmadL0CQueueLocal(TPosition::CO1, 0, L0C_SLOT_ELEMS * L0C_QUEUE_DEPTH);
        LocalTensor<float> outputL0CLocal(
            TPosition::CO1, OUTPUT_L0C_ADDR, OUTPUT_L0C_SLOT_ELEMS * OUTPUT_L0C_QUEUE_DEPTH);
        LocalTensor<float> predUBLocal(TPosition::VECCALC, PRED_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> deltaUBLocal(TPosition::VECCALC, DELTA_UB_ADDR, STATE_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> uUBLocal(TPosition::VECCALC, U_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);

        for (uint32_t taskId = GetBlockIdx(); taskId < data.stateNumTasks; taskId += data.stateUseAicNum) {
            const uint32_t batchId = taskId / DV_TILE_COUNT;
            const uint32_t dvTileId = taskId % DV_TILE_COUNT;
            const uint32_t valueColumn = dvTileId * DV_TILE;
            for (uint32_t slot = 0; slot < DB_SLOT_NUM; ++slot) {
                SetAicToAiv<PIPE_MTE1>(SlotFlagId(FLAG_STATE_HANDOFF_BASE, slot));
            }
            SetAicToAiv<PIPE_MTE1>(FLAG_SINGLE_LANE_R_HANDOFF);

            uint32_t l0aOpIdx = 0;
            uint32_t l0cOpIdx = 0;
            if (data.chunkCount > 0) {
                const uint64_t firstChunkIndex = static_cast<uint64_t>(batchId) * data.chunkCount;
                auto firstChunkMatrixSlotL1Local = chunkMatrixL1Local[0];
                auto firstWL1Local =
                    firstChunkMatrixSlotL1Local[CHUNK_MATRIX_W_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                auto firstQPlusL1Local =
                    firstChunkMatrixSlotL1Local[CHUNK_MATRIX_Q_PLUS_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                auto firstML1Local =
                    firstChunkMatrixSlotL1Local[CHUNK_MATRIX_M_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                auto firstKTailL1Local =
                    firstChunkMatrixSlotL1Local[CHUNK_MATRIX_K_TAIL_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                auto firstAL1Local =
                    firstChunkMatrixSlotL1Local[CHUNK_MATRIX_A_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                auto firstValueL1Local = valueL1Local;
                const uint32_t firstValidLen = data.seqLen < CHUNK_SIZE ? data.seqLen : CHUNK_SIZE;
                const uint64_t firstValueOffset =
                    static_cast<uint64_t>(batchId) * data.seqLen * VALUE_DIM + valueColumn;
                PreloadChunkInputs(
                    firstWL1Local, firstQPlusL1Local, firstML1Local, firstKTailL1Local, firstAL1Local,
                    firstValueL1Local, wGlobal, qPlusGlobal, mGlobal, kTailGlobal, aGlobal, valueGlobal,
                    firstChunkIndex * CHUNK_D_ELEMS, firstChunkIndex * CHUNK_C_ELEMS, firstValueOffset, firstValidLen,
                    MUTEX_CHUNK_MATRIX_L1_BASE);

                const uint32_t uAIdx = l0aOpIdx++ % DB_SLOT_NUM;
                const uint32_t uCIdx = l0cOpIdx++ % L0C_QUEUE_DEPTH;
                auto uAL0Local = aL0ALocal[uAIdx * L0A_SLOT_ELEMS];
                auto uCL0Local = mmadL0CQueueLocal[uCIdx * L0C_SLOT_ELEMS];
                auto uSlotUBLocal = uUBLocal;
                auto valueSlotL1Local = valueL1Local;
                IssueUForSlot(
                    firstML1Local, valueSlotL1Local, valueL0BLocal, uAL0Local, uCL0Local, uSlotUBLocal, 0,
                    MUTEX_CHUNK_MATRIX_L1_BASE, MUTEX_L0A_BASE + static_cast<MutexId>(uAIdx),
                    MUTEX_L0C_BASE + static_cast<MutexId>(uCIdx));
            }

            for (uint32_t chunkId = 0; chunkId < data.chunkCount; ++chunkId) {
                const uint32_t slot = chunkId % DB_SLOT_NUM;
                const uint32_t nextSlot = slot ^ 1;
                const uint64_t chunkIndex = static_cast<uint64_t>(batchId) * data.chunkCount + chunkId;
                auto chunkMatrixSlotL1Local = chunkMatrixL1Local[slot * CHUNK_MATRIX_SLOT_BYTES];
                const MutexId chunkMatrixMutexId = MUTEX_CHUNK_MATRIX_L1_BASE + static_cast<MutexId>(slot);

                if (chunkId + 1 < data.chunkCount) {
                    const uint64_t nextChunkIndex = chunkIndex + 1;
                    const uint32_t nextFirstToken = (chunkId + 1) * CHUNK_SIZE;
                    const uint32_t nextValidLen =
                        data.seqLen - nextFirstToken < CHUNK_SIZE ? data.seqLen - nextFirstToken : CHUNK_SIZE;
                    const uint64_t nextValueOffset =
                        (static_cast<uint64_t>(batchId) * data.seqLen + nextFirstToken) * VALUE_DIM + valueColumn;
                    auto nextChunkMatrixSlotL1Local = chunkMatrixL1Local[nextSlot * CHUNK_MATRIX_SLOT_BYTES];
                    auto nextWL1Local =
                        nextChunkMatrixSlotL1Local[CHUNK_MATRIX_W_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                    auto nextQPlusL1Local =
                        nextChunkMatrixSlotL1Local[CHUNK_MATRIX_Q_PLUS_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                    auto nextML1Local =
                        nextChunkMatrixSlotL1Local[CHUNK_MATRIX_M_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                    auto nextKTailL1Local =
                        nextChunkMatrixSlotL1Local[CHUNK_MATRIX_K_TAIL_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                    auto nextAL1Local =
                        nextChunkMatrixSlotL1Local[CHUNK_MATRIX_A_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                    auto nextValueL1Local = valueL1Local[nextSlot * CHUNK_DV_TILE_ELEMS];
                    PreloadChunkInputs(
                        nextWL1Local, nextQPlusL1Local, nextML1Local, nextKTailL1Local, nextAL1Local, nextValueL1Local,
                        wGlobal, qPlusGlobal, mGlobal, kTailGlobal, aGlobal, valueGlobal,
                        nextChunkIndex * CHUNK_D_ELEMS, nextChunkIndex * CHUNK_C_ELEMS, nextValueOffset, nextValidLen,
                        MUTEX_CHUNK_MATRIX_L1_BASE + static_cast<MutexId>(nextSlot));
                }

                auto wL1Local = chunkMatrixSlotL1Local[CHUNK_MATRIX_W_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                auto qPlusL1Local = chunkMatrixSlotL1Local[CHUNK_MATRIX_Q_PLUS_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                auto mL1Local = chunkMatrixSlotL1Local[CHUNK_MATRIX_M_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                auto kTailL1Local = chunkMatrixSlotL1Local[CHUNK_MATRIX_K_TAIL_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                auto aL1Local = chunkMatrixSlotL1Local[CHUNK_MATRIX_A_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                auto stateSlotL1Local = stateL1Local[slot * STATE_TILE_ELEMS];

                Mutex::Lock<PIPE_MTE1>(chunkMatrixMutexId);
                WaitAivToAic<PIPE_MTE1>(SlotFlagId(FLAG_STATE_HANDOFF_BASE, slot));
                Mutex::Lock<PIPE_MTE1>(MUTEX_STATE_L0B);
                CopyL1ToL0B(stateL0BLocal, stateSlotL1Local, HEAD_DIM, HEAD_DIM, DV_TILE, true);
                Mutex::Unlock<PIPE_MTE1>(MUTEX_STATE_L0B);
                SetAicToAiv<PIPE_MTE1>(SlotFlagId(FLAG_STATE_HANDOFF_BASE, slot));

                const uint32_t predAIdx = l0aOpIdx++ % DB_SLOT_NUM;
                const uint32_t predCIdx = l0cOpIdx++ % L0C_QUEUE_DEPTH;
                auto predAL0Local = aL0ALocal[predAIdx * L0A_SLOT_ELEMS];
                auto predCL0Local = mmadL0CQueueLocal[predCIdx * L0C_SLOT_ELEMS];
                KDALite::LoadA(
                    predAL0Local, wL1Local, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, false,
                    MUTEX_L0A_BASE + static_cast<MutexId>(predAIdx));
                KDALite::IssueMmadHoldL0B(
                    predCL0Local, predAL0Local, stateL0BLocal, CHUNK_SIZE, DV_TILE, HEAD_DIM,
                    MUTEX_L0A_BASE + static_cast<MutexId>(predAIdx), MUTEX_STATE_L0B,
                    MUTEX_L0C_BASE + static_cast<MutexId>(predCIdx));

                const uint32_t historyAIdx = l0aOpIdx++ % DB_SLOT_NUM;
                auto historyAL0Local = aL0ALocal[historyAIdx * L0A_SLOT_ELEMS];
                const uint32_t outputSlot = chunkId % OUTPUT_L0C_QUEUE_DEPTH;
                auto outputSlotL0CLocal = outputL0CLocal[outputSlot * OUTPUT_L0C_SLOT_ELEMS];
                const MutexId outputMutexId = MUTEX_OUTPUT_L0C_BASE + static_cast<MutexId>(outputSlot);
                KDALite::LoadA(
                    historyAL0Local, qPlusL1Local, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, false,
                    MUTEX_L0A_BASE + static_cast<MutexId>(historyAIdx));
                KDALite::IssueHistoryMmad(
                    outputSlotL0CLocal, historyAL0Local, stateL0BLocal,
                    MUTEX_L0A_BASE + static_cast<MutexId>(historyAIdx), MUTEX_STATE_L0B, outputMutexId);
                auto predSlotUBLocal = predUBLocal[slot * CHUNK_DV_HALF_ELEMS];
                KDALite::FixToAivEnd(
                    predSlotUBLocal, predCL0Local, CHUNK_SIZE, DV_TILE, SlotFlagId(FLAG_U_PRED_HANDOFF_BASE, slot),
                    MUTEX_L0C_BASE + static_cast<MutexId>(predCIdx));

                if (chunkId + 1 < data.chunkCount) {
                    auto nextChunkMatrixSlotL1Local = chunkMatrixL1Local[nextSlot * CHUNK_MATRIX_SLOT_BYTES];
                    auto nextML1Local =
                        nextChunkMatrixSlotL1Local[CHUNK_MATRIX_M_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
                    auto nextValueSlotL1Local = valueL1Local[nextSlot * CHUNK_DV_TILE_ELEMS];
                    auto nextUSlotUBLocal = uUBLocal[nextSlot * CHUNK_DV_HALF_ELEMS];
                    const uint32_t uAIdx = l0aOpIdx++ % DB_SLOT_NUM;
                    const uint32_t uCIdx = l0cOpIdx++ % L0C_QUEUE_DEPTH;
                    auto uAL0Local = aL0ALocal[uAIdx * L0A_SLOT_ELEMS];
                    auto uCL0Local = mmadL0CQueueLocal[uCIdx * L0C_SLOT_ELEMS];
                    IssueUForSlot(
                        nextML1Local, nextValueSlotL1Local, valueL0BLocal, uAL0Local, uCL0Local, nextUSlotUBLocal,
                        nextSlot, MUTEX_CHUNK_MATRIX_L1_BASE + static_cast<MutexId>(nextSlot),
                        MUTEX_L0A_BASE + static_cast<MutexId>(uAIdx), MUTEX_L0C_BASE + static_cast<MutexId>(uCIdx));
                }

                WaitAivToAic<PIPE_MTE1>(FLAG_SINGLE_LANE_R_HANDOFF);
                Mutex::Lock<PIPE_MTE1>(MUTEX_R_L0B);
                CopyL1ToL0B(rL0BLocal, rL1Local, CHUNK_SIZE, CHUNK_SIZE, DV_TILE, true);
                Mutex::Unlock<PIPE_MTE1>(MUTEX_R_L0B);
                SetAicToAiv<PIPE_MTE1>(FLAG_SINGLE_LANE_R_HANDOFF);

                const uint32_t deltaAIdx = l0aOpIdx++ % DB_SLOT_NUM;
                const uint32_t deltaCIdx = l0cOpIdx++ % L0C_QUEUE_DEPTH;
                auto deltaAL0Local = aL0ALocal[deltaAIdx * L0A_SLOT_ELEMS];
                auto deltaCL0Local = mmadL0CQueueLocal[deltaCIdx * L0C_SLOT_ELEMS];
                KDALite::LoadA(
                    deltaAL0Local, kTailL1Local, CHUNK_SIZE, HEAD_DIM, CHUNK_SIZE, true,
                    MUTEX_L0A_BASE + static_cast<MutexId>(deltaAIdx));
                KDALite::IssueMmadHoldL0B(
                    deltaCL0Local, deltaAL0Local, rL0BLocal, HEAD_DIM, DV_TILE, CHUNK_SIZE,
                    MUTEX_L0A_BASE + static_cast<MutexId>(deltaAIdx), MUTEX_R_L0B,
                    MUTEX_L0C_BASE + static_cast<MutexId>(deltaCIdx));

                const uint32_t localAIdx = l0aOpIdx++ % DB_SLOT_NUM;
                auto localAL0Local = aL0ALocal[localAIdx * L0A_SLOT_ELEMS];
                KDALite::LoadA(
                    localAL0Local, aL1Local, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, false,
                    MUTEX_L0A_BASE + static_cast<MutexId>(localAIdx));
                Mutex::Unlock<PIPE_MTE1>(chunkMatrixMutexId);
                KDALite::IssueLocalAccumulateMmad(
                    outputSlotL0CLocal, localAL0Local, rL0BLocal, MUTEX_L0A_BASE + static_cast<MutexId>(localAIdx),
                    MUTEX_R_L0B, outputMutexId);
                auto deltaSlotUBLocal = deltaUBLocal[slot * STATE_HALF_ELEMS];
                KDALite::FixToAiv(
                    deltaSlotUBLocal, deltaCL0Local, HEAD_DIM, DV_TILE,
                    SlotFlagId(FLAG_SINGLE_LANE_DELTA_HANDOFF_BASE, slot),
                    MUTEX_L0C_BASE + static_cast<MutexId>(deltaCIdx));
                const uint32_t firstToken = chunkId * CHUNK_SIZE;
                const uint32_t validLen = data.seqLen - firstToken < CHUNK_SIZE ? data.seqLen - firstToken : CHUNK_SIZE;
                const uint64_t outputOffset =
                    (static_cast<uint64_t>(batchId) * data.seqLen + firstToken) * VALUE_DIM + dvTileId * DV_TILE;
                KDALite::FixOutputToGm(outputGlobal, outputSlotL0CLocal, outputOffset, validLen, outputMutexId);
            }

            for (uint32_t slot = 0; slot < DB_SLOT_NUM; ++slot) {
                WaitAivToAic<PIPE_FIX>(SlotFlagId(FLAG_U_PRED_HANDOFF_BASE, slot));
                WaitAivToAic<PIPE_FIX>(SlotFlagId(FLAG_SINGLE_LANE_DELTA_HANDOFF_BASE, slot));
            }
        }
    }
}

} // namespace KDALite::SingleLane
