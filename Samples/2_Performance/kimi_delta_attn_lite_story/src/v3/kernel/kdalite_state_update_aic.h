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

// chunkMatrixL1Local 将 W/QPlus/M/KTail/A 打包到同一槽, 并按 chunkId%2 使用双槽.
// 同一 chunk 的 lane 0 取得槽所有权,
// 最后一条 lane 在 C2 读完 A 后归还. 地址规划保留 4 个物理槽, 仅使用前 2 个.
// Value 只跨一个局部阶段, 使用双槽.
constexpr uint32_t CHUNK_MATRIX_W_SLOT_ADDR = 0;
constexpr uint32_t CHUNK_MATRIX_Q_PLUS_SLOT_ADDR = CHUNK_MATRIX_W_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t CHUNK_MATRIX_M_SLOT_ADDR = CHUNK_MATRIX_Q_PLUS_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t CHUNK_MATRIX_K_TAIL_SLOT_ADDR = CHUNK_MATRIX_M_SLOT_ADDR + CHUNK_C_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t CHUNK_MATRIX_A_SLOT_ADDR = CHUNK_MATRIX_K_TAIL_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t CHUNK_MATRIX_SLOT_BYTES = CHUNK_MATRIX_A_SLOT_ADDR + CHUNK_C_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t STATE_L1_ADDR = CHUNK_MATRIX_SLOT_BYTES * STATE_PIPELINE_MAX_LANE_NUM;
constexpr uint32_t VALUE_L1_ADDR = STATE_L1_ADDR + STATE_TILE_ELEMS * sizeof(bfloat16_t) * STATE_PIPELINE_MAX_LANE_NUM;
constexpr uint32_t R_L1_ADDR = VALUE_L1_ADDR + CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t) * DB_SLOT_NUM;
constexpr uint32_t AIC_L1_END_ADDR = R_L1_ADDR + CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t) * STATE_R_MAX_QUEUE_DEPTH;

constexpr uint32_t L0A_SLOT_ELEMS = CHUNK_D_ELEMS;
constexpr uint32_t STATE_L0B_ADDR = 0;
constexpr uint32_t VALUE_L0B_ADDR =
    STATE_L0B_ADDR + STATE_TILE_ELEMS * sizeof(bfloat16_t) * STATE_PIPELINE_MAX_LANE_NUM;
constexpr uint32_t R_L0B_ADDR = VALUE_L0B_ADDR + CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t L0C_SLOT_ELEMS = STATE_TILE_ELEMS;
constexpr uint32_t OUTPUT_L0C_ADDR = L0C_SLOT_ELEMS * sizeof(float) * L0C_QUEUE_DEPTH;
constexpr uint32_t OUTPUT_L0C_SLOT_ELEMS = CHUNK_DV_TILE_ELEMS;
constexpr uint32_t OUTPUT_L0C_QUEUE_DEPTH = L0C_QUEUE_DEPTH;

static_assert(AIC_L1_END_ADDR <= 512 * 1024, "FusedRecurrentOutput L1 allocation exceeds 512 KiB");
static_assert(L0A_SLOT_ELEMS * sizeof(bfloat16_t) * DB_SLOT_NUM <= 64 * 1024, "L0A allocation exceeds 64 KiB");
static_assert(R_L0B_ADDR + CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t) <= 64 * 1024, "L0B allocation exceeds 64 KiB");
static_assert(
    OUTPUT_L0C_ADDR + OUTPUT_L0C_SLOT_ELEMS * sizeof(float) * OUTPUT_L0C_QUEUE_DEPTH <= 256 * 1024,
    "L0C allocation exceeds 256 KiB");

constexpr MutexId MUTEX_CHUNK_MATRIX_L1_BASE = 0;
constexpr MutexId MUTEX_VALUE_L1_BASE = MUTEX_CHUNK_MATRIX_L1_BASE + STATE_PIPELINE_MAX_LANE_NUM;
constexpr MutexId MUTEX_L0A_BASE = MUTEX_VALUE_L1_BASE + DB_SLOT_NUM;
constexpr MutexId MUTEX_STATE_L0B_BASE = MUTEX_L0A_BASE + DB_SLOT_NUM;
constexpr MutexId MUTEX_VALUE_L0B = MUTEX_STATE_L0B_BASE + STATE_PIPELINE_MAX_LANE_NUM;
constexpr MutexId MUTEX_R_L0B = MUTEX_VALUE_L0B + 1;
constexpr MutexId MUTEX_L0C_BASE = MUTEX_R_L0B + 1;
constexpr MutexId MUTEX_OUTPUT_L0C_BASE = MUTEX_L0C_BASE + L0C_QUEUE_DEPTH;
static_assert(MUTEX_OUTPUT_L0C_BASE + OUTPUT_L0C_QUEUE_DEPTH - 1 <= 27, "StateOutput AIC MutexID exceeds 27");

__aicore__ inline void PreloadChunkInputs(
    AscendC::LocalTensor<bfloat16_t>& wL1Local, AscendC::LocalTensor<bfloat16_t>& qPlusL1Local,
    AscendC::LocalTensor<bfloat16_t>& mL1Local, AscendC::LocalTensor<bfloat16_t>& kTailL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL1Local, const AscendC::GlobalTensor<bfloat16_t>& wGlobal,
    const AscendC::GlobalTensor<bfloat16_t>& qPlusGlobal, const AscendC::GlobalTensor<bfloat16_t>& mGlobal,
    const AscendC::GlobalTensor<bfloat16_t>& kTailGlobal, const AscendC::GlobalTensor<bfloat16_t>& aGlobal,
    uint64_t chunkDOffset, uint64_t chunkCOffset, MutexId chunkMatrixMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_MTE2>(chunkMatrixMutexId);
    CopyGmToL1(wL1Local, wGlobal[chunkDOffset], CHUNK_SIZE, HEAD_DIM, HEAD_DIM);
    CopyGmToL1(qPlusL1Local, qPlusGlobal[chunkDOffset], CHUNK_SIZE, HEAD_DIM, HEAD_DIM);
    CopyGmToL1(mL1Local, mGlobal[chunkCOffset], CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);
    CopyGmToL1(kTailL1Local, kTailGlobal[chunkDOffset], CHUNK_SIZE, HEAD_DIM, HEAD_DIM);
    CopyGmToL1(aL1Local, aGlobal[chunkCOffset], CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);
    Mutex::Unlock<PIPE_MTE2>(chunkMatrixMutexId);
}

__aicore__ inline void PreloadValue(
    AscendC::LocalTensor<bfloat16_t>& valueL1Local, const AscendC::GlobalTensor<bfloat16_t>& valueGlobal,
    uint64_t valueOffset, uint32_t validLen, MutexId valueMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_MTE2>(valueMutexId);
    if (validLen < CHUNK_SIZE) {
        Fill(valueL1Local, {1, CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t) / C0_BYTES, 0, static_cast<bfloat16_t>(0)});
    }
    CopyGmToL1(valueL1Local, valueGlobal[valueOffset], validLen, DV_TILE, VALUE_DIM, CHUNK_SIZE);
    Mutex::Unlock<PIPE_MTE2>(valueMutexId);
}

__aicore__ inline void LoadA(
    AscendC::LocalTensor<bfloat16_t>& dstL0ALocal, AscendC::LocalTensor<bfloat16_t>& srcL1Local, uint32_t l1Rows,
    uint32_t m, uint32_t k, bool transpose, MutexId l0aMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_MTE1>(l0aMutexId);
    CopyL1ToL0A(dstL0ALocal, srcL1Local, l1Rows, m, k, transpose);
    Mutex::Unlock<PIPE_MTE1>(l0aMutexId);
}

__aicore__ inline void IssueMmad(
    AscendC::LocalTensor<float>& dstL0CLocal, AscendC::LocalTensor<bfloat16_t>& aL0ALocal,
    AscendC::LocalTensor<bfloat16_t>& bL0BLocal, uint32_t m, uint32_t n, uint32_t k, MutexId l0aMutexId,
    MutexId l0bMutexId, MutexId l0cMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_M>(l0aMutexId);
    Mutex::Lock<PIPE_M>(l0bMutexId);
    Mutex::Lock<PIPE_M>(l0cMutexId);
    CubeMmad<float, bfloat16_t, bfloat16_t>(dstL0CLocal, aL0ALocal, bL0BLocal, m, n, k);
    Mutex::Unlock<PIPE_M>(l0aMutexId);
    Mutex::Unlock<PIPE_M>(l0cMutexId);
    Mutex::Unlock<PIPE_M>(l0bMutexId);
}

// 两次分离的 Mmad 共用同一块 L0B, 期间由 PIPE_M 持有.
// 第二次读取完成前, 后续 MTE1 写入不能覆盖该块数据.
__aicore__ inline void IssueMmadHoldL0B(
    AscendC::LocalTensor<float>& dstL0CLocal, AscendC::LocalTensor<bfloat16_t>& aL0ALocal,
    AscendC::LocalTensor<bfloat16_t>& bL0BLocal, uint32_t m, uint32_t n, uint32_t k, MutexId l0aMutexId,
    MutexId l0bMutexId, MutexId l0cMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_M>(l0aMutexId);
    Mutex::Lock<PIPE_M>(l0bMutexId);
    Mutex::Lock<PIPE_M>(l0cMutexId);
    CubeMmad<float, bfloat16_t, bfloat16_t>(dstL0CLocal, aL0ALocal, bL0BLocal, m, n, k);
    Mutex::Unlock<PIPE_M>(l0aMutexId);
    Mutex::Unlock<PIPE_M>(l0cMutexId);
}

__aicore__ inline void IssueMmadReleaseL0B(
    AscendC::LocalTensor<float>& dstL0CLocal, AscendC::LocalTensor<bfloat16_t>& aL0ALocal,
    AscendC::LocalTensor<bfloat16_t>& bL0BLocal, uint32_t m, uint32_t n, uint32_t k, MutexId l0aMutexId,
    MutexId l0bMutexId, MutexId l0cMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_M>(l0aMutexId);
    Mutex::Lock<PIPE_M>(l0cMutexId);
    CubeMmad<float, bfloat16_t, bfloat16_t>(dstL0CLocal, aL0ALocal, bL0BLocal, m, n, k);
    Mutex::Unlock<PIPE_M>(l0aMutexId);
    Mutex::Unlock<PIPE_M>(l0cMutexId);
    Mutex::Unlock<PIPE_M>(l0bMutexId);
}

// history 与 local 共享一块输出累加槽. history 取得 L0C 槽后一直持有到
// local 原地累加完成, 随后交给 Fixpipe 直接写 GM.
__aicore__ inline void IssueHistoryMmad(
    AscendC::LocalTensor<float>& outputL0CLocal, AscendC::LocalTensor<bfloat16_t>& aL0ALocal,
    AscendC::LocalTensor<bfloat16_t>& stateL0BLocal, MutexId l0aMutexId, MutexId stateL0bMutexId,
    MutexId outputL0cMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_M>(l0aMutexId);
    Mutex::Lock<PIPE_M>(outputL0cMutexId);
    CubeMmad<float, bfloat16_t, bfloat16_t>(outputL0CLocal, aL0ALocal, stateL0BLocal, CHUNK_SIZE, DV_TILE, HEAD_DIM);
    Mutex::Unlock<PIPE_M>(l0aMutexId);
    Mutex::Unlock<PIPE_M>(stateL0bMutexId);
}

__aicore__ inline void IssueLocalAccumulateMmad(
    AscendC::LocalTensor<float>& outputL0CLocal, AscendC::LocalTensor<bfloat16_t>& aL0ALocal,
    AscendC::LocalTensor<bfloat16_t>& rL0BLocal, MutexId l0aMutexId, MutexId rL0bMutexId, MutexId outputL0cMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_M>(l0aMutexId);
    CubeMmad<float, bfloat16_t, bfloat16_t>(
        outputL0CLocal, aL0ALocal, rL0BLocal, CHUNK_SIZE, DV_TILE, CHUNK_SIZE, false);
    Mutex::Unlock<PIPE_M>(l0aMutexId);
    Mutex::Unlock<PIPE_M>(outputL0cMutexId);
    Mutex::Unlock<PIPE_M>(rL0bMutexId);
}

__aicore__ inline void FixOutputToGm(
    const AscendC::GlobalTensor<bfloat16_t>& outputGlobal, AscendC::LocalTensor<float>& outputL0CLocal,
    uint64_t outputOffset, uint32_t validLen, MutexId outputL0cMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_FIX>(outputL0cMutexId);
    FixpipeToGmRows<bfloat16_t, float>(
        outputGlobal[outputOffset], outputL0CLocal, validLen, DV_TILE, CHUNK_SIZE, VALUE_DIM);
    Mutex::Unlock<PIPE_FIX>(outputL0cMutexId);
}

__aicore__ inline void FixToAiv(
    AscendC::LocalTensor<float>& dstUBLocal, AscendC::LocalTensor<float>& srcL0CLocal, uint32_t m, uint32_t n,
    uint16_t handoffFlagId, MutexId l0cMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_FIX>(l0cMutexId);
    WaitAivToAic<PIPE_FIX>(handoffFlagId);
    FixpipeToVecUB(dstUBLocal, srcL0CLocal, m, n);
    SetAicToAiv<PIPE_FIX>(handoffFlagId);
    Mutex::Unlock<PIPE_FIX>(l0cMutexId);
}

__aicore__ inline void FixToAivBegin(
    AscendC::LocalTensor<float>& dstUBLocal, AscendC::LocalTensor<float>& srcL0CLocal, uint32_t m, uint32_t n,
    uint16_t handoffFlagId, MutexId l0cMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_FIX>(l0cMutexId);
    WaitAivToAic<PIPE_FIX>(handoffFlagId);
    FixpipeToVecUB(dstUBLocal, srcL0CLocal, m, n);
    Mutex::Unlock<PIPE_FIX>(l0cMutexId);
}

__aicore__ inline void FixToAivEnd(
    AscendC::LocalTensor<float>& dstUBLocal, AscendC::LocalTensor<float>& srcL0CLocal, uint32_t m, uint32_t n,
    uint16_t handoffFlagId, MutexId l0cMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_FIX>(l0cMutexId);
    FixpipeToVecUB(dstUBLocal, srcL0CLocal, m, n);
    SetAicToAiv<PIPE_FIX>(handoffFlagId);
    Mutex::Unlock<PIPE_FIX>(l0cMutexId);
}

__aicore__ inline void IssueStateInputPreload(
    const AscendC::GlobalTensor<bfloat16_t>& wGlobal, const AscendC::GlobalTensor<bfloat16_t>& qPlusGlobal,
    const AscendC::GlobalTensor<bfloat16_t>& mGlobal, const AscendC::GlobalTensor<bfloat16_t>& kTailGlobal,
    const AscendC::GlobalTensor<bfloat16_t>& aGlobal, const AscendC::GlobalTensor<bfloat16_t>& valueGlobal,
    AscendC::LocalTensor<uint8_t>& chunkMatrixL1Local, AscendC::LocalTensor<bfloat16_t>& valueL1Local,
    const KimiDeltaAttnLiteTilingData& data, uint32_t taskId, uint32_t chunkMatrixSlot, uint32_t chunkId,
    uint32_t itemId, bool preloadChunkMatrices)
{
    using namespace AscendC;
    const uint32_t batchId = taskId / DV_TILE_COUNT;
    const uint32_t dvTileId = taskId % DV_TILE_COUNT;
    const uint32_t valueColumn = dvTileId * DV_TILE;
    const uint32_t firstToken = chunkId * CHUNK_SIZE;
    const uint32_t validLen = data.seqLen - firstToken < CHUNK_SIZE ? data.seqLen - firstToken : CHUNK_SIZE;
    const uint64_t chunkIndex = static_cast<uint64_t>(batchId) * data.chunkCount + chunkId;
    const uint64_t valueOffset = (static_cast<uint64_t>(batchId) * data.seqLen + firstToken) * VALUE_DIM + valueColumn;
    const uint32_t valueSlot = itemId % DB_SLOT_NUM;
    const MutexId chunkMatrixMutexId = MUTEX_CHUNK_MATRIX_L1_BASE + static_cast<MutexId>(chunkMatrixSlot);
    const MutexId valueMutexId = MUTEX_VALUE_L1_BASE + static_cast<MutexId>(valueSlot);

    auto chunkMatrixSlotL1Local = chunkMatrixL1Local[chunkMatrixSlot * CHUNK_MATRIX_SLOT_BYTES];
    auto wL1Local = chunkMatrixSlotL1Local[CHUNK_MATRIX_W_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto qPlusL1Local = chunkMatrixSlotL1Local[CHUNK_MATRIX_Q_PLUS_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto mL1Local = chunkMatrixSlotL1Local[CHUNK_MATRIX_M_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto kTailL1Local = chunkMatrixSlotL1Local[CHUNK_MATRIX_K_TAIL_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto aL1Local = chunkMatrixSlotL1Local[CHUNK_MATRIX_A_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto valueSlotL1Local = valueL1Local[valueSlot * CHUNK_DV_TILE_ELEMS];
    if (preloadChunkMatrices) {
        PreloadChunkInputs(
            wL1Local, qPlusL1Local, mL1Local, kTailL1Local, aL1Local, wGlobal, qPlusGlobal, mGlobal, kTailGlobal,
            aGlobal, chunkIndex * CHUNK_D_ELEMS, chunkIndex * CHUNK_C_ELEMS, chunkMatrixMutexId);
    }
    PreloadValue(valueSlotL1Local, valueGlobal, valueOffset, validLen, valueMutexId);
}

__aicore__ inline void IssueStateU(
    AscendC::LocalTensor<uint8_t>& chunkMatrixL1Local, AscendC::LocalTensor<bfloat16_t>& valueL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL0ALocal, AscendC::LocalTensor<bfloat16_t>& valueL0BLocal,
    AscendC::LocalTensor<float>& mmadL0CQueueLocal, AscendC::LocalTensor<float>& uUBLocal, uint32_t chunkMatrixSlot,
    uint32_t itemId, bool acquireChunkMatrices, uint32_t& l0aOpIdx, uint32_t& l0cOpIdx)
{
    using namespace AscendC;
    const uint32_t valueSlot = itemId % DB_SLOT_NUM;
    const uint32_t uPredSlot = itemId % DB_SLOT_NUM;
    const MutexId chunkMatrixMutexId = MUTEX_CHUNK_MATRIX_L1_BASE + static_cast<MutexId>(chunkMatrixSlot);
    const MutexId valueMutexId = MUTEX_VALUE_L1_BASE + static_cast<MutexId>(valueSlot);
    auto chunkMatrixSlotL1Local = chunkMatrixL1Local[chunkMatrixSlot * CHUNK_MATRIX_SLOT_BYTES];
    auto mL1Local = chunkMatrixSlotL1Local[CHUNK_MATRIX_M_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto valueSlotL1Local = valueL1Local[valueSlot * CHUNK_DV_TILE_ELEMS];
    auto uSlotUBLocal = uUBLocal[uPredSlot * CHUNK_DV_HALF_ELEMS];

    // 同一 chunk 的首条 lane 取得Chunk 只读矩阵槽, 最后一条 lane 在 C2 读完 A 后归还.
    if (acquireChunkMatrices) {
        Mutex::Lock<PIPE_MTE1>(chunkMatrixMutexId);
    }

    Mutex::Lock<PIPE_MTE1>(valueMutexId);
    Mutex::Lock<PIPE_MTE1>(MUTEX_VALUE_L0B);
    CopyL1ToL0B(valueL0BLocal, valueSlotL1Local, CHUNK_SIZE, CHUNK_SIZE, DV_TILE, true);
    Mutex::Unlock<PIPE_MTE1>(valueMutexId);
    Mutex::Unlock<PIPE_MTE1>(MUTEX_VALUE_L0B);

    const uint32_t uAIdx = l0aOpIdx++ % DB_SLOT_NUM;
    const uint32_t uCIdx = l0cOpIdx++ % L0C_QUEUE_DEPTH;
    auto uAL0Local = aL0ALocal[uAIdx * L0A_SLOT_ELEMS];
    auto uCL0Local = mmadL0CQueueLocal[uCIdx * L0C_SLOT_ELEMS];
    LoadA(uAL0Local, mL1Local, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, false, MUTEX_L0A_BASE + uAIdx);
    IssueMmad(
        uCL0Local, uAL0Local, valueL0BLocal, CHUNK_SIZE, DV_TILE, CHUNK_SIZE, MUTEX_L0A_BASE + uAIdx, MUTEX_VALUE_L0B,
        MUTEX_L0C_BASE + uCIdx);
    FixToAivBegin(
        uSlotUBLocal, uCL0Local, CHUNK_SIZE, DV_TILE, SlotFlagId(FLAG_U_PRED_HANDOFF_BASE, uPredSlot),
        MUTEX_L0C_BASE + uCIdx);
}

__aicore__ inline void IssueStateC1Post(
    AscendC::LocalTensor<uint8_t>& chunkMatrixL1Local, AscendC::LocalTensor<bfloat16_t>& stateL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL0ALocal, AscendC::LocalTensor<bfloat16_t>& stateL0BLocal,
    AscendC::LocalTensor<float>& mmadL0CQueueLocal, AscendC::LocalTensor<float>& predUBLocal, uint32_t lane,
    uint32_t chunkMatrixSlot, uint32_t itemId, uint32_t& l0aOpIdx, uint32_t& l0cOpIdx)
{
    using namespace AscendC;
    const uint32_t uPredSlot = itemId % DB_SLOT_NUM;
    const MutexId stateMutexId = MUTEX_STATE_L0B_BASE + static_cast<MutexId>(lane);
    auto chunkMatrixSlotL1Local = chunkMatrixL1Local[chunkMatrixSlot * CHUNK_MATRIX_SLOT_BYTES];
    auto wL1Local = chunkMatrixSlotL1Local[CHUNK_MATRIX_W_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto stateSlotL1Local = stateL1Local[lane * STATE_TILE_ELEMS];
    auto stateSlotL0BLocal = stateL0BLocal[lane * STATE_TILE_ELEMS];
    auto predSlotUBLocal = predUBLocal[uPredSlot * CHUNK_DV_HALF_ELEMS];

    WaitAivToAic<PIPE_MTE1>(SlotFlagId(FLAG_STATE_HANDOFF_BASE, lane));
    Mutex::Lock<PIPE_MTE1>(stateMutexId);
    CopyL1ToL0B(stateSlotL0BLocal, stateSlotL1Local, HEAD_DIM, HEAD_DIM, DV_TILE, true);
    SetAicToAiv<PIPE_MTE1>(SlotFlagId(FLAG_STATE_HANDOFF_BASE, lane));
    Mutex::Unlock<PIPE_MTE1>(stateMutexId);

    const uint32_t predAIdx = l0aOpIdx++ % DB_SLOT_NUM;
    const uint32_t predCIdx = l0cOpIdx++ % L0C_QUEUE_DEPTH;
    auto predAL0Local = aL0ALocal[predAIdx * L0A_SLOT_ELEMS];
    auto predCL0Local = mmadL0CQueueLocal[predCIdx * L0C_SLOT_ELEMS];
    LoadA(predAL0Local, wL1Local, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, false, MUTEX_L0A_BASE + predAIdx);
    IssueMmadHoldL0B(
        predCL0Local, predAL0Local, stateSlotL0BLocal, CHUNK_SIZE, DV_TILE, HEAD_DIM, MUTEX_L0A_BASE + predAIdx,
        stateMutexId, MUTEX_L0C_BASE + predCIdx);
    FixToAivEnd(
        predSlotUBLocal, predCL0Local, CHUNK_SIZE, DV_TILE, SlotFlagId(FLAG_U_PRED_HANDOFF_BASE, uPredSlot),
        MUTEX_L0C_BASE + predCIdx);
}

__aicore__ inline void IssueStateC2History(
    AscendC::LocalTensor<uint8_t>& chunkMatrixL1Local, AscendC::LocalTensor<bfloat16_t>& aL0ALocal,
    AscendC::LocalTensor<bfloat16_t>& stateL0BLocal, AscendC::LocalTensor<float>& outputL0CLocal, uint32_t lane,
    uint32_t chunkMatrixSlot, uint32_t itemId, uint32_t& l0aOpIdx)
{
    using namespace AscendC;
    const uint32_t outputSlot = itemId % OUTPUT_L0C_QUEUE_DEPTH;
    const MutexId stateMutexId = MUTEX_STATE_L0B_BASE + static_cast<MutexId>(lane);
    const MutexId outputMutexId = MUTEX_OUTPUT_L0C_BASE + static_cast<MutexId>(outputSlot);
    auto chunkMatrixSlotL1Local = chunkMatrixL1Local[chunkMatrixSlot * CHUNK_MATRIX_SLOT_BYTES];
    auto qPlusL1Local = chunkMatrixSlotL1Local[CHUNK_MATRIX_Q_PLUS_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto stateSlotL0BLocal = stateL0BLocal[lane * STATE_TILE_ELEMS];
    auto outputSlotL0CLocal = outputL0CLocal[outputSlot * OUTPUT_L0C_SLOT_ELEMS];

    const uint32_t historyAIdx = l0aOpIdx++ % DB_SLOT_NUM;
    auto historyAL0Local = aL0ALocal[historyAIdx * L0A_SLOT_ELEMS];
    LoadA(historyAL0Local, qPlusL1Local, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, false, MUTEX_L0A_BASE + historyAIdx);
    IssueHistoryMmad(
        outputSlotL0CLocal, historyAL0Local, stateSlotL0BLocal, MUTEX_L0A_BASE + historyAIdx, stateMutexId,
        outputMutexId);
}

__aicore__ inline void IssueStateC2Remainder(
    AscendC::LocalTensor<uint8_t>& chunkMatrixL1Local, AscendC::LocalTensor<bfloat16_t>& rL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL0ALocal, AscendC::LocalTensor<bfloat16_t>& rL0BLocal,
    AscendC::LocalTensor<float>& mmadL0CQueueLocal, AscendC::LocalTensor<float>& deltaUBLocal,
    AscendC::LocalTensor<float>& outputL0CLocal, uint32_t chunkMatrixSlot, uint32_t itemId, uint32_t rQueueDepth,
    bool releaseChunkMatrices, uint32_t& l0aOpIdx, uint32_t& l0cOpIdx)
{
    using namespace AscendC;
    const uint32_t rSlot = itemId % rQueueDepth;
    const uint32_t deltaSlot = itemId % DB_SLOT_NUM;
    const uint32_t outputSlot = itemId % OUTPUT_L0C_QUEUE_DEPTH;
    const MutexId chunkMatrixMutexId = MUTEX_CHUNK_MATRIX_L1_BASE + static_cast<MutexId>(chunkMatrixSlot);
    const MutexId outputMutexId = MUTEX_OUTPUT_L0C_BASE + static_cast<MutexId>(outputSlot);
    const uint16_t deltaFlagId = SlotFlagId(FLAG_DELTA_HANDOFF_BASE, deltaSlot);
    auto chunkMatrixSlotL1Local = chunkMatrixL1Local[chunkMatrixSlot * CHUNK_MATRIX_SLOT_BYTES];
    auto kTailL1Local = chunkMatrixSlotL1Local[CHUNK_MATRIX_K_TAIL_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto aL1Local = chunkMatrixSlotL1Local[CHUNK_MATRIX_A_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto rSlotL1Local = rL1Local[rSlot * CHUNK_DV_TILE_ELEMS];
    auto deltaSlotUBLocal = deltaUBLocal[deltaSlot * STATE_HALF_ELEMS];
    auto outputSlotL0CLocal = outputL0CLocal[outputSlot * OUTPUT_L0C_SLOT_ELEMS];

    WaitAivToAic<PIPE_MTE1>(SlotFlagId(FLAG_R_HANDOFF_BASE, rSlot));
    Mutex::Lock<PIPE_MTE1>(MUTEX_R_L0B);
    CopyL1ToL0B(rL0BLocal, rSlotL1Local, CHUNK_SIZE, CHUNK_SIZE, DV_TILE, true);
    SetAicToAiv<PIPE_MTE1>(SlotFlagId(FLAG_R_HANDOFF_BASE, rSlot));
    Mutex::Unlock<PIPE_MTE1>(MUTEX_R_L0B);

    const uint32_t deltaAIdx = l0aOpIdx++ % DB_SLOT_NUM;
    const uint32_t deltaCIdx = l0cOpIdx++ % L0C_QUEUE_DEPTH;
    auto deltaAL0Local = aL0ALocal[deltaAIdx * L0A_SLOT_ELEMS];
    auto deltaCL0Local = mmadL0CQueueLocal[deltaCIdx * L0C_SLOT_ELEMS];
    LoadA(deltaAL0Local, kTailL1Local, CHUNK_SIZE, HEAD_DIM, CHUNK_SIZE, true, MUTEX_L0A_BASE + deltaAIdx);
    IssueMmadHoldL0B(
        deltaCL0Local, deltaAL0Local, rL0BLocal, HEAD_DIM, DV_TILE, CHUNK_SIZE, MUTEX_L0A_BASE + deltaAIdx, MUTEX_R_L0B,
        MUTEX_L0C_BASE + deltaCIdx);

    const uint32_t localAIdx = l0aOpIdx++ % DB_SLOT_NUM;
    auto localAL0Local = aL0ALocal[localAIdx * L0A_SLOT_ELEMS];
    LoadA(localAL0Local, aL1Local, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, false, MUTEX_L0A_BASE + localAIdx);
    if (releaseChunkMatrices) {
        Mutex::Unlock<PIPE_MTE1>(chunkMatrixMutexId);
    }
    IssueLocalAccumulateMmad(
        outputSlotL0CLocal, localAL0Local, rL0BLocal, MUTEX_L0A_BASE + localAIdx, MUTEX_R_L0B, outputMutexId);

    FixToAiv(deltaSlotUBLocal, deltaCL0Local, HEAD_DIM, DV_TILE, deltaFlagId, MUTEX_L0C_BASE + deltaCIdx);
}

__aicore__ inline void IssueStateOutput(
    const AscendC::GlobalTensor<bfloat16_t>& outputGlobal, AscendC::LocalTensor<float>& outputL0CLocal,
    const KimiDeltaAttnLiteTilingData& data, uint32_t taskId, uint32_t chunkId, uint32_t itemId)
{
    const uint32_t outputSlot = itemId % OUTPUT_L0C_QUEUE_DEPTH;
    const MutexId outputMutexId = MUTEX_OUTPUT_L0C_BASE + static_cast<MutexId>(outputSlot);
    auto outputSlotL0CLocal = outputL0CLocal[outputSlot * OUTPUT_L0C_SLOT_ELEMS];
    const uint32_t batchId = taskId / DV_TILE_COUNT;
    const uint32_t dvTileId = taskId % DV_TILE_COUNT;
    const uint32_t firstToken = chunkId * CHUNK_SIZE;
    const uint32_t validLen = data.seqLen - firstToken < CHUNK_SIZE ? data.seqLen - firstToken : CHUNK_SIZE;
    const uint64_t outputOffset =
        (static_cast<uint64_t>(batchId) * data.seqLen + firstToken) * VALUE_DIM + dvTileId * DV_TILE;
    FixOutputToGm(outputGlobal, outputSlotL0CLocal, outputOffset, validLen, outputMutexId);
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

        LocalTensor<uint8_t> chunkMatrixL1Local(
            TPosition::A1, 0, CHUNK_MATRIX_SLOT_BYTES * STATE_PIPELINE_MAX_LANE_NUM);
        LocalTensor<bfloat16_t> stateL1Local(
            TPosition::A1, STATE_L1_ADDR, STATE_TILE_ELEMS * STATE_PIPELINE_MAX_LANE_NUM);
        LocalTensor<bfloat16_t> valueL1Local(TPosition::A1, VALUE_L1_ADDR, CHUNK_DV_TILE_ELEMS * DB_SLOT_NUM);
        LocalTensor<bfloat16_t> rL1Local(TPosition::A1, R_L1_ADDR, CHUNK_DV_TILE_ELEMS * STATE_R_MAX_QUEUE_DEPTH);
        LocalTensor<bfloat16_t> aL0ALocal(TPosition::A2, 0, L0A_SLOT_ELEMS * DB_SLOT_NUM);
        LocalTensor<bfloat16_t> stateL0BLocal(
            TPosition::B2, STATE_L0B_ADDR, STATE_TILE_ELEMS * STATE_PIPELINE_MAX_LANE_NUM);
        LocalTensor<bfloat16_t> valueL0BLocal(TPosition::B2, VALUE_L0B_ADDR, CHUNK_DV_TILE_ELEMS);
        LocalTensor<bfloat16_t> rL0BLocal(TPosition::B2, R_L0B_ADDR, CHUNK_DV_TILE_ELEMS);
        // 通用 L0C 队列承载 U、prediction 和 delta; outputL0C 单独承载 history+local.
        LocalTensor<float> mmadL0CQueueLocal(TPosition::CO1, 0, L0C_SLOT_ELEMS * L0C_QUEUE_DEPTH);
        LocalTensor<float> outputL0CLocal(
            TPosition::CO1, OUTPUT_L0C_ADDR, OUTPUT_L0C_SLOT_ELEMS * OUTPUT_L0C_QUEUE_DEPTH);
        LocalTensor<float> predUBLocal(TPosition::VECCALC, PRED_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> deltaUBLocal(TPosition::VECCALC, DELTA_UB_ADDR, STATE_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> uUBLocal(TPosition::VECCALC, U_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);

        const uint32_t aicIdx = GetBlockIdx();
        const uint32_t configuredLaneCount = data.statePipelineLaneCount;
        const uint64_t waveStride = static_cast<uint64_t>(data.stateUseAicNum) * configuredLaneCount;
        uint32_t l0aOpIdx = 0;
        uint32_t l0cOpIdx = 0;
        for (uint64_t waveTaskBase = static_cast<uint64_t>(aicIdx) * configuredLaneCount;
             waveTaskBase < data.stateNumTasks; waveTaskBase += waveStride) {
            const uint32_t remainingTasks = data.stateNumTasks - static_cast<uint32_t>(waveTaskBase);
            const uint32_t laneCount = remainingTasks < configuredLaneCount ? remainingTasks : configuredLaneCount;
            const uint32_t rQueueDepth = laneCount > 1 ? laneCount - 1 : 1;
            for (uint32_t lane = 0; lane < STATE_PIPELINE_MAX_LANE_NUM; ++lane) {
                SetAicToAiv<PIPE_MTE1>(SlotFlagId(FLAG_STATE_HANDOFF_BASE, lane));
            }
            for (uint32_t slot = 0; slot < STATE_R_MAX_QUEUE_DEPTH; ++slot) {
                SetAicToAiv<PIPE_MTE1>(SlotFlagId(FLAG_R_HANDOFF_BASE, slot));
            }

            const uint32_t itemCount = data.chunkCount * laneCount;
            if (laneCount == 1) {
                const uint32_t taskId = static_cast<uint32_t>(waveTaskBase);
                IssueStateInputPreload(
                    wGlobal, qPlusGlobal, mGlobal, kTailGlobal, aGlobal, valueGlobal, chunkMatrixL1Local, valueL1Local,
                    data, taskId, 0, 0, 0, true);
                IssueStateU(
                    chunkMatrixL1Local, valueL1Local, aL0ALocal, valueL0BLocal, mmadL0CQueueLocal, uUBLocal, 0, 0, true,
                    l0aOpIdx, l0cOpIdx);

                for (uint32_t itemId = 0; itemId < itemCount; ++itemId) {
                    const uint32_t chunkMatrixSlot = itemId % DB_SLOT_NUM;
                    const uint32_t nextItemId = itemId + 1;
                    if (nextItemId < itemCount) {
                        const uint32_t nextChunkMatrixSlot = nextItemId % DB_SLOT_NUM;
                        IssueStateInputPreload(
                            wGlobal, qPlusGlobal, mGlobal, kTailGlobal, aGlobal, valueGlobal, chunkMatrixL1Local,
                            valueL1Local, data, taskId, nextChunkMatrixSlot, nextItemId, nextItemId, true);
                    }

                    IssueStateC1Post(
                        chunkMatrixL1Local, stateL1Local, aL0ALocal, stateL0BLocal, mmadL0CQueueLocal, predUBLocal, 0,
                        chunkMatrixSlot, itemId, l0aOpIdx, l0cOpIdx);
                    IssueStateC2History(
                        chunkMatrixL1Local, aL0ALocal, stateL0BLocal, outputL0CLocal, 0, chunkMatrixSlot, itemId,
                        l0aOpIdx);

                    if (nextItemId < itemCount) {
                        const uint32_t nextChunkMatrixSlot = nextItemId % DB_SLOT_NUM;
                        IssueStateU(
                            chunkMatrixL1Local, valueL1Local, aL0ALocal, valueL0BLocal, mmadL0CQueueLocal, uUBLocal,
                            nextChunkMatrixSlot, nextItemId, true, l0aOpIdx, l0cOpIdx);
                    }

                    IssueStateC2Remainder(
                        chunkMatrixL1Local, rL1Local, aL0ALocal, rL0BLocal, mmadL0CQueueLocal, deltaUBLocal,
                        outputL0CLocal, chunkMatrixSlot, itemId, rQueueDepth, true, l0aOpIdx, l0cOpIdx);
                    IssueStateOutput(outputGlobal, outputL0CLocal, data, taskId, itemId, itemId);
                }
            } else {
                // 同一 chunk 的 lane 对应同一 batch 的不同 Dv tile, 共用一份Chunk 只读矩阵.
                // Chunk 只读矩阵按 chunk 双槽滚动, Value 仍按 item 双槽分别搬入.
                const uint32_t epochCount = itemCount + laneCount;
                for (uint32_t epoch = 0; epoch < epochCount; ++epoch) {
                    const bool hasC1 = epoch < itemCount;
                    const bool hasC2 = epoch + 1 >= laneCount && epoch < itemCount + laneCount - 1;
                    if (hasC1) {
                        const uint32_t itemId = epoch;
                        const uint32_t lane = itemId % laneCount;
                        const uint32_t chunkId = itemId / laneCount;
                        const uint32_t taskId = static_cast<uint32_t>(waveTaskBase) + lane;
                        const uint32_t chunkMatrixSlot = chunkId % DB_SLOT_NUM;
                        IssueStateInputPreload(
                            wGlobal, qPlusGlobal, mGlobal, kTailGlobal, aGlobal, valueGlobal, chunkMatrixL1Local,
                            valueL1Local, data, taskId, chunkMatrixSlot, chunkId, itemId, lane == 0);
                        IssueStateU(
                            chunkMatrixL1Local, valueL1Local, aL0ALocal, valueL0BLocal, mmadL0CQueueLocal, uUBLocal,
                            chunkMatrixSlot, itemId, lane == 0, l0aOpIdx, l0cOpIdx);
                    }
                    if (hasC2) {
                        const uint32_t itemId = epoch - laneCount + 1;
                        const uint32_t lane = itemId % laneCount;
                        const uint32_t chunkId = itemId / laneCount;
                        const uint32_t chunkMatrixSlot = chunkId % DB_SLOT_NUM;
                        IssueStateC2History(
                            chunkMatrixL1Local, aL0ALocal, stateL0BLocal, outputL0CLocal, lane, chunkMatrixSlot, itemId,
                            l0aOpIdx);
                        IssueStateC2Remainder(
                            chunkMatrixL1Local, rL1Local, aL0ALocal, rL0BLocal, mmadL0CQueueLocal, deltaUBLocal,
                            outputL0CLocal, chunkMatrixSlot, itemId, rQueueDepth, lane + 1 == laneCount, l0aOpIdx,
                            l0cOpIdx);
                    }
                    if (hasC1) {
                        const uint32_t itemId = epoch;
                        const uint32_t lane = itemId % laneCount;
                        const uint32_t chunkId = itemId / laneCount;
                        const uint32_t chunkMatrixSlot = chunkId % DB_SLOT_NUM;
                        IssueStateC1Post(
                            chunkMatrixL1Local, stateL1Local, aL0ALocal, stateL0BLocal, mmadL0CQueueLocal, predUBLocal,
                            lane, chunkMatrixSlot, itemId, l0aOpIdx, l0cOpIdx);
                    }
                    if (hasC2) {
                        const uint32_t itemId = epoch - laneCount + 1;
                        const uint32_t lane = itemId % laneCount;
                        const uint32_t chunkId = itemId / laneCount;
                        IssueStateOutput(
                            outputGlobal, outputL0CLocal, data, static_cast<uint32_t>(waveTaskBase) + lane, chunkId,
                            itemId);
                    }
                }
            }

            for (uint32_t slot = 0; slot < DB_SLOT_NUM; ++slot) {
                WaitAivToAic<PIPE_FIX>(SlotFlagId(FLAG_U_PRED_HANDOFF_BASE, slot));
                WaitAivToAic<PIPE_FIX>(SlotFlagId(FLAG_DELTA_HANDOFF_BASE, slot));
            }
            // state lane 0 的常规交接已排空. 同一 FlagID 的额外一次 Wait 表示整个
            // wave 已结束; 收到该信号后, 下一个 wave 才会重新发布 state 空槽.
            WaitAivToAic<PIPE_MTE1>(FLAG_STATE_WAVE_DONE);
        }
    }
}

} // namespace KDALite
