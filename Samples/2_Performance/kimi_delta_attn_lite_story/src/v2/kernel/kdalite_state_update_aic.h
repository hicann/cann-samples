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

// staticL1Local 中的只读矩阵按 chunkId%2 使用双槽. 同一 chunk 的 lane 0 取得槽所有权,
// 最后一条 lane 在 C2 读完 A 后归还. 地址规划保留 4 个物理槽, 仅使用前 2 个.
// Value 和 KPlusState 只跨一个局部阶段, 使用双槽.
constexpr uint32_t STATIC_K_PLUS_SLOT_ADDR = 0;
constexpr uint32_t STATIC_Q_PLUS_SLOT_ADDR = STATIC_K_PLUS_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t STATIC_M_SLOT_ADDR = STATIC_Q_PLUS_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t STATIC_K_TAIL_SLOT_ADDR = STATIC_M_SLOT_ADDR + CHUNK_C_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t STATIC_A_SLOT_ADDR = STATIC_K_TAIL_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t STATIC_SLOT_BYTES = STATIC_A_SLOT_ADDR + CHUNK_C_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t STATE_L1_ADDR = STATIC_SLOT_BYTES * STATE_PIPELINE_MAX_LANE_NUM;
constexpr uint32_t VALUE_L1_ADDR = STATE_L1_ADDR + STATE_TILE_ELEMS * sizeof(bfloat16_t) * STATE_PIPELINE_MAX_LANE_NUM;
constexpr uint32_t R_L1_ADDR = VALUE_L1_ADDR + CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t) * DB_SLOT_NUM;
constexpr uint32_t K_PLUS_STATE_L1_ADDR =
    R_L1_ADDR + CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t) * STATE_R_MAX_QUEUE_DEPTH;
constexpr uint32_t AIC_L1_END_ADDR = K_PLUS_STATE_L1_ADDR + CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t) * DB_SLOT_NUM;

constexpr uint32_t L0A_SLOT_ELEMS = CHUNK_D_ELEMS;
constexpr uint32_t STATE_L0B_ADDR = 0;
constexpr uint32_t VALUE_L0B_ADDR =
    STATE_L0B_ADDR + STATE_TILE_ELEMS * sizeof(bfloat16_t) * STATE_PIPELINE_MAX_LANE_NUM;
constexpr uint32_t R_L0B_ADDR = VALUE_L0B_ADDR + CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t K_PLUS_STATE_L0B_ADDR = R_L0B_ADDR + CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t L0C_SLOT_ELEMS = STATE_TILE_ELEMS;

static_assert(AIC_L1_END_ADDR <= 512 * 1024, "FusedRecurrentOutput L1 allocation exceeds 512 KiB");
static_assert(L0A_SLOT_ELEMS * sizeof(bfloat16_t) * DB_SLOT_NUM <= 64 * 1024, "L0A allocation exceeds 64 KiB");
static_assert(
    K_PLUS_STATE_L0B_ADDR + CHUNK_DV_TILE_ELEMS * sizeof(bfloat16_t) <= 64 * 1024, "L0B allocation exceeds 64 KiB");
static_assert(L0C_SLOT_ELEMS * sizeof(float) * L0C_QUEUE_DEPTH <= 256 * 1024, "L0C allocation exceeds 256 KiB");

constexpr MutexId MUTEX_STATIC_L1_BASE = 0;
constexpr MutexId MUTEX_VALUE_L1_BASE = MUTEX_STATIC_L1_BASE + STATE_PIPELINE_MAX_LANE_NUM;
constexpr MutexId MUTEX_L0A_BASE = MUTEX_VALUE_L1_BASE + DB_SLOT_NUM;
constexpr MutexId MUTEX_STATE_L0B_BASE = MUTEX_L0A_BASE + DB_SLOT_NUM;
constexpr MutexId MUTEX_VALUE_L0B = MUTEX_STATE_L0B_BASE + STATE_PIPELINE_MAX_LANE_NUM;
constexpr MutexId MUTEX_R_L0B = MUTEX_VALUE_L0B + 1;
constexpr MutexId MUTEX_L0C_BASE = MUTEX_R_L0B + 1;
constexpr MutexId MUTEX_K_PLUS_STATE_L1_BASE = MUTEX_L0C_BASE + L0C_QUEUE_DEPTH;
constexpr MutexId MUTEX_K_PLUS_STATE_L0B = MUTEX_K_PLUS_STATE_L1_BASE + DB_SLOT_NUM;
static_assert(MUTEX_K_PLUS_STATE_L0B <= 27, "StateOutput AIC MutexID exceeds 27");

__aicore__ inline void PreloadStaticInputs(
    AscendC::LocalTensor<bfloat16_t>& kPlusL1Local, AscendC::LocalTensor<bfloat16_t>& qPlusL1Local,
    AscendC::LocalTensor<bfloat16_t>& mL1Local, AscendC::LocalTensor<bfloat16_t>& kTailL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL1Local, const AscendC::GlobalTensor<bfloat16_t>& kPlusGlobal,
    const AscendC::GlobalTensor<bfloat16_t>& qPlusGlobal, const AscendC::GlobalTensor<bfloat16_t>& mGlobal,
    const AscendC::GlobalTensor<bfloat16_t>& kTailGlobal, const AscendC::GlobalTensor<bfloat16_t>& aGlobal,
    uint64_t chunkDOffset, uint64_t chunkCOffset, MutexId staticMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_MTE2>(staticMutexId);
    CopyGmToL1(kPlusL1Local, kPlusGlobal[chunkDOffset], CHUNK_SIZE, HEAD_DIM, HEAD_DIM);
    CopyGmToL1(qPlusL1Local, qPlusGlobal[chunkDOffset], CHUNK_SIZE, HEAD_DIM, HEAD_DIM);
    CopyGmToL1(mL1Local, mGlobal[chunkCOffset], CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);
    CopyGmToL1(kTailL1Local, kTailGlobal[chunkDOffset], CHUNK_SIZE, HEAD_DIM, HEAD_DIM);
    CopyGmToL1(aL1Local, aGlobal[chunkCOffset], CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);
    Mutex::Unlock<PIPE_MTE2>(staticMutexId);
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

__aicore__ inline void FixKPlusStateToL1(
    AscendC::LocalTensor<bfloat16_t>& dstL1Local, AscendC::LocalTensor<float>& srcL0CLocal, MutexId l1MutexId,
    MutexId l0cMutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_FIX>(l0cMutexId);
    Mutex::Lock<PIPE_FIX>(l1MutexId);
    FixpipeParamsArch3510<CO2Layout::NZ> params(
        DV_TILE, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE * static_cast<uint32_t>(CUBE_BLOCK));
    params.quantPre = QuantMode_t::F322BF16;
    Fixpipe<bfloat16_t, float, CFG_NZ>(dstL1Local, srcL0CLocal, params);
    Mutex::Unlock<PIPE_FIX>(l0cMutexId);
    Mutex::Unlock<PIPE_FIX>(l1MutexId);
}

__aicore__ inline void LoadKPlusState(
    AscendC::LocalTensor<bfloat16_t>& dstL0BLocal, AscendC::LocalTensor<bfloat16_t>& srcL1Local, MutexId l1MutexId)
{
    using namespace AscendC;
    Mutex::Lock<PIPE_MTE1>(l1MutexId);
    Mutex::Lock<PIPE_MTE1>(MUTEX_K_PLUS_STATE_L0B);
    CopyL1ToL0B(dstL0BLocal, srcL1Local, CHUNK_SIZE, CHUNK_SIZE, DV_TILE, true);
    Mutex::Unlock<PIPE_MTE1>(l1MutexId);
    Mutex::Unlock<PIPE_MTE1>(MUTEX_K_PLUS_STATE_L0B);
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
    const AscendC::GlobalTensor<bfloat16_t>& kPlusGlobal, const AscendC::GlobalTensor<bfloat16_t>& qPlusGlobal,
    const AscendC::GlobalTensor<bfloat16_t>& mGlobal, const AscendC::GlobalTensor<bfloat16_t>& kTailGlobal,
    const AscendC::GlobalTensor<bfloat16_t>& aGlobal, const AscendC::GlobalTensor<bfloat16_t>& valueGlobal,
    AscendC::LocalTensor<uint8_t>& staticL1Local, AscendC::LocalTensor<bfloat16_t>& valueL1Local,
    const KimiDeltaAttnLiteTilingData& data, uint32_t taskId, uint32_t staticSlot, uint32_t chunkId, uint32_t itemId,
    bool preloadStatic)
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
    const MutexId staticMutexId = MUTEX_STATIC_L1_BASE + static_cast<MutexId>(staticSlot);
    const MutexId valueMutexId = MUTEX_VALUE_L1_BASE + static_cast<MutexId>(valueSlot);

    auto staticSlotL1Local = staticL1Local[staticSlot * STATIC_SLOT_BYTES];
    auto kPlusL1Local = staticSlotL1Local[STATIC_K_PLUS_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto qPlusL1Local = staticSlotL1Local[STATIC_Q_PLUS_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto mL1Local = staticSlotL1Local[STATIC_M_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto kTailL1Local = staticSlotL1Local[STATIC_K_TAIL_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto aL1Local = staticSlotL1Local[STATIC_A_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto valueSlotL1Local = valueL1Local[valueSlot * CHUNK_DV_TILE_ELEMS];
    if (preloadStatic) {
        PreloadStaticInputs(
            kPlusL1Local, qPlusL1Local, mL1Local, kTailL1Local, aL1Local, kPlusGlobal, qPlusGlobal, mGlobal,
            kTailGlobal, aGlobal, chunkIndex * CHUNK_D_ELEMS, chunkIndex * CHUNK_C_ELEMS, staticMutexId);
    }
    PreloadValue(valueSlotL1Local, valueGlobal, valueOffset, validLen, valueMutexId);
}

__aicore__ inline void IssueStateU(
    AscendC::LocalTensor<uint8_t>& staticL1Local, AscendC::LocalTensor<bfloat16_t>& valueL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL0ALocal, AscendC::LocalTensor<bfloat16_t>& valueL0BLocal,
    AscendC::LocalTensor<float>& resultL0CLocal, AscendC::LocalTensor<float>& uUBLocal, uint32_t staticSlot,
    uint32_t itemId, bool acquireStatic, uint32_t& l0aOpIdx, uint32_t& l0cOpIdx)
{
    using namespace AscendC;
    const uint32_t valueSlot = itemId % DB_SLOT_NUM;
    const uint32_t uPredSlot = itemId % DB_SLOT_NUM;
    const MutexId staticMutexId = MUTEX_STATIC_L1_BASE + static_cast<MutexId>(staticSlot);
    const MutexId valueMutexId = MUTEX_VALUE_L1_BASE + static_cast<MutexId>(valueSlot);
    auto staticSlotL1Local = staticL1Local[staticSlot * STATIC_SLOT_BYTES];
    auto mL1Local = staticSlotL1Local[STATIC_M_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto valueSlotL1Local = valueL1Local[valueSlot * CHUNK_DV_TILE_ELEMS];
    auto uSlotUBLocal = uUBLocal[uPredSlot * CHUNK_DV_HALF_ELEMS];

    // 同一 chunk 的首条 lane 取得静态矩阵槽, 最后一条 lane 在 C2 读完 A 后归还.
    if (acquireStatic) {
        Mutex::Lock<PIPE_MTE1>(staticMutexId);
    }

    Mutex::Lock<PIPE_MTE1>(valueMutexId);
    Mutex::Lock<PIPE_MTE1>(MUTEX_VALUE_L0B);
    CopyL1ToL0B(valueL0BLocal, valueSlotL1Local, CHUNK_SIZE, CHUNK_SIZE, DV_TILE, true);
    Mutex::Unlock<PIPE_MTE1>(valueMutexId);
    Mutex::Unlock<PIPE_MTE1>(MUTEX_VALUE_L0B);

    const uint32_t uAIdx = l0aOpIdx++ % DB_SLOT_NUM;
    const uint32_t uCIdx = l0cOpIdx++ % L0C_QUEUE_DEPTH;
    auto uAL0Local = aL0ALocal[uAIdx * L0A_SLOT_ELEMS];
    auto uCL0Local = resultL0CLocal[uCIdx * L0C_SLOT_ELEMS];
    LoadA(uAL0Local, mL1Local, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, false, MUTEX_L0A_BASE + uAIdx);
    IssueMmad(
        uCL0Local, uAL0Local, valueL0BLocal, CHUNK_SIZE, DV_TILE, CHUNK_SIZE, MUTEX_L0A_BASE + uAIdx, MUTEX_VALUE_L0B,
        MUTEX_L0C_BASE + uCIdx);
    FixToAivBegin(
        uSlotUBLocal, uCL0Local, CHUNK_SIZE, DV_TILE, SlotFlagId(FLAG_U_PRED_HANDOFF_BASE, uPredSlot),
        MUTEX_L0C_BASE + uCIdx);
}

__aicore__ inline void IssueStateC1Post(
    AscendC::LocalTensor<uint8_t>& staticL1Local, AscendC::LocalTensor<bfloat16_t>& stateL1Local,
    AscendC::LocalTensor<bfloat16_t>& kPlusStateL1Local, AscendC::LocalTensor<bfloat16_t>& aL0ALocal,
    AscendC::LocalTensor<bfloat16_t>& stateL0BLocal, AscendC::LocalTensor<bfloat16_t>& kPlusStateL0BLocal,
    AscendC::LocalTensor<float>& resultL0CLocal, AscendC::LocalTensor<float>& predUBLocal, uint32_t lane,
    uint32_t staticSlot, uint32_t itemId, uint32_t& l0aOpIdx, uint32_t& l0cOpIdx)
{
    using namespace AscendC;
    const uint32_t kPlusStateSlot = itemId % DB_SLOT_NUM;
    const uint32_t uPredSlot = itemId % DB_SLOT_NUM;
    const MutexId stateMutexId = MUTEX_STATE_L0B_BASE + static_cast<MutexId>(lane);
    auto staticSlotL1Local = staticL1Local[staticSlot * STATIC_SLOT_BYTES];
    auto kPlusL1Local = staticSlotL1Local[STATIC_K_PLUS_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto mL1Local = staticSlotL1Local[STATIC_M_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto stateSlotL1Local = stateL1Local[lane * STATE_TILE_ELEMS];
    auto stateSlotL0BLocal = stateL0BLocal[lane * STATE_TILE_ELEMS];
    auto kPlusStateSlotL1Local = kPlusStateL1Local[kPlusStateSlot * CHUNK_DV_TILE_ELEMS];
    auto predSlotUBLocal = predUBLocal[uPredSlot * CHUNK_DV_HALF_ELEMS];

    WaitAivToAic<PIPE_MTE1>(SlotFlagId(FLAG_STATE_HANDOFF_BASE, lane));
    Mutex::Lock<PIPE_MTE1>(stateMutexId);
    CopyL1ToL0B(stateSlotL0BLocal, stateSlotL1Local, HEAD_DIM, HEAD_DIM, DV_TILE, true);
    SetAicToAiv<PIPE_MTE1>(SlotFlagId(FLAG_STATE_HANDOFF_BASE, lane));
    Mutex::Unlock<PIPE_MTE1>(stateMutexId);

    const uint32_t kPlusStateAIdx = l0aOpIdx++ % DB_SLOT_NUM;
    const uint32_t kPlusStateCIdx = l0cOpIdx++ % L0C_QUEUE_DEPTH;
    auto kPlusStateAL0Local = aL0ALocal[kPlusStateAIdx * L0A_SLOT_ELEMS];
    auto kPlusStateCL0Local = resultL0CLocal[kPlusStateCIdx * L0C_SLOT_ELEMS];
    LoadA(kPlusStateAL0Local, kPlusL1Local, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, false, MUTEX_L0A_BASE + kPlusStateAIdx);
    IssueMmadHoldL0B(
        kPlusStateCL0Local, kPlusStateAL0Local, stateSlotL0BLocal, CHUNK_SIZE, DV_TILE, HEAD_DIM,
        MUTEX_L0A_BASE + kPlusStateAIdx, stateMutexId, MUTEX_L0C_BASE + kPlusStateCIdx);
    FixKPlusStateToL1(
        kPlusStateSlotL1Local, kPlusStateCL0Local, MUTEX_K_PLUS_STATE_L1_BASE + static_cast<MutexId>(kPlusStateSlot),
        MUTEX_L0C_BASE + kPlusStateCIdx);

    const uint32_t predAIdx = l0aOpIdx++ % DB_SLOT_NUM;
    const uint32_t predCIdx = l0cOpIdx++ % L0C_QUEUE_DEPTH;
    auto predAL0Local = aL0ALocal[predAIdx * L0A_SLOT_ELEMS];
    auto predCL0Local = resultL0CLocal[predCIdx * L0C_SLOT_ELEMS];
    LoadA(predAL0Local, mL1Local, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, false, MUTEX_L0A_BASE + predAIdx);
    LoadKPlusState(
        kPlusStateL0BLocal, kPlusStateSlotL1Local, MUTEX_K_PLUS_STATE_L1_BASE + static_cast<MutexId>(kPlusStateSlot));
    IssueMmad(
        predCL0Local, predAL0Local, kPlusStateL0BLocal, CHUNK_SIZE, DV_TILE, CHUNK_SIZE, MUTEX_L0A_BASE + predAIdx,
        MUTEX_K_PLUS_STATE_L0B, MUTEX_L0C_BASE + predCIdx);
    FixToAivEnd(
        predSlotUBLocal, predCL0Local, CHUNK_SIZE, DV_TILE, SlotFlagId(FLAG_U_PRED_HANDOFF_BASE, uPredSlot),
        MUTEX_L0C_BASE + predCIdx);
}

__aicore__ inline void IssueStateC2History(
    AscendC::LocalTensor<uint8_t>& staticL1Local, AscendC::LocalTensor<bfloat16_t>& aL0ALocal,
    AscendC::LocalTensor<bfloat16_t>& stateL0BLocal, AscendC::LocalTensor<float>& resultL0CLocal,
    AscendC::LocalTensor<float>& historyUBLocal, uint32_t lane, uint32_t staticSlot, uint32_t itemId,
    uint32_t& l0aOpIdx, uint32_t& l0cOpIdx)
{
    using namespace AscendC;
    const uint32_t v2Slot = itemId % DB_SLOT_NUM;
    const MutexId stateMutexId = MUTEX_STATE_L0B_BASE + static_cast<MutexId>(lane);
    const uint16_t v2FlagId = SlotFlagId(FLAG_V2_PHASE_HANDOFF_BASE, v2Slot);
    auto staticSlotL1Local = staticL1Local[staticSlot * STATIC_SLOT_BYTES];
    auto qPlusL1Local = staticSlotL1Local[STATIC_Q_PLUS_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto stateSlotL0BLocal = stateL0BLocal[lane * STATE_TILE_ELEMS];
    auto historySlotUBLocal = historyUBLocal[v2Slot * CHUNK_DV_HALF_ELEMS];

    const uint32_t historyAIdx = l0aOpIdx++ % DB_SLOT_NUM;
    const uint32_t historyCIdx = l0cOpIdx++ % L0C_QUEUE_DEPTH;
    auto historyAL0Local = aL0ALocal[historyAIdx * L0A_SLOT_ELEMS];
    auto historyCL0Local = resultL0CLocal[historyCIdx * L0C_SLOT_ELEMS];
    LoadA(historyAL0Local, qPlusL1Local, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, false, MUTEX_L0A_BASE + historyAIdx);
    IssueMmadReleaseL0B(
        historyCL0Local, historyAL0Local, stateSlotL0BLocal, CHUNK_SIZE, DV_TILE, HEAD_DIM,
        MUTEX_L0A_BASE + historyAIdx, stateMutexId, MUTEX_L0C_BASE + historyCIdx);
    FixToAivBegin(historySlotUBLocal, historyCL0Local, CHUNK_SIZE, DV_TILE, v2FlagId, MUTEX_L0C_BASE + historyCIdx);
}

__aicore__ inline void IssueStateC2Remainder(
    AscendC::LocalTensor<uint8_t>& staticL1Local, AscendC::LocalTensor<bfloat16_t>& rL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL0ALocal, AscendC::LocalTensor<bfloat16_t>& rL0BLocal,
    AscendC::LocalTensor<float>& resultL0CLocal, AscendC::LocalTensor<float>& deltaUBLocal,
    AscendC::LocalTensor<float>& localUBLocal, uint32_t staticSlot, uint32_t itemId, uint32_t rQueueDepth,
    bool releaseStatic, uint32_t& l0aOpIdx, uint32_t& l0cOpIdx)
{
    using namespace AscendC;
    const uint32_t rSlot = itemId % rQueueDepth;
    const uint32_t v2Slot = itemId % DB_SLOT_NUM;
    const MutexId staticMutexId = MUTEX_STATIC_L1_BASE + static_cast<MutexId>(staticSlot);
    const uint16_t v2FlagId = SlotFlagId(FLAG_V2_PHASE_HANDOFF_BASE, v2Slot);
    auto staticSlotL1Local = staticL1Local[staticSlot * STATIC_SLOT_BYTES];
    auto kTailL1Local = staticSlotL1Local[STATIC_K_TAIL_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto aL1Local = staticSlotL1Local[STATIC_A_SLOT_ADDR].ReinterpretCast<bfloat16_t>();
    auto rSlotL1Local = rL1Local[rSlot * CHUNK_DV_TILE_ELEMS];
    auto deltaSlotUBLocal = deltaUBLocal[v2Slot * STATE_HALF_ELEMS];
    auto localSlotUBLocal = localUBLocal[v2Slot * CHUNK_DV_HALF_ELEMS];

    WaitAivToAic<PIPE_MTE1>(SlotFlagId(FLAG_R_HANDOFF_BASE, rSlot));
    Mutex::Lock<PIPE_MTE1>(MUTEX_R_L0B);
    CopyL1ToL0B(rL0BLocal, rSlotL1Local, CHUNK_SIZE, CHUNK_SIZE, DV_TILE, true);
    SetAicToAiv<PIPE_MTE1>(SlotFlagId(FLAG_R_HANDOFF_BASE, rSlot));
    Mutex::Unlock<PIPE_MTE1>(MUTEX_R_L0B);

    const uint32_t deltaAIdx = l0aOpIdx++ % DB_SLOT_NUM;
    const uint32_t deltaCIdx = l0cOpIdx++ % L0C_QUEUE_DEPTH;
    auto deltaAL0Local = aL0ALocal[deltaAIdx * L0A_SLOT_ELEMS];
    auto deltaCL0Local = resultL0CLocal[deltaCIdx * L0C_SLOT_ELEMS];
    LoadA(deltaAL0Local, kTailL1Local, CHUNK_SIZE, HEAD_DIM, CHUNK_SIZE, true, MUTEX_L0A_BASE + deltaAIdx);
    IssueMmadHoldL0B(
        deltaCL0Local, deltaAL0Local, rL0BLocal, HEAD_DIM, DV_TILE, CHUNK_SIZE, MUTEX_L0A_BASE + deltaAIdx, MUTEX_R_L0B,
        MUTEX_L0C_BASE + deltaCIdx);

    const uint32_t localAIdx = l0aOpIdx++ % DB_SLOT_NUM;
    const uint32_t localCIdx = l0cOpIdx++ % L0C_QUEUE_DEPTH;
    auto localAL0Local = aL0ALocal[localAIdx * L0A_SLOT_ELEMS];
    auto localCL0Local = resultL0CLocal[localCIdx * L0C_SLOT_ELEMS];
    LoadA(localAL0Local, aL1Local, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, false, MUTEX_L0A_BASE + localAIdx);
    if (releaseStatic) {
        Mutex::Unlock<PIPE_MTE1>(staticMutexId);
    }
    IssueMmadReleaseL0B(
        localCL0Local, localAL0Local, rL0BLocal, CHUNK_SIZE, DV_TILE, CHUNK_SIZE, MUTEX_L0A_BASE + localAIdx,
        MUTEX_R_L0B, MUTEX_L0C_BASE + localCIdx);

    FixToAivEnd(deltaSlotUBLocal, deltaCL0Local, HEAD_DIM, DV_TILE, v2FlagId, MUTEX_L0C_BASE + deltaCIdx);
    FixToAiv(localSlotUBLocal, localCL0Local, CHUNK_SIZE, DV_TILE, v2FlagId, MUTEX_L0C_BASE + localCIdx);
}

__aicore__ inline void KernelProcessStateUpdateForAIC(
    __gm__ bfloat16_t* valueGMAddr, __gm__ uint8_t* workspaceGMAddr, const KimiDeltaAttnLiteTilingData& data)
{
    using namespace AscendC;
    if ASCEND_IS_AIC {
        GlobalTensor<bfloat16_t> kPlusGlobal, qPlusGlobal, kTailGlobal, mGlobal, aGlobal, valueGlobal;
        kPlusGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.kPlusOffset));
        qPlusGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.qPlusOffset));
        kTailGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.kTailOffset));
        mGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.mOffset));
        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.aOffset));
        valueGlobal.SetGlobalBuffer(valueGMAddr);

        LocalTensor<uint8_t> staticL1Local(TPosition::A1, 0, STATIC_SLOT_BYTES * STATE_PIPELINE_MAX_LANE_NUM);
        LocalTensor<bfloat16_t> stateL1Local(
            TPosition::A1, STATE_L1_ADDR, STATE_TILE_ELEMS * STATE_PIPELINE_MAX_LANE_NUM);
        LocalTensor<bfloat16_t> valueL1Local(TPosition::A1, VALUE_L1_ADDR, CHUNK_DV_TILE_ELEMS * DB_SLOT_NUM);
        LocalTensor<bfloat16_t> rL1Local(TPosition::A1, R_L1_ADDR, CHUNK_DV_TILE_ELEMS * STATE_R_MAX_QUEUE_DEPTH);
        LocalTensor<bfloat16_t> kPlusStateL1Local(
            TPosition::A1, K_PLUS_STATE_L1_ADDR, CHUNK_DV_TILE_ELEMS * DB_SLOT_NUM);
        LocalTensor<bfloat16_t> aL0ALocal(TPosition::A2, 0, L0A_SLOT_ELEMS * DB_SLOT_NUM);
        LocalTensor<bfloat16_t> stateL0BLocal(
            TPosition::B2, STATE_L0B_ADDR, STATE_TILE_ELEMS * STATE_PIPELINE_MAX_LANE_NUM);
        LocalTensor<bfloat16_t> valueL0BLocal(TPosition::B2, VALUE_L0B_ADDR, CHUNK_DV_TILE_ELEMS);
        LocalTensor<bfloat16_t> rL0BLocal(TPosition::B2, R_L0B_ADDR, CHUNK_DV_TILE_ELEMS);
        LocalTensor<bfloat16_t> kPlusStateL0BLocal(TPosition::B2, K_PLUS_STATE_L0B_ADDR, CHUNK_DV_TILE_ELEMS);
        LocalTensor<float> resultL0CLocal(TPosition::CO1, 0, L0C_SLOT_ELEMS * L0C_QUEUE_DEPTH);
        LocalTensor<float> predUBLocal(TPosition::VECCALC, PRED_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> historyUBLocal(TPosition::VECCALC, HISTORY_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> deltaUBLocal(TPosition::VECCALC, DELTA_UB_ADDR, STATE_HALF_ELEMS * DB_SLOT_NUM);
        LocalTensor<float> localUBLocal(TPosition::VECCALC, LOCAL_UB_ADDR, CHUNK_DV_HALF_ELEMS * DB_SLOT_NUM);
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
                    kPlusGlobal, qPlusGlobal, mGlobal, kTailGlobal, aGlobal, valueGlobal, staticL1Local, valueL1Local,
                    data, taskId, 0, 0, 0, true);
                IssueStateU(
                    staticL1Local, valueL1Local, aL0ALocal, valueL0BLocal, resultL0CLocal, uUBLocal, 0, 0, true,
                    l0aOpIdx, l0cOpIdx);

                for (uint32_t itemId = 0; itemId < itemCount; ++itemId) {
                    const uint32_t staticSlot = itemId % DB_SLOT_NUM;
                    const uint32_t nextItemId = itemId + 1;
                    if (nextItemId < itemCount) {
                        const uint32_t nextStaticSlot = nextItemId % DB_SLOT_NUM;
                        IssueStateInputPreload(
                            kPlusGlobal, qPlusGlobal, mGlobal, kTailGlobal, aGlobal, valueGlobal, staticL1Local,
                            valueL1Local, data, taskId, nextStaticSlot, nextItemId, nextItemId, true);
                    }

                    IssueStateC1Post(
                        staticL1Local, stateL1Local, kPlusStateL1Local, aL0ALocal, stateL0BLocal, kPlusStateL0BLocal,
                        resultL0CLocal, predUBLocal, 0, staticSlot, itemId, l0aOpIdx, l0cOpIdx);
                    IssueStateC2History(
                        staticL1Local, aL0ALocal, stateL0BLocal, resultL0CLocal, historyUBLocal, 0, staticSlot, itemId,
                        l0aOpIdx, l0cOpIdx);

                    if (nextItemId < itemCount) {
                        const uint32_t nextStaticSlot = nextItemId % DB_SLOT_NUM;
                        IssueStateU(
                            staticL1Local, valueL1Local, aL0ALocal, valueL0BLocal, resultL0CLocal, uUBLocal,
                            nextStaticSlot, nextItemId, true, l0aOpIdx, l0cOpIdx);
                    }

                    IssueStateC2Remainder(
                        staticL1Local, rL1Local, aL0ALocal, rL0BLocal, resultL0CLocal, deltaUBLocal, localUBLocal,
                        staticSlot, itemId, rQueueDepth, true, l0aOpIdx, l0cOpIdx);
                }
            } else {
                // 同一 chunk 的 lane 对应同一 batch 的不同 Dv tile, 共用一份静态矩阵.
                // 静态矩阵按 chunk 双槽滚动, Value 仍按 item 双槽分别搬入.
                const uint32_t epochCount = itemCount + laneCount;
                for (uint32_t epoch = 0; epoch < epochCount; ++epoch) {
                    const bool hasC2 = epoch + 1 >= laneCount && epoch < itemCount + laneCount - 1;
                    if (epoch < itemCount) {
                        const uint32_t itemId = epoch;
                        const uint32_t lane = itemId % laneCount;
                        const uint32_t chunkId = itemId / laneCount;
                        const uint32_t taskId = static_cast<uint32_t>(waveTaskBase) + lane;
                        const uint32_t staticSlot = chunkId % DB_SLOT_NUM;
                        IssueStateInputPreload(
                            kPlusGlobal, qPlusGlobal, mGlobal, kTailGlobal, aGlobal, valueGlobal, staticL1Local,
                            valueL1Local, data, taskId, staticSlot, chunkId, itemId, lane == 0);
                        IssueStateU(
                            staticL1Local, valueL1Local, aL0ALocal, valueL0BLocal, resultL0CLocal, uUBLocal, staticSlot,
                            itemId, lane == 0, l0aOpIdx, l0cOpIdx);
                        IssueStateC1Post(
                            staticL1Local, stateL1Local, kPlusStateL1Local, aL0ALocal, stateL0BLocal,
                            kPlusStateL0BLocal, resultL0CLocal, predUBLocal, lane, staticSlot, itemId, l0aOpIdx,
                            l0cOpIdx);
                    }
                    if (hasC2) {
                        const uint32_t itemId = epoch - laneCount + 1;
                        const uint32_t lane = itemId % laneCount;
                        const uint32_t chunkId = itemId / laneCount;
                        const uint32_t staticSlot = chunkId % DB_SLOT_NUM;
                        IssueStateC2History(
                            staticL1Local, aL0ALocal, stateL0BLocal, resultL0CLocal, historyUBLocal, lane, staticSlot,
                            itemId, l0aOpIdx, l0cOpIdx);
                        IssueStateC2Remainder(
                            staticL1Local, rL1Local, aL0ALocal, rL0BLocal, resultL0CLocal, deltaUBLocal, localUBLocal,
                            staticSlot, itemId, rQueueDepth, lane + 1 == laneCount, l0aOpIdx, l0cOpIdx);
                    }
                }
            }

            for (uint32_t slot = 0; slot < DB_SLOT_NUM; ++slot) {
                WaitAivToAic<PIPE_FIX>(SlotFlagId(FLAG_U_PRED_HANDOFF_BASE, slot));
                WaitAivToAic<PIPE_FIX>(SlotFlagId(FLAG_V2_PHASE_HANDOFF_BASE, slot));
            }
            // state lane 0 的常规交接已排空. 同一 FlagID 的额外一次 Wait 表示整个
            // wave 已结束; 收到该信号后, 下一个 wave 才会重新发布 state 空槽.
            WaitAivToAic<PIPE_MTE1>(FLAG_STATE_WAVE_DONE);
        }
    }
}

} // namespace KDALite
