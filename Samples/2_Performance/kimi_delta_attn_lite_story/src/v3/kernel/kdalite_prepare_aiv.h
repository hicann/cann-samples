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

// 每个物理槽依次在 MTE2(VP)->V(VP)->MTE3(VP)->V(VS)->MTE3(VS) 之间交接.
constexpr MutexId MUTEX_PREP_SLOT_BASE = 0;

static constexpr AscendC::Reg::CastTrait PREP_B16_TO_B32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait PREP_B32_TO_B16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

__simd_vf__ inline void PrepareTransformsVF(
    __ubuf__ bfloat16_t* q, __ubuf__ bfloat16_t* k, __ubuf__ float* cumulativeG, __ubuf__ bfloat16_t* kPlus,
    __ubuf__ bfloat16_t* qFactor, __ubuf__ bfloat16_t* kFactor, __ubuf__ bfloat16_t* kInvFactor,
    __ubuf__ bfloat16_t* kTail, __ubuf__ float* stateDecay, uint16_t validLen)
{
    using namespace AscendC;
    Reg::RegTensor<bfloat16_t> qB16Reg, kB16Reg, outB16Reg, zeroB16Reg;
    Reg::RegTensor<float> qReg, kReg, gReg, cumulativeReg, anchorReg, anchorDecayReg;
    Reg::RegTensor<float> decayReg, scaledDecayReg, outReg, zeroReg;
    Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Duplicate(zeroReg, 0.0F, all);
    Reg::Cast<bfloat16_t, float, PREP_B32_TO_B16>(zeroB16Reg, zeroReg, all);
    const uint16_t tailLen = CHUNK_SIZE - validLen;

    for (uint16_t segment = 0; segment < HEAD_DIM / 64; ++segment) {
        const uint32_t segmentOffset = static_cast<uint32_t>(segment) * 64;
        Reg::Duplicate(cumulativeReg, 0.0F, all);
        for (uint16_t row = 0; row < validLen; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * HEAD_DIM + segmentOffset;
            Reg::LoadAlign(gReg, cumulativeG + offset);
            Reg::Add(cumulativeReg, cumulativeReg, gReg, all);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(cumulativeG + offset, cumulativeReg, all);
        }

        // Cube 只读取重标定后的 factor. C=32 且 -5<=g<=0 时, anchor=G_tail/2 将
        // 两侧指数的范围从约 [-160,160] 缩小到 [-80,80], 且不改变点积结果.
        Reg::Muls(anchorReg, cumulativeReg, 0.5F, all);
        Reg::Exp(anchorDecayReg, anchorReg, all);
        Reg::Mul(decayReg, anchorDecayReg, anchorDecayReg, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(stateDecay + segmentOffset, decayReg, all);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        for (uint16_t row = 0; row < validLen; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * HEAD_DIM + segmentOffset;
            Reg::LoadAlign(gReg, cumulativeG + offset);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(qB16Reg, q + offset);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(kB16Reg, k + offset);
            Reg::Cast<float, bfloat16_t, PREP_B16_TO_B32>(qReg, qB16Reg, all);
            Reg::Cast<float, bfloat16_t, PREP_B16_TO_B32>(kReg, kB16Reg, all);

            Reg::Sub(decayReg, gReg, anchorReg, all);
            Reg::Exp(decayReg, decayReg, all);

            // exp(G)=exp(G-anchor)*exp(anchor), 因此无需再发射一次 Exp.
            Reg::Mul(scaledDecayReg, decayReg, anchorDecayReg, all);
            Reg::Mul(outReg, qReg, scaledDecayReg, all);
            Reg::Cast<bfloat16_t, float, PREP_B32_TO_B16>(outB16Reg, outReg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(q + offset, outB16Reg, all);
            Reg::Mul(outReg, kReg, scaledDecayReg, all);
            Reg::Cast<bfloat16_t, float, PREP_B32_TO_B16>(outB16Reg, outReg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(kPlus + offset, outB16Reg, all);

            Reg::Mul(outReg, qReg, decayReg, all);
            Reg::Cast<bfloat16_t, float, PREP_B32_TO_B16>(outB16Reg, outReg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(qFactor + offset, outB16Reg, all);
            Reg::Mul(outReg, kReg, decayReg, all);
            Reg::Cast<bfloat16_t, float, PREP_B32_TO_B16>(outB16Reg, outReg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(kFactor + offset, outB16Reg, all);

            Reg::Sub(decayReg, anchorReg, gReg, all);
            Reg::Exp(decayReg, decayReg, all);
            Reg::Mul(outReg, kReg, decayReg, all);
            Reg::Cast<bfloat16_t, float, PREP_B32_TO_B16>(outB16Reg, outReg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(kInvFactor + offset, outB16Reg, all);

            // exp(G_tail-G)=exp(anchor-G)*exp(anchor).
            Reg::Mul(scaledDecayReg, decayReg, anchorDecayReg, all);
            Reg::Mul(outReg, kReg, scaledDecayReg, all);
            Reg::Cast<bfloat16_t, float, PREP_B32_TO_B16>(outB16Reg, outReg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(kTail + offset, outB16Reg, all);
        }
        for (uint16_t tail = 0; tail < tailLen; ++tail) {
            const uint32_t offset = static_cast<uint32_t>(validLen + tail) * HEAD_DIM + segmentOffset;
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(q + offset, zeroB16Reg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(kPlus + offset, zeroB16Reg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(qFactor + offset, zeroB16Reg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(kFactor + offset, zeroB16Reg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(kInvFactor + offset, zeroB16Reg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(kTail + offset, zeroB16Reg, all);
        }
    }
}

// AIC 以 BF16 输入计算 Pair/Araw, 并在 FP32 中累加. AIV 求解 M 时保留 FP32 前序行,
// 仅在写 workspace 前将 M/A 转为 BF16.
__simd_vf__ inline void PrepareSolveMVF(
    __ubuf__ float* pair, __ubuf__ float* aRaw, __ubuf__ bfloat16_t* beta, __ubuf__ float* m,
    __ubuf__ bfloat16_t* mBf16, __ubuf__ bfloat16_t* aBf16)
{
    using namespace AscendC;
    Reg::RegTensor<int32_t> indexReg, rowIndexReg;
    Reg::RegTensor<bfloat16_t> b16Reg;
    Reg::RegTensor<float> pairScalarReg;
    Reg::RegTensor<float> betaReg, rowReg, aRowReg, zeroReg;
    Reg::RegTensor<float> previousRowReg, termReg;
    constexpr Reg::MaskPattern CHUNK_MASK_PATTERN = CHUNK_SIZE == 16 ? Reg::MaskPattern::VL16 : Reg::MaskPattern::VL32;
    Reg::MaskReg chunkMask = Reg::CreateMask<float, CHUNK_MASK_PATTERN>();
    Reg::MaskReg columnMask, activeColumnMask;
    Reg::Arange(indexReg, 0);
    Reg::Duplicate(zeroReg, 0.0F, chunkMask);

    for (uint16_t row = 0; row < CHUNK_SIZE; ++row) {
        // 后续行会读取已写入 UB 的 M 前序行. 每行统一发射一次 store->load barrier,
        // row=0 也走同一条无分支循环.
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        const uint32_t rowOffset = static_cast<uint32_t>(row) * CHUNK_SIZE;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_BRC_B16>(b16Reg, beta + row);
        Reg::Cast<float, bfloat16_t, PREP_B16_TO_B32>(betaReg, b16Reg, chunkMask);
        Reg::CompareScalar<int32_t, CMPMODE::EQ>(columnMask, indexReg, static_cast<int32_t>(row), chunkMask);
        Reg::Select(rowReg, betaReg, zeroReg, columnMask);
        Reg::LoadAlign(aRowReg, aRaw + rowOffset);
        Reg::CompareScalar<int32_t, CMPMODE::LE>(columnMask, indexReg, static_cast<int32_t>(row), chunkMask);
        Reg::Select(aRowReg, aRowReg, zeroReg, columnMask);

        Reg::Duplicate(rowIndexReg, static_cast<int32_t>(row), chunkMask);
        for (uint16_t column = 0; column < CHUNK_SIZE; ++column) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(pairScalarReg, pair + rowOffset + column);
            Reg::CompareScalar<int32_t, CMPMODE::GT>(
                activeColumnMask, rowIndexReg, static_cast<int32_t>(column), chunkMask);
            Reg::Select(pairScalarReg, pairScalarReg, zeroReg, activeColumnMask);
            Reg::Mul(pairScalarReg, betaReg, pairScalarReg, chunkMask);
            Reg::LoadAlign(previousRowReg, m + column * CHUNK_SIZE);
            Reg::Select(previousRowReg, previousRowReg, zeroReg, activeColumnMask);
            Reg::Mul(termReg, previousRowReg, pairScalarReg, chunkMask);
            Reg::Sub(rowReg, rowReg, termReg, chunkMask);
        }

        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(m + row * CHUNK_SIZE, rowReg, chunkMask);
        Reg::Cast<bfloat16_t, float, PREP_B32_TO_B16>(b16Reg, aRowReg, chunkMask);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(aBf16 + row * CHUNK_SIZE, b16Reg, chunkMask);
    }

    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint16_t row = 0; row < CHUNK_SIZE; ++row) {
        Reg::LoadAlign(rowReg, m + row * CHUNK_SIZE);
        Reg::Cast<bfloat16_t, float, PREP_B32_TO_B16>(b16Reg, rowReg, chunkMask);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(mBf16 + row * CHUNK_SIZE, b16Reg, chunkMask);
    }
}

__aicore__ inline void CopyPrepareInputs(
    AscendC::LocalTensor<bfloat16_t>& qUBLocal, AscendC::LocalTensor<bfloat16_t>& kUBLocal,
    AscendC::LocalTensor<float>& gUBLocal, AscendC::LocalTensor<bfloat16_t>& betaUBLocal,
    const AscendC::GlobalTensor<bfloat16_t>& qGlobal, const AscendC::GlobalTensor<bfloat16_t>& kGlobal,
    const AscendC::GlobalTensor<float>& gGlobal, const AscendC::GlobalTensor<bfloat16_t>& betaGlobal,
    uint64_t tokenOffset, uint64_t betaOffset, uint32_t validLen)
{
    using namespace AscendC;
    CopyGmToUbRows(qUBLocal, qGlobal[tokenOffset], validLen, HEAD_DIM, HEAD_DIM);
    CopyGmToUbRows(kUBLocal, kGlobal[tokenOffset], validLen, HEAD_DIM, HEAD_DIM);
    CopyGmToUbRows(gUBLocal, gGlobal[tokenOffset], validLen, HEAD_DIM, HEAD_DIM);
    DataCopyExtParams betaParams;
    betaParams.blockCount = 1;
    betaParams.blockLen = validLen * sizeof(bfloat16_t);
    betaParams.srcStride = 0;
    betaParams.dstStride = 0;
    DataCopyPadExtParams<bfloat16_t> betaPadParams;
    betaPadParams.isPad = true;
    betaPadParams.leftPadding = 0;
    constexpr uint32_t BF16_ELEMS_PER_BLOCK = C0_BYTES / sizeof(bfloat16_t);
    betaPadParams.rightPadding =
        static_cast<uint8_t>((BF16_ELEMS_PER_BLOCK - validLen % BF16_ELEMS_PER_BLOCK) % BF16_ELEMS_PER_BLOCK);
    betaPadParams.paddingValue = static_cast<bfloat16_t>(0);
    DataCopyPad(betaUBLocal, betaGlobal[betaOffset], betaParams, betaPadParams);
}

template <typename T>
__aicore__ inline void CopyPrepareNdToNzGroupsToL1(
    AscendC::LocalTensor<T>& dstL1Local, const AscendC::LocalTensor<T>& srcUBLocal, uint32_t rowElems)
{
    using namespace AscendC;
    constexpr uint32_t c0Elems = C0ElemNum<T>();
    DataCopyParams params(CHUNK_SIZE, 1, rowElems / c0Elems - 1, 0);
    for (uint16_t group = 0; group < rowElems / c0Elems; ++group) {
        DataCopy(dstL1Local[group * CHUNK_SIZE * c0Elems], srcUBLocal[group * c0Elems], params);
    }
}

__aicore__ inline void IssuePrepareVpForAIV(
    const AscendC::GlobalTensor<bfloat16_t>& qGlobal, const AscendC::GlobalTensor<bfloat16_t>& kGlobal,
    const AscendC::GlobalTensor<float>& gGlobal, const AscendC::GlobalTensor<bfloat16_t>& betaGlobal,
    const AscendC::GlobalTensor<bfloat16_t>& qPlusGlobal, const AscendC::GlobalTensor<bfloat16_t>& kTailGlobal,
    const AscendC::GlobalTensor<float>& stateDecayGlobal, const KimiDeltaAttnLiteTilingData& data, uint32_t pairTaskId,
    uint32_t cvSlot, uint32_t subAivIdx)
{
    using namespace AscendC;
    const uint32_t taskId = pairTaskId * PREP_SUB_AIV_NUM + subAivIdx;
    const bool validTask = taskId < data.prepareNumTasks;
    const uint32_t prepUbSlotAddr = cvSlot * PREP_SLOT_BYTES;
    const uint32_t l1SlotAddr = cvSlot * PREP_L1_CV_SLOT_BYTES + subAivIdx * PREP_L1_SLOT_BYTES;
    const MutexId prepUbSlotMutexId = MUTEX_PREP_SLOT_BASE + static_cast<MutexId>(cvSlot);
    const uint16_t l1HandoffFlagId = SlotFlagId(FLAG_PREP_L1_HANDOFF_BASE, cvSlot);

    LocalTensor<bfloat16_t> kFactorL1Local(TPosition::A1, l1SlotAddr + PREP_K_FACTOR_L1_ADDR, CHUNK_D_ELEMS);
    LocalTensor<bfloat16_t> qFactorL1Local(TPosition::A1, l1SlotAddr + PREP_Q_FACTOR_L1_ADDR, CHUNK_D_ELEMS);
    LocalTensor<bfloat16_t> kInvFactorL1Local(TPosition::A1, l1SlotAddr + PREP_K_INV_FACTOR_L1_ADDR, CHUNK_D_ELEMS);
    LocalTensor<bfloat16_t> qUBLocal(TPosition::VECCALC, prepUbSlotAddr + PREP_Q_BF16_SLOT_ADDR, CHUNK_D_ELEMS);
    LocalTensor<bfloat16_t> kUBLocal(TPosition::VECCALC, prepUbSlotAddr + PREP_K_BF16_SLOT_ADDR, CHUNK_D_ELEMS);
    LocalTensor<float> gUBLocal(TPosition::VECCALC, prepUbSlotAddr + PREP_G_FP32_SLOT_ADDR, CHUNK_D_ELEMS);
    LocalTensor<bfloat16_t> betaUBLocal(TPosition::VECCALC, prepUbSlotAddr + PREP_BETA_BF16_SLOT_ADDR, CHUNK_SIZE);
    LocalTensor<bfloat16_t> kPlusUBLocal(
        TPosition::VECCALC, prepUbSlotAddr + PREP_K_PLUS_BF16_SLOT_ADDR, CHUNK_D_ELEMS);
    LocalTensor<bfloat16_t> qFactorUBLocal(
        TPosition::VECCALC, prepUbSlotAddr + PREP_Q_FACTOR_BF16_SLOT_ADDR, CHUNK_D_ELEMS);
    LocalTensor<bfloat16_t> kFactorUBLocal(
        TPosition::VECCALC, prepUbSlotAddr + PREP_K_FACTOR_BF16_SLOT_ADDR, CHUNK_D_ELEMS);
    LocalTensor<bfloat16_t> kInvFactorUBLocal(
        TPosition::VECCALC, prepUbSlotAddr + PREP_K_INV_FACTOR_BF16_SLOT_ADDR, CHUNK_D_ELEMS);
    LocalTensor<bfloat16_t> kTailUBLocal(
        TPosition::VECCALC, prepUbSlotAddr + PREP_K_TAIL_BF16_SLOT_ADDR, CHUNK_D_ELEMS);
    LocalTensor<float> stateDecayUBLocal(
        TPosition::VECCALC, prepUbSlotAddr + PREP_STATE_DECAY_FP32_SLOT_ADDR, HEAD_DIM);

    if (validTask) {
        const uint32_t batchId = taskId / data.chunkCount;
        const uint32_t chunkId = taskId % data.chunkCount;
        const uint32_t firstToken = chunkId * CHUNK_SIZE;
        const uint32_t validLen = data.seqLen - firstToken < CHUNK_SIZE ? data.seqLen - firstToken : CHUNK_SIZE;
        const uint64_t tokenOffset = (static_cast<uint64_t>(batchId) * data.seqLen + firstToken) * HEAD_DIM;
        const uint64_t betaOffset = static_cast<uint64_t>(batchId) * data.seqLen + firstToken;
        const uint64_t chunkDOffset = static_cast<uint64_t>(taskId) * CHUNK_D_ELEMS;

        if constexpr (CHUNK_SIZE > CUBE_BLOCK) {
            Mutex::Lock<PIPE_V>(prepUbSlotMutexId);
            Duplicate<uint16_t>(betaUBLocal.ReinterpretCast<uint16_t>(), 0, CHUNK_SIZE);
            Mutex::Unlock<PIPE_V>(prepUbSlotMutexId);
        }
        Mutex::Lock<PIPE_MTE2>(prepUbSlotMutexId);
        CopyPrepareInputs(
            qUBLocal, kUBLocal, gUBLocal, betaUBLocal, qGlobal, kGlobal, gGlobal, betaGlobal, tokenOffset, betaOffset,
            validLen);
        Mutex::Unlock<PIPE_MTE2>(prepUbSlotMutexId);

        Mutex::Lock<PIPE_V>(prepUbSlotMutexId);
        asc_vf_call<PrepareTransformsVF>(
            reinterpret_cast<__ubuf__ bfloat16_t*>(qUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(kUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(gUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(kPlusUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(qFactorUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(kFactorUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(kInvFactorUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(kTailUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(stateDecayUBLocal.GetPhyAddr()), static_cast<uint16_t>(validLen));
        Mutex::Unlock<PIPE_V>(prepUbSlotMutexId);

        WaitAicToAiv<PIPE_MTE3>(l1HandoffFlagId);
        Mutex::Lock<PIPE_MTE3>(prepUbSlotMutexId);
        CopyPrepareNdToNzGroupsToL1(kFactorL1Local, kFactorUBLocal, HEAD_DIM);
        CopyPrepareNdToNzGroupsToL1(qFactorL1Local, qFactorUBLocal, HEAD_DIM);
        CopyPrepareNdToNzGroupsToL1(kInvFactorL1Local, kInvFactorUBLocal, HEAD_DIM);
        SetAivToAic<PIPE_MTE3>(l1HandoffFlagId);
        DataCopy(qPlusGlobal[chunkDOffset], qUBLocal, CHUNK_D_ELEMS);
        DataCopy(kTailGlobal[chunkDOffset], kTailUBLocal, CHUNK_D_ELEMS);
        DataCopy(stateDecayGlobal[static_cast<uint64_t>(taskId) * HEAD_DIM], stateDecayUBLocal, HEAD_DIM);
        Mutex::Unlock<PIPE_MTE3>(prepUbSlotMutexId);
    } else {
        // 尾部只有一个 task 时, AIV1 不计算, 但必须与 AIV0 保持相同的 mode2 握手次数.
        WaitAicToAiv<PIPE_MTE3>(l1HandoffFlagId);
        SetAivToAic<PIPE_MTE3>(l1HandoffFlagId);
    }
}

__aicore__ inline void IssuePrepareVsForAIV(
    const AscendC::GlobalTensor<bfloat16_t>& mGlobal, const AscendC::GlobalTensor<bfloat16_t>& aGlobal,
    const KimiDeltaAttnLiteTilingData& data, uint32_t pairTaskId, uint32_t cvSlot, uint32_t subAivIdx)
{
    using namespace AscendC;
    const uint32_t taskId = pairTaskId * PREP_SUB_AIV_NUM + subAivIdx;
    const bool validTask = taskId < data.prepareNumTasks;
    const uint32_t prepUbSlotAddr = cvSlot * PREP_SLOT_BYTES;
    const uint32_t l1SlotAddr = cvSlot * PREP_L1_CV_SLOT_BYTES + subAivIdx * PREP_L1_SLOT_BYTES;
    const MutexId prepUbSlotMutexId = MUTEX_PREP_SLOT_BASE + static_cast<MutexId>(cvSlot);
    const uint16_t pairArawFlagId = SlotFlagId(FLAG_PREP_PAIR_ARAW_HANDOFF_BASE, cvSlot);
    const uint16_t wFlagId = SlotFlagId(FLAG_PREP_W_HANDOFF_BASE, cvSlot);

    LocalTensor<bfloat16_t> mL1Local(TPosition::A1, l1SlotAddr + PREP_W_M_L1_ADDR, CHUNK_C_ELEMS);
    LocalTensor<bfloat16_t> kPlusL1Local(TPosition::A1, l1SlotAddr + PREP_W_K_PLUS_L1_ADDR, CHUNK_D_ELEMS);
    LocalTensor<bfloat16_t> betaUBLocal(TPosition::VECCALC, prepUbSlotAddr + PREP_BETA_BF16_SLOT_ADDR, CHUNK_SIZE);
    LocalTensor<bfloat16_t> kPlusUBLocal(
        TPosition::VECCALC, prepUbSlotAddr + PREP_K_PLUS_BF16_SLOT_ADDR, CHUNK_D_ELEMS);
    LocalTensor<float> pairUBLocal(TPosition::VECCALC, prepUbSlotAddr + PREP_PAIR_FP32_UB_ADDR, CHUNK_C_ELEMS);
    LocalTensor<float> aRawUBLocal(TPosition::VECCALC, prepUbSlotAddr + PREP_A_RAW_FP32_UB_ADDR, CHUNK_C_ELEMS);
    LocalTensor<float> mFp32UBLocal(TPosition::VECCALC, prepUbSlotAddr + PREP_M_FP32_SLOT_ADDR, CHUNK_C_ELEMS);
    LocalTensor<bfloat16_t> mUBLocal(TPosition::VECCALC, prepUbSlotAddr + PREP_M_BF16_SLOT_ADDR, CHUNK_C_ELEMS);
    LocalTensor<bfloat16_t> aUBLocal(TPosition::VECCALC, prepUbSlotAddr + PREP_A_BF16_SLOT_ADDR, CHUNK_C_ELEMS);

    WaitAicToAiv<PIPE_V>(pairArawFlagId);
    if (validTask) {
        const uint64_t chunkCOffset = static_cast<uint64_t>(taskId) * CHUNK_C_ELEMS;
        Mutex::Lock<PIPE_V>(prepUbSlotMutexId);
        asc_vf_call<PrepareSolveMVF>(
            reinterpret_cast<__ubuf__ float*>(pairUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(aRawUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(betaUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(mFp32UBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(mUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(aUBLocal.GetPhyAddr()));
        Mutex::Unlock<PIPE_V>(prepUbSlotMutexId);
        // Vector 读完 Pair/Araw 后即可归还结果槽. 后续 L1/GM 写回由 Mutex 单独排序.
        SetAivToAic<PIPE_V>(pairArawFlagId);

        WaitAicToAiv<PIPE_MTE3>(wFlagId);
        Mutex::Lock<PIPE_MTE3>(prepUbSlotMutexId);
        CopyPrepareNdToNzGroupsToL1(mL1Local, mUBLocal, CHUNK_SIZE);
        CopyPrepareNdToNzGroupsToL1(kPlusL1Local, kPlusUBLocal, HEAD_DIM);
        SetAivToAic<PIPE_MTE3>(wFlagId);
        DataCopy(mGlobal[chunkCOffset], mUBLocal, CHUNK_C_ELEMS);
        DataCopy(aGlobal[chunkCOffset], aUBLocal, CHUNK_C_ELEMS);
        Mutex::Unlock<PIPE_MTE3>(prepUbSlotMutexId);
    } else {
        SetAivToAic<PIPE_V>(pairArawFlagId);
        // 尾部只有一个 task 时不产生 W, 但 mode2 仍要求两路 AIV 完成同样的交接.
        WaitAicToAiv<PIPE_MTE3>(wFlagId);
        SetAivToAic<PIPE_MTE3>(wFlagId);
    }
}

__aicore__ inline void KernelProcessPrepareForAIV(
    __gm__ bfloat16_t* qGMAddr, __gm__ bfloat16_t* kGMAddr, __gm__ float* gGMAddr, __gm__ bfloat16_t* betaGMAddr,
    __gm__ uint8_t* workspaceGMAddr, const KimiDeltaAttnLiteTilingData& data)
{
    using namespace AscendC;
    if ASCEND_IS_AIV {
        GlobalTensor<bfloat16_t> qGlobal, kGlobal, betaGlobal;
        GlobalTensor<float> gGlobal;
        GlobalTensor<bfloat16_t> qPlusGlobal, kTailGlobal, mGlobal, aGlobal;
        GlobalTensor<float> stateDecayGlobal;
        qGlobal.SetGlobalBuffer(qGMAddr);
        kGlobal.SetGlobalBuffer(kGMAddr);
        gGlobal.SetGlobalBuffer(gGMAddr);
        betaGlobal.SetGlobalBuffer(betaGMAddr);
        qPlusGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.qPlusOffset));
        kTailGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.kTailOffset));
        mGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.mOffset));
        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.aOffset));
        stateDecayGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGMAddr + data.stateDecayOffset));

        const uint32_t subAivIdx = GetSubBlockIdx();
        const uint32_t aicIdx = GetBlockIdx() / GetSubBlockNum();
        if (aicIdx >= data.preparePairNumTasks) {
            return;
        }
        const uint32_t pairTaskCount = CeilDiv<uint32_t>(data.preparePairNumTasks - aicIdx, data.prepareUseAicNum);
        const uint32_t preloadCount = pairTaskCount < PREP_CV_SLOT_NUM ? pairTaskCount : PREP_CV_SLOT_NUM;

        for (uint32_t cvSlot = 0; cvSlot < PREP_CV_SLOT_NUM; ++cvSlot) {
            SetAivToAic<PIPE_V>(SlotFlagId(FLAG_PREP_PAIR_ARAW_HANDOFF_BASE, cvSlot));
        }

        // 预热阶段连续发射两个 VP, 避免 VS 对结果槽的等待阻塞下一次 VP Transform.
        for (uint32_t ordinal = 0; ordinal < preloadCount; ++ordinal) {
            const uint32_t pairTaskId = aicIdx + ordinal * data.prepareUseAicNum;
            IssuePrepareVpForAIV(
                qGlobal, kGlobal, gGlobal, betaGlobal, qPlusGlobal, kTailGlobal, stateDecayGlobal, data, pairTaskId,
                ordinal, subAivIdx);
        }

        uint32_t ordinal = 0;
        // 稳态阶段在 VS(t) 释放 slot 后, 立即在同槽发射 VP(t+2).
        for (; ordinal + PREP_CV_SLOT_NUM < pairTaskCount; ++ordinal) {
            const uint32_t cvSlot = ordinal % PREP_CV_SLOT_NUM;
            const uint32_t pairTaskId = aicIdx + ordinal * data.prepareUseAicNum;
            const uint32_t futurePairTaskId = pairTaskId + PREP_CV_SLOT_NUM * data.prepareUseAicNum;
            IssuePrepareVsForAIV(mGlobal, aGlobal, data, pairTaskId, cvSlot, subAivIdx);
            IssuePrepareVpForAIV(
                qGlobal, kGlobal, gGlobal, betaGlobal, qPlusGlobal, kTailGlobal, stateDecayGlobal, data,
                futurePairTaskId, cvSlot, subAivIdx);
        }

        // 收尾阶段只排空最后两个 VS.
        for (; ordinal < pairTaskCount; ++ordinal) {
            const uint32_t cvSlot = ordinal % PREP_CV_SLOT_NUM;
            const uint32_t pairTaskId = aicIdx + ordinal * data.prepareUseAicNum;
            IssuePrepareVsForAIV(mGlobal, aGlobal, data, pairTaskId, cvSlot, subAivIdx);
        }

        // 循环前为每个槽发布空闲信号; 此处统一消费各槽最终归还的信号.
        for (uint32_t cvSlot = 0; cvSlot < PREP_CV_SLOT_NUM; ++cvSlot) {
            WaitAicToAiv<PIPE_MTE3>(SlotFlagId(FLAG_PREP_L1_HANDOFF_BASE, cvSlot));
        }
    }
}

} // namespace KDALite
