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

// 一个物理槽保存一个完整 chunk, 由同一 MutexID 在 MTE2/V/MTE3 间交接.
constexpr uint32_t PREP_Q_BF16_SLOT_ADDR = 0;
constexpr uint32_t PREP_K_BF16_SLOT_ADDR = PREP_Q_BF16_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_G_FP32_SLOT_ADDR = PREP_K_BF16_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_BETA_BF16_SLOT_ADDR = PREP_G_FP32_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(float);
constexpr uint32_t PREP_CUM_G_FP32_SLOT_ADDR = PREP_BETA_BF16_SLOT_ADDR + CHUNK_SIZE * sizeof(bfloat16_t);
constexpr uint32_t PREP_Q_PLUS_BF16_SLOT_ADDR = PREP_CUM_G_FP32_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(float);
constexpr uint32_t PREP_K_PLUS_BF16_SLOT_ADDR = PREP_Q_PLUS_BF16_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_K_TAIL_BF16_SLOT_ADDR = PREP_K_PLUS_BF16_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_STATE_DECAY_FP32_SLOT_ADDR = PREP_K_TAIL_BF16_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_M_FP32_SLOT_ADDR = PREP_STATE_DECAY_FP32_SLOT_ADDR + HEAD_DIM * sizeof(float);
constexpr uint32_t PREP_M_BF16_SLOT_ADDR = PREP_M_FP32_SLOT_ADDR + CHUNK_C_ELEMS * sizeof(float);
constexpr uint32_t PREP_A_BF16_SLOT_ADDR = PREP_M_BF16_SLOT_ADDR + CHUNK_C_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_SLOT_BYTES = PREP_A_BF16_SLOT_ADDR + CHUNK_C_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_UB_END_ADDR = PREP_SLOT_BYTES;
constexpr MutexId MUTEX_PREP_SLOT_BASE = 0;

static_assert(PREP_UB_END_ADDR <= AIV_USABLE_UB_BYTES, "ChunkPrepare UB allocation exceeds 248 KiB");

static constexpr AscendC::Reg::CastTrait PREP_B16_TO_B32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait PREP_B32_TO_B16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

__simd_vf__ inline void PrepareTransformsVF(
    __ubuf__ bfloat16_t* q, __ubuf__ bfloat16_t* k, __ubuf__ float* g, __ubuf__ float* cumulativeG,
    __ubuf__ bfloat16_t* qPlus, __ubuf__ bfloat16_t* kPlus, __ubuf__ bfloat16_t* kTail, __ubuf__ float* stateDecay,
    uint16_t validLen)
{
    using namespace AscendC;
    Reg::RegTensor<bfloat16_t> qB16Reg, kB16Reg, outB16Reg, zeroB16Reg;
    Reg::RegTensor<float> qReg, kReg, gReg, cumulativeReg, decayReg, outReg, zeroReg;
    Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Duplicate(zeroReg, 0.0F, all);
    Reg::Cast<bfloat16_t, float, PREP_B32_TO_B16>(zeroB16Reg, zeroReg, all);
    const uint16_t tailLen = CHUNK_SIZE - validLen;

    for (uint16_t segment = 0; segment < HEAD_DIM / 64; ++segment) {
        const uint32_t segmentOffset = static_cast<uint32_t>(segment) * 64;
        Reg::Duplicate(cumulativeReg, 0.0F, all);
        for (uint16_t row = 0; row < validLen; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * HEAD_DIM + segmentOffset;
            Reg::LoadAlign(gReg, g + offset);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(qB16Reg, q + offset);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(kB16Reg, k + offset);
            Reg::Cast<float, bfloat16_t, PREP_B16_TO_B32>(qReg, qB16Reg, all);
            Reg::Cast<float, bfloat16_t, PREP_B16_TO_B32>(kReg, kB16Reg, all);
            Reg::Add(cumulativeReg, cumulativeReg, gReg, all);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(cumulativeG + offset, cumulativeReg, all);
            Reg::Exp(decayReg, cumulativeReg, all);

            Reg::Mul(outReg, qReg, decayReg, all);
            Reg::Cast<bfloat16_t, float, PREP_B32_TO_B16>(outB16Reg, outReg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(qPlus + offset, outB16Reg, all);

            Reg::Mul(outReg, kReg, decayReg, all);
            Reg::Cast<bfloat16_t, float, PREP_B32_TO_B16>(outB16Reg, outReg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(kPlus + offset, outB16Reg, all);
        }
        for (uint16_t tail = 0; tail < tailLen; ++tail) {
            const uint32_t offset = static_cast<uint32_t>(validLen + tail) * HEAD_DIM + segmentOffset;
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(cumulativeG + offset, cumulativeReg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(q + offset, zeroB16Reg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(k + offset, zeroB16Reg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(qPlus + offset, zeroB16Reg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(kPlus + offset, zeroB16Reg, all);
        }

        Reg::Exp(decayReg, cumulativeReg, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(stateDecay + segmentOffset, decayReg, all);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        for (uint16_t row = 0; row < validLen; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * HEAD_DIM + segmentOffset;
            Reg::LoadAlign(gReg, cumulativeG + offset);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(kB16Reg, k + offset);
            Reg::Cast<float, bfloat16_t, PREP_B16_TO_B32>(kReg, kB16Reg, all);
            Reg::Sub(decayReg, cumulativeReg, gReg, all);
            Reg::Exp(decayReg, decayReg, all);
            Reg::Mul(outReg, kReg, decayReg, all);
            Reg::Cast<bfloat16_t, float, PREP_B32_TO_B16>(outB16Reg, outReg, all);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(kTail + offset, outB16Reg, all);
        }
        for (uint16_t tail = 0; tail < tailLen; ++tail) {
            const uint32_t offset = static_cast<uint32_t>(validLen + tail) * HEAD_DIM + segmentOffset;
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(kTail + offset, zeroB16Reg, all);
        }
    }
}

// C 为编译期常量. 尾块已在 PrepareTransformsVF 中补零,
// 因此这里不需要运行期 validLen 分支.
__simd_vf__ inline void PreparePairASolveMVF(
    __ubuf__ bfloat16_t* q, __ubuf__ bfloat16_t* k, __ubuf__ float* cumulativeG, __ubuf__ bfloat16_t* beta,
    __ubuf__ float* m, __ubuf__ bfloat16_t* mBf16, __ubuf__ bfloat16_t* aBf16)
{
    using namespace AscendC;
    Reg::RegTensor<int32_t> indexReg;
    Reg::RegTensor<bfloat16_t> b16Reg;
    Reg::RegTensor<float> qIRow0Reg, qIRow1Reg, kIRow0Reg, kIRow1Reg, gIRow0Reg, gIRow1Reg;
    Reg::RegTensor<float> gJReg, kJReg;
    Reg::RegTensor<float> pairAccReg, aAccReg, pairScalarReg, aScalarReg;
    Reg::RegTensor<float> betaReg, rowReg, aRowReg, zeroReg;
    Reg::RegTensor<float> previousRowReg, termReg;
    Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    constexpr Reg::MaskPattern CHUNK_MASK_PATTERN = CHUNK_SIZE == 16 ? Reg::MaskPattern::VL16 :
                                                    CHUNK_SIZE == 32 ? Reg::MaskPattern::VL32 :
                                                                       Reg::MaskPattern::VL64;
    Reg::MaskReg chunkMask = Reg::CreateMask<float, CHUNK_MASK_PATTERN>();
    Reg::MaskReg columnMask;
    Reg::Arange(indexReg, 0);
    Reg::Duplicate(zeroReg, 0.0F, chunkMask);

    for (uint16_t row = 0; row < CHUNK_SIZE; ++row) {
        // 后续行会读取已写入 UB 的 M 前序行. 每行统一发射一次 store->load barrier,
        // 使 row=0 与其他行保持相同的无分支循环结构.
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        const uint32_t rowOffset = static_cast<uint32_t>(row) * HEAD_DIM;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, q + rowOffset);
        Reg::Cast<float, bfloat16_t, PREP_B16_TO_B32>(qIRow0Reg, b16Reg, all);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, q + rowOffset + 64);
        Reg::Cast<float, bfloat16_t, PREP_B16_TO_B32>(qIRow1Reg, b16Reg, all);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, k + rowOffset);
        Reg::Cast<float, bfloat16_t, PREP_B16_TO_B32>(kIRow0Reg, b16Reg, all);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, k + rowOffset + 64);
        Reg::Cast<float, bfloat16_t, PREP_B16_TO_B32>(kIRow1Reg, b16Reg, all);
        Reg::LoadAlign(gIRow0Reg, cumulativeG + rowOffset);
        Reg::LoadAlign(gIRow1Reg, cumulativeG + rowOffset + 64);

        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_BRC_B16>(b16Reg, beta + row);
        Reg::Cast<float, bfloat16_t, PREP_B16_TO_B32>(betaReg, b16Reg, all);
        Reg::CompareScalar<int32_t, CMPMODE::EQ>(columnMask, indexReg, static_cast<int32_t>(row), chunkMask);
        Reg::Select(rowReg, betaReg, zeroReg, columnMask);
        Reg::Duplicate(aRowReg, 0.0F, chunkMask);

        // 对角项的相对衰减因子恒为 1, 直接计算 q_i 与 k_i 的内积.
        Reg::Mul(pairAccReg, qIRow0Reg, kIRow0Reg, all);
        Reg::MulAddDst(pairAccReg, qIRow1Reg, kIRow1Reg, all);
        Reg::ReduceSum(aScalarReg, pairAccReg, all);
        Reg::Duplicate<float, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(aScalarReg, aScalarReg, chunkMask);
        Reg::Select(aRowReg, aScalarReg, aRowReg, columnMask);

        const uint16_t columnCount = row;
        for (uint16_t column = 0; column < columnCount; ++column) {
            const uint32_t columnOffset = static_cast<uint32_t>(column) * HEAD_DIM;

            Reg::LoadAlign(gJReg, cumulativeG + columnOffset);
            Reg::Sub(gJReg, gIRow0Reg, gJReg, all);
            Reg::Exp(gJReg, gJReg, all);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, k + columnOffset);
            Reg::Cast<float, bfloat16_t, PREP_B16_TO_B32>(kJReg, b16Reg, all);
            Reg::Mul(kJReg, kJReg, gJReg, all);
            Reg::Mul(pairAccReg, kIRow0Reg, kJReg, all);
            Reg::Mul(aAccReg, qIRow0Reg, kJReg, all);

            Reg::LoadAlign(gJReg, cumulativeG + columnOffset + 64);
            Reg::Sub(gJReg, gIRow1Reg, gJReg, all);
            Reg::Exp(gJReg, gJReg, all);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, k + columnOffset + 64);
            Reg::Cast<float, bfloat16_t, PREP_B16_TO_B32>(kJReg, b16Reg, all);
            Reg::Mul(kJReg, kJReg, gJReg, all);
            Reg::MulAddDst(pairAccReg, kIRow1Reg, kJReg, all);
            Reg::MulAddDst(aAccReg, qIRow1Reg, kJReg, all);

            Reg::ReduceSum(pairScalarReg, pairAccReg, all);
            Reg::ReduceSum(aScalarReg, aAccReg, all);
            Reg::Duplicate<float, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(
                pairScalarReg, pairScalarReg, chunkMask);
            Reg::Duplicate<float, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(
                aScalarReg, aScalarReg, chunkMask);
            Reg::CompareScalar<int32_t, CMPMODE::EQ>(columnMask, indexReg, static_cast<int32_t>(column), chunkMask);
            Reg::Select(aRowReg, aScalarReg, aRowReg, columnMask);
            Reg::Mul(pairScalarReg, betaReg, pairScalarReg, chunkMask);
            Reg::LoadAlign(previousRowReg, m + column * CHUNK_SIZE);
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

__aicore__ inline void KernelProcessPrepareForAIV(
    __gm__ bfloat16_t* qGMAddr, __gm__ bfloat16_t* kGMAddr, __gm__ float* gGMAddr, __gm__ bfloat16_t* betaGMAddr,
    __gm__ uint8_t* workspaceGMAddr, const KimiDeltaAttnLiteTilingData& data)
{
    using namespace AscendC;
    GlobalTensor<bfloat16_t> qGlobal, kGlobal, betaGlobal;
    GlobalTensor<float> gGlobal;
    GlobalTensor<bfloat16_t> kPlusGlobal, qPlusGlobal, kTailGlobal, mGlobal, aGlobal;
    GlobalTensor<float> stateDecayGlobal;
    qGlobal.SetGlobalBuffer(qGMAddr);
    kGlobal.SetGlobalBuffer(kGMAddr);
    gGlobal.SetGlobalBuffer(gGMAddr);
    betaGlobal.SetGlobalBuffer(betaGMAddr);
    kPlusGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.kPlusOffset));
    qPlusGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.qPlusOffset));
    kTailGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.kTailOffset));
    mGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.mOffset));
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.aOffset));
    stateDecayGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGMAddr + data.stateDecayOffset));

    for (uint32_t taskId = GetBlockIdx(); taskId < data.prepareNumTasks; taskId += data.prepareUseAivNum) {
        constexpr uint32_t slotAddr = 0;
        constexpr MutexId slotMutexId = MUTEX_PREP_SLOT_BASE;
        LocalTensor<bfloat16_t> qUBLocal(TPosition::VECCALC, slotAddr + PREP_Q_BF16_SLOT_ADDR, CHUNK_D_ELEMS);
        LocalTensor<bfloat16_t> kUBLocal(TPosition::VECCALC, slotAddr + PREP_K_BF16_SLOT_ADDR, CHUNK_D_ELEMS);
        LocalTensor<float> gUBLocal(TPosition::VECCALC, slotAddr + PREP_G_FP32_SLOT_ADDR, CHUNK_D_ELEMS);
        LocalTensor<bfloat16_t> betaUBLocal(TPosition::VECCALC, slotAddr + PREP_BETA_BF16_SLOT_ADDR, CHUNK_SIZE);
        LocalTensor<float> cumulativeGUBLocal(TPosition::VECCALC, slotAddr + PREP_CUM_G_FP32_SLOT_ADDR, CHUNK_D_ELEMS);
        LocalTensor<bfloat16_t> qPlusUBLocal(TPosition::VECCALC, slotAddr + PREP_Q_PLUS_BF16_SLOT_ADDR, CHUNK_D_ELEMS);
        LocalTensor<bfloat16_t> kPlusUBLocal(TPosition::VECCALC, slotAddr + PREP_K_PLUS_BF16_SLOT_ADDR, CHUNK_D_ELEMS);
        LocalTensor<bfloat16_t> kTailUBLocal(TPosition::VECCALC, slotAddr + PREP_K_TAIL_BF16_SLOT_ADDR, CHUNK_D_ELEMS);
        LocalTensor<float> stateDecayUBLocal(TPosition::VECCALC, slotAddr + PREP_STATE_DECAY_FP32_SLOT_ADDR, HEAD_DIM);
        LocalTensor<float> mFp32UBLocal(TPosition::VECCALC, slotAddr + PREP_M_FP32_SLOT_ADDR, CHUNK_C_ELEMS);
        LocalTensor<bfloat16_t> mUBLocal(TPosition::VECCALC, slotAddr + PREP_M_BF16_SLOT_ADDR, CHUNK_C_ELEMS);
        LocalTensor<bfloat16_t> aUBLocal(TPosition::VECCALC, slotAddr + PREP_A_BF16_SLOT_ADDR, CHUNK_C_ELEMS);

        const uint32_t batchId = taskId / data.chunkCount;
        const uint32_t chunkId = taskId % data.chunkCount;
        const uint32_t firstToken = chunkId * CHUNK_SIZE;
        const uint32_t validLen = data.seqLen - firstToken < CHUNK_SIZE ? data.seqLen - firstToken : CHUNK_SIZE;
        const uint64_t tokenOffset = (static_cast<uint64_t>(batchId) * data.seqLen + firstToken) * HEAD_DIM;
        const uint64_t betaOffset = static_cast<uint64_t>(batchId) * data.seqLen + firstToken;
        const uint64_t chunkDOffset = static_cast<uint64_t>(taskId) * CHUNK_D_ELEMS;
        const uint64_t chunkCOffset = static_cast<uint64_t>(taskId) * CHUNK_C_ELEMS;

        if constexpr (CHUNK_SIZE > CUBE_BLOCK) {
            // DataCopyPad 只补齐当前 DataBlock. C>16 时先清零整个 beta 槽,
            // 防止尾 chunk 中未被 MTE2 覆盖的元素沿用上一个 task 的数据.
            Mutex::Lock<PIPE_V>(slotMutexId);
            Duplicate<uint16_t>(betaUBLocal.ReinterpretCast<uint16_t>(), 0, CHUNK_SIZE);
            Mutex::Unlock<PIPE_V>(slotMutexId);
        }
        Mutex::Lock<PIPE_MTE2>(slotMutexId);
        CopyPrepareInputs(
            qUBLocal, kUBLocal, gUBLocal, betaUBLocal, qGlobal, kGlobal, gGlobal, betaGlobal, tokenOffset, betaOffset,
            validLen);
        Mutex::Unlock<PIPE_MTE2>(slotMutexId);

        Mutex::Lock<PIPE_V>(slotMutexId);
        asc_vf_call<PrepareTransformsVF>(
            reinterpret_cast<__ubuf__ bfloat16_t*>(qUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(kUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(gUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(cumulativeGUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(qPlusUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(kPlusUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(kTailUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(stateDecayUBLocal.GetPhyAddr()), static_cast<uint16_t>(validLen));
        asc_vf_call<PreparePairASolveMVF>(
            reinterpret_cast<__ubuf__ bfloat16_t*>(qUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(kUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(cumulativeGUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(betaUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(mFp32UBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(mUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t*>(aUBLocal.GetPhyAddr()));
        Mutex::Unlock<PIPE_V>(slotMutexId);

        Mutex::Lock<PIPE_MTE3>(slotMutexId);
        DataCopy(kPlusGlobal[chunkDOffset], kPlusUBLocal, CHUNK_D_ELEMS);
        DataCopy(qPlusGlobal[chunkDOffset], qPlusUBLocal, CHUNK_D_ELEMS);
        DataCopy(kTailGlobal[chunkDOffset], kTailUBLocal, CHUNK_D_ELEMS);
        DataCopy(mGlobal[chunkCOffset], mUBLocal, CHUNK_C_ELEMS);
        DataCopy(aGlobal[chunkCOffset], aUBLocal, CHUNK_C_ELEMS);
        DataCopy(stateDecayGlobal[static_cast<uint64_t>(taskId) * HEAD_DIM], stateDecayUBLocal, HEAD_DIM);
        Mutex::Unlock<PIPE_MTE3>(slotMutexId);
    }
}

} // namespace KDALite
