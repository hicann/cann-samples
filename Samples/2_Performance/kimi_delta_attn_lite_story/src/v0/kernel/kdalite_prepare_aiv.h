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

// Prepare 的输入和中间结果从 UB 起始位置顺序排布.
constexpr uint32_t PREP_Q_BF16_UB_ADDR = 0;
constexpr uint32_t PREP_K_BF16_UB_ADDR = PREP_Q_BF16_UB_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_G_UB_ADDR = PREP_K_BF16_UB_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_Q_PLUS_BF16_UB_ADDR = PREP_G_UB_ADDR + CHUNK_D_ELEMS * sizeof(float);
constexpr uint32_t PREP_K_PLUS_BF16_UB_ADDR = PREP_Q_PLUS_BF16_UB_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_K_TAIL_BF16_UB_ADDR = PREP_K_PLUS_BF16_UB_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_BETA_BF16_UB_ADDR = PREP_K_TAIL_BF16_UB_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_M_FP32_UB_ADDR = PREP_BETA_BF16_UB_ADDR + CHUNK_SIZE * sizeof(bfloat16_t);
constexpr uint32_t PREP_M_BF16_UB_ADDR = PREP_M_FP32_UB_ADDR + CHUNK_C_ELEMS * sizeof(float);
constexpr uint32_t PREP_A_BF16_UB_ADDR = PREP_M_BF16_UB_ADDR + CHUNK_C_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_CUM_G_FP32_UB_ADDR = PREP_A_BF16_UB_ADDR + CHUNK_C_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_UB_END_ADDR = PREP_CUM_G_FP32_UB_ADDR + CHUNK_D_ELEMS * sizeof(float);
constexpr uint32_t AIV_USABLE_UB_BYTES = 248 * 1024;

static_assert(PREP_UB_END_ADDR <= AIV_USABLE_UB_BYTES, "Prepare UB allocation exceeds 248 KiB");

constexpr MutexId MUTEX_PREP_INPUT_UB = 0;

static constexpr AscendC::Reg::CastTrait PREP_B16_TO_B32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait PREP_B32_TO_B16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

__simd_vf__ inline void PrepareTransformsVF(
    __ubuf__ bfloat16_t* q, __ubuf__ bfloat16_t* k, __ubuf__ float* g, __ubuf__ float* cumulativeG,
    __ubuf__ bfloat16_t* qPlus, __ubuf__ bfloat16_t* kPlus, __ubuf__ bfloat16_t* kTail)
{
    using namespace AscendC;
    Reg::RegTensor<bfloat16_t> qB16Reg, kB16Reg, outB16Reg;
    Reg::RegTensor<float> qReg, kReg, gReg, cumulativeReg, decayReg, outReg;
    Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    for (uint16_t segment = 0; segment < HEAD_DIM / 64; ++segment) {
        const uint32_t segmentOffset = static_cast<uint32_t>(segment) * 64;
        Reg::Duplicate(cumulativeReg, 0.0F, all);
        for (uint16_t row = 0; row < CHUNK_SIZE; ++row) {
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

        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        for (uint16_t row = 0; row < CHUNK_SIZE; ++row) {
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
    }
}

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
        Reg::Cast<bfloat16_t, float, PREP_B32_TO_B16>(b16Reg, rowReg, chunkMask);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(mBf16 + row * CHUNK_SIZE, b16Reg, chunkMask);
        Reg::Cast<bfloat16_t, float, PREP_B32_TO_B16>(b16Reg, aRowReg, chunkMask);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(aBf16 + row * CHUNK_SIZE, b16Reg, chunkMask);
    }
}

__aicore__ inline void CopyPrepareInputs(
    AscendC::LocalTensor<bfloat16_t>& qUBLocal, AscendC::LocalTensor<bfloat16_t>& kUBLocal,
    AscendC::LocalTensor<float>& gUBLocal, const AscendC::GlobalTensor<bfloat16_t>& qGlobal,
    const AscendC::GlobalTensor<bfloat16_t>& kGlobal, const AscendC::GlobalTensor<float>& gGlobal, uint64_t tokenOffset,
    uint32_t validLen)
{
    using namespace AscendC;
    CopyGmToUbRows(qUBLocal, qGlobal[tokenOffset], validLen, HEAD_DIM, HEAD_DIM);
    CopyGmToUbRows(kUBLocal, kGlobal[tokenOffset], validLen, HEAD_DIM, HEAD_DIM);
    CopyGmToUbRows(gUBLocal, gGlobal[tokenOffset], validLen, HEAD_DIM, HEAD_DIM);
}

__aicore__ inline void CopyPrepareBeta(
    AscendC::LocalTensor<bfloat16_t>& betaUBLocal, const AscendC::GlobalTensor<bfloat16_t>& betaGlobal,
    uint64_t betaOffset, uint32_t validLen)
{
    using namespace AscendC;
    DataCopyExtParams betaParams;
    betaParams.blockCount = 1;
    betaParams.blockLen = validLen * sizeof(bfloat16_t);
    betaParams.srcStride = 0;
    betaParams.dstStride = 0;
    DataCopyPadExtParams<bfloat16_t> betaPadParams;
    betaPadParams.isPad = true;
    betaPadParams.leftPadding = 0;
    constexpr uint32_t BF16_ELEMS_PER_BLOCK = 32 / sizeof(bfloat16_t);
    betaPadParams.rightPadding =
        static_cast<uint8_t>((BF16_ELEMS_PER_BLOCK - validLen % BF16_ELEMS_PER_BLOCK) % BF16_ELEMS_PER_BLOCK);
    betaPadParams.paddingValue = static_cast<bfloat16_t>(0);
    DataCopyPad(betaUBLocal, betaGlobal[betaOffset], betaParams, betaPadParams);
}

__aicore__ inline void CopyPrepareNdToNzGroupsToL1(
    AscendC::LocalTensor<bfloat16_t>& dstL1Local, const AscendC::LocalTensor<bfloat16_t>& srcUBLocal,
    uint32_t matrixCols, uint32_t firstGroup, uint32_t groupCount)
{
    using namespace AscendC;
    DataCopyParams params(CHUNK_SIZE, 1, matrixCols / CUBE_BLOCK - 1, 0);
    for (uint16_t group = 0; group < groupCount; ++group) {
        const uint32_t globalGroup = firstGroup + group;
        DataCopy(dstL1Local[globalGroup * CHUNK_SIZE * CUBE_BLOCK], srcUBLocal[globalGroup * CUBE_BLOCK], params);
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
        GlobalTensor<bfloat16_t> qPlusGlobal, kTailGlobal, aGlobal;
        GlobalTensor<float> gLastGlobal;
        qGlobal.SetGlobalBuffer(qGMAddr);
        kGlobal.SetGlobalBuffer(kGMAddr);
        gGlobal.SetGlobalBuffer(gGMAddr);
        betaGlobal.SetGlobalBuffer(betaGMAddr);
        qPlusGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.qPlusOffset));
        kTailGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.kTailOffset));
        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceGMAddr + data.aOffset));
        gLastGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGMAddr + data.gLastOffset));

        LocalTensor<bfloat16_t> mL1Local(TPosition::A1, PREP_M_L1_ADDR, PREP_M_L1_ELEMS);
        LocalTensor<bfloat16_t> kPlusL1Local(TPosition::A1, PREP_K_PLUS_L1_ADDR, PREP_K_PLUS_L1_ELEMS);
        LocalTensor<bfloat16_t> qUBLocal(TPosition::VECCALC, PREP_Q_BF16_UB_ADDR, CHUNK_D_ELEMS);
        LocalTensor<bfloat16_t> kUBLocal(TPosition::VECCALC, PREP_K_BF16_UB_ADDR, CHUNK_D_ELEMS);
        LocalTensor<float> gUBLocal(TPosition::VECCALC, PREP_G_UB_ADDR, CHUNK_D_ELEMS);
        LocalTensor<bfloat16_t> qPlusUBLocal(TPosition::VECCALC, PREP_Q_PLUS_BF16_UB_ADDR, CHUNK_D_ELEMS);
        LocalTensor<bfloat16_t> kPlusUBLocal(TPosition::VECCALC, PREP_K_PLUS_BF16_UB_ADDR, CHUNK_D_ELEMS);
        LocalTensor<bfloat16_t> kTailUBLocal(TPosition::VECCALC, PREP_K_TAIL_BF16_UB_ADDR, CHUNK_D_ELEMS);
        LocalTensor<bfloat16_t> betaUBLocal(TPosition::VECCALC, PREP_BETA_BF16_UB_ADDR, CHUNK_SIZE);
        LocalTensor<float> mFp32UBLocal(TPosition::VECCALC, PREP_M_FP32_UB_ADDR, CHUNK_C_ELEMS);
        LocalTensor<bfloat16_t> mUBLocal(TPosition::VECCALC, PREP_M_BF16_UB_ADDR, CHUNK_C_ELEMS);
        LocalTensor<bfloat16_t> aUBLocal(TPosition::VECCALC, PREP_A_BF16_UB_ADDR, CHUNK_C_ELEMS);
        LocalTensor<float> cumulativeGUBLocal(TPosition::VECCALC, PREP_CUM_G_FP32_UB_ADDR, CHUNK_D_ELEMS);

        const uint32_t aivIdx = GetBlockIdx();
        const uint32_t subAivIdx = GetSubBlockIdx();
        const uint32_t aicIdx = aivIdx / GetSubBlockNum();
        constexpr uint32_t HALF_D = HEAD_DIM / 2;

        for (uint32_t taskId = aicIdx; taskId < data.prepareNumTasks; taskId += data.prepareUseAicNum) {
            const uint32_t batchId = taskId / data.chunkCount;
            const uint32_t chunkId = taskId % data.chunkCount;
            const uint32_t firstToken = chunkId * CHUNK_SIZE;
            const uint32_t validLen = data.seqLen - firstToken < CHUNK_SIZE ? data.seqLen - firstToken : CHUNK_SIZE;
            const uint64_t tokenOffset = (static_cast<uint64_t>(batchId) * data.seqLen + firstToken) * HEAD_DIM;
            const uint64_t betaOffset = static_cast<uint64_t>(batchId) * data.seqLen + firstToken;
            const uint64_t chunkOffset = static_cast<uint64_t>(taskId) * CHUNK_D_ELEMS;
            const uint64_t halfOffset = chunkOffset + static_cast<uint64_t>(subAivIdx) * HALF_D;

            Mutex::Lock<PIPE_V>(MUTEX_PREP_INPUT_UB);
            Duplicate<uint16_t>(qUBLocal.ReinterpretCast<uint16_t>(), 0, CHUNK_D_ELEMS);
            Duplicate<uint16_t>(kUBLocal.ReinterpretCast<uint16_t>(), 0, CHUNK_D_ELEMS);
            Duplicate<float>(gUBLocal, 0.0F, CHUNK_D_ELEMS);
            if (subAivIdx == 0) {
                Duplicate<uint16_t>(betaUBLocal.ReinterpretCast<uint16_t>(), 0, CHUNK_SIZE);
            }
            Mutex::Unlock<PIPE_V>(MUTEX_PREP_INPUT_UB);

            Mutex::Lock<PIPE_MTE2>(MUTEX_PREP_INPUT_UB);
            CopyPrepareInputs(qUBLocal, kUBLocal, gUBLocal, qGlobal, kGlobal, gGlobal, tokenOffset, validLen);
            if (subAivIdx == 0) {
                CopyPrepareBeta(betaUBLocal, betaGlobal, betaOffset, validLen);
            }
            Mutex::Unlock<PIPE_MTE2>(MUTEX_PREP_INPUT_UB);

            Mutex::Lock<PIPE_V>(MUTEX_PREP_INPUT_UB);
            asc_vf_call<PrepareTransformsVF>(
                reinterpret_cast<__ubuf__ bfloat16_t*>(qUBLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ bfloat16_t*>(kUBLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(gUBLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(cumulativeGUBLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ bfloat16_t*>(qPlusUBLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ bfloat16_t*>(kPlusUBLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ bfloat16_t*>(kTailUBLocal.GetPhyAddr()));
            if (subAivIdx == 0) {
                asc_vf_call<PreparePairASolveMVF>(
                    reinterpret_cast<__ubuf__ bfloat16_t*>(qUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ bfloat16_t*>(kUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(cumulativeGUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ bfloat16_t*>(betaUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(mFp32UBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ bfloat16_t*>(mUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ bfloat16_t*>(aUBLocal.GetPhyAddr()));
            }
            Mutex::Unlock<PIPE_V>(MUTEX_PREP_INPUT_UB);

            Mutex::Lock<PIPE_MTE3>(MUTEX_PREP_INPUT_UB);
            constexpr uint32_t K_PLUS_GROUPS_PER_AIV = HEAD_DIM / CUBE_BLOCK / 2;
            CopyPrepareNdToNzGroupsToL1(
                kPlusL1Local, kPlusUBLocal, HEAD_DIM, subAivIdx * K_PLUS_GROUPS_PER_AIV, K_PLUS_GROUPS_PER_AIV);
            if (subAivIdx == 0) {
                CopyPrepareNdToNzGroupsToL1(mL1Local, mUBLocal, CHUNK_SIZE, 0, CHUNK_SIZE / CUBE_BLOCK);
            }
            // INPUT_READY 只表示 M/K_plus 已写入共享 L1. 后续 workspace 写入与 AIC
            // 的 MTE1/Cube 无依赖, 可继续在同一条 MTE3 流水上排队.
            SetAivToAic<PIPE_MTE3>(FLAG_PREP_INPUT_READY);
            if (subAivIdx == 0) {
                DataCopy(aGlobal[static_cast<uint64_t>(taskId) * CHUNK_C_ELEMS], aUBLocal, CHUNK_C_ELEMS);
            }
            CopyUbToGmRows(
                qPlusGlobal[halfOffset], qPlusUBLocal[subAivIdx * HALF_D], CHUNK_SIZE, HALF_D, HEAD_DIM, HEAD_DIM);
            CopyUbToGmRows(
                kTailGlobal[halfOffset], kTailUBLocal[subAivIdx * HALF_D], CHUNK_SIZE, HALF_D, HEAD_DIM, HEAD_DIM);
            CopyUbToGmRows(
                gLastGlobal[static_cast<uint64_t>(taskId) * HEAD_DIM + subAivIdx * HALF_D],
                cumulativeGUBLocal[(CHUNK_SIZE - 1) * HEAD_DIM + subAivIdx * HALF_D], 1, HALF_D, HALF_D, HALF_D);
            Mutex::Unlock<PIPE_MTE3>(MUTEX_PREP_INPUT_UB);
            // AIC 在第二次 L1->L0 搬运完成后释放共享 L1; 等待放在 MTE3 流水,
            // 不阻塞 Scalar 指令分发.
            WaitAicToAiv<PIPE_MTE3>(FLAG_PREP_L1_FREE);
        }
    }
}

} // namespace KDALite
