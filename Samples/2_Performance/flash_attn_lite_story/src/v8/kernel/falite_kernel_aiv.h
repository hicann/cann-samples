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

#include "falite_kernel_common.h"

namespace FALite {

// AIV0/AIV1 是独立核心, 各自在本地使用同一组 MutexID.
constexpr MutexId MUTEX_P_WORK_UB_BASE = 0;
constexpr MutexId MUTEX_OUTPUT_UB_BASE = MUTEX_P_WORK_UB_BASE + DB_SLOT_NUM;

// ZERO/ONE layout 分别写 BF16 寄存器的低半段和高半段, 随后用 Or 合并.
static constexpr AscendC::Reg::CastTrait castTraitZero = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};
static constexpr AscendC::Reg::CastTrait castTraitOne = {
    AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};

// 输入 Sᵀ[Bc, halfBr] 使用 DN 布局, 每行占 1 个 64-lane FP32 寄存器.
// 第二遍 Softmax 直接将 exp Cast/Pack 到带 padding 的 BF16 NZ PWork.
template <bool IS_FIRST_ITER>
__simd_vf__ inline void OnlineSoftmaxCastPackVF(
    __ubuf__ bfloat16_t* pWorkUBAddr, __ubuf__ float* sUBAddr, __ubuf__ float* mUBAddr, __ubuf__ float* lUBAddr,
    __ubuf__ float* alphaUBAddr, const uint16_t halfBr, const uint16_t bc, const float scale)
{
    using namespace AscendC;
    Reg::RegTensor<float> s0, s1, s2, s3;
    Reg::RegTensor<float> max0, max1, max2, max3;
    Reg::RegTensor<float> mOld, alpha, lOld;
    Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    Reg::Duplicate(max0, FLOAT_LOWEST, all);
    Reg::Duplicate(max1, FLOAT_LOWEST, all);
    Reg::Duplicate(max2, FLOAT_LOWEST, all);
    Reg::Duplicate(max3, FLOAT_LOWEST, all);
    for (uint16_t r = 0; r < bc; r += 4) {
        Reg::LoadAlign(s0, sUBAddr + r * halfBr);
        Reg::LoadAlign(s1, sUBAddr + (r + 1) * halfBr);
        Reg::LoadAlign(s2, sUBAddr + (r + 2) * halfBr);
        Reg::LoadAlign(s3, sUBAddr + (r + 3) * halfBr);
        Reg::Muls(s0, s0, scale, all);
        Reg::Muls(s1, s1, scale, all);
        Reg::Muls(s2, s2, scale, all);
        Reg::Muls(s3, s3, scale, all);
        Reg::Max(max0, max0, s0, all);
        Reg::Max(max1, max1, s1, all);
        Reg::Max(max2, max2, s2, all);
        Reg::Max(max3, max3, s3, all);
    }
    Reg::Max(max0, max0, max1, all);
    Reg::Max(max2, max2, max3, all);
    Reg::Max(max0, max0, max2, all);
    if constexpr (!IS_FIRST_ITER) {
        Reg::LoadAlign(mOld, mUBAddr);
        Reg::Max(max0, max0, mOld, all);
        Reg::Sub(alpha, mOld, max0, all);
        Reg::Exp(alpha, alpha, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(alphaUBAddr, alpha, all);
    }
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(mUBAddr, max0, all);

    Reg::RegTensor<float> exp0, exp1, exp2, exp3;
    Reg::RegTensor<float> sum0, sum1, sum2, sum3;
    Reg::RegTensor<bfloat16_t> bf16_0, bf16_1, bf16_2, bf16_3;
    Reg::RegTensor<bfloat16_t> packed0, packed1, packed2, packed3;
    Reg::MaskReg b16HalfMask = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::H>();
    Reg::Duplicate(sum0, 0.0f, all);
    Reg::Duplicate(sum1, 0.0f, all);
    Reg::Duplicate(sum2, 0.0f, all);
    Reg::Duplicate(sum3, 0.0f, all);

    // 4 个 16 列 NZ 分组各保存 bc 个 DataBlock, 末尾保留 1 个 padding block.
    const uint32_t groupStrideBlocks = bc + 1;
    __ubuf__ bfloat16_t* pDstAddr = pWorkUBAddr;
    for (uint16_t r = 0; r < bc; r += 4) {
        Reg::LoadAlign(s0, sUBAddr + r * halfBr);
        Reg::LoadAlign(s1, sUBAddr + (r + 1) * halfBr);
        Reg::LoadAlign(s2, sUBAddr + (r + 2) * halfBr);
        Reg::LoadAlign(s3, sUBAddr + (r + 3) * halfBr);
        Reg::Muls(s0, s0, scale, all);
        Reg::Muls(s1, s1, scale, all);
        Reg::Muls(s2, s2, scale, all);
        Reg::Muls(s3, s3, scale, all);
        Reg::FusedExpSub(exp0, s0, max0, all);
        Reg::FusedExpSub(exp1, s1, max0, all);
        Reg::FusedExpSub(exp2, s2, max0, all);
        Reg::FusedExpSub(exp3, s3, max0, all);
        Reg::Add(sum0, sum0, exp0, all);
        Reg::Add(sum1, sum1, exp1, all);
        Reg::Add(sum2, sum2, exp2, all);
        Reg::Add(sum3, sum3, exp3, all);

        Reg::Cast<bfloat16_t, float, castTraitZero>(bf16_0, exp0, all);
        Reg::Cast<bfloat16_t, float, castTraitZero>(bf16_1, exp1, all);
        Reg::Cast<bfloat16_t, float, castTraitZero>(bf16_2, exp2, all);
        Reg::Cast<bfloat16_t, float, castTraitZero>(bf16_3, exp3, all);
        Reg::Pack(
            reinterpret_cast<Reg::RegTensor<uint16_t>&>(packed0), reinterpret_cast<Reg::RegTensor<uint32_t>&>(bf16_0));
        Reg::Pack(
            reinterpret_cast<Reg::RegTensor<uint16_t>&>(packed1), reinterpret_cast<Reg::RegTensor<uint32_t>&>(bf16_1));
        Reg::Pack(
            reinterpret_cast<Reg::RegTensor<uint16_t>&>(packed2), reinterpret_cast<Reg::RegTensor<uint32_t>&>(bf16_2));
        Reg::Pack(
            reinterpret_cast<Reg::RegTensor<uint16_t>&>(packed3), reinterpret_cast<Reg::RegTensor<uint32_t>&>(bf16_3));
        Reg::StoreAlign<bfloat16_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            pDstAddr, packed0, groupStrideBlocks, 1, b16HalfMask);
        Reg::StoreAlign<bfloat16_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            pDstAddr, packed1, groupStrideBlocks, 1, b16HalfMask);
        Reg::StoreAlign<bfloat16_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            pDstAddr, packed2, groupStrideBlocks, 1, b16HalfMask);
        Reg::StoreAlign<bfloat16_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            pDstAddr, packed3, groupStrideBlocks, 1, b16HalfMask);
    }

    Reg::Add(sum0, sum0, sum1, all);
    Reg::Add(sum2, sum2, sum3, all);
    Reg::Add(sum0, sum0, sum2, all);
    if constexpr (IS_FIRST_ITER) {
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(lUBAddr, sum0, all);
    } else {
        Reg::LoadAlign(lOld, lUBAddr);
        Reg::MulDstAdd(lOld, alpha, sum0, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(lUBAddr, lOld, all);
    }
}

// 首轮直接以 oDelta 初始化 OAcc; 后续使用 MulDstAdd 计算 alpha * OAcc + oDelta.
template <bool IS_FIRST_ITER>
__simd_vf__ inline void OnlineUpdateVF(
    __ubuf__ float* oAccUBAddr, __ubuf__ float* oDeltaUBAddr, __ubuf__ float* alphaUBAddr, const uint16_t halfBr,
    const uint16_t d)
{
    using namespace AscendC;
    Reg::RegTensor<float> alpha0, alpha1;
    Reg::RegTensor<float> oAcc00, oAcc01, oAcc10, oAcc11;
    Reg::RegTensor<float> oDelta00, oDelta01, oDelta10, oDelta11;
    Reg::MaskReg pregAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    for (uint16_t i = 0; i < halfBr; i += 2) {
        const uint32_t row0Offset = i * d;
        const uint32_t row1Offset = (i + 1) * d;
        if constexpr (IS_FIRST_ITER) {
            Reg::LoadAlign(oAcc00, oDeltaUBAddr + row0Offset);
            Reg::LoadAlign(oAcc01, oDeltaUBAddr + row0Offset + VL_B32);
            Reg::LoadAlign(oAcc10, oDeltaUBAddr + row1Offset);
            Reg::LoadAlign(oAcc11, oDeltaUBAddr + row1Offset + VL_B32);
        } else {
            Reg::LoadAlign(oDelta00, oDeltaUBAddr + row0Offset);
            Reg::LoadAlign(oDelta01, oDeltaUBAddr + row0Offset + VL_B32);
            Reg::LoadAlign(oDelta10, oDeltaUBAddr + row1Offset);
            Reg::LoadAlign(oDelta11, oDeltaUBAddr + row1Offset + VL_B32);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(alpha0, alphaUBAddr + i);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(alpha1, alphaUBAddr + i + 1);
            Reg::LoadAlign(oAcc00, oAccUBAddr + row0Offset);
            Reg::LoadAlign(oAcc01, oAccUBAddr + row0Offset + VL_B32);
            Reg::LoadAlign(oAcc10, oAccUBAddr + row1Offset);
            Reg::LoadAlign(oAcc11, oAccUBAddr + row1Offset + VL_B32);
            Reg::MulDstAdd(oAcc00, alpha0, oDelta00, pregAll);
            Reg::MulDstAdd(oAcc01, alpha0, oDelta01, pregAll);
            Reg::MulDstAdd(oAcc10, alpha1, oDelta10, pregAll);
            Reg::MulDstAdd(oAcc11, alpha1, oDelta11, pregAll);
        }
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(oAccUBAddr + row0Offset, oAcc00, pregAll);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(oAccUBAddr + row0Offset + VL_B32, oAcc01, pregAll);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(oAccUBAddr + row1Offset, oAcc10, pregAll);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(oAccUBAddr + row1Offset + VL_B32, oAcc11, pregAll);
    }
}

// 原地计算 OAcc / l, 并将 FP32 OAcc 压缩为 DataCopy 所需的 BF16 ND 布局.
// 输出占据同一槽的前半段; 按行正序处理时只覆盖已经读取的 FP32 数据.
__simd_vf__ inline void FusedDivCastInplaceVF(
    __ubuf__ float* oAccUBAddr, __ubuf__ float* lUBAddr, const uint16_t halfBr, const uint16_t d)
{
    using namespace AscendC;
    __ubuf__ bfloat16_t* dstUBAddr = reinterpret_cast<__ubuf__ bfloat16_t*>(oAccUBAddr);
    Reg::RegTensor<float> lReg, srcEven, srcOdd, divEven, divOdd;
    Reg::RegTensor<bfloat16_t> castEven, castOdd, packed;
    Reg::MaskReg pregAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregAllB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    constexpr uint16_t dintlvLen = VL_B32 * 2;
    const uint16_t dLoops = d / dintlvLen;
    for (uint16_t i = 0; i < halfBr; ++i) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(lReg, lUBAddr + i);
        for (uint16_t c = 0; c < dLoops; ++c) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(srcEven, srcOdd, oAccUBAddr + i * d + c * dintlvLen);
            Reg::Div(divEven, srcEven, lReg, pregAll);
            Reg::Div(divOdd, srcOdd, lReg, pregAll);
            Reg::Cast<bfloat16_t, float, castTraitZero>(castEven, divEven, pregAllB16);
            Reg::Cast<bfloat16_t, float, castTraitOne>(castOdd, divOdd, pregAllB16);
            Reg::Or(
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(packed),
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(castEven),
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(castOdd), pregAllB16);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_NORM_B32>(
                dstUBAddr + i * d + c * dintlvLen, packed, pregAllB16);
        }
    }
}

// V1: 在同一次 VF 内完成 online softmax 和 Pᵀ 的 FP32 DN→BF16 NZ.
__aicore__ inline void VectorStage1(
    AscendC::LocalTensor<float>& sUBLocal, AscendC::LocalTensor<float>& mUBLocal, AscendC::LocalTensor<float>& lUBLocal,
    AscendC::LocalTensor<float>& alphaUBLocal, AscendC::LocalTensor<bfloat16_t>& pWorkUBLocal,
    const FlashAttnLiteTilingData& data, uint32_t j)
{
    using namespace AscendC;

    if ASCEND_IS_AIV {
        __ubuf__ float* sUBAddr = reinterpret_cast<__ubuf__ float*>(sUBLocal.GetPhyAddr());
        __ubuf__ float* mUBAddr = reinterpret_cast<__ubuf__ float*>(mUBLocal.GetPhyAddr());
        __ubuf__ float* lUBAddr = reinterpret_cast<__ubuf__ float*>(lUBLocal.GetPhyAddr());
        __ubuf__ float* alphaUBAddr = reinterpret_cast<__ubuf__ float*>(alphaUBLocal.GetPhyAddr());
        if (j == 0) {
            asc_vf_call<OnlineSoftmaxCastPackVF<true>>(
                reinterpret_cast<__ubuf__ bfloat16_t*>(pWorkUBLocal.GetPhyAddr()), sUBAddr, mUBAddr, lUBAddr,
                alphaUBAddr, static_cast<uint16_t>(data.br / 2), static_cast<uint16_t>(data.bc), data.scale);
        } else {
            asc_vf_call<OnlineSoftmaxCastPackVF<false>>(
                reinterpret_cast<__ubuf__ bfloat16_t*>(pWorkUBLocal.GetPhyAddr()), sUBAddr, mUBAddr, lUBAddr,
                alphaUBAddr, static_cast<uint16_t>(data.br / 2), static_cast<uint16_t>(data.bc), data.scale);
        }
    }
}

__aicore__ inline void CopyPWorkToL1(
    const AscendC::LocalTensor<bfloat16_t>& dstL1Local, const AscendC::LocalTensor<bfloat16_t>& srcUBLocal,
    uint32_t hBr, uint32_t bc)
{
    using namespace AscendC;
    const uint16_t blockCount = static_cast<uint16_t>(hBr / B16_PER_DATABLOCK);
    const uint16_t blockLen = static_cast<uint16_t>(bc);
    // 每个 16 列分组搬运 Bc 个 DataBlock, 并跳过源端的 padding block.
    DataCopyParams copyParams(blockCount, blockLen, 1, 0);
    DataCopy(dstL1Local, srcUBLocal, copyParams);
}

// V2: 更新 OAcc = alpha * OAcc + oDelta.
__aicore__ inline void VectorStage2(
    AscendC::LocalTensor<float>& oAccUBLocal, AscendC::LocalTensor<float>& oDeltaUBLocal,
    AscendC::LocalTensor<float>& alphaUBLocal, const FlashAttnLiteTilingData& data, uint32_t j)
{
    using namespace AscendC;

    if ASCEND_IS_AIV {
        const uint32_t halfBr = data.br / 2;
        if (j == 0) {
            asc_vf_call<OnlineUpdateVF<true>>(
                reinterpret_cast<__ubuf__ float*>(oAccUBLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(oDeltaUBLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(alphaUBLocal.GetPhyAddr()), static_cast<uint16_t>(halfBr),
                static_cast<uint16_t>(HEAD_DIM));
        } else {
            asc_vf_call<OnlineUpdateVF<false>>(
                reinterpret_cast<__ubuf__ float*>(oAccUBLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(oDeltaUBLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(alphaUBLocal.GetPhyAddr()), static_cast<uint16_t>(halfBr),
                static_cast<uint16_t>(HEAD_DIM));
        }
    }
}

__aicore__ inline void KernelProcessForAIV(__gm__ bfloat16_t* outGMAddr, FlashAttnLiteTilingData data)
{
    using namespace AscendC;

    if ASCEND_IS_AIV {
        const uint32_t br = data.br, bc = data.bc, halfBr = br / 2;
        constexpr uint32_t d = HEAD_DIM;
        const uint32_t pTileElements = br * bc;
        const uint32_t pHalfElements = halfBr * bc;
        const uint32_t sTileElements = halfBr * bc;
        const uint32_t oDeltaTileElements = halfBr * d;
        const uint32_t pWorkTileElements = halfBr * (bc + 1);
        const uint32_t qTileElements = br * d;
        const uint32_t outputHalfElements = halfBr * d;
        GlobalTensor<bfloat16_t> outGlobal;
        outGlobal.SetGlobalBuffer(outGMAddr);

        const uint32_t aivIdx = GetBlockIdx();
        const uint32_t subAivIdx = GetSubBlockIdx();
        const uint32_t aicIdx = aivIdx / GetSubBlockNum();

        const auto& aic = data.layoutAIC;
        const auto& aiv = data.layoutAIV;
        // 两路 AIV 各写 P L1 的一半; 地址由 host tiling 规划.
        LocalTensor<bfloat16_t> pL1Local(TPosition::A1, aic.pL1Addr, aic.pL1Elems);
        // UB 按 S/oDelta/OAcc/PWork/m/l/alpha 排布.
        LocalTensor<float> sUBLocal(TPosition::VECCALC, aiv.sUBAddr, aiv.sUBElems);
        LocalTensor<float> oDeltaUBLocal(TPosition::VECCALC, aiv.oDeltaUBAddr, aiv.oDeltaUBElems);
        LocalTensor<float> oAccUBLocal(TPosition::VECCALC, aiv.oAccUBAddr, aiv.oAccUBElems);
        // BF16 输出与 FP32 OAcc 共用物理地址, 每个 I/O slot 内原地压缩.
        LocalTensor<bfloat16_t> outputUBLocal(
            TPosition::VECCALC, aiv.oAccUBAddr, aiv.oAccUBElems * sizeof(float) / sizeof(bfloat16_t));
        // PWork 双槽只保存带 padding 的 NZ P.
        LocalTensor<bfloat16_t> pWorkUBLocal(TPosition::VECCALC, aiv.pWorkUBAddr, aiv.pWorkUBElems);
        LocalTensor<float> mUBLocal(TPosition::VECCALC, aiv.mUBAddr, aiv.rowStatsUBElems);
        LocalTensor<float> lUBLocal(TPosition::VECCALC, aiv.lUBAddr, aiv.rowStatsUBElems);
        LocalTensor<float> alphaUBLocal(
            TPosition::VECCALC, aiv.alphaUBAddr, CV_PIPELINE_SLOT_NUM * aiv.rowStatsUBElems);

        // 两路 AIV 都归还同一槽后, mode2 下 AIC 才能开始写该槽.
        for (uint32_t slot = 0; slot < DB_SLOT_NUM; ++slot) {
            SetAivToAic<PIPE_V>(SlotFlagId(FLAG_S_HANDOFF_BASE, slot));
        }
        for (uint32_t slot = 0; slot < DB_SLOT_NUM; ++slot) {
            SetAivToAic<PIPE_V>(SlotFlagId(FLAG_O_DELTA_HANDOFF_BASE, slot));
        }

        uint32_t ioSlot = 0;
        for (uint32_t taskId = aicIdx; taskId < data.numTasks; taskId += data.useAicNum) {
            const uint32_t batchIdx = taskId / data.tr;
            const uint32_t tileIdx = taskId % data.tr;
            const uint64_t outGMOffset = static_cast<uint64_t>(batchIdx) * data.seqLen * d +
                                         static_cast<uint64_t>(tileIdx) * qTileElements +
                                         static_cast<uint64_t>(subAivIdx) * outputHalfElements;

            auto oAccSlotUBLocal = oAccUBLocal[static_cast<uint64_t>(ioSlot) * outputHalfElements];
            auto outputSlotUBLocal =
                outputUBLocal[static_cast<uint64_t>(ioSlot) * outputHalfElements * sizeof(float) / sizeof(bfloat16_t)];

            const MutexID outputMutexId = MUTEX_OUTPUT_UB_BASE + static_cast<MutexID>(ioSlot);
            Mutex::Lock<PIPE_V>(outputMutexId);

            for (uint32_t epoch = 0; epoch < data.tc + CV_PIPELINE_SLOT_NUM; ++epoch) {
                if (epoch >= 1) {
                    const uint32_t j = epoch - 1;
                    if (j < data.tc) {
                        const uint32_t sSlot = j % DB_SLOT_NUM;
                        const uint32_t pWorkSlot = j % DB_SLOT_NUM;
                        const uint32_t cvPipelineSlot = j % CV_PIPELINE_SLOT_NUM;
                        const MutexID pWorkMutexId = MUTEX_P_WORK_UB_BASE + static_cast<MutexID>(pWorkSlot);
                        Mutex::Lock<PIPE_V>(pWorkMutexId);
                        WaitAicToAiv<PIPE_V>(SlotFlagId(FLAG_S_HANDOFF_BASE, sSlot));
                        auto sSlotUBLocal = sUBLocal[static_cast<uint64_t>(sSlot) * sTileElements];
                        auto alphaSlotUBLocal = alphaUBLocal[static_cast<uint64_t>(cvPipelineSlot) * halfBr];
                        auto pWorkSlotUBLocal = pWorkUBLocal[static_cast<uint64_t>(pWorkSlot) * pWorkTileElements];
                        VectorStage1(sSlotUBLocal, mUBLocal, lUBLocal, alphaSlotUBLocal, pWorkSlotUBLocal, data, j);
                        SetAivToAic<PIPE_V>(SlotFlagId(FLAG_S_HANDOFF_BASE, sSlot));
                        Mutex::Unlock<PIPE_V>(pWorkMutexId);

                        Mutex::Lock<PIPE_MTE3>(pWorkMutexId);
                        auto pSliceL1Local = pL1Local
                            [static_cast<uint64_t>(cvPipelineSlot) * pTileElements +
                             static_cast<uint64_t>(subAivIdx) * pHalfElements];
                        CopyPWorkToL1(pSliceL1Local, pWorkSlotUBLocal, halfBr, bc);
                        Mutex::Unlock<PIPE_MTE3>(pWorkMutexId);
                        SetAivToAic<PIPE_MTE3>(SlotFlagId(FLAG_P_READY_BASE, cvPipelineSlot));
                    }
                }

                if (epoch >= CV_PIPELINE_SLOT_NUM) {
                    const uint32_t j = epoch - CV_PIPELINE_SLOT_NUM;
                    if (j < data.tc) {
                        const uint32_t oDeltaSlot = j % DB_SLOT_NUM;
                        const uint32_t cvPipelineSlot = j % CV_PIPELINE_SLOT_NUM;
                        WaitAicToAiv<PIPE_V>(SlotFlagId(FLAG_O_DELTA_HANDOFF_BASE, oDeltaSlot));
                        auto oDeltaSlotUBLocal = oDeltaUBLocal[static_cast<uint64_t>(oDeltaSlot) * oDeltaTileElements];
                        auto alphaSlotUBLocal = alphaUBLocal[static_cast<uint64_t>(cvPipelineSlot) * halfBr];
                        VectorStage2(oAccSlotUBLocal, oDeltaSlotUBLocal, alphaSlotUBLocal, data, j);
                        SetAivToAic<PIPE_V>(SlotFlagId(FLAG_O_DELTA_HANDOFF_BASE, oDeltaSlot));
                    }
                }
            }
            asc_vf_call<FusedDivCastInplaceVF>(
                reinterpret_cast<__ubuf__ float*>(oAccSlotUBLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(lUBLocal.GetPhyAddr()), static_cast<uint16_t>(halfBr),
                static_cast<uint16_t>(d));
            Mutex::Unlock<PIPE_V>(outputMutexId);
            Mutex::Lock<PIPE_MTE3>(outputMutexId);
            DataCopy(outGlobal[outGMOffset], outputSlotUBLocal, outputHalfElements);
            Mutex::Unlock<PIPE_MTE3>(outputMutexId);
            ioSlot ^= 1;
        }
    }
}

} // namespace FALite
