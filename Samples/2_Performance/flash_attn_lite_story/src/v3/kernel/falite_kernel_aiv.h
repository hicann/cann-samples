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

// V_MTE3 的 ID 0/1 用于 PWork 双槽, ID 2 用于最终输出.
constexpr AscendC::TEventID EVENT_V_MTE3_OUTPUT_READY = 2;
constexpr AscendC::TEventID EVENT_MTE3_V_OUTPUT_DONE = 0;

// ZERO/ONE layout 分别写 BF16 寄存器的低半段和高半段, 随后用 Or 合并.
static constexpr AscendC::Reg::CastTrait castTraitZero = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};
static constexpr AscendC::Reg::CastTrait castTraitOne = {
    AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};

// 输入 Sᵀ[Bc, halfBr] 使用 DN 布局, 每行占 1 个 64-lane FP32 寄存器.
// 沿 Bc 维逐 lane 计算 max/sum; 每个 lane 对应本 AIV 的 1 个 Q 行.
template <bool IS_FIRST_ITER>
__simd_vf__ inline void OnlineColwiseSoftmaxVF(
    __ubuf__ float* sUBAddr, __ubuf__ float* mUBAddr, __ubuf__ float* lUBAddr, __ubuf__ float* alphaUBAddr,
    const uint16_t halfBr, const uint16_t bc, const float scale)
{
    using namespace AscendC;
    Reg::RegTensor<float> s, maxReg, mOld, alpha, expReg, sumReg, lOld;
    Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Duplicate(maxReg, FLOAT_LOWEST, all);
    for (uint16_t r = 0; r < bc; ++r) {
        Reg::LoadAlign(s, sUBAddr + r * halfBr);
        Reg::Muls(s, s, scale, all);
        Reg::Max(maxReg, maxReg, s, all);
    }
    if constexpr (!IS_FIRST_ITER) {
        Reg::LoadAlign(mOld, mUBAddr);
        Reg::Max(maxReg, maxReg, mOld, all);
        Reg::Sub(alpha, mOld, maxReg, all);
        Reg::Exp(alpha, alpha, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(alphaUBAddr, alpha, all);
    }
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(mUBAddr, maxReg, all);

    Reg::Duplicate(sumReg, 0.0f, all);
    for (uint16_t r = 0; r < bc; ++r) {
        Reg::LoadAlign(s, sUBAddr + r * halfBr);
        Reg::Muls(s, s, scale, all);
        Reg::FusedExpSub(expReg, s, maxReg, all);
        Reg::Add(sumReg, sumReg, expReg, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(sUBAddr + r * halfBr, expReg, all);
    }
    if constexpr (IS_FIRST_ITER) {
        Reg::Duplicate(alpha, 1.0f, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(alphaUBAddr, alpha, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(lUBAddr, sumReg, all);
    } else {
        Reg::LoadAlign(lOld, lUBAddr);
        Reg::Mul(lOld, lOld, alpha, all);
        Reg::Add(lOld, lOld, sumReg, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(lUBAddr, lOld, all);
    }
}

// 将 FP32 DN Pᵀ[Bc, col] 转为带 32B 组间 padding 的 BF16 NZ.
// col 为 64 的倍数; 每次处理 64 个 Q 列, Cast 后用 Pack 压紧 BF16 数据.
__simd_vf__ inline void FusedDNToNZCastVF(
    __ubuf__ bfloat16_t* pDstAddr, __ubuf__ float* pSrcAddr, const uint16_t col, const uint16_t row)
{
    using namespace AscendC;

    Reg::RegTensor<float> fp32;
    Reg::RegTensor<bfloat16_t> bf16, packed;

    Reg::MaskReg b32AllMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg b16HalfMask = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::H>();
    // 每个 16 列 NZ 分组含 row 个有效 DataBlock, 末尾保留 1 个 padding block.
    const uint32_t groupStrideBlocks = row + 1;

    __ubuf__ bfloat16_t* dstGroupAddr;
    for (uint16_t qBase = 0; qBase < col; qBase += VL_B32) {
        dstGroupAddr = pDstAddr + (qBase / B16_PER_DATABLOCK) * groupStrideBlocks * B16_PER_DATABLOCK;
        for (uint16_t k = 0; k < row; ++k) {
            Reg::LoadAlign(fp32, pSrcAddr + k * col + qBase);
            Reg::Cast<bfloat16_t, float, castTraitZero>(bf16, fp32, b32AllMask);
            Reg::Pack(
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(packed), reinterpret_cast<Reg::RegTensor<uint32_t>&>(bf16));
            // 当前 k 行向 4 个 16 列分组各写 1 个 DataBlock.
            // POST_MODE_UPDATE 将各分组地址推进到下一 k 行.
            Reg::StoreAlign<bfloat16_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                dstGroupAddr, packed, groupStrideBlocks, 1, b16HalfMask);
        }
    }
}

// 计算 OAcc = alpha * OAcc + DeltaO. OAcc 初始为 0, 无需单独处理首轮.
// alpha 通过 BRC 按行广播.
__simd_vf__ inline void OnlineUpdateVF(
    __ubuf__ float* oAccUBAddr, __ubuf__ float* oDeltaUBAddr, __ubuf__ float* alphaUBAddr, const uint16_t halfBr,
    const uint16_t d)
{
    using namespace AscendC;
    Reg::RegTensor<float> alpha, pre, cur, mul, add;
    Reg::MaskReg pregAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    constexpr uint16_t floatRepSize = VL_B32;
    const uint16_t dLoops = d / floatRepSize;
    for (uint16_t i = 0; i < halfBr; ++i) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(alpha, alphaUBAddr + i);
        for (uint16_t j = 0; j < dLoops; ++j) {
            Reg::LoadAlign(pre, oAccUBAddr + i * d + j * floatRepSize);
            Reg::LoadAlign(cur, oDeltaUBAddr + i * d + j * floatRepSize);
            Reg::Mul(mul, alpha, pre, pregAll);
            Reg::Add(add, mul, cur, pregAll);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(oAccUBAddr + i * d + j * floatRepSize, add, pregAll);
        }
    }
}

// 计算 OAcc / l, 并转为 DataCopy 写回 GM 所需的 BF16 ND 布局.
// l 通过 BRC 按行广播; DINTLV_B32 每次读取 2 个 FP32 寄存器.
__simd_vf__ inline void FusedDivCastVF(
    __ubuf__ bfloat16_t* dstUBAddr, __ubuf__ float* srcUBAddr, __ubuf__ float* lUBAddr, const uint16_t halfBr,
    const uint16_t d)
{
    using namespace AscendC;
    Reg::RegTensor<float> lReg, srcEven, srcOdd, divEven, divOdd;
    Reg::RegTensor<bfloat16_t> castEven, castOdd, packed;
    Reg::MaskReg pregAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregAllB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    constexpr uint16_t dintlvLen = VL_B32 * 2;
    const uint16_t dLoops = d / dintlvLen;
    for (uint16_t i = 0; i < halfBr; ++i) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(lReg, lUBAddr + i);
        for (uint16_t c = 0; c < dLoops; ++c) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(srcEven, srcOdd, srcUBAddr + i * d + c * dintlvLen);
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

// V1: online softmax, 并将 Pᵀ 从 FP32 DN 转为 BF16 NZ.
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
            asc_vf_call<OnlineColwiseSoftmaxVF<true>>(
                sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, static_cast<uint16_t>(data.br / 2),
                static_cast<uint16_t>(data.bc), data.scale);
        } else {
            asc_vf_call<OnlineColwiseSoftmaxVF<false>>(
                sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, static_cast<uint16_t>(data.br / 2),
                static_cast<uint16_t>(data.bc), data.scale);
        }
        asc_vf_call<FusedDNToNZCastVF>(
            reinterpret_cast<__ubuf__ bfloat16_t*>(pWorkUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(sUBLocal.GetPhyAddr()), static_cast<uint16_t>(data.br / 2),
            static_cast<uint16_t>(data.bc));
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

// V2: 更新 OAcc = alpha * OAcc + DeltaO.
__aicore__ inline void VectorStage2(
    AscendC::LocalTensor<float>& oAccUBLocal, AscendC::LocalTensor<float>& oDeltaUBLocal,
    AscendC::LocalTensor<float>& alphaUBLocal, const FlashAttnLiteTilingData& data)
{
    using namespace AscendC;

    if ASCEND_IS_AIV {
        const uint32_t halfBr = data.br / 2;
        asc_vf_call<OnlineUpdateVF>(
            reinterpret_cast<__ubuf__ float*>(oAccUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(oDeltaUBLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(alphaUBLocal.GetPhyAddr()), static_cast<uint16_t>(halfBr),
            static_cast<uint16_t>(HEAD_DIM));
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
        // UB 按 S/DeltaO/OAcc/PWork/m/l/alpha 排布.
        LocalTensor<float> sUBLocal(TPosition::VECCALC, aiv.sUBAddr, aiv.sUBElems);
        LocalTensor<float> oDeltaUBLocal(TPosition::VECCALC, aiv.oDeltaUBAddr, aiv.oDeltaUBElems);
        LocalTensor<float> oAccUBLocal(TPosition::VECCALC, aiv.oAccUBAddr, aiv.oAccUBElems);
        // PWork 双槽保存带 padding 的 NZ P, 循环结束后复用 slot 0 保存 ND O.
        LocalTensor<bfloat16_t> pWorkUBLocal(TPosition::VECCALC, aiv.pWorkUBAddr, aiv.pWorkUBElems);
        LocalTensor<float> mUBLocal(TPosition::VECCALC, aiv.mUBAddr, aiv.rowStatsUBElems);
        LocalTensor<float> lUBLocal(TPosition::VECCALC, aiv.lUBAddr, aiv.rowStatsUBElems);
        LocalTensor<float> alphaUBLocal(TPosition::VECCALC, aiv.alphaUBAddr, PIPELINE_SLOT_NUM * aiv.rowStatsUBElems);

        for (uint32_t taskId = aicIdx; taskId < data.numTasks; taskId += data.useAicNum) {
            const uint32_t batchIdx = taskId / data.tr;
            const uint32_t tileIdx = taskId % data.tr;
            const uint64_t outGMOffset = static_cast<uint64_t>(batchIdx) * data.seqLen * d +
                                         static_cast<uint64_t>(tileIdx) * qTileElements +
                                         static_cast<uint64_t>(subAivIdx) * outputHalfElements;

            Duplicate<float>(oAccUBLocal, 0.0f, outputHalfElements);
            Duplicate<float>(mUBLocal, FLOAT_LOWEST, halfBr);
            Duplicate<float>(lUBLocal, 0.0f, halfBr);

            for (uint32_t groupStart = 0; groupStart < data.tc; groupStart += PIPELINE_SLOT_NUM) {
                const uint32_t groupEnd =
                    groupStart + PIPELINE_SLOT_NUM < data.tc ? groupStart + PIPELINE_SLOT_NUM : data.tc;

                // 两路 AIV 执行相同流程: 先按 slot 顺序消费本组 S, 并写出 P.
                for (uint32_t j = groupStart; j < groupEnd; ++j) {
                    const uint32_t slot = j % PIPELINE_SLOT_NUM;
                    WaitAicToAiv<PIPE_V>(SlotFlagId(FLAG_S_READY_BASE, slot));
                    auto sSlotUBLocal = sUBLocal[static_cast<uint64_t>(slot) * sTileElements];
                    auto alphaSlotUBLocal = alphaUBLocal[static_cast<uint64_t>(slot) * halfBr];
                    auto pWorkSlotUBLocal = pWorkUBLocal[static_cast<uint64_t>(slot) * pWorkTileElements];
                    VectorStage1(sSlotUBLocal, mUBLocal, lUBLocal, alphaSlotUBLocal, pWorkSlotUBLocal, data, j);

                    const TEventID pWorkSlotEventId = static_cast<TEventID>(slot);
                    SetWaitFlag<HardEvent::V_MTE3>(pWorkSlotEventId);
                    auto pSliceL1Local = pL1Local
                        [static_cast<uint64_t>(slot) * pTileElements +
                         static_cast<uint64_t>(subAivIdx) * pHalfElements];
                    CopyPWorkToL1(pSliceL1Local, pWorkSlotUBLocal, halfBr, bc);
                    // P_READY 绑定 PIPE_MTE3, 在 P 写入 L1 后生效.
                    SetAivToAic<PIPE_MTE3>(SlotFlagId(FLAG_P_READY_BASE, slot));
                }

                // 再按相同 slot 顺序处理 DeltaO 和 alpha.
                for (uint32_t j = groupStart; j < groupEnd; ++j) {
                    const uint32_t slot = j % PIPELINE_SLOT_NUM;
                    WaitAicToAiv<PIPE_V>(SlotFlagId(FLAG_O_READY_BASE, slot));
                    auto oDeltaSlotUBLocal = oDeltaUBLocal[static_cast<uint64_t>(slot) * oDeltaTileElements];
                    auto alphaSlotUBLocal = alphaUBLocal[static_cast<uint64_t>(slot) * halfBr];
                    VectorStage2(oAccUBLocal, oDeltaSlotUBLocal, alphaSlotUBLocal, data);
                }
            }
            // 复用 PWork 保存归一化和 Cast 后的输出, 再写回 GM.
            asc_vf_call<FusedDivCastVF>(
                reinterpret_cast<__ubuf__ bfloat16_t*>(pWorkUBLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(oAccUBLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(lUBLocal.GetPhyAddr()), static_cast<uint16_t>(halfBr),
                static_cast<uint16_t>(d));
            SetWaitFlag<HardEvent::V_MTE3>(EVENT_V_MTE3_OUTPUT_READY);
            DataCopy(outGlobal[outGMOffset], pWorkUBLocal, outputHalfElements);
            // MTE3 读完 PWork 后, 下一 task 才能覆写该空间.
            SetWaitFlag<HardEvent::MTE3_V>(EVENT_MTE3_V_OUTPUT_DONE);
        }
    }
}

} // namespace FALite
