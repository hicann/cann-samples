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

// PWork 与最终输出复用同一块 UB，并用同一 Mutex 在 Vector/MTE3 间交接所有权。
constexpr MutexID MUTEX_P_WORK_UB = 0;

// ZERO/ONE layout 分别写 BF16 寄存器的低半段和高半段, 随后用 Or 合并.
static constexpr AscendC::Reg::CastTrait castTraitZero = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};
static constexpr AscendC::Reg::CastTrait castTraitOne = {
    AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};

// 输入 Sᵀ[Bc, halfBr] 使用 DN 布局, 每行占 1 个 64-lane FP32 寄存器.
// 沿 Bc 维逐 lane 计算 max/sum; 每个 lane 对应本 AIV 的 1 个 Q 行.
template <bool IS_FIRST_ITER, bool APPLY_CAUSAL_MASK>
__simd_vf__ inline void OnlineColwiseSoftmaxVF(
    __ubuf__ float* sUBAddr, __ubuf__ float* mUBAddr, __ubuf__ float* lUBAddr, __ubuf__ float* alphaUBAddr,
    const uint16_t halfBr, const uint16_t bc, const uint16_t validBc, const float scale, const uint16_t queryColBegin)
{
    using namespace AscendC;
    Reg::RegTensor<float> s, maxReg, mOld, alpha, expReg, sumReg, lOld;
    Reg::RegTensor<int32_t> queryIndex;
    Reg::RegTensor<float> maskedScore;
    Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg invalid;
    if constexpr (APPLY_CAUSAL_MASK) {
        Reg::Arange(queryIndex, static_cast<int32_t>(queryColBegin));
        Reg::Duplicate(maskedScore, FLOAT_LOWEST, all);
    }
    Reg::Duplicate(maxReg, FLOAT_LOWEST, all);
    for (uint16_t r = 0; r < validBc; ++r) {
        Reg::LoadAlign(s, sUBAddr + r * halfBr);
        Reg::Muls(s, s, scale, all);
        if constexpr (APPLY_CAUSAL_MASK) {
            Reg::CompareScalar<int32_t, CMPMODE::LT>(invalid, queryIndex, static_cast<int32_t>(r), all);
            Reg::Select(s, maskedScore, s, invalid);
        }
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
    for (uint16_t r = 0; r < validBc; ++r) {
        Reg::LoadAlign(s, sUBAddr + r * halfBr);
        Reg::Muls(s, s, scale, all);
        if constexpr (APPLY_CAUSAL_MASK) {
            Reg::CompareScalar<int32_t, CMPMODE::LT>(invalid, queryIndex, static_cast<int32_t>(r), all);
            Reg::Select(s, maskedScore, s, invalid);
        }
        Reg::FusedExpSub(expReg, s, maxReg, all);
        Reg::Add(sumReg, sumReg, expReg, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(sUBAddr + r * halfBr, expReg, all);
    }
    // C2 仍按完整 Bc 计算；不存在的 K/V 行对应权重必须为 0。
    Reg::Duplicate(expReg, 0.0f, all);
    for (uint16_t r = validBc; r < bc; ++r) {
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
    // 每个 16 列 NZ 分组固定含 row 个数据 DataBlock，末尾保留 1 个布局 padding block。
    const uint32_t groupStrideBlocks = row + 1;

    __ubuf__ bfloat16_t* dstGroupAddr;
    for (uint16_t qBase = 0; qBase < col; qBase += VL_B32) {
        dstGroupAddr = pDstAddr + (qBase / B16_PER_DATABLOCK) * groupStrideBlocks * B16_PER_DATABLOCK;
        for (uint16_t k = 0; k < row; ++k) {
            Reg::LoadAlign(fp32, pSrcAddr + k * col + qBase);
            Reg::Cast<bfloat16_t, float, castTraitZero>(bf16, fp32, b32AllMask);
            Reg::Pack(
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(packed), reinterpret_cast<Reg::RegTensor<uint32_t>&>(bf16));
            // 第 k 行向 4 个 16 列分组各写 1 个 DataBlock。
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

// V1：更新分块 Softmax 状态，并将未归一化权重 Pᵀ 从 FP32 DN 转为 BF16 NZ。
template <bool CAUSAL_MASK>
__aicore__ inline void VectorStage1(
    AscendC::LocalTensor<float>& sUBLocal, AscendC::LocalTensor<float>& mUBLocal, AscendC::LocalTensor<float>& lUBLocal,
    AscendC::LocalTensor<float>& alphaUBLocal, AscendC::LocalTensor<bfloat16_t>& pWorkUBLocal,
    const FlashAttnLiteTilingData& data, uint32_t j, uint32_t qTileIdx, uint32_t subAivIdx)
{
    using namespace AscendC;

    if ASCEND_IS_AIV {
        __ubuf__ float* sUBAddr = reinterpret_cast<__ubuf__ float*>(sUBLocal.GetPhyAddr());
        __ubuf__ float* mUBAddr = reinterpret_cast<__ubuf__ float*>(mUBLocal.GetPhyAddr());
        __ubuf__ float* lUBAddr = reinterpret_cast<__ubuf__ float*>(lUBLocal.GetPhyAddr());
        __ubuf__ float* alphaUBAddr = reinterpret_cast<__ubuf__ float*>(alphaUBLocal.GetPhyAddr());
        const uint16_t halfBr = static_cast<uint16_t>(data.br / 2);
        const uint16_t bc = static_cast<uint16_t>(data.bc);
        const uint16_t validBc = static_cast<uint16_t>(GetTileValidRows(data.seqLen, j, data.bc));
        const uint16_t queryColBegin = static_cast<uint16_t>(subAivIdx * halfBr);
        if constexpr (CAUSAL_MASK) {
            // 未来 K/V 整块已被裁掉；这里只屏蔽对角块内的上三角。
            const bool isDiagonalTile = (j == qTileIdx);
            if (j == 0) {
                if (isDiagonalTile) {
                    asc_vf_call<OnlineColwiseSoftmaxVF<true, true>>(
                        sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, halfBr, bc, validBc, data.scale, queryColBegin);
                } else {
                    asc_vf_call<OnlineColwiseSoftmaxVF<true, false>>(
                        sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, halfBr, bc, validBc, data.scale, queryColBegin);
                }
            } else if (isDiagonalTile) {
                asc_vf_call<OnlineColwiseSoftmaxVF<false, true>>(
                    sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, halfBr, bc, validBc, data.scale, queryColBegin);
            } else {
                asc_vf_call<OnlineColwiseSoftmaxVF<false, false>>(
                    sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, halfBr, bc, validBc, data.scale, queryColBegin);
            }
        } else if (j == 0) {
            asc_vf_call<OnlineColwiseSoftmaxVF<true, false>>(
                sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, halfBr, bc, validBc, data.scale, queryColBegin);
        } else {
            asc_vf_call<OnlineColwiseSoftmaxVF<false, false>>(
                sUBAddr, mUBAddr, lUBAddr, alphaUBAddr, halfBr, bc, validBc, data.scale, queryColBegin);
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

template <bool CAUSAL_MASK>
__aicore__ inline void KernelProcessForAIV(__gm__ bfloat16_t* outGMAddr, FlashAttnLiteTilingData data)
{
    using namespace AscendC;

    if ASCEND_IS_AIV {
        const uint32_t br = data.br, bc = data.bc, halfBr = br / 2;
        constexpr uint32_t d = HEAD_DIM;
        const uint32_t pHalfElements = halfBr * bc;
        const uint32_t qTileElements = br * d;
        const uint32_t outputHalfElements = halfBr * d;
        GlobalTensor<bfloat16_t> outGlobal;
        outGlobal.SetGlobalBuffer(outGMAddr);

        const uint32_t aivIdx = GetBlockIdx();
        // 两路 AIV 分别处理 Q 块的前、后 Br/2 行，并共享同组 AIC。
        const uint32_t subAivIdx = GetSubBlockIdx();
        const uint32_t aicIdx = aivIdx / GetSubBlockNum();

        const auto& aic = data.layoutAIC;
        const auto& aiv = data.layoutAIV;
        // 两路 AIV 各写 P L1 的一半; 地址由 host tiling 规划.
        LocalTensor<bfloat16_t> pL1Local(TPosition::A1, aic.pL1Addr, aic.pL1Elems);
        LocalTensor<float> sUBLocal(TPosition::VECCALC, aiv.sUBAddr, aiv.sUBElems);
        LocalTensor<float> oDeltaUBLocal(TPosition::VECCALC, aiv.oDeltaUBAddr, aiv.oDeltaUBElems);
        LocalTensor<float> oAccUBLocal(TPosition::VECCALC, aiv.oAccUBAddr, aiv.oAccUBElems);
        // PWork 在 j 循环中保存带 padding 的 NZ P, 循环结束后复用为 ND O.
        LocalTensor<bfloat16_t> pWorkUBLocal(TPosition::VECCALC, aiv.pWorkUBAddr, aiv.pWorkUBElems);
        LocalTensor<float> mUBLocal(TPosition::VECCALC, aiv.mUBAddr, aiv.rowStatsUBElems);
        LocalTensor<float> lUBLocal(TPosition::VECCALC, aiv.lUBAddr, aiv.rowStatsUBElems);
        LocalTensor<float> alphaUBLocal(TPosition::VECCALC, aiv.alphaUBAddr, aiv.rowStatsUBElems);

        for (uint32_t taskId = aicIdx; taskId < data.numTasks; taskId += data.useAicNum) {
            const uint32_t batchHeadIdx = taskId / data.tr;
            const uint32_t tileIdx = taskId % data.tr;
            const uint32_t kvTileCount = GetKvTileCount<CAUSAL_MASK>(data, tileIdx);
            const uint32_t qValidRows = GetTileValidRows(data.seqLen, tileIdx, br);
            const uint32_t subRowBegin = subAivIdx * halfBr;
            const uint32_t outputRows =
                qValidRows > subRowBegin ? (qValidRows - subRowBegin < halfBr ? qValidRows - subRowBegin : halfBr) : 0;
            const uint64_t outGMOffset = static_cast<uint64_t>(batchHeadIdx) * data.seqLen * d +
                                         static_cast<uint64_t>(tileIdx) * qTileElements +
                                         static_cast<uint64_t>(subAivIdx) * outputHalfElements;

            Duplicate<float>(oAccUBLocal, 0.0f, outputHalfElements);
            Duplicate<float>(mUBLocal, FLOAT_LOWEST, halfBr);
            Duplicate<float>(lUBLocal, 0.0f, halfBr);

            for (uint32_t j = 0; j < kvTileCount; ++j) {
                // Vector 先取得 PWork 槽，再等待 S，防止 MTE3 提前取得同一槽。
                Mutex::Lock<PIPE_V>(MUTEX_P_WORK_UB);
                WaitAicToAiv<PIPE_V>(FLAG_S_READY);
                VectorStage1<CAUSAL_MASK>(
                    sUBLocal, mUBLocal, lUBLocal, alphaUBLocal, pWorkUBLocal, data, j, tileIdx, subAivIdx);
                Mutex::Unlock<PIPE_V>(MUTEX_P_WORK_UB);

                Mutex::Lock<PIPE_MTE3>(MUTEX_P_WORK_UB);
                auto pSliceL1Local = pL1Local[static_cast<uint64_t>(subAivIdx) * pHalfElements];
                CopyPWorkToL1(pSliceL1Local, pWorkUBLocal, halfBr, bc);
                Mutex::Unlock<PIPE_MTE3>(MUTEX_P_WORK_UB);
                // P_READY 绑定 PIPE_MTE3, 在 P 写入 L1 后生效.
                SetAivToAic<PIPE_MTE3>(FLAG_P_READY);
                WaitAicToAiv<PIPE_V>(FLAG_O_READY);
                VectorStage2(oAccUBLocal, oDeltaUBLocal, alphaUBLocal, data);
                // DONE 绑定 PIPE_V, 在 V2 完成后生效.
                SetAivToAic<PIPE_V>(FLAG_DONE);
            }

            if (outputRows > 0) {
                // 复用 PWork 保存归一化和 Cast 后的输出，再写回 GM。
                Mutex::Lock<PIPE_V>(MUTEX_P_WORK_UB);
                asc_vf_call<FusedDivCastVF>(
                    reinterpret_cast<__ubuf__ bfloat16_t*>(pWorkUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(oAccUBLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(lUBLocal.GetPhyAddr()), static_cast<uint16_t>(outputRows),
                    static_cast<uint16_t>(d));
                Mutex::Unlock<PIPE_V>(MUTEX_P_WORK_UB);

                Mutex::Lock<PIPE_MTE3>(MUTEX_P_WORK_UB);
                DataCopy(outGlobal[outGMOffset], pWorkUBLocal, outputRows * d);
                Mutex::Unlock<PIPE_MTE3>(MUTEX_P_WORK_UB);
            }
        }
    }
}

} // namespace FALite
