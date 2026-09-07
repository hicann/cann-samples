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

// 每个 ID 对应一个物理缓冲区。C1 和 C2 串行复用同一组 L0A/L0B/L0C。
constexpr MutexID MUTEX_Q_L1 = 0;
constexpr MutexID MUTEX_K_L1 = 1;
constexpr MutexID MUTEX_V_L1 = 2;
constexpr MutexID MUTEX_P_L1 = 3;
constexpr MutexID MUTEX_L0AB = 4;
constexpr MutexID MUTEX_L0C = 5;

template <typename T>
__aicore__ inline uint32_t C0ElemNum()
{
    return C0_BYTES / sizeof(T);
}

template <typename T>
__aicore__ inline void CopyGmToL1(
    const AscendC::LocalTensor<T>& dstL1Local, const AscendC::GlobalTensor<T>& srcGlobal, uint32_t validRows,
    uint32_t tileRows, uint32_t tileCols, uint32_t gmColStride)
{
    using namespace AscendC;
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = validRows;
    p.dValue = tileCols;
    p.srcNdMatrixStride = 1;
    p.srcDValue = gmColStride;
    p.dstNzC0Stride = CeilAlign<uint16_t>(tileRows, BLOCK_CUBE);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 1;
    DataCopy(dstL1Local, srcGlobal, p);

    if (validRows < tileRows) {
        // ND->NZ 后，同一通道分块内的行连续；逐通道分块清零末尾行。
        InitConstValueParams<T> zeroParams;
        zeroParams.repeatTimes = CeilDiv<uint16_t>(tileCols, C0ElemNum<T>());
        zeroParams.blockNum = static_cast<uint16_t>(tileRows - validRows);
        zeroParams.dstGap = static_cast<uint16_t>(validRows);
        zeroParams.initValue = 0;
        Fill(dstL1Local[validRows * C0ElemNum<T>()], zeroParams);
    }
}

template <typename T>
__aicore__ inline void CopyL1ToL0A(
    const AscendC::LocalTensor<T>& dstL0ALocal, const AscendC::LocalTensor<T>& srcL1Local, uint32_t l1Row,
    uint32_t l1Col, uint32_t l0Row, uint32_t l0Col, bool transpose = false)
{
    using namespace AscendC;
    LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = CeilDiv<uint16_t>(l0Row, BLOCK_CUBE);
    p.kStep = CeilDiv<uint16_t>(l0Col, C0ElemNum<T>());
    p.srcStride = CeilDiv<int32_t>(l1Row, BLOCK_CUBE);
    p.dstStride = p.mStep;
    p.ifTranspose = transpose;
    LoadData(dstL0ALocal, srcL1Local, p);
}

template <typename T>
__aicore__ inline void CopyL1ToL0B(
    const AscendC::LocalTensor<T>& dstL0BLocal, const AscendC::LocalTensor<T>& srcL1Local, uint32_t l1Row,
    uint32_t l1Col, uint32_t l0Row, uint32_t l0Col, bool transpose)
{
    using namespace AscendC;
    LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = CeilDiv<uint16_t>(l0Row, BLOCK_CUBE);
    p.kStep = CeilDiv<uint16_t>(l0Col, BLOCK_CUBE);
    p.srcStride = CeilDiv<int32_t>(l1Row, BLOCK_CUBE);
    p.dstStride = CeilDiv<uint16_t>(l0Col, BLOCK_CUBE);
    p.ifTranspose = transpose;
    LoadData(dstL0BLocal, srcL1Local, p);
}

template <typename dstT, typename aT, typename bT>
__aicore__ inline void CubeMmad(
    const AscendC::LocalTensor<dstT>& dstL0CLocal, const AscendC::LocalTensor<aT>& aL0ALocal,
    const AscendC::LocalTensor<bT>& bL0BLocal, uint32_t m, uint32_t n, uint32_t k)
{
    using namespace AscendC;
    MmadParams p;
    p.m = m;
    p.n = n;
    p.k = k;
    p.cmatrixSource = false;
    p.cmatrixInitVal = true;
    p.unitFlag = 0;
    Mmad(dstL0CLocal, aL0ALocal, bL0BLocal, p);
}

template <typename dstT, typename srcT>
__aicore__ inline void FixpipeToGm(
    const AscendC::GlobalTensor<dstT>& dstGlobal, const AscendC::LocalTensor<srcT>& srcL0CLocal, uint32_t m, uint32_t n,
    uint32_t gmStride)
{
    using namespace AscendC;
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> p;
    constexpr uint32_t FIXPIPE_N_ALIGN = 8;
    constexpr uint32_t FIXPIPE_M_ALIGN = 2;
    p.nSize = CeilAlign<uint32_t>(n, FIXPIPE_N_ALIGN);
    p.mSize = CeilAlign<uint32_t>(m, FIXPIPE_M_ALIGN);
    p.srcStride = CeilAlign<uint32_t>(p.mSize, BLOCK_CUBE);
    p.dstStride = gmStride;
    p.dualDstCtl = 1;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    Fixpipe<dstT, srcT, PFA_CFG_GM>(dstGlobal, srcL0CLocal, p);
}

// C1：K × Qᵀ -> Sᵀ，并将 Sᵀ 写入 GM 中间缓冲区。
__aicore__ inline void CubeStage1(
    AscendC::LocalTensor<bfloat16_t>& qL1Local, AscendC::LocalTensor<bfloat16_t>& kL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL0ALocal, AscendC::LocalTensor<bfloat16_t>& bL0BLocal,
    AscendC::LocalTensor<float>& l0CLocal, AscendC::GlobalTensor<float>& sGlobal,
    AscendC::GlobalTensor<bfloat16_t>& kGlobal, const FlashAttnLiteTilingData& data, uint32_t taskId, uint32_t j,
    uint32_t batchHeadIdx, uint32_t kvTileCount)
{
    using namespace AscendC;
    const uint32_t br = data.br, bc = data.bc;
    constexpr uint32_t d = HEAD_DIM;
    const uint32_t kvValidRows = GetTileValidRows(data.seqLen, j, bc);
    const uint64_t kOffset = static_cast<uint64_t>(batchHeadIdx) * data.seqLen * d + static_cast<uint64_t>(j) * bc * d;

    Mutex::Lock<PIPE_MTE2>(MUTEX_K_L1);
    CopyGmToL1<bfloat16_t>(kL1Local, kGlobal[kOffset], kvValidRows, bc, d, d);
    Mutex::Unlock<PIPE_MTE2>(MUTEX_K_L1);

    if (j == 0) {
        // Q 在一个 task 的全部 C1 中保持只读，最后一次 C1 后再释放。
        Mutex::Lock<PIPE_MTE1>(MUTEX_Q_L1);
    }
    Mutex::Lock<PIPE_MTE1>(MUTEX_L0AB);
    Mutex::Lock<PIPE_MTE1>(MUTEX_K_L1);
    CopyL1ToL0A<bfloat16_t>(aL0ALocal, kL1Local, bc, d, bc, d);
    Mutex::Unlock<PIPE_MTE1>(MUTEX_K_L1);
    CopyL1ToL0B<bfloat16_t>(bL0BLocal, qL1Local, br, d, d, br, false);
    Mutex::Unlock<PIPE_MTE1>(MUTEX_L0AB);

    Mutex::Lock<PIPE_M>(MUTEX_L0AB);
    Mutex::Lock<PIPE_M>(MUTEX_L0C);
    CubeMmad(l0CLocal, aL0ALocal, bL0BLocal, bc, br, d);
    Mutex::Unlock<PIPE_M>(MUTEX_L0AB);
    Mutex::Unlock<PIPE_M>(MUTEX_L0C);

    const uint64_t sOffset = static_cast<uint64_t>(taskId) * bc * br;
    Mutex::Lock<PIPE_FIX>(MUTEX_L0C);
    FixpipeToGm(sGlobal[sOffset], l0CLocal, bc, br, br);
    Mutex::Unlock<PIPE_FIX>(MUTEX_L0C);

    if (j + 1 == kvTileCount) {
        // Q 槽由首个 C1 取得，并在末次 C1 后归还。
        Mutex::Unlock<PIPE_MTE1>(MUTEX_Q_L1);
    }
}

// C2：P × V -> ΔO；P 和 ΔO 都经过 GM，用于展示基础数据通路。
__aicore__ inline void CubeStage2(
    AscendC::LocalTensor<bfloat16_t>& pL1Local, AscendC::LocalTensor<bfloat16_t>& vL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL0ALocal, AscendC::LocalTensor<bfloat16_t>& bL0BLocal,
    AscendC::LocalTensor<float>& l0CLocal, AscendC::GlobalTensor<float>& deltaOGlobal,
    AscendC::GlobalTensor<bfloat16_t>& pGlobal, AscendC::GlobalTensor<bfloat16_t>& vGlobal,
    const FlashAttnLiteTilingData& data, uint32_t taskId, uint32_t j, uint32_t batchHeadIdx)
{
    using namespace AscendC;
    const uint32_t br = data.br, bc = data.bc;
    constexpr uint32_t d = HEAD_DIM;
    const uint32_t kvValidRows = GetTileValidRows(data.seqLen, j, bc);
    const uint64_t pOffset = static_cast<uint64_t>(taskId) * bc * br;

    Mutex::Lock<PIPE_MTE2>(MUTEX_P_L1);
    CopyGmToL1<bfloat16_t>(pL1Local, pGlobal[pOffset], bc, bc, br, br);
    Mutex::Unlock<PIPE_MTE2>(MUTEX_P_L1);

    Mutex::Lock<PIPE_MTE1>(MUTEX_L0AB);
    Mutex::Lock<PIPE_MTE1>(MUTEX_P_L1);
    CopyL1ToL0A<bfloat16_t>(aL0ALocal, pL1Local, bc, br, br, bc, true);
    Mutex::Unlock<PIPE_MTE1>(MUTEX_P_L1);

    const uint64_t vOffset = static_cast<uint64_t>(batchHeadIdx) * data.seqLen * d + static_cast<uint64_t>(j) * bc * d;
    Mutex::Lock<PIPE_MTE2>(MUTEX_V_L1);
    CopyGmToL1<bfloat16_t>(vL1Local, vGlobal[vOffset], kvValidRows, bc, d, d);
    Mutex::Unlock<PIPE_MTE2>(MUTEX_V_L1);
    Mutex::Lock<PIPE_MTE1>(MUTEX_V_L1);
    CopyL1ToL0B<bfloat16_t>(bL0BLocal, vL1Local, bc, d, bc, d, true);
    Mutex::Unlock<PIPE_MTE1>(MUTEX_V_L1);
    Mutex::Unlock<PIPE_MTE1>(MUTEX_L0AB);

    Mutex::Lock<PIPE_M>(MUTEX_L0AB);
    Mutex::Lock<PIPE_M>(MUTEX_L0C);
    CubeMmad(l0CLocal, aL0ALocal, bL0BLocal, br, d, bc);
    Mutex::Unlock<PIPE_M>(MUTEX_L0AB);
    Mutex::Unlock<PIPE_M>(MUTEX_L0C);

    const uint64_t deltaOOffset = static_cast<uint64_t>(taskId) * br * d;
    Mutex::Lock<PIPE_FIX>(MUTEX_L0C);
    FixpipeToGm(deltaOGlobal[deltaOOffset], l0CLocal, br, d, d);
    Mutex::Unlock<PIPE_FIX>(MUTEX_L0C);
}

template <bool CAUSAL_MASK>
__aicore__ inline void KernelProcessForAIC(
    __gm__ bfloat16_t* qGMAddr, __gm__ bfloat16_t* kGMAddr, __gm__ bfloat16_t* vGMAddr, __gm__ float* sGMAddr,
    __gm__ bfloat16_t* pGMAddr, __gm__ float* deltaOGMAddr, FlashAttnLiteTilingData data)
{
    using namespace AscendC;
    if ASCEND_IS_AIC {
        constexpr uint32_t d = HEAD_DIM;
        const uint32_t qTileElements = data.br * d;
        GlobalTensor<bfloat16_t> qGlobal, kGlobal, vGlobal, pGlobal;
        GlobalTensor<float> sGlobal, deltaOGlobal;
        qGlobal.SetGlobalBuffer(qGMAddr);
        kGlobal.SetGlobalBuffer(kGMAddr);
        vGlobal.SetGlobalBuffer(vGMAddr);
        sGlobal.SetGlobalBuffer(sGMAddr);
        pGlobal.SetGlobalBuffer(pGMAddr);
        deltaOGlobal.SetGlobalBuffer(deltaOGMAddr);

        const auto& aic = data.layoutAIC;
        LocalTensor<bfloat16_t> pL1Local(TPosition::A1, aic.pL1Addr, aic.pL1Elems);
        LocalTensor<bfloat16_t> qL1Local(TPosition::A1, aic.qL1Addr, aic.qL1Elems);
        LocalTensor<bfloat16_t> kL1Local(TPosition::A1, aic.kL1Addr, aic.kL1Elems);
        LocalTensor<bfloat16_t> vL1Local(TPosition::A1, aic.vL1Addr, aic.vL1Elems);
        LocalTensor<bfloat16_t> aL0ALocal(TPosition::A2, aic.aL0AAddr, aic.aL0AElems);
        LocalTensor<bfloat16_t> bL0BLocal(TPosition::B2, aic.bL0BAddr, aic.bL0BElems);
        LocalTensor<float> l0CLocal(TPosition::CO1, aic.cL0CAddr, aic.cL0CElems);

        const uint32_t firstTaskId = GetBlockIdx();
        for (uint32_t taskId = firstTaskId; taskId < data.numTasks; taskId += GetBlockNum()) {
            const uint32_t batchHeadIdx = taskId / data.tr;
            const uint32_t qTileIdx = taskId % data.tr;
            const uint32_t kvTileCount = GetKvTileCount<CAUSAL_MASK>(data, qTileIdx);
            const uint32_t qValidRows = GetTileValidRows(data.seqLen, qTileIdx, data.br);
            const uint64_t qOffset =
                static_cast<uint64_t>(batchHeadIdx) * data.seqLen * d + static_cast<uint64_t>(qTileIdx) * qTileElements;

            Mutex::Lock<PIPE_MTE2>(MUTEX_Q_L1);
            CopyGmToL1<bfloat16_t>(qL1Local, qGlobal[qOffset], qValidRows, data.br, d, d);
            Mutex::Unlock<PIPE_MTE2>(MUTEX_Q_L1);

            for (uint32_t j = 0; j < kvTileCount; ++j) {
                if (j > 0 || taskId != firstTaskId) {
                    WaitAivToAic<PIPE_MTE1>(FLAG_DONE);
                }
                CubeStage1(
                    qL1Local, kL1Local, aL0ALocal, bL0BLocal, l0CLocal, sGlobal, kGlobal, data, taskId, j, batchHeadIdx,
                    kvTileCount);
                SetAicToAiv<PIPE_FIX>(FLAG_S_READY);
                WaitAivToAic<PIPE_MTE2>(FLAG_P_READY);
                CubeStage2(
                    pL1Local, vL1Local, aL0ALocal, bL0BLocal, l0CLocal, deltaOGlobal, pGlobal, vGlobal, data, taskId, j,
                    batchHeadIdx);
                SetAicToAiv<PIPE_FIX>(FLAG_O_READY);
            }
        }
        // AIV 的 V2 使用单槽 OAcc/DeltaO；消费末轮 DONE 后才能退出 AIC。
        WaitAivToAic<PIPE_MTE1>(FLAG_DONE);
    }
}

} // namespace FALite
