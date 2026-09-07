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

// 每个物理槽对应一个 MutexID. 相邻流水用同一 ID 交接该槽的所有权.
constexpr MutexID MUTEX_K_L1_BASE = 0;
constexpr MutexID MUTEX_V_L1_BASE = MUTEX_K_L1_BASE + PIPELINE_SLOT_NUM;
constexpr MutexID MUTEX_Q_L1 = MUTEX_V_L1_BASE + PIPELINE_SLOT_NUM;
constexpr MutexID MUTEX_L0AB_BASE = MUTEX_Q_L1 + 1;
constexpr MutexID MUTEX_L0C_BASE = MUTEX_L0AB_BASE + PIPELINE_SLOT_NUM;

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
    const AscendC::LocalTensor<bT>& bL0BLocal, uint32_t m, uint32_t n, uint32_t k, bool initC)
{
    using namespace AscendC;
    MmadParams p;
    p.m = m;
    p.n = n;
    p.k = k;
    p.cmatrixSource = false;
    p.cmatrixInitVal = initC;
    p.unitFlag = 0;
    Mmad(dstL0CLocal, aL0ALocal, bL0BLocal, p);
}

template <typename dstT, typename srcT>
__aicore__ inline void FixpipeToVecUB(
    const AscendC::LocalTensor<dstT>& dstUBLocal, const AscendC::LocalTensor<srcT>& srcL0CLocal, uint32_t m, uint32_t n,
    uint8_t dualDstCtl = 1)
{
    using namespace AscendC;
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> p;
    constexpr uint32_t FIXPIPE_N_ALIGN = 8;
    constexpr uint32_t FIXPIPE_M_ALIGN = 2;
    p.nSize = CeilAlign<uint32_t>(n, FIXPIPE_N_ALIGN);
    p.mSize = CeilAlign<uint32_t>(m, FIXPIPE_M_ALIGN);
    p.srcStride = CeilAlign<uint32_t>(p.mSize, BLOCK_CUBE);
    p.dstStride = dualDstCtl == 2 ? p.nSize / 2 : CeilAlign<uint32_t>(p.nSize, BLOCK_CUBE);
    p.dualDstCtl = dualDstCtl;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    Fixpipe<dstT, srcT, PFA_CFG_UB>(dstUBLocal, srcL0CLocal, p);
}

// C1：K × Qᵀ -> Sᵀ；Fixpipe 将 Br 个 Query 行均分到两路 AIV UB。
__aicore__ inline void CubeStage1(
    AscendC::LocalTensor<bfloat16_t>& qL1Local, AscendC::LocalTensor<bfloat16_t>& kL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL0ALocal, AscendC::LocalTensor<bfloat16_t>& bL0BLocal,
    AscendC::LocalTensor<float>& mmadL0CLocal, AscendC::LocalTensor<float>& sUBLocal,
    AscendC::GlobalTensor<bfloat16_t>& kGlobal, const FlashAttnLiteTilingData& data, uint32_t j, uint32_t batchHeadIdx,
    uint32_t slot, uint32_t kvTileCount)
{
    using namespace AscendC;

    if ASCEND_IS_AIC {
        const uint32_t br = data.br, bc = data.bc;
        constexpr uint32_t d = HEAD_DIM;
        const uint32_t kvValidRows = GetTileValidRows(data.seqLen, j, bc);
        const uint64_t kGMOffset =
            static_cast<uint64_t>(batchHeadIdx) * data.seqLen * d + static_cast<uint64_t>(j) * bc * d;
        const MutexID kMutexId = MUTEX_K_L1_BASE + static_cast<MutexID>(slot);
        const MutexID l0abMutexId = MUTEX_L0AB_BASE + static_cast<MutexID>(slot);
        const MutexID l0cMutexId = MUTEX_L0C_BASE + static_cast<MutexID>(slot);

        Mutex::Lock<PIPE_MTE2>(kMutexId);
        CopyGmToL1<bfloat16_t>(kL1Local, kGlobal[kGMOffset], kvValidRows, bc, d, d);
        Mutex::Unlock<PIPE_MTE2>(kMutexId);

        if (j == 0) {
            Mutex::Lock<PIPE_MTE1>(MUTEX_Q_L1);
        }
        Mutex::Lock<PIPE_MTE1>(l0abMutexId);
        CopyL1ToL0B<bfloat16_t>(bL0BLocal, qL1Local, br, d, d, br, false);
        Mutex::Lock<PIPE_MTE1>(kMutexId);
        // Sᵀ 使用 DN [Bc, Br] 布局; 两路 AIV 各接收 Br/2 列.
        CopyL1ToL0A<bfloat16_t>(aL0ALocal, kL1Local, bc, d, bc, d);
        Mutex::Unlock<PIPE_MTE1>(kMutexId);
        Mutex::Unlock<PIPE_MTE1>(l0abMutexId);
        if (j + 1 == kvTileCount) {
            Mutex::Unlock<PIPE_MTE1>(MUTEX_Q_L1);
        }

        Mutex::Lock<PIPE_M>(l0abMutexId);
        Mutex::Lock<PIPE_M>(l0cMutexId);
        CubeMmad<float, bfloat16_t, bfloat16_t>(mmadL0CLocal, aL0ALocal, bL0BLocal, bc, br, d, true);
        Mutex::Unlock<PIPE_M>(l0abMutexId);
        Mutex::Unlock<PIPE_M>(l0cMutexId);

        Mutex::Lock<PIPE_FIX>(l0cMutexId);
        FixpipeToVecUB<float, float>(sUBLocal, mmadL0CLocal, bc, br, 2);
        Mutex::Unlock<PIPE_FIX>(l0cMutexId);
    }
}

// C2：P × V -> ΔO；Fixpipe 将 Br 行均分到两路 AIV UB。
__aicore__ inline void CubeStage2(
    AscendC::LocalTensor<bfloat16_t>& pL1Local, AscendC::LocalTensor<bfloat16_t>& vL1Local,
    AscendC::LocalTensor<bfloat16_t>& aL0ALocal, AscendC::LocalTensor<bfloat16_t>& bL0BLocal,
    AscendC::LocalTensor<float>& mmadL0CLocal, AscendC::LocalTensor<float>& oDeltaUBLocal,
    AscendC::GlobalTensor<bfloat16_t>& vGlobal, const FlashAttnLiteTilingData& data, uint32_t j, uint32_t batchHeadIdx,
    uint32_t slot)
{
    using namespace AscendC;

    if ASCEND_IS_AIC {
        const uint32_t br = data.br, bc = data.bc;
        constexpr uint32_t d = HEAD_DIM;
        const MutexID vMutexId = MUTEX_V_L1_BASE + static_cast<MutexID>(slot);
        const MutexID l0abMutexId = MUTEX_L0AB_BASE + static_cast<MutexID>(slot);
        const MutexID l0cMutexId = MUTEX_L0C_BASE + static_cast<MutexID>(slot);

        // 先发射 P 的 MTE1, 再等待 P_READY_MTE2 并搬运 V, 让两条 Pipe 尽量重叠.
        Mutex::Lock<PIPE_MTE1>(l0abMutexId);
        CopyL1ToL0A<bfloat16_t>(aL0ALocal, pL1Local, bc, br, br, bc, true);

        // P_READY_MTE2 使 V 的 MTE2 等待两路 AIV 写完 P.
        WaitAivToAic<PIPE_MTE2>(SlotFlagId(FLAG_P_READY_MTE2_BASE, slot));
        const uint64_t vGMOffset =
            static_cast<uint64_t>(batchHeadIdx) * data.seqLen * d + static_cast<uint64_t>(j) * bc * d;
        Mutex::Lock<PIPE_MTE2>(vMutexId);
        CopyGmToL1<bfloat16_t>(vL1Local, vGlobal[vGMOffset], GetTileValidRows(data.seqLen, j, bc), bc, d, d);
        Mutex::Unlock<PIPE_MTE2>(vMutexId);

        Mutex::Lock<PIPE_MTE1>(vMutexId);
        CopyL1ToL0B<bfloat16_t>(bL0BLocal, vL1Local, bc, d, bc, d, true);
        Mutex::Unlock<PIPE_MTE1>(vMutexId);
        Mutex::Unlock<PIPE_MTE1>(l0abMutexId);

        Mutex::Lock<PIPE_M>(l0abMutexId);
        Mutex::Lock<PIPE_M>(l0cMutexId);
        CubeMmad<float, bfloat16_t, bfloat16_t>(mmadL0CLocal, aL0ALocal, bL0BLocal, br, d, bc, true);
        Mutex::Unlock<PIPE_M>(l0abMutexId);
        Mutex::Unlock<PIPE_M>(l0cMutexId);

        Mutex::Lock<PIPE_FIX>(l0cMutexId);
        FixpipeToVecUB<float, float>(oDeltaUBLocal, mmadL0CLocal, br, d);
        Mutex::Unlock<PIPE_FIX>(l0cMutexId);
    }
}

template <bool CAUSAL_MASK>
__aicore__ inline void KernelProcessForAIC(
    __gm__ bfloat16_t* qGMAddr, __gm__ bfloat16_t* kGMAddr, __gm__ bfloat16_t* vGMAddr, FlashAttnLiteTilingData data)
{
    using namespace AscendC;

    if ASCEND_IS_AIC {
        const uint32_t br = data.br;
        constexpr uint32_t d = HEAD_DIM;
        const uint32_t qTileElements = br * d;
        const uint32_t pTileElements = br * data.bc;
        const uint32_t kTileElements = data.bc * d;
        const uint32_t vTileElements = data.bc * d;
        const uint32_t sHalfTileElements = br / 2 * data.bc;
        const uint32_t oDeltaHalfTileElements = br / 2 * d;
        GlobalTensor<bfloat16_t> qGlobal, kGlobal, vGlobal;
        qGlobal.SetGlobalBuffer(qGMAddr);
        kGlobal.SetGlobalBuffer(kGMAddr);
        vGlobal.SetGlobalBuffer(vGMAddr);

        const auto& aic = data.layoutAIC;
        const auto& aiv = data.layoutAIV;
        const uint32_t aL0ATileElements = aic.aL0AElems / PIPELINE_SLOT_NUM;
        const uint32_t bL0BTileElements = aic.bL0BElems / PIPELINE_SLOT_NUM;
        const uint32_t cL0CTileElements = aic.cL0CElems / PIPELINE_SLOT_NUM;
        LocalTensor<bfloat16_t> pL1Local(TPosition::A1, aic.pL1Addr, aic.pL1Elems);
        LocalTensor<bfloat16_t> qL1Local(TPosition::A1, aic.qL1Addr, aic.qL1Elems);
        LocalTensor<bfloat16_t> kL1Local(TPosition::A1, aic.kL1Addr, aic.kL1Elems);
        LocalTensor<bfloat16_t> vL1Local(TPosition::A1, aic.vL1Addr, aic.vL1Elems);
        // L0A/L0B/L0C 均从物理地址 0 开始, 每类空间按地址分为双槽.
        LocalTensor<bfloat16_t> aL0ALocal(TPosition::A2, aic.aL0AAddr, aic.aL0AElems);
        LocalTensor<bfloat16_t> bL0BLocal(TPosition::B2, aic.bL0BAddr, aic.bL0BElems);
        LocalTensor<float> mmadL0CLocal(TPosition::CO1, aic.cL0CAddr, aic.cL0CElems);
        // Fixpipe 双目的地址必须与两路 AIV 的 S/DeltaO UB 地址一致.
        LocalTensor<float> sUBLocal(TPosition::VECCALC, aiv.sUBAddr, aiv.sUBElems);
        LocalTensor<float> oDeltaUBLocal(TPosition::VECCALC, aiv.oDeltaUBAddr, aiv.oDeltaUBElems);

        for (uint32_t taskId = GetBlockIdx(); taskId < data.numTasks; taskId += GetBlockNum()) {
            const uint32_t batchHeadIdx = taskId / data.tr;
            const uint32_t tileIdx = taskId % data.tr;
            const uint32_t kvTileCount = GetKvTileCount<CAUSAL_MASK>(data, tileIdx);
            const uint64_t qGMOffset =
                static_cast<uint64_t>(batchHeadIdx) * data.seqLen * d + static_cast<uint64_t>(tileIdx) * qTileElements;

            Mutex::Lock<PIPE_MTE2>(MUTEX_Q_L1);
            CopyGmToL1<bfloat16_t>(qL1Local, qGlobal[qGMOffset], GetTileValidRows(data.seqLen, tileIdx, br), br, d, d);
            Mutex::Unlock<PIPE_MTE2>(MUTEX_Q_L1);

            for (uint32_t groupStart = 0; groupStart < kvTileCount; groupStart += PIPELINE_SLOT_NUM) {
                const uint32_t groupEnd =
                    groupStart + PIPELINE_SLOT_NUM < kvTileCount ? groupStart + PIPELINE_SLOT_NUM : kvTileCount;

                // 先连续发射本组 C1, 将各 S slot 依次交给 AIV.
                for (uint32_t j = groupStart; j < groupEnd; ++j) {
                    const uint32_t slot = j % PIPELINE_SLOT_NUM;
                    auto kSlotL1Local = kL1Local[static_cast<uint64_t>(slot) * kTileElements];
                    auto aSlotL0ALocal = aL0ALocal[static_cast<uint64_t>(slot) * aL0ATileElements];
                    auto bSlotL0BLocal = bL0BLocal[static_cast<uint64_t>(slot) * bL0BTileElements];
                    auto cSlotL0CLocal = mmadL0CLocal[static_cast<uint64_t>(slot) * cL0CTileElements];
                    auto sSlotUBLocal = sUBLocal[static_cast<uint64_t>(slot) * sHalfTileElements];
                    CubeStage1(
                        qL1Local, kSlotL1Local, aSlotL0ALocal, bSlotL0BLocal, cSlotL0CLocal, sSlotUBLocal, kGlobal,
                        data, j, batchHeadIdx, slot, kvTileCount);
                    SetAicToAiv<PIPE_FIX>(SlotFlagId(FLAG_S_READY_BASE, slot));
                }

                // P_READY 到达后连续发射本组 C2, 将各 DeltaO slot 依次交给 AIV.
                for (uint32_t j = groupStart; j < groupEnd; ++j) {
                    const uint32_t slot = j % PIPELINE_SLOT_NUM;
                    WaitAivToAic<PIPE_MTE1>(SlotFlagId(FLAG_P_READY_BASE, slot));
                    auto pSlotL1Local = pL1Local[static_cast<uint64_t>(slot) * pTileElements];
                    auto vSlotL1Local = vL1Local[static_cast<uint64_t>(slot) * vTileElements];
                    auto aSlotL0ALocal = aL0ALocal[static_cast<uint64_t>(slot) * aL0ATileElements];
                    auto bSlotL0BLocal = bL0BLocal[static_cast<uint64_t>(slot) * bL0BTileElements];
                    auto cSlotL0CLocal = mmadL0CLocal[static_cast<uint64_t>(slot) * cL0CTileElements];
                    auto oDeltaSlotUBLocal = oDeltaUBLocal[static_cast<uint64_t>(slot) * oDeltaHalfTileElements];
                    CubeStage2(
                        pSlotL1Local, vSlotL1Local, aSlotL0ALocal, bSlotL0BLocal, cSlotL0CLocal, oDeltaSlotUBLocal,
                        vGlobal, data, j, batchHeadIdx, slot);
                    SetAicToAiv<PIPE_FIX>(SlotFlagId(FLAG_O_READY_BASE, slot));
                }
            }
        }

    } // ASCEND_IS_AIC
}

} // namespace FALite
