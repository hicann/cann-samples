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

template <typename T> __aicore__ inline uint32_t C0ElemNum() {
    return C0_BYTES / sizeof(T);
}

template <typename T>
__aicore__ inline void CopyGmToL1(const AscendC::LocalTensor<T> &dstL1Local,
                                  const AscendC::GlobalTensor<T> &srcGlobal,
                                  uint32_t tileRow, uint32_t tileCol,
                                  uint32_t gmColStride) {
    using namespace AscendC;
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = tileRow;
    p.dValue = tileCol;
    p.srcNdMatrixStride = 1;
    p.srcDValue = gmColStride;
    p.dstNzC0Stride = CeilAlign<uint16_t>(tileRow, BLOCK_CUBE);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 1;
    DataCopy(dstL1Local, srcGlobal, p);
}

template <typename T>
__aicore__ inline void CopyL1ToL0A(const AscendC::LocalTensor<T> &dstL0ALocal,
                                   const AscendC::LocalTensor<T> &srcL1Local,
                                   uint32_t l1Row, uint32_t l1Col,
                                   uint32_t l0Row, uint32_t l0Col,
                                   bool transpose = false) {
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
__aicore__ inline void
CopyL1ToL0B(const AscendC::LocalTensor<T> &dstL0BLocal,
            const AscendC::LocalTensor<T> &srcL1Local, uint32_t l1Row,
            uint32_t l1Col, uint32_t l0Row, uint32_t l0Col, bool transpose) {
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
__aicore__ inline void CubeMmad(const AscendC::LocalTensor<dstT> &dstL0CLocal,
                                const AscendC::LocalTensor<aT> &aL0ALocal,
                                const AscendC::LocalTensor<bT> &bL0BLocal,
                                uint32_t m, uint32_t n, uint32_t k,
                                bool initC) {
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
__aicore__ inline void
FixpipeToVecUB(const AscendC::LocalTensor<dstT> &dstUBLocal,
               const AscendC::LocalTensor<srcT> &srcL0CLocal, uint32_t m,
               uint32_t n, uint8_t dualDstCtl = 1) {
    using namespace AscendC;
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> p;
    constexpr uint32_t FIXPIPE_N_ALIGN = 8;
    constexpr uint32_t FIXPIPE_M_ALIGN = 2;
    p.nSize = CeilAlign<uint32_t>(n, FIXPIPE_N_ALIGN);
    p.mSize = CeilAlign<uint32_t>(m, FIXPIPE_M_ALIGN);
    p.srcStride = CeilAlign<uint32_t>(p.mSize, BLOCK_CUBE);
    p.dstStride = dualDstCtl == 2 ? p.nSize / 2
                                  : CeilAlign<uint32_t>(p.nSize, BLOCK_CUBE);
    p.dualDstCtl = dualDstCtl;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    Fixpipe<dstT, srcT, PFA_CFG_UB>(dstUBLocal, srcL0CLocal, p);
}

// C1: GEMM1 K x Qᵀ -> Sᵀ, Fixpipe 沿 N 维拆分到两路 AIV UB.
__aicore__ inline void CubeStage1(AscendC::LocalTensor<bfloat16_t> &qL1Local,
                                  AscendC::LocalTensor<bfloat16_t> &kL1Local,
                                  AscendC::LocalTensor<bfloat16_t> &aL0ALocal,
                                  AscendC::LocalTensor<bfloat16_t> &bL0BLocal,
                                  AscendC::LocalTensor<float> &mmadL0CLocal,
                                  AscendC::LocalTensor<float> &sUBLocal,
                                  AscendC::GlobalTensor<bfloat16_t> &kGlobal,
                                  const FlashAttnLiteTilingData &data,
                                  uint32_t j, uint32_t batchIdx) {
    using namespace AscendC;

    if ASCEND_IS_AIC {
        const uint32_t br = data.br, bc = data.bc;
        constexpr uint32_t d = HEAD_DIM;
        const uint64_t kGMOffset =
            static_cast<uint64_t>(batchIdx) * data.seqLen * d +
            static_cast<uint64_t>(j) * bc * d;
        CopyGmToL1<bfloat16_t>(kL1Local, kGlobal[kGMOffset], bc, d, d);
        SetWaitFlag<HardEvent::MTE2_MTE1>(
            STATIC_EVENT_ID1); // 等待 K 写入 L1, 再由 LoadData 读取 kL1Local.
        // K x Qᵀ 输出 DN 布局 Sᵀ[Bc, Br], 每路 AIV 接收 Br/2 个 Q 列.
        CopyL1ToL0A<bfloat16_t>(aL0ALocal, kL1Local, bc, d, bc, d);
        CopyL1ToL0B<bfloat16_t>(bL0BLocal, qL1Local, br, d, d, br, false);
        SetWaitFlag<HardEvent::MTE1_M>(
            STATIC_EVENT_ID0); // 等待 Q/K 写入 L0A/L0B, 再由 Mmad 读取.
        CubeMmad<float, bfloat16_t, bfloat16_t>(mmadL0CLocal, aL0ALocal,
                                                bL0BLocal, bc, br, d, true);
        SetWaitFlag<HardEvent::M_FIX>(
            STATIC_EVENT_ID0); // 等待 S 写入 L0C, 再由 Fixpipe 读取.
        FixpipeToVecUB<float, float>(sUBLocal, mmadL0CLocal, bc, br, 2);
        // P_READY 依赖 S_READY -> V1 -> MTE3, 后续 Wait(P_READY) 已覆盖
        // C1 的 MTE1/M/FIX 依赖, 此处无需 PIPE_ALL.
    }
}

// C2: GEMM2 P x V -> ΔO, Fixpipe 将结果拆分到两路 AIV UB.
__aicore__ inline void CubeStage2(AscendC::LocalTensor<bfloat16_t> &pL1Local,
                                  AscendC::LocalTensor<bfloat16_t> &vL1Local,
                                  AscendC::LocalTensor<bfloat16_t> &aL0ALocal,
                                  AscendC::LocalTensor<bfloat16_t> &bL0BLocal,
                                  AscendC::LocalTensor<float> &mmadL0CLocal,
                                  AscendC::LocalTensor<float> &oDeltaUBLocal,
                                  AscendC::GlobalTensor<bfloat16_t> &vGlobal,
                                  const FlashAttnLiteTilingData &data,
                                  uint32_t j, uint32_t batchIdx) {
    using namespace AscendC;

    if ASCEND_IS_AIC {
        const uint32_t br = data.br, bc = data.bc;
        constexpr uint32_t d = HEAD_DIM;
        // L1 保存 Pᵀ, LoadData 转置后执行 P x V.
        CopyL1ToL0A<bfloat16_t>(aL0ALocal, pL1Local, bc, br, br, bc, true);
        SetWaitFlag<HardEvent::MTE1_MTE2>(
            STATIC_EVENT_ID0); // 等待 P 读出 L1, 再由 MTE2 覆写 V 区.
        const uint64_t vGMOffset =
            static_cast<uint64_t>(batchIdx) * data.seqLen * d +
            static_cast<uint64_t>(j) * bc * d;
        CopyGmToL1<bfloat16_t>(vL1Local, vGlobal[vGMOffset], bc, d, d);
        SetWaitFlag<HardEvent::MTE2_MTE1>(
            STATIC_EVENT_ID2); // 等待 V 写入 L1, 再由 LoadData 读取 vL1Local.
        CopyL1ToL0B<bfloat16_t>(bL0BLocal, vL1Local, bc, d, bc, d, true);
        SetWaitFlag<HardEvent::MTE1_M>(
            STATIC_EVENT_ID1); // 等待 P/V 写入 L0A/L0B, 再由 Mmad 读取.
        CubeMmad<float, bfloat16_t, bfloat16_t>(mmadL0CLocal, aL0ALocal,
                                                bL0BLocal, br, d, bc, true);
        SetWaitFlag<HardEvent::M_FIX>(
            STATIC_EVENT_ID1); // 等待 ΔO 写入 L0C, 再由 Fixpipe 读取.
        FixpipeToVecUB<float, float>(oDeltaUBLocal, mmadL0CLocal, br, d);
        // DONE 依赖 O_READY -> V2 -> PIPE_V, 下一轮 Wait(DONE) 已覆盖
        // C2 的 M/FIX 依赖, 此处无需 PIPE_ALL.
    }
}

__aicore__ inline void KernelProcessForAIC(__gm__ bfloat16_t *qGMAddr,
                                           __gm__ bfloat16_t *kGMAddr,
                                           __gm__ bfloat16_t *vGMAddr,
                                           FlashAttnLiteTilingData data) {
    using namespace AscendC;

    if ASCEND_IS_AIC {
        const uint32_t br = data.br;
        constexpr uint32_t d = HEAD_DIM;
        const uint32_t qTileElements = br * d;
        GlobalTensor<bfloat16_t> qGlobal, kGlobal, vGlobal;
        qGlobal.SetGlobalBuffer(qGMAddr);
        kGlobal.SetGlobalBuffer(kGMAddr);
        vGlobal.SetGlobalBuffer(vGMAddr);

        const auto &aic = data.layoutAIC;
        const auto &aiv = data.layoutAIV;
        // L1 按 P/Q/K/V 排布, 地址由 host tiling 规划.
        LocalTensor<bfloat16_t> pL1Local(TPosition::A1, aic.pL1Addr,
                                         aic.pL1Elems);
        LocalTensor<bfloat16_t> qL1Local(TPosition::A1, aic.qL1Addr,
                                         aic.qL1Elems);
        LocalTensor<bfloat16_t> kL1Local(TPosition::A1, aic.kL1Addr,
                                         aic.kL1Elems);
        LocalTensor<bfloat16_t> vL1Local(TPosition::A1, aic.vL1Addr,
                                         aic.vL1Elems);
        // L0A/L0B/L0C 均从物理地址 0 开始, C1 和 C2 原位复用.
        LocalTensor<bfloat16_t> aL0ALocal(TPosition::A2, aic.aL0AAddr,
                                          aic.aL0AElems);
        LocalTensor<bfloat16_t> bL0BLocal(TPosition::B2, aic.bL0BAddr,
                                          aic.bL0BElems);
        LocalTensor<float> mmadL0CLocal(TPosition::CO1, aic.cL0CAddr,
                                        aic.cL0CElems);
        // Fixpipe 双目的地址必须与两路 AIV 的 S/O_DELTA UB 地址一致.
        LocalTensor<float> sUBLocal(TPosition::VECCALC, aiv.sUBAddr,
                                    aiv.sUBElems);
        LocalTensor<float> oDeltaUBLocal(TPosition::VECCALC, aiv.oDeltaUBAddr,
                                         aiv.oDeltaUBElems);

        // 首个 task 的 j=0 无前序轮次, 其余 C1 在启动前等待上一轮 DONE.
        const uint32_t firstTaskId = GetBlockIdx();
        for (uint32_t taskId = GetBlockIdx(); taskId < data.numTasks;
             taskId += GetBlockNum()) {
            const uint32_t batchIdx = taskId / data.tr;
            const uint32_t tileIdx = taskId % data.tr;
            const uint64_t qGMOffset =
                static_cast<uint64_t>(batchIdx) * data.seqLen * d +
                static_cast<uint64_t>(tileIdx) * qTileElements;

            CopyGmToL1<bfloat16_t>(qL1Local, qGlobal[qGMOffset], br, d, d);
            SetWaitFlag<HardEvent::MTE2_MTE1>(
                STATIC_EVENT_ID0); // 等待 Q 写入 L1, 并在 j 循环中常驻.

            for (uint32_t j = 0; j < data.tc; ++j) {
                if (j > 0 || taskId != firstTaskId) {
#ifdef SIM_COMPATIBLE
                    // mode4 不聚合两路 AIV, AIC 分别等待 ID 和 ID+16.
                    CrossCoreWaitFlag<PAIR_CROSS_MODE, PIPE_MTE1>(FLAG_DONE);
                    CrossCoreWaitFlag<PAIR_CROSS_MODE, PIPE_MTE1>(
                        FLAG_DONE + AIV1_FLAG_OFFSET);
#else
                    // mode2 聚合同组两路 AIV, 1 次 Wait 等待二者.
                    CrossCoreWaitFlag<GROUP_CROSS_MODE, PIPE_MTE1>(FLAG_DONE);
#endif
                }
                CubeStage1(qL1Local, kL1Local, aL0ALocal, bL0BLocal,
                           mmadL0CLocal, sUBLocal, kGlobal, data, j, batchIdx);
#ifdef SIM_COMPATIBLE
                // mode4 下 AIC flag 0..10 对应 AIV0, 16..26 对应 AIV1.
                CrossCoreSetFlag<PAIR_CROSS_MODE, PIPE_FIX>(FLAG_S_READY);
                CrossCoreSetFlag<PAIR_CROSS_MODE, PIPE_FIX>(FLAG_S_READY +
                                                            AIV1_FLAG_OFFSET);
#else
                CrossCoreSetFlag<GROUP_CROSS_MODE, PIPE_FIX>(FLAG_S_READY);
#endif
                // P 由 C2 的 L1 -> L0A 首先消费, 因此 Wait 绑定 MTE1.
                // mode4 仅改变 AIV 配对方式, 不改变消费流水.
#ifdef SIM_COMPATIBLE
                CrossCoreWaitFlag<PAIR_CROSS_MODE, PIPE_MTE1>(FLAG_P_READY);
                CrossCoreWaitFlag<PAIR_CROSS_MODE, PIPE_MTE1>(FLAG_P_READY +
                                                              AIV1_FLAG_OFFSET);
#else
                CrossCoreWaitFlag<GROUP_CROSS_MODE, PIPE_MTE1>(FLAG_P_READY);
#endif
                CubeStage2(pL1Local, vL1Local, aL0ALocal, bL0BLocal,
                           mmadL0CLocal, oDeltaUBLocal, vGlobal, data, j,
                           batchIdx);
#ifdef SIM_COMPATIBLE
                CrossCoreSetFlag<PAIR_CROSS_MODE, PIPE_FIX>(FLAG_O_READY);
                CrossCoreSetFlag<PAIR_CROSS_MODE, PIPE_FIX>(FLAG_O_READY +
                                                            AIV1_FLAG_OFFSET);
#else
                CrossCoreSetFlag<GROUP_CROSS_MODE, PIPE_FIX>(FLAG_O_READY);
#endif
            }
        }

        // DONE 通过 MTE1 门控下一轮 CubeStage1, 循环外补齐末轮 Wait.
#ifdef SIM_COMPATIBLE
        CrossCoreWaitFlag<PAIR_CROSS_MODE, PIPE_MTE1>(FLAG_DONE);
        CrossCoreWaitFlag<PAIR_CROSS_MODE, PIPE_MTE1>(FLAG_DONE +
                                                      AIV1_FLAG_OFFSET);
#else
        CrossCoreWaitFlag<GROUP_CROSS_MODE, PIPE_MTE1>(FLAG_DONE);
#endif
    } // ASCEND_IS_AIC
}

} // namespace FALite
