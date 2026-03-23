/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_BLOCK_MMAD_MX_SWAT_H
#define MATMUL_BLOCK_MMAD_MX_SWAT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../../../common/kernel_utils/layout_utils.h"
#include "../../../../common/kernel_utils/common_utils.h"
#include "../../../../common/kernel_utils/tuple_utils.h"
#include "../utils/quant_matmul_constant.h"

struct TileL1L0Param {
    uint64_t curM = 0;
    uint64_t curN = 0;
    uint64_t curAlignM = 0;
    uint64_t curAlignN = 0;
    uint64_t curGmAKL1 = 0;
    uint64_t curGmBKL1 = 0;
    uint64_t curPadAKL1 = 0;  // padded to 64 alignment
    uint64_t curPadBKL1 = 0;  // padded to 64 alignment
    uint64_t curKL0 = 0;
};

namespace Block {
template <class AType_, class BType_, class CType_>
class BlockMmadMx {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using MxL0AType = AType_;
    using MxL0BType = BType_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool transA = false;
    static constexpr bool transB = true;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1{0};  //  baseK_*4 by default
        uint64_t scaleKL1{0};
        uint64_t l1BufNum{2};
    };

    __aicore__ inline BlockMmadMx()
    {
        #pragma unroll
        for(uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    }

    __aicore__ inline ~BlockMmadMx()
    {
        #pragma unroll
        for(uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    }

    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t baseM_{256};
    uint64_t baseN_{256};
    uint64_t baseK_{256};
    uint64_t kL1_{1024};
    uint64_t kL1TileNum;
    uint64_t C0_SIZE{64};
    uint64_t l1BufNum_{2};
    uint64_t scaleKL1_{1024*32};

    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t HALF_L0C_SIZE = L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(float);
    constexpr static uint64_t BLOCK_CUBE = 16UL;
    constexpr static uint64_t MXFP_GROUP_SIZE = 32UL;
    constexpr static uint64_t MXFP_DIVISOR_SIZE = 64UL;
    constexpr static uint64_t MXFP_MULTI_BASE_SIZE = 2;
    constexpr static uint64_t SCALE_BUFFER_NUM = 2;
    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool enableL0cPingPong_{false};

    __aicore__ inline void Init(const TupleShape& problemShape, const BlockShape& l0TileShape,
                                const L1Params& l1Params)
    {
        m_ = static_cast<uint64_t>(Get<IDX_M_IDX>(problemShape));
        n_ = static_cast<uint64_t>(Get<IDX_N_IDX>(problemShape));
        k_ = static_cast<uint64_t>(Get<IDX_K_IDX>(problemShape));

        kL1_ = (l1Params.kL1 > 0) ? l1Params.kL1 : (baseK_ * 4);
        scaleKL1_ = l1Params.scaleKL1;
        baseM_ = static_cast<uint64_t>(Get<IDX_M_IDX>(l0TileShape));
        baseN_ = static_cast<uint64_t>(Get<IDX_N_IDX>(l0TileShape));
        baseK_ = static_cast<uint64_t>(Get<IDX_K_IDX>(l0TileShape));
        if (baseK_ == 0) {
            baseK_ = 128 / sizeof(fp4x2_e2m1_t);
        }
        l1BufNum_ = l1Params.l1BufNum;
        bL1OneBuffer_ = baseN_ * kL1_;
        scaleBL1OneBuffer_ = baseN_ * CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

        aL1OneBuffer_ = baseM_ * Align(kL1_, MXFP_DIVISOR_SIZE);
        scaleAL1OneBuffer_ = baseM_ * CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        // L1 layout: sequential (no Bank separation) for Double-Buffer-only version.
        for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            uint64_t l1Offset = L1_SIZE * (bufferId & 1);
            l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (bufferId >> 1);
            l1BufferBOffset_[bufferId] =
                l1Offset + aL1OneBuffer_ * (l1BufNum_ >> 1) + bL1OneBuffer_ * (bufferId >> 1);
        }
        for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
            l1BufferScaleAOffset_[bufferId] = (l1BufferScaleAOffset_[bufferId] + 1) >> 1;
            l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
        }

        kL1TileNum = CeilDiv(k_, kL1_);
    }

    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<AType> &aGlobal,
        const AscendC::LocalTensor<AType> &al1Local, TileL1L0Param &tileL1L0Param)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = tileL1L0Param.curM;
        uint64_t dDim = tileL1L0Param.curGmAKL1;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = (dDim + 1) >> 1;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = (k_ + 1) >> 1;
        nd2nzParams.dstNzC0Stride = tileL1L0Param.curAlignM;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<BType> &bGlobal,
        const AscendC::LocalTensor<BType> &bl1Local, TileL1L0Param &tileL1L0Param)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = tileL1L0Param.curN;
        uint64_t dDim = tileL1L0Param.curGmBKL1;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = (dDim + 1) >> 1;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = (k_ + 1) >> 1;
        nd2nzParams.dstNzC0Stride = tileL1L0Param.curAlignN;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInScaleA(const AscendC::GlobalTensor<fp8_e8m0_t> &aScaleGlobal,
                                        const AscendC::LocalTensor<fp8_e8m0_t> &aScaleL1Local, uint64_t curML1, uint64_t curKL1,
                                        uint64_t kL1Offset)
    {
        uint64_t curScaleKL1 = curKL1;
        if (kL1Offset + curScaleKL1 > k_) {
            curScaleKL1 = k_ - kL1Offset;
        }
        uint64_t nDim = curML1;
        uint64_t dDim = CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);

        uint64_t offsetScaleAGM = kL1Offset / MXFP_DIVISOR_SIZE;

        AscendC::GlobalTensor<half> aScaleGlobalB16;
        aScaleGlobalB16.SetGlobalBuffer(((__gm__ half*)(aScaleGlobal.GetPhyAddr())));
        auto aScaleL1LocalImpl = aScaleL1Local.template ReinterpretCast<half>();

        AscendC::Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = 1;
        dn2nzParams.dValue = nDim;
        dn2nzParams.nValue = dDim;
        dn2nzParams.srcDnMatrixStride = 0;
        dn2nzParams.srcDValue = CeilDiv(k_, MXFP_DIVISOR_SIZE);
        dn2nzParams.dstNzC0Stride = CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
        dn2nzParams.dstNzNStride = 1;
        dn2nzParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(aScaleL1LocalImpl, aScaleGlobalB16[offsetScaleAGM], dn2nzParams);
    }

    __aicore__ inline void CopyInScaleB(const AscendC::GlobalTensor<fp8_e8m0_t> &bScaleGlobal,
                                        const AscendC::LocalTensor<fp8_e8m0_t> &bScaleL1Local, uint64_t curNL1,
                                        uint64_t kL1Offset)
    {
        uint64_t curScaleKL1 = scaleKL1_;
        if (kL1Offset + curScaleKL1 > k_) {
            curScaleKL1 = k_ - kL1Offset;
        }
        uint64_t nDim = curNL1;
        uint64_t dDim = CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);

        AscendC::GlobalTensor<half> bScaleGlobalB16;
        bScaleGlobalB16.SetGlobalBuffer(((__gm__ half*)(bScaleGlobal.GetPhyAddr())));
        auto bScaleL1LocalImpl = bScaleL1Local.template ReinterpretCast<half>();

        uint64_t offsetScaleBGM = kL1Offset / MXFP_DIVISOR_SIZE;

        AscendC::Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = 1;
        dn2nzParams.dValue = nDim;
        dn2nzParams.nValue = dDim;
        dn2nzParams.srcDnMatrixStride = 0;
        dn2nzParams.srcDValue = CeilDiv(k_, MXFP_DIVISOR_SIZE);
        dn2nzParams.dstNzC0Stride = CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
        dn2nzParams.dstNzNStride = 1;
        dn2nzParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(bScaleL1LocalImpl, bScaleGlobalB16[offsetScaleBGM], dn2nzParams);
    }

    __aicore__ inline void CopyScalesInL1(AscendC::GlobalTensor<fp8_e8m0_t> &scaleAGlobal,
                                           AscendC::GlobalTensor<fp8_e8m0_t> &scaleBGlobal, TileL1L0Param &tileL1L0Param,
                                           uint64_t l1Iter, uint64_t scaleL1BufId)
    {
        uint64_t kL1Offset = l1Iter * kL1_;
        if (l1Iter % (scaleKL1_ / kL1_) == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
            CopyInScaleA(scaleAGlobal, scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId]], tileL1L0Param.curM,
                        scaleKL1_, kL1Offset);
            CopyInScaleB(scaleBGlobal, scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId]], tileL1L0Param.curN,
                        kL1Offset);
        }
    }

    __aicore__ inline void CopyAInL1(AscendC::GlobalTensor<AType> aGlobal, TileL1L0Param tileL1L0Param, uint64_t offsetA,
                                     uint64_t offsetAL1, uint64_t l1Iter)
    {
        CopyInA1(aGlobal[offsetA], aL1Local_[offsetAL1], tileL1L0Param);
    }

    __aicore__ inline void CopyBInL1(AscendC::GlobalTensor<BType> bGlobal, TileL1L0Param tileL1L0Param, uint64_t l1BufId,
                                    uint64_t l1Iter)
    {
        uint64_t offsetB = l1Iter * kL1_;
        CopyInB1(bGlobal[offsetB], bL1Local_[l1BufferBOffset_[l1BufId]], tileL1L0Param);
    }

    __aicore__ inline void CopyInL0A(
        const AscendC::LocalTensor<MxL0AType>& l0aLocal, const AscendC::LocalTensor<AType>& al1Local,
        const AscendC::LocalTensor<fp8_e8m0_t>& scaleAl1Local, uint64_t iter, TileL1L0Param& tileL1L0Param,
        uint64_t curScaleKL1)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        AscendC::LoadData2DMxParams loadData2DMxParams;
        uint64_t m1 = CeilDiv(tileL1L0Param.curM, AscendC::BLOCK_CUBE);
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = CeilDiv(iter * baseK_, C0_SIZE);
        loadDataParams.mStep = m1;
        loadDataParams.kStep = CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
        loadDataParams.srcStride = loadDataParams.mStep;
        loadDataParams.dstStride = loadDataParams.mStep;
        loadDataParams.ifTranspose = false;

        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.yStartPosition = CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.xStep = m1;
        loadData2DMxParams.yStep = CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.srcStride = CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        AscendC::LoadData(l0aLocal, al1Local, scaleAl1Local, loadDataParams, loadData2DMxParams);
    }

    __aicore__ inline void CopyInL0B(const AscendC::LocalTensor<MxL0BType> &l0bLocal,
                                     const AscendC::LocalTensor<BType> &bl1Local,
                                     const AscendC::LocalTensor<fp8_e8m0_t> &scaleBl1Local, uint64_t iter,
                                     TileL1L0Param &tileL1L0Param)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        AscendC::LoadData2DMxParams loadData2DMxParams;
        uint64_t n1 = CeilDiv(tileL1L0Param.curN, AscendC::BLOCK_CUBE);
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = CeilDiv(iter * baseK_, C0_SIZE);
        loadDataParams.mStep = n1;
        loadDataParams.kStep = CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
        loadDataParams.srcStride = loadDataParams.mStep;
        loadDataParams.dstStride = loadDataParams.mStep;
        loadDataParams.ifTranspose = false;

        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.yStartPosition = CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.xStep = n1;
        loadData2DMxParams.yStep = CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.srcStride = CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        AscendC::LoadData(l0bLocal, bl1Local, scaleBl1Local, loadDataParams, loadData2DMxParams);
    }

    __aicore__ inline void Mmad(
        AscendC::MmadParams &mmadParams, uint64_t l0cOffset, uint64_t l0abOffset, uint64_t biasOffset, bool needBias)
    {
        mmadParams.cmatrixSource = false;
        AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], mmadParams);
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<CType> &cGlobal, AscendC::LocalTensor<float> &c1Local,
                                   uint64_t baseM, uint64_t baseN)
    {
        AscendC::DataCopyCO12DstParams intriParams;
        intriParams.nSize = baseN;
        intriParams.mSize = baseM;
        intriParams.dstStride = n_;
        intriParams.srcStride = Align(baseM, AscendC::BLOCK_CUBE);
        if constexpr (AscendC::IsSameType<CType, bfloat16_t>::value) {
            intriParams.quantPre = QuantMode_t::F322BF16;
        } else if (AscendC::IsSameType<CType, half>::value) {
            intriParams.quantPre = QuantMode_t::F322F16;
        } else if (AscendC::IsSameType<CType, float>::value) {
            intriParams.quantPre = QuantMode_t::NoQuant;
        }
        intriParams.nz2ndEn = true;
        intriParams.unitFlag = 0;  // No UnitFlag: add manual SET/WAIT for FIXPIPE
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::DataCopy(cGlobal, c1Local, intriParams);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
    }

    __aicore__ inline void UpdateKL1(TileL1L0Param &tileL1L0Param, uint64_t iter0)
    {
        tileL1L0Param.curGmBKL1 = (iter0 + 1 == kL1TileNum) ? (k_ - iter0 * kL1_) : kL1_;
        tileL1L0Param.curPadBKL1 = CeilAlign(tileL1L0Param.curGmBKL1, MXFP_DIVISOR_SIZE);
        tileL1L0Param.curGmAKL1 = tileL1L0Param.curGmBKL1;
        tileL1L0Param.curPadAKL1 = tileL1L0Param.curPadBKL1;
    }

    __aicore__ inline void UpdateKL0(TileL1L0Param &tileL1L0Param, uint64_t iter1)
    {
        if (iter1 * baseK_ + baseK_ > tileL1L0Param.curPadBKL1) {
            tileL1L0Param.curKL0 = tileL1L0Param.curPadBKL1 - iter1 * baseK_;
        } else {
            tileL1L0Param.curKL0 = baseK_;
        }
    }

    __aicore__ inline void GetAlignMN(TileL1L0Param& tileL1L0Param)
    {
        tileL1L0Param.curAlignM = CeilAlign(tileL1L0Param.curM, BLOCK_CUBE);
        tileL1L0Param.curAlignN = CeilAlign(tileL1L0Param.curN, BLOCK_CUBE);
    }

    __aicore__ inline void Iterate(TileL1L0Param &tileL1L0Param, AscendC::MmadParams &mmadParams, uint64_t l1Iter, uint64_t l1BufId,
                                   uint64_t scaleL1BufId, uint64_t offsetAl1, uint64_t l0cOffset)
    {
        uint64_t kL0Iter = CeilDiv(tileL1L0Param.curGmBKL1, baseK_);
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            UpdateKL0(tileL1L0Param, iter1);
            uint64_t l0Offset = (HALF_L0_SIZE << 1) * (l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            uint64_t offsetScaleL1 = BLOCK_CUBE * (l1Iter % (scaleKL1_ / kL1_)) * (kL1_ / MXFP_GROUP_SIZE);
            CopyInL0A(l0aLocal_[l0Offset], aL1Local_[offsetAl1],
                      scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId] + offsetScaleL1], iter1, tileL1L0Param,
                      scaleKL1_);

            CopyInL0B(l0bLocal_[l0Offset], bL1Local_[l1BufferBOffset_[l1BufId]],
                      scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId] + offsetScaleL1], iter1, tileL1L0Param);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            mmadParams.k = CeilAlign(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
            mmadParams.unitFlag = 0;  // No UnitFlag in this version
            mmadParams.cmatrixInitVal = (l1Iter == 0 && iter1 == 0);
            Mmad(mmadParams, l0cOffset, l0Offset, baseN_ * biasBufId_, false);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            l0PingPong_++;
        }
    }

    __aicore__ inline void operator()(
        AscendC::GlobalTensor<AType> aGlobal,
        AscendC::GlobalTensor<BType> bGlobal,
        AscendC::GlobalTensor<fp8_e8m0_t> scaleAGlobal,
        AscendC::GlobalTensor<fp8_e8m0_t> scaleBGlobal,
        AscendC::GlobalTensor<CType> cGlobal,
        const BlockShape& singleShape)
    {
        TileL1L0Param tileL1L0Param;
        tileL1L0Param.curM = Get<IDX_M_TILEIDX>(singleShape);
        tileL1L0Param.curN = Get<IDX_N_TILEIDX>(singleShape);
        GetAlignMN(tileL1L0Param);
        AscendC::MmadParams mmadParams;
        mmadParams.m = tileL1L0Param.curM;
        mmadParams.n = tileL1L0Param.curN;
        mmadParams.disableGemv = true;
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        for (uint64_t iter0 = 0; iter0 < kL1TileNum; ++iter0) {
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
            uint64_t offsetA = iter0 * kL1_;
            uint64_t offsetAl1 = l1BufferAOffset_[l1BufId];
            CopyScalesInL1(scaleAGlobal, scaleBGlobal, tileL1L0Param, iter0, scaleL1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            UpdateKL1(tileL1L0Param, iter0);
            CopyAInL1(aGlobal, tileL1L0Param, offsetA, offsetAl1, iter0);
            CopyBInL1(bGlobal, tileL1L0Param, l1BufId, iter0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            Iterate(tileL1L0Param, mmadParams, iter0, l1BufId, scaleL1BufId, offsetAl1, l0cOffset);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            if ((iter0 + 1) % (scaleKL1_ / kL1_) == 0 || iter0 == kL1TileNum - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
                scaleLoopCnt_++;
            }
            abL1LoopCnt_++;
        }
        AscendC::LocalTensor<float> c1Local = c1Local_[l0cOffset];
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        CopyOut(cGlobal, c1Local, mmadParams.m, mmadParams.n);
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }
private:
    uint16_t biasBufId_ = 0;
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t scaleAL1OneBuffer_ = 0UL;
    uint64_t scaleBL1OneBuffer_ = 0UL;
    uint64_t l1BufferAOffset_[4] = {0UL};
    uint64_t l1BufferBOffset_[4] = {0UL};
    uint64_t l1BufferScaleAOffset_[2] = {0UL};
    uint64_t l1BufferScaleBOffset_[2] = {0UL};
    AscendC::LocalTensor<MxL0AType> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<MxL0BType> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> c1Local_{AscendC::TPosition::CO1, 0, L0C_SIZE};
    AscendC::LocalTensor<AType> aL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<BType> bL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<fp8_e8m0_t> scaleAL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<fp8_e8m0_t> scaleBL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
};
}
#endif
