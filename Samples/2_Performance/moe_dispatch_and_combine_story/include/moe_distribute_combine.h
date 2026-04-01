/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_H
#define MOE_DISTRIBUTE_COMBINE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "shmem.h"
#include "moe_distribute_comm.h"

struct MoeDistributeCombineShmemTilingData {
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t moeExpertNum;
    uint32_t moeExpertPerRankNum;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t aivNum;
    uint64_t totalWinSize;
};

namespace MoeDistributeCombineShmemImpl {
constexpr uint32_t STATE_SIZE = 1024UL * 1024UL;
constexpr uint32_t COMBINE_STATE_OFFSET = 64U * 1024U;
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U;
constexpr uint64_t WIN_STATE_OFFSET = 500UL * 1024UL;
constexpr uint64_t STATE_WIN_OFFSET = 975UL * 1024UL;
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint32_t FLAG_AFTER_WAIT = 10;

__aicore__ inline int64_t GetShmemDataAddr(__gm__ uint8_t* shmemSpace, int32_t pe) {
    return (int64_t)aclshmem_ptr(shmemSpace, pe);
}

__aicore__ inline int64_t GetShmemSignalAddr(__gm__ uint8_t* shmemSpace, int32_t pe) {
    return (int64_t)aclshmem_ptr(shmemSpace, pe) + 1022 * 1024 * 1024;
}

#define TemplateMC2TypeClass typename ExpandXType, typename XType, typename ExpandIdxType
#define TemplateMC2TypeFunc ExpandXType, XType, ExpandIdxType

using namespace Mc2Kernel;
using namespace AscendC;

template <TemplateMC2TypeClass>
class MoeDistributeCombineShmem {
public:
    __aicore__ inline MoeDistributeCombineShmem(){};
    __aicore__ inline void Init(GM_ADDR shmemSpace, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                GM_ADDR epSendCount, GM_ADDR expertScales, GM_ADDR XOut, GM_ADDR workspaceGM,
                                TPipe* pipe, const MoeDistributeCombineShmemTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitDataStatus(const MoeDistributeCombineShmemTilingData& tilingData);
    __aicore__ inline void InitInputAndOutput(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount,
                                              GM_ADDR expertScales, GM_ADDR XOut);
    __aicore__ inline void InitAttrs(const MoeDistributeCombineShmemTilingData& tilingData);
    __aicore__ inline void AlltoAllBuffInit();
    __aicore__ inline void SetWaitStatusAndDisPatch();
    __aicore__ inline void ExpertAlltoAllDispatchInnerCopyAdd(uint32_t toRankId, uint32_t tokenId, uint32_t topkId,
                                                              uint32_t tkIndex, AscendC::TEventID copyEventId);
    __aicore__ inline void ExpertAlltoAllDispatchCopyAdd();
    __aicore__ inline void ProcessMoeExpert(uint32_t tokenIndexOffset, uint32_t topkId, float scaleVal);
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void WaitDispatch(uint32_t tokenIndex);

    __aicore__ GM_ADDR GetWinAddrByRankId(const int32_t rankId) {
        return (GM_ADDR)GetShmemDataAddr(shmemContextGM_, rankId) + winDataSizeOffset_;
    }

    __aicore__ GM_ADDR GetWinStateAddrByRankId(const int32_t rankId) {
        return (GM_ADDR)GetShmemSignalAddr(shmemContextGM_, rankId) + winStatusOffset_;
    }

    __aicore__ GM_ADDR GetShmemWinAddr(const int32_t rankId) {
        return (GM_ADDR)shmemContextGM_ + winDataSizeOffset_;
    }

    __aicore__ GM_ADDR GetShmemWinStateAddr(const int32_t rankId) {
        return (GM_ADDR)shmemContextGM_ + 1022 * 1024 * 1024 + winStatusOffset_;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y) {
        return (x < y) ? x : y;
    }

    TPipe* tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGM_;
    GlobalTensor<int32_t> expertIdsGM_;
    GlobalTensor<ExpandIdxType> expandIdxGM_;
    GlobalTensor<ExpandIdxType> epSendCountGM_;
    GlobalTensor<float> expertScalesGM_;
    GlobalTensor<XType> expandOutGlobal_;
    GlobalTensor<XType> rankWindow_;
    GlobalTensor<XType> rowTmpGlobal_;
    GlobalTensor<uint32_t> selfDataStatusTensor_;

    GM_ADDR shmemContextGM_;
    GM_ADDR epWindowGM_;
    GM_ADDR statusDataSpaceGm_;

    LocalTensor<ExpandXType> gmTpSendCountTensor_;
    LocalTensor<uint32_t> datastateLocalTensor_;
    LocalTensor<float> stateResetTensor_;

    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epRankId_{0};
    uint32_t coreIdx_{0};
    uint32_t moeExpertPerRankNum_{0};
    uint32_t moeSendNum_{0};
    uint32_t moeExpertNum_{0};
    uint32_t bsKNum_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendCntNum_{0};
    uint32_t dataState_{0};
    uint32_t stateOffset_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t winStatusOffset_{0};
    uint64_t totalWinSize_{0};
    uint32_t selfSendCnt_{0};
    uint32_t hExpandXTypeSize_{0};
    uint32_t hAlign32Size_{0};
    uint32_t hFloatAlign32Size_{0};
    uint32_t hExpandXAlign32Size_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t hAlignWinCnt_{0};
    uint32_t flagRcvCount_{0};
    uint32_t opCnt_{0};

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> moeQueue_;
    TQue<QuePosition::VECIN, 1> moeSumQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> gmTpSendCountQueue_;
    TBuf<> readStateBuf_;
    TBuf<> expertScalesBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> indexCountsBuf_;
    TBuf<> tokenBuf_;
    TBuf<> stateBuf_;
    TBuf<> stateResetBuf_;

    LocalTensor<float> expertScalesLocal_;
    LocalTensor<float> rowTmpFloatLocal_;
    LocalTensor<float> mulBufLocal_;
    LocalTensor<float> sumFloatBufLocal_;
};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::InitDataStatus(
    const MoeDistributeCombineShmemTilingData& tilingData) {
    statusDataSpaceGm_ = (GM_ADDR)(GetShmemSignalAddr(shmemContextGM_, epRankId_));
    selfDataStatusTensor_.SetGlobalBuffer(
        (__gm__ uint32_t*)(statusDataSpaceGm_ + STATE_WIN_OFFSET + coreIdx_ * WIN_ADDR_ALIGN));
    TBuf<> datastateBuf;
    tpipe_->InitBuffer(datastateBuf, UB_ALIGN + UB_ALIGN);
    LocalTensor<uint64_t> datastateLocalTensor64;
    datastateLocalTensor_ = datastateBuf.Get<uint32_t>();
    datastateLocalTensor64 = datastateBuf.Get<uint64_t>();
    DataCopy(datastateLocalTensor_, selfDataStatusTensor_, (UB_ALIGN + UB_ALIGN) / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    dataState_ = datastateLocalTensor_.GetValue(0);
    datastateLocalTensor_.SetValue(0, dataState_ == 0 ? 1 : 0);
    datastateLocalTensor_.SetValue(1, 1);
    datastateLocalTensor_.SetValue(2, tilingData.epRankId);
    datastateLocalTensor_.SetValue(3, tilingData.moeExpertNum);
    datastateLocalTensor_.SetValue(4, tilingData.epWorldSize);
    datastateLocalTensor_.SetValue(5, tilingData.globalBs);
    opCnt_ = datastateLocalTensor64.GetValue(3);
    opCnt_ = opCnt_ + 1;
    datastateLocalTensor64.SetValue(3, opCnt_);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(selfDataStatusTensor_, datastateLocalTensor_, (UB_ALIGN + UB_ALIGN) / sizeof(uint32_t));
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::InitInputAndOutput(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR expertScales, GM_ADDR XOut) {
    expandXGM_.SetGlobalBuffer((__gm__ ExpandXType*)expandX);
    expertIdsGM_.SetGlobalBuffer((__gm__ ExpandIdxType*)expertIds);
    expandIdxGM_.SetGlobalBuffer((__gm__ ExpandIdxType*)expandIdx);
    epSendCountGM_.SetGlobalBuffer((__gm__ int32_t*)epSendCount);
    expertScalesGM_.SetGlobalBuffer((__gm__ float*)expertScales);
    expandOutGlobal_.SetGlobalBuffer((__gm__ XType*)XOut);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::InitAttrs(
    const MoeDistributeCombineShmemTilingData& tilingData) {
    axisBS_ = tilingData.bs;
    axisH_ = tilingData.h;
    axisK_ = tilingData.k;
    aivNum_ = tilingData.aivNum;
    epRankId_ = tilingData.epRankId;
    epWorldSize_ = tilingData.epWorldSize;
    moeExpertPerRankNum_ = tilingData.moeExpertPerRankNum;
    moeSendNum_ = epWorldSize_ * moeExpertPerRankNum_;

    totalWinSize_ = tilingData.totalWinSize;
    moeExpertNum_ = tilingData.moeExpertNum;

    stateOffset_ = STATE_OFFSET;
    uint32_t hFloatSize = axisH_ * static_cast<uint32_t>(sizeof(float));
    hAlign32Size_ = Ceil(axisH_, UB_ALIGN) * UB_ALIGN;
    hFloatAlign32Size_ = Ceil(hFloatSize, UB_ALIGN) * UB_ALIGN;
    hExpandXTypeSize_ = axisH_ * sizeof(ExpandXType);
    hExpandXAlign32Size_ = Ceil(hExpandXTypeSize_, UB_ALIGN) * UB_ALIGN;
    hAlignWinSize_ = Ceil(hExpandXTypeSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(ExpandXType);
    bsKNum_ = axisBS_ * axisK_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::Init(
    GM_ADDR shmemSpace, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount,
    GM_ADDR expertScales, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe* pipe,
    const MoeDistributeCombineShmemTilingData& tilingData) {
    tpipe_ = pipe;
    shmemContextGM_ = shmemSpace;

    coreIdx_ = GetBlockIdx();

    InitInputAndOutput(expandX, expertIds, expandIdx, epSendCount, expertScales, XOut);

    InitAttrs(tilingData);
    InitDataStatus(tilingData);

    PipeBarrier<PIPE_ALL>();

    winDataSizeOffset_ = static_cast<uint64_t>(dataState_) * (totalWinSize_ / 2UL);
    winStatusOffset_ = COMBINE_STATE_OFFSET + dataState_ * WIN_STATE_OFFSET;
    epWindowGM_ = GetWinAddrByRankId(epRankId_);
    DataCacheCleanAndInvalid<ExpandIdxType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        epSendCountGM_[moeSendNum_ - 1]);
    selfSendCnt_ = epSendCountGM_(moeSendNum_ - 1);

    SplitCoreCal();
    tpipe_->InitBuffer(moeQueue_, BUFFER_NUM, hExpandXAlign32Size_);
    flagRcvCount_ = axisK_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::BuffInit() {
    tpipe_->Reset();
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN);

    tpipe_->InitBuffer(gmTpSendCountQueue_, BUFFER_NUM, hExpandXAlign32Size_);
    tpipe_->InitBuffer(indexCountsBuf_, sendCntNum_ * EXPAND_IDX_INFO * sizeof(int32_t));
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::AlltoAllBuffInit() {
    tpipe_->Reset();
    uint32_t maxSizeTokenBuf = hExpandXAlign32Size_;
    uint32_t maxSizeRowTmpFloatBuf = hFloatAlign32Size_;
    tpipe_->InitBuffer(expertScalesBuf_, axisBS_ * axisK_ * sizeof(float));
    tpipe_->InitBuffer(tokenBuf_, maxSizeRowTmpFloatBuf);
    tpipe_->InitBuffer(rowTmpFloatBuf_, maxSizeRowTmpFloatBuf);
    tpipe_->InitBuffer(mulBuf_, hFloatAlign32Size_);
    tpipe_->InitBuffer(sumFloatBuf_, hFloatAlign32Size_);
    tpipe_->InitBuffer(moeSumQueue_, BUFFER_NUM, hExpandXAlign32Size_);
    tpipe_->InitBuffer(stateBuf_, (flagRcvCount_) * STATE_OFFSET);
    tpipe_->InitBuffer(stateResetBuf_, (flagRcvCount_) * STATE_OFFSET);
    stateResetTensor_ = stateResetBuf_.Get<float>();
    Duplicate<float>(stateResetTensor_, (float)0.0, static_cast<uint32_t>(flagRcvCount_ * FLOAT_PER_UB_ALIGN));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::SplitCoreCal() {
    sendCntNum_ = selfSendCnt_ / aivNum_;
    uint32_t remainderRankNum = selfSendCnt_ % aivNum_;

    startTokenId_ = sendCntNum_ * coreIdx_;

    if (coreIdx_ < remainderRankNum) {
        sendCntNum_++;
        startTokenId_ += coreIdx_;
    } else {
        startTokenId_ += remainderRankNum;
    }
    endTokenId_ = startTokenId_ + sendCntNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::SetWaitStatusAndDisPatch() {
    PipeBarrier<PIPE_ALL>();
    if (coreIdx_ >= selfSendCnt_) {
        return;
    }
    ExpertAlltoAllDispatchCopyAdd();
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::ExpertAlltoAllDispatchCopyAdd() {
    if (sendCntNum_ == 0U) {
        return;
    }

    LocalTensor<ExpandIdxType> expandIdxLocal = indexCountsBuf_.Get<ExpandIdxType>();
    const DataCopyExtParams bskParams{1U,
                                      static_cast<uint32_t>(sendCntNum_ * EXPAND_IDX_INFO * sizeof(uint32_t)), 0U, 0U,
                                      0U};
    const DataCopyPadExtParams<ExpandIdxType> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expandIdxLocal, expandIdxGM_[startTokenId_ * EXPAND_IDX_INFO], bskParams, copyPadParams);
    LocalTensor<float> statusTensor = readStateBuf_.AllocTensor<float>();
    Duplicate<float>(statusTensor, (float)1, FLOAT_PER_UB_ALIGN);

    __gm__ aclshmem_device_host_state_t *deviceState = aclshmemi_get_state();
    AscendC::TEventID copyEventId = (AscendC::TEventID)deviceState->mte_config.sync_id;

    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t loop = 0; loop < sendCntNum_; loop++) {
        uint32_t tkIndex = startTokenId_ + ((loop + epRankId_) % sendCntNum_);
        uint32_t baseOffset = (tkIndex - startTokenId_) * EXPAND_IDX_INFO;
        uint32_t rankIdExpandIdx = static_cast<uint32_t>(expandIdxLocal(baseOffset));
        uint32_t toRankId = rankIdExpandIdx;
        uint32_t tokenId = static_cast<uint32_t>(expandIdxLocal(baseOffset + 1));
        uint32_t topkId = static_cast<uint32_t>(expandIdxLocal(baseOffset + 2));
        ExpertAlltoAllDispatchInnerCopyAdd(toRankId, tokenId, topkId, tkIndex, copyEventId);
        PipeBarrier<PIPE_MTE3>();
        GM_ADDR stateGM =
            GetShmemWinStateAddr(toRankId) + tokenId * flagRcvCount_ * stateOffset_ + topkId * stateOffset_;
        GlobalTensor<float> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ float*)stateGM);
        aclshmemx_mte_put_nbi<float>(stateGMTensor, statusTensor, FLOAT_PER_UB_ALIGN, toRankId, copyEventId);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::ExpertAlltoAllDispatchInnerCopyAdd(
    uint32_t toRankId, uint32_t tokenId, uint32_t topkId, uint32_t tkIndex, AscendC::TEventID copyEventId) {
    uint32_t dataCnt = axisH_;
    uint32_t epOffset = tokenId * axisK_ + topkId;
    const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    uint32_t tokenGMOffset = tkIndex * axisH_;
    uint32_t tokenWinOffset = tkIndex * hAlignWinCnt_;
    GM_ADDR rankGM = GetShmemWinAddr(toRankId) + epOffset * hAlignWinSize_;
    rankWindow_.SetGlobalBuffer((__gm__ XType*)rankGM);
    gmTpSendCountTensor_ = gmTpSendCountQueue_.AllocTensor<ExpandXType>();
    DataCopyPad(gmTpSendCountTensor_, expandXGM_[tokenGMOffset], expandXCopyParams, copyPadExtParams);
    gmTpSendCountQueue_.EnQue(gmTpSendCountTensor_);
    gmTpSendCountTensor_ = gmTpSendCountQueue_.DeQue<ExpandXType>();
    aclshmemx_mte_put_nbi<float16_t>(rankWindow_, gmTpSendCountTensor_, axisH_, toRankId, copyEventId);
    gmTpSendCountQueue_.FreeTensor<ExpandXType>(gmTpSendCountTensor_);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::WaitDispatch(uint32_t tokenIndex) {
    uint32_t copyCount = flagRcvCount_ * FLOAT_PER_UB_ALIGN;
    uint32_t targetCount = copyCount;
    GM_ADDR stateGM = GetWinStateAddrByRankId(epRankId_) + tokenIndex * flagRcvCount_ * stateOffset_;
    GlobalTensor<float> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ float*)stateGM);
    float localState = 0;
    float target = (float)1.0 * targetCount;
    float minTarget = target - (float)0.5;
    float maxTarget = target + (float)0.5;
    SumParams sumParams{1, copyCount, copyCount};
    LocalTensor<float> stateTensor = stateBuf_.Get<float>();
    while ((localState < minTarget) || (localState > maxTarget)) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy<float>(stateTensor, stateGMTensor, copyCount);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(stateTensor, stateTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        localState = stateTensor(0);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy<float>(stateGMTensor, stateResetTensor_, copyCount);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::ProcessMoeExpert(uint32_t tokenIndexOffset,
                                                                                          uint32_t topkId,
                                                                                          float scaleVal) {
    uint32_t processLen = axisH_;
    const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    const DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};

    GM_ADDR wAddr = (__gm__ uint8_t*)(epWindowGM_) + (tokenIndexOffset + topkId) * hAlignWinSize_;
    rowTmpGlobal_.SetGlobalBuffer((__gm__ XType*)wAddr);
    LocalTensor<XType> tmpUb = moeSumQueue_.AllocTensor<XType>();

    DataCopyPad(tmpUb, rowTmpGlobal_, expandXCopyParams, copyPadExtParams);

    moeSumQueue_.EnQue(tmpUb);
    tmpUb = moeSumQueue_.DeQue<XType>();

    Cast(rowTmpFloatLocal_, tmpUb, AscendC::RoundMode::CAST_NONE, processLen);
    PipeBarrier<PIPE_V>();
    AscendC::Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, processLen);
    PipeBarrier<PIPE_V>();
    AscendC::Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, processLen);
    moeSumQueue_.FreeTensor<XType>(tmpUb);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::LocalWindowCopy() {
    if (axisBS_ == 0U) {
        return;
    }
    uint32_t beginIndex = 0U;
    uint32_t endIndex = 0U;
    uint32_t processLen = 0U;
    uint32_t tokenOffset = 0U;
    uint32_t tokenPerAivNum = axisBS_ / aivNum_;
    uint32_t remainderToken = axisBS_ % aivNum_;

    beginIndex = tokenPerAivNum * coreIdx_;
    if (coreIdx_ < remainderToken) {
        tokenPerAivNum++;
        beginIndex += coreIdx_;
    } else {
        beginIndex += remainderToken;
    }
    endIndex = beginIndex + tokenPerAivNum;
    if (tokenPerAivNum == 0U) {
        return;
    }
    processLen = axisH_;
    expertScalesLocal_ = expertScalesBuf_.Get<float>();
    rowTmpFloatLocal_ = rowTmpFloatBuf_.Get<float>();
    mulBufLocal_ = mulBuf_.Get<float>();
    sumFloatBufLocal_ = sumFloatBuf_.Get<float>();

    const DataCopyExtParams bskParams{1U, static_cast<uint32_t>(bsKNum_ * sizeof(uint32_t)), 0U, 0U, 0U};
    const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};
    const DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    DataCopyPad(expertScalesLocal_, expertScalesGM_, bskParams, copyPadFloatParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t curIdx = beginIndex; curIdx < endIndex; curIdx++) {
        uint32_t tokenIndex = curIdx;
        WaitDispatch(tokenIndex);
        uint32_t index = tokenIndex * axisK_;
        float scaleVal = 0.0;
        GM_ADDR wAddr;
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Duplicate(sumFloatBufLocal_, static_cast<float>(0), axisH_);
        LocalTensor<XType> tmpUb;
        uint32_t tokenIndexOffset = tokenIndex * axisK_;
        for (uint32_t topkId = 0U; topkId < axisK_; topkId++) {
            uint32_t expert_id = expertIdsGM_.GetValue(tokenIndex * axisK_ + topkId);
            scaleVal = expertScalesLocal_.GetValue(index);

            if (expert_id < moeExpertNum_) {
                ProcessMoeExpert(tokenIndexOffset, topkId, scaleVal);
                index++;
            }
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<XType> sumBufLocal = tokenBuf_.Get<XType>();
        Cast(sumBufLocal, sumFloatBufLocal_, AscendC::RoundMode::CAST_RINT, processLen);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(expandOutGlobal_[tokenIndex * axisH_ + tokenOffset], sumBufLocal, expandXCopyParams);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::Process() {
    if ASCEND_IS_AIV {
        BuffInit();
        SetWaitStatusAndDisPatch();
        AlltoAllBuffInit();
        LocalWindowCopy();
        DataCopyParams dataCopyParams{1U, sizeof(uint32_t), 0U, 0U};
        datastateLocalTensor_ = expertScalesBuf_.Get<uint32_t>();
        datastateLocalTensor_.SetValue(0, FLAG_AFTER_WAIT);
        selfDataStatusTensor_.SetGlobalBuffer(
            (__gm__ uint32_t*)(statusDataSpaceGm_ + STATE_WIN_OFFSET + coreIdx_ * WIN_ADDR_ALIGN + sizeof(uint32_t)));
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(selfDataStatusTensor_, datastateLocalTensor_, dataCopyParams);
        PipeBarrier<PIPE_MTE3>();
    }
}

}  // namespace MoeDistributeCombineShmemImpl
#endif  // MOE_DISTRIBUTE_COMBINE_IMPL_H