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
 * \file 0_native.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <libgen.h>
#include <linux/limits.h>
#include <unistd.h>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "moe_mrgsort.h"
#include "moe_mrgsort_out.h"
#include "moe_sort_base.h"
#include "moe_tiling_def.h"
#include "moe_util.h"
#include "moe_kernel_common.h"

using namespace AscendC;

class ExpertIdxSort : public MoeSortBase {
private:
    constexpr static int64_t WORK_GM_NUM = 2;

    GlobalTensor<float> workspaceGms[2];

    int64_t srcWsIndex = 0;
    int64_t listNum;
    int64_t perListElements;
    int64_t lastListElements;

    int64_t sortTotalLength;
    int64_t sortCoreLoops;
    int64_t sortCoreLoopElements;
    int64_t sortCoreLastLoopElements;

    int64_t perCoreExpert;
    int64_t needInitExpertCore;
    int64_t currentCoreExpert;

    MoeVBSComputeTilingData *vbsTilingData;

    MoeMrgsort mrgsorter;
    MoeMrgsortParam mrgsortParam;

public:
    __aicore__ inline ExpertIdxSort(){};

    __aicore__ inline void Init(__gm__ int32_t *expertIdx,  __gm__ int32_t *workspace, __gm__ int32_t *expandedRowIdx,
        MoeInitRoutingTilingData *tilingData, TPipe *tPipe)
    {
        this->totalLength = tilingData->n * tilingData->k;
        this->vbsTilingData = &(tilingData->vbsComputeTilingData);
        this->sortOutOneLoopMaxElements = tilingData->sortOutOneLoopMaxElements;

        this->tileLength = this->vbsTilingData->perCorePerLoopElements;
        this->sortTotalLength = this->vbsTilingData->perCoreElements;
        this->n = tilingData->n;
        this->k = tilingData->k;

        expertStart = tilingData->expertStart;
        expertEnd = tilingData->expertEnd;
        actualExpertNum = expertEnd - expertStart;

        sortCoreLoops = this->vbsTilingData->perCoreLoops;
        sortCoreLoopElements = this->vbsTilingData->perCorePerLoopElements;
        sortCoreLastLoopElements = this->vbsTilingData->perCoreLastLoopElements;

        this->pipe = tPipe;
        int64_t totalLengthAlign = Align(this->totalLength, sizeof(int32_t));
        expertIdxGm.SetGlobalBuffer(expertIdx, this->sortTotalLength);
        sortedExpertIdxGm.SetGlobalBuffer(workspace, totalLengthAlign);
        expandedRowIdxGm.SetGlobalBuffer(expandedRowIdx, totalLengthAlign);

        expertCountTempGm.SetGlobalBuffer(workspace + Align(n * k, sizeof(int32_t)) * 2, actualExpertNum);
        InitGlobalMemory(expertCountTempGm, actualExpertNum, 0);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

        // key and value
        workspaceGms[0].SetGlobalBuffer((__gm__ float *)workspace + totalLengthAlign * 2 + actualExpertNum, 
                                        totalLengthAlign * KV_FACTOR);
        workspaceGms[1].SetGlobalBuffer((__gm__ float *)workspace + totalLengthAlign * (KV_FACTOR + 2) + 
                                        actualExpertNum, totalLengthAlign * KV_FACTOR);

        int64_t bufferSize = Ceil(Max(this->sortOutOneLoopMaxElements * MAX_MRGSORT_LIST, sortCoreLoopElements), 
            ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM * sizeof(int32_t) * KV_FACTOR;
        pipe->InitBuffer(sortDataCopyInQueue, 1, bufferSize);
        pipe->InitBuffer(sortDataCopyOutQueue, 1, bufferSize);
        pipe->InitBuffer(sortedBuffer, bufferSize);
        pipe->InitBuffer(tempBuffer, bufferSize);
    }

    __aicore__ inline void VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum)
    {
        LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
        int64_t inOffset = progress * sortCoreLoopElements;
        DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), 
                                         static_cast<uint32_t>(size * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
        DataCopyPad(inLocal[0], expertIdxGm[inOffset], dataCopyParams, dataCopyPadParams);

        LocalTensor<int32_t> rowIdxLocal = inLocal[sortNum];

        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        ArithProgression<int32_t>(rowIdxLocal, inOffset, 1, size);
        sortDataCopyInQueue.EnQue(inLocal);
    }

    __aicore__ inline void UBSortCompute(int64_t progress, int64_t size, int64_t sortNum)
    {
        LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
        LocalTensor<int32_t> expertForSourceRowLocal = inLocal[0];
        LocalTensor<float> expertForSourceRowLocalFp32;

        expertForSourceRowLocalFp32 = expertForSourceRowLocal.ReinterpretCast<float>();
        Cast(expertForSourceRowLocalFp32, expertForSourceRowLocal, RoundMode::CAST_ROUND, sortNum);

        uint16_t repeatTimes = Ceil(sortNum, FLOAT_REG_TENSOR_LENGTH);
        uint32_t sreg = static_cast<uint32_t>(sortNum);
        __ubuf__ float *inUbAddr = (__ubuf__ float *)expertForSourceRowLocalFp32.GetPhyAddr();
        SortVf(inUbAddr, expertStart, sreg, repeatTimes);

        int64_t duplicateNum = size % ONE_REPEAT_SORT_NUM;
        if (duplicateNum > 0) {
            int duplicateIndex = size - duplicateNum;
            uint64_t mask0 = UINT64_MAX;
            mask0 = mask0 << duplicateNum;
            mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
            uint64_t mask[2] = {mask0, 0};
            Duplicate(expertForSourceRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
        }

        LocalTensor<float> concatLocal = expertForSourceRowLocalFp32;
        LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(sortNum));
        LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
        LocalTensor<uint32_t> sourceRowLocal;
        sourceRowLocal = inLocal[sortNum].ReinterpretCast<uint32_t>();
        Sort<float, true>(outLocal, concatLocal, sourceRowLocal, sortedLocal, sortNum / ONE_REPEAT_SORT_NUM);

        sortDataCopyOutQueue.EnQue<float>(outLocal);
        sortDataCopyInQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum)
    {
        LocalTensor<float> outLocal = sortDataCopyOutQueue.DeQue<float>();
        DataCopy(workspaceGms[0][GetSortLen<float>(progress * sortCoreLoopElements)],
                outLocal, Align(GetSortLen<float>(size), sizeof(float)));
        sortDataCopyOutQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t sortNum)
    {
        VBSCopyIn(progress, size, sortNum);
        UBSortCompute(progress, size, sortNum);
        VBSCopyOut(progress, size, sortNum);
    }

    __aicore__ inline void InitMoeMrgSort(MoeMrgsort *sorter, int64_t listNum, int64_t coreOffset,
                                          int64_t loopOffset)
    {
        GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex][loopOffset];
        LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();
        LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
        for (int64_t i = 0; i < listNum; i++) {
            LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(this->sortOutOneLoopMaxElements) * i];
            sorter->SetInput(srcWsGm, inLocalT);
        }
        GlobalTensor<float> dstWsGm = workspaceGms[1 - srcWsIndex][loopOffset];
        sorter->SetOutput(dstWsGm, outLocal);
        sortDataCopyInQueue.FreeTensor(inLocal);
        sortDataCopyOutQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void OneCoreVMSProcess(int64_t listNum, int64_t perListElements,
                                                           int64_t lastListElements)
    {
        int64_t coreOffset = GetSortLen<float>(this->vbsTilingData->perCoreElements);
        mrgsortParam.oneLoopMaxElements = this->sortOutOneLoopMaxElements;

        for (int64_t i = 0; listNum >= 1; i++) {
            int64_t loops = (listNum + MAX_MRGSORT_LIST - 1) / MAX_MRGSORT_LIST;
            int64_t remainListNum = listNum - (loops - 1) * MAX_MRGSORT_LIST;

            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = perListElements;

            int64_t loopOffset = GetSortLen<float>(mrgsortParam.perListElements * MAX_MRGSORT_LIST);
            for (int64_t loop = 0; loop < loops - 1; loop++) {
                InitMoeMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, loop * loopOffset);
                mrgsorter.Init(&mrgsortParam);
                mrgsorter.Process();
            }

            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = lastListElements;
            InitMoeMrgSort(&mrgsorter, remainListNum, coreOffset, (loops - 1) * loopOffset);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();

            listNum = loops;
            lastListElements = perListElements * (remainListNum - 1) + lastListElements;
            perListElements = perListElements * MAX_MRGSORT_LIST;
            srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;
            if (loops == 1) {
                break;
            }
        }
    }

    __aicore__ inline void VBSProcess()
    {
        int64_t sortNum = Ceil(sortCoreLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        for (int64_t loop = 0; loop < sortCoreLoops - 1; loop++) {
            UBSortProcess(loop, sortCoreLoopElements, sortNum);
        }

        sortNum = Ceil(sortCoreLastLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        UBSortProcess(sortCoreLoops - 1, sortCoreLastLoopElements, sortNum);

        if (sortCoreLoops > 1) {
            OneCoreVMSProcess(sortCoreLoops, sortCoreLoopElements, sortCoreLastLoopElements);
        }
    }

    __aicore__ inline void InitMoeMrgSortOut(MoeMrgsortOut *sorter, int64_t listNum, int64_t coreOffset)
    {
        GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex];
        LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();
        LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();

        for (int64_t i = 0; i < listNum; i++) {
            LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(this->sortOutOneLoopMaxElements) * i];
            sorter->SetInput(srcWsGm, inLocalT);
        }

        LocalTensor<float> outLocalV = outLocal[this->sortOutOneLoopMaxElements * MAX_MRGSORT_LIST];
        sorter->SetOutput(this->sortedExpertIdxGm, this->expandedRowIdxGm, outLocal, outLocalV);

        LocalTensor<float> tempBuffer =
            sortedBuffer.Get<float>(GetSortLen<float>(this->sortOutOneLoopMaxElements) * MAX_MRGSORT_LIST);
        sorter->SetBuffer(tempBuffer);
        sortDataCopyInQueue.FreeTensor(inLocal);
        sortDataCopyOutQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void SortOutProcess()
    {
        mrgsortParam.perListElements = this->vbsTilingData->perCoreElements;
        mrgsortParam.lastListElements = this->vbsTilingData->perCoreElements;
        mrgsortParam.oneLoopMaxElements = this->sortOutOneLoopMaxElements;

        MoeMrgsortOut sorter;
        InitMoeMrgSortOut(&sorter, 1, GetSortLen<float>(perListElements));
        sorter.Init(&mrgsortParam, pipe);
        sorter.Process();
        SyncAll();
    }

    __aicore__ inline void Process()
    {
        VBSProcess();
        SortOutProcess();
    }
};

class ExpertTokensCount {
private:
    GlobalTensor<int32_t> sortedExpertIdxGm_;
    GlobalTensor<int32_t> expertCountTempGm_;
    GlobalTensor<int64_t> expertTokensCountGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;
    TPipe *pipe_;

    TQue<QuePosition::VECIN, 1> sortedExpertIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> expertCountOutToTempQueue_;
    TQue<QuePosition::VECIN, 1> expertCountTempInQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxCountOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertTotalCountQueue_;

    const MoeTokensCountTilingData *expertTokensCountTilingData_;
    int64_t perCoreElements_ = 0;
    int64_t curCoreElements_ = 0;
    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t actualExpertNum_ = 0;
    int64_t coreLoopsNum_ = 0;
    int64_t perCorePerLoopElements_ = 0;
    int64_t perCoreLastLoopElements_ = 0;
    int64_t actualExpertTotalNum_ = 0;
public:
    __aicore__ inline ExpertTokensCount(){};

    __aicore__ inline void Init(__gm__ int64_t *expertTokensCount, __gm__ int32_t *workspace,
                                MoeInitRoutingTilingData *tilingData, TPipe *tPipe)
    {
        pipe_ = tPipe;
        expertTokensCountTilingData_ = &(tilingData->countTilingData);
        perCoreElements_ = expertTokensCountTilingData_->perCoreElements;
        expertStart_ = tilingData->expertStart;
        expertEnd_ = tilingData->expertEnd;
        actualExpertNum_ = expertEnd_ - expertStart_;

        curCoreElements_ = expertTokensCountTilingData_->perCoreElements;
        coreLoopsNum_ = expertTokensCountTilingData_->perCoreLoops;
        perCorePerLoopElements_ = expertTokensCountTilingData_->perCorePerLoopElements;
        perCoreLastLoopElements_ = expertTokensCountTilingData_->perCoreLastLoopElements;

        expertTokensCountGm_.SetGlobalBuffer(expertTokensCount, actualExpertNum_);
        sortedExpertIdxGm_.SetGlobalBuffer(workspace, curCoreElements_);

        int64_t expertIdxOffset = Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2;
        int64_t expertCountTempOffset = Align(actualExpertNum_, sizeof(int32_t));
        expertCountTempGm_.SetGlobalBuffer(workspace + expertIdxOffset, actualExpertNum_);
        expertTotalCountGm_.SetGlobalBuffer(workspace + expertIdxOffset + expertCountTempOffset, actualExpertNum_);
       
        int64_t sortedExpertIdxInLen = Max(perCorePerLoopElements_, perCoreLastLoopElements_);
        pipe_->InitBuffer(sortedExpertIdxInQueue_, 1, AlignBytes(sortedExpertIdxInLen, sizeof(int32_t)));
        pipe_->InitBuffer(expertCountOutToTempQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
        pipe_->InitBuffer(expertCountTempInQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
        pipe_->InitBuffer(expertIdxCountOutQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int64_t)));
        pipe_->InitBuffer(expertTotalCountQueue_, 1, AlignBytes(1, sizeof(int32_t)));
        
        InitGlobalMemory(expertTotalCountGm_, 1, 0);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }

    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < coreLoopsNum_; i++) {
            int64_t perLoopElements = (i == (coreLoopsNum_ - 1)) ? perCoreLastLoopElements_ : perCorePerLoopElements_;
            CopyIn(i, perLoopElements);
            Compute(perLoopElements);
            CopyOut();
        }
        CopyOutExpertTotalCount();

        ExpertCountCopyIn();
        ExpertCountCompute();
        ExpertCountCopyOut();
    }

private:
    __aicore__ inline void CopyIn(int64_t loop, int64_t curLoopElements)
    {
        LocalTensor<int32_t> sortedExpertIdxInLocal = sortedExpertIdxInQueue_.AllocTensor<int32_t>();
        DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopElements * sizeof(int32_t)),
                                         0, 0, 0};
        DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
        int64_t sortedexpertIdxOffset = loop * perCorePerLoopElements_;
        DataCopyPad(sortedExpertIdxInLocal, sortedExpertIdxGm_[sortedexpertIdxOffset], dataCopyParams, dataCopyPadParams);
        sortedExpertIdxInQueue_.EnQue(sortedExpertIdxInLocal);
    }

    __aicore__ inline void Compute(int64_t curLoopElements)
    {
        LocalTensor<int32_t> sortedExpertIdxInLocal = sortedExpertIdxInQueue_.DeQue<int32_t>();
        LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.AllocTensor<int32_t>();
        Duplicate(expertCountOutLocal, static_cast<int32_t>(0), static_cast<int32_t>(actualExpertNum_));
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

        int64_t i = 0;
        int32_t lastExpertId = sortedExpertIdxInLocal.GetValue(0);
        int32_t lastIndex = 0;

        for (i = 1; i < curLoopElements; i++) {
            if ((lastExpertId >= expertEnd_) || (lastExpertId < expertStart_)) {
                break;
            }
            int32_t curExpertId = sortedExpertIdxInLocal.GetValue(i);
            if (curExpertId != lastExpertId || curExpertId >= expertEnd_) {
                expertCountOutLocal.SetValue(lastExpertId - expertStart_, i - lastIndex);
                actualExpertTotalNum_ += i - lastIndex;
                lastIndex = i;
                lastExpertId = curExpertId;
            }
        }

        if ((i == curLoopElements) && ((lastExpertId >= expertStart_) && (lastExpertId < expertEnd_))) {
            expertCountOutLocal.SetValue(lastExpertId - expertStart_, i - lastIndex);
            actualExpertTotalNum_ += i - lastIndex;
        }

        expertCountOutToTempQueue_.EnQue<int32_t>(expertCountOutLocal);
        sortedExpertIdxInQueue_.FreeTensor(sortedExpertIdxInLocal);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.DeQue<int32_t>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)),
                                     0, 0, 0};
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(expertCountTempGm_, expertCountOutLocal, copyParams);
        expertCountOutToTempQueue_.FreeTensor(expertCountOutLocal);
    }

    __aicore__ inline void CopyOutExpertTotalCount()
    {
        LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.AllocTensor<int32_t>();
        DataCopyExtParams copyTotalCountParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        expertTotalCountLocal.SetValue(0, static_cast<int32_t>(actualExpertTotalNum_));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(expertTotalCountGm_, expertTotalCountLocal, copyTotalCountParams);
        expertTotalCountQueue_.FreeTensor(expertTotalCountLocal);
    }

    __aicore__ inline void ExpertCountCopyIn()
    {
        LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.AllocTensor<int32_t>();
        DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                         static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
        DataCopyPad(expertCountTempInLocal, expertCountTempGm_, dataCopyParams, dataCopyPadParams);
        expertCountTempInQueue_.EnQue(expertCountTempInLocal);
    }

    __aicore__ inline void ExpertCountCompute()
    {
        LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.DeQue<int32_t>();
        LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.AllocTensor<int64_t>();
        Cast(expertCountOutLocal, expertCountTempInLocal, RoundMode::CAST_NONE, actualExpertNum_);
        expertIdxCountOutQueue_.EnQue<int64_t>(expertCountOutLocal);
        expertCountTempInQueue_.FreeTensor(expertCountTempInLocal);
    }

    __aicore__ inline void ExpertCountCopyOut()
    {
        LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.DeQue<int64_t>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(actualExpertNum_ * sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(expertTokensCountGm_, expertCountOutLocal, copyParams);
        expertIdxCountOutQueue_.FreeTensor(expertCountOutLocal);
    }
};

class GatherOut {
private:
    GlobalTensor<float> xGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> expandedScaleGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    TQue<QuePosition::VECIN, 1> expandedRowIdxCopyInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> xCopyInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> scaleCopyInQueue_;

    TPipe *pipe_;
    int64_t cols_ = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;
    int64_t indicesLoops_ = 0;
    int64_t curCoreIndicesElements_ = 0;
    int64_t curCorePerLoopIndicesElements_ = 0;
    int64_t curCoreLastLoopIndicesElements_ = 0;
    int64_t perCoreIndicesElements_ = 0;
    int64_t perCorePerLoopIndicesElements_ = 0;
    int64_t colsLoops_ = 0;
    int64_t perLoopCols_  = 0;
    int64_t lastLoopCols_ = 0;
    int64_t expertTotalCount_ = 0;
    MoeInitRoutingTilingData *tilingData_;

public:
    __aicore__ inline GatherOut()
    {}
    
    __aicore__ inline void Init(__gm__ float *x, __gm__ float *scale, __gm__ int32_t *workspace, 
                                __gm__ int32_t *expandedRowIdx, __gm__ float *expandedX, __gm__ float *expandedScale, 
                                MoeInitRoutingTilingData *tilingData, TPipe *tPipe)
    {
        tilingData_ = tilingData;
        pipe_ = tPipe;
        n_ = tilingData_->n;
        k_ = tilingData_->k;
        cols_ = tilingData_->cols;

        colsLoops_ = tilingData_->gatherTilingData.colsLoops;
        perLoopCols_ = tilingData_->gatherTilingData.perLoopCols;
        lastLoopCols_ = tilingData_->gatherTilingData.lastLoopCols;

        expertTotalCountGm_.SetGlobalBuffer(workspace + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                            Align((tilingData_->expertEnd - tilingData_->expertStart), 
                                            sizeof(int32_t)), 1);

        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            expertTotalCountGm_);
        expertTotalCount_ = expertTotalCountGm_.GetValue(0);

        perCorePerLoopIndicesElements_ = tilingData_->gatherTilingData.perCorePerLoopIndicesElements;
        perCoreIndicesElements_ = expertTotalCount_;

        curCoreIndicesElements_ = perCoreIndicesElements_;
        curCorePerLoopIndicesElements_ = Min(perCorePerLoopIndicesElements_, curCoreIndicesElements_);

        indicesLoops_ = Ceil(curCoreIndicesElements_, curCorePerLoopIndicesElements_);
        curCoreLastLoopIndicesElements_ = curCoreIndicesElements_ - (indicesLoops_ - 1) * curCorePerLoopIndicesElements_;

        pipe_->InitBuffer(expandedRowIdxCopyInQueue_, 1, AlignBytes(curCorePerLoopIndicesElements_, sizeof(int32_t)));
        pipe_->InitBuffer(xCopyInQueue_, 1, AlignBytes(perLoopCols_, sizeof(float)));
        pipe_->InitBuffer(scaleCopyInQueue_, 1, AlignBytes(1, sizeof(float)));

        xGm_.SetGlobalBuffer(x, n_ * cols_);
        scaleGm_.SetGlobalBuffer(scale, n_);
        expandedXGm_.SetGlobalBuffer(expandedX, curCoreIndicesElements_ * cols_);
        expandedScaleGm_.SetGlobalBuffer(expandedScale, curCoreIndicesElements_);
        expandedRowIdxGm_.SetGlobalBuffer(expandedRowIdx, Align(curCoreIndicesElements_, sizeof(int32_t)));
    }

    __aicore__ inline void CopyExpertIn(int64_t curExpertLoopOffset, int64_t curLoopElements)
    {
        LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.AllocTensor<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopElements * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(subRowIdxLocal, expandedRowIdxGm_[curExpertLoopOffset], copyParams, padParams);
        expandedRowIdxCopyInQueue_.EnQue(subRowIdxLocal);
    }

    __aicore__ inline void CopyX(int64_t xSrcOffset, int64_t xDstOffset, int64_t curLoopCols)
    {
        LocalTensor<float> xLocal = xCopyInQueue_.AllocTensor<float>();
        DataCopyExtParams copyInParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopCols * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[xSrcOffset], copyInParams, padParams);
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(curLoopCols * sizeof(float)), 0, 0, 0};
        DataCopyPad(expandedXGm_[xDstOffset], xLocal, copyOutParams);
        xCopyInQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void Process()
    {
        int64_t curLoopElements = curCorePerLoopIndicesElements_;
        for (int64_t indicesLoop = 0; indicesLoop < indicesLoops_; indicesLoop++) {
            if (indicesLoop == indicesLoops_ - 1) {
                curLoopElements = curCoreLastLoopIndicesElements_;
            }
            int64_t curExpertLoopOffset = indicesLoop * curCorePerLoopIndicesElements_;
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            CopyExpertIn(curExpertLoopOffset, curLoopElements);

            LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.DeQue<int32_t>();
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

            for (int64_t indicesIndex = 0; indicesIndex < curLoopElements; indicesIndex++) {
                int64_t rowIdx = subRowIdxLocal.GetValue(indicesIndex);
                int64_t xSrcOffset = rowIdx / k_ * cols_;
                int64_t scaleSrcOffset = rowIdx / k_;
                int64_t xDstOffset = (curExpertLoopOffset + indicesIndex) * cols_;
                SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);

                // inputscale is not supported yet
                int64_t curLoopCols = perLoopCols_;
                for (int64_t colsLoop = 0; colsLoop < colsLoops_; colsLoop++) {
                    if (colsLoop == colsLoops_ - 1) {
                        curLoopCols = lastLoopCols_;
                    }
                    int64_t colsLoopOffset = colsLoop * perLoopCols_;
                    CopyX(xSrcOffset + colsLoopOffset, xDstOffset + colsLoopOffset, curLoopCols);
                }
            }
            expandedRowIdxCopyInQueue_.FreeTensor(subRowIdxLocal);
        }
    }
};

__global__ __aicore__ __vector__ void MoeInitRouting(
    __gm__ float *x, __gm__ int32_t *expertIdx, __gm__ float *scale, __gm__ float *offset, 
    __gm__ int32_t *workspace, 
    __gm__ float *expandedX, __gm__ int32_t *expandedRowIdx,
    __gm__ int64_t *expertTokensCountOrCumsum, __gm__ float *expandedScale, 
    MoeInitRoutingTilingData tiling)
{
    TPipe sortPipe;
    ExpertIdxSort sort;
    sort.Init(expertIdx, workspace, expandedRowIdx, &tiling, &sortPipe);
    sort.Process();
    sortPipe.Destroy();

    TPipe countPipe;
    ExpertTokensCount countOp;
    countOp.Init(expertTokensCountOrCumsum, workspace, &tiling, &countPipe);
    countOp.Process();
    countPipe.Destroy();

    // gather mode is not supported yet
    // gatherout
    TPipe gatherPipe;
    GatherOut gatherOp;
    gatherOp.Init(x, scale, workspace, expandedRowIdx, expandedX, expandedScale, &tiling, &gatherPipe);
    gatherOp.Process();
    gatherPipe.Destroy();
}

void CalGatherTiling(MoeInitRoutingTilingData *tilingData)
{
    auto *gatherOutTiling = &(tilingData->gatherTilingData);
    int64_t totalLength = tilingData->n * tilingData->k;
    int64_t perCoreIndicesElements = totalLength;

    int64_t inputXDtypeSize = sizeof(float);
    int64_t perLoopCols = tilingData->cols;
    int64_t inputXSize = AlignBytes(perLoopCols, inputXDtypeSize);
    int64_t inputScaleSize = BLOCK_BYTES;
    int64_t perLoopMaxIndicesElements = 
        (tilingData->ubSize - inputXSize - inputScaleSize) / static_cast<int64_t>(sizeof(int32_t));
    while (perLoopMaxIndicesElements <= 0) {
        perLoopCols = CeilDiv(perLoopCols, 2);
        inputXSize = AlignBytes(perLoopCols, inputXDtypeSize);
        perLoopMaxIndicesElements = 
            (tilingData->ubSize - inputXSize - inputScaleSize) / static_cast<int64_t>(sizeof(int32_t));
    }
    int64_t colsLoops = CeilDiv(tilingData->cols, perLoopCols);
    int64_t lastLoopCols = tilingData->cols - (colsLoops - 1) * perLoopCols;
    gatherOutTiling->perCoreIndicesElements = perCoreIndicesElements;
    gatherOutTiling->colsLoops = colsLoops;
    gatherOutTiling->perLoopCols = perLoopCols;
    gatherOutTiling->lastLoopCols = lastLoopCols;

    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreIndicesLoops = perCoreIndicesLoops;
    gatherOutTiling->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreLastLoopIndicesElements = perCoreLastLoopIndicesElements;
}

void CalCountTiling(MoeInitRoutingTilingData *tilingData)
{
    auto *tokensCountTiling = &(tilingData->countTilingData);
    int64_t totalElements = tilingData->n * tilingData->k;
    int64_t perCoreElements = totalElements;

    tokensCountTiling->perCoreElements = perCoreElements;

    int64_t expertNumElement = tilingData->expertEnd - tilingData->expertStart;
    int64_t maxElementsPerLoop =
        (tilingData->ubSize - CeilAlign(expertNumElement, UB_BLOCK_SIZE) *
            (static_cast<int64_t>(sizeof(int32_t)) * 2 + static_cast<int64_t>(sizeof(int64_t))) -
                UB_BLOCK_SIZE) / static_cast<int64_t>(sizeof(int32_t));
    int64_t perCoreLoops = CeilDiv(perCoreElements, maxElementsPerLoop);
    int64_t perCorePerLoopElements = CeilDiv(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;
    tokensCountTiling->perCoreLoops = perCoreLoops;
    tokensCountTiling->perCorePerLoopElements = perCorePerLoopElements;
    tokensCountTiling->perCoreLastLoopElements = perCoreLastLoopElements;
}

void CalSortTiling(MoeInitRoutingTilingData *tilingData)
{
    // Tiling4VBSCompute
    int64_t queueNum = 4; // sortDataCopyInQueue|sortDataCopyOutQueue|sortedBuffer|tempBuffer
    int64_t sortLoopMaxElement = tilingData->ubSize / (queueNum * KV_FACTOR * MRG_LIST_NUM) / 
        ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;
    // 限制单核排序的元素个数在AscendC::Sort全排序的能力范围内
    sortLoopMaxElement = std::min(sortLoopMaxElement, SORT_API_MAX_ELEM); 

    int64_t totalLength = tilingData->n * tilingData->k;
    auto *vbsTiling = &(tilingData->vbsComputeTilingData);
    vbsTiling->oneLoopMaxElements = sortLoopMaxElement;

    int64_t perCoreElements = totalLength;
    vbsTiling->perCoreElements = perCoreElements;
    vbsTiling->perCoreLoops = CeilDiv(vbsTiling->perCoreElements, sortLoopMaxElement);
    vbsTiling->perCorePerLoopElements = std::min(vbsTiling->perCoreElements, sortLoopMaxElement);
    vbsTiling->perCoreLastLoopElements =
        vbsTiling->perCoreElements - (vbsTiling->perCoreLoops - 1) * vbsTiling->perCorePerLoopElements;
    tilingData->sortOutOneLoopMaxElements = MRG_SORT_API_MAX_ELEM;
}

int main(int argc, char* argv[])
{
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    int64_t n = (argc > 1) ? std::stoll(argv[1]) : 2048;
    int64_t k = (argc > 2) ? std::stoll(argv[2]) : 8;
    int64_t c = (argc > 3) ? std::stoll(argv[3]) : 32;

    MoeInitRoutingTilingData tilingData;
    tilingData.n = n;
    tilingData.cols = c;
    tilingData.k = k;
    tilingData.expertStart = 0;
    tilingData.expertEnd = 8;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = 1;
    tilingData.ubSize = ubSize;
    tilingData.coreNum = coreNum;

    CalSortTiling(&tilingData);
    CalCountTiling(&tilingData);
    CalGatherTiling(&tilingData);

    float *xDevice;
    float *scaleDevice;
    float *offsetDevice;
    float *expandedXDevice;
    float *expandedScaleDevice;
    float *expandedXHost;
    float *expandedScaleHost;
    int32_t *expertIdxDevice;
    int32_t *workspaceDevice;
    int32_t *expandedRowIdxDevice;
    int32_t *expandedRowIdxHost;
    int64_t *tokenCountDevice;
    int64_t *tokenCountHost;

    int64_t totalLength = n * k;
    int64_t xSize = n * c * sizeof(float);
    int64_t expertIdxSize = totalLength * sizeof(int32_t);
    int64_t scaleSize = n * sizeof(float);
    int64_t offsetSize = n * sizeof(float);
    int64_t expandedXSize = totalLength * c * sizeof(float);
    int64_t actualExpertNum = tilingData.expertEnd - tilingData.expertStart;
    int64_t tokenCountSize = actualExpertNum * sizeof(int64_t);

    int64_t workspaceSize = 0;
    // 排序需要的空间：sortedexpertIdxGm、expandedRowIdx and workspaceGms[2] * KV_FACTOR for mrgsort middle compute
    int64_t sortWorkspaceSize = totalLength * sizeof(float) * SORT_BUFFER_FACTOR;
    int64_t scatterWorkspaceSize = totalLength * sizeof(int32_t);
    int64_t expertTokensCountWorkspaceSize = actualExpertNum * sizeof(int32_t);
    int64_t expertTokenTotalCountWorkspace = AlignBytes(1, sizeof(int32_t));

    workspaceSize = sortWorkspaceSize + scatterWorkspaceSize +
                    expertTokensCountWorkspaceSize + expertTokenTotalCountWorkspace;

    CHECK_ACL(aclrtMalloc((void **)&xDevice, xSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&expertIdxDevice, expertIdxSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&scaleDevice, scaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&offsetDevice, offsetSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&expandedXDevice, expandedXSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&expandedRowIdxDevice, expertIdxSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&tokenCountDevice, tokenCountSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&expandedScaleDevice, scaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost((void **)&expandedXHost, expandedXSize));
    CHECK_ACL(aclrtMallocHost((void **)&expandedRowIdxHost, expertIdxSize));
    CHECK_ACL(aclrtMallocHost((void **)&tokenCountHost, tokenCountSize));
    CHECK_ACL(aclrtMallocHost((void **)&expandedScaleHost, scaleSize));

    std::string exeDir = GetExeDir();
    std::vector<float> xData;
    if (!GetDataFromBin(exeDir + "/x.bin", xData)) {
        std::cerr << "Failed to load x.bin" << std::endl;
        return 1;
    }

    std::vector<int32_t> expertIdxData;
    if (!GetDataFromBin(exeDir + "/expert_idx.bin", expertIdxData)) {
        std::cerr << "Failed to load expert_idx.bin" << std::endl;
        return 1;
    }

    // scale and offset is none
    CHECK_ACL(aclrtMemcpy(xDevice, xSize, xData.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(expertIdxDevice, expertIdxSize, expertIdxData.data(), expertIdxSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // kernel call
    CHECK_ACL(aclrtSynchronizeStream(stream));
    MoeInitRouting<<<coreNum, nullptr, stream>>>(xDevice, expertIdxDevice, scaleDevice, offsetDevice,
        workspaceDevice, expandedXDevice, expandedRowIdxDevice, tokenCountDevice, expandedScaleDevice, tilingData);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(expandedXHost, expandedXSize, expandedXDevice, expandedXSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(expandedRowIdxHost, expertIdxSize, expandedRowIdxDevice, expertIdxSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(tokenCountHost, tokenCountSize, tokenCountDevice, tokenCountSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(expandedScaleHost, scaleSize, expandedScaleDevice, scaleSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // write result to files
    if (!WriteDataToBin(exeDir + "/result_expanded_x.bin", expandedXHost, expandedXSize / sizeof(float))) {
        std::cerr << "Failed to write result_expanded_x.bin" << std::endl;
        return 1;
    }
    if (!WriteDataToBin(exeDir + "/result_expanded_row_idx.bin", expandedRowIdxHost, expertIdxSize / sizeof(int32_t))) {
        std::cerr << "Failed to write result_expanded_row_idx.bin" << std::endl;
        return 1;
    }
    if (!WriteDataToBin(exeDir + "/result_expert_token_count.bin", tokenCountHost, tokenCountSize / sizeof(int64_t))) {
        std::cerr << "Failed to write result_expert_token_count.bin" << std::endl;
        return 1;
    }

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(expertIdxDevice));
    CHECK_ACL(aclrtFree(scaleDevice));
    CHECK_ACL(aclrtFree(offsetDevice));
    CHECK_ACL(aclrtFree(workspaceDevice));
    CHECK_ACL(aclrtFree(expandedXDevice));
    CHECK_ACL(aclrtFree(expandedRowIdxDevice));
    CHECK_ACL(aclrtFree(tokenCountDevice));
    CHECK_ACL(aclrtFree(expandedScaleDevice));
    CHECK_ACL(aclrtFreeHost(expandedXHost));
    CHECK_ACL(aclrtFreeHost(expandedRowIdxHost));
    CHECK_ACL(aclrtFreeHost(tokenCountHost));
    CHECK_ACL(aclrtFreeHost(expandedScaleHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}