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
 * \file 5_simt.cpp
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
#include "simt_api/asc_simt.h"

#include "moe_mrgsort.h"
#include "moe_mrgsort_out.h"
#include "moe_sort_base.h"
#include "moe_tiling_def.h"
#include "moe_util.h"
#include "moe_kernel_common.h"

using namespace AscendC;

class ExpertIdxSortOneCore : public MoeSortBase {
public:
    __aicore__ inline ExpertIdxSortOneCore(){};

    __aicore__ inline void Init(__gm__ int32_t *expertIdx,  __gm__ int32_t *workspace, __gm__ int32_t *expandedRowIdx,
        MoeInitRoutingTilingData *tilingData, TPipe *tPipe)
    {
        this->pipe = tPipe;
        this->tileLength = Align(tilingData->vbsComputeTilingData.lastCorePerLoopElements, sizeof(int32_t));
        this->sortNum = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        this->totalLength = tilingData->n * tilingData->k;
        this->n = tilingData->n;
        this->k = tilingData->k;
        this->blockIdx = GetBlockIdx();

        expertStart = tilingData->expertStart;
        expertEnd = tilingData->expertEnd;
        actualExpertNum = expertEnd - expertStart;

        expertIdxGm.SetGlobalBuffer(expertIdx, this->tileLength);
        sortedExpertIdxGm.SetGlobalBuffer(workspace, this->tileLength);
        expandedRowIdxGm.SetGlobalBuffer(expandedRowIdx,  this->tileLength);

        if (this->blockIdx == 0) {
            expertCountTempGm.SetGlobalBuffer(workspace + Align(totalLength, sizeof(int32_t)) * KV_FACTOR, 
                                                                actualExpertNum);
            InitGlobalMemory(expertCountTempGm, actualExpertNum, 0);
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }

        // key and value
        int64_t bufferSize = this->sortNum * sizeof(int32_t) * KV_FACTOR;
        pipe->InitBuffer(sortDataCopyInQueue, PIPELINE_DEPTH, bufferSize);
        pipe->InitBuffer(sortDataCopyOutQueue, PIPELINE_DEPTH, bufferSize);
        pipe->InitBuffer(sortedBuffer, bufferSize);
        pipe->InitBuffer(tempBuffer, bufferSize);
    }

    __aicore__ inline void CopyIn()
    {
        LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
        DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                         static_cast<uint32_t>(this->totalLength * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
        DataCopyPad(inLocal[0], expertIdxGm, dataCopyParams, dataCopyPadParams);
        LocalTensor<int32_t> rowIdxLocal = inLocal[this->sortNum];
        ArithProgression<int32_t>(rowIdxLocal, 0, 1, this->sortNum);
        sortDataCopyInQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
        LocalTensor<int32_t> expertIdx = inLocal[0];
        LocalTensor<float> expertIdxFp32 = expertIdx.ReinterpretCast<float>();
        Cast(expertIdxFp32, expertIdx, RoundMode::CAST_ROUND, this->tileLength);

        uint16_t repeatTimes = Ceil(this->tileLength, FLOAT_REG_TENSOR_LENGTH);
        uint32_t sreg = static_cast<uint32_t>(this->tileLength);
        __ubuf__ float *inUbAddr = (__ubuf__ float *)expertIdxFp32.GetPhyAddr();
        SortVf(inUbAddr, expertStart, sreg, repeatTimes);

        int64_t duplicateNum = this->totalLength % ONE_REPEAT_SORT_NUM;
        if (duplicateNum > 0) {
            int duplicateIndex = this->totalLength - duplicateNum;
            uint64_t mask0 = (UINT64_MAX << duplicateNum) & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
            uint64_t mask[2] = {mask0, 0};
            Duplicate(expertIdxFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
        }

        LocalTensor<float> concatLocal;
        LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum));
        Concat(concatLocal, expertIdxFp32, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);

        LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum));
        LocalTensor<uint32_t> sourceRowLocal;
        sourceRowLocal = inLocal[this->sortNum].ReinterpretCast<uint32_t>();
        Sort<float, true>(sortedLocal, concatLocal, sourceRowLocal, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);

        LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
        LocalTensor<float> sortedExpertForSourceRowLocal = outLocal[0];
        LocalTensor<uint32_t> expandDstToSrcRowLocal = outLocal[this->sortNum].ReinterpretCast<uint32_t>();
        Extract(sortedExpertForSourceRowLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum / ONE_REPEAT_SORT_NUM);
        Muls(sortedExpertForSourceRowLocal, sortedExpertForSourceRowLocal, (float)-1, this->tileLength);

        LocalTensor<int32_t> expertForSourceRowLocalInt32 = sortedExpertForSourceRowLocal.ReinterpretCast<int32_t>();
        Cast(expertForSourceRowLocalInt32, sortedExpertForSourceRowLocal, RoundMode::CAST_ROUND, this->tileLength);
        sortDataCopyOutQueue.EnQue<float>(outLocal);
        sortDataCopyInQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = this->totalLength * sizeof(int32_t);
        DataCopyPad(sortedExpertIdxGm, outLocal[0], intriParams);
        DataCopyPad(expandedRowIdxGm, outLocal[this->sortNum], intriParams);
        sortDataCopyOutQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx < 1) {
            CopyIn();
            Compute();
            CopyOut();
        }
        SyncAll();
    }
};

class ExpertIdxSortMultiCore : public MoeSortBase {
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
    __aicore__ inline ExpertIdxSortMultiCore(){};

    __aicore__ inline void Init(__gm__ int32_t *expertIdx,  __gm__ int32_t *workspace, __gm__ int32_t *expandedRowIdx,
        MoeInitRoutingTilingData *tilingData, TPipe *tPipe)
    {
        this->totalLength = tilingData->n * tilingData->k;
        this->vbsTilingData = &(tilingData->vbsComputeTilingData);
        this->vmsNeedCoreNum = tilingData->vmsNeedCoreNum;
        this->sortOutOneLoopMaxElements = tilingData->sortOutOneLoopMaxElements;

        this->blockIdx = GetBlockIdx();
        this->tileLength = this->vbsTilingData->perCorePerLoopElements;
        this->sortTotalLength = this->vbsTilingData->perCoreElements;
        if (this->blockIdx == tilingData->vbsComputeTilingData.needCoreNum - 1) {
            this->tileLength = this->vbsTilingData->lastCorePerLoopElements;
            this->sortTotalLength = this->vbsTilingData->lastCoreElements;
        }
        this->n = tilingData->n;
        this->k = tilingData->k;

        expertStart = tilingData->expertStart;
        expertEnd = tilingData->expertEnd;
        actualExpertNum = expertEnd - expertStart;

        // VBS param init
        if (this->blockIdx == this->vbsTilingData->needCoreNum - 1) {
            sortCoreLoops = this->vbsTilingData->lastCoreLoops;
            sortCoreLoopElements = this->vbsTilingData->lastCorePerLoopElements;
            sortCoreLastLoopElements = this->vbsTilingData->lastCoreLastLoopElements;
        } else {
            sortCoreLoops = this->vbsTilingData->perCoreLoops;
            sortCoreLoopElements = this->vbsTilingData->perCorePerLoopElements;
            sortCoreLastLoopElements = this->vbsTilingData->perCoreLastLoopElements;
        }

        this->pipe = tPipe;
        int64_t totalLengthAlign = Align(this->totalLength, sizeof(int32_t));
        expertIdxGm.SetGlobalBuffer(expertIdx + this->blockIdx * tilingData->vbsComputeTilingData.perCoreElements, 
                                    this->sortTotalLength);
        sortedExpertIdxGm.SetGlobalBuffer(workspace, totalLengthAlign);
        expandedRowIdxGm.SetGlobalBuffer(expandedRowIdx, totalLengthAlign);

        if (this->blockIdx == 0) {
            expertCountTempGm.SetGlobalBuffer(workspace + Align(n * k, sizeof(int32_t)) * 2, actualExpertNum);
            InitGlobalMemory(expertCountTempGm, actualExpertNum, 0);
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }

        // key and value
        workspaceGms[0].SetGlobalBuffer((__gm__ float *)workspace + totalLengthAlign * 2 + actualExpertNum, 
                                        totalLengthAlign * KV_FACTOR);
        workspaceGms[1].SetGlobalBuffer((__gm__ float *)workspace + totalLengthAlign * (KV_FACTOR + 2) + 
                                        actualExpertNum, totalLengthAlign * KV_FACTOR);

        int64_t bufferSize = Ceil(Max(this->sortOutOneLoopMaxElements * MAX_MRGSORT_LIST, sortCoreLoopElements), 
            ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM * sizeof(int32_t) * KV_FACTOR;
        pipe->InitBuffer(sortDataCopyInQueue, PIPELINE_DEPTH, bufferSize);
        pipe->InitBuffer(sortDataCopyOutQueue, PIPELINE_DEPTH, bufferSize);
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
        int64_t startValue = this->blockIdx * this->vbsTilingData->perCoreElements + inOffset;

        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        ArithProgression<int32_t>(rowIdxLocal, startValue, 1, size);
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
        DataCopy(workspaceGms[0][this->blockIdx * GetSortLen<float>(this->vbsTilingData->perCoreElements) +
                                GetSortLen<float>(progress * sortCoreLoopElements)],
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
        GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex][blockIdx * coreOffset + loopOffset];
        LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();
        LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
        for (int64_t i = 0; i < listNum; i++) {
            LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(this->sortOutOneLoopMaxElements) * i];
            sorter->SetInput(srcWsGm, inLocalT);
        }
        GlobalTensor<float> dstWsGm = workspaceGms[1 - srcWsIndex][blockIdx * coreOffset + loopOffset];
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
        if (this->blockIdx < this->vbsTilingData->needCoreNum) {
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
        SyncAll();
    }

    __aicore__ inline void VMSProcess()
    {
        int64_t currentStageNeedCoreNum = this->vmsNeedCoreNum;
        perListElements = this->vbsTilingData->perCoreElements;
        lastListElements = this->vbsTilingData->lastCoreElements;
        listNum = this->vbsTilingData->needCoreNum;

        for (; listNum > MAX_MRGSORT_LIST;) {
            currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);
            int64_t coreOffset = GetSortLen<float>(perListElements * MAX_MRGSORT_LIST);
            int64_t remainListNum = listNum - (currentStageNeedCoreNum - 1) * MAX_MRGSORT_LIST;

            if (this->blockIdx < currentStageNeedCoreNum - 1) {
                mrgsortParam.perListElements = perListElements;
                mrgsortParam.lastListElements = perListElements;
                mrgsortParam.oneLoopMaxElements = this->sortOutOneLoopMaxElements;
                InitMoeMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, 0);
                mrgsorter.Init(&mrgsortParam);
                mrgsorter.Process();
            } else if (this->blockIdx == currentStageNeedCoreNum - 1) {
                mrgsortParam.perListElements = perListElements;
                mrgsortParam.lastListElements = lastListElements;
                mrgsortParam.oneLoopMaxElements = this->sortOutOneLoopMaxElements;
                InitMoeMrgSort(&mrgsorter, remainListNum, coreOffset, 0);
                mrgsorter.Init(&mrgsortParam);
                mrgsorter.Process();
            }
            listNum = currentStageNeedCoreNum;
            currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);
            srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;

            lastListElements = perListElements * (remainListNum - 1) + lastListElements;
            perListElements = perListElements * MAX_MRGSORT_LIST;

            SyncAll();
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
        if (this->blockIdx < 1) {
            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = lastListElements;
            mrgsortParam.oneLoopMaxElements = this->sortOutOneLoopMaxElements;

            MoeMrgsortOut sorter;
            InitMoeMrgSortOut(&sorter, listNum, GetSortLen<float>(perListElements));
            sorter.Init(&mrgsortParam, pipe);
            sorter.Process();
        }
        SyncAll();
    }

    __aicore__ inline void Process()
    {
        VBSProcess();
        VMSProcess();
        SortOutProcess();
    }
};

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertFirstIndexSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd, __gm__ int32_t *sortedExpertIdGmAddr,
    __ubuf__ int32_t *expertFirstIndexLocalAddr)
{
    auto threadIdx = static_cast<int32_t>(Simt::GetThreadIdx());
    auto threadNum = static_cast<int32_t>(Simt::GetThreadNum());
    for (auto i = threadIdx; i < elementNum; i += threadNum) {
        auto currExpertId = sortedExpertIdGmAddr[i];
        if (currExpertId >= expertEnd) {
            break;
        }
        auto prevExpertId = (i == 0 ? -1 : sortedExpertIdGmAddr[i - 1]);
        if (currExpertId != prevExpertId) {
            expertFirstIndexLocalAddr[currExpertId - expertStart] = i;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertCountOutSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd, __gm__ int32_t *sortedExpertIdGmAddr,
    __ubuf__ int32_t *expertFirstIndexLocalAddr, __ubuf__ int32_t *expertCountOutLocalAddr)
{
    auto threadIdx = static_cast<int32_t>(Simt::GetThreadIdx());
    auto threadNum = static_cast<int32_t>(Simt::GetThreadNum());
    for (auto i = threadIdx; i < elementNum; i += threadNum) {
        auto currExpertId = sortedExpertIdGmAddr[i];
        if (currExpertId >= expertEnd) {
            break;
        }
        if (i == elementNum - 1 || currExpertId != sortedExpertIdGmAddr[i + 1]) {
            expertCountOutLocalAddr[currExpertId - expertStart] =
                i + 1 - expertFirstIndexLocalAddr[currExpertId - expertStart];
        }
    }
}

class ExpertTokensCount {
private:
    GlobalTensor<int32_t> sortedExpertIdxGm_;
    GlobalTensor<int32_t> expertCountTempGm_;
    GlobalTensor<int64_t> expertTokensCountGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    TQue<QuePosition::VECIN, 1> sortedExpertIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> expertCountOutToTempQueue_;
    TQue<QuePosition::VECIN, 1> expertCountTempInQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxCountOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertTotalCountQueue_;

    TPipe *pipe_;
    int64_t blockIdx_;
    int64_t needCoreNum_ = 0;
    int64_t perCoreElements_ = 0;
    int64_t curCoreElements_ = 0;
    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t actualExpertNum_ = 0;
    int64_t coreLoopsNum_ = 0;
    int64_t perCorePerLoopElements_ = 0;
    int64_t perCoreLastLoopElements_ = 0;
    int64_t actualExpertTotalNum_ = 0;
    int64_t expertCountElements_ = 0;
    MoeTokensCountTilingData *expertTokensCountTilingData_;

public:
    __aicore__ inline ExpertTokensCount(){};

    __aicore__ inline void Init(__gm__ int64_t *expertTokensCount, 
                                __gm__ int32_t *workspace,
                                MoeInitRoutingTilingData *tilingData, TPipe *tPipe)
    {
        pipe_ = tPipe;
        blockIdx_ = AscendC::GetBlockIdx();
        expertTokensCountTilingData_ = &(tilingData->countTilingData);

        needCoreNum_ = expertTokensCountTilingData_->needCoreNum;
        perCoreElements_ = expertTokensCountTilingData_->perCoreElements;
        expertStart_ = tilingData->expertStart;
        expertEnd_ = tilingData->expertEnd;
        actualExpertNum_ = expertEnd_ - expertStart_;
        expertCountElements_ = actualExpertNum_;

        if (blockIdx_ == needCoreNum_ - 1) {
            curCoreElements_ = expertTokensCountTilingData_->lastCoreElements;
            coreLoopsNum_ = expertTokensCountTilingData_->lastCoreLoops;
            perCorePerLoopElements_ = expertTokensCountTilingData_->lastCorePerLoopElements;
            perCoreLastLoopElements_ = expertTokensCountTilingData_->lastCoreLastLoopElements;
        } else {
            curCoreElements_ = expertTokensCountTilingData_->perCoreElements;
            coreLoopsNum_ = expertTokensCountTilingData_->perCoreLoops;
            perCorePerLoopElements_ = expertTokensCountTilingData_->perCorePerLoopElements;
            perCoreLastLoopElements_ = expertTokensCountTilingData_->perCoreLastLoopElements;
        }

        expertTokensCountGm_.SetGlobalBuffer(expertTokensCount, expertCountElements_);
        sortedExpertIdxGm_.SetGlobalBuffer(workspace + blockIdx_ * perCoreElements_, curCoreElements_);

        int64_t expertIdxOffset = Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2;
        int64_t expertCountTempOffset = Align(actualExpertNum_, sizeof(int32_t));
        expertCountTempGm_.SetGlobalBuffer(workspace + expertIdxOffset, actualExpertNum_);
        expertTotalCountGm_.SetGlobalBuffer(workspace + expertIdxOffset + expertCountTempOffset, actualExpertNum_);
       
        int64_t sortedExpertIdxInLen = Max(perCorePerLoopElements_, perCoreLastLoopElements_);
        pipe_->InitBuffer(sortedExpertIdxInQueue_, 1, AlignBytes(sortedExpertIdxInLen, sizeof(int32_t)));
        pipe_->InitBuffer(expertCountOutToTempQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
        pipe_->InitBuffer(expertCountTempInQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
        pipe_->InitBuffer(expertIdxCountOutQueue_, 1, AlignBytes(expertCountElements_, sizeof(int64_t)));
        pipe_->InitBuffer(expertTotalCountQueue_, 1, AlignBytes(1, sizeof(int32_t)));

        if (blockIdx_ == 0) {
            InitGlobalMemory(expertTotalCountGm_, 1, 0);
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
        SyncAll();
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ < needCoreNum_) {
            LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.AllocTensor<int32_t>();
            Duplicate(expertCountOutLocal, 0, actualExpertNum_);

            __gm__ int32_t *sortedExpertIdxGmAddr = (__gm__ int32_t *)sortedExpertIdxGm_.GetPhyAddr();
            __ubuf__ int32_t *expertCountOutLocalAddr = (__ubuf__ int32_t *)expertCountOutLocal.GetPhyAddr();

            Simt::VF_CALL<ComputeExpertFirstIndexSimt>(Simt::Dim3{SIMT_THREAD_NUM, 1, 1}, curCoreElements_, expertStart_,
                                                       expertEnd_, sortedExpertIdxGmAddr, expertCountOutLocalAddr);
            Simt::VF_CALL<ComputeExpertCountOutSimt>(Simt::Dim3{SIMT_THREAD_NUM, 1, 1}, curCoreElements_, expertStart_,
                                                     expertEnd_, sortedExpertIdxGmAddr, expertCountOutLocalAddr,
                                                     expertCountOutLocalAddr);
                                    
            expertCountOutToTempQueue_.EnQue<int32_t>(expertCountOutLocal);
            CopyOut();
        }

        SyncAll();
        /* copy expert tokens count result from worksapce to output GM. */
        if (blockIdx_ == 0) {
            ExpertCountCopyIn();
            ExpertCountCompute();
            ExpertCountCopyOut();
        }
        SyncAll();
    }

private:
    __aicore__ inline void CopyOut()
    {
        LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.DeQue<int32_t>();

        DataCopyExtParams copyParams{static_cast<uint16_t>(1), 
                                     static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)),
                                     0, 0, 0};
        SetAtomicAdd<int32_t>();
        DataCopyPad(expertCountTempGm_, expertCountOutLocal, copyParams);
        SetAtomicNone();
        expertCountOutToTempQueue_.FreeTensor(expertCountOutLocal);
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
        LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.AllocTensor<int32_t>();
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        // key value mode is not supported yet
        for (int64_t i = 0; i < actualExpertNum_; i++) {
            int64_t expertCount = static_cast<int64_t>(expertCountTempInLocal.GetValue(i));
            expertCountOutLocal.SetValue(i, expertCount);
            actualExpertTotalNum_ += expertCount;
        }
        expertTotalCountLocal.SetValue(0, static_cast<int32_t>(actualExpertTotalNum_));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        expertIdxCountOutQueue_.EnQue<int64_t>(expertCountOutLocal);
        expertTotalCountQueue_.EnQue<int32_t>(expertTotalCountLocal);
        expertCountTempInQueue_.FreeTensor(expertCountTempInLocal);
    }

    __aicore__ inline void ExpertCountCopyOut()
    {
        LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.DeQue<int64_t>();
        LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.DeQue<int32_t>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(expertCountElements_ * sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(expertTokensCountGm_, expertCountOutLocal, copyParams);
        copyParams.blockLen = sizeof(int32_t);
        DataCopyPad(expertTotalCountGm_, expertTotalCountLocal, copyParams);
        expertIdxCountOutQueue_.FreeTensor(expertCountOutLocal);
        expertTotalCountQueue_.FreeTensor(expertTotalCountLocal);
    }
};

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void GatherOutSimt(
    int32_t curLoopElements, int32_t cols, int32_t k,
    __gm__ float *xGmAddr, __ubuf__ int32_t *expandedRowIdxLocalAddr, __ubuf__ float *xLocalAddr)
{
    auto threadIdx0 = static_cast<int32_t>(Simt::GetThreadIdx<0>());
    auto threadNum0 = static_cast<int32_t>(Simt::GetThreadNum<0>());

    auto threadIdx1 = static_cast<int32_t>(Simt::GetThreadIdx<1>());
    auto threadNum1 = static_cast<int32_t>(Simt::GetThreadNum<1>());

    for (auto i = threadIdx0; i < curLoopElements; i += threadNum0) {
        int64_t rowIdx = expandedRowIdxLocalAddr[i];
        int64_t xSrcOffset = rowIdx * cols;
        int64_t xLocalOffset = i * cols;
        for (auto j = threadIdx1; j < cols; j += threadNum1) {
            xLocalAddr[xLocalOffset + j]  = xGmAddr[xSrcOffset + j];
        }
    }
}

class GatherOut {
private:
    GlobalTensor<float> xGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> expandedScaleGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> expandedRowIdxCopyInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, DOUBLE_BUFFER> xCopyInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, DOUBLE_BUFFER> scaleCopyInQueue_;

    TPipe *pipe_;
    int64_t cols_ = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;
    int64_t blockIdx_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t indicesLoops_ = 0;
    int64_t curCoreIndicesElements_ = 0;
    int64_t curCorePerLoopIndicesElements_ = 0;
    int64_t curCoreLastLoopIndicesElements_ = 0;
    int64_t perCoreIndicesElements_ = 0;
    int64_t lastCoreIndicesElements_ = 0;
    int64_t perCorePerLoopIndicesElements_ = 0;
    int64_t lastCorePerLoopIndicesElements_ = 0;
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
        blockIdx_ = AscendC::GetBlockIdx();
        tilingData_ = tilingData;
        pipe_ = tPipe;
        needCoreNum_ = tilingData_->gatherTilingData.needCoreNum;
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
        lastCorePerLoopIndicesElements_ = tilingData_->gatherTilingData.lastCorePerLoopIndicesElements;
        perCoreIndicesElements_ = Ceil(expertTotalCount_, tilingData->coreNum);
        needCoreNum_ = Ceil(expertTotalCount_, perCoreIndicesElements_);
        lastCoreIndicesElements_ = expertTotalCount_ - (needCoreNum_ - 1) * perCoreIndicesElements_;

        if (blockIdx_ == needCoreNum_ - 1) {
            curCoreIndicesElements_ = lastCoreIndicesElements_;
            curCorePerLoopIndicesElements_ = Min(lastCorePerLoopIndicesElements_, curCoreIndicesElements_);
        } else {
            curCoreIndicesElements_ = perCoreIndicesElements_;
            curCorePerLoopIndicesElements_ = Min(perCorePerLoopIndicesElements_, curCoreIndicesElements_);
        }
        indicesLoops_ = Ceil(curCoreIndicesElements_, curCorePerLoopIndicesElements_);
        curCoreLastLoopIndicesElements_ = curCoreIndicesElements_ - (indicesLoops_ - 1) * curCorePerLoopIndicesElements_;

        pipe_->InitBuffer(expandedRowIdxCopyInQueue_, DOUBLE_BUFFER, 
                          AlignBytes(curCorePerLoopIndicesElements_, sizeof(int32_t)));
        pipe_->InitBuffer(xCopyInQueue_, DOUBLE_BUFFER, AlignBytes(perLoopCols_, sizeof(float)));
        pipe_->InitBuffer(scaleCopyInQueue_, DOUBLE_BUFFER, AlignBytes(1, sizeof(float)));

        xGm_.SetGlobalBuffer(x, n_ * cols_);
        scaleGm_.SetGlobalBuffer(scale, n_);
        expandedXGm_.SetGlobalBuffer(expandedX + blockIdx_ * perCoreIndicesElements_ * cols_,
                                     curCoreIndicesElements_ * cols_);
        expandedScaleGm_.SetGlobalBuffer(expandedScale + blockIdx_ * perCoreIndicesElements_,
                                         curCoreIndicesElements_);
        expandedRowIdxGm_.SetGlobalBuffer(expandedRowIdx + blockIdx_ * perCoreIndicesElements_,
                                          Align(curCoreIndicesElements_, sizeof(int32_t)));
    }

    __aicore__ inline void CopyExpertIn(int64_t curExpertLoopOffset, int64_t curLoopElements)
    {
        LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.AllocTensor<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopElements * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(subRowIdxLocal, expandedRowIdxGm_[curExpertLoopOffset], copyParams, padParams);
        expandedRowIdxCopyInQueue_.EnQue(subRowIdxLocal);
    }

    __aicore__ inline void CopyXOut(int64_t xDstOffset, int64_t curLoopCols)
    {
        LocalTensor<float> xLocal = xCopyInQueue_.DeQue<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopCols * sizeof(float)), 0, 0, 0};
        DataCopyPad(expandedXGm_[xDstOffset], xLocal, copyParams);
        xCopyInQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= needCoreNum_) {
            return;
        }
        
        int64_t curLoopElements = curCorePerLoopIndicesElements_;
        for (int64_t indicesLoop = 0; indicesLoop < indicesLoops_; indicesLoop++) {
            if (indicesLoop == indicesLoops_ - 1) {
                curLoopElements = curCoreLastLoopIndicesElements_;
            }
            int64_t curExpertLoopOffset = indicesLoop * curCorePerLoopIndicesElements_;
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            CopyExpertIn(curExpertLoopOffset, curLoopElements);

            LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.DeQue<int32_t>();
            LocalTensor<float> xLocal = xCopyInQueue_.AllocTensor<float>();

            __gm__ float *xGmAddr = (__gm__ float *)xGm_.GetPhyAddr();
            __ubuf__ int32_t *expandedRowIdxLocalAddr = (__ubuf__ int32_t *)subRowIdxLocal.GetPhyAddr();
            __ubuf__ float *xLocalAddr = (__ubuf__ float *)xLocal.GetPhyAddr();
            
            Simt::VF_CALL<GatherOutSimt>(Simt::Dim3{16, 128, 1}, 
                                        static_cast<int32_t>(curLoopElements), static_cast<int32_t>(cols_), 
                                        static_cast<int32_t>(k_), xGmAddr, expandedRowIdxLocalAddr, xLocalAddr);

            xCopyInQueue_.EnQue(xLocal);
            SyncAll();

            // 一次搬出
            CopyXOut(curExpertLoopOffset * cols_, curLoopElements * cols_);
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
    if (tiling.vbsComputeTilingData.needCoreNum == 1) {
        ExpertIdxSortOneCore sort;
        sort.Init(expertIdx, workspace, expandedRowIdx, &tiling, &sortPipe);
        sort.Process();
    } else {
        ExpertIdxSortMultiCore sort;
        sort.Init(expertIdx, workspace, expandedRowIdx, &tiling, &sortPipe);
        sort.Process();
    }
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
    int64_t perCoreIndicesElements = CeilDiv(totalLength, tilingData->coreNum);
    if (perCoreIndicesElements <= 0) {
        gatherOutTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = CeilDiv(totalLength, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = totalLength - (needCoreNum - 1) * perCoreIndicesElements;
    int64_t inputXDtypeSize = sizeof(float);

    int64_t perLoopCols = tilingData->cols;
    int64_t inputXSize = AlignBytes(perLoopCols, inputXDtypeSize);
    int64_t inputScaleSize = BLOCK_BYTES;
    int64_t perLoopMaxIndicesElements = 
        tilingData->ubSize / DOUBLE_BUFFER / (inputXSize + inputScaleSize + static_cast<int64_t>(sizeof(int32_t)));
    while (perLoopMaxIndicesElements <= 0) {
        perLoopCols = CeilDiv(perLoopCols, 2);
        inputXSize = AlignBytes(perLoopCols, inputXDtypeSize);
        perLoopMaxIndicesElements = 
            tilingData->ubSize / DOUBLE_BUFFER / (inputXSize + inputScaleSize + static_cast<int64_t>(sizeof(int32_t)));
    }
    int64_t colsLoops = CeilDiv(tilingData->cols, perLoopCols);
    int64_t lastLoopCols = tilingData->cols - (colsLoops - 1) * perLoopCols;
    gatherOutTiling->needCoreNum = needCoreNum;
    gatherOutTiling->perCoreIndicesElements = perCoreIndicesElements;
    gatherOutTiling->lastCoreIndicesElements = lastCoreIndicesElements;
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

    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreIndicesLoops = lastCoreIndicesLoops;
    gatherOutTiling->lastCorePerLoopIndicesElements = lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreLastLoopIndicesElements = lastCoreLastLoopIndicesElements;
}

void CalCountTiling(MoeInitRoutingTilingData *tilingData)
{
    auto *tokensCountTiling = &(tilingData->countTilingData);
    int64_t totalElements = tilingData->n * tilingData->k;
    int64_t perCoreElements = CeilDiv(totalElements, tilingData->coreNum);
    int64_t needCoreNum = CeilDiv(totalElements, perCoreElements);
    int64_t lastCoreElements = totalElements - (needCoreNum - 1) * perCoreElements;
    tokensCountTiling->needCoreNum = needCoreNum;
    tokensCountTiling->perCoreElements = perCoreElements;
    tokensCountTiling->lastCoreElements = lastCoreElements;

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

    int64_t lastCoreLoops = CeilDiv(lastCoreElements, maxElementsPerLoop);
    int64_t lastCorePerLoopElements = CeilDiv(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;
    tokensCountTiling->lastCoreLoops = lastCoreLoops;
    tokensCountTiling->lastCorePerLoopElements = lastCorePerLoopElements;
    tokensCountTiling->lastCoreLastLoopElements = lastCoreLastLoopElements;
}

void VBSOneCoreCompute(MoeVBSComputeTilingData *vbsTiling, int64_t totalLength)
{
    vbsTiling->needCoreNum = 1;
    vbsTiling->perCoreElements = totalLength;
    vbsTiling->perCoreLoops = 1;
    vbsTiling->perCorePerLoopElements = vbsTiling->perCoreElements;
    vbsTiling->perCoreLastLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLoops = 1;
    vbsTiling->lastCorePerLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLastLoopElements = vbsTiling->perCoreElements;
}

void VBSMultiCoreCompute(MoeVBSComputeTilingData *vbsTiling, int64_t totalLength, 
                            int64_t sortLoopMaxElement, int64_t coreNum)
{
    int64_t needCoreNum = CeilDiv(totalLength, sortLoopMaxElement);
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));
    needCoreNum = std::min(needCoreNum, coreNum);

    int64_t perCoreElements = (needCoreNum == 0) ? 0 : (totalLength / needCoreNum);
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % ONE_REPEAT_SORT_NUM;
    int64_t lastCoreElement = totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + ONE_REPEAT_SORT_NUM - perCoreElements % ONE_REPEAT_SORT_NUM;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = CeilDiv(totalLength, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    vbsTiling->needCoreNum = needCoreNum;
    do {
        vbsTiling->perCoreElements = perCoreElements;
        vbsTiling->perCoreLoops = CeilDiv(vbsTiling->perCoreElements, sortLoopMaxElement);
        vbsTiling->perCorePerLoopElements = std::min(vbsTiling->perCoreElements, sortLoopMaxElement);

        vbsTiling->perCoreLastLoopElements =
            vbsTiling->perCoreElements - (vbsTiling->perCoreLoops - 1) * vbsTiling->perCorePerLoopElements;

        vbsTiling->lastCoreElements = totalLength - (vbsTiling->needCoreNum - 1) * vbsTiling->perCoreElements;
        vbsTiling->lastCoreLoops = vbsTiling->perCoreLoops;
        int64_t lastCorePerLoopElements = CeilDiv(CeilDiv(vbsTiling->lastCoreElements, vbsTiling->lastCoreLoops),
                                            ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        vbsTiling->lastCorePerLoopElements = lastCorePerLoopElements;
        vbsTiling->lastCoreLastLoopElements =
            vbsTiling->lastCoreElements - (vbsTiling->lastCoreLoops - 1) * vbsTiling->lastCorePerLoopElements;
        perCoreElements -= ONE_REPEAT_SORT_NUM;
    } while (vbsTiling->lastCoreLastLoopElements <= 0 && perCoreElements > 0);
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

    if (totalLength <= sortLoopMaxElement) { // 排序只用到一个核排序
        VBSOneCoreCompute(vbsTiling, totalLength);
    } else {
        VBSMultiCoreCompute(vbsTiling, totalLength, sortLoopMaxElement, tilingData->coreNum);
    }

    // Tiling4VMSMiddleCompute
    if (vbsTiling->needCoreNum <= MRG_LIST_NUM) { // 队列数小于一次vms则没有中间归并
        tilingData->vmsNeedCoreNum = 0;
    } else {
        tilingData->vmsNeedCoreNum = CeilDiv(vbsTiling->needCoreNum, MRG_LIST_NUM);
    }

    // Tiling4SortOutCompute
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
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    tilingData.ubSize = ubSize - SIMT_DCACHE_SIZE;
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
    int64_t coreSyncWorkspaceSize = coreNum * UB_BLOCK_SIZE * 2; // 多核同步需要的空间
    int64_t scatterWorkspaceSize = totalLength * sizeof(int32_t);
    int64_t expertTokensCountWorkspaceSize = actualExpertNum * sizeof(int32_t);
    int64_t expertTokenTotalCountWorkspace = AlignBytes(1, sizeof(int32_t));

    workspaceSize = sortWorkspaceSize + coreSyncWorkspaceSize + scatterWorkspaceSize +
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
    MoeInitRouting<<<coreNum, tilingData.ubSize, stream>>>(xDevice, expertIdxDevice, scaleDevice, offsetDevice,
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