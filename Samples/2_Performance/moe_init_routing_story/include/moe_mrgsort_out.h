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
 * \file moe_mrgsort_out.h
 * \brief
 */

#ifndef MOE_MRGSORT_OUT_H
#define MOE_MRGSORT_OUT_H

#include "moe_mrgsort.h"
#include "kernel_operator.h"

using namespace AscendC;

class MoeMrgsortOut {
public:
    __aicore__ inline MoeMrgsortOut(){};
    __aicore__ inline void Init(MoeMrgsortParam *param, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void SetInput(GlobalTensor<float> &gmInput, LocalTensor<float> &ubInput);
    __aicore__ inline void SetOutput(GlobalTensor<int32_t> &gmOutput1, GlobalTensor<int32_t> &gmOutput2,
                                     LocalTensor<float> &ubOutput1, LocalTensor<float> &ubOutput2);
    __aicore__ inline void SetBuffer(LocalTensor<float> &tempBuffer);

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void UpdateMrgParam();
    __aicore__ inline void MrgsortCompute();
    __aicore__ inline void UpdateSortInfo();
    __aicore__ inline void Extract();
    __aicore__ inline void CopyOut();
    __aicore__ inline void ClearCache();

private:
    MoeMrgsortParam *param = nullptr;

    GlobalTensor<float> gmInputs[4];
    GlobalTensor<int32_t> gmOutput1;
    GlobalTensor<int32_t> gmOutput2;

    LocalTensor<float> ubInputs[4];
    LocalTensor<float> tempBuffer;

    // for extract
    LocalTensor<float> ubOutput1;
    LocalTensor<uint32_t> ubOutput2;

    // for copy out
    LocalTensor<int32_t> ubOutputInt1;
    LocalTensor<int32_t> ubOutputInt2;

    int64_t listNum{0};
    int64_t remainListNum{0};
    int64_t outOffset{0};
    int64_t offsets[4];
    int64_t listRemainElements[4];
    int64_t lengths[4];
    int64_t allRemainElements{0};
    int64_t curLoopSortedNum{0};

    // for MrgSort
    uint16_t validBitTail;
    uint16_t elementCountListTail[4];
    uint32_t listSortedNums[4];
    LocalTensor<float> *tmpUbInputs[4];

    // 核内流水同步 mutex（ISASI），取代原 SetFlag/WaitFlag
    uint8_t sortMte3Mte2Mutex_;   // MTE3 -> 下一轮 MTE2
    uint8_t sortMte2VMutex_;      // MTE2 -> V
    uint8_t sortVMte3Mutex_;      // V -> MTE3
};

__aicore__ inline void MoeMrgsortOut::ClearCache()
{
    this->listNum = 0;
    this->allRemainElements = 0;
    this->outOffset = 0;
}

__aicore__ inline void MoeMrgsortOut::SetInput(GlobalTensor<float> &gmInput, LocalTensor<float> &ubInput)
{
    this->gmInputs[listNum].SetGlobalBuffer(gmInput.GetPhyAddr(0), gmInput.GetSize());
    this->ubInputs[listNum] = ubInput;
    this->listNum += 1;
}

__aicore__ inline void MoeMrgsortOut::SetOutput(GlobalTensor<int32_t> &gmOutput1, GlobalTensor<int32_t> &gmOutput2,
                                                LocalTensor<float> &ubOutput1, LocalTensor<float> &ubOutput2)
{
    this->gmOutput1.SetGlobalBuffer(gmOutput1.GetPhyAddr(0), gmOutput1.GetSize());
    this->ubOutput1 = ubOutput1;
    this->ubOutputInt1 = ubOutput1.ReinterpretCast<int32_t>();

    this->gmOutput2.SetGlobalBuffer(gmOutput2.GetPhyAddr(0), gmOutput2.GetSize());
    this->ubOutput2 = ubOutput2.ReinterpretCast<uint32_t>();
    this->ubOutputInt2 = ubOutput2.ReinterpretCast<int32_t>();
}

__aicore__ inline void MoeMrgsortOut::SetBuffer(LocalTensor<float> &tempBuffer)
{
    this->tempBuffer = tempBuffer;
}

__aicore__ inline void MoeMrgsortOut::UpdateMrgParam()
{
    if (this->remainListNum == MERGE_LIST_TWO) {
        elementCountListTail[MERGE_LIST_IDX_TWO] = 0;
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0011;
    } else if (this->remainListNum == MERGE_LIST_THREE) {
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0111;
    } else if (this->remainListNum == MERGE_LIST_FOUR) {
        validBitTail = 0b1111;
    } else {
        validBitTail = 0b0001;
    }
}

__aicore__ inline void MoeMrgsortOut::CopyIn()
{
    this->remainListNum = 0;
    SyncMte3ToMte2(sortMte3Mte2Mutex_); // 局部 MTE3->MTE2 fence（等价原 HardEvent MTE3_MTE2 的 SetWaitFlag）
    AscendC::Mutex::Lock<PIPE_MTE2>(sortMte2VMutex_);    // 拷贝前先锁（环形平衡：等上一轮 V 消费完输入）
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        lengths[i] = Min(param->oneLoopMaxElements, listRemainElements[i]);
        if (lengths[i] > 0) {
            DataCopy(this->ubInputs[i], this->gmInputs[i][offsets[i]],
                     Align(GetSortLen<float>(lengths[i]), sizeof(float)));
            tmpUbInputs[j] = &this->ubInputs[i];
            elementCountListTail[j] = lengths[i];
            this->remainListNum += 1;
            j++;
        }
    }
    AscendC::Mutex::Unlock<PIPE_MTE2>(sortMte2VMutex_); // producer 侧：DataCopy 之后，通知 V 输入就绪
}

__aicore__ inline void MoeMrgsortOut::MrgsortCompute()
{
    AscendC::Mutex::Lock<PIPE_V>(sortMte2VMutex_); // V 等 MTE2 数据就绪（原 MTE2_V）
    AscendC::Mutex::Lock<PIPE_V>(sortVMte3Mutex_); // V 等上一轮 MTE3 写完输出（护 ubOutput 的 WAR）
    if (this->remainListNum == MERGE_LIST_TWO) {
        MrgSortSrcList sortListTail = MrgSortSrcList(*tmpUbInputs[0], *tmpUbInputs[1], *tmpUbInputs[0],
                                                     *tmpUbInputs[0]);
        MrgSort<float, true>(this->tempBuffer, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
    } else if (this->remainListNum == MERGE_LIST_THREE) {
        MrgSortSrcList sortListTail =
            MrgSortSrcList(*tmpUbInputs[0], *tmpUbInputs[1], *tmpUbInputs[MERGE_LIST_IDX_TWO], *tmpUbInputs[0]);
        MrgSort<float, true>(this->tempBuffer, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
    } else if (this->remainListNum == MERGE_LIST_FOUR) {
        MrgSortSrcList sortListTail = MrgSortSrcList(*tmpUbInputs[0], *tmpUbInputs[1], *tmpUbInputs[MERGE_LIST_IDX_TWO],
                                                     *tmpUbInputs[MERGE_LIST_IDX_THREE]);
        MrgSort<float, true>(this->tempBuffer, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
    } else {
        DataCopy(this->tempBuffer, *this->tmpUbInputs[0],
                 Align(GetSortLen<float>(elementCountListTail[0]), sizeof(float)));
        listSortedNums[0] = elementCountListTail[0];
    }
    AscendC::Mutex::Unlock<PIPE_V>(sortMte2VMutex_); // 输入消费完（环平衡，下一轮 MTE2 才能覆盖输入）
}

__aicore__ inline void MoeMrgsortOut::UpdateSortInfo()
{
    curLoopSortedNum = 0;
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        if (lengths[i] > 0) {
            // update remain size
            listRemainElements[i] -= listSortedNums[j];
            allRemainElements -= listSortedNums[j];
            // update offset
            offsets[i] += GetSortOffset<float>(listSortedNums[j]);
            // update current loop sorted nums
            curLoopSortedNum += listSortedNums[j];
            j += 1;
        }
    }
}

__aicore__ inline void MoeMrgsortOut::Extract()
{
    AscendC::Extract(this->ubOutput1, this->ubOutput2, this->tempBuffer, Ceil(curLoopSortedNum, ONE_REPEAT_SORT_NUM));
    Muls(this->ubOutput1, this->ubOutput1, (float)-1, Align(curLoopSortedNum, sizeof(float)));
    Cast(this->ubOutputInt1, this->ubOutput1, RoundMode::CAST_ROUND, Align(curLoopSortedNum, sizeof(float)));
    AscendC::Mutex::Unlock<PIPE_V>(sortVMte3Mutex_); // 输出就绪（原 V_MTE3 producer 侧）：须在 Extract 的 V 指令之后
}

__aicore__ inline void MoeMrgsortOut::CopyOut()
{
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = curLoopSortedNum * sizeof(int32_t);
    AscendC::Mutex::Lock<PIPE_MTE3>(sortVMte3Mutex_); // MTE3 等 V 输出就绪（原 V_MTE3 consumer 侧）
    DataCopyPad(this->gmOutput1[outOffset], this->ubOutputInt1, intriParams);
    DataCopyPad(this->gmOutput2[outOffset], this->ubOutputInt2, intriParams);
    AscendC::Mutex::Unlock<PIPE_MTE3>(sortVMte3Mutex_);

    outOffset += curLoopSortedNum;
}

__aicore__ inline void MoeMrgsortOut::Init(MoeMrgsortParam *param, TPipe *tPipe)
{
    this->param = param;
    this->allRemainElements = 0;

    sortMte3Mte2Mutex_ = AscendC::AllocMutexID();
    sortMte2VMutex_ = AscendC::AllocMutexID();
    sortVMte3Mutex_ = AscendC::AllocMutexID();

    for (int64_t i = 0; i < listNum; i++) {
        offsets[i] = GetSortOffset<float>(param->perListElements * i);
        if (i == listNum - 1) {
            listRemainElements[i] = param->lastListElements;
        } else {
            listRemainElements[i] = param->perListElements;
        }
        allRemainElements += listRemainElements[i];
    }
}

__aicore__ inline void MoeMrgsortOut::Process()
{
    for (; allRemainElements > 0;) {
        CopyIn();
        UpdateMrgParam();
        MrgsortCompute();
        UpdateSortInfo();
        Extract();
        CopyOut();
    }

    // 尾部 drain：按每条边的 consumer 侧真实排空，随后释放
    // sortMte3Mte2Mutex_ 已改为 CopyIn 内局部 MTE3->MTE2 fence，不跨迭代持锁，无需 drain
    AscendC::Mutex::Lock<PIPE_V>(sortMte2VMutex_);
    AscendC::Mutex::Unlock<PIPE_V>(sortMte2VMutex_);
    AscendC::Mutex::Lock<PIPE_MTE3>(sortVMte3Mutex_);
    AscendC::Mutex::Unlock<PIPE_MTE3>(sortVMte3Mutex_);
    AscendC::ReleaseMutexID(sortMte3Mte2Mutex_);
    AscendC::ReleaseMutexID(sortMte2VMutex_);
    AscendC::ReleaseMutexID(sortVMte3Mutex_);

    ClearCache();
}
#endif
