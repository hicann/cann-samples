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
 * \file moe_sort_base.h
 * \brief
 */

#ifndef MOE_MRGSORTBASE_H
#define MOE_MRGSORTBASE_H

using namespace AscendC;

class MoeSortBase {
protected:
    constexpr static int64_t DST_BLK_STRIDE = 1;
    constexpr static int64_t DST_REP_STRIDE = 8;
    constexpr static int64_t MAX_MRGSORT_LIST = 4;
    constexpr static uint16_t FLOAT_REG_TENSOR_LENGTH = 256 / sizeof(float);

    GlobalTensor<int32_t> expertIdxGm;
    GlobalTensor<int32_t> expandedRowIdxGm;
    GlobalTensor<int32_t> sortedExpertForSourceRowGm;
    GlobalTensor<int32_t> expandDstToSrcRowGm;
    GlobalTensor<int32_t> sortedExpertIdxGm;
    GlobalTensor<int32_t> expertCountTempGm;

    TQue<QuePosition::VECIN, 1> sortDataCopyInQueue;
    TQue<QuePosition::VECOUT, 1> sortDataCopyOutQueue;
    TBuf<TPosition::VECCALC> tempBuffer;
    TBuf<TPosition::VECCALC> sortedBuffer;

    TPipe *pipe;
    int64_t blockIdx = 0;
    int64_t totalLength = 0;
    int64_t sortNum = 0;
    int64_t tileLength = 0;
    int64_t expertStart = 0;
    int64_t expertEnd = 0;
    int64_t actualExpertNum = 0;
    int64_t n = 0;
    int64_t k = 0;
    int64_t rowIdxType = 0;
    int64_t vmsNeedCoreNum = 0;
    int64_t sortOutOneLoopMaxElements = 0;

public:
    __aicore__ inline MoeSortBase(){};
};
#endif