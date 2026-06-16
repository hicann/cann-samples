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
 * \file broadcast_common.h
 * \brief
 */

#pragma once

#include "acl/acl.h"
#include "kernel_operator.h"
#include "../../include/host_test_utils.h"
#include "../../include/kernel_utils.h"

#include <cassert>
#include <cstdio>
#include <vector>

using BrcGlobal = AscendC::GlobalTensor<float>;
using BrcLocal = AscendC::LocalTensor<float>;
using BrcInQueue = AscendC::TQue<AscendC::QuePosition::VECIN, 1>;
using BrcOutQueue = AscendC::TQue<AscendC::QuePosition::VECOUT, 1>;

__host_aicore__ inline uint32_t ElemAlignedForB32(uint32_t elemNum)
{
    return CeilAlign<uint32_t>(elemNum * sizeof(float), UBBLOCKSIZE) / sizeof(float);
}

__aicore__ inline uint16_t VFLoopNumForB32(uint32_t n)
{
    return static_cast<uint16_t>(CeilDiv<uint32_t>(n, VL_B32));
}

__aicore__ inline void RowiseCopyIn(
    BrcLocal& dstLocal, const BrcGlobal& srcGlobal, uint32_t rowNum, uint32_t rowElems,
    uint32_t rowElemsAligned)
{
    for (uint32_t row = 0; row < rowNum; ++row) {
        ElemwiseCopyIn<float>(dstLocal[row * rowElemsAligned], srcGlobal[row * rowElems], rowElems);
    }
}

__aicore__ inline void RowiseCopyOut(
    const BrcGlobal& dstGlobal, const BrcLocal& srcLocal, uint32_t rowNum, uint32_t rowElems,
    uint32_t rowElemsAligned)
{
    for (uint32_t row = 0; row < rowNum; ++row) {
        ElemwiseCopyOut<float>(dstGlobal[row * rowElems], srcLocal[row * rowElemsAligned], rowElems);
    }
}
