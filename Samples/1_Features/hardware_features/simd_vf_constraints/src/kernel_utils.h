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

#include "acl/acl.h"
#include "kernel_operator.h"
#include "common_utils.h"

template <typename T>
__aicore__ inline void ElemwiseCopyIn(
    const AscendC::LocalTensor<T>& dstLocal, const AscendC::GlobalTensor<T>& srcGlobal, uint32_t copyElemNum)
{
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(1);
    copyParams.blockLen = copyElemNum * static_cast<uint32_t>(sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = static_cast<T>(0);
    AscendC::DataCopyPad<T>(dstLocal, srcGlobal, copyParams, padParams);
}

template <typename T>
__aicore__ inline void ElemwiseCopyOut(
    const AscendC::GlobalTensor<T>& dstGlobal, const AscendC::LocalTensor<T>& srcLocal, uint32_t copyElemNum)
{
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(1);
    copyParams.blockLen = copyElemNum * static_cast<uint32_t>(sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad<T>(dstGlobal, srcLocal, copyParams);
}
