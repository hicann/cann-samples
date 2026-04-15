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
 * \file tensor_utils.h
 * \brief Shared tensor type helpers using AscendC::Hardware position tags.
 */

#pragma once

#include "kernel_basic_intf.h"
#include "include/tensor.h"

namespace kernel_utils {
template <AscendC::Hardware position, typename DType>
struct TensorMemPtrFactory;

template <typename DType>
struct TensorMemPtrFactory<AscendC::Hardware::L1, DType> {
    __aicore__ static inline auto Make()
    {
        return AscendC::Te::MakeL1memPtr(reinterpret_cast<__cbuf__ DType*>(0));
    }
};

template <AscendC::Hardware position, typename DType, typename Layout>
using TensorType =
    decltype(AscendC::Te::MakeTensor(TensorMemPtrFactory<position, DType>::Make(), Layout{}(16UL, 16UL)));
} // namespace kernel_utils
