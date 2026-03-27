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
 * \file dispatch_policy.h
 * \brief
 */
#ifndef DISPATCH_POLICY_H
#define DISPATCH_POLICY_H

#include "kernel_utils/common_utils.h"
#include "kernel_utils/integral_constant.h"

/* block schedule policies */
struct KernelMultiBlockOnKAxisWithScale {};

/**
 * @struct QuantMatmulMxMultiBlockWithSwat
 * @brief Matrix multiplication with scaleA and scaleB
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0>
 */
template <uint64_t FULL_LOAD_MODE_ = 0>
struct QuantMatmulMxMultiBlockWithSwat {
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
};

#endif
