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
 * \brief Dispatch policy tags used by the quantized matmul recipe kernels.
 */
#ifndef DISPATCH_POLICY_H
#define DISPATCH_POLICY_H

#include "block/block_scheduler_policy.h"
#include "kernel_utils/integral_constant.h"

// Tag for kernels that split work along K while still carrying dedicated
// scaleA and scaleB tensors through the pipeline.
struct KernelMultiBlockOnKAxisWithScale {};

/**
 * @brief Dispatch tag for MXFP4 quantized matmul kernels that use the SWAT
 *        scheduling family.
 * @tparam FULL_LOAD_MODE_ Selects the SWAT variant: streaming or A-full-load.
 */
template <uint64_t FULL_LOAD_MODE_ = SWAT_NO_FULL_LOAD_MODE>
struct QuantMatmulMxMultiBlockWithSwat {
    using ScheduleType = KernelMultiBlockOnKAxisWithScale;
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
};

#endif
