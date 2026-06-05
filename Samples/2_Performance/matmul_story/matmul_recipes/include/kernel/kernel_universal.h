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
 * \file kernel_universal.h
 * \brief Common GemmUniversal template declaration.
 */

#pragma once

#include "blaze/utils/common_utils.h"

namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, typename Enable_ = void>
class GemmUniversal {
    static_assert(
        Blaze::Gemm::always_false_v<BlockEpilogue_> && Blaze::Gemm::always_false_v<BlockMmad_>,
        "KernelStreamk is not implemented for this BlockEpilogue or BlockMmad");
};

} // namespace Kernel

// Include concrete GemmUniversal specializations here.
#include "weight_quant_matmul_mxfp8fp4_kernel_swat.h"
