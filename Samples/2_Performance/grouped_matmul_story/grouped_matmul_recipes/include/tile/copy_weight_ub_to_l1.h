/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GROUPED_MATMUL_RECIPE_COPY_WEIGHT_UB_TO_L1_H
#define GROUPED_MATMUL_RECIPE_COPY_WEIGHT_UB_TO_L1_H

#include "include/tensor.h"
#include "include/utils/utils.h"

namespace AscendC {
namespace Te {

// Custom UB to L1 copy class for 8-bit weight
// Both UB and L1 use NZ (fractal) format
struct CopyUB2L1Custom {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        using type = typename U::elementType;
        const auto& dstLayout = dst.Layout();
        const auto& srcLayout = src.Layout();

        // Get shape and stride tuples
        auto srcShape = AscendC::Te::GetShape(srcLayout);
        auto srcStrideTuple = AscendC::Te::GetStride(srcLayout);

        // Extract dimensions from srcShape = ((c0, k1), (n0, n1))
        // Std::get<0>(srcShape) = (c0, k1), Std::get<1>(srcShape) = (n0, n1)
        uint16_t c0 = AscendC::Std::get<0>(AscendC::Std::get<0>(srcShape));
        uint16_t k1 = AscendC::Std::get<1>(AscendC::Std::get<0>(srcShape));
        uint16_t n0 = AscendC::Std::get<0>(AscendC::Std::get<1>(srcShape));
        uint16_t n1 = AscendC::Std::get<1>(AscendC::Std::get<1>(srcShape));

        // Extract innerStride from srcStride = ((1, n1*InnerStride), (c0, InnerStride))
        // For UB2L1, use column stride (InnerStride) not row stride (n1*InnerStride)
        int64_t innerStride = AscendC::Std::get<1>(AscendC::Std::get<1>(srcStrideTuple));

        // Total number of fractal blocks to copy
        uint16_t blockCount = k1 * n1;

        // Block length in 32B units
        uint32_t blockLen = (n0 * c0 * sizeof(type)) / 32;

        // Source stride in 32B units
        int64_t srcStride = (innerStride * sizeof(type)) / 32 - blockLen;

        // Destination stride in 32B units (contiguous in L1)
        int64_t dstStride = 0;

        // Use basic intrinsic directly
        copy_ubuf_to_cbuf((__cbuf__ void*)dst.Data().Get(), (__ubuf__ void*)src.Data().Get(),
                          0, blockCount, blockLen, srcStride, dstStride);
    }
};

} // namespace Te
} // namespace AscendC

// Register CopyTraits with standard trait
template <>
struct AscendC::Te::CopyTraits<AscendC::Te::CopyUB2L1Custom>
    : public AscendC::Te::CopyTraits<
          AscendC::Te::CopyUB2L1Custom, AscendC::Te::DataCopyTraitDefault,
          AscendC::Te::CopyUB2L1Custom, AscendC::Te::DataCopyTraitDefault> {};

#endif
