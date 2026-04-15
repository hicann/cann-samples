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
 * \file copy_gm_to_ub.h
 * \brief Custom copy primitive for packed B4 weight movement from GM to UB.
 */
#pragma once

#include "include/tensor.h"
#include "include/utils/utils.h"

namespace AscendC {
namespace Te {

// Custom GM-to-UB copy for packed 4-bit weight tensors.
struct CopyGM2UBWeight {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        const auto& dstLayout = dst.Layout();
        const auto& srcLayout = src.Layout();

        uint8_t cacheMode = GetCacheModeFromTensor(src);

        // Get shape and stride
        auto srcShape = AscendC::Te::GetShape(srcLayout);
        auto srcStrideTuple = AscendC::Te::GetStride(srcLayout);
        auto dstStrideTuple = AscendC::Te::GetStride(dstLayout);

        // Extract k1 from shape: ((c0, k1), (n0, n1)).
        uint16_t blockCount = AscendC::Std::get<1>(AscendC::Std::get<0>(srcShape));

        // For packed B4 (2 elements per byte), convert element strides to byte strides via >> 1.
        uint32_t blockLen = AscendC::Std::get<1>(AscendC::Std::get<0>(dstStrideTuple)) >> 1;
        int64_t srcStride = AscendC::Std::get<1>(AscendC::Std::get<0>(srcStrideTuple)) >> 1;
        int64_t dstStride = AscendC::Std::get<1>(AscendC::Std::get<0>(dstStrideTuple)) >> 1;

        // Execute the hardware intrinsic with explicit cache/stride controls.
        copy_gm_to_ubuf_align_v2(
            (__ubuf__ uint8_t*)dst.Data().Get(), (__gm__ uint8_t*)src.Data().Get(), 0, blockCount, blockLen, 0, 0,
            false, cacheMode, srcStride, dstStride);
    }
};

} // namespace Te
} // namespace AscendC

// Register CopyTraits for the custom GM-to-UB packed-weight copy.
template <>
struct AscendC::Te::CopyTraits<AscendC::Te::CopyGM2UBWeight>
    : public AscendC::Te::CopyTraits<
          AscendC::Te::CopyGM2UBWeight, AscendC::Te::DataCopyTraitDefault, AscendC::Te::CopyGM2UBWeight,
          AscendC::Te::DataCopyTraitDefault> {};
