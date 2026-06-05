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
 * \file copy_weight_gm_to_ub.h
 * \brief Tensor API copy trait for packed 4-bit NZ weight GM to UB movement.
 */
#pragma once

#include "include/tensor_api/tensor.h"

namespace Tile {

struct CopyGM2UBPackedWeight {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        const auto& dstLayout = dst.Layout();
        const auto& srcLayout = src.Layout();
        uint8_t cacheMode = src.Engine().GetCacheMode();

        auto srcShape = AscendC::Te::GetShape(srcLayout);
        auto srcStride = AscendC::Te::GetStride(srcLayout);
        auto dstStride = AscendC::Te::GetStride(dstLayout);

        uint16_t blockCount = AscendC::Std::get<1>(AscendC::Std::get<0>(srcShape));
        uint32_t blockLen = AscendC::Std::get<1>(AscendC::Std::get<0>(dstStride)) >> 1;
        int64_t srcStrideBytes = AscendC::Std::get<1>(AscendC::Std::get<0>(srcStride)) >> 1;
        int64_t dstStrideBytes = AscendC::Std::get<1>(AscendC::Std::get<0>(dstStride)) >> 1;

        asc_copy_gm2ub_align(
            (__ubuf__ uint8_t*)dst.Data().Get(), (__gm__ uint8_t*)src.Data().Get(), blockCount, blockLen, 0, 0, false,
            cacheMode, srcStrideBytes, dstStrideBytes);
    }
};

} // namespace Tile

template <>
struct AscendC::Te::CopyTraits<::Tile::CopyGM2UBPackedWeight>
    : public AscendC::Te::CopyTraits<
          ::Tile::CopyGM2UBPackedWeight, AscendC::Te::CopyGM2UBTraitDefault, ::Tile::CopyGM2UBPackedWeight,
          AscendC::Te::CopyGM2UBTraitDefault> {};
