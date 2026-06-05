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
 * \file copy_weight_ub_to_l1.h
 * \brief Tensor API copy trait for converted 8-bit weight UB to L1 movement.
 */
#pragma once

#include "include/tensor_api/tensor.h"

namespace Tile {

struct CopyUB2L1Weight {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        using type = typename U::elementType;
        const auto& srcLayout = src.Layout();
        auto srcShape = AscendC::Te::GetShape(srcLayout);
        auto srcStride = AscendC::Te::GetStride(srcLayout);

        uint16_t c0 = AscendC::Std::get<0>(AscendC::Std::get<0>(srcShape));
        uint16_t k1 = AscendC::Std::get<1>(AscendC::Std::get<0>(srcShape));
        uint16_t n0 = AscendC::Std::get<0>(AscendC::Std::get<1>(srcShape));
        uint16_t n1 = AscendC::Std::get<1>(AscendC::Std::get<1>(srcShape));
        int64_t innerStride = AscendC::Std::get<1>(AscendC::Std::get<1>(srcStride));

        uint16_t blockCount = k1 * n1;
        uint32_t blockLen = (n0 * c0 * sizeof(type)) / 32;
        int64_t srcStrideBlocks = (innerStride * sizeof(type)) / 32 - blockLen;

        asc_copy_ub2l1(
            (__cbuf__ void*)dst.Data().Get(), (__ubuf__ void*)src.Data().Get(), blockCount, blockLen, srcStrideBlocks,
            0);
    }
};

} // namespace Tile

template <>
struct AscendC::Te::CopyTraits<::Tile::CopyUB2L1Weight>
    : public AscendC::Te::CopyTraits<
          ::Tile::CopyUB2L1Weight, AscendC::Te::CopyUB2L1TraitDefault, ::Tile::CopyUB2L1Weight,
          AscendC::Te::CopyUB2L1TraitDefault> {};
