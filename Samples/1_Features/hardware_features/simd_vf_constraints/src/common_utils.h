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

#include <cstdint>
#include <limits>
#include <type_traits>

static constexpr uint32_t AVAIL_UBSIZE = (256 - 16) * 1024u;
static constexpr uint32_t UBBLOCKSIZE = 32u;
static constexpr uint32_t VREGLEN = 256u;
static constexpr uint32_t VL_B32 = VREGLEN / sizeof(float);

template <typename R = uint32_t, typename T1 = R, typename T2 = R>
__host_aicore__ __inline__ R CeilDiv(T1 a, T2 b)
{
    static_assert(std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,
                  "CeilDiv args must be arithmetic types");
    static_assert(std::is_arithmetic<R>::value,
                  "CeilDiv return type must be arithmetic");
    auto ra = static_cast<R>(a);
    auto rb = static_cast<R>(b);
    if (rb == 0) {
        return std::numeric_limits<R>::max();
    }
    // Overflow guard: if a + b - 1 wraps around, return max
    if (ra + rb - 1 < ra) {
        return std::numeric_limits<R>::max();
    }
    return (ra + rb - 1) / rb;
}

template <typename R = uint32_t, typename T1 = R, typename T2 = R>
__host_aicore__ __inline__ R CeilAlign(T1 val, T2 align)
{
    return CeilDiv<R>(val, align) * static_cast<R>(align);
}
