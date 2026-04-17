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
 * \file block_scheduler_utils.h
 * \brief Scheduler traits shared by grouped MX recipe wrappers.
 */
#pragma once

#include "kernel_utils/tuple_utils.h"

#include "../policy/dispatch_policy.h"

namespace Block {

template <size_t N, typename Tp>
__aicore__ constexpr inline decltype(auto) GetIntegralConstant()
{
    static_assert(AscendC::Std::is_tuple_v<Tp>, "Input must be a tuple type");
    return AscendC::Std::tuple_element<N, Tp>::type::value;
}

__aicore__ inline bool IsMTail(int64_t mTileIdx, int64_t mTileNum)
{
    return (mTileIdx - (mTileNum - 1)) % mTileNum == 0;
}

__aicore__ inline bool IsNTail(int64_t nTileIdx, int64_t nTileNum)
{
    return nTileIdx == (nTileNum - 1);
}


} // namespace Block

