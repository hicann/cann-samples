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
 * \file grouped_matmul_constant.h
 * \brief Shared constants for grouped MX recipe wrappers.
 */
#ifndef GROUPED_MATMUL_CONSTANT_H
#define GROUPED_MATMUL_CONSTANT_H

#include <cstdint>

namespace GroupedMatmulRecipe {
// Generic (m, n, k) tuple indices shared by host and device helpers.
constexpr int32_t MNK_M = 0;
constexpr int32_t MNK_N = 1;
constexpr int32_t MNK_K = 2;

constexpr int8_t GROUP_TYPE_SPLIT_M = 0;
constexpr int8_t GROUP_TYPE_NO_SPLIT = -1;

constexpr uint64_t MX_GROUP_SIZE = 32UL;
constexpr uint64_t MX_DIVISOR_SIZE = 64UL;
constexpr uint64_t MX_MULTI_SIZE = 2UL;
constexpr uint64_t CUBE_BLOCK = 16UL;
constexpr uint64_t DOUBLE_BUFFER = 2UL;
constexpr uint8_t MTE1_MTE2_EVENT_ID_NUM = 6;
constexpr static uint32_t FINAL_ACCUMULATION = 3U;
constexpr static uint32_t NON_FINAL_ACCUMULATION = 2U;

inline uint64_t GetShapeWithDataTypeFp4(uint64_t shape)
{
    return shape << 1;
}

inline uint64_t GetSizeWithDataTypeFp4(uint64_t shape)
{
    return (shape + 1UL) >> 1;
}
} // namespace GroupedMatmulRecipe

#endif
