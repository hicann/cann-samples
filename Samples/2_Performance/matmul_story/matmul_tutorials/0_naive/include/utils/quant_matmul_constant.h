/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UTILS_QUANT_MATMUL_CONSTANT_H
#define UTILS_QUANT_MATMUL_CONSTANT_H

constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_X1SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_X2SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_C_OFFSET = 4UL;
constexpr uint64_t IDX_M_TILEIDX = 0UL;
constexpr uint64_t IDX_N_TILEIDX = 1UL;
constexpr uint64_t IDX_M_TAIL_SPLIT_TILEIDX = 2UL;
constexpr uint64_t IDX_N_TAIL_SPLIT_TILEIDX = 3UL;
 
constexpr uint64_t IDX_M_IDX = 0UL;
constexpr uint64_t IDX_N_IDX = 1UL;
constexpr uint64_t IDX_K_IDX = 2UL;

constexpr uint32_t FINAL_ACCUMULATION = 3;
constexpr uint32_t NON_FINAL_ACCUMULATION = 2;
constexpr uint64_t B8_MIN_STEP = 2UL;

constexpr uint16_t ZERO_FLAG = 0;
constexpr uint16_t FIRST_FLAG = 1;
constexpr uint16_t SCALE_BUFFER_FLAG_0 = 4;
constexpr uint16_t SCALE_BUFFER_FLAG_1 = 5;
constexpr uint8_t MTE1_MTE2_EVENT_ID_NUM = 6;

constexpr int32_t MXFP_GROUP_SIZE = 32UL;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
static constexpr uint64_t NONE_FULL_LOAD_MODE = 0UL;
static constexpr uint64_t A_FULL_LOAD_MODE = 1UL;
constexpr int64_t DOUBLE_BUFFER_COUNT = 2LL;
constexpr int64_t OFFSET_4_8 = 2LL;

#endif
