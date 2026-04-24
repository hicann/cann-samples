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
 * \file matmul_a16w16_constant.h
 * \brief Shared constants and helper types for A16W16 matmul.
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_BIAS_OFFSET = 2UL;
constexpr uint64_t IDX_C_OFFSET = 3UL;

constexpr uint64_t IDX_M_IDX = 0UL;
constexpr uint64_t IDX_N_IDX = 1UL;
constexpr uint64_t IDX_K_IDX = 2UL;

// Set unitflag state: 3 = final accumulation, 2 = non-final accumulation
constexpr uint32_t FINAL_ACCUMULATION = 3;
constexpr uint32_t NON_FINAL_ACCUMULATION = 2;

constexpr uint64_t DOUBLE_BUFFER_COUNT = 2LL;
constexpr uint64_t MB_SIZE = 1024 * 1024UL;
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint64_t NUM_THREE = 3UL;
constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
constexpr uint64_t BASIC_BLOCK_SIZE_64 = 64UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BLOCK_BYTE_SIZE = 32UL;
constexpr uint64_t DATA_SIZE_FP16 = 2UL;
constexpr uint64_t DATA_SIZE_FP32 = 4UL;
constexpr uint64_t CACHELINE = 512UL;
constexpr uint64_t WINDOW_LEN = 4UL;
constexpr uint64_t L1_FOUR_BUFFER = 4UL;
constexpr uint64_t BIAS_TABLE_NUM = 256UL;
constexpr uint64_t RPC_WORKSIZE = 20UL;

constexpr uint16_t AIC_SYNC_AIV_MODE_4 = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 8;
constexpr uint16_t FLAG_ID_MAX = 16;
constexpr uint16_t BLOCK_BASE_M = 256;
constexpr uint16_t BLOCK_BASE_N = 256;

constexpr uint16_t ZERO_FLAG = 0;
constexpr uint16_t FIRST_FLAG = 1;
constexpr uint16_t SECOND_FLAG = 2;
constexpr uint16_t THIRD_FLAG = 3;
constexpr uint16_t SIXTH_FLAG = 6;
constexpr uint16_t SEVENTH_FLAG = 7;
constexpr uint8_t MTE1_MTE2_EVENT_ID_NUM = 4;

