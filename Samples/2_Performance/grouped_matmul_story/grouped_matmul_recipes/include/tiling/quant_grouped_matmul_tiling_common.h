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
 * \file quant_grouped_matmul_tiling_common.h
 * \brief Shared runtime state for grouped MX tiling generation.
 */
#ifndef QUANT_GROUPED_MATMUL_TILING_COMMON_H
#define QUANT_GROUPED_MATMUL_TILING_COMMON_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#include "tiling/platform/platform_ascendc.h"

// Mirrors `optiling::GmmConstant` / sample layout (arch35 grouped quant matmul).
namespace QuantGroupedMatmulTilingConst {
inline constexpr uint64_t DB_SIZE = 2UL;
inline constexpr uint64_t POWER_OF_TWO = 2UL;
inline constexpr uint64_t MTE2_MIN_LOAD_SIZE_V120 = 64UL * 1024UL;
inline constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
inline constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
inline constexpr uint64_t BASIC_BLOCK_SIZE_512 = 512UL;
inline constexpr uint32_t DATA_SIZE_L0C = 4U;
inline constexpr uint32_t SCALE_FACTOR_MIN = 1U;
inline constexpr uint32_t SCALE_FACTOR_MAX = 127U;
inline constexpr bool TRANS_A = false;
inline constexpr bool TRANS_B = true;
inline constexpr bool WEIGHT_FRACTAL_NZ = false;
} // namespace QuantGroupedMatmulTilingConst

// Original grouped problem shape provided by the caller.
struct QuantGroupedMatmulTilingArgs {
    uint64_t groupNum{0UL};
    uint64_t m{0UL};
    uint64_t n{0UL};
    uint64_t k{0UL};
};

// Static hardware information queried from the Ascend platform runtime.
struct QuantGroupedMatmulPlatformInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{512 * 1024UL};
    uint64_t l2Size{0UL};
    uint64_t l0cSize{256 * 1024UL};
    uint64_t l0aSize{0UL};
    uint64_t l0bSize{0UL};
    uint64_t btSize{0UL};
    platform_ascendc::SocVersion socVersion{};
};

// Intermediate tiling state built step by step by the grouped tiling engine.
struct QuantGroupedMatmulRunInfo {
    uint64_t baseM{0UL};
    uint64_t baseN{0UL};
    uint64_t baseK{0UL};
    uint64_t kAL1{0UL};
    uint64_t kBL1{0UL};
    uint64_t scaleKAL1{0UL};
    uint64_t scaleKBL1{0UL};
    uint8_t dbL0C{1U};
    uint64_t depthA1{1UL};
    uint64_t depthB1{1UL};
    uint64_t stepKa{1UL};
    uint64_t stepKb{1UL};
    uint32_t scaleFactorA{1U};
    uint32_t scaleFactorB{1U};
};
#endif // QUANT_GROUPED_MATMUL_TILING_COMMON_H
