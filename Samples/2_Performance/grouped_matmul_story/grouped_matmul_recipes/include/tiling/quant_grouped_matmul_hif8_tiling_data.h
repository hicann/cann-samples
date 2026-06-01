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
 * \file quant_grouped_matmul_hif8_tiling_data.h
 * \brief Serialized tiling data for grouped HiFloat8 matmul samples.
 */
#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

struct QuantGroupedMatmulHif8TilingData {
    uint32_t groupNum{0U};
    uint32_t m{0U};
    uint32_t n{0U};
    uint32_t k{0U};
    uint32_t usedCoreNum{0U};
    uint32_t baseM{0U};
    uint32_t baseN{0U};
    uint32_t baseK{0U};
    uint32_t stepKa{0U};
    uint32_t stepKb{0U};
    uint32_t kAL1{0U};
    uint32_t kBL1{0U};
    uint32_t nBufferNum{0U};
    uint32_t dbL0C{0U};
    // Values match GroupedMatmulRecipe::QuantMode on device:
    // 0=DEFAULT, 1=PERTENSOR, 2=PERCHANNEL.
    uint32_t x1QuantMode{0U};
    uint32_t x2QuantMode{0U};
};