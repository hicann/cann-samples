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
 * \file quant_grouped_matmul_mx_tiling_data.h
 * \brief POD tiling payload shared by grouped MX recipe host and kernel.
 */
#pragma once

#include <cstdint>

struct QuantGroupedMatmulMxTilingData {
    uint32_t groupNum = 0;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t kAL1 = 0;
    uint32_t kBL1 = 0;
    uint32_t scaleKAL1 = 0;
    uint32_t scaleKBL1 = 0;
    uint32_t usedCoreNum = 0;
    uint8_t dbL0C = 1;
};

using QuantGroupedMatmulMxfp4TilingData = QuantGroupedMatmulMxTilingData;
using QuantGroupedMatmulMxfp8TilingData = QuantGroupedMatmulMxTilingData;

