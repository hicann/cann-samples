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
 * \file weight_quant_matmul_mxfp8fp4_tiling_data.h
 * \brief Serialized tiling data for the MXFP8 input + MXFP4 packed-weight matmul recipe.
 */
#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#pragma pack(push, 8)
struct alignas(8) WeightQuantMatmulMxfp8Fp4TilingData {
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};

    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t baseK{0};
    // L1 K extents consumed by block MMAD. Scale K is serialized as an extent, not a derived scale factor.
    uint32_t tileShapeKL1{0};
    uint32_t tileShapeScaleKL1{0};

    uint32_t usedCoreNum{1};
    uint32_t cubeNumBlocksM{1};
    uint32_t cubeNumBlocksN{1};
    uint32_t iterateOrder{1};
    uint32_t mTailTile{1};
    uint32_t nTailTile{1};
    uint32_t mBaseTailSplitCnt{1};
    uint32_t nBaseTailSplitCnt{1};
    uint32_t mTailMain{0};
    uint32_t nTailMain{0};

    // Weight prologue keeps N intact and splits only K inside UB.
    uint32_t nBubSize{0};
    uint32_t kBubSize{0};
};
#pragma pack(pop)
