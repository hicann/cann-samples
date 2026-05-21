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
 * \file quant_matmul_hifp8_tiling_data.h
 * \brief Serialized tiling data passed from host launcher to kernel.
 */

#ifndef QUANT_MATMUL_HIFP8_TILING_DATA_H
#define QUANT_MATMUL_HIFP8_TILING_DATA_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#pragma pack(push, 8)
struct alignas(8) QuantMatmulHifp8TilingData {
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    uint32_t usedCoreNum{0};
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t baseK{0};
    uint32_t stepKa{0};
    uint32_t stepKb{0};
    uint32_t kAL1{0};
    uint32_t kBL1{0};
    uint32_t nBufferNum{0};
    uint32_t dbL0c{0};
    uint32_t mTailTile{1};
    uint32_t nTailTile{1};
    uint32_t mBaseTailSplitCnt{1};
    uint32_t nBaseTailSplitCnt{1};
    uint32_t mTailMain{0};
    uint32_t nTailMain{0};
    // x1/x2: QuantBatchMatmul::QuantMode。0=DEFAULT 表示该侧无独立 scale GM 张量（scale 为空）。
    // 1=PERCHANNEL，2=PERTENSOR。
    uint32_t x1QuantMode{0};
    uint32_t x2QuantMode{0};
};
#pragma pack(pop)

#endif // QUANT_MATMUL_HIFP8_TILING_DATA_H
