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

namespace FALite {

// 样例固定 D=128；Br 和 Bc 由 Host 侧 TilingData 指定。
constexpr uint32_t HEAD_DIM = 128;

// 字段名中的 Addr 以字节为单位，Elems 表示 LocalTensor 的元素数。
struct SRAMLayoutAIC {
    uint32_t pL1Addr;
    uint32_t pL1Elems;
    uint32_t qL1Addr;
    uint32_t qL1Elems;
    uint32_t kL1Addr;
    uint32_t kL1Elems;
    uint32_t vL1Addr;
    uint32_t vL1Elems;
    uint32_t aL0AAddr;
    uint32_t aL0AElems;
    uint32_t bL0BAddr;
    uint32_t bL0BElems;
    uint32_t cL0CAddr;
    uint32_t cL0CElems;
};

struct SRAMLayoutAIV {
    uint32_t sUBAddr;
    uint32_t sUBElems;
    uint32_t oDeltaUBAddr;
    uint32_t oDeltaUBElems;
    uint32_t oAccUBAddr;
    uint32_t oAccUBElems;
    uint32_t pUBAddr;
    uint32_t pUBElems;
    uint32_t mUBAddr;
    uint32_t lUBAddr;
    uint32_t alphaUBAddr;
    uint32_t rowStatsUBElems;
};

// TilingData 仅保存 Host 与 Kernel 共用的参数。
struct FlashAttnLiteTilingData {
    uint32_t batchHeadNum;
    uint32_t seqLen;
    float scale;
    uint32_t br;
    uint32_t bc;
    uint32_t tr;
    uint32_t tc;
    uint32_t useAicNum;
    uint32_t numTasks;

    SRAMLayoutAIC layoutAIC;
    SRAMLayoutAIV layoutAIV;
};

void LaunchFlashAttnLiteKernel(
    uint8_t* dQ, uint8_t* dK, uint8_t* dV, uint8_t* dP, uint8_t* dOut, const FlashAttnLiteTilingData& data,
    void* stream);

} // namespace FALite
