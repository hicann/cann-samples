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

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace FALite {

constexpr uint32_t L1_CAP = 512 * 1024;
constexpr uint32_t L0A_CAP = 64 * 1024;
constexpr uint32_t L0B_CAP = 64 * 1024;
constexpr uint32_t L0C_CAP = 256 * 1024;
constexpr uint32_t UB_CAP = 248 * 1024;

template <typename TilingData>
const char* InitBaseTiling(uint32_t B, uint32_t S, float scale, uint32_t aicNum, TilingData& data)
{
    if (B == 0 || S == 0)
        return "B/S must >0";
    if (S % BR != 0)
        return "S must be BR(128) multiple";
    if (scale == 0.0f || !std::isfinite(scale))
        return "scale must be non-zero finite";
    if (aicNum == 0)
        return "aicNum must >0";
    uint64_t nt = (uint64_t)B * (S / BR);
    if (nt > UINT32_MAX)
        return "B*tr exceeds UINT32_MAX";
    data.batchSize = B;
    data.seqLen = S;
    data.headDim = HEAD_DIM;
    data.scale = scale;
    data.br = BR;
    data.bc = BC;
    data.tr = S / data.br;
    data.tc = S / data.bc;
    data.numTasks = (uint32_t)nt;
    data.useAicNum = data.numTasks < aicNum ? data.numTasks : aicNum;
    return nullptr;
}

template <typename TilingData>
const char* CheckSRAMCapacity(const TilingData& data)
{
    if (data.layoutAIC.pL1Addr + data.layoutAIC.pL1Elems * sizeof(uint16_t) > L1_CAP)
        return "L1 overflow";
    if (data.layoutAIC.aL0AElems * sizeof(uint16_t) > L0A_CAP)
        return "L0A overflow";
    if (data.layoutAIC.bL0BElems * sizeof(uint16_t) > L0B_CAP)
        return "L0B overflow";
    if (data.layoutAIC.cL0CElems * sizeof(float) > L0C_CAP)
        return "L0C overflow";
    if (data.layoutAIV.alphaUBAddr + data.layoutAIV.rowStatsUBElems * sizeof(float) > UB_CAP)
        return "UB overflow";
    return nullptr;
}

static uint32_t FloatToU32(float f)
{
    uint32_t u = 0;
    const unsigned char* src = reinterpret_cast<const unsigned char*>(&f);
    unsigned char* dst = reinterpret_cast<unsigned char*>(&u);
    for (int i = 0; i < static_cast<int>(sizeof(uint32_t)); ++i) {
        dst[i] = src[i];
    }
    return u;
}

inline void F32ToBf16(const float* fp32, uint16_t* bf16, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        uint32_t u = FloatToU32(fp32[i]);
        bf16[i] = static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
    }
}

// 各版本共用 InitTiling 骨架: InitBaseTiling → 版本 layout → 容量校验
template <typename TilingData, typename ComputeLayout>
const char* InitAndCheckTiling(
    uint32_t B, uint32_t S, float scale, uint32_t aicNum, TilingData& data, ComputeLayout computeLayout)
{
    const char* err = InitBaseTiling(B, S, scale, aicNum, data);
    if (err)
        return err;
    computeLayout(data);
    err = CheckSRAMCapacity(data);
    if (err)
        return err;
    if (data.layoutAIV.pUBElems < (data.br / 2) * data.headDim)
        return "P reuse workspace too small";
    return nullptr;
}

} // namespace FALite
