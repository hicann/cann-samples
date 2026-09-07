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

#include "../flash_attn_lite_common.h"

#include <basic_api/reg_compute/kernel_reg_compute_utils.h>
#include <kernel_operator.h>

namespace FALite {

// 使用 uint8_t 作为 Mutex ID 类型，以兼容未声明 AscendC::MutexID 的 CANN 9.0。
using MutexID = uint8_t;

// 真机使用 CrossCore mode2 同步一个 Mix 组（1 AIC + 2 AIV）。
// CANNsim 使用 mode4 分别同步 AIV0 和 AIV1，以兼容多轮仿真。
constexpr uint8_t GROUP_CROSS_MODE = 2;
#ifdef SIM_COMPATIBLE
constexpr uint8_t PAIR_CROSS_MODE = 4;
constexpr uint16_t AIV1_FLAG_OFFSET = 16;
#endif
constexpr uint16_t FLAG_S_READY_BASE = 0;
constexpr uint16_t FLAG_P_READY_BASE = FLAG_S_READY_BASE + PIPELINE_SLOT_NUM;
constexpr uint16_t FLAG_O_READY_BASE = FLAG_P_READY_BASE + PIPELINE_SLOT_NUM;

constexpr AscendC::FixpipeConfig PFA_CFG_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

// Ascend 950 的 RegBase 向量宽度为 256B; 固定该值以兼容 CANN 9.0.0 的不同小版本.
constexpr uint32_t VECTOR_REG_WIDTH = 256;
constexpr uint32_t VL_B32 = VECTOR_REG_WIDTH / sizeof(float);
constexpr uint32_t VL_B16 = VECTOR_REG_WIDTH / sizeof(bfloat16_t);
constexpr uint32_t C0_BYTES = 32;
constexpr uint32_t B16_PER_DATABLOCK = C0_BYTES / sizeof(bfloat16_t);
// -FLT_MAX 是 float 的最低有限值；FLT_MIN 表示最小正正规数。
constexpr float FLOAT_LOWEST = -3.402823466e+38F;

__aicore__ inline uint16_t SlotFlagId(uint16_t baseFlagId, uint32_t slot)
{
    return baseFlagId + static_cast<uint16_t>(slot);
}

template <pipe_t PIPE>
__aicore__ inline void SetAicToAiv(uint16_t flagId)
{
    using namespace AscendC;
#ifdef SIM_COMPATIBLE
    CrossCoreSetFlag<PAIR_CROSS_MODE, PIPE>(flagId);
    CrossCoreSetFlag<PAIR_CROSS_MODE, PIPE>(flagId + AIV1_FLAG_OFFSET);
#else
    CrossCoreSetFlag<GROUP_CROSS_MODE, PIPE>(flagId);
#endif
}

template <pipe_t PIPE>
__aicore__ inline void WaitAicToAiv(uint16_t flagId)
{
    using namespace AscendC;
#ifdef SIM_COMPATIBLE
    CrossCoreWaitFlag<PAIR_CROSS_MODE, PIPE>(flagId);
#else
    CrossCoreWaitFlag<GROUP_CROSS_MODE, PIPE>(flagId);
#endif
}

template <pipe_t PIPE>
__aicore__ inline void SetAivToAic(uint16_t flagId)
{
    using namespace AscendC;
#ifdef SIM_COMPATIBLE
    CrossCoreSetFlag<PAIR_CROSS_MODE, PIPE>(flagId);
#else
    CrossCoreSetFlag<GROUP_CROSS_MODE, PIPE>(flagId);
#endif
}

template <pipe_t PIPE>
__aicore__ inline void WaitAivToAic(uint16_t flagId)
{
    using namespace AscendC;
#ifdef SIM_COMPATIBLE
    CrossCoreWaitFlag<PAIR_CROSS_MODE, PIPE>(flagId);
    CrossCoreWaitFlag<PAIR_CROSS_MODE, PIPE>(flagId + AIV1_FLAG_OFFSET);
#else
    CrossCoreWaitFlag<GROUP_CROSS_MODE, PIPE>(flagId);
#endif
}

template <typename R, typename T1, typename T2>
__aicore__ inline R CeilDiv(T1 x, T2 y)
{
    if (y == 0) {
        return static_cast<R>(0);
    }
    return static_cast<R>((x + y - 1) / y);
}

template <typename R, typename T1, typename T2>
__aicore__ inline R CeilAlign(T1 x, T2 base)
{
    return static_cast<R>(CeilDiv<R, T1, T2>(x, base) * base);
}

// 非因果模式遍历全部 K/V 块；因果模式只遍历该 Q 块及之前的 K/V 块。
template <bool CAUSAL_MASK>
__aicore__ inline uint32_t GetKvTileCount(const FlashAttnLiteTilingData& data, uint32_t qTileIdx)
{
    if constexpr (CAUSAL_MASK) {
        return qTileIdx + 1;
    }
    return data.tc;
}

// 返回序列 Tile 中真实存在的行数；除末块外均为 tileRows。
__aicore__ inline uint32_t GetTileValidRows(uint32_t seqLen, uint32_t tileIdx, uint32_t tileRows)
{
    const uint32_t tileBegin = tileIdx * tileRows;
    const uint32_t remaining = seqLen - tileBegin;
    return remaining < tileRows ? remaining : tileRows;
}

} // namespace FALite
