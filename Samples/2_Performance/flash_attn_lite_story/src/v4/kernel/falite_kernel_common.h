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

// 正式执行使用 mode2 同步同组的 1 个 AIC 和 2 个 AIV.
// SIM_COMPATIBLE 改用 mode4 分别同步两路 AIV, 规避 CANNsim 的多轮 mode2 异常.
constexpr uint8_t GROUP_CROSS_MODE = 2;
#ifdef SIM_COMPATIBLE
constexpr uint8_t PAIR_CROSS_MODE = 4;
constexpr uint16_t AIV1_FLAG_OFFSET = 16;
#endif
constexpr uint16_t FLAG_S_READY_BASE = 0;
constexpr uint16_t FLAG_P_READY_BASE = FLAG_S_READY_BASE + PIPELINE_SLOT_NUM;
constexpr uint16_t FLAG_O_READY_BASE = FLAG_P_READY_BASE + PIPELINE_SLOT_NUM;
// P 写入 L1 后, 分别通知 AIC 的 MTE1 和 MTE2.
constexpr uint16_t FLAG_P_READY_MTE2_BASE = FLAG_O_READY_BASE + PIPELINE_SLOT_NUM;

constexpr AscendC::FixpipeConfig PFA_CFG_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

// Ascend 950 的 RegBase 向量宽度为 256B; 固定该值以兼容 CANN 9.0.0 的不同小版本.
constexpr uint32_t VECTOR_REG_WIDTH = 256;
constexpr uint32_t VL_B32 = VECTOR_REG_WIDTH / sizeof(float);
constexpr uint32_t VL_B16 = VECTOR_REG_WIDTH / sizeof(bfloat16_t);
constexpr uint32_t C0_BYTES = 32;
constexpr uint32_t B16_PER_DATABLOCK = C0_BYTES / sizeof(bfloat16_t);
// float 的最低有限值, 即 -FLT_MAX; FLT_MIN 是最小正正规数.
constexpr float FLOAT_LOWEST = -3.402823466e+38F;

template <AscendC::HardEvent E>
__aicore__ inline void SetWaitFlag(const AscendC::TEventID eventId)
{
    using namespace AscendC;
    SetFlag<E>(eventId);
    WaitFlag<E>(eventId);
}

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

} // namespace FALite
