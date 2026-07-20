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

#include <basic_api/reg_compute/kernel_reg_compute_utils.h> // Reg Cast 配置
#include <kernel_operator.h>

namespace FALite {

// 默认使用 mode2 聚合 AIC 和两路 AIV. SIM_COMPATIBLE 使用
// mode4 分别配对 AIV, 规避 CANNsim 反向同 ID 多代复用的计数异常.
constexpr uint8_t GROUP_CROSS_MODE = 2;
#ifdef SIM_COMPATIBLE
constexpr uint8_t PAIR_CROSS_MODE = 4;
constexpr uint16_t AIV1_FLAG_OFFSET = 16;
#endif
constexpr uint16_t FLAG_S_READY = 0;
constexpr uint16_t FLAG_O_READY = 1;
constexpr uint16_t FLAG_DONE = 2;
constexpr uint16_t FLAG_P_READY = 4;

constexpr AscendC::FixpipeConfig PFA_CFG_UB = {AscendC::CO2Layout::ROW_MAJOR,
                                               true};

// Ascend950 RegBase 向量长度，适配 CANN-9.0.0 多版本保留兼容性固定写 256B
constexpr uint32_t VECTOR_REG_WIDTH = 256;
constexpr uint32_t VL_B32 = VECTOR_REG_WIDTH / sizeof(float);
constexpr uint32_t VL_B16 = VECTOR_REG_WIDTH / sizeof(bfloat16_t);
constexpr uint32_t C0_BYTES = 32;
constexpr uint32_t B16_PER_DATABLOCK = C0_BYTES / sizeof(bfloat16_t);
// float 的最低有限值, 即 -FLT_MAX, 避免将 FLT_MIN 误解为最低负数.
constexpr float FLOAT_LOWEST = -3.402823466e+38F;

// 静态 Tensor 编程需显式管理 event ID, Ascend950 保留 ID 6/7.
constexpr AscendC::TEventID STATIC_EVENT_ID0 = 0;
constexpr AscendC::TEventID STATIC_EVENT_ID1 = 1;
constexpr AscendC::TEventID STATIC_EVENT_ID2 = 2;

template <AscendC::HardEvent E>
__aicore__ inline void SetWaitFlag(const AscendC::TEventID eventId) {
    using namespace AscendC;
    SetFlag<E>(eventId);
    WaitFlag<E>(eventId);
}

template <typename R, typename T1, typename T2>
__aicore__ inline R CeilDiv(T1 x, T2 y) {
    if (y == 0) {
        return static_cast<R>(0);
    }
    return static_cast<R>((x + y - 1) / y);
}

template <typename R, typename T1, typename T2>
__aicore__ inline R CeilAlign(T1 x, T2 base) {
    return static_cast<R>(CeilDiv<R, T1, T2>(x, base) * base);
}

} // namespace FALite
