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

#include <acl/acl.h>

#include <cstdint>

// 物理布局: Q/K/V/O 为 BF16 [B,S,128], log_decay 为 FP32 [B,S,128],
// beta 为 BF16 [B,S], final_state 为 FP32 [B,128,128]. 当前固定 N=1、Dk=Dv=128,
// Head 轴不落盘.
// 查询当前版本所需的设备 workspace 字节数.
bool GetKimiDeltaAttnLiteWorkspaceSize(uint32_t batchSize, uint32_t seqLen, uint64_t& workspaceBytes);

// 返回 true 表示参数校验通过, 且当前版本所需的 kernel 已提交到 stream.
// requestedMixCoreNum 为 0 时使用本卡全部 Mix 组. 一个 Mix 组包含 1 个 AIC
// 和 2 个 AIV; 纯 AIV kernel 可使用所选 Mix 组数两倍的 AIV 核. 调用方必须保证
// 所有设备指针、workspace 和 stream 在 stream 执行完成前一直有效.
bool KimiDeltaAttnLiteNPU(
    uint8_t* dQ, uint8_t* dK, uint8_t* dV, uint8_t* dLogDecay, uint8_t* dBeta, uint8_t* dO, uint8_t* dFinalState,
    uint8_t* dWorkspace, uint64_t workspaceBytes, uint32_t batchSize, uint32_t seqLen, uint32_t requestedMixCoreNum,
    aclrtStream stream);
