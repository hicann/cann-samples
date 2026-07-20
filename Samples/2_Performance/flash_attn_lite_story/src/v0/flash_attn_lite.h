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

// 返回 true 表示参数校验通过且已发起 kernel launch.
// requestedAicCoreNum 为 0 时使用本卡全部 AIC.
bool FlashAttnLiteNPU(uint8_t *dQ, uint8_t *dK, uint8_t *dV, uint8_t *dOut,
                         uint32_t batchSize, uint32_t seqLen,
                         float softmaxScale, uint32_t requestedAicCoreNum,
                         aclrtStream stream);
