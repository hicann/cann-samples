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
#include "_shared_k.h"

namespace FALite {

constexpr uint16_t FLAG_S_READY = 0;
constexpr uint16_t FLAG_O_READY = 1;
constexpr uint16_t FLAG_DONE = 2;
constexpr uint16_t FLAG_P_READY = 4;

} // namespace FALite
