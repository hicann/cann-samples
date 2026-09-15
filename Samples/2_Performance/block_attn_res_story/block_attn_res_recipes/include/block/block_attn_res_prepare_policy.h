/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_policy.h
 * \brief
 */
#pragma once

namespace BlockAttnResPrepareMix {
namespace Attention {

// Select the AttentionUniversal specialization that orchestrates BlockAttnResPrepare MM1, AIV epilogue, and MM2.
struct KernelBlockAttnResPrepareSchedule {};

// Compile-time component contract shared by the Kernel and its BlockEpilogue.
struct BlockAttnResPreparePolicy {
    using ScheduleType = KernelBlockAttnResPrepareSchedule;
};

} // namespace Attention
} // namespace BlockAttnResPrepareMix
