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

#include "tiling/block_attn_res_prepare_tiling_data.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

namespace BlockAttnResPrepareMix {
namespace Attention {
namespace Kernel {

namespace {
namespace BlockAttnResPrepareDetail {
constexpr uint32_t MMAD_BLOCK_NUM = 2U;
constexpr uint32_t MM1_INDEX = 0U;
constexpr uint32_t MM2_INDEX = 1U;
constexpr int32_t S_DIM_INDEX = 0;
constexpr int32_t N_DIM_INDEX = 1;
constexpr int32_t D_DIM_INDEX = 2;
constexpr int32_t T_DIM_INDEX = 3;
constexpr int64_t MMAD_BATCH_OFFSET = 0;
constexpr uint32_t SINGLE_BUFFER_NUM = 1U;
constexpr uint32_t E_BUFFER_NUM = optiling::BLOCK_ATTN_RES_PREPARE_E_BUFFER_NUM;
constexpr uint16_t MODE4_LOCAL_FLAG_COUNT = 10U;

// Mode-4 maps the sibling AIV into the peer flag space by adding 16 to the same local logical flag ID.
constexpr uint16_t DOT_READY_FLAG = 0U;
constexpr uint16_t E_READY_FLAG = 1U;
constexpr uint16_t E_BUFFER_FREE_FLAG = 3U;
constexpr uint16_t AIV1_FLAG_OFFSET = 16U;
constexpr uint8_t WORKSPACE_STORE_EVENT_ID = 0U;
constexpr uint8_t SYNC_MODE = 4U;
static_assert(
    E_BUFFER_FREE_FLAG + E_BUFFER_NUM <= MODE4_LOCAL_FLAG_COUNT,
    "BlockAttnResPrepare local mode-4 flag ID exceeds the hardware range.");

struct TypedGmParams {
    __gm__ float* blockResidual{nullptr};
    __gm__ float* effectiveQuery{nullptr};
    __gm__ uint64_t* validBlocks{nullptr};
    __gm__ float* softmaxMax{nullptr};
    __gm__ float* weightedOutput{nullptr};
    __gm__ float* softmaxSum{nullptr};
    __gm__ float* workspace{nullptr};
};

template <typename T>
__aicore__ inline auto MakeNDExtLayout(int64_t rows, int64_t columns, int64_t rowPitch)
{
    auto shape = AscendC::Te::MakeShape(
        AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, rows), AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, columns));
    auto stride = AscendC::Te::MakeStride(
        AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, rowPitch),
        AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}));
    return AscendC::Te::MakePatternLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<T>>(
        shape, stride);
}

template <typename T>
__aicore__ inline auto MakeBatchedDNExtLayout(
    int64_t batchCount, int64_t rows, int64_t columns, int64_t batchStride, int64_t columnStride)
{
    auto shape = AscendC::Te::MakeShape(
        batchCount, AscendC::Te::MakeShape(
                        AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, rows),
                        AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, columns)));
    auto stride = AscendC::Te::MakeStride(
        batchStride, AscendC::Te::MakeStride(
                         AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}),
                         AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, columnStride)));
    return AscendC::Te::MakePatternLayout<AscendC::Te::DNExtLayoutPtn, AscendC::Te::LayoutTraitDefault<T>>(
        shape, stride);
}

class Sync {
public:
    __aicore__ inline static void NotifyDotReady()
    {
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(DOT_READY_FLAG);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(DOT_READY_FLAG + AIV1_FLAG_OFFSET);
    }

    __aicore__ inline static void WaitDotReady()
    {
        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(DOT_READY_FLAG);
    }

    __aicore__ inline static void NotifyEReady(uint16_t eSlotIdx)
    {
        // Wait until the E workspace GM write has completed before publishing readiness to the consuming AIC.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(WORKSPACE_STORE_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(WORKSPACE_STORE_EVENT_ID);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_S>(E_READY_FLAG + eSlotIdx);
    }

    __aicore__ inline static void WaitEReady(uint16_t eSlotIdx)
    {
        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(E_READY_FLAG + eSlotIdx);
        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(E_READY_FLAG + eSlotIdx + AIV1_FLAG_OFFSET);
    }

    __aicore__ inline static void NotifyEBufferFree(uint16_t eSlotIdx)
    {
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(E_BUFFER_FREE_FLAG + eSlotIdx);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(E_BUFFER_FREE_FLAG + eSlotIdx + AIV1_FLAG_OFFSET);
    }

    __aicore__ inline static void WaitEBufferFree(uint16_t eSlotIdx)
    {
        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(E_BUFFER_FREE_FLAG + eSlotIdx);
    }
};

} // namespace BlockAttnResPrepareDetail
} // namespace

} // namespace Kernel
} // namespace Attention
} // namespace BlockAttnResPrepareMix
