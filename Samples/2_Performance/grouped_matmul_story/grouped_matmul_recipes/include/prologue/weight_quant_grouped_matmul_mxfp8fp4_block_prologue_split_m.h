/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_grouped_matmul_mxfp8fp4_block_prologue_split_m.h
 * \brief Weight-quantized grouped matmul prologue with MXFP8/FP4 casting
 *
 * ## Overview
 * This prologue prepares weight data for grouped matrix multiplication by:
 * 1. Loading 4-bit quantized weights from Global Memory (GM)
 * 2. Anti-quantizing to 8-bit format using vector instructions
 * 3. Moving processed weights to L1 cache for Cube unit consumption
 *
 * ## Pipeline Architecture (3-Stage)
 *
 * ```
 * ┌─────────┐    ┌─────────┐    ┌─────────┐
 * │  MTE2   │───→│  Vector │───→│  MTE3   │
 * │  (Load) │    │(Compute)│    │ (Store) │
 * └─────────┘    └─────────┘    └─────────┘
 *      │              │              │
 *      ▼              ▼              ▼
 *   GM→UB       Anti-quant      UB→L1
 *   4-bit       4-bit→8-bit    8-bit
 * ```
 *
 * ### Stage 1: MTE2 (Memory Transfer Engine 2)
 * - Copies 4-bit weight from GM to Unified Buffer (UB)
 * - Uses quad-buffering for 4-way parallel access
 * - Triggered per K-block iteration
 *
 * ### Stage 2: Vector (Anti-quantization)
 * - Converts 4-bit weights to 8-bit using SIMD instructions
 * - Uses quad-buffering for 8-bit weight output
 *
 * ### Stage 3: MTE3 (Memory Transfer Engine 3)
 * - Moves 8-bit weights from UB to L1 cache
 * - Signals AIC (Cube unit) when data is ready
 *
 * ## Unified Buffer Layout
 *
 * | Offset    | Size  | Content                    | Buffering |
 * |-----------|-------|---------------------------|-----------|
 * | 0KB       | 64KB  | 4-bit weight quad         | Quad      |
 * | 64KB      | 128KB | 8-bit weight quad         | Quad      |
 * | **Total (used)** | **192KB** | Within 248KB UB hardware limit | |
 *
 * ## L1 Buffer Layout (Non-Contiguous)
 *
 * ### 8-bit Weight (Double Buffered, 128KB Total)
 * Two separate 64KB buffers at different L1 locations:
 * - Buffer 0: 64KB at `WEIGHT_L1_INIT_OFFSET`
 * - Buffer 1: 64KB at `WEIGHT_L1_INIT_OFFSET + WEIGHT_L1_DB_OFFSET`
 *
 * Note: Buffers are NOT contiguous in L1 memory
 *
 * ## Key Design Decisions
 *
 * 1. **Quad buffering for 4-bit weight**: Allows MTE2 to load multiple blocks ahead
 *    while Vector unit processes earlier blocks
 *
 * 2. **Quad buffering for 8-bit weight**: Supports the 3-stage pipeline where
 *    MTE2, Vector, and MTE3 can operate on different buffers simultaneously
 *
 * 3. **K-block splitting**: Large K dimensions are split to fit in UB/L1 constraints,
 *    with each sub-block processed independently
 *
 * ## SIMD Model
 *
 * - **AIV (AI Vector) cores**: Execute MTE2, Vector, and MTE3 operations
 * - **AIC (AI Cube) cores**: Consume prepared weights from L1
 * - **Synchronization**: Cross-core flags (0/1) coordinate between AIV and AIC
 *
 * ## Data Types
 *
 * - Input weight: 4-bit per element (packed as 2 elements per byte)
 * - Output weight: 8-bit per element (fp8_e4m3)

 */
#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "../utils/grouped_matmul_constant.h"
#include "../tile/copy_gm_to_ub.h"
#include "../tile/shift_w4_to_w8.h"
#include "../tile/copy_weight_ub_to_l1.h"
#include "../utils/layout_struct.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::GetSubBlockIdx;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::WaitFlag;

using GroupedMatmulRecipe::DOUBLE_BUFFER;
using GroupedMatmulRecipe::SYNC_MODE4;

namespace Prologue {

static constexpr int32_t QUADRUPLE_BUFFER_NUM = 4;

struct PrologueMxCastWOffsetParam {
    uint64_t kSize;
    uint64_t kbL1Size;
    uint64_t nL1Size;
    uint64_t nOffset;
    uint64_t nAlign;
};

// Macro aliases keep this dispatch-policy specialization concise and readable.
#define WQBMM_PROLOGUE_TEMPLATE_PARAM template <class OutType_, class InType_>

#define WQBMM_PROLOGUE_CLASS BlockPrologue<KernelMixDynamicKL1NTailResplit, OutType_, InType_>

WQBMM_PROLOGUE_TEMPLATE_PARAM
class WQBMM_PROLOGUE_CLASS {
public:
    using DispatchPolicy = KernelMixDynamicKL1NTailResplit;
    using OutType = OutType_;
    using InType = InType_;

    static constexpr uint64_t kUbMte2BufferNum = 4;

    struct Params {
        __gm__ InType* ptrB;
    };

    __aicore__ inline BlockPrologue();
    template <typename GMWeightTensorType>
    __aicore__ inline void operator()(
        const GMWeightTensorType& gmWeightTensor, uint64_t mL1Size, uint64_t kSize, uint64_t nL1Size, uint64_t nOffset,
        uint64_t nAlign);
    __aicore__ inline ~BlockPrologue();

protected:
    __aicore__ inline uint64_t CalcDynamicKBlock(uint64_t mL1Size, uint64_t nL1Size) const;
    __aicore__ inline void SetAivToAic();
    __aicore__ inline void WaitAicToAiv();
    template <typename GMWeightTensorType>
    __aicore__ inline void ComputeBasicBlockAivNdKnNzNk(
        const PrologueMxCastWOffsetParam& offsetParam, const GMWeightTensorType& gmWeightTensor);

    __aicore__ inline void WaitVectorToMTE2();
    __aicore__ inline void SetVectorToMTE2();
    template <typename Weight4BitTensorType, typename Weight8BitTensorType>
    __aicore__ inline void WeightAntiQuantComputeNzNk(
        const Weight4BitTensorType& weight4BitTensor, const Weight8BitTensorType& weight8BitTensor);
    template <typename Weight8BitTensorType, typename L1TensorType>
    __aicore__ inline void CopyWeightToL1(
        uint64_t mte2RealK, const Weight8BitTensorType& weight8BitTensor, const L1TensorType& l1Tensor);

    __aicore__ inline void FinalizeVectorCompute();

    // GM to UB copy function
    template <typename GMWeightBaseTensorType, typename Weight4BitTensorType>
    __aicore__ inline void CopyGmToUb(
        uint64_t kOffset, uint64_t mte2RealK, const PrologueMxCastWOffsetParam& param,
        const GMWeightBaseTensorType& gmWeightBaseTensor, const Weight4BitTensorType& weight4BitTensor);

    // === Tensor Creation Helper Functions ===
    // Weight tensor creation functions
    __aicore__ inline auto MakeWeight4BitTensor(uint64_t mte2RealK, uint64_t nL1Size);
    __aicore__ inline auto MakeWeight8BitTensor(uint64_t mte2RealK, uint64_t nL1Size);

    // L1 tensor creation functions
    __aicore__ inline auto MakeL1WeightTensor(uint64_t mte2RealK, uint64_t nL1Size, uint64_t l1SplitOffset);

    uint64_t cvLoopIdx_ = 0;

    uint64_t ubMte2LoopIdx_ = 0;
    uint64_t ubComputeLoopIdx_ = 0;

    // === Buffer Size Unit ===
    static constexpr uint64_t KB = 1024;
    static constexpr uint64_t WEIGHT_L1_INIT_OFFSET = 0;
    static constexpr uint64_t WEIGHT_L1_DB_OFFSET = 384 * KB;
    static constexpr uint64_t L1_WEIGHT_OFFSETS[DOUBLE_BUFFER] = {
        WEIGHT_L1_INIT_OFFSET * sizeof(OutType), WEIGHT_L1_DB_OFFSET * sizeof(OutType)};

    // === Pipeline Buffer Configuration ===
    static constexpr uint64_t WEIGHT_8BIT_BUFFER_NUM = QUADRUPLE_BUFFER_NUM; // 4

    // === UB Memory Layout (192KB actively used in this pipeline) ===
    // [0-64KB): 4-bit weight quad buffers
    static constexpr uint64_t WEIGHT_4BIT_TOTAL_SIZE = 64 * KB;
    static constexpr uint64_t WEIGHT_4BIT_SINGLE_BUFFER_SIZE = WEIGHT_4BIT_TOTAL_SIZE / kUbMte2BufferNum;

    // [64KB-192KB): 8-bit weight quad buffers
    static constexpr uint64_t WEIGHT_8BIT_TOTAL_SIZE = 128 * KB;

    // Compile-time verification: active use stays within the 248KB UB hardware limit.
    static_assert(
        WEIGHT_4BIT_TOTAL_SIZE + WEIGHT_8BIT_TOTAL_SIZE <= 248 * KB,
        "UB buffer total must not exceed 248KB hardware limit");

    // === UB Buffer Base Offsets ===
    static constexpr uint64_t WEIGHT_4BIT_INIT_OFFSET = 0;
    static constexpr uint64_t WEIGHT_8BIT_INIT_OFFSET = WEIGHT_4BIT_TOTAL_SIZE;

    // === UB Buffer Offset Arrays (Compile-time computed for fast lookup) ===
    static constexpr uint64_t WEIGHT_4BIT_OFFSETS[4] = {
        WEIGHT_4BIT_INIT_OFFSET + 0 * WEIGHT_4BIT_SINGLE_BUFFER_SIZE,
        WEIGHT_4BIT_INIT_OFFSET + 1 * WEIGHT_4BIT_SINGLE_BUFFER_SIZE,
        WEIGHT_4BIT_INIT_OFFSET + 2 * WEIGHT_4BIT_SINGLE_BUFFER_SIZE,
        WEIGHT_4BIT_INIT_OFFSET + 3 * WEIGHT_4BIT_SINGLE_BUFFER_SIZE};

    // === Hardware/Architecture Parameters ===
#if __CCE_AICORE__ == 310
    constexpr static uint64_t VEC_REG_ELEM = AscendC::VECTOR_REG_WIDTH;
#else
    constexpr static uint64_t VEC_REG_ELEM = 256;
#endif

    static constexpr uint64_t WEIGHT_8BIT_OFFSETS[4] = {
        WEIGHT_8BIT_INIT_OFFSET + 0 * VEC_REG_ELEM * sizeof(OutType),
        WEIGHT_8BIT_INIT_OFFSET + 1 * VEC_REG_ELEM * sizeof(OutType),
        WEIGHT_8BIT_INIT_OFFSET + 2 * VEC_REG_ELEM * sizeof(OutType),
        WEIGHT_8BIT_INIT_OFFSET + 3 * VEC_REG_ELEM * sizeof(OutType)};
    static constexpr uint64_t WEIGHT_8BIT_LAYOUT_INNER_SIZE = VEC_REG_ELEM * WEIGHT_8BIT_BUFFER_NUM;

    // === Event IDs for Pipeline Synchronization ===
    constexpr static TEventID vecEventIdVToMte2_ = 0;
    constexpr static TEventID vecEventIdMte3ToV_ = 0;
    constexpr static TEventID EVENT_ID_MTE2_TO_V = 0;

    // === Cross-Core Synchronization Flags ===
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 0;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 1;

    // === Dynamic Tiling Configuration ===
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_512 = 512;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256 = 256;

    static constexpr uint64_t FINALIZE_AIC_WAIT_COUNT = 2;
};

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline WQBMM_PROLOGUE_CLASS::BlockPrologue()
{
    for (uint16_t idx = 0; idx < kUbMte2BufferNum; idx++) {
        SetFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_ + idx);
    }

    for (uint16_t idx = 0; idx < WEIGHT_8BIT_BUFFER_NUM; idx++) {
        SetFlag<HardEvent::MTE3_V>(vecEventIdMte3ToV_ + idx);
    }
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline uint64_t WQBMM_PROLOGUE_CLASS::CalcDynamicKBlock(uint64_t mL1Size, uint64_t nL1Size) const
{
    return (mL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256 &&
            nL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD) ?
               MX_A8W4_L1_K_CONFIG_512 :
               MX_A8W4_L1_K_CONFIG_256;
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
template <typename GMWeightTensorType>
__aicore__ inline void WQBMM_PROLOGUE_CLASS::ComputeBasicBlockAivNdKnNzNk(
    const PrologueMxCastWOffsetParam& param, const GMWeightTensorType& gmWeightTensor)
{
    // Setup loop constants
    const uint64_t kMte2BaseSize = param.kbL1Size >> 1;
    const uint64_t l1SplitOffset = GetSubBlockIdx() * kMte2BaseSize;

    // Main processing loop
    for (uint64_t kOffset = 0; kOffset < param.kSize; kOffset += param.kbL1Size, cvLoopIdx_++) {
        // Calculate K block sizes
        uint64_t l1RealLen = (kOffset + param.kbL1Size) > param.kSize ? param.kSize - kOffset : param.kbL1Size;
        uint64_t mte2RealK = GetSubBlockIdx() == 0     ? min(kMte2BaseSize, l1RealLen) :
                             l1RealLen > kMte2BaseSize ? l1RealLen - kMte2BaseSize :
                                                         0;

        // Create tensors using helper functions
        auto weight4BitTensor = MakeWeight4BitTensor(mte2RealK, param.nL1Size);
        auto weight8BitTensor = MakeWeight8BitTensor(mte2RealK, param.nL1Size);
        auto l1Tensor = MakeL1WeightTensor(mte2RealK, param.nL1Size, l1SplitOffset);

        // Pipeline: Stage 1 - Wait and Load from GM
        WaitVectorToMTE2();

        CopyGmToUb(kOffset + GetSubBlockIdx() * kMte2BaseSize, mte2RealK, param, gmWeightTensor, weight4BitTensor);

        // Pipeline: Stage 2 - Wait for AIC and Compute
        WaitAicToAiv();
        WeightAntiQuantComputeNzNk(weight4BitTensor, weight8BitTensor);
        SetVectorToMTE2();
        ubMte2LoopIdx_++;

        // Pipeline: Stage 3 - Copy to L1 and Signal
        CopyWeightToL1(mte2RealK, weight8BitTensor, l1Tensor);
        SetAivToAic();
    }
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
template <typename GMWeightTensorType>
__aicore__ inline void WQBMM_PROLOGUE_CLASS::operator()(
    const GMWeightTensorType& gmWeightTensor, uint64_t mL1Size, uint64_t kSize, uint64_t nL1Size, uint64_t nOffset,
    uint64_t nAlign)
{
    // Type assertions - __aicore__ guarantees these types are valid
    static_assert(std::is_same_v<OutType, __fp8e4m3>, "OutType must be __fp8e4m3");
    static_assert(std::is_same_v<InType, __fp4e2m1x2>, "InType must be __fp4e2m1x2");

    PrologueMxCastWOffsetParam offsetParam = {};
    offsetParam.kSize = kSize;
    offsetParam.nL1Size = nL1Size;
    offsetParam.nOffset = nOffset;
    offsetParam.kbL1Size = CalcDynamicKBlock(mL1Size, nL1Size);
    offsetParam.nAlign = nAlign;
    ComputeBasicBlockAivNdKnNzNk(offsetParam, gmWeightTensor);
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline WQBMM_PROLOGUE_CLASS::~BlockPrologue()
{
    for (uint64_t idx = 0; idx < FINALIZE_AIC_WAIT_COUNT; ++idx) {
        WaitAicToAiv();
    }
    FinalizeVectorCompute();
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::SetAivToAic()
{
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::WaitAicToAiv()
{
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::WaitVectorToMTE2()
{
    WaitFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_ + (ubMte2LoopIdx_ & (kUbMte2BufferNum - 1)));
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::SetVectorToMTE2()
{
    SetFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_ + (ubMte2LoopIdx_ & (kUbMte2BufferNum - 1)));
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
template <typename GMWeightBaseTensorType, typename Weight4BitTensorType>
__aicore__ inline void WQBMM_PROLOGUE_CLASS::CopyGmToUb(
    uint64_t kOffset, uint64_t mte2RealK, const PrologueMxCastWOffsetParam& param,
    const GMWeightBaseTensorType& gmWeightBaseTensor, const Weight4BitTensorType& weight4BitTensor)
{
    if (mte2RealK > 0) {
        auto gmSliceTensor = gmWeightBaseTensor(
            AscendC::Te::MakeCoord(kOffset, param.nOffset), AscendC::Te::MakeShape(mte2RealK, param.nL1Size));
        auto copyGM2UBWeight = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UBWeight{});
        AscendC::Te::Copy(copyGM2UBWeight, weight4BitTensor, gmSliceTensor);
    }

    // Synchronization point after copy completes
    SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
template <typename Weight4BitTensorType, typename Weight8BitTensorType>
__aicore__ inline void WQBMM_PROLOGUE_CLASS::WeightAntiQuantComputeNzNk(
    const Weight4BitTensorType& weight4BitTensor, const Weight8BitTensorType& weight8BitTensor)
{
    WaitFlag<HardEvent::MTE3_V>(vecEventIdMte3ToV_ + (ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1)));

    // Pure compute: anti-quantization using tile helper
    Tile::ShiftW4ToW8<OutType, InType>(weight4BitTensor, weight8BitTensor);

    // Set/Wait flags AFTER compute
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
template <typename Weight8BitTensorType, typename L1TensorType>
__aicore__ inline void WQBMM_PROLOGUE_CLASS::CopyWeightToL1(
    uint64_t mte2RealK, const Weight8BitTensorType& weight8BitTensor, const L1TensorType& l1Tensor)
{
    if (likely(mte2RealK > 0)) {
        // Copy weight 8-bit from UB to L1 (inlined from CopyWeight8BitForAligned)
        auto copyUB2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyUB2L1Custom{});
        AscendC::Te::Copy(copyUB2L1, l1Tensor, weight8BitTensor);
    }
    SetFlag<HardEvent::MTE3_V>(vecEventIdMte3ToV_ + (ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1)));
    ubComputeLoopIdx_++;
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::FinalizeVectorCompute()
{
    for (uint16_t idx = 0; idx < WEIGHT_8BIT_BUFFER_NUM; idx++) {
        WaitFlag<HardEvent::MTE3_V>(vecEventIdMte3ToV_ + idx);
    }

    for (uint16_t idx = 0; idx < kUbMte2BufferNum; idx++) {
        WaitFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_ + idx);
    }
}

// === Tensor Creation Helper Function Implementations ===

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline auto WQBMM_PROLOGUE_CLASS::MakeWeight4BitTensor(uint64_t mte2RealK, uint64_t nL1Size)
{
    return AscendC::Te::MakeTensor(
        AscendC::Te::MakeUBmemPtr<InType>(WEIGHT_4BIT_OFFSETS[ubMte2LoopIdx_ & (kUbMte2BufferNum - 1)]),
        AscendC::Te::Weight4BitLayout<InType>{}(static_cast<int64_t>(mte2RealK), static_cast<int64_t>(nL1Size)));
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline auto WQBMM_PROLOGUE_CLASS::MakeWeight8BitTensor(uint64_t mte2RealK, uint64_t nL1Size)
{
    return AscendC::Te::MakeTensor(
        AscendC::Te::MakeUBmemPtr<OutType>(WEIGHT_8BIT_OFFSETS[ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1)]),
        AscendC::Te::Weight8BitUBLayout<OutType, WEIGHT_8BIT_LAYOUT_INNER_SIZE>{}(mte2RealK, nL1Size));
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline auto WQBMM_PROLOGUE_CLASS::MakeL1WeightTensor(
    uint64_t mte2RealK, uint64_t nL1Size, uint64_t l1SplitOffset)
{
    auto l1BaseLayout = AscendC::Te::MakeZnLayout<OutType>(mte2RealK, nL1Size);
    auto l1BaseTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeL1memPtr<OutType>(L1_WEIGHT_OFFSETS[cvLoopIdx_ & (DOUBLE_BUFFER - 1)]), l1BaseLayout);
    return l1BaseTensor(AscendC::Te::MakeCoord(l1SplitOffset, 0), AscendC::Te::MakeShape(mte2RealK, nL1Size));
}

#undef WQBMM_PROLOGUE_CLASS
#undef WQBMM_PROLOGUE_TEMPLATE_PARAM
} // namespace Prologue
