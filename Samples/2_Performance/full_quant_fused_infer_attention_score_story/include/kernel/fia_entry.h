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
 * \file fia_entry.h
 * \brief
 */

#ifndef PROMPT_FLASH_ATTENTION_ENTRY_310_H_
#define PROMPT_FLASH_ATTENTION_ENTRY_310_H_
#include "flash_attention_score_tiling_regbase.h"
#include "flash_attention_score_kernel_infer.h"
#include "fia_enum.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
static constexpr uint8_t inOutLayoutType = 0;
static constexpr bool hasAttenMask = 0;
static constexpr uint16_t config = 17;

__aicore__ inline void CopyTiling(FlashAttentionScoreSimplifiedTilingData *tilingData, __gm__ uint8_t *tilingGM)
{
    int64_t *ptr = reinterpret_cast<int64_t *>(tilingData);
    auto tiling32 = reinterpret_cast<__gm__ int64_t *>(tilingGM);
    for (int64_t i = 0; i < sizeof(FlashAttentionScoreSimplifiedTilingData) / sizeof(int64_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

inline __aicore__ void FlashAttentionEntry(__gm__ uint8_t *query, __gm__ uint8_t *key,
    __gm__ uint8_t *value, __gm__ uint8_t *key_antiquant_scale, __gm__ uint8_t *value_antiquant_scale, __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    FlashAttentionScoreSimplifiedTilingData  tilingDataTemp;
    CopyTiling(&tilingDataTemp, tiling);

    FlashAttentionScoreSimplifiedTilingData* __restrict tilingData = &tilingDataTemp;
    
    PARSE_PARAMS_NoQuant(inOutLayoutType, config, hasAttenMask);
    constexpr uint64_t qkvSizeRsv2 =
        MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                static_cast<uint64_t>(dTemplateType),
            static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) *
        2;
    constexpr uint64_t vec1ResultSize =
        static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
    TPipe tPipe;

    if ASCEND_IS_AIC {  // CUBE 实现
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,
            BaseApi::FABlockCube<fp8_e4m3fn_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                static_cast<PseTypeEnum>(9),
                hasAttenMask,
                false,
                false,
                true,
                false,
                false,
                false>,
            BaseApi::FABlockCubeDummy<fp8_e4m3fn_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                static_cast<PseTypeEnum>(9),
                hasAttenMask,
                false,
                false,
                true,
                false,
                false,
                false>>::type;
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,
            BaseApi::FABlockVecDummy<fp8_e4m3fn_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                static_cast<PseTypeEnum>(9),
                hasAttenMask,
                false,
                false,
                true,
                false,
                false,
                false>,
            BaseApi::FABlockVecInfer<fp8_e4m3fn_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                static_cast<PseTypeEnum>(9),
                hasAttenMask,
                false,
                false,
                true,
                false,
                false,
                false>>::type;

        BaseApi::FlashAttentionScoreKernelInfer<CubeBlockType, VecBlockType> op;

        op.InitBaseAPI(query, key, value, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, dequantScaleQuery, key_antiquant_scale, value_antiquant_scale, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, attentionOut, user, nullptr, &tPipe);
        op.Process();

    } else {  // VECTOR 实现
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,
            BaseApi::FABlockCube<fp8_e4m3fn_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                static_cast<PseTypeEnum>(9),
                hasAttenMask,
                false,
                false,
                true,
                false,
                false,
                false>,
            BaseApi::FABlockCubeDummy<fp8_e4m3fn_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                static_cast<PseTypeEnum>(9),
                hasAttenMask,
                false,
                false,
                true,
                false,
                false,
                false>>::type;
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,
            BaseApi::FABlockVecDummy<fp8_e4m3fn_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                static_cast<PseTypeEnum>(9),
                hasAttenMask,
                false,
                false,
                true,
                false,
                false,
                false>,
            BaseApi::FABlockVecInfer<fp8_e4m3fn_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                static_cast<PseTypeEnum>(9),
                hasAttenMask,
                false,
                false,
                true,
                false,
                false,
                false>>::type;
        BaseApi::FlashAttentionScoreKernelInfer<CubeBlockType, VecBlockType> op;
        op.InitBaseAPI(query, key, value, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, dequantScaleQuery, key_antiquant_scale, value_antiquant_scale, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, attentionOut, user, tilingData, &tPipe);
        op.Process();
    }
}
#endif  // end of PROMPT_FLASH_ATTENTION_ENTRY_310_H_