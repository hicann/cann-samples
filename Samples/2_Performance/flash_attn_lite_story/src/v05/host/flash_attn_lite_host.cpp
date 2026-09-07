/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_lite.h"
#include "../flash_attn_lite_common.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <tiling/platform/platform_ascendc.h>

namespace {

constexpr uint32_t L1_CAPACITY_BYTES = 512 * 1024;
constexpr uint32_t L0A_CAPACITY_BYTES = 64 * 1024;
constexpr uint32_t L0B_CAPACITY_BYTES = 64 * 1024;
constexpr uint32_t L0C_CAPACITY_BYTES = 256 * 1024;
constexpr uint32_t UB_CAPACITY_BYTES = 248 * 1024;

const char* ComputeFlashAttnLiteTilingData(
    uint32_t batchSize, uint32_t headNum, uint32_t seqLen, float scale, uint32_t aicoreNum,
    FALite::FlashAttnLiteTilingData& data)
{
    data = {};
    data.br = 128;
    data.bc = 128;
    if (batchSize == 0) {
        return "B 必须大于 0";
    }
    if (headNum == 0) {
        return "N 必须大于 0";
    }
    if (seqLen == 0) {
        return "S 必须大于 0";
    }
    if (scale == 0.0f || !std::isfinite(scale)) {
        return "scale 必须非 0 且为有限值";
    }
    if (aicoreNum == 0) {
        return "可用 AIC 核数必须大于 0";
    }
    const uint64_t batchHeadNum = static_cast<uint64_t>(batchSize) * headNum;
    if (batchHeadNum > std::numeric_limits<uint32_t>::max()) {
        return "B*N 超出 uint32_t 表示范围";
    }
    data.batchHeadNum = static_cast<uint32_t>(batchHeadNum);
    data.seqLen = seqLen;
    data.scale = scale;
    data.tr = seqLen / data.br + static_cast<uint32_t>(seqLen % data.br != 0);
    data.tc = seqLen / data.bc + static_cast<uint32_t>(seqLen % data.bc != 0);
    const uint64_t numTasks = batchHeadNum * data.tr;
    if (numTasks > std::numeric_limits<uint32_t>::max()) {
        return "任务数 B*N*ceil(S/128) 超出 uint32_t 表示范围";
    }
    data.numTasks = static_cast<uint32_t>(numTasks);
    data.useAicNum = data.numTasks < aicoreNum ? data.numTasks : aicoreNum;
    data.pLayoutMode = FALite::PLayoutMode::DN_TO_NZ;

    auto& aic = data.layoutAIC;
    // Host 侧用等宽的 uint16_t 计算 BF16 缓冲区字节数.
    // P/K/Q/V 在 L1 中连续排布.
    aic.pL1Addr = 0;
    aic.pL1Elems = FALite::PIPELINE_SLOT_NUM * data.br * data.bc;
    aic.kL1Addr = aic.pL1Addr + aic.pL1Elems * sizeof(uint16_t);
    aic.kL1Elems = FALite::PIPELINE_SLOT_NUM * data.bc * FALite::HEAD_DIM;
    aic.qL1Addr = aic.kL1Addr + aic.kL1Elems * sizeof(uint16_t);
    aic.qL1Elems = data.br * FALite::HEAD_DIM;
    aic.vL1Addr = aic.qL1Addr + aic.qL1Elems * sizeof(uint16_t);
    aic.vL1Elems = FALite::PIPELINE_SLOT_NUM * data.bc * FALite::HEAD_DIM;

    // C1 和 C2 复用同一组 L0 slot; 每槽按两阶段的较大需求分配.
    aic.aL0AAddr = 0;
    aic.aL0AElems = FALite::PIPELINE_SLOT_NUM * data.bc * (data.br > FALite::HEAD_DIM ? data.br : FALite::HEAD_DIM);
    aic.bL0BAddr = 0;
    aic.bL0BElems = FALite::PIPELINE_SLOT_NUM * FALite::HEAD_DIM * (data.br > data.bc ? data.br : data.bc);
    aic.cL0CAddr = 0;
    aic.cL0CElems = FALite::PIPELINE_SLOT_NUM * data.br * (data.bc > FALite::HEAD_DIM ? data.bc : FALite::HEAD_DIM);

    auto& aiv = data.layoutAIV;
    const uint32_t halfBr = data.br / 2;
    aiv.sUBAddr = 0;
    aiv.sUBElems = FALite::PIPELINE_SLOT_NUM * halfBr * data.bc;
    aiv.oDeltaUBAddr = aiv.sUBAddr + aiv.sUBElems * sizeof(float);
    aiv.oDeltaUBElems = FALite::PIPELINE_SLOT_NUM * halfBr * FALite::HEAD_DIM;
    aiv.oAccUBAddr = aiv.oDeltaUBAddr + aiv.oDeltaUBElems * sizeof(float);
    aiv.oAccUBElems = halfBr * FALite::HEAD_DIM;
    aiv.pWorkUBAddr = aiv.oAccUBAddr + aiv.oAccUBElems * sizeof(float);
    // 每个 16 列 NZ 分组固定含 Bc 个数据 DataBlock 和 1 个布局 padding block；
    // 每个 PWork 槽占 halfBr * (Bc + 1) 个 BF16 元素。
    aiv.pWorkUBElems = FALite::PIPELINE_SLOT_NUM * halfBr * (data.bc + 1);
    aiv.mUBAddr = aiv.pWorkUBAddr + aiv.pWorkUBElems * sizeof(uint16_t);
    aiv.rowStatsUBElems = halfBr;
    aiv.lUBAddr = aiv.mUBAddr + aiv.rowStatsUBElems * sizeof(float);
    aiv.alphaUBAddr = aiv.lUBAddr + aiv.rowStatsUBElems * sizeof(float);

    if (aic.vL1Addr + aic.vL1Elems * sizeof(uint16_t) > L1_CAPACITY_BYTES ||
        aic.aL0AElems * sizeof(uint16_t) > L0A_CAPACITY_BYTES ||
        aic.bL0BElems * sizeof(uint16_t) > L0B_CAPACITY_BYTES || aic.cL0CElems * sizeof(float) > L0C_CAPACITY_BYTES ||
        aiv.alphaUBAddr + FALite::PIPELINE_SLOT_NUM * aiv.rowStatsUBElems * sizeof(float) > UB_CAPACITY_BYTES) {
        return "SRAM 布局超过片上空间容量";
    }
    if (aiv.pWorkUBElems < FALite::PIPELINE_SLOT_NUM * halfBr * FALite::HEAD_DIM) {
        return "P 工作区不足以复用为最终输出";
    }
    return nullptr;
}

} // namespace

bool FlashAttnLiteNPU(
    uint8_t* dQ, uint8_t* dK, uint8_t* dV, uint8_t* dOut, uint32_t batchSize, uint32_t headNum, uint32_t seqLen,
    float softmaxScale, uint32_t requestedAicCoreNum, aclrtStream stream)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr) {
        std::fprintf(stderr, "获取 AscendC 平台信息失败\n");
        return false;
    }
    const uint32_t deviceAicCoreNum = ascendcPlatform->GetCoreNumAic();
    if (requestedAicCoreNum > deviceAicCoreNum) {
        std::fprintf(
            stderr, "请求 AIC 核数 %u 超过设备 AIC 核数 %u，kernel 未启动\n", requestedAicCoreNum, deviceAicCoreNum);
        return false;
    }
    const uint32_t aicCoreNum = requestedAicCoreNum == 0 ? deviceAicCoreNum : requestedAicCoreNum;

    FALite::FlashAttnLiteTilingData data{};
    const char* error = ComputeFlashAttnLiteTilingData(batchSize, headNum, seqLen, softmaxScale, aicCoreNum, data);
    if (error != nullptr) {
        std::fprintf(stderr, "FALite 参数不受支持：%s\n", error);
        return false;
    }
    if (dQ == nullptr || dK == nullptr || dV == nullptr || dOut == nullptr) {
        std::fprintf(stderr, "kernel 未启动：Q/K/V/O 设备指针不能为空\n");
        return false;
    }

    std::printf("falite: 请求启动 kernel，AIC 核数=%u，scale=%g\n", aicCoreNum, static_cast<double>(softmaxScale));
    FALite::LaunchFlashAttnLiteKernel(dQ, dK, dV, dOut, data, stream);
    return true;
}
