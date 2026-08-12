/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kimi_delta_attn_lite.h"
#include "../kimi_delta_attn_lite_common.h"

#include <cstdio>
#include <limits>
#include <tiling/platform/platform_ascendc.h>

namespace {

constexpr uint64_t WORKSPACE_ALIGNMENT = 32;
constexpr uint64_t CHUNK_D_BF16_BYTES = KDALite::CHUNK_SIZE * KDALite::HEAD_DIM * sizeof(uint16_t);
constexpr uint64_t CHUNK_C_BF16_BYTES = KDALite::CHUNK_SIZE * KDALite::CHUNK_SIZE * sizeof(uint16_t);
constexpr uint64_t STATE_DECAY_BYTES = KDALite::HEAD_DIM * sizeof(float);

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool AppendWorkspaceSegment(uint64_t segmentBytes, uint64_t& cursor, uint64_t& offset)
{
    uint64_t aligned = 0;
    if (!CheckedAdd(cursor, WORKSPACE_ALIGNMENT - 1, aligned)) {
        return false;
    }
    aligned = aligned / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
    offset = aligned;
    return CheckedAdd(aligned, segmentBytes, cursor);
}

bool AppendChunkWorkspaceSegment(uint64_t chunkCount, uint64_t bytesPerChunk, uint64_t& cursor, uint64_t& offset)
{
    uint64_t segmentBytes = 0;
    return CheckedMul(chunkCount, bytesPerChunk, segmentBytes) && AppendWorkspaceSegment(segmentBytes, cursor, offset);
}

const char* ComputeKimiDeltaAttnLiteTilingData(
    uint32_t batchSize, uint32_t seqLen, uint32_t mixCoreNum, KDALite::KimiDeltaAttnLiteTilingData& data)
{
    data = {};
    if (batchSize == 0) {
        return "B 必须大于 0";
    }
    if (seqLen == 0) {
        return "S 必须大于 0";
    }
    if (mixCoreNum == 0) {
        return "可用 Mix 组数必须大于 0";
    }
    uint64_t availableAivCoreNum = 0;
    if (!CheckedMul(mixCoreNum, 2, availableAivCoreNum) || availableAivCoreNum > std::numeric_limits<uint32_t>::max()) {
        return "Mix 组对应的 AIV 核数超出 uint32_t 表示范围";
    }

    const uint64_t chunkCount = (static_cast<uint64_t>(seqLen) + KDALite::CHUNK_SIZE - 1) / KDALite::CHUNK_SIZE;
    uint64_t totalChunks = 0;
    uint64_t stateTasks = 0;
    if (!CheckedMul(batchSize, chunkCount, totalChunks) ||
        !CheckedMul(batchSize, KDALite::VALUE_DIM / KDALite::DV_TILE, stateTasks) ||
        chunkCount > std::numeric_limits<uint32_t>::max() || totalChunks > std::numeric_limits<uint32_t>::max() ||
        stateTasks > std::numeric_limits<uint32_t>::max()) {
        return "任务数超出 uint32_t 表示范围";
    }

    data.batchSize = batchSize;
    data.seqLen = seqLen;
    data.chunkCount = static_cast<uint32_t>(chunkCount);
    data.prepareNumTasks = static_cast<uint32_t>(totalChunks);
    const uint32_t aivCoreNum = static_cast<uint32_t>(availableAivCoreNum);
    data.prepareUseAivNum = data.prepareNumTasks < aivCoreNum ? data.prepareNumTasks : aivCoreNum;
    data.stateNumTasks = static_cast<uint32_t>(stateTasks);
    data.stateUseAicNum = data.stateNumTasks < mixCoreNum ? data.stateNumTasks : mixCoreNum;

    uint64_t cursor = 0;
    if (!AppendChunkWorkspaceSegment(totalChunks, CHUNK_D_BF16_BYTES, cursor, data.kPlusOffset) ||
        !AppendChunkWorkspaceSegment(totalChunks, CHUNK_D_BF16_BYTES, cursor, data.qPlusOffset) ||
        !AppendChunkWorkspaceSegment(totalChunks, CHUNK_D_BF16_BYTES, cursor, data.kTailOffset) ||
        !AppendChunkWorkspaceSegment(totalChunks, CHUNK_C_BF16_BYTES, cursor, data.mOffset) ||
        !AppendChunkWorkspaceSegment(totalChunks, CHUNK_C_BF16_BYTES, cursor, data.aOffset) ||
        !AppendChunkWorkspaceSegment(totalChunks, STATE_DECAY_BYTES, cursor, data.stateDecayOffset)) {
        return "workspace 字节数超出 uint64_t 表示范围";
    }

    data.workspaceBytes = cursor;
    return nullptr;
}

} // namespace

bool GetKimiDeltaAttnLiteWorkspaceSize(uint32_t batchSize, uint32_t seqLen, uint64_t& workspaceBytes)
{
    KDALite::KimiDeltaAttnLiteTilingData data{};
    const char* error = ComputeKimiDeltaAttnLiteTilingData(batchSize, seqLen, 1, data);
    if (error != nullptr) {
        std::fprintf(stderr, "无法计算 KDALite workspace：%s\n", error);
        workspaceBytes = 0;
        return false;
    }
    workspaceBytes = data.workspaceBytes;
    return true;
}

bool KimiDeltaAttnLiteNPU(
    uint8_t* dQ, uint8_t* dK, uint8_t* dV, uint8_t* dLogDecay, uint8_t* dBeta, uint8_t* dO, uint8_t* dFinalState,
    uint8_t* dWorkspace, uint64_t workspaceBytes, uint32_t batchSize, uint32_t seqLen, uint32_t requestedMixCoreNum,
    aclrtStream stream)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr) {
        std::fprintf(stderr, "获取 AscendC 平台信息失败\n");
        return false;
    }
    const uint32_t deviceMixCoreNum = ascendcPlatform->GetCoreNumAic();
    if (requestedMixCoreNum > deviceMixCoreNum) {
        std::fprintf(
            stderr, "请求 Mix 组数 %u 超过本卡 Mix 组数 %u，kernel 未启动\n", requestedMixCoreNum, deviceMixCoreNum);
        return false;
    }
    const uint32_t mixCoreNum = requestedMixCoreNum == 0 ? deviceMixCoreNum : requestedMixCoreNum;

    KDALite::KimiDeltaAttnLiteTilingData data{};
    const char* error = ComputeKimiDeltaAttnLiteTilingData(batchSize, seqLen, mixCoreNum, data);
    if (error != nullptr) {
        std::fprintf(stderr, "不支持当前 KDALite 参数：%s\n", error);
        return false;
    }
    if (dQ == nullptr || dK == nullptr || dV == nullptr || dLogDecay == nullptr || dBeta == nullptr || dO == nullptr ||
        dFinalState == nullptr || dWorkspace == nullptr) {
        std::fprintf(stderr, "kernel 未启动：输入、输出和 workspace 设备指针不能为空\n");
        return false;
    }
    if (workspaceBytes < data.workspaceBytes) {
        std::fprintf(
            stderr, "kernel 未启动：workspace 不足，提供 %llu bytes，需要 %llu bytes\n",
            static_cast<unsigned long long>(workspaceBytes), static_cast<unsigned long long>(data.workspaceBytes));
        return false;
    }

    std::printf(
        "kdalite: 请求启动 kernel，ChunkPrepare=%u AIV，FusedRecurrentOutput=%u Mix，"
        "workspace=%llu bytes\n",
        data.prepareUseAivNum, data.stateUseAicNum, static_cast<unsigned long long>(data.workspaceBytes));
    KDALite::LaunchKimiDeltaAttnLiteKernels(
        dQ, dK, dV, dLogDecay, dBeta, dO, dFinalState, dWorkspace, data, reinterpret_cast<void*>(stream));
    return true;
}
