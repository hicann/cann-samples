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

#include <cstdint>

namespace KDALite {

constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t VALUE_DIM = 128;
#ifndef KDALITE_V0_CHUNK_SIZE
#define KDALITE_V0_CHUNK_SIZE 32
#endif
constexpr uint32_t CHUNK_SIZE = KDALITE_V0_CHUNK_SIZE;
constexpr uint32_t DV_TILE = 32;
constexpr uint32_t AIV_DV_TILE = DV_TILE / 2;

static_assert(
    CHUNK_SIZE >= 16 && CHUNK_SIZE <= 64 && (CHUNK_SIZE & (CHUNK_SIZE - 1)) == 0,
    "v0 CHUNK_SIZE must be 16, 32, or 64");
static_assert(VALUE_DIM % DV_TILE == 0, "VALUE_DIM must be divisible by DV_TILE");
static_assert(DV_TILE == 2 * AIV_DV_TILE, "one Mix task must split evenly across two AIVs");
static_assert(HEAD_DIM == VALUE_DIM, "KDALite requires Dk and Dv to have the same fixed size");

// Workspace offset 的单位均为字节. LocalTensor 的地址和元素数在各 Kernel
// 内按固定 shape 显式计算, 不通过 TilingData 传递.
struct KimiDeltaAttnLiteTilingData {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t chunkCount;

    uint32_t prepareNumTasks;
    uint32_t prepareUseAicNum;
    uint32_t stateNumTasks;
    uint32_t stateUseAicNum;
    uint32_t outputNumTasks;
    uint32_t outputUseAicNum;

    uint64_t wOffset;
    uint64_t qPlusOffset;
    uint64_t kTailOffset;
    uint64_t aOffset;
    uint64_t uOffset;
    uint64_t rOffset;
    uint64_t gLastOffset;
    uint64_t oHistoryOffset;
    uint64_t workspaceBytes;
};

void LaunchKimiDeltaAttnLiteKernels(
    uint8_t* dQ, uint8_t* dK, uint8_t* dV, uint8_t* dLogDecay, uint8_t* dBeta, uint8_t* dO, uint8_t* dFinalState,
    uint8_t* dWorkspace, const KimiDeltaAttnLiteTilingData& data, void* stream);

} // namespace KDALite
