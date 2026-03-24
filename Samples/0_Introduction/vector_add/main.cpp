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
 * \file main.cpp
 * \brief Vector Addition Example
 */

#include <iostream>
#include <random>
#include <tuple>
#include <algorithm>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

// ACL 错误检查宏
#define CHECK_ACL(call)                                              \
    do {                                                             \
        aclError err = (call);                                       \
        if (err != ACL_SUCCESS) {                                    \
            std::cerr << "ACL error: " << err << " at " << __FILE__ \
                      << ":" << __LINE__ << std::endl;              \
            return 1;                                                \
        }                                                            \
    } while (0)

// 自定义删除器，安全处理空指针
struct AclrtFreeDeleter {
    void operator()(void* ptr) const {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

std::tuple<int64_t, int64_t, int64_t> calc_tiling_params(int64_t totalLength)
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
    constexpr static int64_t PIPELINE_DEPTH = 2;
    constexpr static int64_t BUFFER_NUM = 3;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    numBlocks = std::max(numBlocks, static_cast<int64_t>(1));
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    int64_t tileSize = ubSize / PIPELINE_DEPTH / BUFFER_NUM;
    return std::make_tuple(numBlocks, blockLength, tileSize);
}

template <typename T>
__global__ __aicore__ __vector__ void add_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    constexpr static int64_t PIPELINE_DEPTH = 2;
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm, yGm, zGm;
    AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> outQueueZ;
    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(inQueueY, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(outQueueZ, PIPELINE_DEPTH, tileSize);
    xGm.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
    yGm.SetGlobalBuffer((__gm__ T *)y + blockLength * AscendC::GetBlockIdx());
    zGm.SetGlobalBuffer((__gm__ T *)z + blockLength * AscendC::GetBlockIdx());

    int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
    if (currentBlockLength > blockLength) {
        currentBlockLength = blockLength;
    }
    int64_t elementNumPerTile = tileSize / sizeof(T);
    int64_t tileNum = currentBlockLength / elementNumPerTile;
    int64_t tailTileElementNum = currentBlockLength - tileNum * elementNumPerTile;

    for (int64_t i = 0; i < tileNum; ++i) {
        int64_t offset = i * elementNumPerTile;
        // CopyIn
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = elementNumPerTile * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        // Compute
        xLocal = inQueueX.DeQue<T>();
        yLocal = inQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        AscendC::Add(zLocal, xLocal, yLocal, elementNumPerTile);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        // CopyOut
        zLocal = outQueueZ.DeQue<T>();
        AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

    if (tailTileElementNum > 0) {
        int64_t offset = tileNum * elementNumPerTile;
        // CopyIn
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = tailTileElementNum * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        // Compute
        xLocal = inQueueX.DeQue<T>();
        yLocal = inQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        AscendC::Add(zLocal, xLocal, yLocal, tailTileElementNum);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        // CopyOut
        zLocal = outQueueZ.DeQue<T>();
        AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }
}

int run_vector_add(aclrtStream stream, int64_t numElements)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);

    size_t size = static_cast<size_t>(numElements) * sizeof(float);

    // Host 内存
    std::vector<float> h_A(numElements);
    std::vector<float> h_B(numElements);
    std::vector<float> h_C(numElements);

    for (int64_t i = 0; i < numElements; ++i) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
        h_C[i] = 0.0f;
    }

    // Device 内存 - 使用智能指针管理
    GM_ADDR d_A = nullptr;
    GM_ADDR d_B = nullptr;
    GM_ADDR d_C = nullptr;
    CHECK_ACL(aclrtMalloc((void **)&d_A, size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&d_B, size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&d_C, size, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_A_guard(d_A);
    std::unique_ptr<void, AclrtFreeDeleter> d_B_guard(d_B);
    std::unique_ptr<void, AclrtFreeDeleter> d_C_guard(d_C);

    CHECK_ACL(aclrtMemcpy(d_A, size, h_A.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_B, size, h_B.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));

    // Kernel Call
    int64_t numBlocks, blockLength, tileSize;
    std::tie(numBlocks, blockLength, tileSize) = calc_tiling_params(numElements);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    add_kernel<float><<<numBlocks, nullptr, stream>>>(d_A, d_B, d_C, numElements, blockLength, tileSize);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(h_C.data(), size, d_C, size, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // 验证结果
    bool success = true;
    for (int64_t i = 0; i < numElements; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Vector add completed successfully!" << std::endl;
    } else {
        std::cout << "Vector add failed!" << std::endl;
    }

    return success ? 0 : 1;
}

int main()
{
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    int result = run_vector_add(stream, 409600);

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return result;
}
