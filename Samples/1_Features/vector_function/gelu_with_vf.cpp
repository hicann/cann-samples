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
 * \file gelu_with_vf.cpp
 * \brief
 */

#include "acl/acl.h"
#include "gelu_cpu.h"
#include "kernel_operator.h"
#include <iostream>
#include <vector>
#include <random>

__simd_vf__ inline void gelu_vf(__ubuf__ float *xAddr, __ubuf__ float *yAddr, uint32_t n, uint32_t loopNum)
{
    const float NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715;
    const float TANH_APPROX_FACTOR = 1 / 0.044715;
    constexpr static uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::MicroAPI::MaskReg pMask;
    AscendC::MicroAPI::RegTensor<float> xReg, yReg, cubeReg, tReg;
    uint32_t count;
    count = static_cast<uint32_t>(n);
    
    for (uint16_t i = 0; i < loopNum; ++i) {
        pMask = AscendC::MicroAPI::UpdateMask<float>(count);
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
            xReg, (__ubuf__ float *)xAddr + i * vectorLength);
        AscendC::MicroAPI::Mul(cubeReg, xReg, xReg, pMask);
        AscendC::MicroAPI::Mul(cubeReg, cubeReg, xReg, pMask);
        AscendC::MicroAPI::Muls(tReg, xReg, TANH_APPROX_FACTOR, pMask);
        AscendC::MicroAPI::Add(cubeReg, cubeReg, tReg, pMask);
        AscendC::MicroAPI::Muls(cubeReg, cubeReg, NEG_SQRT_EIGHT_OVER_PI, pMask);
        AscendC::MicroAPI::Exp(cubeReg, cubeReg, pMask);
        AscendC::MicroAPI::Adds(cubeReg, cubeReg, 1.0f, pMask);
        AscendC::MicroAPI::Div(yReg, xReg, cubeReg, pMask);
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ float *)yAddr + i * vectorLength, yReg, pMask);
    }
}

__aicore__ inline void gelu_compute(const AscendC::LocalTensor<float> &xLocal, const AscendC::LocalTensor<float> &yLocal,
    const AscendC::LocalTensor<float> &xCube, const AscendC::LocalTensor<float> &tLocal, int64_t n)
{
    constexpr static uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t loopNum = (n + vectorLength - 1) / vectorLength;
    __ubuf__ float *xAddr = (__ubuf__ float *)xLocal.GetPhyAddr();
    __ubuf__ float *yAddr = (__ubuf__ float *)yLocal.GetPhyAddr();
    gelu_vf(xAddr, yAddr, static_cast<uint32_t>(n), loopNum);
}

__global__ __aicore__ __vector__ void gelu_kernel(
    GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    constexpr static int64_t PIPELINE_DEPTH = 2;
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> xGm, yGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf2;
    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(outQueueY, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(tempBuf1, tileSize);
    pipe.InitBuffer(tempBuf2, tileSize);
    xGm.SetGlobalBuffer((__gm__ float *)x + blockLength * AscendC::GetBlockIdx());
    yGm.SetGlobalBuffer((__gm__ float *)y + blockLength * AscendC::GetBlockIdx());

    int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
    if (currentBlockLength > blockLength) {
        currentBlockLength = blockLength;
    }
    int64_t elementNumPerTile = tileSize / sizeof(float);
    int64_t tileNum = currentBlockLength / elementNumPerTile;
    int64_t tailTileElementNum = currentBlockLength - tileNum * elementNumPerTile;

    for (int64_t i = 0; i < tileNum; ++i) {
        int64_t offset = i * elementNumPerTile;
        // CopyIn
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = elementNumPerTile * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        // Compute
        xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> xCube = tempBuf1.Get<float>();
        AscendC::LocalTensor<float> tLocal = tempBuf2.Get<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        gelu_compute(xLocal, yLocal, xCube, tLocal, elementNumPerTile);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        // CopyOut
        yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

    if (tailTileElementNum > 0) {
        int64_t offset = tileNum * elementNumPerTile;
        // CopyIn
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = tailTileElementNum * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        // Compute
        xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> xCube = tempBuf1.Get<float>();
        AscendC::LocalTensor<float> tLocal = tempBuf2.Get<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        gelu_compute(xLocal, yLocal, xCube, tLocal, tailTileElementNum);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        // CopyOut
        yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }
}

int main()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    // Generate input data and CPU result
    int numElements = 409600;
    size_t size = numElements * sizeof(float);
    std::vector<float> input(numElements);
    std::vector<float> cpu_result(numElements);
    std::vector<float> npu_result(numElements);
    for (int i = 0; i < numElements; i++) {
        input[i] = dist(gen);
        cpu_result[i] = 0.0f;  // 初始化为0
        npu_result[i] = 0.0f;  // 初始化为0
    }
    gelu_cpu(input, cpu_result);

    // Init npu resources
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // Copy host data to device
    GM_ADDR d_input = nullptr;
    GM_ADDR d_result = nullptr;
    aclrtMalloc((void **)&d_input, size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&d_result, size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(d_input, size, input.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);

    // Call Gelu Kernel
    int64_t numBlocks, blockLength, tileSize;
    numBlocks = 1;
    blockLength = numElements;
    tileSize = 32 * 1024;
    aclrtSynchronizeStream(stream);
    for (int64_t i = 0; i < 5; ++i) {
        gelu_kernel<<<numBlocks, nullptr, stream>>>(d_input, d_result, numElements, blockLength, tileSize);
    }
    aclrtSynchronizeStream(stream);

    // Copy npu result back to host
    aclrtMemcpy(npu_result.data(), size, d_result, size, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtSynchronizeStream(stream);

    // check result
    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        float a = cpu_result[i];
        float b = npu_result[i];
        if (std::abs(a - b) > 0.001) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "GeLU completed successfully!" << std::endl;
    } else {
        std::cout << "GeLU failed!" << std::endl;
    }

    // free resources
    aclrtFree(d_input);
    aclrtFree(d_result);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
