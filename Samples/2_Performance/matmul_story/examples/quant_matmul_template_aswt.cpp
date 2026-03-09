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
 * \file quant_matmul_mxfp4.cpp
 * \brief
 */
#include <iostream>
#include <cstdlib>
#include <memory>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernel_operator.h"

#include "host_utils/common_utils.h"
#include "host_utils/io_utils.h"
#include "kernel_utils/common_utils.h"
#include "kernel_utils/layout_utils.h"
#include "block/block_mmad_mx.h"
#include "block/block_scheduler_policy.h"
#include "block/block_scheduler_mx.h"
#include "kernel/quant_matmul_mx_kernel_aswt_impl.h"
#include "policy/dispatch_policy.h"
#include "utils/quant_matmul_tiling_data.h"

__global__ __aicore__ void QuantMatmulMxfp4Kernel(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dScaleA, GM_ADDR dScaleB, GM_ADDR dBias, GM_ADDR dC,
    const QuantMatmulTilingData quantMatmulTilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    using AType = fp4x2_e2m1_t;
    using BType = fp4x2_e2m1_t;
    using BiasType = float;
    using CType = float;

    using layoutA = layout::RowMajor;
    using layoutB = layout::ColumnMajor;
    using layoutC = layout::RowMajor;
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;

    using BlockScheduler = QuantMatmulMxAswtScheduler;
    // disable A full load: QuantMatmulMxMultiBlockWithAswt<>
    // enable A full load: QuantMatmulMxMultiBlockWithAswt<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE>
    using DispatchPolicy = QuantMatmulMxMultiBlockWithAswt<>;
    using BlockMmad = Block::BlockMmadMx<
        DispatchPolicy, L1TileShape, L0TileShape, AType, layoutA, BType, layoutB, CType, layoutC, BiasType, layoutC, void>;
    using ProblemShape = MatmulShape;
    using QuantMatmulKernelImpl = Kernel::QuantMatmulMxKernelAswtImpl<ProblemShape, BlockMmad, BlockScheduler>;

    using Params = typename QuantMatmulKernelImpl::Params;

    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    QBMMTiling qbmmParams{quantMatmulTilingData.baseM, quantMatmulTilingData.baseN, quantMatmulTilingData.baseK,
                          static_cast<uint32_t>(quantMatmulTilingData.isBias),
                          static_cast<uint32_t>(quantMatmulTilingData.dbL0C)};

    Params params = {
        {quantMatmulTilingData.m, quantMatmulTilingData.n, quantMatmulTilingData.k, 1},
        {dA, dB, dC, dBias, dScaleA, dScaleB},
        {quantMatmulTilingData.stepK * quantMatmulTilingData.baseK, quantMatmulTilingData.scaleKL1, quantMatmulTilingData.nBufferNum},
        {quantMatmulTilingData.baseM, quantMatmulTilingData.baseN, quantMatmulTilingData.mTailTile, quantMatmulTilingData.nTailTile,
         quantMatmulTilingData.mBaseTailSplitCnt, quantMatmulTilingData.nBaseTailSplitCnt, quantMatmulTilingData.mTailMain,
         quantMatmulTilingData.nTailMain},
        qbmmParams};
    QuantMatmulKernelImpl quantMatmulKernelImpl;
    quantMatmulKernelImpl(params);
}

// 打印使用说明
void printUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName << " m k n" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "Example: " << programName << " 100 50 200" << std::endl;
}

// 解析命令行参数
void parseArguments(int argc, char* argv[], int& m, int& k, int& n)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        printUsage(argv[0]);
        exit(1);
    }
    if (argc < 4) {
        throw std::invalid_argument("ERROR: Lacks Arguments");
    }
    try {
        m = std::stoi(argv[1]);
        k = std::stoi(argv[2]);
        n = std::stoi(argv[3]);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument("ERROR: m k n must be Integer");
    }

    if (m <= 0 || k <= 0 || n <= 0) {
        throw std::invalid_argument("ERROR: m k n must be positive");
    }

    if (k % 2 != 0) {
        throw std::invalid_argument("ERROR: k must be an even number");
    }

    if (CeilDiv(k, 32) % 2 != 0) {
        throw std::invalid_argument("ERROR: k should satisfy that CeilDiv(k, 32) is an even number");
    }
}

void SetTilingData(QuantMatmulTilingData& quantMatmulTilingData, int m, int n, int k)
{
    quantMatmulTilingData.m = m;
    quantMatmulTilingData.n = n;
    quantMatmulTilingData.k = k;
    quantMatmulTilingData.baseM = 256;
    quantMatmulTilingData.baseN = 256;
    quantMatmulTilingData.baseK = 256;
    quantMatmulTilingData.scaleKL1 = 8192;
    quantMatmulTilingData.stepK = 2;
    quantMatmulTilingData.nBufferNum = 2;
    quantMatmulTilingData.isBias = 0;
    quantMatmulTilingData.dbL0C = 1;

    quantMatmulTilingData.mTailTile = 1;
    quantMatmulTilingData.nTailTile = 1;
    quantMatmulTilingData.mBaseTailSplitCnt = 1;
    quantMatmulTilingData.nBaseTailSplitCnt = 1;
    quantMatmulTilingData.mTailMain = 0;
    quantMatmulTilingData.nTailMain = 0;
}

int main(int argc, char* argv[])
{
    // get inputShape
    int m, k, n;
    try {
        parseArguments(argc, argv, m, k, n);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // init
    int32_t deviceId = 0;
    aclrtStream stream;
    uint32_t deviceCount;
    CHECK_COND(aclrtGetDeviceCount(&deviceCount) == ACL_SUCCESS, "Failed to get ACLRT devices.");
    CHECK_COND(deviceCount > 0U, "No ACLRT devices found.");
    CHECK_COND(aclInit(nullptr) == ACL_SUCCESS, "aclInit failed.");
    CHECK_COND(aclrtSetDevice(deviceId) == ACL_SUCCESS, "aclrtSetDevice failed.");
    CHECK_COND(aclrtCreateStream(&stream) == ACL_SUCCESS, "aclrtCreateStream failed.");

    // host data
    uint8_t* hA = nullptr;
    uint8_t* hB = nullptr;
    uint8_t* hScaleA = nullptr;
    uint8_t* hScaleB = nullptr;
    float* hBias = nullptr;
    float* hC = nullptr;

    // device addr
    GM_ADDR dA = nullptr;
    GM_ADDR dB = nullptr;
    GM_ADDR dScaleA = nullptr;
    GM_ADDR dScaleB = nullptr;
    GM_ADDR dBias = nullptr;
    GM_ADDR dC = nullptr;

    // fp4 needs to be divided by 2.
    size_t sizeA = ((m * k + 1) >> 1) * sizeof(uint8_t);
    size_t sizeB = ((k * n + 1) >> 1) * sizeof(uint8_t);
    size_t sizeScaleA = (m * CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE) * sizeof(uint8_t);
    size_t sizeScaleB = (n * CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE) * sizeof(uint8_t);
    size_t sizeBias = n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);

    QuantMatmulTilingData quantMatmulTilingData;
    SetTilingData(quantMatmulTilingData, m, n, k);

    // malloc pinned memory
    CHECK_COND(aclrtMallocHost((void**)&hA, sizeA) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostA(hA, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&hB, sizeB) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostB(hB, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&hScaleA, sizeScaleA) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostScaleA(hScaleA, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&hScaleB, sizeScaleB) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostScaleB(hScaleB, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&hBias, sizeBias) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostBias(hBias, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&hC, sizeC) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostC(hC, aclrtFreeHost);

    ReadFile("./input/input_a.bin", sizeA, hA, sizeA);
    ReadFile("./input/input_b.bin", sizeB, hB, sizeB);
    ReadFile("./input/input_scaleA.bin", sizeScaleA, hScaleA, sizeScaleA);
    ReadFile("./input/input_scaleB.bin", sizeScaleB, hScaleB, sizeScaleB);

    // malloc device memory
    CHECK_COND(aclrtMalloc((void**)&dA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceA(dA, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&dB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceB(dB, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&dScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceScaleA(dScaleA, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&dScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceScaleB(dScaleB, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&dBias, sizeBias, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceBias(dBias, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&dC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceC(dC, aclrtFree);

    // memcpy h2d
    CHECK_COND(
        aclrtMemcpyAsync(dA, sizeA, hA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(dB, sizeB, hB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(dScaleA, sizeScaleA, hScaleA, sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(dScaleB, sizeScaleB, hScaleB, sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(dBias, sizeBias, hBias, sizeBias, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");

    // get platform info
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    CHECK_COND(ascendcPlatform != nullptr, "Get ascendcPlatform failed.");
    uint32_t numBlocks = ascendcPlatform->GetCoreNumAic();

    // kernel launch
    QuantMatmulMxfp4Kernel<<<numBlocks, nullptr, stream>>>(dA, dB, dScaleA, dScaleB, dBias, dC, quantMatmulTilingData);

    // memcpy d2h
    CHECK_COND(
        aclrtMemcpyAsync(hC, sizeC, dC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");

    // Sync
    CHECK_COND(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "aclrtSynchronizeStream failed.");

    WriteFile("./output/npu_out.bin", hC, sizeC);

    // 资源释放
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
