/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include <cmath>
 #include <cstdlib>
 #include <iostream>
 #include <limits.h>
 #include <unistd.h>
 #include <memory>
 #include <random>
 #include <string>
 #include <vector>

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "../common/host_utils/io_utils.h"
#include "include/kernel/quant_matmul_mx_kernel_impl_base.h"
#include "include/block/quant_matmul_mxfp4_block_mmad_a_full_load.h"
#include "include/block/quant_matmul_mxfp4_block_mmad_swat.h"
#include "include/block/block_scheduler_mx_base.h"
#include "include/utils/quant_matmul_constant.h"

namespace Kernel {
__global__ __aicore__ __cube__ void QuantMatmulMxfp4BaseKernel(uint64_t m, uint64_t k, uint64_t n,
        GM_ADDR aGM, GM_ADDR bGM, GM_ADDR aScaleGM, GM_ADDR bScaleGM, GM_ADDR cGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    using AType = fp4x2_e2m1_t;
    using BType = fp4x2_e2m1_t;
    using CType = bfloat16_t;

    using BlockScheduler = Block::QuantMatmulMxLastRoundTileBalanceScheduler;
    // use fullload mode
    using BlockMmadT = Block::BlockMmadMxAFullLoad<AType, BType, CType>;
    using ProblemShape = MatmulShape;
    using QuantMatmulKernelImpl = Kernel::QuantMatmulMxKernelBaseImpl<ProblemShape, BlockMmadT, BlockScheduler>;
    using Params = typename QuantMatmulKernelImpl::Params;

    constexpr uint32_t BASE_M = 128;
    constexpr uint32_t BASE_N = 512;
    constexpr uint32_t BASE_K = 128;
    constexpr uint32_t PINGPONG_NUM = 2;
    constexpr uint32_t M_TAIL_TILE = 1;
    constexpr uint32_t N_TAIL_TILE = 1;
    constexpr uint32_t L1_BUFFER_NUM = 2;

    Params params;
    params.problemShape.m = static_cast<int64_t>(m);
    params.problemShape.n = static_cast<int64_t>(n);
    params.problemShape.k = static_cast<int64_t>(k);
    params.mmadParams.aGmAddr = aGM;
    params.mmadParams.bGmAddr = bGM;
    params.mmadParams.scaleAGmAddr = aScaleGM;
    params.mmadParams.scaleBGmAddr = bScaleGM;
    params.mmadParams.cGmAddr = cGM;
    params.l1Params.kL1 = BASE_K * L1_BUFFER_NUM;
    params.l1Params.scaleKL1 = BASE_K * L1_BUFFER_NUM;
    params.l1Params.l1BufNum = PINGPONG_NUM;
    params.schParams.baseM = BASE_M;
    params.schParams.baseN = BASE_N;
    params.schParams.mTailTile = M_TAIL_TILE;
    params.schParams.nTailTile = N_TAIL_TILE;
    params.qbmmParams.baseM = BASE_M;
    params.qbmmParams.baseN = BASE_N;
    params.qbmmParams.baseK = BASE_K;

    QuantMatmulKernelImpl impl;
    impl(params);
}
} // namespace Kernel

void printUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName << " m k n" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "Example: " << programName << " 100 50 200" << std::endl;
}

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
int main(int argc, char* argv[])
{
    int m, k, n;
    try {
        parseArguments(argc, argv, m, k, n);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    int32_t deviceId = 0;
    aclrtStream stream;

    auto ret = aclInit(nullptr);
    CHECK_COND(ret == ACL_SUCCESS, "aclInit failed.");
    ret = aclrtSetDevice(deviceId);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtSetDevice failed.");
    ret = aclrtCreateStream(&stream);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtCreateStream failed.");

    std::vector<uint8_t> hostA((m * k + 1) >> 1, 0);
    std::vector<uint8_t> hostB((k * n + 1) >> 1, 0);
    std::vector<uint8_t> hostScaleA(m * CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, 0);
    std::vector<uint8_t> hostScaleB(n * CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, 0);
    std::vector<half> hostOutput(m * n, 0);
    auto sizeA = static_cast<size_t>(1) * hostA.size() * sizeof(uint8_t);
    auto sizeB = static_cast<size_t>(1) * hostB.size() * sizeof(uint8_t);
    auto sizeScaleA = static_cast<size_t>(1) * hostScaleA.size() * sizeof(uint8_t);
    auto sizeScaleB = static_cast<size_t>(1) * hostScaleB.size() * sizeof(uint8_t);
    auto sizeOutput = static_cast<size_t>(1) * hostOutput.size() * sizeof(half);
    // Resolve scripts/input|output next to gen_data.py (readlink avoids std::filesystem for older GCC).
    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    std::string baseDir = ".";
    if (len > 0) {
        exePath[len] = '\0';
        baseDir = exePath;
        for (int up = 0; up < 2; ++up) {
            size_t lastSlash = baseDir.find_last_of('/');
            if (lastSlash != std::string::npos && lastSlash > 0) {
                baseDir.resize(lastSlash);
            } else {
                break;
            }
        }
        baseDir += "/scripts";
    }
    std::string inputDir = baseDir + "/input";
    std::string outputDir = baseDir + "/output";
    ReadFile(inputDir + "/input_a.bin", sizeA, hostA.data(), sizeA);
    ReadFile(inputDir + "/input_b.bin", sizeB, hostB.data(), sizeB);
    ReadFile(inputDir + "/input_scaleA.bin", sizeScaleA, hostScaleA.data(), sizeScaleA);
    ReadFile(inputDir + "/input_scaleB.bin", sizeScaleB, hostScaleB.data(), sizeScaleB);

    GM_ADDR deviceA = nullptr;
    GM_ADDR deviceB = nullptr;
    GM_ADDR deviceScaleA = nullptr;
    GM_ADDR deviceScaleB = nullptr;
    GM_ADDR deviceOutput = nullptr;
    ret = aclrtMalloc((void**)&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST);
    std::unique_ptr<void, aclError (*)(void*)> DeviceAAddr(deviceA, aclrtFree);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceA failed.");
    ret = aclrtMalloc((void**)&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST);
    std::unique_ptr<void, aclError (*)(void*)> DeviceBAddr(deviceB, aclrtFree);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceB failed.");
    ret = aclrtMalloc((void**)&deviceScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_FIRST);
    std::unique_ptr<void, aclError (*)(void*)> DeviceScaleAAddr(deviceScaleA, aclrtFree);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceScaleA failed.");
    ret = aclrtMalloc((void**)&deviceScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_FIRST);
    std::unique_ptr<void, aclError (*)(void*)> DeviceScaleBAddr(deviceScaleB, aclrtFree);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceScaleB failed.");
    ret = aclrtMalloc((void**)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_FIRST);
    std::unique_ptr<void, aclError (*)(void*)> DeviceOutputAddr(deviceOutput, aclrtFree);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceOutput failed.");

    ret = aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceA failed.");
    ret = aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceB failed.");
    ret = aclrtMemcpy(deviceScaleA, sizeScaleA, hostScaleA.data(), sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceScaleA failed.");
    ret = aclrtMemcpy(deviceScaleB, sizeScaleB, hostScaleB.data(), sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceScaleB failed.");

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    CHECK_COND(ascendcPlatform != nullptr, "get ascendcPlatform failed.");
    uint32_t blockDim = ascendcPlatform->GetCoreNumAic();

    Kernel::QuantMatmulMxfp4BaseKernel<<<blockDim, nullptr, stream>>>(
        m, k, n, deviceA, deviceB, deviceScaleA, deviceScaleB, deviceOutput);

    ret = aclrtSynchronizeStream(stream);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed.");

    ret = aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceOutput failed.");

    WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);

    std::string cmd = "cd \"" + baseDir + "\" && python3 verify_result.py " + std::to_string(m) + " " +
                      std::to_string(n);
    if (std::system(cmd.c_str()) != 0) {
        return 1;
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
