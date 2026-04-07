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
 * \file quant_matmul_mxfp4_swat.cpp
 * \brief Sample launcher for the MXFP4 SWAT streaming example.
 */

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "acl/acl.h"
#include "kernel_operator.h"

#include "block/block_scheduler_policy.h"
#include "host_utils/common_utils.h"
#include "host_utils/io_utils.h"
#include "kernel/quant_matmul_mx_kernel_swat.h"
#include "tiling/quant_matmul_mx_tiling_swat.h"
#include "tiling/quant_matmul_tiling_data.h"

__global__ __aicore__ void QuantMatmulMxfp4SwatKernel(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dScaleA, GM_ADDR dScaleB, GM_ADDR dC,
    const QuantMatmulTilingData quantMatmulTilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    // Keep the sample explicit about the datatype/layout combination that this
    // executable demonstrates so host tiling and kernel traits stay in sync.
    using AType = fp4x2_e2m1_t;
    using BType = fp4x2_e2m1_t;
    using CType = bfloat16_t;

    using layoutA = layout::RowMajor;
    using layoutB = layout::ColumnMajor;
    using layoutC = layout::RowMajor;

    using BlockScheduler = QuantMatmulMxSwatScheduler<SWAT_NO_FULL_LOAD_MODE>;
    using DispatchPolicy =
        QuantMatmulMxMultiBlockWithSwat<SWAT_NO_FULL_LOAD_MODE>;
    using BlockMmad = Block::BlockMmad<
        DispatchPolicy, AType, layoutA, BType, layoutB, CType, layoutC>;
    using ProblemShape = MatmulShape;
    using QuantMatmulKernelImpl =
        Kernel::QuantMatmulMxKernelSwat<ProblemShape, BlockMmad, BlockScheduler>;

    using Params = typename QuantMatmulKernelImpl::Params;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    // Translate the serialized host tiling data into the strongly typed kernel
    // parameter bundle expected by the device-side implementation.
    QBMMTiling qbmmParams{quantMatmulTilingData.baseM, quantMatmulTilingData.baseN, quantMatmulTilingData.baseK,
                          quantMatmulTilingData.dbL0c};
    Params params = {
        {quantMatmulTilingData.m, quantMatmulTilingData.n, quantMatmulTilingData.k, 1UL},
        {dA, dB, dC, dScaleA, dScaleB},
        {quantMatmulTilingData.stepK * quantMatmulTilingData.baseK, quantMatmulTilingData.scaleKL1,
         quantMatmulTilingData.nBufferNum},
        {quantMatmulTilingData.baseM, quantMatmulTilingData.baseN, quantMatmulTilingData.mTailTile,
         quantMatmulTilingData.nTailTile, quantMatmulTilingData.mBaseTailSplitCnt,
         quantMatmulTilingData.nBaseTailSplitCnt, quantMatmulTilingData.mTailMain, quantMatmulTilingData.nTailMain},
        qbmmParams};
    QuantMatmulKernelImpl quantMatmulKernelImpl;
    quantMatmulKernelImpl(params);
}

namespace {
uint64_t ParsePositiveUint64(const char* arg, const char* name)
{
    std::string value(arg);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(std::string("ERROR: ") + name + " must be a positive integer");
    }

    try {
        uint64_t parsed = std::stoull(value);
        if (parsed == 0UL) {
            throw std::invalid_argument(std::string("ERROR: ") + name + " must be greater than 0");
        }
        return parsed;
    } catch (const std::out_of_range&) {
        throw std::invalid_argument(std::string("ERROR: ") + name + " is out of range for uint64_t");
    }
}

void PrintUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName << " m k n" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "Example: " << programName << " 100 50 200" << std::endl;
}

void ParseArguments(int argc, char* argv[], uint64_t& m, uint64_t& k, uint64_t& n)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        PrintUsage(argv[0]);
        exit(1);
    }
    if (argc != 4) {
        throw std::invalid_argument("ERROR: Invalid number of arguments, expected exactly 3 arguments: m k n");
    }
    m = ParsePositiveUint64(argv[1], "m");
    k = ParsePositiveUint64(argv[2], "k");
    n = ParsePositiveUint64(argv[3], "n");

    if (k % 2 != 0) {
        throw std::invalid_argument("ERROR: k must be an even number");
    }
}

}

int main(int argc, char* argv[])
{
    uint64_t m = 0;
    uint64_t k = 0;
    uint64_t n = 0;
    try {
        ParseArguments(argc, argv, m, k, n);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }

    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    bool aclInitialized = false;
    bool deviceSet = false;
    auto cleanupAcl = [&]() {
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
            stream = nullptr;
        }
        if (deviceSet) {
            aclrtResetDevice(deviceId);
            deviceSet = false;
        }
        if (aclInitialized) {
            aclFinalize();
            aclInitialized = false;
        }
    };

    try {
        QuantMatmulTilingData tilingData;
        // Host tiling picks the block shape, tail strategy, and buffering plan
        // that will later be consumed by the device kernel.
        QuantMatmulTilingSwat<DataType::FP4, DataType::FP4> tilingEngine;
        tilingEngine.GetTilingData(m, n, k, tilingData);

        uint32_t deviceCount = 0;
        CHECK_COND(aclrtGetDeviceCount(&deviceCount) == ACL_SUCCESS, "Failed to query ACL device count.");
        CHECK_COND(deviceCount > 0U, "No ACL devices are available.");
        CHECK_COND(aclInit(nullptr) == ACL_SUCCESS, "Failed to initialize ACL runtime.");
        aclInitialized = true;
        CHECK_COND(aclrtSetDevice(deviceId) == ACL_SUCCESS, "Failed to set the ACL device.");
        deviceSet = true;
        CHECK_COND(aclrtCreateStream(&stream) == ACL_SUCCESS, "Failed to create the ACL stream.");

        uint64_t sizeA = ((m * k) >> 1) * sizeof(uint8_t);
        uint64_t sizeB = ((k * n) >> 1) * sizeof(uint8_t);
        uint64_t sizeScaleA = (m * CeilDiv(k, TILING_MXFP_DIVISOR_SIZE) * TILING_MXFP_MULTI_BASE_SIZE) * sizeof(uint8_t);
        uint64_t sizeScaleB = (n * CeilDiv(k, TILING_MXFP_DIVISOR_SIZE) * TILING_MXFP_MULTI_BASE_SIZE) * sizeof(uint8_t);
        uint64_t sizeC = m * n * sizeof(half);

        char exePath[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
        std::string baseDir = ".";
        if (len > 0) {
            // Keep all example assets next to the installed executable so the
            // launcher, generator, and verifier agree on one local layout.
            exePath[len] = '\0';
            baseDir = exePath;
            size_t lastSlash = baseDir.find_last_of('/');
            if (lastSlash != std::string::npos) {
                baseDir.resize(lastSlash);
            }
        }
        std::string inputDir = baseDir + "/input";
        std::string outputDir = baseDir + "/output";

        uint8_t* hA = nullptr;
        uint8_t* hB = nullptr;
        uint8_t* hScaleA = nullptr;
        uint8_t* hScaleB = nullptr;
        half* hC = nullptr;

        GM_ADDR dA = nullptr;
        GM_ADDR dB = nullptr;
        GM_ADDR dScaleA = nullptr;
        GM_ADDR dScaleB = nullptr;
        GM_ADDR dC = nullptr;

        CHECK_COND(aclrtMallocHost((void**)&hA, sizeA) == ACL_SUCCESS, "Failed to allocate the host buffer for input A.");
        std::unique_ptr<void, aclError (*)(void*)> hostA(hA, aclrtFreeHost);
        CHECK_COND(aclrtMallocHost((void**)&hB, sizeB) == ACL_SUCCESS, "Failed to allocate the host buffer for input B.");
        std::unique_ptr<void, aclError (*)(void*)> hostB(hB, aclrtFreeHost);
        CHECK_COND(
            aclrtMallocHost((void**)&hScaleA, sizeScaleA) == ACL_SUCCESS,
            "Failed to allocate the host buffer for scaleA.");
        std::unique_ptr<void, aclError (*)(void*)> hostScaleA(hScaleA, aclrtFreeHost);
        CHECK_COND(
            aclrtMallocHost((void**)&hScaleB, sizeScaleB) == ACL_SUCCESS,
            "Failed to allocate the host buffer for scaleB.");
        std::unique_ptr<void, aclError (*)(void*)> hostScaleB(hScaleB, aclrtFreeHost);
        CHECK_COND(aclrtMallocHost((void**)&hC, sizeC) == ACL_SUCCESS, "Failed to allocate the host buffer for output C.");
        std::unique_ptr<void, aclError (*)(void*)> hostC(hC, aclrtFreeHost);

        ReadFile(inputDir + "/input_a.bin", sizeA, hA, sizeA);
        ReadFile(inputDir + "/input_b.bin", sizeB, hB, sizeB);
        ReadFile(inputDir + "/input_scaleA.bin", sizeScaleA, hScaleA, sizeScaleA);
        ReadFile(inputDir + "/input_scaleB.bin", sizeScaleB, hScaleB, sizeScaleB);

        CHECK_COND(
            aclrtMalloc((void**)&dA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
            "Failed to allocate the device buffer for input A.");
        std::unique_ptr<void, aclError (*)(void*)> deviceA(dA, aclrtFree);
        CHECK_COND(
            aclrtMalloc((void**)&dB, sizeB, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
            "Failed to allocate the device buffer for input B.");
        std::unique_ptr<void, aclError (*)(void*)> deviceB(dB, aclrtFree);
        CHECK_COND(
            aclrtMalloc((void**)&dScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
            "Failed to allocate the device buffer for scaleA.");
        std::unique_ptr<void, aclError (*)(void*)> deviceScaleA(dScaleA, aclrtFree);
        CHECK_COND(
            aclrtMalloc((void**)&dScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
            "Failed to allocate the device buffer for scaleB.");
        std::unique_ptr<void, aclError (*)(void*)> deviceScaleB(dScaleB, aclrtFree);
        CHECK_COND(
            aclrtMalloc((void**)&dC, sizeC, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
            "Failed to allocate the device buffer for output C.");
        std::unique_ptr<void, aclError (*)(void*)> deviceC(dC, aclrtFree);

        CHECK_COND(
            aclrtMemcpyAsync(dA, sizeA, hA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
            "Failed to copy input A from host to device.");
        CHECK_COND(
            aclrtMemcpyAsync(dB, sizeB, hB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
            "Failed to copy input B from host to device.");
        CHECK_COND(
            aclrtMemcpyAsync(dScaleA, sizeScaleA, hScaleA, sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
            "Failed to copy scaleA from host to device.");
        CHECK_COND(
            aclrtMemcpyAsync(dScaleB, sizeScaleB, hScaleB, sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
            "Failed to copy scaleB from host to device.");

        // Launch exactly the number of AICs requested by the tiling result so
        // scheduler-side load balancing and runtime launch geometry match.
        QuantMatmulMxfp4SwatKernel<<<tilingData.usedCoreNum, nullptr, stream>>>(
            dA, dB, dScaleA, dScaleB, dC, tilingData);

        CHECK_COND(
            aclrtMemcpyAsync(hC, sizeC, dC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS,
            "Failed to copy output C from device to host.");
        CHECK_COND(
            aclrtSynchronizeStream(stream) == ACL_SUCCESS,
            "Failed to synchronize the ACL stream after kernel execution.");

        WriteFile(outputDir + "/npu_out.bin", hC, sizeC);
        std::string cmd = "cd \"" + baseDir + "\" && python3 verify_result.py " + std::to_string(m) + " " +
                          std::to_string(n);
        if (std::system(cmd.c_str()) != 0) {
            cleanupAcl();
            return 1;
        }
        cleanupAcl();
        return 0;
    } catch (const std::exception& e) {
        cleanupAcl();
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
