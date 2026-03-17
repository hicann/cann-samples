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
 * \brief
 */
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <unistd.h>

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
#include "kernel/quant_matmul_mx_kernel_swat_impl.h"
#include "policy/dispatch_policy.h"
#include "utils/quant_matmul_tiling_data.h"

// -----------------------------------------------------------------------------
// Device entry
// -----------------------------------------------------------------------------
//
// This sample kernel is intentionally thin:
// 1. Select the concrete data types / layouts / policies.
// 2. Convert runtime tiling data into the templated kernel parameter structure.
// 3. Forward execution to `QuantMatmulMxKernelSwatImpl`.
//
// The actual compute pipeline lives in:
// - `BlockSchedulerQuantMatmulMx`  : maps logical tiles to hardware blocks.
// - `BlockMmadMx`                  : manages L1/L0 movement and MMAD execution.
// - `QuantMatmulMxKernelSwatImpl`  : connects scheduling, address mapping, and compute.
__global__ __aicore__ void QuantMatmulMxfp4Kernel(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dScaleA, GM_ADDR dScaleB, GM_ADDR dBias, GM_ADDR dC,
    const QuantMatmulTilingData quantMatmulTilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    // Matrix element types used by this sample.
    //
    // Conceptually, this is an MXFP4 / fp4 matmul sample.
    // The concrete storage type for A/B is `fp4x2_e2m1_t`, which packs two fp4
    // values into one byte in GM.
    // Accumulation and output both use fp32 here to keep the demo easy to inspect.
    using AType = fp4x2_e2m1_t;
    using BType = fp4x2_e2m1_t;
    using BiasType = float;
    using CType = bfloat16_t;

    // Logical tensor layouts as seen by the matmul template.
    //
    // This sample computes:
    //   C[M, N] = A[M, K] * B[K, N]
    //
    // `layoutB = ColumnMajor` matches the internal expectations of this MX path.
    using layoutA = layout::RowMajor;
    using layoutB = layout::ColumnMajor;
    using layoutC = layout::RowMajor;
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;

    using BlockScheduler = QuantMatmulMxSwatScheduler;
    // Dispatch policy controls how data is staged across blocks.
    //
    // - `QuantMatmulMxMultiBlockWithSwat<>`
    //      Default path. Both A and B are staged tile-by-tile.
    // - `QuantMatmulMxMultiBlockWithSwat<..., A_FULL_LOAD_MODE>`
    //      A stays resident in L1 when the shape is friendly enough.
    //
    // This sample uses the simpler default path because it is easier to learn.
    using DispatchPolicy = QuantMatmulMxMultiBlockWithSwat<>;
    using BlockMmad = Block::BlockMmadMx<
        DispatchPolicy, L1TileShape, L0TileShape, AType, layoutA, BType, layoutB, CType, layoutC, BiasType, layoutC, void>;
    using ProblemShape = MatmulShape;
    using QuantMatmulKernelImpl = Kernel::QuantMatmulMxKernelSwatImpl<ProblemShape, BlockMmad, BlockScheduler>;

    using Params = typename QuantMatmulKernelImpl::Params;

    // `QBMMTiling` contains the "shape of one compute tile":
    // - baseM/baseN/baseK: block-level matmul tile shape
    // - isBias           : whether bias is enabled
    // - dbL0C            : whether L0C ping-pong is enabled
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    QBMMTiling qbmmParams{quantMatmulTilingData.baseM, quantMatmulTilingData.baseN, quantMatmulTilingData.baseK,
                          static_cast<uint32_t>(quantMatmulTilingData.isBias),
                          static_cast<uint32_t>(quantMatmulTilingData.dbL0C)};

    // Runtime parameters consumed by the templated kernel implementation.
    //
    // The nesting here mirrors the kernel structure:
    // - problemShape : global M/N/K
    // - mmadParams   : GM tensor base addresses
    // - l1Params     : how much K is staged in L1 each iteration
    // - schParams    : how tiles are scheduled and how tail tiles are split
    // - qbmmParams   : tile geometry and optional features
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

// Print usage information for the standalone demo binary.
//
// The sample is intentionally command-line driven so that new developers can
// experiment with shape changes without touching the kernel code first.
void printUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName << " m k n" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "Example: " << programName << " 100 50 200" << std::endl;
}

// Parse command-line arguments and validate the shape contract required by this sample.
//
// The checks here are not generic matmul requirements. They are requirements of
// this specific MXFP4 example and its current packing / scale-layout assumptions.
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

    // The GM storage for fp4 values is packed two-per-byte in this sample, so K
    // must be even for the current MXFP4 data layout.
    if (k % 2 != 0) {
        throw std::invalid_argument("ERROR: k must be an even number");
    }

    // Scale tensors are grouped by 32 logical K elements and then further aligned
    // to the MX divisor granularity used by the template implementation.
    if (CeilDiv(k, 32) % 2 != 0) {
        throw std::invalid_argument("ERROR: k should satisfy that CeilDiv(k, 32) is an even number");
    }
}

// Populate a "good default" tiling setup for the sample.
//
// The goal is not to be universally optimal. The goal is to provide a stable,
// readable configuration that exposes the main pieces of the pipeline:
// block tiling, staged K iteration, scale movement, and optional tail handling.
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
    // -------------------------------------------------------------------------
    // 1. Parse the problem shape from the command line.
    // -------------------------------------------------------------------------
    int m, k, n;
    try {
        parseArguments(argc, argv, m, k, n);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // -------------------------------------------------------------------------
    // 2. Initialize the ACL runtime and create a stream.
    //
    // The sample assumes one device and one stream for clarity. More advanced
    // applications may use multiple streams or pre-created runtime contexts.
    // -------------------------------------------------------------------------
    int32_t deviceId = 0;
    aclrtStream stream;
    uint32_t deviceCount;
    CHECK_COND(aclrtGetDeviceCount(&deviceCount) == ACL_SUCCESS, "Failed to get ACLRT devices.");
    CHECK_COND(deviceCount > 0U, "No ACLRT devices found.");
    CHECK_COND(aclInit(nullptr) == ACL_SUCCESS, "aclInit failed.");
    CHECK_COND(aclrtSetDevice(deviceId) == ACL_SUCCESS, "aclrtSetDevice failed.");
    CHECK_COND(aclrtCreateStream(&stream) == ACL_SUCCESS, "aclrtCreateStream failed.");

    // -------------------------------------------------------------------------
    // 3. Declare host-side buffers.
    //
    // Host buffers hold the packed input tensors loaded from disk and receive
    // the output tensor copied back from device memory.
    // -------------------------------------------------------------------------
    uint8_t* hA = nullptr;
    uint8_t* hB = nullptr;
    uint8_t* hScaleA = nullptr;
    uint8_t* hScaleB = nullptr;
    float* hBias = nullptr;
    half* hC = nullptr;

    // Device buffers mirror the host buffers.
    //
    // `GM_ADDR` is the generic "global memory address" type used by the kernel
    // launch interface in Ascend C samples.
    GM_ADDR dA = nullptr;
    GM_ADDR dB = nullptr;
    GM_ADDR dScaleA = nullptr;
    GM_ADDR dScaleB = nullptr;
    GM_ADDR dBias = nullptr;
    GM_ADDR dC = nullptr;

    // -------------------------------------------------------------------------
    // 4. Compute tensor sizes in bytes.
    // -------------------------------------------------------------------------
    //
    // A and B:
    //   this MXFP4 sample stores fp4 values in packed form, with two logical
    //   values per byte, so the logical element count is divided by 2
    //   (rounded up).
    //
    // scaleA and scaleB:
    //   each scale value covers `MXFP_DIVISOR_SIZE` logical K elements and uses
    //   `MXFP_MULTI_BASE_SIZE` bytes in storage.
    //
    // bias:
    //   one fp32 bias per output column N.
    //
    // C:
    //   full bf16 output matrix of shape [M, N].
    size_t sizeA = ((m * k + 1) >> 1) * sizeof(uint8_t);
    size_t sizeB = ((k * n + 1) >> 1) * sizeof(uint8_t);
    size_t sizeScaleA = (m * CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE) * sizeof(uint8_t);
    size_t sizeScaleB = (n * CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE) * sizeof(uint8_t);
    size_t sizeBias = n * sizeof(float);
    size_t sizeC = m * n * sizeof(half);

    // Materialize the default tiling configuration for this problem shape.
    QuantMatmulTilingData quantMatmulTilingData;
    SetTilingData(quantMatmulTilingData, m, n, k);

    // -------------------------------------------------------------------------
    // 5. Allocate pinned host memory.
    //
    // Pinned buffers are used because they are the typical choice for explicit
    // async H2D / D2H copies in standalone performance samples.
    // -------------------------------------------------------------------------
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

    // Load pre-generated test vectors from disk.
    //
    // The sample keeps input generation out of the main executable so that
    // compute code stays easy to follow and data can be reproduced offline.
    ReadFile("./input/input_a.bin", sizeA, hA, sizeA);
    ReadFile("./input/input_b.bin", sizeB, hB, sizeB);
    ReadFile("./input/input_scaleA.bin", sizeScaleA, hScaleA, sizeScaleA);
    ReadFile("./input/input_scaleB.bin", sizeScaleB, hScaleB, sizeScaleB);

    // -------------------------------------------------------------------------
    // 6. Allocate global memory on device.
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // 7. Copy host inputs to device memory.
    //
    // These copies are queued on the same stream that will later launch the
    // kernel, which preserves execution order without extra synchronization.
    // -------------------------------------------------------------------------
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

    // Query the platform object to learn how many AIC cores are available.
    //
    // This sample launches one block per AIC core and lets the scheduler assign
    // multiple tiles to each block when the problem is larger than the machine.
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    CHECK_COND(ascendcPlatform != nullptr, "Get ascendcPlatform failed.");
    uint32_t numBlocks = ascendcPlatform->GetCoreNumAic();

    // -------------------------------------------------------------------------
    // 8. Launch the kernel.
    //
    // The kernel itself is small because most of the interesting logic is
    // encoded in the template stack and the runtime tiling parameters.
    // -------------------------------------------------------------------------
    QuantMatmulMxfp4Kernel<<<numBlocks, nullptr, stream>>>(dA, dB, dScaleA, dScaleB, dBias, dC, quantMatmulTilingData);

    // Queue the output copy after the kernel launch on the same stream.
    CHECK_COND(
        aclrtMemcpyAsync(hC, sizeC, dC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");

    // -------------------------------------------------------------------------
    // 9. Synchronize, dump the output, and tear everything down.
    // -------------------------------------------------------------------------
    CHECK_COND(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "aclrtSynchronizeStream failed.");

    WriteFile("./output/npu_out.bin", hC, sizeC);
    std::filesystem::path exe_path = std::filesystem::canonical("/proc/self/exe");
    std::filesystem::path exe_dir = exe_path.parent_path();
    std::filesystem::path script =
        exe_dir.parent_path().parent_path().parent_path() / "common/golden/quant_matmul_mxfp4/verify_result.py";
    std::string cmd = "python3 \"" + script.string() + "\" " + std::to_string(m) + " " + std::to_string(n);
    system(cmd.c_str());
    // `unique_ptr` takes care of freeing host/device buffers.
    // The runtime objects still need explicit destruction/finalization.
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
