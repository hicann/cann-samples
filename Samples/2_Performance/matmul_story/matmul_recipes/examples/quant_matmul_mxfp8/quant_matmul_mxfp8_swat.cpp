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
 * \file quant_matmul_mxfp8_swat.cpp
 * \brief Sample launcher for the MXFP8 SWAT streaming example.
 */

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "acl/acl.h"
#include "kernel_operator.h"

#include "block/block_scheduler_policy.h"
#include "host_utils/acl_utils.h"
#include "host_utils/common_utils.h"
#include "host_utils/io_utils.h"
#include "kernel_utils/layout_utils.h"
#include "kernel/quant_matmul_mx_kernel_swat.h"
#include "tiling/quant_matmul_mx_tiling_swat.h"
#include "tiling/quant_matmul_tiling_data.h"

template <uint64_t Stages>
__global__ __aicore__ __cube__ void QuantMatmulMxfp8SwatKernel(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dScaleA, GM_ADDR dScaleB, GM_ADDR dC,
    const QuantMatmulTilingData quantMatmulTilingData)
{
    // Keep the sample explicit about the datatype/layout combination that this
    // executable demonstrates so host tiling and kernel traits stay in sync.
    using TypeA = fp8_e4m3fn_t;
    using TypeB = fp8_e4m3fn_t;
    using TypeC = bfloat16_t;

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;

    using BlockScheduler = QuantMatmulMxSwatScheduler<NO_FULL_LOAD_MODE>;
    using DispatchPolicy = QuantMatmulMxMultiBlockWithSwat<NO_FULL_LOAD_MODE, Stages>;
    using ProblemShape = MatmulShape;

    using BlockMmad = Block::BlockMmad<DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC>;
    using QuantMatmulKernelImpl = Kernel::QuantMatmulMxKernelSwat<ProblemShape, BlockMmad, BlockScheduler>;
    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;

    // Translate the serialized host tiling packet once at the launch
    // boundary, then pass only the typed parameter bundles each lower layer
    // actually consumes.
    ProblemShape problemShape{quantMatmulTilingData.m, quantMatmulTilingData.n, quantMatmulTilingData.k, 1UL};
    BlockMmadParams mmadParams{dA, dB, dC, dScaleA, dScaleB};
    L1Params l1Params{
        static_cast<uint64_t>(quantMatmulTilingData.stepK) * quantMatmulTilingData.baseK,
        quantMatmulTilingData.scaleKL1};
    BlockSchedulerParams schedulerParams{
        quantMatmulTilingData.baseM,
        quantMatmulTilingData.baseN,
        quantMatmulTilingData.mTailTile,
        quantMatmulTilingData.nTailTile,
        quantMatmulTilingData.mBaseTailSplitCnt,
        quantMatmulTilingData.nBaseTailSplitCnt,
        quantMatmulTilingData.mTailMain,
        quantMatmulTilingData.nTailMain};
    QBMMTiling qbmmParams{
        quantMatmulTilingData.baseM,
        quantMatmulTilingData.baseN,
        quantMatmulTilingData.baseK,
        quantMatmulTilingData.dbL0c};
    Params params{problemShape, mmadParams, l1Params, schedulerParams, qbmmParams};
    QuantMatmulKernelImpl quantMatmulKernelImpl;
    quantMatmulKernelImpl(params);
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

    try {
        QuantMatmulTilingData tilingData;
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;
        tilingEngine.GetTilingData(m, n, k, tilingData);

        AclRtSession aclSession(deviceId);
        aclSession.Init();
        aclrtStream stream = aclSession.GetStream();

        // MXFP8 stores one element per byte.
        uint64_t sizeA = (m * k) * sizeof(uint8_t);
        uint64_t sizeB = (k * n) * sizeof(uint8_t);
        uint64_t sizeScaleA =
            (m * CeilDiv(k, TILING_MXFP_DIVISOR_SIZE) * TILING_MXFP_MULTI_BASE_SIZE) * sizeof(uint8_t);
        uint64_t sizeScaleB =
            (n * CeilDiv(k, TILING_MXFP_DIVISOR_SIZE) * TILING_MXFP_MULTI_BASE_SIZE) * sizeof(uint8_t);
        uint64_t sizeC = m * n * sizeof(half);

        ExampleIoPaths paths = GetExampleIoPaths();

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

        CHECK_COND(
            aclrtMallocHost((void**)&hA, sizeA) == ACL_SUCCESS, "Failed to allocate the host buffer for input A.");
        std::unique_ptr<void, aclError (*)(void*)> hostA(hA, aclrtFreeHost);
        CHECK_COND(
            aclrtMallocHost((void**)&hB, sizeB) == ACL_SUCCESS, "Failed to allocate the host buffer for input B.");
        std::unique_ptr<void, aclError (*)(void*)> hostB(hB, aclrtFreeHost);
        CHECK_COND(
            aclrtMallocHost((void**)&hScaleA, sizeScaleA) == ACL_SUCCESS,
            "Failed to allocate the host buffer for scaleA.");
        std::unique_ptr<void, aclError (*)(void*)> hostScaleA(hScaleA, aclrtFreeHost);
        CHECK_COND(
            aclrtMallocHost((void**)&hScaleB, sizeScaleB) == ACL_SUCCESS,
            "Failed to allocate the host buffer for scaleB.");
        std::unique_ptr<void, aclError (*)(void*)> hostScaleB(hScaleB, aclrtFreeHost);
        CHECK_COND(
            aclrtMallocHost((void**)&hC, sizeC) == ACL_SUCCESS, "Failed to allocate the host buffer for output C.");
        std::unique_ptr<void, aclError (*)(void*)> hostC(hC, aclrtFreeHost);

        CHECK_COND(ReadExactFile(paths.inputDir + "/input_a.bin", hA, sizeA), "Failed to read input_a.bin.");
        CHECK_COND(ReadExactFile(paths.inputDir + "/input_b.bin", hB, sizeB), "Failed to read input_b.bin.");
        CHECK_COND(
            ReadExactFile(paths.inputDir + "/input_scaleA.bin", hScaleA, sizeScaleA),
            "Failed to read input_scaleA.bin.");
        CHECK_COND(
            ReadExactFile(paths.inputDir + "/input_scaleB.bin", hScaleB, sizeScaleB),
            "Failed to read input_scaleB.bin.");

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
            aclrtMemcpyAsync(dScaleA, sizeScaleA, hScaleA, sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE, stream) ==
                ACL_SUCCESS,
            "Failed to copy scaleA from host to device.");
        CHECK_COND(
            aclrtMemcpyAsync(dScaleB, sizeScaleB, hScaleB, sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE, stream) ==
                ACL_SUCCESS,
            "Failed to copy scaleB from host to device.");

        // `nBufferNum` selects the compile-time pipeline depth: instantiate
        // the 4-stage kernel for four L1 buffers, otherwise use the 2-stage variant.
        if (tilingData.nBufferNum == 4U) {
            QuantMatmulMxfp8SwatKernel<4>
                <<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dScaleA, dScaleB, dC, tilingData);
        } else {
            QuantMatmulMxfp8SwatKernel<2>
                <<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dScaleA, dScaleB, dC, tilingData);
        }

        CHECK_COND(
            aclrtMemcpyAsync(hC, sizeC, dC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS,
            "Failed to copy output C from device to host.");
        CHECK_COND(
            aclrtSynchronizeStream(stream) == ACL_SUCCESS,
            "Failed to synchronize the ACL stream after kernel execution.");

        CHECK_COND(WriteFile(paths.outputDir + "/npu_out.bin", hC, sizeC), "Failed to write npu_out.bin.");
        std::string cmd =
            "cd \"" + paths.baseDir + "\" && python3 verify_result.py " + std::to_string(m) + " " + std::to_string(n);
        if (std::system(cmd.c_str()) != 0) {
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
