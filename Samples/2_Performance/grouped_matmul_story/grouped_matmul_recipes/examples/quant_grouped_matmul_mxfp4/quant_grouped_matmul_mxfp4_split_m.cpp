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
 * \file quant_grouped_matmul_mxfp4_split_m.cpp
 * \brief Minimal launcher for the grouped MXFP4 split-M recipe.
 */

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel_operator.h"

#include "host_utils/acl_utils.h"
#include "host_utils/common_utils.h"
#include "host_utils/io_utils.h"
#include "kernel/quant_grouped_matmul_mxfp4_kernel_split_m.h"
#include "policy/dispatch_policy.h"
#include "tiling/quant_grouped_matmul_mxfp4_tiling_data.h"
#include "tiling/quant_grouped_matmul_mxfp4_tiling_split_m.h"
#include "utils/grouped_matmul_constant.h"

// mxfp4: only for cube core
__global__ __aicore__ __cube__ void QuantGroupedMatmulMxfp4SplitMKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR xScale, GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR y,
    const QuantGroupedMatmulMxfp4TilingData tilingData)
{
    using AType = fp4x2_e2m1_t;
    using BType = fp4x2_e2m1_t;
    using CType = bfloat16_t;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;

    using ProblemShape = MatmulShape;
    using DispatchPolicy = QuantMatmulMxMultiBlockMmad;
    using BlockMmad = Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC>;
    using BlockScheduler = Block::BlockSchedulerGmmAswtWithTailSplit<ProblemShape, false, true>;
    using GroupedKernel = Kernel::QuantGroupedMatmulMxfp4KernelSplitM<ProblemShape, BlockMmad, BlockScheduler>;
    using Params = typename GroupedKernel::Params;

    Params params = {
        {static_cast<int64_t>(tilingData.maxM), static_cast<int64_t>(tilingData.n), static_cast<int64_t>(tilingData.k), 1},
        {x, weight, groupList, xScale, weightScale, y},
        &tilingData,
    };
    GroupedKernel kernel;
    kernel(params);
}

int main(int argc, char* argv[])
{
    GroupedMatmulMxfp4Args args{};
    try {
        args = ParseArguments(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }

    constexpr int32_t deviceId = 0;

    try {
        std::string baseDir = GetExecutableDir();
        CHECK_COND(chdir(baseDir.c_str()) == 0, "Failed to switch to the executable directory.");
        uint64_t groupNum = args.groupNum;
        uint64_t m = args.m;
        uint64_t k = args.k;
        uint64_t n = args.n;
        const size_t groupListBytes = args.groupListBytes;
        std::string inputDir = "./input";
        std::string outputDir = "./output";
        std::vector<int64_t> groupListHost(static_cast<size_t>(groupNum), 0);

        size_t groupListFileSize = groupListBytes;
        CHECK_COND(
            ReadFile(inputDir + "/input_groupList.bin", groupListFileSize, groupListHost.data(), groupListBytes),
            "Failed to read input_groupList.bin.");
        CHECK_COND(
            groupListFileSize == groupListBytes, "input_groupList.bin size does not match the expected tensor size.");

        std::vector<uint32_t> groupMList = ParseGroupList(groupListHost);
        uint64_t sumGroupM = std::accumulate(groupMList.begin(), groupMList.end(), 0ULL);
        CHECK_COND(sumGroupM <= m, "m must be greater than or equal to sum(group_m_list).");

        QuantGroupedMatmulMxfp4TilingData tilingData;
        QuantGroupedMatmulMxfp4TilingSplitM tiler;
        tiler.GetTilingData(static_cast<uint32_t>(groupMList.size()), static_cast<uint32_t>(m),
                            static_cast<uint32_t>(n), static_cast<uint32_t>(k), tilingData);

        uint64_t scaleK =
            CeilDiv<uint64_t>(k, GroupedMatmulRecipe::MX_DIVISOR_SIZE) * GroupedMatmulRecipe::MX_MULTI_SIZE;
        // Tensor and GM sizes follow the declared M budget (m), not sum(group_m_list), so buffers stay valid when
        // some groups contribute zero rows.
        size_t xByteSize = (m * k / 2UL) * sizeof(uint8_t);
        size_t weightByteSize = (static_cast<uint64_t>(groupMList.size()) * n * k / 2UL) * sizeof(uint8_t);
        size_t xScaleByteSize = m * scaleK * sizeof(uint8_t);
        size_t weightScaleByteSize = static_cast<uint64_t>(groupMList.size()) * n * scaleK * sizeof(uint8_t);
        size_t yByteSize = m * n * sizeof(uint16_t);

        std::vector<uint8_t> hostX(xByteSize, 0);
        std::vector<uint8_t> hostWeight(weightByteSize, 0);
        std::vector<uint8_t> hostXScale(xScaleByteSize, 0);
        std::vector<uint8_t> hostWeightScale(weightScaleByteSize, 0);
        std::vector<uint8_t> hostY(yByteSize, 0);
        size_t xFileSize = xByteSize;
        size_t weightFileSize = weightByteSize;
        size_t xScaleFileSize = xScaleByteSize;
        size_t weightScaleFileSize = weightScaleByteSize;

        CHECK_COND(
            ReadFile(inputDir + "/input_a.bin", xFileSize, hostX.data(), xByteSize),
            "Failed to read input_a.bin.");
        CHECK_COND(xFileSize == xByteSize, "input_a.bin size does not match the expected tensor size.");
        CHECK_COND(
            ReadFile(inputDir + "/input_b.bin", weightFileSize, hostWeight.data(), weightByteSize),
            "Failed to read input_b.bin.");
        CHECK_COND(weightFileSize == weightByteSize, "input_b.bin size does not match the expected tensor size.");
        CHECK_COND(
            ReadFile(inputDir + "/input_scaleA.bin", xScaleFileSize, hostXScale.data(), xScaleByteSize),
            "Failed to read input_scaleA.bin.");
        CHECK_COND(xScaleFileSize == xScaleByteSize, "input_scaleA.bin size does not match the expected tensor size.");
        CHECK_COND(
            ReadFile(inputDir + "/input_scaleB.bin", weightScaleFileSize, hostWeightScale.data(), weightScaleByteSize),
            "Failed to read input_scaleB.bin.");
        CHECK_COND(
            weightScaleFileSize == weightScaleByteSize, "input_scaleB.bin size does not match the expected tensor size.");

        AclRtSession aclSession(deviceId);
        aclSession.Init();

        {
            AclDeviceBuffers dBuffers;
            GM_ADDR dX = nullptr;
            GM_ADDR dWeight = nullptr;
            GM_ADDR dXScale = nullptr;
            GM_ADDR dWeightScale = nullptr;
            GM_ADDR dGroupList = nullptr;
            GM_ADDR dY = nullptr;
            CHECK_COND(aclrtMalloc(reinterpret_cast<void**>(&dX), xByteSize, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
                       "Failed to allocate device X.");
            dBuffers.Push(dX);
            CHECK_COND(
                aclrtMalloc(reinterpret_cast<void**>(&dWeight), weightByteSize, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
                "Failed to allocate device weight.");
            dBuffers.Push(dWeight);
            CHECK_COND(
                aclrtMalloc(reinterpret_cast<void**>(&dXScale), xScaleByteSize, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
                "Failed to allocate device xScale.");
            dBuffers.Push(dXScale);
            CHECK_COND(
                aclrtMalloc(reinterpret_cast<void**>(&dWeightScale), weightScaleByteSize, ACL_MEM_MALLOC_HUGE_ONLY) ==
                    ACL_SUCCESS,
                "Failed to allocate device weightScale.");
            dBuffers.Push(dWeightScale);
            CHECK_COND(
                aclrtMalloc(reinterpret_cast<void**>(&dGroupList), groupListBytes, ACL_MEM_MALLOC_HUGE_ONLY) ==
                    ACL_SUCCESS,
                "Failed to allocate device groupList.");
            dBuffers.Push(dGroupList);
            CHECK_COND(aclrtMalloc(reinterpret_cast<void**>(&dY), yByteSize, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
                       "Failed to allocate device Y.");
            dBuffers.Push(dY);

            aclrtStream stream = aclSession.GetStream();
            CHECK_COND(
                aclrtMemcpyAsync(dX, xByteSize, hostX.data(), xByteSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
                "Failed to copy X to device.");
            CHECK_COND(
                aclrtMemcpyAsync(dWeight, weightByteSize, hostWeight.data(), weightByteSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) ==
                    ACL_SUCCESS,
                "Failed to copy weight to device.");
            CHECK_COND(
                aclrtMemcpyAsync(dXScale, xScaleByteSize, hostXScale.data(), xScaleByteSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) ==
                    ACL_SUCCESS,
                "Failed to copy xScale to device.");
            CHECK_COND(
                aclrtMemcpyAsync(
                    dWeightScale, weightScaleByteSize, hostWeightScale.data(), weightScaleByteSize, ACL_MEMCPY_HOST_TO_DEVICE,
                    stream) == ACL_SUCCESS,
                "Failed to copy weightScale to device.");
            CHECK_COND(
                aclrtMemcpyAsync(
                    dGroupList, groupListBytes, groupListHost.data(), groupListBytes, ACL_MEMCPY_HOST_TO_DEVICE, stream) ==
                    ACL_SUCCESS,
                "Failed to copy groupList to device.");

            QuantGroupedMatmulMxfp4SplitMKernel<<<tilingData.usedCoreNum, nullptr, stream>>>(
                dX, dWeight, dXScale, dWeightScale, dGroupList, dY, tilingData);
            CHECK_COND(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "Failed to synchronize stream.");
            CHECK_COND(
                aclrtMemcpy(hostY.data(), yByteSize, dY, yByteSize, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
                "Failed to copy Y back to host.");
            CHECK_COND(WriteFile(outputDir + "/npu_out.bin", hostY.data(), yByteSize), "Failed to write npu_out.bin.");
        }
        std::string cmd = "cd \"" + baseDir + "\" && python3 verify_result.py " + std::to_string(groupNum) + " " +
                          std::to_string(m) + " " + std::to_string(k) + " " + std::to_string(n);
        if (std::system(cmd.c_str()) != 0) {
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
