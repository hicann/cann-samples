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
 * \file weight_quant_grouped_matmul_mxfp8fp4_split_m.cpp
 * \brief Minimal launcher for the grouped mxfp8fp4 split-M recipe.
 */

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "acl/acl.h"

#include "host_utils/acl_utils.h"
#include "host_utils/common_utils.h"
#include "host_utils/io_utils.h"

#include "prologue/block_prologue.h"
#include "block/block_mmad.h"
#include "block/weight_quant_grouped_matmul_mxfp8fp4_block_scheduler_split_m.h"
#include "policy/dispatch_policy.h"
#include "kernel/weight_quant_grouped_matmul_mxfp8fp4_kernel_split_m.h"
#include "utils/layout_struct.h"

#include "tiling/weight_quant_grouped_matmul_mxfp8fp4_tiling.h"

#include "include/tensor.h"

namespace detail {

__global__ __aicore__ __mix__(1, 2) void WeightQuantGroupedMatmulKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR xScale, GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR y,
    const WeightQuantGroupedMatmulTilingData tilingData)
{
    using AType = float8_e4m3_t;
    using BType = fp4x2_e2m1_t;
    using ScaleBType = float8_e8m0_t;
    using ScaleAType = float8_e8m0_t;
    using CType = bfloat16_t;

    using DispatchPolicy = KernelMixDynamicKL1NTailResplit;
    using LayoutA = typename AscendC::Te::NDLayoutFormat<AType>;
    using LayoutB = AscendC::Te::Weight4BitLayout<BType>;
    using LayoutC = typename AscendC::Te::NDLayoutFormat<AType>;
    using LayoutScaleA = typename AscendC::Te::ScaleANDLayoutFormat<fp8_e8m0_t>;
    using LayoutScaleB = typename AscendC::Te::ScaleBDNLayoutFormat<fp8_e8m0_t>;
    using ProblemShape = decltype(AscendC::Te::MakeShape(0UL, 0UL, 0UL, 0UL));
    using BlockScheduler = Block::GroupedMatmulSchedulerNResplit<decltype(AscendC::Te::MakeShape(0UL, 0UL, 0UL))>;
    using BlockMmad = Block::BlockMmad<
        DispatchPolicy, AscendC::Std::tuple<AType, ScaleAType>, AscendC::Std::tuple<LayoutA, LayoutScaleA>,
        AscendC::Std::tuple<BType, ScaleBType>, AscendC::Std::tuple<LayoutB, LayoutScaleB>, CType, LayoutC>;
    using BlockEpilogue = void;
    using BlockPrologue = Prologue::BlockPrologue<DispatchPolicy, AType, BType>;
    using KernelImpl = Kernel::WeightQuantGroupedMatmulMxfp8fp4Kernel<
        ProblemShape, BlockMmad, BlockScheduler, BlockEpilogue, BlockPrologue>;

    typename BlockMmad::Params mmadParams{
        reinterpret_cast<__gm__ AType*>(x), reinterpret_cast<__gm__ ScaleAType*>(xScale),
        reinterpret_cast<__gm__ ScaleBType*>(weightScale), reinterpret_cast<__gm__ CType*>(y)};
    typename BlockScheduler::Params schedulerParams{
        tilingData.mainBlockCount,
        tilingData.mainBlockSize,
        tilingData.firstTailBlockCount,
        tilingData.firstTailBlockSize,
        tilingData.secondTailBlockCount,
        tilingData.secondTailBlockSize,
        tilingData.coreNum,
        tilingData.cubeNumBlocksN,
        tilingData.baseM,
        tilingData.nSize};
    typename BlockPrologue::Params prologueParams{reinterpret_cast<__gm__ BType*>(weight)};
    typename KernelImpl::Params params = {
        {0UL, tilingData.kSize, tilingData.nSize, tilingData.groupNum},
        mmadParams,
        schedulerParams,
        prologueParams,
        groupList};
    KernelImpl kernelImpl;
    kernelImpl(params);
}
} // namespace detail

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

    try {
        constexpr int32_t deviceId = 0;
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

        WeightQuantGroupedMatmulMxfp8fp4Tiling tiler;
        WeightQuantGroupedMatmulTilingData tilingData;
        tiler.GetTilingData(
            static_cast<uint32_t>(groupMList.size()), static_cast<uint64_t>(n), static_cast<uint64_t>(k), tilingData);

        uint64_t k1 = CeilDiv<uint64_t>(k, 32UL);
        uint64_t n1 = CeilDiv<uint64_t>(n, 16UL);
        size_t xSize = (m * k) * sizeof(uint8_t);
        size_t weightSize = (groupNum * k1 * n1 * 16UL * 16UL) * sizeof(uint8_t);
        uint64_t scaleK = CeilDiv<uint64_t>(k, 64UL) * 2UL;
        size_t xScaleSize = m * scaleK * sizeof(uint8_t);
        size_t weightScaleSize = groupNum * n * scaleK * sizeof(uint8_t);
        size_t ySize = m * n * sizeof(uint16_t);

        std::vector<uint8_t> hostX(xSize, 0);
        std::vector<uint8_t> hostWeight(weightSize, 0);
        std::vector<uint8_t> hostXScale(xScaleSize, 0);
        std::vector<uint8_t> hostWeightScale(weightScaleSize, 0);
        std::vector<uint16_t> hostY(m * n, 0);

        size_t xFileSize = xSize;
        size_t weightFileSize = weightSize;
        size_t xScaleFileSize = xScaleSize;
        size_t weightScaleFileSize = weightScaleSize;

        CHECK_COND(ReadFile(inputDir + "/input_a.bin", xFileSize, hostX.data(), xSize), "Failed to read input_a.bin.");
        CHECK_COND(xFileSize == xSize, "input_a.bin size does not match the expected tensor size.");
        CHECK_COND(
            ReadFile(inputDir + "/input_b.bin", weightFileSize, hostWeight.data(), weightSize),
            "Failed to read input_b.bin.");
        CHECK_COND(weightFileSize == weightSize, "input_b.bin size does not match the expected tensor size.");
        CHECK_COND(
            ReadFile(inputDir + "/input_scaleA.bin", xScaleFileSize, hostXScale.data(), xScaleSize),
            "Failed to read input_scaleA.bin.");
        CHECK_COND(xScaleFileSize == xScaleSize, "input_scaleA.bin size does not match the expected tensor size.");

        CHECK_COND(
            ReadFile(inputDir + "/input_scaleB.bin", weightScaleFileSize, hostWeightScale.data(), weightScaleSize),
            "Failed to read input_scaleB.bin.");
        CHECK_COND(
            weightScaleFileSize == weightScaleSize, "input_scaleB.bin size does not match the expected tensor size.");

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

            CHECK_COND(
                aclrtMalloc(reinterpret_cast<void**>(&dX), xSize, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
                "Failed to allocate device X.");
            dBuffers.Push(dX);
            CHECK_COND(
                aclrtMalloc(reinterpret_cast<void**>(&dWeight), weightSize, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
                "Failed to allocate device weight.");
            dBuffers.Push(dWeight);
            CHECK_COND(
                aclrtMalloc(reinterpret_cast<void**>(&dXScale), xScaleSize, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
                "Failed to allocate device xScale.");
            dBuffers.Push(dXScale);
            CHECK_COND(
                aclrtMalloc(reinterpret_cast<void**>(&dWeightScale), weightScaleSize, ACL_MEM_MALLOC_HUGE_ONLY) ==
                    ACL_SUCCESS,
                "Failed to allocate device weightScale.");
            dBuffers.Push(dWeightScale);
            CHECK_COND(
                aclrtMalloc(reinterpret_cast<void**>(&dGroupList), groupListBytes, ACL_MEM_MALLOC_HUGE_ONLY) ==
                    ACL_SUCCESS,
                "Failed to allocate device groupList.");
            dBuffers.Push(dGroupList);
            CHECK_COND(
                aclrtMalloc(reinterpret_cast<void**>(&dY), ySize, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
                "Failed to allocate device Y.");
            dBuffers.Push(dY);

            aclrtStream stream = aclSession.GetStream();
            CHECK_COND(
                aclrtMemcpyAsync(dX, xSize, hostX.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
                "Failed to copy X to device.");
            CHECK_COND(
                aclrtMemcpyAsync(
                    dWeight, weightSize, hostWeight.data(), weightSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) ==
                    ACL_SUCCESS,
                "Failed to copy weight to device.");
            CHECK_COND(
                aclrtMemcpyAsync(
                    dXScale, xScaleSize, hostXScale.data(), xScaleSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) ==
                    ACL_SUCCESS,
                "Failed to copy xScale to device.");
            CHECK_COND(
                aclrtMemcpyAsync(
                    dWeightScale, weightScaleSize, hostWeightScale.data(), weightScaleSize, ACL_MEMCPY_HOST_TO_DEVICE,
                    stream) == ACL_SUCCESS,
                "Failed to copy weightScale to device.");
            CHECK_COND(
                aclrtMemcpyAsync(
                    dGroupList, groupListBytes, groupListHost.data(), groupListBytes, ACL_MEMCPY_HOST_TO_DEVICE,
                    stream) == ACL_SUCCESS,
                "Failed to copy groupList to device.");

            // Kernel stub does not write y, initialize it to keep output deterministic.
            CHECK_COND(aclrtMemset(dY, ySize, 0, ySize) == ACL_SUCCESS, "Failed to memset Y.");

            uint32_t coreNum = tilingData.coreNum;
            detail::WeightQuantGroupedMatmulKernel<<<coreNum, nullptr, stream>>>(
                dX, dWeight, dXScale, dWeightScale, dGroupList, dY, tilingData);

            CHECK_COND(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "Failed to synchronize stream.");
            CHECK_COND(
                aclrtMemcpy(hostY.data(), ySize, dY, ySize, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
                "Failed to copy Y back to host.");
            CHECK_COND(
                WriteFile(outputDir + "/output_npu.bin", hostY.data(), ySize), "Failed to write output_npu.bin.");
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
