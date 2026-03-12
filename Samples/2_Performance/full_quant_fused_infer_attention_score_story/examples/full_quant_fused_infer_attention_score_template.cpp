/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

#include <iostream>
#include <cstdlib>
#include <memory>

#define FIA_ENABLE_MLA
#include "common_utils.h"
#include "io_utils.h"
#include "flash_attention_score_tiling_regbase.h"
#include "fia_entry.h"

__global__ __aicore__ void FiaKernelFullQuant(
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR keyAntiquantScale,
        GM_ADDR valueAntiquantScale, GM_ADDR dequantScaleQuery, GM_ADDR attentionOut,
        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    FlashAttentionEntry(
        query, key, value,
        keyAntiquantScale, valueAntiquantScale, dequantScaleQuery,  attentionOut,
        workspace, tiling);
    return;
}

int main(int argc, char* argv[])
{
    std::cerr << "Start fused_infer_attention_score demo." << std::endl;
    // -------------------------------------------------------------------------
    // 1. Set the problem shape.
    // -------------------------------------------------------------------------
    uint32_t batchSize = 1;
    uint32_t numHeadsQ = 1;
    uint32_t numHeadsKV = 1;
    uint64_t seqLengthsQ = 8192;
    uint64_t seqLengthsKV = 8192;
    uint32_t headDim = 128;

    // -------------------------------------------------------------------------
    // 2. Initialize the ACL runtime and create a stream.
    //
    // The sample assumes one device and one stream for clarity. More advanced
    // applications may use multiple streams or pre-created runtime contexts.
    // -------------------------------------------------------------------------
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
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
    uint8_t* queryHost = nullptr;
    uint8_t* keyHost = nullptr;
    uint8_t* valueHost = nullptr;
    uint8_t* keyAntiquantScaleHost = nullptr;
    uint8_t* valueAntiquantScaleHost = nullptr;
    uint8_t* dequantScaleQueryHost = nullptr;
    uint8_t* outputHost = nullptr;

    // -------------------------------------------------------------------------
    // 4. Declare device-side buffers.
    //
    // Device buffers mirror the host buffers.
    // `GM_ADDR` is the generic "global memory address" type used by the kernel
    // launch interface in Ascend C samples.
    // -------------------------------------------------------------------------
    GM_ADDR queryDevice = nullptr;
    GM_ADDR keyDevice = nullptr;
    GM_ADDR valueDevice = nullptr;
    GM_ADDR keyAntiquantScaleDevice = nullptr;
    GM_ADDR valueAntiquantScaleDevice = nullptr;
    GM_ADDR dequantScaleQueryDevice = nullptr;
    GM_ADDR outputDevice = nullptr;
    GM_ADDR workspace = nullptr;
    GM_ADDR tilingDataDevice = nullptr;

    // -------------------------------------------------------------------------
    // 4. Compute tensor sizes in bytes.
    // -------------------------------------------------------------------------
    //
    // query:
    //   float8_e4m3fn input tensor of shape [batchSize, numHeadsQ, seqLengthsQ, headDim].
    //
    // key and value:
    //   float8_e4m3fn input tensor of shape [batchSize, numHeadsKV, seqLengthsKV, headDim].
    //
    // dequantScaleQuery:
    //   float32 input tensor shape [batchSize, numHeadsQ, seqLengthsQ / 128, 1].
    //
    // keyAntiquantScale and valueAntiquantScale:
    //   float32 input tensor shape [batchSize, numHeadsKV, seqLengthsKV / 256, 1].
    size_t querySize = (batchSize * numHeadsQ * seqLengthsQ * headDim) * sizeof(uint8_t);
    size_t keySize = (batchSize * numHeadsKV * seqLengthsKV * headDim) * sizeof(uint8_t);
    size_t valueSize = (batchSize * numHeadsKV * seqLengthsKV * headDim) * sizeof(uint8_t);
    size_t queryQuantScaleSize = (batchSize * numHeadsQ * (seqLengthsQ / 128) * 1) * sizeof(float);
    size_t keyAntiquantScaleSize = (batchSize * numHeadsKV * (seqLengthsKV / 256) * 1) * sizeof(float);
    size_t valueAntiquantScaleSize = (batchSize * numHeadsKV * (seqLengthsKV / 256) * 1) * sizeof(float);
    size_t outputSize = (batchSize * numHeadsQ * seqLengthsQ * headDim) * sizeof(uint16_t);
    size_t workspaceSize = 200 * 2048 * 1024 * sizeof(uint8_t);
    size_t tilingDataSize = sizeof(optiling::FlashAttentionScoreSimplifiedTilingData);

    // -------------------------------------------------------------------------
    // 4. Materialize the default tiling configuration for this problem shape.
    // -------------------------------------------------------------------------
    optiling::FlashAttentionScoreSimplifiedTilingData tilingData;
    SetTilingData(tilingData);

    // -------------------------------------------------------------------------
    // 5. Allocate pinned host memory.
    //
    // Pinned buffers are used because they are the typical choice for explicit
    // async H2D / D2H copies in standalone performance samples.
    // -------------------------------------------------------------------------
    CHECK_COND(aclrtMallocHost((void**)&queryHost, querySize) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostQ(queryHost, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&keyHost, keySize) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostK(keyHost, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&valueHost, valueSize) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostV(valueHost, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&keyAntiquantScaleHost, keyAntiquantScaleSize) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostKScale(keyAntiquantScaleHost, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&valueAntiquantScaleHost, valueAntiquantScaleSize) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostVScale(valueAntiquantScaleHost, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&dequantScaleQueryHost, queryQuantScaleSize) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostQScale(dequantScaleQueryHost, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&outputHost, outputSize) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostO(outputHost, aclrtFreeHost);

    // Load pre-generated test tensors from disk.
    //
    // The sample keeps input generation out of the main executable so that
    // compute code stays easy to follow and data can be reproduced offline.
    ReadFile("./input/input_0.bin", querySize, queryHost, querySize);
    ReadFile("./input/input_1.bin", keySize, keyHost, keySize);
    ReadFile("./input/input_2.bin", valueSize, valueHost, valueSize);
    ReadFile("./input/input_15.bin", keyAntiquantScaleSize, keyAntiquantScaleHost, keyAntiquantScaleSize);
    ReadFile("./input/input_17.bin", valueAntiquantScaleSize, valueAntiquantScaleHost, valueAntiquantScaleSize);
    ReadFile("./input/input_27.bin", queryQuantScaleSize, dequantScaleQueryHost, queryQuantScaleSize);

    // -------------------------------------------------------------------------
    // 6. Allocate global memory on device.
    // -------------------------------------------------------------------------
    CHECK_COND(aclrtMalloc((void**)&queryDevice, querySize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceQ(queryDevice, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&keyDevice, keySize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceK(keyDevice, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&valueDevice, valueSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceV(valueDevice, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&keyAntiquantScaleDevice, keyAntiquantScaleSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceKScale(keyAntiquantScaleDevice, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&valueAntiquantScaleDevice, valueAntiquantScaleSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceVScale(valueAntiquantScaleDevice, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&dequantScaleQueryDevice, queryQuantScaleSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceQScale(dequantScaleQueryDevice, aclrtFree);

    CHECK_COND(aclrtMalloc((void**)&outputDevice, outputSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceO(outputDevice, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceWS(workspace, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&tilingDataDevice, tilingDataSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceTD(tilingDataDevice, aclrtFree);

    // -------------------------------------------------------------------------
    // 7. Copy host inputs to device memory.
    //
    // These copies are queued on the same stream that will later launch the
    // kernel, which preserves execution order without extra synchronization.
    // -------------------------------------------------------------------------
    CHECK_COND(
        aclrtMemcpyAsync(queryDevice, querySize, queryHost, querySize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(keyDevice, keySize, keyHost, keySize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(valueDevice, valueSize, valueHost, valueSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(keyAntiquantScaleDevice, keyAntiquantScaleSize, keyAntiquantScaleHost, keyAntiquantScaleSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(valueAntiquantScaleDevice, valueAntiquantScaleSize, valueAntiquantScaleHost, valueAntiquantScaleSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(dequantScaleQueryDevice, queryQuantScaleSize, dequantScaleQueryHost, queryQuantScaleSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(tilingDataDevice, tilingDataSize, &tilingData, tilingDataSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");

    // Query the platform object to learn how many AIC cores are available.
    //
    // This sample launches one block per AIC core and lets the scheduler assign
    // multiple tiles to each block when the problem is larger than the machine.
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    CHECK_COND(ascendcPlatform != nullptr, "Get ascendcPlatform failed.");
    uint32_t blockDimToBeSet = ascendcPlatform->CalcTschBlockDim(ascendcPlatform->GetCoreNumAiv(),
                    ascendcPlatform->GetCoreNumAic(), ascendcPlatform->GetCoreNumAiv());

    // -------------------------------------------------------------------------
    // 8. Launch the kernel.
    //
    // The kernel itself is small because most of the interesting logic is
    // encoded in the template stack and the runtime tiling parameters.
    // -------------------------------------------------------------------------
    constexpr uint8_t inOutLayoutType = 0;
    constexpr bool hasAttenMask = false;
    FiaKernelFullQuant<<<blockDimToBeSet, nullptr, stream>>>(
        queryDevice, keyDevice, valueDevice,
        keyAntiquantScaleDevice, valueAntiquantScaleDevice, dequantScaleQueryDevice,
        outputDevice,
        workspace,
        tilingDataDevice
    );
    
    // Queue the output copy after the kernel launch on the same stream.
    CHECK_COND(
        aclrtMemcpyAsync(outputHost, outputSize, outputDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");

    // -------------------------------------------------------------------------
    // 9. Synchronize, dump the output, and tear everything down.
    // -------------------------------------------------------------------------
    CHECK_COND(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "aclrtSynchronizeStream failed.");
    WriteFile("./output/npu_out.bin", outputHost, outputSize);

    // `unique_ptr` takes care of freeing host/device buffers.
    // The runtime objects still need explicit destruction/finalization.
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

