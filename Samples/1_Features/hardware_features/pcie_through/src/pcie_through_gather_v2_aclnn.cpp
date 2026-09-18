/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pcie_through_gather_v2_aclnn.cpp
 * \brief 通过 aclnn 接口直调 GatherV2 算子，演示 PCIe through 特性的完整用法
 *
 * PCIe through 特性允许 Device 直接访问 Host 内存，避免显式 H2D/D2H 拷贝。
 * 本样例的完整流程如下：
 *   1. 通过 aclrtMallocHost 在 Host 侧分配内存
 *   2. 通过 aclrtHostRegister(ACL_HOST_REGISTER_MAPPED) 将 Host 内存映射到 Device 地址空间
 *   3. 将映射后的 Device 指针直接作为 aclTensor 的数据地址传给算子
 *   4. 算子内部自动检测 PCIe through 场景，选择 PCIe 安全的 SIMD tiling 路径
 *   5. 计算完成后结果直接写入 Host 内存，无需 D2H 拷贝
 */

#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_gather_v2.h"

#define CHECK_ACL(call)                                                                                   \
    do {                                                                                                   \
        aclError err = (call);                                                                             \
        if (err != ACL_SUCCESS) {                                                                          \
            std::fprintf(stderr, "ACL error %d at %s:%d\n", static_cast<int>(err), __FILE__, __LINE__);   \
            return 1;                                                                                      \
        }                                                                                                  \
    } while (0)

namespace {

constexpr int64_t DIM = 0;
constexpr int64_t M = 4;
constexpr int64_t N = 8;
constexpr int64_t IDX_NUM = 3;

/**
 * Host 内存管理 RAII 封装：自动在析构时 unregister + free
 */
struct HostMemGuard {
    void *ptr = nullptr;           // Host 侧内存地址
    bool registered = false;       // 是否已通过 aclrtHostRegister 注册映射
    void *devPtr = nullptr;        // 映射后的 Device 侧地址
    size_t bytes = 0;
};

/**
 * 判断当前环境是否为 PCIe through 场景
 * 需满足三个条件：
 * 1. 环境变量 OP_PCIE_THROUGH_ACCESS_HOST_MEM_CHECK_ENABLE=1
 * 2. Host-Device 连接类型为 PCIe（ACL_HOST_DEVICE_CONNECT_TYPE_PCIE）
 * 3. Host地址到Device地址是否映射成功
 */
bool IsPcieThrough(uint32_t deviceId, void *deviceAddr)
{
    bool isPcieThrough = false;
    const char *pcieThroughEnv = std::getenv("OP_PCIE_THROUGH_ACCESS_HOST_MEM_CHECK_ENABLE");
    int64_t pcieThroughValue = (pcieThroughEnv != nullptr && std::string(pcieThroughEnv) == "1") ? 1 : 0;
    if (!pcieThroughValue) {
        return false;
    }

    int64_t hdConnectType = -1;
    aclError ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_HD_CONNECT_TYPE, &hdConnectType);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtGetDeviceInfo failed, ret=%d\n", ret);
        return false;
    }

    if (hdConnectType == ACL_HOST_DEVICE_CONNECT_TYPE_PCIE) {
        isPcieThrough = deviceAddr != nullptr;
    }

    return isPcieThrough;
}

void PrintResult(const float *hostPtr, int64_t count)
{
    std::printf("GatherV2 result (PCIe through):\n");
    for (int64_t i = 0; i < count; ++i) {
        std::printf("  out[%lld] = %.1f\n", static_cast<long long>(i), hostPtr[i]);
    }
}

bool CheckResult(const float *out, const float *x, const int64_t *index, int64_t idxNum, int64_t n)
{
    for (int64_t i = 0; i < idxNum; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float expected = x[index[i] * n + j];
            if (out[i * n + j] != expected) {
                std::fprintf(stderr, "Mismatch at out[%lld][%lld]: got %.1f, expected %.1f\n",
                             static_cast<long long>(i), static_cast<long long>(j),
                             out[i * n + j], expected);
                return false;
            }
        }
    }
    return true;
}

void CleanupGatherV2(void *workspace, aclOpExecutor *executor,
                     aclTensor *xTensor, aclTensor *idxTensor, aclTensor *outTensor)
{
    if (workspace != nullptr) { aclrtFree(workspace); }
    aclDestroyTensor(xTensor);
    aclDestroyTensor(idxTensor);
    aclDestroyTensor(outTensor);
}

int32_t PrepareHostMemory(HostMemGuard &xMem, HostMemGuard &idxMem, HostMemGuard &outMem)
{
    int64_t xTotal = M * N;
    int64_t outTotal = IDX_NUM * N;
    size_t xBytes = static_cast<size_t>(xTotal) * sizeof(float);
    size_t idxBytes = static_cast<size_t>(IDX_NUM) * sizeof(int64_t);
    size_t outBytes = static_cast<size_t>(outTotal) * sizeof(float);

    CHECK_ACL(aclrtMallocHost(&xMem.ptr, xBytes));
    xMem.bytes = xBytes;
    CHECK_ACL(aclrtMallocHost(&idxMem.ptr, idxBytes));
    idxMem.bytes = idxBytes;
    CHECK_ACL(aclrtMallocHost(&outMem.ptr, outBytes));
    outMem.bytes = outBytes;

    std::printf("[MallocHost] xHost=%p idxHost=%p outHost=%p\n", xMem.ptr, idxMem.ptr, outMem.ptr);

    for (int64_t i = 0; i < xTotal; ++i) {
        (static_cast<float *>(xMem.ptr))[i] = static_cast<float>(i);
    }
    int64_t idxData[IDX_NUM] = {3, 0, 2};
    std::copy_n(idxData, IDX_NUM, static_cast<int64_t *>(idxMem.ptr));
    std::fill_n(static_cast<float *>(outMem.ptr), outTotal, 0.0f);
    return 0;
}

int32_t RegisterHostMemory(HostMemGuard &xMem, HostMemGuard &idxMem, HostMemGuard &outMem)
{
    CHECK_ACL(aclrtHostRegister(xMem.ptr, xMem.bytes, ACL_HOST_REGISTER_MAPPED, &xMem.devPtr));
    xMem.registered = true;
    CHECK_ACL(aclrtHostRegister(idxMem.ptr, idxMem.bytes, ACL_HOST_REGISTER_MAPPED, &idxMem.devPtr));
    idxMem.registered = true;
    CHECK_ACL(aclrtHostRegister(outMem.ptr, outMem.bytes, ACL_HOST_REGISTER_MAPPED, &outMem.devPtr));
    outMem.registered = true;

    std::printf("[HostRegister] xDev=%p idxDev=%p outDev=%p\n", xMem.devPtr, idxMem.devPtr, outMem.devPtr);
    return 0;
}

int32_t ExecuteGatherV2(HostMemGuard &xMem, HostMemGuard &idxMem, HostMemGuard &outMem,
                        aclrtStream stream)
{
    std::printf("[PCIeThrough] Detected PCIe through scenario, operator will use SIMD tiling path\n");
    std::vector<int64_t> xShape = {M, N};
    std::vector<int64_t> idxShape = {IDX_NUM};
    std::vector<int64_t> outShape = {IDX_NUM, N};

    aclTensor *xTensor = aclCreateTensor(xShape.data(), xShape.size(), ACL_FLOAT, nullptr, 0,
                                         ACL_FORMAT_ND, nullptr, 0, xMem.devPtr);
    aclTensor *idxTensor = aclCreateTensor(idxShape.data(), idxShape.size(), ACL_INT64, nullptr, 0,
                                           ACL_FORMAT_ND, nullptr, 0, idxMem.devPtr);
    aclTensor *outTensor = aclCreateTensor(outShape.data(), outShape.size(), ACL_FLOAT, nullptr, 0,
                                           ACL_FORMAT_ND, nullptr, 0, outMem.devPtr);
    if (xTensor == nullptr || idxTensor == nullptr || outTensor == nullptr) {
        std::fprintf(stderr, "aclCreateTensor failed\n");
        if (xTensor != nullptr) { aclDestroyTensor(xTensor); }
        if (idxTensor != nullptr) { aclDestroyTensor(idxTensor); }
        if (outTensor != nullptr) { aclDestroyTensor(outTensor); }
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus status = aclnnGatherV2GetWorkspaceSize(xTensor, DIM, idxTensor, outTensor,
                                                        &workspaceSize, &executor);
    if (status != 0) {
        std::fprintf(stderr, "aclnnGatherV2GetWorkspaceSize failed, status=%d\n", status);
        aclDestroyTensor(xTensor);
        aclDestroyTensor(idxTensor);
        aclDestroyTensor(outTensor);
        return 1;
    }
    std::printf("[Tiling] workspaceSize=%llu\n", static_cast<unsigned long long>(workspaceSize));

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    status = aclnnGatherV2(workspace, workspaceSize, executor, stream);
    if (status != 0) {
        std::fprintf(stderr, "aclnnGatherV2 failed, status=%d\n", status);
        CleanupGatherV2(workspace, executor, xTensor, idxTensor, outTensor);
        return 1;
    }

    CHECK_ACL(aclrtSynchronizeStream(stream));
    std::printf("[Execute] GatherV2 completed\n");

    CleanupGatherV2(workspace, executor, xTensor, idxTensor, outTensor);
    return 0;
}

int32_t VerifyAndPrint(HostMemGuard &xMem, HostMemGuard &idxMem, HostMemGuard &outMem)
{
    int64_t outTotal = IDX_NUM * N;

    PrintResult(static_cast<float *>(outMem.ptr), outTotal);

    bool ok = CheckResult(static_cast<float *>(outMem.ptr), static_cast<float *>(xMem.ptr),
                          static_cast<int64_t *>(idxMem.ptr), IDX_NUM, N);
    if (ok) {
        std::printf("[Verify] PASSED: output matches expected values\n");
    } else {
        std::fprintf(stderr, "[Verify] FAILED: output does not match\n");
    }
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    std::printf("[Init] ACL initialized, device=%d\n", deviceId);

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    HostMemGuard xMem;
    HostMemGuard idxMem;
    HostMemGuard outMem;

    if (PrepareHostMemory(xMem, idxMem, outMem) != 0) {
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    if (RegisterHostMemory(xMem, idxMem, outMem) != 0) {
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    if (!IsPcieThrough(static_cast<uint32_t>(deviceId), xMem.devPtr)) {
        std::printf("[PCIeThrough] Non-PCIe scenario\n");
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 0;
    }

    int32_t ret = ExecuteGatherV2(xMem, idxMem, outMem, stream);

    if (ret == 0) {
        ret = VerifyAndPrint(xMem, idxMem, outMem);
    }

    aclrtDestroyStream(stream);
    if (outMem.registered) { aclrtHostUnregister(outMem.ptr); }
    if (idxMem.registered) { aclrtHostUnregister(idxMem.ptr); }
    if (xMem.registered) { aclrtHostUnregister(xMem.ptr); }
    if (outMem.ptr != nullptr) { aclrtFreeHost(outMem.ptr); }
    if (idxMem.ptr != nullptr) { aclrtFreeHost(idxMem.ptr); }
    if (xMem.ptr != nullptr) { aclrtFreeHost(xMem.ptr); }
    aclrtResetDevice(deviceId);
    aclFinalize();

    return ret;
}
