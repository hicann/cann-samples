/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_RELU_CV_STORY_HOST_RUNNER_H
#define MATMUL_RELU_CV_STORY_HOST_RUNNER_H

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <libgen.h>
#include <linux/limits.h>

#include "acl/acl.h"
#include "common_utils.h"
#include "kernel_common.h"

inline std::string GetExeDir()
{
    char path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len != -1) {
        path[len] = 0;
        return std::string(dirname(path));
    }
    return ".";
}

inline void EnsureDir(const std::string& path)
{
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        mkdir(path.c_str(), 0755);
    }
}

struct HostBuffers {
    uint8_t* aHost = nullptr;
    uint8_t* bHost = nullptr;
    uint8_t* cHost = nullptr;
    uint8_t* aDevice = nullptr;
    uint8_t* bDevice = nullptr;
    uint8_t* cDevice = nullptr;
    size_t aFileSize = 0;
    size_t bFileSize = 0;
    size_t cFileSize = 0;
    aclrtStream stream = nullptr;
    int32_t deviceId = 0;
    std::string workDir;
};

inline void ReleaseHostBuffers(HostBuffers& buf)
{
    if (buf.aDevice != nullptr) {
        aclrtFree(buf.aDevice);
        buf.aDevice = nullptr;
    }
    if (buf.bDevice != nullptr) {
        aclrtFree(buf.bDevice);
        buf.bDevice = nullptr;
    }
    if (buf.cDevice != nullptr) {
        aclrtFree(buf.cDevice);
        buf.cDevice = nullptr;
    }
    if (buf.aHost != nullptr) {
        aclrtFreeHost(buf.aHost);
        buf.aHost = nullptr;
    }
    if (buf.bHost != nullptr) {
        aclrtFreeHost(buf.bHost);
        buf.bHost = nullptr;
    }
    if (buf.cHost != nullptr) {
        aclrtFreeHost(buf.cHost);
        buf.cHost = nullptr;
    }
    if (buf.stream != nullptr) {
        aclrtDestroyStream(buf.stream);
        buf.stream = nullptr;
    }
}

inline bool FailInitAndCleanup(HostBuffers& buf, bool deviceReady)
{
    ReleaseHostBuffers(buf);
    if (deviceReady) {
        aclrtResetDevice(buf.deviceId);
    }
    aclFinalize();
    return false;
}

inline bool CheckAcl(aclError ret, const char* apiName)
{
    if (ret == ACL_SUCCESS) {
        return true;
    }
    ERROR_LOG("%s failed, ret=%d", apiName, ret);
    return false;
}

inline bool InitAclRuntime(HostBuffers& buf)
{
    if (!CheckAcl(aclInit(nullptr), "aclInit")) {
        return false;
    }
    if (!CheckAcl(aclrtSetDevice(buf.deviceId), "aclrtSetDevice")) {
        aclFinalize();
        return false;
    }
    if (!CheckAcl(aclrtCreateStream(&buf.stream), "aclrtCreateStream")) {
        return FailInitAndCleanup(buf, true);
    }
    return true;
}

inline bool AllocOneHost(uint8_t*& ptr, size_t bytes, const char* name, HostBuffers& buf)
{
    if (CheckAcl(aclrtMallocHost(reinterpret_cast<void**>(&ptr), bytes), name)) {
        return true;
    }
    return FailInitAndCleanup(buf, true);
}

inline bool AllocOneDevice(uint8_t*& ptr, size_t bytes, const char* name, HostBuffers& buf)
{
    if (CheckAcl(aclrtMalloc(reinterpret_cast<void**>(&ptr), bytes, ACL_MEM_MALLOC_HUGE_FIRST), name)) {
        return true;
    }
    return FailInitAndCleanup(buf, true);
}

inline bool AllocHostAndDeviceBuffers(HostBuffers& buf)
{
    if (!AllocOneHost(buf.aHost, buf.aFileSize, "aclrtMallocHost(A)", buf) ||
        !AllocOneHost(buf.bHost, buf.bFileSize, "aclrtMallocHost(B)", buf) ||
        !AllocOneHost(buf.cHost, buf.cFileSize, "aclrtMallocHost(C)", buf)) {
        return false;
    }
    if (!AllocOneDevice(buf.aDevice, buf.aFileSize, "aclrtMalloc(A)", buf) ||
        !AllocOneDevice(buf.bDevice, buf.bFileSize, "aclrtMalloc(B)", buf) ||
        !AllocOneDevice(buf.cDevice, buf.cFileSize, "aclrtMalloc(C)", buf)) {
        return false;
    }
    return true;
}

inline bool CopyHostToDevice(uint8_t* dst, const uint8_t* src, size_t bytes, const char* name, HostBuffers& buf)
{
    if (CheckAcl(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE), name)) {
        return true;
    }
    return FailInitAndCleanup(buf, true);
}

inline bool LoadInputsAndCopyH2D(HostBuffers& buf)
{
    size_t aReadSize = 0;
    size_t bReadSize = 0;
    std::string aPath = buf.workDir + "/input/x1_gm.bin";
    std::string bPath = buf.workDir + "/input/x2_gm.bin";
    if (!ReadFile(aPath, aReadSize, buf.aHost, buf.aFileSize) || aReadSize != buf.aFileSize) {
        ERROR_LOG("Read A failed or size mismatch: %s read=%zu expected=%zu",
            aPath.c_str(), aReadSize, buf.aFileSize);
        return FailInitAndCleanup(buf, true);
    }
    if (!ReadFile(bPath, bReadSize, buf.bHost, buf.bFileSize) || bReadSize != buf.bFileSize) {
        ERROR_LOG("Read B failed or size mismatch: %s read=%zu expected=%zu",
            bPath.c_str(), bReadSize, buf.bFileSize);
        return FailInitAndCleanup(buf, true);
    }
    if (!CopyHostToDevice(buf.aDevice, buf.aHost, buf.aFileSize, "aclrtMemcpy(A H2D)", buf) ||
        !CopyHostToDevice(buf.bDevice, buf.bHost, buf.bFileSize, "aclrtMemcpy(B H2D)", buf)) {
        return false;
    }
    return true;
}

inline bool InitHostAndLoadInputs(HostBuffers& buf)
{
    buf.workDir = GetExeDir();
    buf.aFileSize = MATMUL_RELU_M * MATMUL_RELU_K * sizeof(uint16_t);
    buf.bFileSize = MATMUL_RELU_K * MATMUL_RELU_N * sizeof(uint16_t);
    buf.cFileSize = MATMUL_RELU_M * MATMUL_RELU_N * sizeof(float);

    if (!InitAclRuntime(buf) || !AllocHostAndDeviceBuffers(buf) || !LoadInputsAndCopyH2D(buf)) {
        return false;
    }
    return true;
}

inline std::string OutputBinPath(const HostBuffers& buf)
{
    return buf.workDir + "/output/output.bin";
}

// Remove stale output so a failed run cannot leave a previous pass artifact.
inline bool PrepareFreshOutput(HostBuffers& buf)
{
    EnsureDir(buf.workDir + "/output");
    const std::string outPath = OutputBinPath(buf);
    if (unlink(outPath.c_str()) != 0 && errno != ENOENT) {
        ERROR_LOG("failed to remove stale output: %s", outPath.c_str());
        return false;
    }
    return true;
}

inline bool SyncStream(HostBuffers& buf, const char* stage)
{
    return CheckAcl(aclrtSynchronizeStream(buf.stream), stage);
}

inline bool FinalizeRuntime(HostBuffers& buf)
{
    ReleaseHostBuffers(buf);
    if (!CheckAcl(aclrtResetDevice(buf.deviceId), "aclrtResetDevice")) {
        aclFinalize();
        return false;
    }
    if (!CheckAcl(aclFinalize(), "aclFinalize")) {
        return false;
    }
    return true;
}

// D2H + write output.bin. On any failure, remove partial output and still try to release ACL.
inline bool DumpOutputAndFinalize(HostBuffers& buf)
{
    bool ok = true;
    if (!CheckAcl(aclrtMemcpy(buf.cHost, buf.cFileSize, buf.cDevice, buf.cFileSize, ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy(C D2H)")) {
        ok = false;
    }
    const std::string outPath = OutputBinPath(buf);
    if (ok) {
        EnsureDir(buf.workDir + "/output");
        if (!WriteFile(outPath, buf.cHost, buf.cFileSize)) {
            ERROR_LOG("WriteFile output failed: %s", outPath.c_str());
            ok = false;
        }
    }
    if (!ok) {
        (void)unlink(outPath.c_str());
    }
    if (!FinalizeRuntime(buf)) {
        ok = false;
    }
    return ok;
}

#endif // MATMUL_RELU_CV_STORY_HOST_RUNNER_H
