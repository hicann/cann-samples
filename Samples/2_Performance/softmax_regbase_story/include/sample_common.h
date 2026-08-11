/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SOFTMAX_REGBASE_SAMPLE_COMMON_H_
#define SOFTMAX_REGBASE_SAMPLE_COMMON_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <libgen.h>
#include <linux/limits.h>
#include <unistd.h>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "kernel_operator.h"

#ifndef SOURCE_DIR
#define SOURCE_DIR "."
#endif

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace SoftmaxRegbaseSample {

constexpr uint32_t TOTAL_M = 256;
constexpr uint32_t TOTAL_N = 2048;
constexpr uint32_t SINGLE_CORE_M = 8;
constexpr uint32_t SINGLE_CORE_N = 2048;
constexpr uint32_t TILE_LEN = 2 * SINGLE_CORE_N;
constexpr uint32_t BLOCKS =
    ((TOTAL_M + SINGLE_CORE_M - 1) / SINGLE_CORE_M) * ((TOTAL_N + SINGLE_CORE_N - 1) / SINGLE_CORE_N);
constexpr float COMPARE_TOL = 1e-3f;
constexpr int32_t MAX_ERROR_ELEM_NUM = 20;

inline std::string GetExeDir()
{
    char path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len != -1) {
        path[len] = '\0';
        return std::string(dirname(path));
    }
    return ".";
}

template <typename T>
inline void ReadBin(const std::string& filename, std::vector<T>& data)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Can not open file: " + filename);
    }
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t elemNum = static_cast<size_t>(fileSize) / sizeof(T);
    data.resize(elemNum);
    if (elemNum > 0) {
        file.read(reinterpret_cast<char*>(data.data()), elemNum * sizeof(T));
    }
}

inline int InitAcl(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

inline std::string FindGenDataScript(const std::string& exeDir)
{
    std::vector<std::string> candidates = {
        exeDir + "/gen_data.py",
        exeDir + "/scripts/gen_data.py",
        std::string(SOURCE_DIR) + "/scripts/gen_data.py",
    };
    for (const auto& path : candidates) {
        std::ifstream script(path);
        if (script.is_open()) {
            return path;
        }
    }
    return candidates.back();
}

inline int GenerateData(const std::string& exeDir)
{
    std::ostringstream cmd;
    cmd << "env -u LD_LIBRARY_PATH python3 " << FindGenDataScript(exeDir) << " --output " << exeDir;
    int ret = std::system(cmd.str().c_str());
    if (ret != 0) {
        std::cerr << "Generate data failed, command: " << cmd.str() << std::endl;
    }
    return ret;
}

inline int CompareFloat(
    const std::string& name, const float* actual, const std::vector<float>& golden, float tol)
{
    int errorCount = 0;
    float maxDiff = 0.0f;
    size_t maxDiffIndex = 0;
    for (size_t i = 0; i < golden.size(); ++i) {
        float diff = std::abs(actual[i] - golden[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIndex = i;
        }
        if (diff > tol) {
            if (errorCount < MAX_ERROR_ELEM_NUM) {
                std::cout << name << " mismatch index " << i << ", expected " << golden[i] << ", actual "
                          << actual[i] << ", diff " << diff << std::endl;
            }
            ++errorCount;
        }
    }
    float precision = golden.empty() ? 100.0f
                                     : static_cast<float>(golden.size() - errorCount) / golden.size() * 100.0f;
    std::cout << name << " precision " << precision << "%, errors " << errorCount << ", max diff " << maxDiff
              << " at " << maxDiffIndex << std::endl;
    return errorCount;
}

template <int kStep>
inline int RunSample(void (*launchKernel)(uint32_t, aclrtStream, uint8_t*, uint8_t*), const std::string& sampleName)
{
    const char* devEnv = std::getenv("SAMPLE_DEVICE_ID");
    int32_t deviceId = devEnv ? std::atoi(devEnv) : 7;
    aclrtStream stream = nullptr;
    auto ret = InitAcl(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::string exeDir = GetExeDir();
    ret = GenerateData(exeDir);
    CHECK_RET(ret == 0, return ret);

    size_t bytes = TOTAL_M * TOTAL_N * sizeof(float);
    std::vector<float> hostIn;
    std::vector<float> golden;
    try {
        ReadBin(exeDir + "/input/input_x.bin", hostIn);
        ReadBin(exeDir + "/output/golden.bin", golden);
    } catch (const std::exception& e) {
        std::cerr << "Read input/golden failed: " << e.what() << std::endl;
        return 1;
    }

    uint8_t *dIn = nullptr, *dOut = nullptr;
    ret = aclrtMalloc(reinterpret_cast<void**>(&dIn), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMalloc(reinterpret_cast<void**>(&dOut), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dIn); return ret);

    ret = aclrtMemcpy(dIn, bytes, hostIn.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dIn); aclrtFree(dOut); return ret);

    launchKernel(BLOCKS, stream, dIn, dOut);
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        aclrtFree(dIn); aclrtFree(dOut);
        return ret;
    }

    std::vector<float> hostOut(TOTAL_M * TOTAL_N);
    ret = aclrtMemcpy(hostOut.data(), bytes, dOut, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dIn); aclrtFree(dOut); return ret);

    int errors = CompareFloat("output", hostOut.data(), golden, COMPARE_TOL);
    std::cout << "[" << sampleName << "] step " << kStep << (errors == 0 ? " PASSED" : " FAILED") << std::endl;

    aclrtFree(dIn);
    aclrtFree(dOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return errors == 0 ? 0 : 1;
}

} // namespace SoftmaxRegbaseSample

#endif // SOFTMAX_REGBASE_SAMPLE_COMMON_H_
