/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KV_RMS_NORM_ROPE_CACHE_SAMPLE_COMMON_H_
#define KV_RMS_NORM_ROPE_CACHE_SAMPLE_COMMON_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
#include "platform/platform_ascendc.h"

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

namespace KvRmsNormRopeCacheSample {

using DataType = bfloat16_t;
using HostBf16 = uint16_t;

constexpr int64_t SAMPLE_BATCH = 8;
constexpr int64_t SAMPLE_NUM_HEAD = 1;
constexpr int64_t SAMPLE_SEQ = 128;
constexpr int64_t SAMPLE_CACHE_LENGTH = SAMPLE_SEQ;
constexpr int64_t SAMPLE_DV = 512;
constexpr int64_t SAMPLE_DK = 128;
constexpr int64_t SAMPLE_D = SAMPLE_DV + SAMPLE_DK;
constexpr int64_t SAMPLE_UB_FACTOR = 8;
constexpr float SAMPLE_EPSILON = 1e-5f;
constexpr float BF16_COMPARE_TOL = 6e-2f;
constexpr int32_t MAX_ERROR_ELEM_NUM = 20;
constexpr size_t BF16_BYTES = sizeof(HostBf16);

struct KvRmsNormRopeCacheSampleTilingData {
    int64_t batchSize;
    int64_t numHead;
    int64_t seqLength;
    int64_t cacheLength;
    int64_t dv;
    int64_t dk;
    int64_t blockFactor;
    int64_t ubFactor;
    float epsilon;
    float reciprocal;
};

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
inline void ReadBin(const std::string &filename, std::vector<T> &data)
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
        file.read(reinterpret_cast<char *>(data.data()), elemNum * sizeof(T));
    }
}

inline void CheckAcl(aclError ret, const char *expr, int line)
{
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "ACL error at line " << line << " for " << expr << ": " << ret << std::endl;
    }
}

#define CHECK_ACL(expr) KvRmsNormRopeCacheSample::CheckAcl((expr), #expr, __LINE__)

inline float Bf16ToFloat(HostBf16 value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline int CompareBf16(
    const std::string &name, const HostBf16 *actual, const std::vector<HostBf16> &golden, float tol)
{
    int errorCount = 0;
    float maxDiff = 0.0f;
    size_t maxDiffIndex = 0;
    for (size_t i = 0; i < golden.size(); ++i) {
        float actualValue = Bf16ToFloat(actual[i]);
        float goldenValue = Bf16ToFloat(golden[i]);
        float diff = std::abs(actualValue - goldenValue);
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIndex = i;
        }
        if (diff > tol) {
            if (errorCount < MAX_ERROR_ELEM_NUM) {
                std::cout << name << " mismatch index " << i << ", expected " << goldenValue << ", actual "
                          << actualValue << ", diff " << diff << std::endl;
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

inline int InitAcl(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

inline size_t BuildTiling(KvRmsNormRopeCacheSampleTilingData &tiling)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    int64_t totalRows = SAMPLE_BATCH * SAMPLE_NUM_HEAD * SAMPLE_SEQ;
    int64_t blockFactor = (totalRows + coreNum - 1) / coreNum;
    int64_t blockNum = (totalRows + blockFactor - 1) / blockFactor;

    tiling.batchSize = SAMPLE_BATCH;
    tiling.numHead = SAMPLE_NUM_HEAD;
    tiling.seqLength = SAMPLE_SEQ;
    tiling.cacheLength = SAMPLE_CACHE_LENGTH;
    tiling.dv = SAMPLE_DV;
    tiling.dk = SAMPLE_DK;
    tiling.blockFactor = blockFactor;
    tiling.ubFactor = std::min<int64_t>(SAMPLE_UB_FACTOR, blockFactor);
    tiling.epsilon = SAMPLE_EPSILON;
    tiling.reciprocal = 1.0f / static_cast<float>(SAMPLE_DV);
    return static_cast<size_t>(blockNum);
}

inline std::string FindGenDataScript(const std::string &exeDir)
{
    std::vector<std::string> candidates = {
        exeDir + "/scripts/gen_data.py",
        exeDir + "/../scripts/gen_data.py",
        std::string(SOURCE_DIR) + "/scripts/gen_data.py",
    };
    for (const auto &path : candidates) {
        std::ifstream script(path);
        if (script.is_open()) {
            return path;
        }
    }
    return candidates.back();
}

inline int GenerateData(const std::string &exeDir)
{
    std::ostringstream cmd;
    cmd << "env -u LD_LIBRARY_PATH python3 " << FindGenDataScript(exeDir) << " --batch " << SAMPLE_BATCH << " --seq " << SAMPLE_SEQ
        << " --dv " << SAMPLE_DV << " --dk " << SAMPLE_DK << " --output " << exeDir;
    int ret = std::system(cmd.str().c_str());
    if (ret != 0) {
        std::cerr << "Generate data failed, command: " << cmd.str() << std::endl;
    }
    return ret;
}

inline int RunSample(
    void (*launchKernel)(uint32_t, aclrtStream, DataType *, DataType *, DataType *, DataType *, int64_t *, DataType *,
        DataType *, DataType *, DataType *, KvRmsNormRopeCacheSampleTilingData),
    const std::string &sampleName)
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = InitAcl(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::string exeDir = GetExeDir();
    ret = GenerateData(exeDir);
    CHECK_RET(ret == 0, return ret);

    std::vector<HostBf16> kvData;
    std::vector<HostBf16> gammaData;
    std::vector<HostBf16> cosData;
    std::vector<HostBf16> sinData;
    std::vector<int64_t> indexData;
    std::vector<HostBf16> kCacheInit;
    std::vector<HostBf16> vCacheInit;
    std::vector<HostBf16> goldenKCache;
    std::vector<HostBf16> goldenVCache;
    std::vector<HostBf16> goldenKOut;
    std::vector<HostBf16> goldenVOut;

    try {
        ReadBin(exeDir + "/input/kv.bin", kvData);
        ReadBin(exeDir + "/input/gamma.bin", gammaData);
        ReadBin(exeDir + "/input/cos.bin", cosData);
        ReadBin(exeDir + "/input/sin.bin", sinData);
        ReadBin(exeDir + "/input/index.bin", indexData);
        ReadBin(exeDir + "/input/k_cache.bin", kCacheInit);
        ReadBin(exeDir + "/input/v_cache.bin", vCacheInit);
        ReadBin(exeDir + "/output/k_cache_golden.bin", goldenKCache);
        ReadBin(exeDir + "/output/v_cache_golden.bin", goldenVCache);
        ReadBin(exeDir + "/output/k_out_golden.bin", goldenKOut);
        ReadBin(exeDir + "/output/v_out_golden.bin", goldenVOut);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    size_t kvSize = kvData.size() * BF16_BYTES;
    size_t gammaSize = gammaData.size() * BF16_BYTES;
    size_t cosSize = cosData.size() * BF16_BYTES;
    size_t sinSize = sinData.size() * BF16_BYTES;
    size_t indexSize = indexData.size() * sizeof(int64_t);
    size_t kCacheSize = kCacheInit.size() * BF16_BYTES;
    size_t vCacheSize = vCacheInit.size() * BF16_BYTES;
    size_t kOutSize = goldenKOut.size() * BF16_BYTES;
    size_t vOutSize = goldenVOut.size() * BF16_BYTES;

    DataType *kvDevice = nullptr;
    DataType *gammaDevice = nullptr;
    DataType *cosDevice = nullptr;
    DataType *sinDevice = nullptr;
    int64_t *indexDevice = nullptr;
    DataType *kCacheDevice = nullptr;
    DataType *vCacheDevice = nullptr;
    DataType *kOutDevice = nullptr;
    DataType *vOutDevice = nullptr;
    HostBf16 *kCacheHost = nullptr;
    HostBf16 *vCacheHost = nullptr;
    HostBf16 *kOutHost = nullptr;
    HostBf16 *vOutHost = nullptr;

    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&kvDevice), kvSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&gammaDevice), gammaSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&cosDevice), cosSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&sinDevice), sinSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&indexDevice), indexSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&kCacheDevice), kCacheSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&vCacheDevice), vCacheSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&kOutDevice), kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&vOutDevice), vOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&kCacheHost), kCacheSize));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&vCacheHost), vCacheSize));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&kOutHost), kOutSize));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&vOutHost), vOutSize));

    CHECK_ACL(aclrtMemcpy(kvDevice, kvSize, kvData.data(), kvSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(gammaDevice, gammaSize, gammaData.data(), gammaSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(cosDevice, cosSize, cosData.data(), cosSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(sinDevice, sinSize, sinData.data(), sinSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(indexDevice, indexSize, indexData.data(), indexSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(kCacheDevice, kCacheSize, kCacheInit.data(), kCacheSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(vCacheDevice, vCacheSize, vCacheInit.data(), vCacheSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemset(kOutDevice, kOutSize, 0, kOutSize));
    CHECK_ACL(aclrtMemset(vOutDevice, vOutSize, 0, vOutSize));

    KvRmsNormRopeCacheSampleTilingData tilingData;
    uint32_t blockNum = static_cast<uint32_t>(BuildTiling(tilingData));
    launchKernel(
        blockNum, stream, kvDevice, gammaDevice, cosDevice, sinDevice, indexDevice, kCacheDevice, vCacheDevice,
        kOutDevice, vOutDevice, tilingData);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(kCacheHost, kCacheSize, kCacheDevice, kCacheSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(vCacheHost, vCacheSize, vCacheDevice, vCacheSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(kOutHost, kOutSize, kOutDevice, kOutSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(vOutHost, vOutSize, vOutDevice, vOutSize, ACL_MEMCPY_DEVICE_TO_HOST));

    int errors = 0;
    std::cout << "Run " << sampleName << ", blockNum " << blockNum << ", blockFactor " << tilingData.blockFactor
              << ", ubFactor " << tilingData.ubFactor << std::endl;
    errors += CompareBf16("k_cache", kCacheHost, goldenKCache, BF16_COMPARE_TOL);
    errors += CompareBf16("v_cache", vCacheHost, goldenVCache, BF16_COMPARE_TOL);
    errors += CompareBf16("k_out", kOutHost, goldenKOut, BF16_COMPARE_TOL);
    errors += CompareBf16("v_out", vOutHost, goldenVOut, BF16_COMPARE_TOL);
    std::cout << (errors == 0 ? "PASS" : "FAIL") << std::endl;

    CHECK_ACL(aclrtFree(kvDevice));
    CHECK_ACL(aclrtFree(gammaDevice));
    CHECK_ACL(aclrtFree(cosDevice));
    CHECK_ACL(aclrtFree(sinDevice));
    CHECK_ACL(aclrtFree(indexDevice));
    CHECK_ACL(aclrtFree(kCacheDevice));
    CHECK_ACL(aclrtFree(vCacheDevice));
    CHECK_ACL(aclrtFree(kOutDevice));
    CHECK_ACL(aclrtFree(vOutDevice));
    CHECK_ACL(aclrtFreeHost(kCacheHost));
    CHECK_ACL(aclrtFreeHost(vCacheHost));
    CHECK_ACL(aclrtFreeHost(kOutHost));
    CHECK_ACL(aclrtFreeHost(vOutHost));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return errors == 0 ? 0 : 1;
}

} // namespace KvRmsNormRopeCacheSample

#endif // KV_RMS_NORM_ROPE_CACHE_SAMPLE_COMMON_H_
