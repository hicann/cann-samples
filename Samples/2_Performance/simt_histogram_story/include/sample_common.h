/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIMT_HISTOGRAM_STORY_SAMPLE_COMMON_H_
#define SIMT_HISTOGRAM_STORY_SAMPLE_COMMON_H_

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_rt.h"

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

namespace SimtHistogramStorySample {

// 输入元素类型：对标 histogram_v2 支持 float / float16 / int32 等，此处以 float 为主
using XType = float;
using OutType = int32_t;

// 数据规模：100 万元素 + 100 个 bins
constexpr int32_t INPUT_ELEMS = 1000000;
constexpr int32_t BINS = 100;
constexpr uint32_t BLOCKS = 4;
constexpr int32_t SIMT_THREADS_PER_BLOCK = 1024;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t MAX_ERROR_ELEM_NUM = 20;

enum class HistogramDataCase {
    UNIFORM = 0,  // 均匀分布
};

using LaunchKernelFunc = void (*)(uint32_t blocks, aclrtStream stream, XType* x, XType* min, XType* max, OutType* y,
                                  int32_t totalLength, int32_t bins);

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

template <typename T>
inline void CheckSize(const std::string& name, const std::vector<T>& data, size_t expected)
{
    if (data.size() != expected) {
        std::ostringstream oss;
        oss << name << " size mismatch, expected " << expected << ", actual " << data.size();
        throw std::runtime_error(oss.str());
    }
}

inline int CompareHistogram(const std::string& name, const OutType* actual, const std::vector<OutType>& golden)
{
    int errorCount = 0;
    for (size_t i = 0; i < golden.size(); ++i) {
        if (actual[i] != golden[i]) {
            if (errorCount < MAX_ERROR_ELEM_NUM) {
                std::cout << name << " mismatch index " << i << ", expected " << golden[i] << ", actual "
                          << actual[i] << std::endl;
            }
            ++errorCount;
        }
    }
    float precision = golden.empty() ? 100.0f
                                     : static_cast<float>(golden.size() - errorCount) / golden.size() * 100.0f;
    std::cout << name << " precision " << precision << "%, errors " << errorCount << std::endl;
    return errorCount;
}

// 释放设备资源
inline void CleanUpAcl(XType* dx, XType* dMin, XType* dMax, OutType* dOut, aclrtStream stream, int32_t deviceId)
{
    aclrtFree(dx);
    aclrtFree(dMin);
    aclrtFree(dMax);
    aclrtFree(dOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

// 加载 golden 数据
inline int LoadHistogramData(std::vector<XType>& x, std::vector<XType>& min, std::vector<XType>& max,
                             std::vector<OutType>& golden)
{
    try {
        ReadBin(DATA_DIR "/input/x.bin", x);
        ReadBin(DATA_DIR "/input/min.bin", min);
        ReadBin(DATA_DIR "/input/max.bin", max);
        ReadBin(DATA_DIR "/output/golden.bin", golden);
        CheckSize("x", x, INPUT_ELEMS);
        CheckSize("min", min, 1);
        CheckSize("max", max, 1);
        CheckSize("golden", golden, BINS);
    } catch (const std::exception& e) {
        std::cerr << "Read input/golden failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

// 分配设备内存并拷贝输入数据
inline aclError AllocAndCopyH2D(const std::vector<XType>& x, const std::vector<XType>& min,
                                const std::vector<XType>& max, size_t xBytes, size_t scalarBytes,
                                size_t outputBytes, XType*& dx, XType*& dMin, XType*& dMax, OutType*& dOut)
{
    aclError ret;
    ret = aclrtMalloc(reinterpret_cast<void**>(&dx), xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMalloc(reinterpret_cast<void**>(&dMin), scalarBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dx); return ret);
    ret = aclrtMalloc(reinterpret_cast<void**>(&dMax), scalarBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dx); aclrtFree(dMin); return ret);
    ret = aclrtMalloc(reinterpret_cast<void**>(&dOut), outputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dx); aclrtFree(dMin); aclrtFree(dMax); return ret);

    std::vector<OutType> zeroOut(BINS, 0);
    ret = aclrtMemcpy(dOut, outputBytes, zeroOut.data(), outputBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(dx, xBytes, x.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(dMin, scalarBytes, min.data(), scalarBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(dMax, scalarBytes, max.data(), scalarBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    return ret;
}

template <int kStep>
inline int RunSample(LaunchKernelFunc launchKernel, const std::string& sampleName, HistogramDataCase dataCase)
{
    (void)dataCase;
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = InitAcl(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<XType> x;
    std::vector<XType> min;
    std::vector<XType> max;
    std::vector<OutType> golden;
    CHECK_RET(LoadHistogramData(x, min, max, golden) == 0,
              aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize(); return 1);

    const size_t xBytes = static_cast<size_t>(INPUT_ELEMS) * sizeof(XType);
    const size_t scalarBytes = sizeof(XType);
    const size_t outputBytes = static_cast<size_t>(BINS) * sizeof(OutType);
    XType* dx = nullptr;
    XType* dMin = nullptr;
    XType* dMax = nullptr;
    OutType* dOut = nullptr;
    ret = AllocAndCopyH2D(x, min, max, xBytes, scalarBytes, outputBytes, dx, dMin, dMax, dOut);
    CHECK_RET(ret == ACL_SUCCESS, CleanUpAcl(dx, dMin, dMax, dOut, stream, deviceId); return ret);

    launchKernel(BLOCKS, stream, dx, dMin, dMax, dOut, INPUT_ELEMS, BINS);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, CleanUpAcl(dx, dMin, dMax, dOut, stream, deviceId); return ret);

    std::vector<OutType> hostOut(BINS);
    ret = aclrtMemcpy(hostOut.data(), outputBytes, dOut, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, CleanUpAcl(dx, dMin, dMax, dOut, stream, deviceId); return ret);

    int errors = CompareHistogram("output", hostOut.data(), golden);
    std::cout << "[" << sampleName << "] step " << kStep << (errors == 0 ? " PASSED" : " FAILED") << std::endl;

    CleanUpAcl(dx, dMin, dMax, dOut, stream, deviceId);
    return errors == 0 ? 0 : 1;
}

} // namespace SimtHistogramStorySample

#endif // SIMT_HISTOGRAM_STORY_SAMPLE_COMMON_H_
