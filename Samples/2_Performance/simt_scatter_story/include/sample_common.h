/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIMT_SCATTER_STORY_SAMPLE_COMMON_H_
#define SIMT_SCATTER_STORY_SAMPLE_COMMON_H_

#include <algorithm>
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

namespace SimtScatterStorySample {

using DataType = int32_t;
using IndexType = int32_t;

constexpr int32_t DST_ROWS = 4096;
constexpr int32_t INNER_DIM = 8;
constexpr int32_t UNIQUE_UPDATES = 4096;
constexpr int32_t CONFLICT_UPDATES = 8192;
constexpr int32_t OUTPUT_ELEMS = DST_ROWS * INNER_DIM;
constexpr uint32_t BLOCKS = 4;
constexpr int32_t SIMT_THREAD_NUM = 2048;
constexpr int32_t SIMT_X_THREAD_NUM = 256;
constexpr int32_t SIMT_Y_THREAD_NUM = 8;
constexpr int32_t MAX_ERROR_ELEM_NUM = 20;

enum class ScatterDataCase {
    UNIQUE = 0,
    CONFLICT = 1,
};

using LaunchKernelFunc = void (*)(uint32_t blocks, aclrtStream stream, IndexType* indices, DataType* updates,
                                  DataType* y, int32_t updateRows, int32_t dstRows, int32_t innerDim);

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

inline const char* CasePrefix(ScatterDataCase dataCase)
{
    return dataCase == ScatterDataCase::UNIQUE ? "unique" : "conflict";
}

inline int32_t CaseUpdateRows(ScatterDataCase dataCase)
{
    return dataCase == ScatterDataCase::UNIQUE ? UNIQUE_UPDATES : CONFLICT_UPDATES;
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

inline int CompareInt(const std::string& name, const DataType* actual, const std::vector<DataType>& golden)
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

inline int PrepareInputs(ScatterDataCase dataCase, int32_t& deviceId, aclrtStream& stream, IndexType*& dIndices,
    DataType*& dUpdates, DataType*& dOut, int32_t& updateRows, std::vector<DataType>& golden)
{
    deviceId = 0;
    stream = nullptr;
    auto ret = InitAcl(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::string exeDir = GetExeDir();
    ret = GenerateData(exeDir);
    CHECK_RET(ret == 0, return ret);

    const std::string prefix = CasePrefix(dataCase);
    updateRows = CaseUpdateRows(dataCase);
    std::vector<DataType> base;
    std::vector<IndexType> indices;
    std::vector<DataType> updates;
    try {
        ReadBin(exeDir + "/input/base.bin", base);
        ReadBin(exeDir + "/input/" + prefix + "_indices.bin", indices);
        ReadBin(exeDir + "/input/" + prefix + "_updates.bin", updates);
        ReadBin(exeDir + "/output/" + prefix + "_golden.bin", golden);
        CheckSize("base", base, OUTPUT_ELEMS);
        CheckSize("indices", indices, static_cast<size_t>(updateRows));
        CheckSize("updates", updates, static_cast<size_t>(updateRows) * INNER_DIM);
        CheckSize("golden", golden, OUTPUT_ELEMS);
    } catch (const std::exception& e) {
        std::cerr << "Read input/golden failed: " << e.what() << std::endl;
        return 1;
    }

    const size_t indicesBytes = static_cast<size_t>(updateRows) * sizeof(IndexType);
    const size_t updatesBytes = static_cast<size_t>(updateRows) * INNER_DIM * sizeof(DataType);
    const size_t outputBytes = static_cast<size_t>(OUTPUT_ELEMS) * sizeof(DataType);

    dIndices = nullptr;
    dUpdates = nullptr;
    dOut = nullptr;
    ret = aclrtMalloc(reinterpret_cast<void**>(&dIndices), indicesBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMalloc(reinterpret_cast<void**>(&dUpdates), updatesBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dIndices); return ret);
    ret = aclrtMalloc(reinterpret_cast<void**>(&dOut), outputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dIndices); aclrtFree(dUpdates); return ret);

    ret = aclrtMemcpy(dIndices, indicesBytes, indices.data(), indicesBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dIndices); aclrtFree(dUpdates); aclrtFree(dOut); return ret);
    ret = aclrtMemcpy(dUpdates, updatesBytes, updates.data(), updatesBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dIndices); aclrtFree(dUpdates); aclrtFree(dOut); return ret);
    ret = aclrtMemcpy(dOut, outputBytes, base.data(), outputBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dIndices); aclrtFree(dUpdates); aclrtFree(dOut); return ret);
    return ACL_SUCCESS;
}

template <int kStep>
inline int RunSample(LaunchKernelFunc launchKernel, const std::string& sampleName, ScatterDataCase dataCase)
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    IndexType* dIndices = nullptr;
    DataType* dUpdates = nullptr;
    DataType* dOut = nullptr;
    int32_t updateRows = 0;
    std::vector<DataType> golden;
    auto ret = PrepareInputs(dataCase, deviceId, stream, dIndices, dUpdates, dOut, updateRows, golden);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    launchKernel(BLOCKS, stream, dIndices, dUpdates, dOut, updateRows, DST_ROWS, INNER_DIM);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dIndices); aclrtFree(dUpdates); aclrtFree(dOut); return ret);

    std::vector<DataType> hostOut(OUTPUT_ELEMS);
    const size_t outputBytes = static_cast<size_t>(OUTPUT_ELEMS) * sizeof(DataType);
    ret = aclrtMemcpy(hostOut.data(), outputBytes, dOut, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dIndices); aclrtFree(dUpdates); aclrtFree(dOut); return ret);

    int errors = CompareInt("output", hostOut.data(), golden);
    std::cout << "[" << sampleName << "] step " << kStep << (errors == 0 ? " PASSED" : " FAILED") << std::endl;

    aclrtFree(dIndices);
    aclrtFree(dUpdates);
    aclrtFree(dOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return errors == 0 ? 0 : 1;
}

} // namespace SimtScatterStorySample

#endif // SIMT_SCATTER_STORY_SAMPLE_COMMON_H_
