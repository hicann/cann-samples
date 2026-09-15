/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_STORY_SAMPLE_COMMON_H_
#define BLOCK_ATTN_RES_STORY_SAMPLE_COMMON_H_

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <libgen.h>
#include <linux/limits.h>
#include <unistd.h>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "tiling/block_attn_res_prepare_tiling.h"
#include "tiling/block_attn_res_update_tiling.h"
#include "sample_process.h"

#ifndef SOURCE_DIR
#define SOURCE_DIR "."
#endif

namespace BlockAttnResStory {

inline void CheckAcl(aclError ret, const char* operation)
{
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed, ACL error=" + std::to_string(ret));
    }
}

inline uint64_t CheckedMul(uint64_t lhs, uint64_t rhs, const char* name)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        throw std::overflow_error(std::string(name) + " size overflows uint64");
    }
    return lhs * rhs;
}

inline size_t CheckedSize(uint64_t elements, size_t elementBytes, const char* name)
{
    const uint64_t bytes = CheckedMul(elements, elementBytes, name);
    if (bytes > std::numeric_limits<size_t>::max()) {
        throw std::overflow_error(std::string(name) + " size overflows size_t");
    }
    return static_cast<size_t>(bytes);
}

inline std::string GetExeDir()
{
    char path[PATH_MAX];
    const ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len < 0)
        return ".";
    path[len] = '\0';
    return std::string(dirname(path));
}

inline std::string FindScript(const std::string& exeDir, const std::string& name)
{
    const std::vector<std::string> candidates = {
        exeDir + "/" + name, exeDir + "/scripts/" + name, std::string(SOURCE_DIR) + "/scripts/" + name};
    for (const auto& path : candidates) {
        std::ifstream file(path);
        if (file.is_open())
            return path;
    }
    return candidates.back();
}

inline void ReadExactFile(const std::string& path, void* data, size_t bytes)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("cannot open " + path);
    if (static_cast<uint64_t>(file.tellg()) != bytes)
        throw std::runtime_error("unexpected size: " + path);
    file.seekg(0, std::ios::beg);
    if (bytes != 0 && !file.read(reinterpret_cast<char*>(data), bytes))
        throw std::runtime_error("cannot read " + path);
}

inline void WriteFile(const std::string& path, const void* data, size_t bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open() || !file.write(reinterpret_cast<const char*>(data), bytes))
        throw std::runtime_error("cannot write " + path);
}

inline uint64_t ParseU64(const char* text, const char* name)
{
    const std::string input = text == nullptr ? "" : text;
    if (input.empty() || input.front() == '-')
        throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
    size_t parsed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(input, &parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
    }
    if (parsed != input.size())
        throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
    return static_cast<uint64_t>(value);
}

inline uint32_t ParseU32(const char* text, const char* name)
{
    const uint64_t value = ParseU64(text, name);
    if (value > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument(std::string(name) + " is too large");
    return static_cast<uint32_t>(value);
}

inline float ParseFloat(const char* text, const char* name)
{
    const std::string input = text == nullptr ? "" : text;
    size_t parsed = 0;
    float value = 0.0F;
    try {
        value = std::stof(input, &parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " must be a floating-point number");
    }
    if (parsed != input.size())
        throw std::invalid_argument(std::string(name) + " must be a floating-point number");
    return value;
}

inline void PrintUsage(const char* program)
{
    std::cerr << "Usage: " << program << " T N S D [options]\n"
              << "  --valid-blocks <n>       runtime valid N, default N\n"
              << "  --slot <n>               slot consumed by Update, default 0\n"
              << "  --template auto|vector|mix  Prepare/E2E only\n"
              << "  --eps <value>            default 1e-6\n"
              << "  --warmup <n>             default 1\n"
              << "  --repeat <n>             default 1\n"
              << "  --compare isclose|stat_rel_err  default isclose\n";
}

inline void ParseOption(ShapeConfig& shape, const std::string& option, const char* value)
{
    if (option == "--valid-blocks")
        shape.validBlocks = ParseU64(value, "valid-blocks");
    else if (option == "--slot")
        shape.slot = ParseU32(value, "slot");
    else if (option == "--eps")
        shape.eps = ParseFloat(value, "eps");
    else if (option == "--warmup")
        shape.warmup = ParseU32(value, "warmup");
    else if (option == "--repeat")
        shape.repeat = ParseU32(value, "repeat");
    else if (option == "--compare") {
        shape.compareMethod = value;
        if (shape.compareMethod != "isclose" && shape.compareMethod != "stat_rel_err")
            throw std::invalid_argument("compare must be isclose or stat_rel_err");
    } else if (option == "--template") {
        const std::string mode = value;
        if (mode == "auto")
            shape.prepareTemplate = PrepareTemplate::AUTO;
        else if (mode == "vector")
            shape.prepareTemplate = PrepareTemplate::VECTOR;
        else if (mode == "mix")
            shape.prepareTemplate = PrepareTemplate::MIX;
        else
            throw std::invalid_argument("template must be auto, vector or mix");
    } else
        throw std::invalid_argument("unknown option: " + option);
}

inline ShapeConfig ParseArguments(int argc, char** argv)
{
    if (argc < 5) {
        PrintUsage(argv[0]);
        throw std::invalid_argument("T N S D are required");
    }
    ShapeConfig shape;
    shape.t = ParseU32(argv[1], "T");
    shape.n = ParseU32(argv[2], "N");
    shape.s = ParseU32(argv[3], "S");
    shape.d = ParseU32(argv[4], "D");
    shape.validBlocks = shape.n;
    for (int index = 5; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 >= argc)
            throw std::invalid_argument("missing value for " + option);
        const char* value = argv[++index];
        ParseOption(shape, option, value);
    }
    if (!std::isfinite(shape.eps) || shape.eps <= 0 || shape.repeat == 0) {
        throw std::invalid_argument("eps must be finite positive and repeat must be positive");
    }
    ValidatePrepareShape(shape);
    return shape;
}

class AclContext {
public:
    AclContext()
    {
        const char* deviceEnv = std::getenv("SAMPLE_DEVICE_ID");
        const uint32_t deviceId = deviceEnv == nullptr ? 0 : ParseU32(deviceEnv, "SAMPLE_DEVICE_ID");
        if (deviceId > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
            throw std::invalid_argument("SAMPLE_DEVICE_ID is too large");
        deviceId_ = static_cast<int32_t>(deviceId);
        CheckAcl(aclInit(nullptr), "aclInit");
        initialized_ = true;
        try {
            CheckAcl(aclrtSetDevice(deviceId_), "aclrtSetDevice");
            deviceSet_ = true;
            CheckAcl(aclrtCreateStream(&stream_), "aclrtCreateStream");
        } catch (...) {
            Cleanup();
            throw;
        }
    }
    ~AclContext()
    {
        Cleanup();
    }
    AclContext(const AclContext&) = delete;
    AclContext& operator=(const AclContext&) = delete;
    aclrtStream Stream() const
    {
        return stream_;
    }

private:
    void Cleanup() noexcept
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (initialized_) {
            aclFinalize();
            initialized_ = false;
        }
    }

    int32_t deviceId_ = 0;
    aclrtStream stream_ = nullptr;
    bool initialized_ = false;
    bool deviceSet_ = false;
};

} // namespace BlockAttnResStory

#endif // BLOCK_ATTN_RES_STORY_SAMPLE_COMMON_H_
