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
 * \file common_utils.h
 * \brief
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gmm {
// Quantized MX element layouts for non-type template parameters.
enum class DataType {
    DT_FLOAT4_E2M1,
    DT_FLOAT8_E4M3FN,
};
} // namespace gmm

#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)
#define CHECK_COND(cond, msg)                                                                                  \
    do {                                                                                                       \
        if (!(cond)) {                                                                                         \
            throw std::runtime_error(                                                                          \
                std::string("Error: ") + msg + "\nFile: " + __FILE__ + "\nLine: " + std::to_string(__LINE__)); \
        }                                                                                                      \
    } while (0)

template <typename T>
inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
inline T Align(T a, T b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
inline T FloorAlign(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return a / b * b;
}

template <gmm::DataType dataType, typename T>
constexpr T GetShapeWithDataType(T size)
{
    if constexpr (dataType == gmm::DataType::DT_FLOAT4_E2M1) {
        return size << 1;
    } else {
        return size;
    }
}

template <gmm::DataType dataType, typename T>
constexpr T GetSizeWithDataType(T shape)
{
    if constexpr (dataType == gmm::DataType::DT_FLOAT4_E2M1) {
        return (shape + 1) >> 1;
    } else {
        return shape;
    }
}

struct GroupedMatmulMxfp4Args {
    uint64_t groupNum{0};
    uint64_t m{0};
    uint64_t k{0};
    uint64_t n{0};
    size_t groupListBytes{0};
};

inline std::vector<uint32_t> ParseGroupList(const std::vector<int64_t>& groupListHost)
{
    CHECK_COND(!groupListHost.empty(), "group_list must not be empty.");
    std::vector<uint32_t> groupMList;
    groupMList.reserve(groupListHost.size());
    for (int64_t groupM : groupListHost) {
        CHECK_COND(groupM >= 0, "Each group M value must be greater than or equal to zero.");
        CHECK_COND(
            groupM <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
            "Each group M value must not exceed INT32_MAX.");
        groupMList.push_back(static_cast<uint32_t>(groupM));
    }
    return groupMList;
}

inline uint64_t ParsePositiveUint64(const char* arg, const char* name)
{
    std::string value(arg);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(std::string("ERROR: ") + name + " must be a positive integer");
    }

    try {
        uint64_t parsed = std::stoull(value);
        if (parsed == 0UL) {
            throw std::invalid_argument(std::string("ERROR: ") + name + " must be greater than 0");
        }
        return parsed;
    } catch (const std::out_of_range&) {
        throw std::invalid_argument(std::string("ERROR: ") + name + " is out of range for uint64_t");
    }
}

inline void PrintUsage(const char* program)
{
    std::cerr << "Usage: " << program << " group_num m k n" << std::endl;
    std::cerr << "Example: " << program << " 2 256 4096 1024" << std::endl;
}

inline GroupedMatmulMxfp4Args ParseArguments(int argc, char* argv[])
{
    if (argc != 5) {
        throw std::invalid_argument("ERROR: Invalid number of arguments, expected exactly 4 arguments: group_num m k n");
    }

    GroupedMatmulMxfp4Args args{
        ParsePositiveUint64(argv[1], "group_num"),
        ParsePositiveUint64(argv[2], "m"),
        ParsePositiveUint64(argv[3], "k"),
        ParsePositiveUint64(argv[4], "n"),
        0U};

    constexpr uint64_t int32MaxU64 = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    CHECK_COND(args.m <= int32MaxU64, "m must not exceed INT32_MAX.");
    CHECK_COND(args.k <= int32MaxU64, "k must not exceed INT32_MAX.");
    CHECK_COND(args.n <= int32MaxU64, "n must not exceed INT32_MAX.");
    CHECK_COND(args.k % 2UL == 0UL, "k must be even for fp4 packed storage.");
    CHECK_COND(
        args.groupNum <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
        "group_num exceeds addressable size on this platform.");
    CHECK_COND(
        args.groupNum <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / sizeof(int64_t),
        "group list byte size exceeds addressable size on this platform.");
    args.groupListBytes = static_cast<size_t>(args.groupNum * sizeof(int64_t));
    return args;
}

