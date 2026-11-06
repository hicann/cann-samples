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
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)
#define CHECK_COND(cond, msg) \
    do { \
        if (!(cond)) { \
            throw std::runtime_error(std::string("Error: ") + msg + "\nFile: " + __FILE__ + \
                                     "\nLine: " + std::to_string(__LINE__)); \
        } \
    } while (0)

inline int ParseTutorialDimension(const char* arg)
{
    const std::string value(arg);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("ERROR: m k n must be positive integers");
    }
    const int parsed = std::stoi(value);
    if (parsed <= 0) {
        throw std::invalid_argument("ERROR: m k n must be positive");
    }
    return parsed;
}

inline void ParseTutorialDimensions(int argc, char* argv[], int& m, int& k, int& n)
{
    if (argc != 4) {
        throw std::invalid_argument("ERROR: Expected exactly three arguments: m k n");
    }
    m = ParseTutorialDimension(argv[1]);
    k = ParseTutorialDimension(argv[2]);
    n = ParseTutorialDimension(argv[3]);
    if (k % 2 != 0) {
        throw std::invalid_argument("ERROR: k must be an even number");
    }
    const int maxElements = std::numeric_limits<int>::max();
    if (m > maxElements / k || n > maxElements / k || m > maxElements / n) {
        throw std::invalid_argument("ERROR: Tensor element counts must not exceed INT_MAX");
    }
}

template <typename T>
inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return a / b + static_cast<T>(a % b != 0);
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

template <typename T>
T GetShapeWithDataTypeFP4(T size)
{
    return size << 1;
}

template <typename T>
T GetSizeWithDataTypeFP4(T shape)
{
    return (shape + 1) >> 1;
}
