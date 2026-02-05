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
 * \file matmul_golden.h
 * \brief
 */

#ifndef MATMUL_GOLDEN_H
#define MATMUL_GOLDEN_H

#include <vector>

namespace matmul {

template <typename T>
void FillRandomData(std::vector<T>& data, T min, T max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(min, max);
        for (auto& elem : data) elem = dist(gen);
    } else if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dist(min, max);
        for (auto& elem : data) elem = dist(gen);
    }
}

template <typename T>
void ComputeGolden(int m, int k, int n, std::vector<T>& hostInput, std::vector<T>& hostWeight,
                   std::vector<T>& goldenOutput)
{
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            size_t offsetGolden = row * n + col;
            T sum = 0;
            for (uint32_t iter = 0; iter < k; ++iter) {
                size_t offsetInput = row * k + iter;
                size_t offsetWeight = iter * n + col;
                sum += hostInput[offsetInput] * hostWeight[offsetWeight];
            }
            goldenOutput[offsetGolden] = sum;
        }
    }
}

template <typename T>
std::vector<uint64_t> Compare(std::vector<T>& hostOutput, std::vector<T>& goldenOutput)
{
    std::vector<uint64_t> errorIndices;
    const float rtol = 1.0f / 256;
    for (uint64_t i = 0; i < hostOutput.size(); ++i) {
        T actualValue = hostOutput[i];
        T expectValue = goldenOutput[i];
        T diff = std::fabs(actualValue - expectValue);
        if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
            errorIndices.push_back(i);
        }
    }
    return errorIndices;
}

} // namespace matmul

#endif // MATMUL_GOLDEN_H