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
 * \file gelu_cpu.cpp
 * \brief
 */


#include "gelu_cpu.h"
#include <cmath>
#include <algorithm>
#include <iostream>


void gelu_cpu(const std::vector<float>& input, std::vector<float>& output)
{
    const float TANH_APPROX_FACTOR = 1 / 0.044715;
    const float NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715;
    if (output.size() != input.size()) {
        output.resize(input.size());
    }

    for (std::size_t i = 0; i < input.size(); ++i)
    {
        float x = input[i];
        float x_cube = x * x * x;
        output[i] = x / (1.0f + std::exp((x * TANH_APPROX_FACTOR + x_cube) * NEG_SQRT_EIGHT_OVER_PI));
    }
}
