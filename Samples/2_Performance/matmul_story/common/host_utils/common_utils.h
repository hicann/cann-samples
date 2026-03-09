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

#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H
#include <iostream>
#include <fstream>

#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)
#define CHECK_COND(cond, msg)                                                                                  \
    do {                                                                                                       \
        if (!(cond)) {                                                                                         \
            throw std::runtime_error(                                                                          \
                std::string("Error: ") + msg + "\nFile: " + __FILE__ + "\nLine: " + std::to_string(__LINE__)); \
        }                                                                                                      \
    } while (0)

template <typename T>
static T CeilDiv(T num1, T num2)
{
    return (num1 + num2 - 1) / num2;
}

template <typename T>
static T CeilAlign(T num1, T num2)
{
    return CeilDiv(num1, num2) * num2;
}

template <typename T>
static T FloorAlign(T num1, T num2)
{
    return num1 / num2 * num2;
}

#endif // COMMON_UTILS_H
