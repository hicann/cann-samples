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
 * \file layout_utils.h
 * \brief
 */

#pragma once

#include "include/tensor_api/tensor.h"

template <typename LayoutPattern>
constexpr bool GetTransValue()
{
    constexpr bool isNonTrans =
        AscendC::Std::is_one_of_v<LayoutPattern, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn>;
    constexpr bool isTrans =
        AscendC::Std::is_one_of_v<LayoutPattern, AscendC::Te::DNExtLayoutPtn, AscendC::Te::ZNLayoutPtn>;

    constexpr bool isKnown = isNonTrans || isTrans;
    static_assert(isKnown, "IsTrans is not implemented for this layout pattern");

    return !isNonTrans && isTrans;
}

template <typename LayoutPattern>
struct IsTrans {
    static constexpr bool value = GetTransValue<LayoutPattern>();
};

template <typename LayoutPattern>
constexpr bool GetWeightNzValue()
{
    constexpr bool isNonWeightNz =
        AscendC::Std::is_one_of_v<LayoutPattern, AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn>;
    constexpr bool isWeightNz =
        AscendC::Std::is_one_of_v<LayoutPattern, AscendC::Te::NZLayoutPtn, AscendC::Te::ZNLayoutPtn>;

    constexpr bool isKnown = isNonWeightNz || isWeightNz;
    static_assert(isKnown, "IsWeightNz is not implemented for this layout");

    return !isNonWeightNz && isWeightNz;
}

template <typename LayoutPattern>
struct IsWeightNz {
    static constexpr bool value = GetWeightNzValue<LayoutPattern>();
};

