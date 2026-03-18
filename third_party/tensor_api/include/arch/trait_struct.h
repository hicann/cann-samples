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
 * \file trait_struct.h
 * \brief
 */
#ifndef EXPERIMENTAL_TENSOR_API_ARCH_TRAIT_STRUCT_H
#define EXPERIMENTAL_TENSOR_API_ARCH_TRAIT_STRUCT_H

#include "impl/utils/sys_macros.h"

namespace AscendC {
namespace Te {

enum class MmadType { NORMAL = 0, MX = 1 };

struct MmadTrait {
    int32_t fmOffset = 0;
    bool kDirectionAlign = false;
    bool cmatrixSource = false;
    bool disableGemv = false;
    MmadType mmadType = MmadType::NORMAL; 

    __aicore__ constexpr MmadTrait () {};

    __aicore__ constexpr MmadTrait (int32_t fmOffsetIn, bool kDirectionAlignIn, bool cmatrixSourceIn,
            bool disableGemvIn, MmadType mmadTypeIn) 
        {
            fmOffset = fmOffsetIn;
            kDirectionAlign = kDirectionAlignIn;
            cmatrixSource = cmatrixSourceIn;
            disableGemv = disableGemvIn;
            mmadType = mmadTypeIn;
        };
};

struct MmadParams {
    uint8_t unitFlag;
    bool cmatrixInitVal;
    uint16_t m = 0;
    uint16_t n = 0;
    uint16_t k = 0;

    __aicore__ MmadParams() {};

    __aicore__ constexpr MmadParams(uint16_t mIn, uint16_t nIn, uint16_t kIn, uint8_t unitFlagIn, bool cmatrixInitValIn) : 
        m(mIn), n(nIn), k(kIn), unitFlag(unitFlagIn), cmatrixInitVal(cmatrixInitValIn){};
};

struct DataCopyTrait {};

enum class RoundMode : uint8_t {
    DEFAULT = 0,
    HYBRID,
};

struct FixpipeTrait {
    __aicore__ constexpr FixpipeTrait() {}
    __aicore__ constexpr FixpipeTrait(
        RoundMode roundModeIn,
        bool enableReluIn,
        bool enableChannelSplitIn,
        uint8_t dualDstCtlIn
    ) :
        roundMode(roundModeIn),
        enableRelu(enableReluIn),
        enableChannelSplit(enableChannelSplitIn),
        dualDstCtl(dualDstCtlIn)
    {}

    RoundMode roundMode = RoundMode::DEFAULT;
    bool enableRelu = false;
    bool enableChannelSplit = false;
    uint8_t dualDstCtl = false;
};

struct FixpipeParams {
   uint8_t unitFlag = 0;
   __aicore__ FixpipeParams() {};

   __aicore__ constexpr FixpipeParams(uint8_t unitFlagIn) : 
      unitFlag(unitFlagIn) {};
};

struct LoadDataTrait {
    __aicore__ constexpr LoadDataTrait() {}

    __aicore__ constexpr LoadDataTrait(const bool transposedIn) : transposed(transposedIn) {}

    __aicore__ constexpr LoadDataTrait(const LoadDataTrait& trait, const bool transposedIn) : transposed(transposedIn) {}

    bool transposed = false;
};

} // namespace Te
} // namespace AscendC

#endif // EXPERIMENTAL_TENSOR_API_ARCH_TRAIT_STRUCT_H