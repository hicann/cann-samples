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
 * \file pad_mx_kl1.h
 * \brief zero-pad A/B L1 buffers along K when the GM slice is shorter than the L1-aligned layout.
 */
#pragma once

#include "kernel_utils/common_utils.h"
#include "include/tensor.h"

using AscendC::Te::AttrInfo;
using AscendC::Te::C0_SIZE;
using AscendC::Te::GetEleFromLayout;

namespace Tile {
struct PadMxKL1Base {
    template <typename T>
    __aicore__ inline static void PadZero(const T& tensorL1, uint64_t repeatTimes, uint64_t blockNum, uint64_t dstGap)
    {
        create_cbuf_matrix((__cbuf__ half*)tensorL1.Data().Get(), (blockNum << 16) | (dstGap << 32) | repeatTimes, 0);
    }

    template <typename T>
    __aicore__ inline static constexpr bool IsMxFp4()
    {
        using type = typename T::elementType;
        return AscendC::Std::is_one_of_v<type, __cbuf__ fp4x2_e1m2_t, __cbuf__ fp4x2_e2m1_t>;
    }

    template <typename T>
    __aicore__ inline static constexpr bool IsMxFp8()
    {
        using type = typename T::elementType;
        return AscendC::Std::is_one_of_v<type, __cbuf__ fp8_e5m2_t, __cbuf__ fp8_e4m3fn_t>;
    }
};

struct PadMxKAL1 : public PadMxKL1Base {
    template <typename T, typename U>
    __aicore__ inline static void PadZero(const T& tensorL1, const U& tensorGm)
    {
        static_assert(IsMxFp4<T>() || IsMxFp8<T>(), "Only support mxfp4/mxfp8!");
        auto layoutL1 = tensorL1.Layout();
        auto layoutGm = tensorGm.Layout();
        auto kAxis = GetEleFromLayout<decltype(layoutGm), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(layoutGm);
        auto kAxisL1Align = GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(layoutL1) *
                            GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(layoutL1);

        if constexpr (AscendC::Te::IsNDFormat<U>::value) {
            if constexpr (IsMxFp4<T>()) {
                return;
            }

            if (kAxisL1Align - kAxis < C0_SIZE<T>) {
                return;
            }
            auto mAlign = GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::ROW, 0>(layoutL1) *
                          GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::ROW, 1>(layoutL1);
            auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, C0_SIZE<T>); // K方向的坐标是ND2NZ指令对齐后的值
            auto sliceTensor = tensorL1(AscendC::Te::MakeCoord(0, kAxisND2NZAlign));
            PadMxKL1Base::PadZero(sliceTensor, 1, mAlign, 0);
        } else if constexpr (AscendC::Te::IsDNFormat<U>::value) {
            // ND2NZ指令只支持给最内轴（m0）补零，外轴需要自己清零
            if (kAxis == kAxisL1Align) {
                return;
            }
            // shape: [m1, k1, k0, m0] 清零迭代次数为NZ最外轴大小，左矩阵就是m1
            auto m1 = GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::ROW, 1>(layoutL1);
            auto sliceTensor = tensorL1(AscendC::Te::MakeCoord(0, kAxis));
            PadMxKL1Base::PadZero(sliceTensor, m1, kAxisL1Align - kAxis, kAxis);
        }
    }
};

struct PadMxKBL1 : public PadMxKL1Base {
    template <typename T, typename U>
    __aicore__ inline static void PadZero(const T& tensorL1, const U& tensorGm)
    {
        static_assert(IsMxFp4<T>() || IsMxFp8<T>(), "Only support mxfp4/mxfp8!");
        auto layoutL1 = tensorL1.Layout();
        auto layoutGm = tensorGm.Layout();

        auto kAxis = GetEleFromLayout<decltype(layoutGm), AttrInfo::SHAPE, AttrInfo::ROW, 1>(layoutGm);
        auto kAxisL1Align = GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::ROW, 0>(layoutL1) *
                            GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::ROW, 1>(layoutL1);

        if constexpr (AscendC::Te::IsNDFormat<U>::value) {
            // ND2NZ指令只支持给最内轴（n0）补零，外轴需要自己清零
            if (kAxis == kAxisL1Align) {
                return;
            }

            // shape: [n1, k1, k0, n0] 清零迭代次数为NZ最外轴大小，右矩阵就是n1
            auto n1 = GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(layoutL1);
            auto sliceTensor = tensorL1(AscendC::Te::MakeCoord(kAxis, 0));
            PadMxKL1Base::PadZero(sliceTensor, n1, kAxisL1Align - kAxis, kAxis);
        } else if constexpr (AscendC::Te::IsDNFormat<U>::value) {
            if constexpr (IsMxFp4<T>()) {
                return;
            }

            if (kAxisL1Align - kAxis < C0_SIZE<T>) {
                return;
            }
            auto nAlign = GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(layoutL1) *
                          GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(layoutL1);
            auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, C0_SIZE<T>); // K方向的坐标是ND2NZ指令对齐后的值
            auto sliceTensor = tensorL1(AscendC::Te::MakeCoord(kAxisND2NZAlign, 0));
            PadMxKL1Base::PadZero(sliceTensor, 1, nAlign, 0);
        } else if constexpr (AscendC::Te::IsNZFormat<U>::value) {
            // ND2NZ指令只支持给最内轴（n0）补零，外轴需要自己清零
            auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, AscendC::BLOCK_CUBE);
            if (kAxisND2NZAlign == kAxisL1Align) {
                return;
            }

            // shape: [n1, k1, k0, n0] 清零迭代次数为NZ最外轴大小，右矩阵就是n1
            auto n1 = GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(layoutL1);
            auto sliceTensor = tensorL1(AscendC::Te::MakeCoord(kAxis, 0));
            PadMxKL1Base::PadZero(sliceTensor, n1, kAxisL1Align - kAxis, kAxis);
        }
    }
};
} // namespace Tile

