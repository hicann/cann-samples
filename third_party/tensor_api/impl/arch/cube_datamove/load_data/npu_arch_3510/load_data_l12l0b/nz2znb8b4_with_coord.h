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
 * \file zn2nzb8b4_with_coord.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0B_NZ2ZNB8B4_WITH_COORD_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0B_NZ2ZNB8B4_WITH_COORD_H

#include "impl/arch/cube_datamove/load_data/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {
class LoadDataFourDim3510L12L0BNZ2ZNB8B4WithCoord {

public:
    template <const LoadDataTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        LoadDataImpl<TraitHolder<trait, true>::traitTransposed, T, U, Coord>(dst, src, coord);
    }

private:
    template<const LoadDataTrait& trait, bool transpose>
    struct TraitHolder {
        static constexpr LoadDataTrait traitTransposed = LoadDataTrait(trait, transpose);
    };

    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        CheckFormat::CheckZNTemplate<T>();
        CheckFormat::CheckNZTemplate<U>();
        CheckDataTypeFor3510::CheckL12L0BDataType<T, U>();
    }

    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline void LoadDataImplB4(const T& dst, const U& src, uint16_t mStartPosition,
        uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride, uint16_t dstStride)
    {
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        constexpr const int KHALF = 2;
        constexpr const int SHIFT_M_STEP_B4 = 2;
        constexpr const int M_STEP_MIN_VAL_B4 = 4;
        uint16_t nLoop = mStep >> SHIFT_M_STEP_B4;
        uint16_t dstAddrStride = kStep * FRACTAL_FIXED * C0_SIZE;
        mStep = M_STEP_MIN_VAL_B4;
        int x = 0, y = 0;
        int Col = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout) *
                    GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
        LoadCbufToCbS4Base loadCbufToCbS4;
        for (uint16_t idx = 0; idx < nLoop; ++idx) {
            auto sliceDst = dst(MakeCoord(x, y));
            loadCbufToCbS4.template LoadData<trait>(sliceDst, src, mStartPosition, kStartPosition / KHALF, 
                                                    mStep, kStep / KHALF, srcStride, dstStride);
            x += C0_SIZE * KHALF;
            mStartPosition += M_STEP_MIN_VAL_B4;
        }
    }

    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline void LoadDataImplB8(const T& dst, const U& src, uint16_t mStartPosition,
        uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride, uint16_t dstStride)
    {
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        constexpr const int SHIFT_M_STEP_B8 = 1;
        constexpr const int M_STEP_MIN_VAL_B8 = 2;
        uint16_t nLoop = mStep >> SHIFT_M_STEP_B8;
        uint16_t dstAddrStride = kStep * FRACTAL_FIXED * C0_SIZE;
        mStep = M_STEP_MIN_VAL_B8;
        int x = 0, y = 0;
        int Col = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout) *
                    GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
        LoadCbufToCbBase loadCbufToCb;
        for (uint16_t idx = 0; idx < nLoop; ++idx) {
            auto sliceDst = dst(MakeCoord(x, y));
            loadCbufToCb.template LoadData<trait>(sliceDst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
            x += C0_SIZE;
            mStartPosition += M_STEP_MIN_VAL_B8;
        }
    }

    template <const LoadDataTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void LoadDataImpl(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<trait, T, U>();
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        auto mStartPosition = Std::get<0>(coord) / FRACTAL_FIXED;
        auto kStartPosition = Std::get<1>(coord) * sizeof(DstType) / C0_SIZE;
        auto n1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) *
                GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout) / FRACTAL_FIXED;
        auto mStep = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) *
                GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout) / FRACTAL_FIXED;
        auto kStep = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) *
                GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout) * sizeof(DstType) / C0_SIZE;
        // Nz -> Zn
        constexpr bool isFp4Type = Std::is_one_of_v<
                Std::tuple<typename T::elementType, typename U::elementType>,
                Std::tuple<__cb__ fp4x2_e2m1_t, __cbuf__ fp4x2_e2m1_t>,
                Std::tuple<__cb__ fp4x2_e1m2_t, __cbuf__ fp4x2_e1m2_t>>;
        constexpr int KHALF = 2;
        uint32_t STRIDE_UNIT = isFp4Type ? FRACTAL_FIXED * (C0_SIZE / sizeof(DstType) * KHALF) : FRACTAL_FIXED * (C0_SIZE / sizeof(DstType));
        auto srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / STRIDE_UNIT;
        auto dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout) / STRIDE_UNIT;
        if constexpr (isFp4Type) {
            if ((n1 & 1) == 0) {
                LoadCbufToCbS4Base loadCbufToCbS4;
                loadCbufToCbS4.template LoadData<trait>(dst, src, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF, srcStride, dstStride);
            } else {
                LoadDataImplB4<trait, T, U>(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
            }
        } else {
            if ((n1 & 1) == 0) {
                LoadCbufToCbBase loadCbufToCb;
                loadCbufToCb.template LoadData<trait>(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
            } else {
                LoadDataImplB8<trait, T, U>(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
            }
        }
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0B_NZ2ZNB8B4_WITH_COORD_H