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
 * \file zn2nz_with_coord.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0A_ZN2NZ_WITH_COORD_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0A_ZN2NZ_WITH_COORD_H

#include "impl/arch/cube_datamove/load_data/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {
class LoadDataFourDim3510L12L0AZN2NZWithCoord {

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
        CheckFormat::CheckNZTemplate<T>();
        CheckFormat::CheckZNTemplate<U>();
        CheckDataTypeFor3510::CheckL12L0ADataType<T, U>();
    }

    template <const LoadDataTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void LoadDataImpl(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<trait, T, U>();
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint16_t mStartPosition = Std::get<1>(coord) / FRACTAL_FIXED;
        uint16_t kStartPosition = Std::get<0>(coord) * sizeof(typename U::elementType) / C0_SIZE;
        auto mStep = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) *
                GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout) / FRACTAL_FIXED;
        auto kStep = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) *
                GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout) * sizeof(DstType) / C0_SIZE;
        // Zn -> Nz
        uint32_t STRIDE_UNIT = FRACTAL_FIXED * (C0_SIZE / sizeof(DstType));
        auto srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout) / STRIDE_UNIT;
        auto dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout) / STRIDE_UNIT;
        LoadCbufToCaBase loadCbufToCa;
        loadCbufToCa.template LoadData<trait>(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0A_ZN2NZ_WITH_COORD_H