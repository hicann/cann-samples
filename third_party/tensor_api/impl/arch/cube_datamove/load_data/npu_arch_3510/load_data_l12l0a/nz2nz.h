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
 * \file nz2nz.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0A_NZ2NZ_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0A_NZ2NZ_H

#include "impl/arch/cube_datamove/load_data/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class LoadDataFourDim3510L12L0ANZ2NZ {

public:
    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline void Run(const T& dst, const U& src) {
        LoadDataImpl<TraitHolder<trait, false>::traitTransposed, T, U>(dst, src);
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
        CheckFormat::CheckNZTemplate<U>();
        CheckDataTypeFor3510::CheckL12L0ADataType<T, U>();
    }

    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline void LoadDataImpl(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint16_t mStartPosition = 0;
        uint16_t kStartPosition = 0;
        auto mStep = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
        auto kStep = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
        // Nz -> Nz
        constexpr bool isFp4Type = Std::is_one_of_v<
                Std::tuple<typename T::elementType, typename U::elementType>,
                Std::tuple<__ca__ fp4x2_e2m1_t, __cbuf__ fp4x2_e2m1_t>,
                Std::tuple<__ca__ fp4x2_e1m2_t, __cbuf__ fp4x2_e1m2_t>>;
        constexpr int KHALF = 2;
        uint32_t STRIDE_UNIT = isFp4Type ? FRACTAL_FIXED * (C0_SIZE / sizeof(DstType) * KHALF) : FRACTAL_FIXED * (C0_SIZE / sizeof(DstType));
        auto srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / STRIDE_UNIT;
        auto dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout) / STRIDE_UNIT;
        if constexpr (isFp4Type) {
            LoadCbufToCaS4Base loadCbufToCaS4;
            loadCbufToCaS4.template LoadData<trait>(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);    
        } else {
            LoadCbufToCaBase loadCbufToCa;
            loadCbufToCa.template LoadData<trait>(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
        }
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0A_NZ2NZ_H