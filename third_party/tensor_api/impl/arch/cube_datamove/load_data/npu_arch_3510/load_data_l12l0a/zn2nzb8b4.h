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
 * \file zn2nzb8b4.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0A_ZN2NZB8B4_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0A_ZN2NZB8B4_H

#include "impl/arch/cube_datamove/load_data/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {
class LoadDataFourDim3510L12L0AZN2NZB8B4 {

public:
    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline void Run(const T& dst, const U& src) {
        LoadDataImpl<TraitHolder<trait, true>::traitTransposed, T, U>(dst, src);
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

        template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline void LoadDataImplB4(const T& dst, const U& src, uint16_t mStartPosition,
        uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride, uint16_t dstStride)
    {
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        constexpr int KHALF = 2;
        constexpr int SHIFT_M_STEP_B4 = 2;
        constexpr int M_STEP_MIN_VAL_B4 = 4;
        uint16_t mLoop = mStep >> SHIFT_M_STEP_B4;
        uint16_t dstAddrStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) *
                GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout) * C0_SIZE;
        mStep = M_STEP_MIN_VAL_B4;
        LoadCbufToCaS4Base loadCbufToCaS4;
        auto iter = dst.Data();
        for (uint16_t idx = 0; idx < mLoop; ++idx) {
            auto sliceDst = MakeTensor(iter, dst.Layout());
            loadCbufToCaS4.template LoadData<trait>(sliceDst, src, mStartPosition, kStartPosition / KHALF, 
                                                    mStep, kStep / KHALF, srcStride, dstStride);
            iter = iter + dstAddrStride;
            mStartPosition += M_STEP_MIN_VAL_B4;
        }
    }

    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline void LoadDataImplB8(const T& dst, const U& src, uint16_t mStartPosition,
        uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride, uint16_t dstStride)
    {
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        constexpr int SHIFT_M_STEP_B8 = 1;
        constexpr int M_STEP_MIN_VAL_B8 = 2;
        uint16_t mLoop = mStep >> SHIFT_M_STEP_B8;
        uint16_t dstAddrStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) *
                GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout) * C0_SIZE;
        mStep = M_STEP_MIN_VAL_B8;
        LoadCbufToCaBase loadCbufToCa;
        auto iter = dst.Data();
        for (uint16_t idx = 0; idx < mLoop; ++idx) {
            auto sliceDst = MakeTensor(iter, dst.Layout());
            loadCbufToCa.template LoadData<trait>(sliceDst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
            iter = iter + dstAddrStride;
            mStartPosition += M_STEP_MIN_VAL_B8;
        }
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
        auto m1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout) *
                  GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) -
                  GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) *
                  GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
        auto mStep = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout) *
                GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) / FRACTAL_FIXED;
        auto kStep = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout) *
                GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) / C0_SIZE;
        // Zn -> Nz
        constexpr bool isFp4Type = Std::is_one_of_v<
                Std::tuple<typename T::elementType, typename U::elementType>,
                Std::tuple<__ca__ fp4x2_e2m1_t, __cbuf__ fp4x2_e2m1_t>,
                Std::tuple<__ca__ fp4x2_e1m2_t, __cbuf__ fp4x2_e1m2_t>>;
        constexpr int KHALF = 2;
        uint32_t STRIDE_UNIT = isFp4Type ? FRACTAL_FIXED * (C0_SIZE / sizeof(DstType) * KHALF) : FRACTAL_FIXED * (C0_SIZE / sizeof(DstType));
        auto srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout) / STRIDE_UNIT;
        auto dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout) / STRIDE_UNIT;
        if constexpr (isFp4Type) {
            if (m1 < FRACTAL_FIXED) {
                LoadCbufToCaS4Base loadCbufToCaS4;
                loadCbufToCaS4.template LoadData<trait>(dst, src, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF, srcStride, dstStride);
            } else {
                LoadDataImplB4<trait, T, U>(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
            }
        } else {
            if (m1 < FRACTAL_FIXED) {
                LoadCbufToCaBase loadCbufToCa;
                loadCbufToCa.template LoadData<trait>(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
            } else {
                LoadDataImplB8<trait, T, U>(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
            }
        }
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0A_ZN2NZB8B4_H