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
 * \file load_data_l12l0a.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_2201_LOAD_DATA_L12L0A_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_2201_LOAD_DATA_L12L0A_H

#include "impl/tensor/pointer_impl.h"
#include "impl/tensor/local_tensor_impl.h"
#include "impl/arch/utils/arch_utils.h"

namespace AscendC {
namespace Te {

class LoadDataFourDim2201L12L0ABase{
public:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        return;
    }

    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        if constexpr (!trait.transposed) {
            CheckFormat::CheckNZTemplate<U>();
        } else {
            CheckFormat::CheckZNTemplate<U>();
        }
        CheckFormat::CheckZZTemplate<T>();
        CheckDataTypeFor2201::CheckL12L0ADataType<T, U>();
    }

    template<typename T>
    __aicore__ inline void SetMatrixL0AImpl(T config)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            set_fmatrix(config);
        }
    }

    template <typename T>
    __aicore__ inline void LoadData3DV2L12L0AImpl(__ca__ T* dst, __cbuf__ T* src, uint16_t kExtension, uint16_t mExtension,
        uint16_t kStartPt, uint16_t mStartPt, bool enTranspose, uint16_t channelSize)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            img2colv2_cbuf_to_ca(dst, src, kExtension, mExtension, kStartPt, mStartPt, 1, 1, 1, 1, 1, 1, false, false, enTranspose, false, channelSize);
        }
    }

    template<typename T>
    __aicore__ inline void LoadDataTransposeL12L0AImpl(__ca__ T* dst, __cbuf__ T* src, uint16_t startIdx, uint16_t repeatTime, uint16_t srcStride,
        uint16_t dstStride, uint16_t dstFracStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            load_cbuf_to_ca_transpose(dst, src, startIdx, repeatTime, srcStride, dstStride, false, dstFracStride);
        }
    }
};

class LoadDataFourDim2201L12L0ANZ2ZZ : public LoadDataFourDim2201L12L0ABase {
public:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto params = GenLoadDataParams<trait, T, U, Coord>(dst, src, coord);
        LoadDataL0AImpl<trait, T, U, decltype(params)>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline auto GenLoadDataParams(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<trait, T, U>();
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        auto dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) * FRACTAL_FIXED;
        auto dstCol = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * C0_SIZE / sizeof(DstType);
        auto srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout) * FRACTAL_FIXED;
        auto srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout) * C0_SIZE / sizeof(DstType);
        auto indexRow = Std::get<1>(coord) * FRACTAL_FIXED;
        auto indexCol = Std::get<1>(coord) * C0_SIZE / sizeof(DstType);
        // NZ
        auto config = srcRow | SHIFT_LEFT_16;
        auto params = Std::make_tuple(dstRow, dstCol, srcRow, srcCol, indexRow, indexCol, config);
        return params;
    }

    template <const LoadDataTrait& trait, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void LoadDataL0AImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        LoadL1ToL0AImpl<trait.transposed>(dst.Data().Get(), src.Data().Get(),
            Std::get<Is>(tupleParams)...);
    }

    template<bool needTranspose, typename T>
    __aicore__ inline void LoadL1ToL0AImpl(__ca__ T* dst, __cbuf__ T* src, uint16_t dstRow, uint16_t dstCol, uint16_t srcRow,
        uint16_t srcCol, uint16_t indexRow, uint16_t indexCol, uint64_t config)
    {
        if ASCEND_IS_AIV {
            return;
        }

        if constexpr(needTranspose) {
            SetMatrixL0AImpl(config);
            LoadData3DV2L12L0AImpl(dst, src, dstRow, dstCol, indexRow, indexCol, true, srcRow);
        } else {
            SetMatrixL0AImpl(config);
            LoadData3DV2L12L0AImpl(dst, src, dstCol, dstRow, indexCol, indexRow, false, srcCol);
        }
    }
};

class LoadDataFourDim2201L12L0AZN2ZZB8 : public LoadDataFourDim2201L12L0ABase {
public:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto params = GenLoadDataParams<trait, T, U, Coord>(dst, src, coord);
        LoadDataL0AImpl<trait, T, U, decltype(params)>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline auto GenLoadDataParams(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<trait, T, U>();
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        auto dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) * FRACTAL_FIXED;
        auto dstCol = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * C0_SIZE / sizeof(DstType);
        // ZN
        auto srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout) * C0_SIZE / sizeof(DstType);
        auto srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout) * FRACTAL_FIXED;
        auto indexRow = Std::get<1>(coord) * C0_SIZE / sizeof(DstType);
        auto indexCol = Std::get<1>(coord) * FRACTAL_FIXED;
        constexpr const uint32_t SHIFT_BLOCK_LEN = 4;
        constexpr const uint32_t SHIFT_BLOCK_BYTE = 5;
        constexpr uint16_t CUBE_BLOCK_SIZE = 512;
        constexpr uint16_t fracNum = 2;
        uint16_t srcColNum = srcCol >> (SHIFT_BLOCK_LEN + fracNum - 1);
        uint16_t dstColNum = dstCol * sizeof(DstType) >> SHIFT_BLOCK_BYTE;
        uint16_t dstRowNum = dstRow >> (SHIFT_BLOCK_LEN + fracNum - 1);
        uint16_t startIdx0 = (indexCol >> (SHIFT_BLOCK_LEN + fracNum - 1)) +
            (indexRow * srcColNum * sizeof(DstType) >> SHIFT_BLOCK_BYTE);
        auto params = Std::make_tuple(dstRowNum, dstColNum, fracNum, startIdx0, srcColNum);
        return params;
    }

    template <const LoadDataTrait& trait, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void LoadDataL0AImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        LoadL1ToL0AImpl<trait.transposed>(dst.Data().Get(), src.Data().Get(),
            Std::get<Is>(tupleParams)...);
    }

    template<bool needTranspose, typename T>
    __aicore__ inline void LoadL1ToL0AImpl(__ca__ T* dst, __cbuf__ T* src, uint16_t dstRowNum, uint16_t dstColNum,
        uint16_t fracNum, uint16_t startIdx0, uint16_t srcColNum)
    {
        if ASCEND_IS_AIV {
            return;
        }
        uint16_t dstGap = 0;
        uint16_t dstFracGap = 0;
        constexpr uint16_t CUBE_BLOCK_SIZE = 512;
        if (dstRowNum >= dstColNum) {
            dstGap = fracNum * dstColNum - 1;
            dstFracGap = dstColNum - 1;
            for (uint16_t i = 0; i < dstColNum; i++) {
                LoadDataTransposeL12L0AImpl(dst, src, startIdx0 + i, dstRowNum, srcColNum, dstGap, dstFracGap);
                dst += CUBE_BLOCK_SIZE;
            }
        } else {
            dstFracGap = dstColNum - 1;
            for (uint16_t i = 0; i < dstRowNum; i++) {
                LoadDataTransposeL12L0AImpl(dst, src, startIdx0 + i * srcColNum, dstColNum, 1, 0, dstFracGap);
                dst += dstColNum * CUBE_BLOCK_SIZE * fracNum;
            }
        }
    }
};

class LoadDataFourDim2201L12L0AZN2ZZ : public LoadDataFourDim2201L12L0ABase {
public:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto params = GenLoadDataParams<trait, T, U, Coord>(dst, src, coord);
        LoadDataL0AImpl<trait, T, U, decltype(params)>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline auto GenLoadDataParams(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<trait, T, U>();
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        auto dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) * FRACTAL_FIXED;
        auto dstCol = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * C0_SIZE / sizeof(DstType);
        auto srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout) * C0_SIZE / sizeof(DstType);
        auto srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout) * FRACTAL_FIXED;
        auto indexRow = Std::get<1>(coord) * C0_SIZE / sizeof(DstType);
        auto indexCol = Std::get<1>(coord) * FRACTAL_FIXED;
        auto config = srcCol | SHIFT_LEFT_16;
        auto params = Std::make_tuple(dstRow, dstCol, srcRow, srcCol, indexRow, indexCol, config);
        return params;
    }

    template <const LoadDataTrait& trait, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void LoadDataL0AImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        LoadL1ToL0AImpl<trait.transposed>(dst.Data().Get(), src.Data().Get(),
            Std::get<Is>(tupleParams)...);
    }

    template<bool needTranspose, typename T>
    __aicore__ inline void LoadL1ToL0AImpl(__ca__ T* dst, __cbuf__ T* src, uint16_t dstRow, uint16_t dstCol, uint16_t srcRow,
        uint16_t srcCol, uint16_t indexRow, uint16_t indexCol, uint64_t config)
    {
        if ASCEND_IS_AIV {
            return;
        }

        if constexpr(needTranspose) {
            SetMatrixL0AImpl(config);
            LoadData3DV2L12L0AImpl(dst, src, dstRow, dstCol, indexRow, indexCol, true, srcRow);
        } else {
            SetMatrixL0AImpl(config);
            LoadData3DV2L12L0AImpl(dst, src, dstCol, dstRow, indexCol, indexRow, false, srcCol);
        }
    }
};

class LoadDataFourDim2201L12L0A : public LoadDataFourDim2201L12L0ANZ2ZZ, public LoadDataFourDim2201L12L0AZN2ZZB8,
    public LoadDataFourDim2201L12L0AZN2ZZ {
public:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        if constexpr (IsNZFormat<U>::value && IsZZFormat<T>::value) {
            LoadDataFourDim2201L12L0ANZ2ZZ::Run<trait, T, U, Coord>(dst, src, coord);
        } else if constexpr (IsZNFormat<U>::value && IsZZFormat<T>::value && (sizeof(typename U::elementType) == 1)) {
            LoadDataFourDim2201L12L0AZN2ZZB8::Run<trait, T, U, Coord>(dst, src, coord);
        } else if constexpr (IsZNFormat<U>::value && IsZZFormat<T>::value) {
            LoadDataFourDim2201L12L0AZN2ZZ::Run<trait, T, U, Coord>(dst, src, coord);
        }
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_2201_LOAD_DATA_L12L0A_H