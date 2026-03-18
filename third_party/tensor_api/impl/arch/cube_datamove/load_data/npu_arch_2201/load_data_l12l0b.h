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
 * \file load_data_l12l0b.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_2201_LOAD_DATA_L12L0B_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_2201_LOAD_DATA_L12L0B_H

#include "impl/tensor/pointer_impl.h"
#include "impl/tensor/local_tensor_impl.h"
#include "impl/arch/utils/arch_utils.h"

namespace AscendC {
namespace Te {

class LoadDataFourDim2201L12L0BBase {
public:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        return;
    }

    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        if constexpr (trait.transposed) {
            CheckFormat::CheckNZTemplate<U>();
        } else {
            CheckFormat::CheckZNTemplate<U>();
        }
        CheckFormat::CheckZNTemplate<T>();
        CheckDataTypeFor2201::CheckL12L0BDataType<T, U>();
    }

    template<typename T>
    __aicore__ inline void SetMatrixL0BImpl(T config)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            set_fmatrix_b(config);
        }
    }

    template <typename T>
    __aicore__ inline void LoadData3DV2L12L0BImpl(__cb__ T* dst, __cbuf__ T* src, uint16_t kExtension, uint16_t mExtension,
        uint16_t kStartPt, uint16_t mStartPt, bool enTranspose, uint16_t channelSize)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            img2colv2_cbuf_to_cb(dst, src, kExtension, mExtension, kStartPt, mStartPt,
                1, 1, 1, 1, 1, 1, false, false, false, enTranspose, channelSize);
        }
    }

    template <typename T>
    __aicore__ inline void LoadDataTransposeL12L0BImpl(__cb__ T* dst, __cbuf__ T* src, uint16_t startIdx,
        uint16_t repeatTimes, uint16_t srcStride, uint16_t dstGap, uint16_t dstFracGap)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            load_cbuf_to_cb_transpose(dst, src, startIdx, repeatTimes, srcStride, dstGap, false, dstFracGap);
        }
    }

    template <typename T>
    __aicore__ inline void LoadData2DL12L0BImpl(__cb__ T* dst, __cbuf__ T* src, uint16_t startIdx, uint16_t repeatTimes,
        uint16_t srcStride, uint16_t dstGap, bool transpose)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            if (transpose) {
                load_cbuf_to_cb(dst, src, startIdx, repeatTimes, srcStride, dstGap, 0, true, addr_cal_mode_t(0));
            } else {
                load_cbuf_to_cb(dst, src, startIdx, repeatTimes, srcStride, dstGap, 0, false, addr_cal_mode_t(0));
            }
        }
    }
};

class LoadDataFourDim2201L12L0BNZ2ZNB8 : public LoadDataFourDim2201L12L0BBase {
public:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto params = GenLoadDataParams<trait, T, U, Coord>(dst, src, coord);
        LoadDataAlignV2Impl<trait, T, U, decltype(params)>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline auto GenLoadDataParams(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<trait, T, U>();
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        auto dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) * C0_SIZE / sizeof(DstType);
        auto dstCol = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * FRACTAL_FIXED;

        constexpr const uint32_t SHIFT_BLOCK_LEN = 4;
        constexpr const uint32_t SHIFT_BLOCK_BYTE = 5;

        auto srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout) * FRACTAL_FIXED;
        auto srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout) * C0_SIZE / sizeof(DstType);
        auto indexRow = Std::get<1>(coord) * FRACTAL_FIXED;
        auto indexCol = Std::get<1>(coord) * C0_SIZE / sizeof(DstType);
        constexpr uint16_t fracNum = 2;
        uint16_t srcColNum = srcCol * sizeof(DstType) >> SHIFT_BLOCK_BYTE;
        uint16_t srcRowNum = srcRow >> (SHIFT_BLOCK_LEN + fracNum - 1);
        uint16_t dstColNum = dstCol >> (SHIFT_BLOCK_LEN + fracNum - 1);
        uint16_t dstRowNum = dstRow * sizeof(DstType) >> SHIFT_BLOCK_BYTE;
        uint16_t startIdx0 = (indexRow >> (SHIFT_BLOCK_LEN + fracNum - 1)) +
            (indexCol * sizeof(DstType) * srcRowNum >> SHIFT_BLOCK_BYTE);
        auto params = Std::make_tuple(dstRowNum, dstColNum, fracNum, startIdx0, srcRowNum);
        return params;
    }

    template <const LoadDataTrait& trait, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void LoadDataAlignV2Impl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        LoadL1ToL0B<trait.transposed>(dst.Data().Get(), src.Data().Get(),
            Std::get<Is>(tupleParams)...);
    }

    template<bool needTranspose, typename T>
    __aicore__ inline void LoadL1ToL0B(__cb__ T* dst, __cbuf__ T* src, uint16_t dstRowNum, uint16_t dstColNum,
        uint16_t fracNum, uint16_t startIdx0, uint16_t srcRowNum)
    {
        if ASCEND_IS_AIV {
            return;
        }
        uint16_t dstGap = 0;
        constexpr uint16_t CUBE_BLOCK_SIZE = 512;
        if (dstRowNum >= dstColNum) {
            dstGap = fracNum * dstColNum - 1;
            for (uint16_t i = 0; i < dstColNum; i ++) {
                LoadDataTransposeL12L0BImpl(dst, src, startIdx0 + i * srcRowNum, dstRowNum, 1, dstGap, 0);
                dst += fracNum * CUBE_BLOCK_SIZE;
            }
        } else {
            dstGap = fracNum - 1;
            for (uint16_t i = 0; i < dstRowNum; i ++) {
                LoadDataTransposeL12L0BImpl(dst, src, startIdx0 + i, dstColNum, srcRowNum, dstGap, 0);
                dst += dstColNum * fracNum * CUBE_BLOCK_SIZE;
            }
        }
    }
};

class LoadDataFourDim2201L12L0BNZ2ZN : public LoadDataFourDim2201L12L0BBase {
public:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto params = GenLoadDataParams<trait, T, U, Coord>(dst, src, coord);
        LoadDataAlignV2Impl<trait, T, U, decltype(params)>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline auto GenLoadDataParams(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<trait, T, U>();
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        auto dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) * C0_SIZE / sizeof(DstType);
        auto dstCol = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * FRACTAL_FIXED;
        auto srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout) * FRACTAL_FIXED;
        auto srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout) * C0_SIZE / sizeof(DstType);
        auto indexRow = Std::get<1>(coord) * FRACTAL_FIXED;
        auto indexCol = Std::get<1>(coord) * C0_SIZE / sizeof(DstType);
        auto config = srcRow | SHIFT_LEFT_16;
        auto params = Std::make_tuple(dstRow, dstCol, srcRow, srcCol, indexRow, indexCol, config);
        return params;
    }

    template <const LoadDataTrait& trait, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void LoadDataAlignV2Impl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        LoadL1ToL0B<trait.transposed>(dst.Data().Get(), src.Data().Get(),
            Std::get<Is>(tupleParams)...);
    }

    template<bool needTranspose, typename T>
    __aicore__ inline void LoadL1ToL0B(__cb__ T* dst, __cbuf__ T* src, uint16_t dstRow, uint16_t dstCol,
        uint16_t srcRow, uint16_t srcCol, uint16_t indexRow, uint16_t indexCol, uint64_t config)
    {
        if ASCEND_IS_AIV {
            return;
        }
        SetMatrixL0BImpl(config);
        LoadData3DV2L12L0BImpl(dst, src, dstCol, dstRow, indexCol, indexRow, needTranspose, srcCol);
    }
};

class LoadDataFourDim2201L12L0BZN2ZN : public LoadDataFourDim2201L12L0BBase {
public:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto params = GenLoadDataParams<trait, T, U, Coord>(dst, src, coord);
        LoadDataAlignV2Impl<trait, T, U, decltype(params)>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline auto GenLoadDataParams(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<trait, T, U>();
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        auto dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) * C0_SIZE / sizeof(DstType);
        auto dstCol = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * FRACTAL_FIXED;

        constexpr const uint32_t SHIFT_BLOCK_LEN = 4;
        constexpr const uint32_t SHIFT_BLOCK_BYTE = 5;
        constexpr const int BLOCK_BYTE_SIZE = C0_SIZE;
        constexpr uint16_t CUBE_BLOCK_SIZE = 512;
        auto srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout) * C0_SIZE / sizeof(DstType);
        auto srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout) * FRACTAL_FIXED;
        auto indexRow = Std::get<1>(coord) * C0_SIZE / sizeof(DstType);
        auto indexCol = Std::get<1>(coord) * FRACTAL_FIXED;
        constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(DstType);
        uint16_t dstRowNum = (dstRow * sizeof(DstType)) >> SHIFT_BLOCK_BYTE;
        uint16_t dstColNum = dstCol >> SHIFT_BLOCK_LEN;
        uint16_t srcColNum = srcCol >> SHIFT_BLOCK_LEN;
        uint16_t srcRowNum = (srcRow * sizeof(DstType)) >> SHIFT_BLOCK_BYTE;
        uint16_t blockNum = CUBE_BLOCK_SIZE >> (sizeof(DstType) == 1 ? 0 :
                                                sizeof(DstType) == 2 ? 1 :
                                                sizeof(DstType) == 4 ? 2 : 0);
        uint16_t startIdx0 = 
            (indexRow * sizeof(DstType) * srcColNum >> SHIFT_BLOCK_BYTE) + (indexCol >> SHIFT_BLOCK_LEN);
        auto params = Std::make_tuple(dstCol, dstRowNum, dstColNum, startIdx0, srcColNum, indexRow, indexCol, blockNum);
        return params;
    }

    template <const LoadDataTrait& trait, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void LoadDataAlignV2Impl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        LoadL1ToL0B<trait.transposed>(dst.Data().Get(), src.Data().Get(),
            Std::get<Is>(tupleParams)...);
    }

    template <bool needTranspose, typename T>
    __aicore__ inline void LoadL1ToL0B(__cb__ T* dst, __cbuf__ T* src, uint16_t dstCol, uint16_t dstRowNum, uint16_t dstColNum,
        uint16_t startIdx0, uint16_t srcColNum, uint16_t indexRow, uint16_t indexCol, uint16_t blockNum)
    {
        if ASCEND_IS_AIV {
            return;
        }
        constexpr const int BLOCK_BYTE_SIZE = C0_SIZE;
        constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(T);
        uint16_t dstGap = 0;
        if (dstRowNum >= dstColNum) {
            dstGap = dstColNum - 1;
            for (uint16_t i = 0; i < dstColNum; i++) {
                LoadData2DL12L0BImpl(dst, src, startIdx0 + i, dstRowNum, srcColNum, dstGap, false);
                dst += blockNum;
            }
        } else {
            for (uint16_t i = 0; i < dstRowNum; i++) {
                LoadData2DL12L0BImpl(dst, src, startIdx0 + i * srcColNum, dstColNum, 1, 0, false);
                dst += dstCol * c0Size;
            }
        }
    }
};

class LoadDataFourDim2201L12L0B : public LoadDataFourDim2201L12L0BNZ2ZNB8, public LoadDataFourDim2201L12L0BNZ2ZN,
    public LoadDataFourDim2201L12L0BZN2ZN {
public:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        using type = typename U::elementType;
        if constexpr (IsNZFormat<U>::value && IsZNFormat<T>::value && (sizeof(type) == 1)) {
            LoadDataFourDim2201L12L0BNZ2ZNB8::Run<trait, T, U, Coord>(dst, src, coord);
        } else if constexpr (IsNZFormat<U>::value && IsZNFormat<T>::value) {
            LoadDataFourDim2201L12L0BNZ2ZN::Run<trait, T, U, Coord>(dst, src, coord);
        } else if constexpr (IsZNFormat<U>::value && IsZNFormat<T>::value) {
            LoadDataFourDim2201L12L0BZN2ZN::Run<trait, T, U, Coord>(dst, src, coord);
        }
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_2201_LOAD_DATA_L12L0B_H