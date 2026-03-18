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
 * \file fixpipe_base.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_BASE_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_BASE_H

#include "impl/arch/cube_datamove/fixpipe/fixpipe_utils.h"
#include "impl/tensor/pointer_impl.h"
#include "impl/tensor/local_tensor_impl.h"
#include "impl/arch/utils/arch_utils.h"
#include "impl/tensor/layout_method.h"

namespace AscendC {
namespace Te {
constexpr uint32_t MAIN_LOOP_N_SIZE_2201 = 512;
constexpr uint32_t CBURST_NUM_2201 = MAIN_LOOP_N_SIZE_2201 / FRACTAL_FIXED;

template <const FixpipeTrait& trait, typename T, typename U, typename S = void>
__aicore__ inline constexpr QuantMode_t GetFixpipe2201QuantPre()
{
    using srcType = typename U::elementType;
    using dstType = typename T::elementType;
    constexpr bool isTensor = IsTileTensorV<S>;
    constexpr bool isScalar = Std::is_same_v<S, uint64_t>;
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
        if constexpr (isTensor) {
            if constexpr (Std::is_same_v<srcType, __cc__ int32_t> && Std::is_same_v<dstType, __gm__ half>) {
                return QuantMode_t::VDEQF16;
            } else if constexpr (
                Std::is_same_v<srcType, __cc__ float> && Std::is_one_of_v<dstType, __gm__ uint8_t, __gm__ int8_t>) {
                return QuantMode_t::VQF322B8_PRE;
            } else if constexpr (
                Std::is_same_v<srcType, __cc__ int32_t> && Std::is_one_of_v<dstType, __gm__ uint8_t, __gm__ int8_t>) {
                return QuantMode_t::VREQ8;
            } else {
                return QuantMode_t::NoQuant;
            }
        } else if constexpr (isScalar) {
            if constexpr (Std::is_same_v<srcType, __cc__ int32_t> && Std::is_same_v<dstType, __gm__ half>) {
                return QuantMode_t::DEQF16;
            } else if constexpr (
                Std::is_same_v<srcType, __cc__ float> && Std::is_one_of_v<dstType, __gm__ uint8_t, __gm__ int8_t>) {
                return QuantMode_t::QF322B8_PRE;
            } else if constexpr (
                Std::is_same_v<srcType, __cc__ int32_t> && Std::is_one_of_v<dstType, __gm__ uint8_t, __gm__ int8_t>) {
                return QuantMode_t::REQ8;
            } else {
                return QuantMode_t::NoQuant;
            }
        } else {
            if constexpr (Std::is_same_v<srcType, __cc__ float> && Std::is_same_v<dstType, __gm__ half>) {
                return QuantMode_t::F322F16;
            } else if constexpr (Std::is_same_v<srcType, __cc__ float> && Std::is_same_v<dstType, __gm__ bfloat16_t>) {
                return QuantMode_t::F322BF16;
            } else {
                return QuantMode_t::NoQuant;
            }
        }
    } else {
        return QuantMode_t::NoQuant;
    }
}

template <typename T, typename U, typename Coord>
__aicore__ inline void CheckCoord(const T& dst, const U& src, const Coord& coord)
{
    auto coordRow = Std::get<0>(coord);
    auto coordCol = Std::get<1>(coord);
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    uint32_t dstRow1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t dstCol1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
    uint32_t srcRow1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    uint32_t srcCol1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
}

template <typename T, typename Coord>
__aicore__ inline auto MakeTensorWithCoord(const T& oldTensor, const Coord& coord, uint32_t offset = 0)
{
    auto oldTensorLayout = oldTensor.Layout();
    auto index = Crd2Idx(coord, oldTensorLayout);
    using oldTensorType = typename T::elementType;
    constexpr Hardware oldTensorPos = GetHardPos<T>();
    auto oldTensorIterator = MakeMemPtr<oldTensorPos>(reinterpret_cast<oldTensorType *>(oldTensor.Data().Get() + offset + index));
    auto oldTensorMatrixLayout = MakeLayout(oldTensorLayout.Shape(), oldTensorLayout.Stride());
    auto newTensor = MakeTensor(oldTensorIterator, oldTensorMatrixLayout); 
    return newTensor;
} 

class Copy2201DeqTensorToFbuf {
public:
    template <typename T>
    __aicore__ inline void CopyDeqTensorToFbufImpl(const T& src, uint16_t calNSize, uint16_t nIterIndex)
    {
        auto params = CopyDeqTensorToFbufGenParams(src, calNSize, nIterIndex);
        uint64_t dstAddr = AllocTempBuf(calNSize);
        DataCopyImpl<T, decltype(params)>(dstAddr, src, params, tuple_sequence<decltype(params)>{});
        SetFpc(dstAddr);
    }

private:
    template <typename T>
    __aicore__ inline constexpr void CheckTemplate()
    {
        CheckFormat::CheckNDTemplate<T>();
        CheckDataTypeFor2201::CheckFixPipeDataType<T>();
        constexpr Hardware srcTPos = GetHardPos<T>();
        static_assert(srcTPos == Hardware::L1, "The hardware of quant must be L1");
    }

    template <typename T>
    __aicore__ inline auto CopyDeqTensorToFbufGenParams(const T& src, uint16_t calNSize, uint16_t nIterIndex)
    {
        CheckTemplate<T>();
        constexpr uint16_t fbufBurstLenUnit = 128;
        using srcType = typename T::elementType;
        auto layout = src.Layout();
        uint16_t colLength = GetEleFromLayout<decltype(layout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(layout);
        uint16_t rowStride = GetEleFromLayout<decltype(layout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(layout);
        uint16_t blockCount = CeilDivision(calNSize, colLength);
        uint16_t blockLen = CeilDivision(colLength * sizeof(srcType), fbufBurstLenUnit);
        uint16_t srcStride = CeilDivision(rowStride * sizeof(srcType), C0_SIZE);
        uint16_t dstStride = blockLen;
        uint32_t deqValueOffset = MAIN_LOOP_N_SIZE_2201 / colLength * rowStride * nIterIndex;

        auto params = Std::make_tuple(blockCount, blockLen, srcStride, dstStride, deqValueOffset);
        return params;
    }

    template <typename T, typename U, size_t... Is>
    __aicore__ inline void DataCopyImpl(
        const uint64_t& dstAddr, const T& src, const U& tupleParams, Std::index_sequence<Is...>)
    {
        using srcType = typename T::elementType;
        CopyCbufToFbuf<srcType, decltype(tupleParams)>(
            dstAddr, (__cbuf__ uint64_t *)src.Data().Get(), Std::get<Is>(tupleParams)...);
    }

    template <typename T, typename U>
    __aicore__ inline void CopyCbufToFbuf(uint64_t dst, __cbuf__ T *src, uint16_t blockCount,
        uint16_t blockLen, uint16_t srcStride, uint16_t dstStride, uint32_t deqValueOffset)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            copy_cbuf_to_fbuf((__fbuf__ uint64_t *)dst, src + deqValueOffset, blockCount, blockLen, srcStride, dstStride);
        }
    }
};

class Copy2201MatrixCcToGmBase {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename S>
    __aicore__ inline void DataCopy(const T& dst, const U& src, const S& params)
    {
        DataCopyImpl<trait, quantPre, T, U, S>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename S, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const S& tupleParams, Std::index_sequence<Is...>)
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        CopyMatrixCcToGm<quantPre, dstType, srcType>(
            (__gm__ dstType *)dst.Data().Get(), (__cc__ srcType *)src.Data().Get(), Std::get<Is>(tupleParams)...);
    }

    template <QuantMode_t quantPre, typename T, typename U>
    __aicore__ inline void CopyMatrixCcToGm(__gm__ T *dst, __cc__ U *src, uint32_t nSize, uint32_t mSize,
        uint32_t srcStride, uint32_t dstStride, bool reluEn, uint8_t unitFlag, bool isChannelSplit, bool nz2ndEn)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
#if defined(ASCENDC_CPU_DEBUG)
            copy_matrix_cc_to_gm(dst, src, 0, nSize, mSize, dstStride, srcStride, unitFlag,
                quantPre, reluEn, isChannelSplit, nz2ndEn);
#else
            copy_matrix_cc_to_gm(dst, src, 0, nSize, mSize, dstStride, srcStride, unitFlag,
                static_cast<uint64_t>(quantPre), reluEn, isChannelSplit, nz2ndEn);
#endif
        }
    }
};

class SetRegister2201Base {
public:
    template <typename T, typename U>
    __aicore__ inline void SetRegister(const T& quant, const U& params)
    {
        SetQuantPre(quant);
        SetRegisterImpl<U>(params, tuple_sequence<decltype(params)>{});
    }
    template <typename T>
    __aicore__ inline void SetRegister(const T& params)
    {
        SetRegisterImpl<T>(params, tuple_sequence<decltype(params)>{});
    }

private:
    template <typename T, size_t... Is>
    __aicore__ inline void SetRegisterImpl(const T& tupleParams, Std::index_sequence<Is...>)
    {
        if constexpr (sizeof...(Is) == 0) {
            return;
        } else {
            SetParamsToRegister<uint64_t>(Std::get<Is>(tupleParams)...);
        }
    }

    template <typename T>
    __aicore__ inline void SetQuantPre(const T& quant)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            set_quant_pre(quant);
        }
    }

    template <typename T>
    __aicore__ inline void SetParamsToRegister(uint64_t ndNum, uint64_t dstNDStride, uint64_t srcNDStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            T ndPara = 0;
            ndPara = ndPara | (static_cast<T>(ndNum));
            ndPara = ndPara | (static_cast<T>(srcNDStride) << 16);
            ndPara = ndPara | (static_cast<T>(dstNDStride) << 32);
            set_nd_para(ndPara);
        }
    }
};

}
}

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_BASE_H