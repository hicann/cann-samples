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
 * \file data_copy_gm2l1.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_2201_DATA_COPY_GM2L1_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_2201_DATA_COPY_GM2L1_H

#include "impl/tensor/pointer_impl.h"
#include "impl/tensor/local_tensor_impl.h"
#include "impl/arch/utils/arch_utils.h"

namespace AscendC {
namespace Te {

class CopyGmToCbufBase {
public:
    template <const DataCopyTrait& trait, typename T, typename U, typename V>
    __aicore__ inline void DataCopy(const T& dst, const U& src, const V& tupleParams) {
        DataCopyImpl<trait, T, U, V>(dst, src, tupleParams, tuple_sequence<V>{});
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        CopyGmToCbuf(dst.Data().Get(), src.Data().Get(), Std::get<Is>(tupleParams)...);
    }

    template <typename T>
    __aicore__ inline void CopyGmToCbuf(__cbuf__ T* dst, __gm__ T* src, uint32_t blockCount, uint32_t blockLen,
                                        int64_t srcStride, int64_t dstStride)
    {
        if ASCEND_IS_AIV {
            return;
        }

        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            copy_gm_to_cbuf(dst, src, 0, blockCount, blockLen, srcStride, dstStride, static_cast<pad_t>(0));
        }
    }
};

class CopyGmToCbufNZBase : public CopyGmToCbufBase {
public:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline void Run(const T& dst, const U& src) {

        auto params = GenDataCopyParams<trait, T, U>(dst, src);
        CopyGmToCbufBase::DataCopy<trait, T, U, decltype(params)>(dst, src, params);
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        CheckFormat::CheckNZTemplate<T>();
        CheckFormat::CheckNZTemplate<U>();
        CheckDataTypeFor2201::CheckGm2L1DataType<T, U>();
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        auto dstRow0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
        auto dstRow1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
        auto dstCol0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
        auto dstCol1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);

        auto srcStrideSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);
        using type = typename U::elementType;

        auto blockCount = dstCol1;
        auto blockLen = dstRow1 * dstRow0 * dstCol0 * sizeof(type);
        auto srcStride = srcStrideSize * sizeof(type);
        auto dstStride = blockLen;
        return Std::make_tuple(blockCount, blockLen, srcStride, dstStride);
    }
};

class CopyGmToCbufNDBase : public CopyGmToCbufBase {
public:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline void Run(const T& dst, const U& src) {
        auto params = GenDataCopyParams<trait, T, U>(dst, src);
        CopyGmToCbufBase::DataCopy<trait, T, U, decltype(params)>(dst, src, params);
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        CheckFormat::CheckNDTemplate<T>();
        CheckFormat::CheckNDTemplate<U>();
        CheckDataTypeFor2201::CheckGm2L1DataType<T, U>();
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
 
        auto dstShapeRows = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
        auto dstShapeColumns = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
        auto dstStrideRows = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);
        auto srcStrideRows = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
 
        using type = typename U::elementType;
 
        // considering block len is encoded num.
        auto blockCount = dstShapeRows;
        auto blockLen = dstShapeColumns * sizeof(type);
        auto srcStride = srcStrideRows * sizeof(type);
        auto dstStride = dstStrideRows * sizeof(type);
 
        return Std::make_tuple(blockCount, blockLen, srcStride, dstStride);
    }
};

class CopyGmToCbufMultiND2NZBase {
public:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline void Run(const T& dst, const U& src) {
        auto params = GenDataCopyParams<trait, T, U>(dst, src);
        DataCopyImpl<trait, T, U, decltype(params)>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        CheckFormat::CheckNDTemplate<U>();
        CheckFormat::CheckNZTemplate<T>();
        CheckDataTypeFor2201::CheckGm2L1DataType<T, U>();
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        using type = typename U::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        auto dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) * FRACTAL_FIXED;
        auto dstCol = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * C0_SIZE / sizeof(type);
        auto srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        auto srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

        auto srcRowStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout); 

        uint16_t ndNum = 1;
        uint64_t srcNdMatrixStride = 0;
        uint32_t dstNzMatrixStride = 0;
        uint16_t nValue = dstRow;
        uint16_t dValue = dstCol;
        uint16_t srcDValue = srcRowStride;
        uint16_t dstNzC0Stride = dstRow;
        uint16_t dstNzNStride = 1;
        return Std::make_tuple(ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                dstNzNStride, dstNzMatrixStride);
    }

    template <const DataCopyTrait& trait, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        CopyGmToCbufMultiNd2nz(dst.Data().Get(), src.Data().Get(), Std::get<Is>(tupleParams)...);
    }

    template <typename T>
    __aicore__ inline void CopyGmToCbufMultiNd2nz(__cbuf__ T* dst, __gm__ T* src,
            uint16_t ndNum, uint16_t nValue, uint16_t dValue, uint16_t srcNdMatrixStride, uint16_t srcDValue,
            uint16_t dstNzC0Stride, uint16_t dstNzNStride, uint16_t dstNzMatrixStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (sizeof(T) == 1) {
            copy_gm_to_cbuf_multi_nd2nz_b8(dst, src, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                dstNzNStride, dstNzMatrixStride);
        } else if constexpr (sizeof(T) == 2) {
            copy_gm_to_cbuf_multi_nd2nz_b16(dst, src, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                dstNzNStride, dstNzMatrixStride);
        } else if constexpr (sizeof(T) == 4) {
            copy_gm_to_cbuf_multi_nd2nz_b32s(dst, src, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
                dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
        }
    }
};

class CopyGmToCbufMultiDN2ZNBase {
public:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline void Run(const T& dst, const U& src) {
        auto params = GenDataCopyParams<trait, T, U>(dst, src);
        DataCopyImpl<trait, T, U, decltype(params)>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        CheckFormat::CheckDNTemplate<U>();
        CheckFormat::CheckZNTemplate<T>();
        CheckDataTypeFor2201::CheckGm2L1DataType<T, U>();
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        using type = typename U::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        auto dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) * C0_SIZE / sizeof(type);
        auto dstCol = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * FRACTAL_FIXED;
        auto srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        auto srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

        uint16_t ndNum = 1;
        uint64_t srcNdMatrixStride = 0;
        uint32_t dstNzMatrixStride = 0;
        uint16_t nValue = dstCol;
        uint16_t dValue = dstRow;
        uint16_t srcDValue = srcRow;
        uint16_t dstNzC0Stride = dstCol;
        uint16_t dstNzNStride = 1;

        return Std::make_tuple(ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                dstNzNStride, dstNzMatrixStride);
    }

    template <const DataCopyTrait& trait, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        CopyGmToCbufMultiNd2nz(dst.Data().Get(), src.Data().Get(), Std::get<Is>(tupleParams)...);
    }

    template <typename T>
    __aicore__ inline void CopyGmToCbufMultiNd2nz(__cbuf__ T* dst, __gm__ T* src,
            uint16_t ndNum, uint16_t nValue, uint16_t dValue, uint16_t srcNdMatrixStride, uint16_t srcDValue,
            uint16_t dstNzC0Stride, uint16_t dstNzNStride, uint16_t dstNzMatrixStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (sizeof(T) == 1) {
            copy_gm_to_cbuf_multi_nd2nz_b8(dst, src, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                dstNzNStride, dstNzMatrixStride);
        } else if constexpr (sizeof(T) == 2) {
            copy_gm_to_cbuf_multi_nd2nz_b16(dst, src, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                dstNzNStride, dstNzMatrixStride);
        } else if constexpr (sizeof(T) == 4) {
            copy_gm_to_cbuf_multi_nd2nz_b32s(dst, src, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
                dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
        }
    }
};

class DataCopyGM2L12201 : public CopyGmToCbufMultiND2NZBase, public CopyGmToCbufMultiDN2ZNBase,
    public CopyGmToCbufNZBase, public CopyGmToCbufNDBase {
public:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline void Run(const T& dst, const U& src) {
        Execute<trait>(dst, src);
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline void Execute(const T& dst, const U& src) {
        if constexpr (IsNZFormat<U>::value && IsNZFormat<T>::value) {
            CopyGmToCbufNZBase::Run<trait, T, U>(dst, src);
        } else if constexpr (IsNDFormat<U>::value && IsNZFormat<T>::value) {
            CopyGmToCbufMultiND2NZBase::Run<trait, T, U>(dst, src);
        } else if constexpr (IsDNFormat<U>::value && IsZNFormat<T>::value) {
            CopyGmToCbufMultiDN2ZNBase::Run<trait, T, U>(dst, src);
        } else if constexpr (IsNDFormat<U>::value && IsNDFormat<T>::value) {
            CopyGmToCbufNDBase::Run<trait, T, U>(dst, src);
        }
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_2201_DATA_COPY_GM2L1_H