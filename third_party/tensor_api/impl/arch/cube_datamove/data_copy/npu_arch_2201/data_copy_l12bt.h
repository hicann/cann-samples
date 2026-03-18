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
 * \file data_copy_l12bt.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_2201_DATA_COPY_L12BT_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_2201_DATA_COPY_L12BT_H

#include "impl/tensor/pointer_impl.h"
#include "impl/tensor/local_tensor_impl.h"
#include "impl/arch/utils/arch_utils.h"

namespace AscendC {
namespace Te {

class CopyCbufToBT2201 {
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
        CheckFormat::CheckNDTemplate<T>();
        CheckFormat::CheckNDTemplate<U>();
        CheckDataTypeFor2201::CheckL12BtDataType<T, U>();
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        constexpr auto L12BT_UNIT = C0_SIZE * 2;
        CheckTemplate<trait, T, U>();

        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        uint16_t dstCol = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
        uint16_t srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
        uint16_t dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);

        using srcType = typename U::elementType;
        using dstType = typename T::elementType;

        bool convControl = false;
        uint16_t blockCount = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
        uint16_t blockLen = dstCol * sizeof(dstType) / L12BT_UNIT;
        uint16_t srcStride = (srcRow - dstCol) * sizeof(srcType) / C0_SIZE;
        uint16_t dstStride = (dstRow - dstCol) * sizeof(dstType) / L12BT_UNIT;

        return Std::make_tuple(convControl, blockCount, blockLen, srcStride, dstStride);
    }

    template <const DataCopyTrait& trait, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        CopyCbufToBt(reinterpret_cast<uint64_t>(dst.Data().Get()), src.Data().Get(), Std::get<Is>(tupleParams)...);
    }

    template <typename T>
    __aicore__ inline void CopyCbufToBt(uint64_t dst, __cbuf__ T* src, bool convControl, uint16_t blockCount, uint16_t blockLen,
        uint16_t srcStride, uint16_t dstStride)
    {
        if ASCEND_IS_AIV {
            return;
        }

        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            copy_cbuf_to_bt(dst, src, convControl, blockCount, blockLen, srcStride, dstStride);
        }
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_2201_DATA_COPY_L12BT_H