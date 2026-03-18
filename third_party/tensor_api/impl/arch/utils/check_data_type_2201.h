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
 * \file check_data_type_2201.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_UTILS_CHECK_DATA_TYPE_2201_H
#define IMPL_TENSOR_API_ARCH_UTILS_CHECK_DATA_TYPE_2201_H

#include "impl/utils/utils_impl.h"

namespace AscendC {
namespace Te {

class CheckDataTypeFor2201 {
public:
    template <typename T, typename U, typename S>
    __aicore__ inline static constexpr void CheckMmadDataType()
    {
        using dstDataType = typename T::elementType;
        using fmDataType = typename U::elementType;
        using filterDataType = typename S::elementType;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        static_assert(Std::is_one_of_v<Std::tuple<dstDataType, fmDataType, filterDataType>,
                                       Std::tuple<__cc__ int32_t, __ca__ int8_t, __cb__ int8_t>,
                                       Std::tuple<__cc__ float, __ca__ half, __cb__ half>,
                                       Std::tuple<__cc__ float, __ca__ bfloat16_t, __cb__ bfloat16_t>,
                                       Std::tuple<__cc__ float, __ca__ float, __cb__ float>>,
                      "The data type is not supported.");
#endif
    }

    template <typename T, typename U, typename S, typename V>
    __aicore__ inline static constexpr void CheckMmadBiasDataType()
    {
        using fmDataType = typename U::elementType;
        using filterDataType = typename S::elementType;
        using biasDataType = typename V::elementType;
        using dstDataType = typename T::elementType;
        constexpr auto biasPos = GetHardPos<V>();

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        static_assert(
            Std::is_one_of_v<Std::tuple<biasDataType, dstDataType, fmDataType, filterDataType>,
                             Std::tuple<__biasbuf__ int32_t, __cc__ int32_t, __ca__ int8_t, __cb__ int8_t>,
                             Std::tuple<__biasbuf__ float, __cc__ float, __ca__ half, __cb__ half>,
                             Std::tuple<__biasbuf__ float, __cc__ float, __ca__ bfloat16_t, __cb__ bfloat16_t>,
                             Std::tuple<__biasbuf__ float, __cc__ float, __ca__ float, __cb__ float>>,
            "The data type is not supported.");
#endif
    }

    template <typename T, typename U>
    __aicore__ inline static constexpr void CheckGm2L1DataType()
    {
        using srcDataType = typename U::elementType;
        using dstDataType = typename T::elementType;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        static_assert(
            Std::is_one_of_v<Std::tuple<dstDataType, srcDataType>, Std::tuple<__cbuf__ bfloat16_t, __gm__ bfloat16_t>,
                             Std::tuple<__cbuf__ half, __gm__ half>, Std::tuple<__cbuf__ float, __gm__ float>,
                             Std::tuple<__cbuf__ int16_t, __gm__ int16_t>, Std::tuple<__cbuf__ int32_t, __gm__ int32_t>,
                             Std::tuple<__cbuf__ int8_t, __gm__ int8_t>, Std::tuple<__cbuf__ uint16_t, __gm__ uint16_t>,
                             Std::tuple<__cbuf__ uint32_t, __gm__ uint32_t>,
                             Std::tuple<__cbuf__ uint8_t, __gm__ uint8_t>>,
            "The data type is not supported.");
#endif
    }

    template <typename T, typename U>
    __aicore__ inline static constexpr void CheckGm2L1NDDataType()
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, Std::tuple<__cbuf__ bfloat16_t, __gm__ bfloat16_t>,
                                       Std::tuple<__cbuf__ half, __gm__ half>, Std::tuple<__cbuf__ float, __gm__ float>,
                                       Std::tuple<__cbuf__ int32_t, __gm__ int32_t>>,
                      "The data type is not supported.");
#endif
    }

    template <typename T, typename U>
    __aicore__ inline static constexpr void CheckL12BtDataType()
    {
        using dstDataType = typename T::elementType;
        using srcDataType = typename U::elementType;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        static_assert(
            Std::is_one_of_v<Std::tuple<dstDataType, srcDataType>, Std::tuple<__cbuf__ bfloat16_t, __gm__ bfloat16_t>,
                             Std::tuple<__cbuf__ half, __gm__ half>, Std::tuple<__cbuf__ float, __gm__ float>,
                             Std::tuple<__cbuf__ int16_t, __gm__ int16_t>, Std::tuple<__cbuf__ int32_t, __gm__ int32_t>,
                             Std::tuple<__cbuf__ int8_t, __gm__ int8_t>, Std::tuple<__cbuf__ uint16_t, __gm__ uint16_t>,
                             Std::tuple<__cbuf__ uint32_t, __gm__ uint32_t>,
                             Std::tuple<__cbuf__ uint8_t, __gm__ uint8_t>>,
            "The data type is not supported.");
#endif
    }

    template <typename T, typename U>
    __aicore__ inline static constexpr void CheckL12FbDataType()
    {
        using srcDataType = typename U::elementType;
        using dstDataType = typename T::elementType;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        static_assert(
            Std::is_one_of_v<
                Std::tuple<dstDataType, srcDataType>, Std::tuple<__fbuf__ bfloat16_t, __cbuf__ bfloat16_t>,
                Std::tuple<__fbuf__ half, __cbuf__ half>, Std::tuple<__fbuf__ float, __cbuf__ float>,
                Std::tuple<__fbuf__ int16_t, __cbuf__ int16_t>, Std::tuple<__fbuf__ int32_t, __cbuf__ int32_t>,
                Std::tuple<__fbuf__ int8_t, __cbuf__ int8_t>, Std::tuple<__fbuf__ uint16_t, __cbuf__ uint16_t>,
                Std::tuple<__fbuf__ uint32_t, __cbuf__ uint32_t>, Std::tuple<__fbuf__ uint8_t, __cbuf__ uint8_t>,
                Std::tuple<__fbuf__ uint64_t, __cbuf__ uint64_t>, Std::tuple<__fbuf__ int64_t, __cbuf__ int64_t>,
                Std::tuple<__fbuf__ double, __cbuf__ double>>,
            "The data type is not supported.");
#endif
    }

    template <typename U>
    __aicore__ inline static constexpr void CheckFixPipeDataType()
    {
        using srcDataType = typename U::elementType;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        static_assert(Std::is_one_of_v<srcDataType, __cbuf__ uint64_t>, "The source data type is not supported.");
#endif
    }

    template <typename T, typename U>
    __aicore__ inline static constexpr void CheckL0c2GmDataType()
    {
        using srcDataType = typename U::elementType;
        using dstDataType = typename T::elementType;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        static_assert(Std::is_one_of_v<Std::tuple<dstDataType, srcDataType>, Std::tuple<__gm__ float, __cc__ float>,
                                       Std::tuple<__gm__ int32_t, __cc__ int32_t>>,
                      "The data type is not supported.");
#endif
    }

    template <typename T, typename U>
    __aicore__ inline static constexpr void CheckL12L0ADataType()
    {
        using srcDataType = typename U::elementType;
        using dstDataType = typename T::elementType;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        static_assert(
            Std::is_one_of_v<Std::tuple<dstDataType, srcDataType>, Std::tuple<__ca__ bfloat16_t, __cbuf__ bfloat16_t>,
                             Std::tuple<__ca__ half, __cbuf__ half>, Std::tuple<__ca__ float, __cbuf__ float>,
                             Std::tuple<__ca__ int16_t, __cbuf__ int16_t>, Std::tuple<__ca__ int32_t, __cbuf__ int32_t>,
                             Std::tuple<__ca__ int8_t, __cbuf__ int8_t>, Std::tuple<__ca__ uint16_t, __cbuf__ uint16_t>,
                             Std::tuple<__ca__ uint32_t, __cbuf__ uint32_t>,
                             Std::tuple<__ca__ uint8_t, __cbuf__ uint8_t>>,
            "The data type is not supported.");
#endif
    }

    template <typename T, typename U>
    __aicore__ inline static constexpr void CheckL12L0BDataType()
    {
        using srcDataType = typename U::elementType;
        using dstDataType = typename T::elementType;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        static_assert(
            Std::is_one_of_v<Std::tuple<dstDataType, srcDataType>, Std::tuple<__cb__ bfloat16_t, __cbuf__ bfloat16_t>,
                             Std::tuple<__cb__ half, __cbuf__ half>, Std::tuple<__cb__ float, __cbuf__ float>,
                             Std::tuple<__cb__ int16_t, __cbuf__ int16_t>, Std::tuple<__cb__ int32_t, __cbuf__ int32_t>,
                             Std::tuple<__cb__ int8_t, __cbuf__ int8_t>, Std::tuple<__cb__ uint16_t, __cbuf__ uint16_t>,
                             Std::tuple<__cb__ uint32_t, __cbuf__ uint32_t>,
                             Std::tuple<__cb__ uint8_t, __cbuf__ uint8_t>>,
            "The data type is not supported.");
#endif
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_UTILS_CHECK_DATA_TYPE_2201_H