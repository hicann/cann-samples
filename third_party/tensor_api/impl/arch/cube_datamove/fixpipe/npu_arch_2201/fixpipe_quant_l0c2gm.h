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
 * \file fixpipe_quant_l0c2gm.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_QUANT_L0C2GM_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_QUANT_L0C2GM_H

#include "impl/arch/cube_datamove/fixpipe/npu_arch_2201/fixpipe_quant_nz2nd_l0c2gm.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_2201/fixpipe_quant_nz2nz_l0c2gm.h"

namespace AscendC {
namespace Te {

enum class Format2201 : uint8_t { None, NZ, ND};
enum class QuantMode2201 : uint8_t { None, Scalar, Vector, Direct };

template <typename T>
__aicore__ inline constexpr Format2201 GetDataFormat2201()
{
    if constexpr (IsL0cNZFormat<T>::value) {
        return Format2201::NZ;
    } else if constexpr (IsNDFormat<T>::value) {
        return Format2201::ND;
    }
    return Format2201::None;
}

template <const QuantMode_t quantPre>
__aicore__ inline constexpr QuantMode2201 GetQuantMode2201()
{
    if constexpr (IsVectorQuantMode<quantPre>()) {
        return QuantMode2201::Vector;
    } else if constexpr (IsScalarQuantMode<quantPre>()) {
        return QuantMode2201::Scalar;
    } else if constexpr (IsDirectQuantMode<quantPre>()) {
        return QuantMode2201::Direct;
    }
    return QuantMode2201::None;
}

class Format2201RegistorIgnore {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename S, typename Params>
    __aicore__ inline void Run(const T& dst, const U& src, const S& quant, const Params& params) {}
};

template <Format2201 dstFormat, Format2201 srcFormat, QuantMode2201 quantMode>
struct Format2201Registor {
    using type = Format2201RegistorIgnore;
};

template <>
struct Format2201Registor<Format2201::NZ, Format2201::NZ, QuantMode2201::Direct> {
    using type = FixpipeNZ2NZ2201SimpleQuant;
};

template <>
struct Format2201Registor<Format2201::ND, Format2201::NZ, QuantMode2201::Direct> {
    using type = FixpipeNZ2ND2201SimpleQuant;
};

template <>
struct Format2201Registor<Format2201::NZ, Format2201::NZ, QuantMode2201::Scalar> {
    using type = FixpipeNZ2NZ2201SimpleQuant;
};

template <>
struct Format2201Registor<Format2201::ND, Format2201::NZ, QuantMode2201::Scalar> {
    using type = FixpipeNZ2ND2201SimpleQuant;
};

template <>
struct Format2201Registor<Format2201::NZ, Format2201::NZ, QuantMode2201::Vector> {
    using type = FixpipeNZ2NZ2201VectorQuant;
};

template <>
struct Format2201Registor<Format2201::ND, Format2201::NZ, QuantMode2201::Vector> {
    using type = FixpipeNZ2ND2201VectorQuant;
};

class FixpipeQuantFourDim2201L0C2GM {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename S, typename Params>
    __aicore__ inline void Run(const T& dst, const U& src, const S& quant, const Params& params)
    {
        Execute<trait>(dst, src, quant, params);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U, typename S, typename Params>
    __aicore__ inline void Execute(const T& dst, const U& src, const S& quant, const Params& params)
    {
        constexpr auto quantPre = GetFixpipe2201QuantPre<trait, T, U, S>();
        using FixpipeQuantCoordL0C2GM =
            typename Format2201Registor<GetDataFormat2201<T>(), GetDataFormat2201<U>(), GetQuantMode2201<quantPre>()>::type;
        FixpipeQuantCoordL0C2GM{}.template Run<trait, quantPre, T, U, S>(dst, src, quant, params);
    }
};
}  // namespace Te
}  // namespace AscendC

#endif  // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_QUANT_L0C2GM_H
