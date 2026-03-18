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
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_L0C2GM_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_L0C2GM_H

#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_quant_l0c2gm/nz2dn.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_quant_l0c2gm/nz2nd.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_quant_l0c2gm/nz2nz.h"

namespace AscendC {
namespace Te {


template <Format3510 dstFormat, Format3510 srcFormat, QuantMode3510 QuantMode3510>
struct FormatRegistorFixpipe2Gm3510 {
    using type = FormatRegistorIgnore3510;
};

template <>
struct FormatRegistorFixpipe2Gm3510<Format3510::NZ, Format3510::NZ, QuantMode3510::Direct> {
    using type = Fixpipe2GmNZ2NZSimpleQuant3510;
};

template <>
struct FormatRegistorFixpipe2Gm3510<Format3510::ND, Format3510::NZ, QuantMode3510::Direct> {
    using type = Fixpipe2GmNZ2NDSimpleQuant3510;
};

template <>
struct FormatRegistorFixpipe2Gm3510<Format3510::DN, Format3510::NZ, QuantMode3510::Direct> {
    using type = Fixpipe2GmNZ2DNSimpleQuant3510;
};

template <>
struct FormatRegistorFixpipe2Gm3510<Format3510::NZ, Format3510::NZ, QuantMode3510::Scalar> {
    using type = Fixpipe2GmNZ2NZSimpleQuant3510;
};

template <>
struct FormatRegistorFixpipe2Gm3510<Format3510::ND, Format3510::NZ, QuantMode3510::Scalar> {
    using type = Fixpipe2GmNZ2NDSimpleQuant3510;
};

template <>
struct FormatRegistorFixpipe2Gm3510<Format3510::DN, Format3510::NZ, QuantMode3510::Scalar> {
    using type = Fixpipe2GmNZ2DNSimpleQuant3510;
};

template <>
struct FormatRegistorFixpipe2Gm3510<Format3510::NZ, Format3510::NZ, QuantMode3510::Vector> {
    using type = Fixpipe2GmNZ2NZVectorQuant3510;
};

template <>
struct FormatRegistorFixpipe2Gm3510<Format3510::ND, Format3510::NZ, QuantMode3510::Vector> {
    using type = Fixpipe2GmNZ2NDVectorQuant3510;
};

template <>
struct FormatRegistorFixpipe2Gm3510<Format3510::DN, Format3510::NZ, QuantMode3510::Vector> {
    using type = Fixpipe2GmNZ2DNVectorQuant3510;
};


class FixpipeQuantFourDimL0C2GM3510 {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename V, typename Params>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant, const Params& params) {
        Execute<trait>(dst, src, quant, params);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U, typename V, typename Params>
    __aicore__ inline void Execute(const T& dst, const U& src, const V& quant, const Params& params)
    {
        constexpr auto quantPre = GetFixpipeQuantPre<trait, T, U, V>();
        using FixpipeQuantL0C2GM =
            typename FormatRegistorFixpipe2Gm3510<GetDataFormat<T>(), GetDataFormat<U>(), GetQuantMode<quantPre>()>::type;
        FixpipeQuantL0C2GM{}.template Run<trait, quantPre, T, U, V>(dst, src, quant, params);
    }
};
}  // namespace Te
}  // namespace AscendC

#endif  // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_L0C2GM_H
