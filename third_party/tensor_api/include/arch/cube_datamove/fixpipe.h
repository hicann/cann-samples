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
 * \file fixpipe.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_H
#define INCLUDE_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_H

#include "impl/arch/cube_datamove/fixpipe/fixpipe_impl.h"

namespace AscendC {
namespace Te {

template <const FixpipeTrait& trait, typename T, typename U>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeTemplate<T, U>, void>::type
Fixpipe(const T& dst, const U& src, const FixpipeParams& params);

template <const FixpipeTrait& trait, typename T, typename U, typename V>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeQuantTemplate<T, U, V>, void>::type
Fixpipe(const T& dst, const U& src, const V& quant, const FixpipeParams& params);

template <const FixpipeTrait& trait, typename T, typename U, typename Coord>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeTemplateWithCoord<T, U, Coord>, void>::type
Fixpipe(const T& dst, const U& src, const Coord& coord, const FixpipeParams& params);

template <const FixpipeTrait& trait, typename T, typename U, typename V, typename Coord>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeQuantTemplateWithCoord<T, U, V, Coord>, void>::type
Fixpipe(const T& dst, const U& src, const V& quant, const Coord& coord, const FixpipeParams& params);

} // namespace Te
} // namespace AscendC

#endif // INCLUDE_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_H