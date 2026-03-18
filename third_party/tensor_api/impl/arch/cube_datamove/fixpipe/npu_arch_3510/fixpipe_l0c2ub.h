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
 * \file fixpipe_l0c2ub.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2UB_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2UB_H

#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_l0c2ub/nz2nz.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_l0c2ub/nz2nd.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_l0c2ub/nz2dn.h"

namespace AscendC {
namespace Te {

class FixpipeFourDimL0C2UB3510 {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename Params>
    __aicore__ inline void Run(const T& dst, const U& src, const Params& params) {
        Execute<trait>(dst, src, params);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U, typename Params>
    __aicore__ inline void Execute(const T& dst, const U& src, const Params& params) {
        constexpr auto quantPre = GetFixpipeQuantPre<trait, T, U>();
        if constexpr (IsL0cNZFormat<U>::value && IsNZFormat<T>::value) {
            Fixpipe2UbNz2NzBase3510 nz2NzStrategy;
            nz2NzStrategy.Run<trait, quantPre, T, U, Params>(dst, src, params);
        } else if constexpr (IsL0cNZFormat<U>::value && IsNDFormat<T>::value) {
            Fixpipe2UbNz2NdBase3510 nz2NdStrategy;
            nz2NdStrategy.Run<trait, quantPre, T, U, Params>(dst, src, params);
        } else if constexpr (IsL0cNZFormat<U>::value && IsDNFormat<T>::value) {
            Fixpipe2UbNz2DnBase3510 nz2DnStrategy;
            nz2DnStrategy.Run<trait, quantPre, T, U, Params>(dst, src, params);
        }
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2UB_H