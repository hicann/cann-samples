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
 * \file fixpipe_routing.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_FIXPIPE_ROUTING_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_FIXPIPE_ROUTING_H

#include "impl/arch/cube_datamove/fixpipe/npu_arch_2201/fixpipe_l0c2gm.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_2201/fixpipe_quant_l0c2gm.h"

#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_l0c2gm.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_l0c2ub.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_quant_l0c2gm.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_quant_l0c2ub.h"

namespace AscendC {
namespace Te {

class FixpipeIgnore {
public:
    template <const FixpipeTrait& trait, typename ...Args>
    __aicore__ inline void Run(const Args&... args) {}
};

template <Hardware dstPos, Hardware srcpos, Hardware quantpos, uint32_t Version, size_t dimension>
struct FixpipeTensor2Tensor {
    using type = FixpipeIgnore;
};

template <>
struct FixpipeTensor2Tensor<Hardware::GM, Hardware::L0C, Hardware::MAX, ArchVersion::V2201, FOUR_DIM_DATA> {
    using type = FixpipeFourDim2201L0C2GM;
};

template <>
struct FixpipeTensor2Tensor<Hardware::GM, Hardware::L0C, Hardware::L1, ArchVersion::V2201, FOUR_DIM_DATA> {
    using type = FixpipeQuantFourDim2201L0C2GM;
};

template <>
struct FixpipeTensor2Tensor<Hardware::GM, Hardware::L0C, Hardware::MAX, ArchVersion::V3510, FOUR_DIM_DATA> {
    using type = FixpipeFourDimL0C2GM3510;
};

template <>
struct FixpipeTensor2Tensor<Hardware::GM, Hardware::L0C, Hardware::L1, ArchVersion::V3510, FOUR_DIM_DATA> {
    using type = FixpipeQuantFourDimL0C2GM3510;
};

// L0C to UB routing for V3510
template <>
struct FixpipeTensor2Tensor<Hardware::UB, Hardware::L0C, Hardware::MAX, ArchVersion::V3510, FOUR_DIM_DATA> {
    using type = FixpipeFourDimL0C2UB3510;
};

template <>
struct FixpipeTensor2Tensor<Hardware::UB, Hardware::L0C, Hardware::L1, ArchVersion::V3510, FOUR_DIM_DATA> {
    using type = FixpipeQuantFourDimL0C2UB3510;
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_FIXPIPE_ROUTING_H