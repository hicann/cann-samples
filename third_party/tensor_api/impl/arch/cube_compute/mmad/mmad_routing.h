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
 * \file mmad_routing.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_MMAD_ROUTING_H
#define IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_MMAD_ROUTING_H

#include "impl/arch/cube_compute/mmad/npu_arch_2201/mmad_details.h"
#include "impl/arch/cube_compute/mmad/npu_arch_2201/mmad_with_bias_details.h"

#include "impl/arch/cube_compute/mmad/npu_arch_3510/mmad.h"
#include "impl/arch/cube_compute/mmad/npu_arch_3510/mmad_with_bias.h"

namespace AscendC {
namespace Te {

class MmadIgnore
{
public:
    template <const MmadTrait& trait, typename ...Args>
    __aicore__ inline void Run(const Args&... args) {}
};

template<Hardware dstPos, Hardware fmPos, Hardware filterPos, Hardware biasPos, uint32_t Version>
struct MmadTensor2Tensor
{
    using type = MmadIgnore;
};

template<>
struct MmadTensor2Tensor<Hardware::L0C, Hardware::L0A, Hardware::L0B, Hardware::MAX, ArchVersion::V2201>
{
    using type = Mmad2201;
};

template<>
struct MmadTensor2Tensor<Hardware::L0C, Hardware::L0A, Hardware::L0B, Hardware::L0C, ArchVersion::V2201>
{
    using type = MmadWithBias2201;
};

template<>
struct MmadTensor2Tensor<Hardware::L0C, Hardware::L0A, Hardware::L0B, Hardware::BIAS, ArchVersion::V2201>
{
    using type = MmadWithBias2201;
};

template<>
struct MmadTensor2Tensor<Hardware::L0C, Hardware::L0A, Hardware::L0B, Hardware::MAX, ArchVersion::V3510>
{
    using type = Mmad3510;
};

template<>
struct MmadTensor2Tensor<Hardware::L0C, Hardware::L0A, Hardware::L0B, Hardware::L0C, ArchVersion::V3510>
{
    using type = MmadWithBias3510;
};

template<>
struct MmadTensor2Tensor<Hardware::L0C, Hardware::L0A, Hardware::L0B, Hardware::BIAS, ArchVersion::V3510>
{
    using type = MmadWithBias3510;
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_MMAD_ROUTING_H