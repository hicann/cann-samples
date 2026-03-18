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
 * \file load_data_l12l0b.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0B_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0B_H

#include "impl/arch/cube_datamove/load_data/npu_arch_3510/load_data_l12l0b/zn2zn.h"
#include "impl/arch/cube_datamove/load_data/npu_arch_3510/load_data_l12l0b/nz2zn.h"
#include "impl/arch/cube_datamove/load_data/npu_arch_3510/load_data_l12l0b/nz2znb8b4.h"
#include "impl/arch/cube_datamove/load_data/npu_arch_3510/load_data_l12l0b/zn2zn_with_coord.h"
#include "impl/arch/cube_datamove/load_data/npu_arch_3510/load_data_l12l0b/nz2zn_with_coord.h"
#include "impl/arch/cube_datamove/load_data/npu_arch_3510/load_data_l12l0b/nz2znb8b4_with_coord.h"

namespace AscendC {
namespace Te {
class LoadDataFourDim3510L12L0B {
public:
    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline void Run(const T& dst, const U& src) {
        Execute<trait>(dst, src);
    }

private:
    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline void Execute(const T& dst, const U& src) {
        if constexpr (IsZNFormat<U>::value && IsZNFormat<T>::value) {
            LoadDataFourDim3510L12L0BZN2ZN zn2znStrategy;
            zn2znStrategy.Run<trait, T, U>(dst, src);
        } else if constexpr (IsNZFormat<U>::value && IsZNFormat<T>::value && (sizeof(typename U::elementType) == 1)) {
            LoadDataFourDim3510L12L0BNZ2ZNB8B4 nz2znb8Strategy;
            nz2znb8Strategy.Run<trait, T, U>(dst, src);
        } else if constexpr (IsNZFormat<U>::value && IsZNFormat<T>::value) {
            LoadDataFourDim3510L12L0BNZ2ZN nz2znStrategy;
            nz2znStrategy.Run<trait, T, U>(dst, src);
        }
    }
};

class LoadDataFourDim3510L12L0BWithCoord {
public:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        Execute<trait>(dst, src, coord);
    }

private:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline void Execute(const T& dst, const U& src, const Coord& coord) {
        if constexpr (IsZNFormat<U>::value && IsZNFormat<T>::value) {
            LoadDataFourDim3510L12L0BZN2ZNWithCoord zn2znStrategy;
            zn2znStrategy.Run<trait, T, U, Coord>(dst, src, coord);
        } else if constexpr (IsNZFormat<U>::value && IsZNFormat<T>::value && (sizeof(typename U::elementType) == 1)) {
            LoadDataFourDim3510L12L0BNZ2ZNB8B4WithCoord nz2znb8Strategy;
            nz2znb8Strategy.Run<trait, T, U, Coord>(dst, src, coord);
        } else if constexpr (IsNZFormat<U>::value && IsZNFormat<T>::value) {
            LoadDataFourDim3510L12L0BNZ2ZNWithCoord nz2znStrategy;
            nz2znStrategy.Run<trait, T, U, Coord>(dst, src, coord);
        }
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0B_H