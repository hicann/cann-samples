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
 * \file layout_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_IMPL_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_IMPL_H

#include "impl/utils/utils_impl.h"
#include "impl/tensor/layout_method.h"
#include "impl/tensor/coord_index.h"
#include "impl/tensor/layout_fractal.h"

namespace AscendC {
namespace Te {

template <typename T>
struct LocalTensor;

template <typename Coord, typename LayoutType, typename TileShape>
__aicore__ inline decltype(auto) MakeTileLayout(const Coord& coord, const LayoutType& layout, const TileShape& tileShape) 
{
    static_assert(Std::is_tuple_v<TileShape>);

    using OriginShape = Std::remove_cvref_t<decltype(layout.Shape())>;
    if constexpr (nesting_depth_v<TileShape> == nesting_depth_v<OriginShape>
                  && Std::tuple_size_v<TileShape> == Std::tuple_size_v<OriginShape>) {
        return MakeLayout(tileShape, layout.Stride());
    } else {
        static_assert(Std::tuple_size_v<TileShape> == TWO_DIM_DATA);

        const uint32_t rows = Std::get<0>(tileShape);
        const uint32_t cols = Std::get<1>(tileShape);

        const auto& innerRow = Std::get<0>(Std::get<0>(layout.Shape()));
        const auto& innerCol = Std::get<0>(Std::get<1>(layout.Shape()));

        using InnerRowType = Std::remove_cvref_t<decltype(innerRow)>;
        using InnerColType = Std::remove_cvref_t<decltype(innerCol)>;

        if constexpr (IsIntegralConstantV<InnerRowType> && IsIntegralConstantV<InnerColType>) {
            return MakeLayout(
                MakeShape(MakeShape(Std::Int<InnerRowType::value>{}, CeilDivision(rows, InnerRowType::value)),
                          MakeShape(Std::Int<InnerColType::value>{}, CeilDivision(cols, InnerColType::value))),
                layout.Stride());
        } else {
            return MakeLayout(
                MakeShape(MakeShape(innerRow, CeilDivision(rows, innerRow)),
                          MakeShape(innerCol, CeilDivision(cols, innerCol))),
                layout.Stride());
        }
    }
}

template <typename Coord, typename LayoutType, typename TensorType>
__aicore__ inline decltype(auto) MakeTileLayout(const Coord& coord, const LayoutType& layout, const LocalTensor<TensorType>& tileTensor) 
{
    static_assert(tileTensor.rank == LayoutType::rank, "tensor rank is not equal layout");

    auto innerRow = Std::get<0>(Std::get<0>(layout.Shape()));
    auto innerCol = Std::get<0>(Std::get<1>(layout.Shape()));

    auto srcRow = innerRow * Std::get<1>(Std::get<0>(layout.Shape())) - Std::get<0>(coord);
    auto srcCol = innerCol * Std::get<1>(Std::get<1>(layout.Shape())) - Std::get<1>(coord);

    auto dstRow = Std::get<0>(Std::get<0>(tileTensor.Shape())) * Std::get<1>(Std::get<0>(tileTensor.Shape()));
    auto dstCol = Std::get<0>(Std::get<1>(tileTensor.Shape())) * Std::get<1>(Std::get<1>(tileTensor.Shape()));

    auto realRow = Min(srcRow, dstRow);
    auto realCol = Min(srcCol, dstCol);

    auto row = MakeShape(innerRow, CeilDivision(realRow, innerRow));
    auto col = MakeShape(innerCol, CeilDivision(realCol, innerCol));

    return MakeTileLayout(coord, layout, MakeShape(row, col));
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_MAKE_LAYOUT_IMPL_H
