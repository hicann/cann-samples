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
 * \file tile_mmad_mx.h
 * \brief Tile-level MMAD traits used by grouped MX kernels.
 */
#ifndef GROUPED_MATMUL_RECIPE_TILE_MMAD_MX_H
#define GROUPED_MATMUL_RECIPE_TILE_MMAD_MX_H

#include "include/tensor.h"

namespace Tile {

struct MmadMx {
    template <typename Tp, const Tp& traits, typename T, typename U, typename S>
    __aicore__ inline static void Mad(
        const T& dst, const U& fm, const S& filter, uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlagCtrl,
        bool btBuffCtrl, bool initCMatrixCtrl)
    {
        mad_mx(
            dst.Data().Get(), fm.Data().Get(), filter.Data().Get(), m, k, n, unitFlagCtrl, true, btBuffCtrl,
            initCMatrixCtrl);
    }
};

struct MmadMxWithBias {
    template <typename Tp, const Tp& traits, typename T, typename U, typename S, typename V>
    __aicore__ inline static void Mad(
        const T& dst, const U& fm, const S& filter, const V& bias, uint16_t m, uint16_t k, uint16_t n,
        uint8_t unitFlagCtrl, bool btBuffCtrl, bool initCMatrixCtrl)
    {
        using dstType = typename T::elementType;
        uint64_t biasAddr = reinterpret_cast<uint64_t>(bias.Data().Get());
        uint64_t cAddr = reinterpret_cast<uint64_t>(dst.Data().Get());
        uint64_t xd = (cAddr) & 0xffffffffULL | ((biasAddr & 0xffffffffULL) << 32);
        mad_mx(
            (dstType*)xd, fm.Data().Get(), filter.Data().Get(), m, k, n, unitFlagCtrl, true, btBuffCtrl,
            initCMatrixCtrl);
    }
};

} // namespace Tile

namespace AscendC {
namespace Te {

template <typename Opration, typename TraitStruct>
struct MmadTraits<Opration, TraitStruct> {
    using TraitType = typename TraitStruct::TraitType;
    static constexpr const TraitType defaultTrait = TraitStruct::value;

    template <const TraitType& trait = defaultTrait, typename... Args>
    __aicore__ inline void MmadUnpack(const Args&... args) const
    {
        Opration::template Mad<TraitType, trait, Args...>(args..., m, k, n, unitFlagCtrl, btBuffCtrl, initCMatrixCtrl);
    }

    uint16_t m = 0;
    uint16_t k = 0;
    uint16_t n = 0;
    uint8_t unitFlagCtrl = 0;
    bool btBuffCtrl = false;
    bool initCMatrixCtrl = false;
};

template <>
struct MmadTraits<::Tile::MmadMx>
    : public MmadTraits<::Tile::MmadMx, MmadTraitDefault, ::Tile::MmadMx, MmadTraitDefault> {};

template <>
struct MmadTraits<::Tile::MmadMxWithBias>
    : public MmadTraits<
        ::Tile::MmadMxWithBias, MmadTraitDefault, ::Tile::MmadMxWithBias, MmadTraitDefault> {};

} // namespace Te
} // namespace AscendC

#endif
