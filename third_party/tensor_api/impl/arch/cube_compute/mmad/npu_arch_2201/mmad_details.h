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
 * \file mmad_details.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_2201_MMAD_DETAILS_H
#define IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_2201_MMAD_DETAILS_H

#include "impl/tensor/pointer_impl.h"
#include "impl/tensor/local_tensor_impl.h"
#include "impl/arch/utils/arch_utils.h"

namespace AscendC {
namespace Te {

class MmadGenParams2201
{
public:
    template <const MmadTrait& trait, typename T, typename U, typename S, typename Params>
    __aicore__ inline auto GenParams(const T& dst, const U& fm, const S& filter, const Params& params)
    {
        return GenParamsImpl<trait, T, U, S, Params>(dst, fm, filter, params);
    }
private:
    template <const MmadTrait& trait, typename T, typename U, typename S>
    __aicore__ inline constexpr void CheckTemplate()
    {
        CheckFormat::CheckL0CNZTemplate<T>();
        CheckFormat::CheckZZTemplate<U>();
        CheckFormat::CheckZNTemplate<S>();
        CheckDataTypeFor2201::CheckMmadDataType<T, U, S>();
    }

    template <const MmadTrait& trait, typename T, typename U, typename S, typename Params>
    __aicore__ inline auto GenParamsImpl(const T& dst, const U& fm, const S& filter, const Params& params)
    {
        CheckTemplate<trait, T, U, S>();
        using fmType = typename U::elementType;
        auto fmLayout = fm.Layout();
        auto dstLayout = dst.Layout();

        uint16_t m = GetEleFromLayout<decltype(fmLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(fmLayout) * FRACTAL_FIXED;
        uint16_t k = GetEleFromLayout<decltype(fmLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(fmLayout) * C0_SIZE / sizeof(fmType);
        uint16_t n = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * FRACTAL_FIXED;
        auto genParams = Std::make_tuple(m, k, n, params.unitFlag, trait.kDirectionAlign, trait.cmatrixSource, params.cmatrixInitVal);
        return genParams;
    }
};

class MmadCore2201
{
public:
    template <const MmadTrait& trait, typename T, typename U, typename S, typename Params, typename V, size_t... Is>
    __aicore__ inline void Mmad(const T& dst, const U& fm, const S& filter, const Params& params, const V& tupleParams, Std::index_sequence<Is...>)
    {
        // MTE2
        MmadImpl(dst.Data().Get(), fm.Data().Get(), filter.Data().Get(), Std::get<Is>(tupleParams)...);
    }
private:
    template <typename T, typename U, typename S>
    __aicore__ inline void MmadImpl(__cc__ T* dst, __ca__ U* fm, __cb__ S* filter, uint16_t m, uint16_t k, uint16_t n,
        int8_t unitFlag, bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal) {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            mad(dst, fm, filter, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, cmatrixInitVal);
        }
    }
};

class Mmad2201 : public MmadCore2201, public MmadGenParams2201
{
public:
    template <const MmadTrait& trait, typename ...Args>
    __aicore__ inline void Run(const Args&... args) 
    {
        auto params = GenParams<trait>(args...);
        Mmad<trait>(args..., params, tuple_sequence<decltype(params)>{});
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_2201_MMAD_DETAILS_H