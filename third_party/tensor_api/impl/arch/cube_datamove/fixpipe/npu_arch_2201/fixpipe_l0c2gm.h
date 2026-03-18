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
 * \file fixpipe_l0c2gm.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_L0C2GM_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_L0C2GM_H

#include "impl/arch/utils/arch_utils.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_2201/fixpipe_base.h"

namespace AscendC {
namespace Te {

class FixpipetNz2Nz2201Base : public Copy2201MatrixCcToGmBase {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename Params>
    __aicore__ inline void Run(const T& dst, const U& src, const Params& inParams) {
        auto params = GenFixpipeParams<trait, T, U, Params>(dst, src, inParams);
        DataCopy<trait, quantPre, T, U, decltype(params)>(dst, src, params);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        CheckFormat::CheckL0CNZTemplate<T>();
        CheckFormat::CheckL0CNZTemplate<U>();
        CheckDataTypeFor2201::CheckL0c2GmDataType<T, U>();
    }

    template <const FixpipeTrait& trait, typename T, typename U, typename Params>
    __aicore__ inline auto GenFixpipeParams(const T& dst, const U& src, const Params& inParams)
    {
        CheckTemplate<trait, T, U>();
        using dstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout) * sizeof(dstType) / C0_SIZE;

        bool reluEn = false;
        uint8_t unitFlag = inParams.unitFlag;
        bool isChannelSplit = false;
        bool nz2ndEn = false;
        auto params = Std::make_tuple(nSize, mSize, srcStride, dstStride, reluEn, unitFlag, isChannelSplit, nz2ndEn);
        return params;
    }
};

class FixpipetNz2Nd2201Base : public Copy2201MatrixCcToGmBase, public SetRegister2201Base {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename Params>
    __aicore__ inline void Run(const T& dst, const U& src, const Params& inParams) {
        auto ndParams = GenRegisterParams<trait, T, U>(dst, src);
        SetRegister<decltype(ndParams)>(ndParams);
        auto params = GenFixpipeParams<trait, T, U, Params>(dst, src, inParams);

        DataCopy<trait, quantPre, T, U, decltype(params)>(dst, src, params);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        CheckFormat::CheckNDTemplate<T>();
        CheckFormat::CheckL0CNZTemplate<U>();
        CheckDataTypeFor2201::CheckL0c2GmDataType<T, U>();
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline auto GenRegisterParams(const T& dst, const U& src)
    {
        uint64_t ndNum = 1;
        uint64_t srcNdStride = 0;
        uint64_t dstNdStride = 0;
        auto params = Std::make_tuple(ndNum, dstNdStride, srcNdStride);
        return params;
    }
    
    template <const FixpipeTrait& trait, typename T, typename U, typename Params>
    __aicore__ inline auto GenFixpipeParams(const T& dst, const U& src, const Params& inParams)
    {
        CheckTemplate<trait, T, U>();
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);

        bool reluEn = false;
        uint8_t unitFlag = inParams.unitFlag;
        bool isChannelSplit = false;
        bool nz2ndEn = true;
        auto params = Std::make_tuple(nSize, mSize, srcStride, dstStride, reluEn, unitFlag, isChannelSplit, nz2ndEn);
        return params;
    }
};

class FixpipeFourDim2201L0C2GM : public FixpipetNz2Nz2201Base, public FixpipetNz2Nd2201Base {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename Params>
    __aicore__ inline void Run(const T& dst, const U& src, const Params& params) {
        Execute<trait>(dst, src, params);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U, typename Params>
    __aicore__ inline void Execute(const T& dst, const U& src, const Params& params) {
        constexpr auto quantPre = GetFixpipe2201QuantPre<trait, T, U>();
        if constexpr (IsL0cNZFormat<U>::value && IsL0cNZFormat<T>::value) {
            FixpipetNz2Nz2201Base::Run<trait, quantPre, T, U>(dst, src, params);
        } else if constexpr (IsL0cNZFormat<U>::value && IsNDFormat<T>::value) {
            FixpipetNz2Nd2201Base::Run<trait, quantPre, T, U>(dst, src, params);
        }
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_L0C2GM_H