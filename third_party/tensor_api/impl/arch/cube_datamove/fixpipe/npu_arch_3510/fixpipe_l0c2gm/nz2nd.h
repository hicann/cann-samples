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
 * \file nz2nd.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2GM_NZ2ND_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2GM_NZ2ND_H

#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class Fixpipe2GmNz2NdBase3510 {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename Params>
    __aicore__ inline void Run(const T& dst, const U& src, const Params& params) {
        SetRegisterImpl<trait, T, U>(dst, src);
        DataCopyImpl<trait, quantPre, T, U, Params>(dst, src, params);
    }

private:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        CheckFormat::CheckNDTemplate<T>();
        CheckFormat::CheckL0CNZTemplate<U>();
        CheckDataTypeFor3510::CheckL0C2GmDataType<quantPre, T, U>();
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline void SetRegisterImpl(const T& dst, const U& src)
    {
        uint32_t ndNum = 1;
        uint32_t srcNdStride = 0;
        uint32_t dstNdStride = 0;
        SetRegisterBase3510 setRegisterInst;
        setRegisterInst.SetRegister(ndNum, dstNdStride, srcNdStride);
    }

    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename Params>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const Params& params)
    {
        CheckTemplate<trait, quantPre, T, U>();
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = Std::min(GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout),
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout) *
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout));
        uint32_t mSize = Std::min(GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout),
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout) *
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout));
        uint32_t srcStride =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);
        uint8_t cacheMode = GetCacheModeFromTensor(dst.Data().Get());

        bool reluEn = trait.enableRelu;
        uint8_t unitFlag = params.unitFlag;
        bool isChannelSplit = trait.enableChannelSplit;
        bool nz2ndEn = true;
        bool nz2dnEn = false;
        CopyMatrixCcToGmBase3510 copyInst;
        copyInst.DataCopy<trait, quantPre, T, U>(dst, src, nSize, mSize, srcStride, dstStride,
            cacheMode, reluEn, unitFlag, isChannelSplit, nz2ndEn, nz2dnEn);
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2GM_NZ2ND_H