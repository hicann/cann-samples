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
 * \file nd2nd.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_ND2ND_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_ND2ND_H

#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class CopyGmToCbufAlignV2NDBase {
public:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static void Run(const T& dst, const U& src) {
        DataCopyImpl<trait, T, U>(dst, src);
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static constexpr void CheckTemplate()
    {
        CheckFormat::CheckNDTemplate<T>();
        CheckFormat::CheckNDTemplate<U>();
        CheckDataTypeFor3510::CheckGm2L1AlignV2NDDataType<T, U>();
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static void DataCopyImpl(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
 
        auto dstShapeRows = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
        auto dstShapeColumns = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
        auto dstStrideRows = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);
        auto srcStrideRows = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
 
        uint8_t leftPaddingCnt = 0;
        uint8_t rightPaddingCnt = 0;
        uint8_t cacheMode = GetCacheModeFromTensor(src.Data().Get());
 
        using type = typename U::elementType;
 
        auto blockCount = dstShapeRows;
        auto blockLen = dstShapeColumns * sizeof(type);
        auto srcStride = srcStrideRows * sizeof(type);
        auto dstStride = dstStrideRows * sizeof(type);

        CopyGmToCbufAlignV2Base::DataCopy(dst, src, blockCount, blockLen, leftPaddingCnt, rightPaddingCnt, cacheMode, srcStride, dstStride);
    }
};

}
}

#endif
