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
 * \file nd2nz.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_ND2NZ_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_ND2NZ_H

#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class CopyGmToCbufMultiNd2nzBase {
public:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static void Run(const T& dst, const U& src) {
        DataCopyImpl<trait>(dst, src);
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static constexpr void CheckTemplate()
    {
        CheckFormat::CheckNDTemplate<U>();
        CheckFormat::CheckNZTemplate<T>();
        CheckDataTypeFor3510::CheckGm2L1Fp4DataType<T, U>();
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static void DataCopyImpl(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        using type = typename U::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        uint16_t ndNum = 1;
        uint16_t nValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t dValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        auto srcRowStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
        auto dstColumnStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);
        if constexpr (is_b4_type<type>) {
            // move fp4 as b8, need to be divided by 2
            dValue = dValue >> 1;
            srcRowStride = srcRowStride >> 1;
        }
        constexpr auto c0Size = is_b4_type<type> ? C0_SIZE * 2 : C0_SIZE;
        uint64_t srcNdMatrixStride = 0;
        uint64_t srcDValue = srcRowStride;
        uint16_t dstNzC0Stride = dstColumnStride * sizeof(type) / c0Size;
        uint16_t dstNzNStride = 1;
        uint32_t dstNzMatrixStride = 0;

        uint64_t loop1SrcStride = srcDValue * sizeof(type);
        uint64_t loop4SrcStride = srcNdMatrixStride * sizeof(type);

        uint16_t loop2DstStride = dstNzNStride;  // loop2_dst_stride = dst_nz_n_stride
        uint16_t loop3DstStride = dstNzC0Stride; // loop3_dst_stride = dst_nz_c0_Stride
        // loop4_dst_stride: dst_nz_matrix_stride * size_of_dst_type / C0_size
        uint16_t loop4DstStride = static_cast<uint16_t>(dstNzMatrixStride * sizeof(type) / c0Size);
        uint8_t cacheMode = GetCacheModeFromTensor(src.Data().Get());

        CopyGmToCbufMultiNd2nzInstr::DataCopy(dst, src, ndNum, loop2DstStride, loop3DstStride, loop4DstStride,
                                         loop1SrcStride, cacheMode, nValue, dValue, loop4SrcStride, false);
    }
};

}
}

#endif
