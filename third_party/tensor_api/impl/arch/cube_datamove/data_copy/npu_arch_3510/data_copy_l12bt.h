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
 * \file data_copy_l12bt.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_L12BT_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_L12BT_H

#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class DataCopyL12BT3510 {
public:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static void Run(const T& dst, const U& src) {
        DataCopyImpl<trait, T, U>(dst, src);
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static constexpr void CheckTemplate()
    {
        CheckDataTypeFor3510::CheckL12BtDataType<T, U>();
        CheckFormat::CheckNDTemplate<T>();
        CheckFormat::CheckNDTemplate<U>();
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static auto DataCopyImpl(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        uint16_t srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint16_t srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
        uint16_t dstCol = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
        uint16_t dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);

        using srcType = typename U::elementType;
        using dstType = typename T::elementType;

        bool convControl = false;

        if (Std::is_same_v<srcType, half> && Std::is_same_v<dstType, float>) {
            convControl = true;
        }
        uint16_t blockCount = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint16_t blockLen = (srcCol * sizeof(dstType) + C0_SIZE - 1) / C0_SIZE;
        uint16_t srcStride = (srcRow - srcCol) * sizeof(srcType) / C0_SIZE;
        uint16_t dstStride = (dstRow - srcCol) * sizeof(dstType) / C0_SIZE;

        CopyL12BTInstr copyInstr;
        copyInstr.DataCopy(dst, src, convControl, blockCount, blockLen, srcStride, dstStride);
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_H