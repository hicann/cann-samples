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
 * \file data_copy_gm2l1.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_H

#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/dn2nz.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/dn2zn.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/nd2nd.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/nd2nz.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/nd2zn.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/nz2nz.h"

namespace AscendC {
namespace Te {

class DataCopyGM2L13510 {
public:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline void Run(const T& dst, const U& src) {
        Execute<trait>(dst, src);
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline void Execute(const T& dst, const U& src) {
        if constexpr (IsNDFormat<U>::value && IsNDFormat<T>::value) {
            CopyGmToCbufAlignV2NDBase::Run<trait, T, U>(dst, src);
        } else if constexpr (IsNDFormat<U>::value && IsNZFormat<T>::value) {
            CopyGmToCbufMultiNd2nzBase::Run<trait, T, U>(dst, src);
        } else if constexpr (IsNDFormat<U>::value && IsZNFormat<T>::value) {
            CopyGmToCbufMultiNd2znBase::Run<trait, T, U>(dst, src);
        } else if constexpr (IsDNFormat<U>::value && IsNZFormat<T>::value) {
            CopyGmToCbufMultiDn2nzBase::Run<trait, T, U>(dst, src);
        } else if constexpr (IsDNFormat<U>::value && IsZNFormat<T>::value) {
            CopyGmToCbufMultiDn2ZnBase::Run<trait, T, U>(dst, src);
        } else if constexpr (IsNZFormat<U>::value && IsNZFormat<T>::value) {
            CopyGmToCbufAlignV2NZBase::Run<trait, T, U>(dst, src);
        } else {
            // assert error
            static_assert(Std::is_same_v<T, U>, "The data format is not supported.");
        }
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_H