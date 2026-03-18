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
* \file copy_gm2l1.h
* \brief
*/
#ifndef IMPL_TENSOR_API_ATOM_CUBE_DATAMOVE_COPY_GM2L1_H
#define IMPL_TENSOR_API_ATOM_CUBE_DATAMOVE_COPY_GM2L1_H

#include "impl/utils/utils_impl.h"

#include "impl/arch/cube_datamove/data_copy/data_copy_impl.h"
#include "impl/atom/copy_traits_impl.h"

namespace AscendC {
namespace Te {

struct DataCopyTraitDefault {
    using TraitType = DataCopyTrait;
    static constexpr const TraitType value = DEFAULT_DATA_COPY_TRAIT;
};

struct CopyGM2L1 {
    template <typename Tp, const Tp& traits, typename... Args>
    __aicore__ inline static void Copy(const Args& ...args)
    {
        DataCopy<traits, Args...>(args...);
    }
};

template <typename Traits>
struct CopyTraits<CopyGM2L1, Traits> : public CopyTraits<CopyGM2L1, Traits, CopyGM2L1, DataCopyTraitDefault> {};

template <>
struct CopyTraits<CopyGM2L1> : public CopyTraits<CopyGM2L1, DataCopyTraitDefault> {};

using CopyL12BT = CopyGM2L1;
using CopyL12FB = CopyGM2L1;

}
}

#endif // IMPL_TENSOR_API_ATOM_CUBE_DATAMOVE_COPY_GM2L1_H