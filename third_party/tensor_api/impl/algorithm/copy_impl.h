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
* \file copy_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_ALGORITHM_COPY_IMPL_H
#define IMPL_TENSOR_API_ALGORITHM_COPY_IMPL_H

#include "impl/atom/copy_atom_impl.h"

namespace AscendC {
namespace Te {

template <typename Tp, const Tp& traits, typename T, typename... Params>
__aicore__ inline void Copy(const CopyAtom<T>& atomCopy, const Params& ...params)
{
    atomCopy.template Call<traits>(params...);
}

template <typename T, typename... Params>
__aicore__ inline void Copy(const CopyAtom<T>& atomCopy, const Params& ...params)
{
    atomCopy.Call(params...);
}

template <typename... Args>
__aicore__ inline auto MakeCopy(const Args& ...traits) {
    return CopyAtom<CopyTraits<Args...>>{};
}

}
}

#endif // IMPL_TENSOR_API_ALGORITHM_COPY_IMPL_H