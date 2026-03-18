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
* \file engine_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_ENGINE_IMPL_H
#define IMPL_TENSOR_API_TENSOR_ENGINE_IMPL_H

#include "impl/utils/utils_impl.h"

namespace AscendC {
namespace Te {

template <typename Iterator>
struct ViewEngine
{
    using iterator     = Iterator;
    using reference    = typename IterRef<iterator>::type; // T&
    using elementType = typename IterEle<iterator>::type; // rm_ref
    using valueType   = typename IterVal<iterator>::type; // rm_cvf
    __aicore__ inline constexpr iterator const& Begin() const {
        return storage;
    }

    __aicore__ inline constexpr iterator& Begin() {
        return storage;
    }
    __aicore__ inline constexpr ViewEngine(iterator storage = {}) : storage(storage) {}
private:
    iterator storage;
};

template <typename Iterator>
struct ConstViewEngine
{
    using iterator     = Iterator;
    using reference    = typename IterRef<iterator>::type; // T&
    using elementType = typename IterEle<iterator>::type; // rm_ref
    using valueType   = typename IterVal<iterator>::type; // rm_cvf

    __aicore__ inline constexpr iterator const& Begin() const {
        return storage;
    }
    __aicore__ inline constexpr ConstViewEngine(iterator storage = {}) : storage(storage) {}
private:
    iterator storage;
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_ENGINE_IMPL_H