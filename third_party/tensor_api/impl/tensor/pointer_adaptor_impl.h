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
* \file pointer_adaptor_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_POINTER_ADAPTOR_IMPL_H
#define IMPL_TENSOR_API_TENSOR_POINTER_ADAPTOR_IMPL_H

#include "impl/utils/utils_impl.h"

namespace AscendC {
namespace Te {

template <typename Iterator, typename DerivedType>
struct IterAdaptor
{
    using iterator     = Iterator;
    using reference    = typename IterRef<iterator>::type; // T&
    using elementType = typename IterEle<iterator>::type; // rm_ref
    using valueType   = typename IterVal<iterator>::type; // rm_cvf

    __aicore__ inline constexpr IterAdaptor(iterator ptr = {}) : ptr(ptr) {}

    __aicore__ inline constexpr reference operator*() const {
        return *ptr;
    }

    template <typename Index>
    __aicore__ inline constexpr reference operator[](const Index& i) const {
        return ptr[i];
    }

    template <typename Index>
    __aicore__ inline constexpr DerivedType operator+(const Index& i) const {
        return {ptr + i};
    }

    __aicore__ inline constexpr iterator Get() const {
        return ptr; 
    }

    __aicore__ inline constexpr friend bool operator==(const DerivedType& x, const DerivedType& y) {
        return x.ptr == y.ptr;
    }

    __aicore__ inline constexpr friend bool operator!=(const DerivedType& x, const DerivedType& y) {
        return x.ptr != y.ptr;
    }

    __aicore__ inline constexpr friend bool operator< (const DerivedType& x, const DerivedType& y) {
        return x.ptr <  y.ptr; 
    }

    __aicore__ inline constexpr friend bool operator<=(const DerivedType& x, const DerivedType& y) {
        return x.ptr <= y.ptr; 
    }

    __aicore__ inline constexpr friend bool operator> (const DerivedType& x, const DerivedType& y) {
        return x.ptr >  y.ptr; 
    }

    __aicore__ inline constexpr friend bool operator>=(const DerivedType& x, const DerivedType& y) {
        return x.ptr >= y.ptr; 
    }
private:
    iterator ptr;
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_POINTER_ADAPTOR_IMPL_H