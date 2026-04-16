/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning                                                                                                               \
    "tensor_api/impl/tensor/engine_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file engine_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_ENGINE_IMPL_H
#define IMPL_TENSOR_API_TENSOR_ENGINE_IMPL_H

#include "impl/utils/utils_impl.h"

namespace AscendC {
namespace Te {

enum class CacheMode {
    CACHE_MODE_DISABLE = 0,
    CACHE_MODE_NORMAL = 1,
    CACHE_MODE_LAST = 2,
    CACHE_MODE_PERSISTENT = 4
};

struct L2CacheAlter3510 {
template <typename Iterator>
    __aicore__ __inline__ static void Run(Iterator& storage, CacheMode mode)
    {
        constexpr uint64_t L2_CACHE_OFFSET = 60;
        constexpr uint64_t L2_CACHE_OFFSET_MASK = (1ul << L2_CACHE_OFFSET) - 1;

        uint64_t value = 0;
        if (mode == CacheMode::CACHE_MODE_DISABLE) {
            value = uint64_t(0b100) << L2_CACHE_OFFSET;
        } else if (mode == CacheMode::CACHE_MODE_NORMAL) {
            value = uint64_t(0b000) << L2_CACHE_OFFSET;
        }
        storage = storage & L2_CACHE_OFFSET_MASK | value;
    }
};

struct EmptyValue {template <typename ...Args> __aicore__ __inline__ static void Run(const Args&... args) {} };

template <typename... Pairs>
class TupleMap {
private:
    using MapType = Std::tuple<Pairs...>;
    static constexpr size_t MapSize = sizeof...(Pairs);

    template <typename Key, size_t Index, size_t MaxSize>
    struct FindImpl {
        using CurrentPair = typename Std::tuple_element<Index, MapType>::type;
        using CurrentKey  = typename Std::tuple_element<0, CurrentPair>::type;
        using CurrentVal  = typename Std::tuple_element<1, CurrentPair>::type;

        using NextResult = typename FindImpl<Key, Index + 1, MaxSize>::type;

        using type = Std::conditional_t<Std::is_same_v<CurrentKey, Key>, CurrentVal, NextResult>;
    };

    template <typename Key, size_t MaxSize>
    struct FindImpl<Key, MaxSize, MaxSize> {
        using type = EmptyValue;
    };

public:
    template <typename Key>
    using Get = typename FindImpl<Key, 0, MapSize>::type;
};

using L2CacheAlterSet = TupleMap<Std::tuple<Std::Int<ArchVersion::V3510>, L2CacheAlter3510>>;

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

    __aicore__ inline constexpr uint8_t GetCacheMode() const {
        return static_cast<uint8_t>(mode);
    }

    __aicore__ inline constexpr void SetCacheMode(CacheMode mode) {
        this->mode = mode;
    }
private:
    iterator storage;
    CacheMode mode = CacheMode::CACHE_MODE_NORMAL;
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

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
