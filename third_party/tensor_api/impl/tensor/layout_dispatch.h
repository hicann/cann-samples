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
* \file layout_dispatch.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_DISPATCH_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_DISPATCH_H

#include "impl/utils/utils_impl.h"

namespace AscendC {
namespace Te {

__aicore__ inline int64_t CeilAlign(int64_t a, int64_t b) {
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

struct MakeTupleCons {
    template <typename... Ts>
    __aicore__ inline decltype(auto) operator()(Ts&&... ts) {
        return Std::make_tuple(Std::forward<Ts>(ts)...);
    }
};

template <typename F, typename T>
__aicore__ inline decltype(auto) Make2Params2Tuple(F&& f, T&& t) {
    return t;
}

template <typename F, typename T0, typename T1>
__aicore__ inline decltype(auto) Make2Params2Tuple(F&& f, T0&& t0, T1&& t1) {
    return f(t0, t1);
}

template <typename F, typename T0, typename T1, typename... Ts>
__aicore__ inline decltype(auto) Make2Params2Tuple(F&& f, T0&& t0, T1&& t1, Ts&&... ts) {
    auto tuple1 = Make2Params2Tuple(f, t0, t1);
    auto tuple2 = Make2Params2Tuple(f, ts...);
    return Make2Params2Tuple(f, tuple1, tuple2);
}

template <typename... Ts>
__aicore__ inline decltype(auto) GetTuple(Ts&&... ts) {
    return Make2Params2Tuple(MakeTupleCons{}, ts...);
}

 template <typename T0, typename T1, typename T2, typename T3, typename... Ts> 
 __aicore__ inline decltype(auto) LayoutConstructor(T0&& t0, T1&& t1, T2&& t2, T3&& t3, Ts&&... ts) { 
     auto shape = GetTuple(t0, t1, t2, t3); 
     auto stride = GetTuple(ts...); 
     return Layout(shape, stride); 
 }

// tuple_dispatch.h
template <LayoutFormat format, TupleFormat tupleFormat, typename T>
struct TupleDispatcher;

// make shape
template <typename T>
struct TupleDispatcher<LayoutFormat::NZ, TupleFormat::Shape, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        constexpr auto c0Size = is_b4_type<T> ? C0_SIZE * 2 : C0_SIZE / sizeof(T);
        return GetTuple(Std::Int<FRACTAL_FIXED>{},  CeilDivision(row, FRACTAL_FIXED), 
                                Std::Int<c0Size / sizeof(T)>{},  CeilDivision(column, (c0Size / sizeof(T)))); 
    }
};

template <>
struct TupleDispatcher<LayoutFormat::NZ, TupleFormat::Shape, Std::ignore_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<FRACTAL_FIXED>{},  CeilDivision(row, FRACTAL_FIXED), 
                                Std::Int<C0_SIZE / sizeof(uint16_t)>{},  CeilDivision(column, (C0_SIZE / sizeof(uint16_t)))); 
    }
};

template <typename T>
struct TupleDispatcher<LayoutFormat::ZN, TupleFormat::Shape, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<C0_SIZE / sizeof(T)>{},  CeilDivision(row, (C0_SIZE / sizeof(T))),
                                Std::Int<FRACTAL_FIXED>{},  CeilDivision(column, FRACTAL_FIXED));
    }
};

template <typename T>
struct TupleDispatcher<LayoutFormat::DN, TupleFormat::Shape, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<1>{}, row, Std::Int<1>{}, column);
    }
};

template <>
struct TupleDispatcher<LayoutFormat::DN, TupleFormat::Shape, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<1>{}, row, Std::Int<2>{}, column / MX_SCALE_K0);
    }
};

template <typename T>
struct TupleDispatcher<LayoutFormat::ND, TupleFormat::Shape, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<1>{}, row, Std::Int<1>{}, column);
    }
};

template <>
struct TupleDispatcher<LayoutFormat::ND, TupleFormat::Shape, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<2>{}, row / MX_SCALE_K0, Std::Int<1>{}, column);
    }
};

template <typename T>
struct TupleDispatcher<LayoutFormat::ZZ, TupleFormat::Shape, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<FRACTAL_FIXED>{}, CeilDivision(row, FRACTAL_FIXED),
                                    Std::Int<C0_SIZE / sizeof(T)>{}, CeilDivision(column, (C0_SIZE / sizeof(T))));
    }
};

template <>
struct TupleDispatcher<LayoutFormat::ZZ, TupleFormat::Shape, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<FRACTAL_FIXED>{}, CeilDivision(row, FRACTAL_FIXED),
                                    Std::Int<MX_SCALE_K0>{}, column / MX_SCALE_K0);
    }
};

template <>
struct TupleDispatcher<LayoutFormat::NN, TupleFormat::Shape, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) { // (scaleK, n)
        return GetTuple(Std::Int<MX_SCALE_K0>{}, row / MX_SCALE_K0,
                                    Std::Int<FRACTAL_FIXED>{}, CeilDivision(column, FRACTAL_FIXED));
    }
};

// make stride
template <typename T>
struct TupleDispatcher<LayoutFormat::NZ, TupleFormat::Stride, T> {
    __aicore__ inline static decltype(auto) apply(size_t stride0, size_t stride1) {
        constexpr auto c0Size = is_b4_type<T> ? C0_SIZE * 2 : C0_SIZE / sizeof(T);
        return GetTuple(Std::Int<c0Size / sizeof(T)>{},  stride0, Std::Int<1>{},  stride1); 
    }
};

template <>
struct TupleDispatcher<LayoutFormat::NZ, TupleFormat::Stride, Std::ignore_t> {
    __aicore__ inline static decltype(auto) apply(size_t stride0, size_t stride1) {
        return GetTuple(Std::Int<C0_SIZE / sizeof(uint16_t)>{},  stride0, Std::Int<1>{},  stride1); 
    }
};

template <typename T>
struct TupleDispatcher<LayoutFormat::ZN, TupleFormat::Stride, T> {
    __aicore__ inline static decltype(auto) apply(size_t stride0, size_t stride1) {
        return GetTuple(Std::Int<1>{},  stride0, Std::Int<C0_SIZE / sizeof(T)>{},  stride1);
    }
};

template <typename T>
struct TupleDispatcher<LayoutFormat::DN, TupleFormat::Stride, T> {
    __aicore__ inline static decltype(auto) apply(size_t stride0, size_t stride1) {
        return GetTuple(Std::Int<0>{}, stride0, Std::Int<0>{}, stride1);
    }
};

template <>
struct TupleDispatcher<LayoutFormat::DN, TupleFormat::Stride, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t stride0, size_t stride1) {
        return GetTuple(Std::Int<0>{}, stride0, Std::Int<1>{}, stride1);
    }
};

template <typename T>
struct TupleDispatcher<LayoutFormat::ND, TupleFormat::Stride, T> {
    __aicore__ inline static decltype(auto) apply(size_t stride0, size_t stride1) {
        return GetTuple(Std::Int<0>{}, stride0, Std::Int<0>{}, stride1);
    }
};

template <>
struct TupleDispatcher<LayoutFormat::ND, TupleFormat::Stride, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t stride0, size_t stride1) {
        return GetTuple(Std::Int<1>{}, stride0, Std::Int<0>{}, stride1);
    }
};

template <typename T>
struct TupleDispatcher<LayoutFormat::ZZ, TupleFormat::Stride, T> {
    __aicore__ inline static decltype(auto) apply(size_t stride0, size_t stride1) {
        return GetTuple(Std::Int<C0_SIZE / sizeof(T)>{}, stride0, Std::Int<1>{}, stride1);
    }
};

template <>
struct TupleDispatcher<LayoutFormat::ZZ, TupleFormat::Stride, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t stride0, size_t stride1) {
        return GetTuple(Std::Int<MX_SCALE_K0>{}, stride0, Std::Int<1>{}, stride1);
    }
};

template <>
struct TupleDispatcher<LayoutFormat::NN, TupleFormat::Stride, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t stride0, size_t stride1) { // (scaleK, n)
        return GetTuple(Std::Int<1>{}, stride0, Std::Int<MX_SCALE_K0>{}, stride1);
    }
};

// make coord
template <typename T>
struct TupleDispatcher<LayoutFormat::NZ, TupleFormat::Coord, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        constexpr auto c0Size = is_b4_type<T> ? C0_SIZE * 2 : C0_SIZE / sizeof(T);
        return GetTuple(Std::Int<0>{},  CeilDivision(row, FRACTAL_FIXED), 
                                Std::Int<0>{},  CeilDivision(column, (c0Size / sizeof(T)))); 
    }
};

template <>
struct TupleDispatcher<LayoutFormat::NZ, TupleFormat::Coord, Std::ignore_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<0>{},  CeilDivision(row, FRACTAL_FIXED), 
                                Std::Int<0>{},  CeilDivision(column, (C0_SIZE / sizeof(uint16_t)))); 
    }
};

template <typename T>
struct TupleDispatcher<LayoutFormat::ZN, TupleFormat::Coord, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<0>{},  CeilDivision(row, (C0_SIZE / sizeof(T))),
                                Std::Int<0>{},  CeilDivision(column, FRACTAL_FIXED));
    }
};

template <typename T>
struct TupleDispatcher<LayoutFormat::DN, TupleFormat::Coord, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<0>{}, row, Std::Int<0>{}, column);
    }
};

template <>
struct TupleDispatcher<LayoutFormat::DN, TupleFormat::Coord, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<0>{}, row, Std::Int<0>{}, column / MX_SCALE_K0);
    }
};

template <typename T>
struct TupleDispatcher<LayoutFormat::ND, TupleFormat::Coord, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<0>{}, row, Std::Int<0>{}, column);
    }
};

template <>
struct TupleDispatcher<LayoutFormat::ND, TupleFormat::Coord, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<0>{}, row / MX_SCALE_K0, Std::Int<0>{}, column);
    }
};

template <typename T>
struct TupleDispatcher<LayoutFormat::ZZ, TupleFormat::Coord, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<0>{}, CeilDivision(row, FRACTAL_FIXED),
                                    Std::Int<0>{}, CeilDivision(column, (C0_SIZE / sizeof(T))));
    }
};

template <>
struct TupleDispatcher<LayoutFormat::ZZ, TupleFormat::Coord, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return GetTuple(Std::Int<0>{}, CeilDivision(row, FRACTAL_FIXED),
                                    Std::Int<0>{}, column / MX_SCALE_K0);
    }
};

template <>
struct TupleDispatcher<LayoutFormat::NN, TupleFormat::Coord, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) { // (scaleK, n)
        return GetTuple(Std::Int<0>{}, row / MX_SCALE_K0,
                                    Std::Int<0>{}, CeilDivision(column, FRACTAL_FIXED));
    }
};

// layout_dispatch.h
template <LayoutFormat format, typename T>
struct LayoutDispatcher;

template <typename T>
struct LayoutDispatcher<LayoutFormat::NZ, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        constexpr auto c0Size = is_b4_type<T> ? C0_SIZE * 2 : C0_SIZE / sizeof(T);
        return LayoutConstructor(Std::Int<FRACTAL_FIXED>{}, CeilDivision(row, FRACTAL_FIXED), Std::Int<c0Size>{},
                                 CeilDivision(column, (c0Size)), Std::Int<c0Size>{}, Std::Int<c0Size * FRACTAL_FIXED>{},
                                 Std::Int<1>{}, c0Size * CeilAlign(row, FRACTAL_FIXED));
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::NZ, Std::ignore_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<FRACTAL_FIXED>{},  CeilDivision(row, FRACTAL_FIXED), 
                                Std::Int<C0_SIZE / sizeof(uint16_t)>{},  CeilDivision(column, (C0_SIZE / sizeof(uint16_t))), 
                                Std::Int<C0_SIZE / sizeof(uint16_t)>{},  Std::Int<C0_SIZE / sizeof(uint16_t) * FRACTAL_FIXED>{},
                                Std::Int<1>{},  C0_SIZE / sizeof(uint16_t) * CeilAlign(row, FRACTAL_FIXED)); 
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::ZN, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        constexpr auto c0Size = is_b4_type<T> ? C0_SIZE * 2 : C0_SIZE / sizeof(T);
        return LayoutConstructor(Std::Int<c0Size>{},  CeilDivision(row, (c0Size)),
                                Std::Int<FRACTAL_FIXED>{},  CeilDivision(column, FRACTAL_FIXED),
                                Std::Int<1>{},  c0Size * CeilAlign(column, FRACTAL_FIXED),
                                Std::Int<c0Size>{},  Std::Int<c0Size * FRACTAL_FIXED>{});
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::DN, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<1>{}, row, Std::Int<1>{}, column,
                                    Std::Int<0>{}, Std::Int<1>{}, Std::Int<0>{}, row);
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::DN, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<1>{}, row, Std::Int<2>{}, column / MX_SCALE_K0,
                                    Std::Int<0>{}, Std::Int<MX_SCALE_K0>{}, Std::Int<1>{}, MX_SCALE_K0 * row);
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::ND, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<1>{}, row, Std::Int<1>{}, column,
                                    Std::Int<0>{}, column, Std::Int<0>{}, Std::Int<1>{});
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::ND, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<2>{}, row / MX_SCALE_K0, Std::Int<1>{}, column,
                                    Std::Int<1>{}, MX_SCALE_K0 * column, Std::Int<0>{}, Std::Int<MX_SCALE_K0>{});
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::ZZ, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<FRACTAL_FIXED>{}, CeilDivision(row, FRACTAL_FIXED),
                                    Std::Int<C0_SIZE / sizeof(T)>{}, CeilDivision(column, (C0_SIZE / sizeof(T))),
                                    Std::Int<C0_SIZE / sizeof(T)>{}, FRACTAL_FIXED * CeilAlign(column, (C0_SIZE / sizeof(T))),
                                    Std::Int<1>{}, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>{});
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::ZZ, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<FRACTAL_FIXED>{}, CeilDivision(row, FRACTAL_FIXED),
                                    Std::Int<MX_SCALE_K0>{}, column / MX_SCALE_K0,
                                    Std::Int<MX_SCALE_K0>{}, column * FRACTAL_FIXED,
                                    Std::Int<1>{}, Std::Int<C0_SIZE>{});
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::NN, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) { // (scaleK, n)
        return LayoutConstructor(Std::Int<MX_SCALE_K0>{}, row / MX_SCALE_K0,
                                    Std::Int<FRACTAL_FIXED>{}, CeilDivision(column, FRACTAL_FIXED),
                                    Std::Int<1>{}, Std::Int<C0_SIZE>{},
                                    Std::Int<MX_SCALE_K0>{}, row * FRACTAL_FIXED);
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_DISPTACH_H