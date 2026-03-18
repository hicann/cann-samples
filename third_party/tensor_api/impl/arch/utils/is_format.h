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
 * \file is_format.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_UTILS_IS_FORMAT_H
#define IMPL_TENSOR_API_ARCH_UTILS_IS_FORMAT_H

#include "impl/utils/utils_impl.h"
#include "impl/tensor/pointer_impl.h"
#include "impl/tensor/local_tensor_impl.h"

namespace AscendC {
namespace Te {

template <typename T>
struct GetTypeFromFourDimTrait;

template <Hardware hPos, typename Pointer, typename Shape1, typename Shape2, typename Stride1, typename Stride2>
struct GetTypeFromFourDimTrait<LocalTensor<TensorAttribute<ViewEngine<HardwareMemPtr<hPos, Pointer>>, Layout<Shape<Shape1, Shape2>, Stride<Stride1, Stride2>>>>> {
    using ShapeRowsZeroDim = typename Std::tuple_element<0, Shape1>::type;
    using ShapeRowsOneDim = typename Std::tuple_element<1, Shape1>::type;
    using ShapeColumnsZeroDim = typename Std::tuple_element<0, Shape2>::type;
    using ShapeColumnsOneDim = typename Std::tuple_element<1, Shape2>::type;

    using StrideRowsZeroDim = typename Std::tuple_element<0, Stride1>::type;
    using StrideRowsOneDim = typename Std::tuple_element<1, Stride1>::type;
    using StrideColumnsZeroDim = typename Std::tuple_element<0, Stride2>::type;
    using StrideColumnsOneDim = typename Std::tuple_element<1, Stride2>::type;
};

enum class AttrInfo : uint8_t {SHAPE, STRIDE, ROW, COLUMN};

template <typename T, AttrInfo info1, AttrInfo info2, size_t dim> 
struct GetFourDimType;

template <typename T>
struct GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::ShapeRowsZeroDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 1> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::ShapeRowsOneDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::ShapeColumnsZeroDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 1> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::ShapeColumnsOneDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::StrideRowsZeroDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::StrideRowsOneDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::StrideColumnsZeroDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::StrideColumnsOneDim>;
};

template <typename T>
struct IsZZFormat {
private:
    __aicore__ inline static constexpr bool IsFractalZZFormatNormal() {
        constexpr bool isShapeRight = Std::is_constant<FRACTAL_FIXED, ShapeRow0>::value 
            && Std::is_constant<C0_SIZE / sizeof(type), ShapeColumn0>::value;
        constexpr bool isStrideRight = Std::is_constant<C0_SIZE / sizeof(type), StrideRow0>::value 
            && Std::is_constant<1, StrideColumn0>::value;
        return (isShapeRight && isStrideRight);
    }

    __aicore__ inline static constexpr bool IsFractalScaleZZFormat() {
        constexpr bool isShapeRight = Std::is_constant<FRACTAL_FIXED, ShapeRow0>::value 
        && Std::is_constant<MX_SCALE_K0, ShapeColumn0>::value;
        constexpr bool isStrideRight = Std::is_constant<MX_SCALE_K0, StrideRow0>::value 
        && Std::is_constant<1, StrideColumn0>::value;
        return (isShapeRight && isStrideRight);
    }

    __aicore__ inline static constexpr bool IsFractalZZFormat() {
        constexpr bool isScaleType = is_one_of_attr_v<type, fp8_e8m0_t>;
        using ResultType = Std::conditional_t<isScaleType,
            Std::bool_constant<IsFractalScaleZZFormat()>,
            Std::bool_constant<IsFractalZZFormatNormal()>>;
        return ResultType::value;
    }
public:
    using type = typename T::elementType;
    using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
    using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
    using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
    using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
    static constexpr bool value = IsFractalZZFormat();
};

template <typename T>
struct IsNNFormat {
private:
    __aicore__ inline static constexpr bool IsFractalNNFormat() {
        using type = typename T::elementType;
        static_assert(Std::is_same_v<type, __cbuf__ fp8_e8m0_t>, "NnFormat Only support fp8_e8m0_t");
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;

        constexpr bool isShapeRight = Std::is_constant<MX_SCALE_K0, ShapeRow0>::value 
        && Std::is_constant<FRACTAL_FIXED, ShapeColumn0>::value;
        constexpr bool isStrideRight = Std::is_constant<1, StrideRow0>::value 
        && Std::is_constant<MX_SCALE_K0, StrideColumn0>::value;
        return (isShapeRight && isStrideRight);
    }
public:
    static constexpr bool value = IsFractalNNFormat();
};

template <typename T>
struct IsZNFormat {
private:
    __aicore__ inline static constexpr bool IsFractalZNFormat() {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        constexpr auto c0Size = is_b4_type<type> ? C0_SIZE * 2 : C0_SIZE / sizeof(type);
        constexpr bool isShapeRight =
            Std::is_constant<c0Size, ShapeRow0>::value && Std::is_constant<FRACTAL_FIXED, ShapeColumn0>::value;
        constexpr bool isStrideRight =
            Std::is_constant<1, StrideRow0>::value && Std::is_constant<c0Size, StrideColumn0>::value;

        return (isShapeRight && isStrideRight);
    }
public:
    static constexpr bool value = IsFractalZNFormat();
};

template <typename T>
struct IsNZFormat {
private:
    __aicore__ inline static constexpr bool IsFractalNZFormat() {
        using type = typename T::elementType;
        // NZ shape (Int<16>, row) , (Int<C0Size>, column))
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        // NZ stride (Int<C0Size>, N * C0Size + 16 * Int<C0Size>) , (Int<1>, row * (N * C0Size + 16 * Int<C0Size>)))
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        constexpr auto c0Size = is_b4_type<type> ? C0_SIZE * 2 : C0_SIZE / sizeof(type);
        constexpr bool isStrideRight =
            Std::is_constant<c0Size, StrideRow0>::value && Std::is_constant<1, StrideColumn0>::value;
        constexpr bool isShapeRight =
            Std::is_constant<FRACTAL_FIXED, ShapeRow0>::value && Std::is_constant<c0Size, ShapeColumn0>::value;

        return (isShapeRight && isStrideRight);
    }
public:
    static constexpr bool value = IsFractalNZFormat();
};

template <typename T>
struct IsL0cNZFormat {
private:
    __aicore__ inline static constexpr bool IsFractalL0cNZFormat() {
        using type = typename T::elementType;
        // NZ shape (Int<16>, row) , (Int<C0Size>, column))
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;

        // NZ stride (Int<C0Size>, N * C0Size + 16 * Int<C0Size>) , (Int<1>, row * (N * C0Size + 16 * Int<C0Size>)))
        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;

        constexpr bool isShapeRight = Std::is_constant<FRACTAL_FIXED, ShapeRow0>::value 
            && Std::is_constant<FRACTAL_FIXED, ShapeColumn0>::value;
        constexpr bool isStrideRight = Std::is_constant<FRACTAL_FIXED, StrideRow0>::value 
            && Std::is_constant<1, StrideColumn0>::value;

        return (isShapeRight && isStrideRight);
    }
public:
    static constexpr bool value = IsFractalL0cNZFormat();
};

template <typename T>
struct IsNDFormat {
private:
    __aicore__ inline static constexpr bool IsFractalNDFormatNormal() {
        // shape = ((1, row),(1,col)) stride = ((0, col),(0, 1))
        constexpr bool isShapeRight = Std::is_constant<1, ShapeRow0>::value && Std::is_constant<1, ShapeColumn0>::value;
        constexpr bool isStrideRight = Std::is_constant<0, StrideRow0>::value && Std::is_constant<0, StrideColumn0>::value 
                                                                                && Std::is_constant<1, StrideColumn1>::value;

        return (isShapeRight && isStrideRight);
    }

    __aicore__ inline static constexpr bool IsFractalScaleNDFormat() {
        // shape = ((2, row/2),(1,col)) stride = ((1, 2*col),(0, 2))
        constexpr bool isShapeRight = Std::is_constant<2, ShapeRow0>::value && Std::is_constant<1, ShapeColumn0>::value;
        constexpr bool isStrideRight = Std::is_constant<1, StrideRow0>::value && Std::is_constant<0, StrideColumn0>::value
                                                                                && Std::is_constant<2, StrideColumn1>::value;

        return (isShapeRight && isStrideRight);
    }

    __aicore__ inline static constexpr bool IsFractalNDFormat() {
        constexpr bool isScaleType = is_one_of_attr_v<type, fp8_e8m0_t>;
        using ResultType = Std::conditional_t<isScaleType,
            Std::bool_constant<IsFractalScaleNDFormat()>,
            Std::bool_constant<IsFractalNDFormatNormal()>>;
        return ResultType::value;
    }
public:
    using type = typename T::elementType;
    using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
    using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;

    using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
    using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
    using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
    static constexpr bool value = IsFractalNDFormat();
    static constexpr bool normalValue = IsFractalNDFormatNormal();
};

template <typename T>
struct IsDNFormat {
private:
    __aicore__ inline static constexpr bool IsFractalDNFormatNormal() {
        // shape = ((1, row),(1,col)) stride = ((0, 1),(0, row))
        constexpr bool isShapeRight = Std::is_constant<1, ShapeRow0>::value && Std::is_constant<1, ShapeColumn0>::value;
        constexpr bool isStrideRight = Std::is_constant<0, StrideRow0>::value && Std::is_constant<1, StrideRow1>::value && Std::is_constant<0, StrideColumn0>::value;

        return (isShapeRight && isStrideRight);
    }

    __aicore__ inline static constexpr bool IsFractalScaleDNFormat() {
         // shape = ((1, row),(2,col/2)) stride = ((0, 2),(1, row*2))
        constexpr bool isShapeRight = Std::is_constant<1, ShapeRow0>::value && Std::is_constant<2, ShapeColumn0>::value;
        constexpr bool isStrideRight = Std::is_constant<0, StrideRow0>::value && Std::is_constant<2, StrideRow1>::value && Std::is_constant<1, StrideColumn0>::value;

        return (isShapeRight && isStrideRight);
    }

     __aicore__ inline static constexpr bool IsFractalDNFormat() {
        constexpr bool isScaleType = is_one_of_attr_v<type, fp8_e8m0_t>;
        using ResultType = Std::conditional_t<isScaleType,
            Std::bool_constant<IsFractalScaleDNFormat()>,
            Std::bool_constant<IsFractalDNFormatNormal()>>;
        return ResultType::value;
    }
public:
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
    static constexpr bool value = IsFractalDNFormat();
    static constexpr bool normalValue = IsFractalDNFormatNormal();
};

template <typename T>
struct IsScaleANDFormat { // shape = ((1, row),(1,col)) stride = ((0, col),(0, 1))
    static constexpr bool value = IsNDFormat<T>::normalValue;
};

template <typename T>
struct IsScaleADNFormat { // shape = ((1, row),(2,col/2)) stride = ((0, 2),(1, row*2))
    static constexpr bool value = IsDNFormat<T>::value;
};

template <typename T>
struct IsScaleBNDFormat { // shape = ((2, row/2),(1,col)) stride = ((1, 2*col),(0, 2))
    static constexpr bool value = IsNDFormat<T>::value;
};

template <typename T>
struct IsScaleBDNFormat { // shape = ((1, row),(1,col)) stride = ((0, 1),(0, row))
    static constexpr bool value = IsDNFormat<T>::normalValue;
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_UTILS_IS_FORMAT_H