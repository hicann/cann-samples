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
 * \file fixpipe_quant_nz2nd_l0c2gm.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_QUANT_NZ2ND_L0C2GM_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_QUANT_NZ2ND_L0C2GM_H

#include "impl/arch/utils/arch_utils.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_2201/fixpipe_base.h"

namespace AscendC {
namespace Te {

class FixpipeNZ2ND2201SimpleQuant : public Copy2201MatrixCcToGmBase, public SetRegister2201Base {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename S, typename Params>
    __aicore__ inline void Run(const T& dst, const U& src, const S& quant, const Params& inParams)
    {
        auto registerParams = GenRegisterParams<trait, T, U>(dst, src);
        SetRegister<S, decltype(registerParams)>(quant, registerParams);
        auto params = GenFixpipeQuantParams<trait, T, U>(dst, src, inParams);

        DataCopy<trait, quantPre, T, U, decltype(params)>(dst, src, params);
    }

private:

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        CheckFormat::CheckNDTemplate<T>();
        CheckFormat::CheckL0CNZTemplate<U>();
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline auto GenRegisterParams(const T& dst, const U& src)
    {
        uint32_t ndNum = 1;
        uint32_t srcNDStride = 0;
        uint32_t dstNDStride = 0;
        auto params = Std::make_tuple(ndNum, dstNDStride, srcNDStride);
        return params;
    }

    template <const FixpipeTrait& trait, typename T, typename U, typename Params>
    __aicore__ inline auto GenFixpipeQuantParams(const T& dst, const U& src, const Params& inParams)
    {
        CheckTemplate<trait, T, U>();
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);

        bool reluEn = false;
        uint8_t unitFlag = inParams.unitFlag;
        bool isChannelSplit = false;
        bool nz2ndEn = true;
        auto params = Std::make_tuple(nSize, mSize, srcStride, dstStride, reluEn, unitFlag, isChannelSplit, nz2ndEn);
        return params;
    }
};

class FixpipeNZ2ND2201VectorBase : public Copy2201MatrixCcToGmBase, public Copy2201DeqTensorToFbuf {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename S, typename V>
    __aicore__ inline void FixpipeNZ2NDVectorEntrance(const T& dst, const U& src, const S& quant, const V& params)
    {
        FixpipeNZ2NDVectorImpl<trait, quantPre, T, U, S, V>(dst, src, quant, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename S, typename V, size_t... Is>
    __aicore__ inline void FixpipeNZ2NDVectorImpl(
        const T& dst, const U& src, const S& quant, const V& tupleParams, Std::index_sequence<Is...>)
    {
        FixpipeNZ2NDVectorCompute<trait, quantPre, T, U, S>(dst, src, quant, Std::get<Is>(tupleParams)...);
    }

    template <const FixpipeTrait& trait, typename T, typename U, bool isTail>
    __aicore__ inline auto GenParams(const T& dst, const U& src)
    {
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        if constexpr (isTail) {
            nSize = nSize % MAIN_LOOP_N_SIZE_2201;
        } else {
            if (nSize > MAIN_LOOP_N_SIZE_2201) {
                nSize = MAIN_LOOP_N_SIZE_2201;
            }
        }
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);

        bool reluEn = false;
        uint8_t unitFlag = 0;
        bool isChannelSplit = false;
        bool nz2ndEn = true;
        auto params = Std::make_tuple(nSize, mSize, srcStride, dstStride, reluEn, unitFlag, isChannelSplit, nz2ndEn);
        return params;
    }

    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename S>
    __aicore__ inline void FixpipeNZ2NDVectorCompute(const T& dst, const U& src, const S& quant,
        uint32_t nIterNum, uint32_t calNSize, uint32_t tailNSize, uint32_t dstOffset, uint32_t srcOffset)
    {
        auto mainLoopParam = GenParams<trait, T, U, false>(dst, src);
        for (uint16_t i = 0; i < nIterNum; ++i) {
            CopyDeqTensorToFbufImpl(quant, calNSize, i);
            InsertSync();
            auto coordZero = MakeCoord(Std::Int<0>{}, Std::Int<0>{});
            auto srcTensor = MakeTensorWithCoord<U>(src, coordZero, srcOffset * i);
            auto dstTensor = MakeTensorWithCoord<T>(dst, coordZero, dstOffset * i);

            DataCopy<trait, quantPre, T, U, decltype(mainLoopParam)>(dstTensor, srcTensor, mainLoopParam);
        }
        auto tailParam = GenParams<trait, T, U, true>(dst, src);
        if (tailNSize) {
            CopyDeqTensorToFbufImpl(quant, tailNSize, nIterNum);
            InsertSync();
            auto coordZero = MakeCoord(Std::Int<0>{}, Std::Int<0>{});
            auto srcTensor = MakeTensorWithCoord<U>(src, coordZero, srcOffset * nIterNum);
            auto dstTensor = MakeTensorWithCoord<T>(dst, coordZero, dstOffset * nIterNum);

            DataCopy<trait, quantPre, T, U, decltype(tailParam)>(dstTensor, srcTensor, tailParam);
        }
    }
};

class FixpipeNZ2ND2201VectorQuant : public FixpipeNZ2ND2201VectorBase, public SetRegister2201Base {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename S, typename Params>
    __aicore__ inline void Run(const T& dst, const U& src, const S& quant, const Params& inParams)
    {
        auto registerParams = GenRegisterParams<trait, T, U>(dst, src);
        SetRegister<decltype(registerParams)>(registerParams);
        auto params = GenParams<trait, T, U>(dst, src);
        FixpipeNZ2NDVectorEntrance<trait, quantPre, T, U, S, decltype(params)>(dst, src, quant, params);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        CheckFormat::CheckNDTemplate<T>();
        CheckFormat::CheckL0CNZTemplate<U>();
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline auto GenParams(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t srcStride =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);

        uint16_t nIterNum = 1;
        uint32_t calNSize = nSize;
        uint32_t tailNSize = 0;
        uint32_t dstOffset = MAIN_LOOP_N_SIZE_2201;
        uint32_t srcOffset = CBURST_NUM_2201 * srcStride * FRACTAL_FIXED;
        if (calNSize > MAIN_LOOP_N_SIZE_2201) {
            nIterNum = nSize / MAIN_LOOP_N_SIZE_2201;
            tailNSize = nSize % MAIN_LOOP_N_SIZE_2201;
            calNSize = MAIN_LOOP_N_SIZE_2201;
        }
        auto params = Std::make_tuple(nIterNum, calNSize, tailNSize, dstOffset, srcOffset);
        return params;
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline auto GenRegisterParams(const T& dst, const U& src)
    {
        uint32_t ndNum = 1;
        uint32_t srcNDStride = 0;
        uint32_t dstNDStride = 0;
        auto params = Std::make_tuple(ndNum, dstNDStride, srcNDStride);
        return params;
    }
};
}  // namespace Te
}  // namespace AscendC

#endif  // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_QUANT_NZ2ND_L0C2GM_H
