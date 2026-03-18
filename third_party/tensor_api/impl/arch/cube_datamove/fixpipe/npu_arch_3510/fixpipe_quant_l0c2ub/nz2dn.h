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
 * \file nz2dn.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_L0C2UB_NZ2DN_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_L0C2UB_NZ2DN_H

#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class Fixpipe2UbNZ2DNSimpleQuant3510 {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename V, typename Params>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant, const Params& params)
    {
        SetRegisterImpl<trait, T, U>(dst, src, quant);
        DataCopyImpl<trait, quantPre, T, U, Params>(dst, src, params);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        CheckFormat::CheckDNTemplate<T>();
        CheckFormat::CheckL0CNZTemplate<U>();
    }

    template <const FixpipeTrait& trait, typename T, typename U, typename V>
    __aicore__ inline void SetRegisterImpl(const T& dst, const U& src, const V& quant)
    {
        uint32_t dnNum = 1;
        uint32_t srcNZMatrixStride = 0;
        uint32_t dstDNMatrixStride = 0;
        uint32_t srcNZC0Stride = 1;
        SetRegisterBase3510 setRegisterInst;
        setRegisterInst.SetRegister(quant, dnNum, dstDNMatrixStride, srcNZMatrixStride, srcNZC0Stride);
    }

    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename Params>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const Params& params)
    {
        CheckTemplate<trait, T, U>();
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);
        uint8_t dualDstCtl = trait.dualDstCtl;

        bool reluEn = trait.enableRelu;
        uint8_t unitFlag = params.unitFlag;
        bool subBlockId = false;
        bool nz2ndEn = false;
        bool nz2dnEn = true;
        CopyMatrixCcToUbBase3510 copyInst;
        copyInst.DataCopy<trait, quantPre, T, U>(dst, src, nSize, mSize, srcStride, dstStride, dualDstCtl,
            reluEn, unitFlag, subBlockId, nz2ndEn, nz2dnEn);
    }
};

class Fixpipe2UbNZ2DNVectorBase3510 {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename V, typename... Params>
    __aicore__ inline void FixpipeNZ2DNVectorEntrance(const T& dst, const U& src, const V& quant, const Params& ...params)
    {
        FixpipeNZ2DNVectorCompute<trait, quantPre, T, U, V>(dst, src, quant, params...);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U, bool isTail, typename Params>
    __aicore__ inline auto GenParams(const T& dst, const U& src, const Params& params)
    {
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        if constexpr (isTail) {
            nSize = nSize % MAIN_LOOP_N_SIZE_3510;
        } else {
            if (nSize > MAIN_LOOP_N_SIZE_3510) {
                nSize = MAIN_LOOP_N_SIZE_3510;
            }
        }
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);
        uint8_t dualDstCtl = trait.dualDstCtl;

        bool reluEn = trait.enableRelu;
        uint8_t unitFlag = params.unitFlag;
        bool subBlockId = false;
        bool nz2ndEn = false;
        bool nz2dnEn = true;
        auto fixpipeParams = Std::make_tuple(
            nSize, mSize, srcStride, dstStride, dualDstCtl, reluEn, unitFlag, subBlockId, nz2ndEn, nz2dnEn);
        return fixpipeParams;
    }

    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename V>
    __aicore__ inline void FixpipeNZ2DNVectorCompute(const T& dst, const U& src, const V& quant, uint32_t nIterNum,
        uint32_t calNSize, uint32_t tailNSize, const FixpipeParams& params)
    {
        auto mainLoopParam = GenParams<trait, T, U, false, FixpipeParams>(dst, src, params);
        CopyMatrixCcToUbBase3510 copyInst;
        CopyDeqTensorToFbuf3510 copyDeqTensorInst;
        for (uint16_t i = 0; i < nIterNum; ++i) {
            copyDeqTensorInst.CopyDeqTensorToFbufImpl(quant, calNSize, i);
            InsertSync();
            auto srcCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, i * CBURST_NUM_3510));
            auto dstCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, i * MAIN_LOOP_N_SIZE_3510));
            DataCopyWrapper<trait, quantPre>(copyInst, dst(dstCoord), src(srcCoord),
                mainLoopParam, tuple_sequence<decltype(mainLoopParam)>{});
        }
        if (tailNSize) {
            auto tailParam = GenParams<trait, T, U, true, FixpipeParams>(dst, src, params);
            copyDeqTensorInst.CopyDeqTensorToFbufImpl(quant, tailNSize, nIterNum);
            InsertSync();
            auto srcCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, nIterNum * CBURST_NUM_3510));
            auto dstCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, nIterNum * MAIN_LOOP_N_SIZE_3510));
            DataCopyWrapper<trait, quantPre>(copyInst, dst(dstCoord), src(srcCoord),
                tailParam, tuple_sequence<decltype(tailParam)>{});
        }
    }

    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void DataCopyWrapper(CopyMatrixCcToUbBase3510& copyInst, const T& dst, const U& src,
        const V& tupleParams, Std::index_sequence<Is...>)
    {
        copyInst.DataCopy<trait, quantPre>(dst, src, Std::get<Is>(tupleParams)...);
    }
};

class Fixpipe2UbNZ2DNVectorQuant3510 : public Fixpipe2UbNZ2DNVectorBase3510 {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename V, typename Params>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant, const Params& params)
    {
        SetRegisterImpl<trait, T, U>(dst, src);
        DataCopyImpl<trait, quantPre, T, U, V, Params>(dst, src, quant, params);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        CheckFormat::CheckDNTemplate<T>();
        CheckFormat::CheckL0CNZTemplate<U>();
    }

    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename V, typename Params>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& quant, const Params& params)
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
        if (calNSize > MAIN_LOOP_N_SIZE_3510) {
            nIterNum = nSize / MAIN_LOOP_N_SIZE_3510;
            tailNSize = nSize % MAIN_LOOP_N_SIZE_3510;
            calNSize = MAIN_LOOP_N_SIZE_3510;
        }
        FixpipeNZ2DNVectorEntrance<trait, quantPre, T, U, V>(dst, src, quant, nIterNum, calNSize, tailNSize, params);
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline void SetRegisterImpl(const T& dst, const U& src)
    {
        uint32_t dnNum = 1;
        uint32_t srcNZMatrixStride = 0;
        uint32_t dstDNMatrixStride = 0;
        uint32_t srcNZC0Stride = 1;
        SetRegisterBase3510 setRegisterInst;
        setRegisterInst.SetRegister(dnNum, dstDNMatrixStride, srcNZMatrixStride, srcNZC0Stride);
    }
};
}  // namespace Te
}  // namespace AscendC

#endif  // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_L0C2UB_NZ2DN_H
