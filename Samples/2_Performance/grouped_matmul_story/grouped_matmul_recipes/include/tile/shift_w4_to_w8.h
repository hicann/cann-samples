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
 * \file shift_w4_to_w8.h
 * \brief Convert packed W4 weights to W8 layout for vector-to-cube consumption.
 */
#pragma once

#include <type_traits>

#include "include/tensor.h"
#include "kernel_operator.h"

using AscendC::BLOCK_CUBE;
static constexpr int32_t C0_SIZE_B8 = 32;

namespace Tile {

template <typename OutType, typename InType>
struct ShiftW4ToW8Params {
    uint64_t loopKNum;
    uint64_t innerLoopNum;
    uint64_t loopKDstStride;
    uint64_t innerDstStride;
    uint64_t nRealSizeAlign;
    __ubuf__ InType* weight4BitPhyAddr;
    __ubuf__ OutType* weight8BitPhyAddr;
};

static constexpr uint32_t E2M1_SHIFT_RIGHT_SIZE = 0x2;
static constexpr uint32_t SHIFT_LEFT_SIZE = 0x4;
static constexpr uint32_t E2M1_AND_MASK = 0x9C;

template <typename OutType, typename InType>
__simd_vf__ inline void ShiftW4ToW8Vf(ShiftW4ToW8Params<OutType, InType> params);

template <typename OutType, typename InType, typename Weight4BitTensorType, typename Weight8BitTensorType>
__aicore__ inline void ShiftW4ToW8(
    const Weight4BitTensorType& weight4BitTensor, const Weight8BitTensorType& weight8BitTensor)
{
    static_assert(std::is_same_v<OutType, __fp8e4m3>, "OutType must be __fp8e4m3");
    static_assert(std::is_same_v<InType, __fp4e2m1x2>, "InType must be __fp4e2m1x2");

    ShiftW4ToW8Params<OutType, InType> params;
    params.weight4BitPhyAddr = (__ubuf__ InType*)weight4BitTensor.Data().Get();
    params.weight8BitPhyAddr = (__ubuf__ OutType*)weight8BitTensor.Data().Get();

    // Derive kernel-loop and stride parameters directly from tensor shape/stride metadata.
    params.loopKNum = AscendC::Std::get<1>(AscendC::Std::get<0>(weight4BitTensor.Layout().Shape()));
    params.nRealSizeAlign = AscendC::Std::get<1>(AscendC::Std::get<1>(weight4BitTensor.Layout().Shape())) * BLOCK_CUBE;
    params.innerDstStride = AscendC::Std::get<1>(AscendC::Std::get<1>(weight8BitTensor.Layout().Stride()));
    params.innerLoopNum = (params.nRealSizeAlign * C0_SIZE_B8) / static_cast<uint64_t>(AscendC::GetVecLen());
    params.loopKDstStride = params.innerLoopNum * params.innerDstStride;

    ShiftW4ToW8Vf<OutType, InType>(params);
}

template <typename OutType, typename InType>
__simd_vf__ inline void ShiftW4ToW8Vf(ShiftW4ToW8Params<OutType, InType> params)
{
    AscendC::Reg::RegTensor<int8_t> wShrReg;
    AscendC::Reg::RegTensor<int8_t> wShlReg;
    AscendC::Reg::RegTensor<int8_t> wAndReg;
    AscendC::Reg::RegTensor<int8_t> wLoad;
    AscendC::Reg::RegTensor<int8_t> wShl;
    AscendC::Reg::RegTensor<int8_t> wShr0;
    AscendC::Reg::RegTensor<int8_t> wShr1;
    AscendC::Reg::RegTensor<int8_t> wSel;
    AscendC::Reg::RegTensor<int8_t> wAnd;

    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg pregVsel = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShrReg, E2M1_SHIFT_RIGHT_SIZE, preg);
    AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShlReg, SHIFT_LEFT_SIZE, preg);
    AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wAndReg, E2M1_AND_MASK, preg);

    for (uint16_t loopKIdx = 0; loopKIdx < params.loopKNum; ++loopKIdx) {
        for (uint16_t innerLoopIdx = 0; innerLoopIdx < params.innerLoopNum; ++innerLoopIdx) {
            // DIST_US_B8 load mode expands each packed B4 byte into lane-aligned B8 slots.
            // Packed B4 address offset (bytes) = logical element index >> 1.
            AscendC::Reg::AddrReg aregWeightB8In = AscendC::Reg::CreateAddrReg<uint8_t>(
                loopKIdx, (C0_SIZE_B8 * params.nRealSizeAlign) >> 1, innerLoopIdx, AscendC::GetVecLen() >> 1);
            AscendC::Reg::LoadAlign<uint8_t, AscendC::Reg::LoadDist::DIST_US_B8>(
                (AscendC::Reg::RegTensor<uint8_t>&)wLoad, (__ubuf__ uint8_t*&)params.weight4BitPhyAddr, aregWeightB8In);

            AscendC::Reg::ShiftRight(wShr0, wLoad, wShrReg, preg);
            AscendC::Reg::ShiftLeft(wShl, wLoad, wShlReg, preg);
            AscendC::Reg::ShiftRight(wShr1, wShl, wShrReg, preg);
            AscendC::Reg::Select(wSel, wShr1, wShr0, pregVsel);
            AscendC::Reg::And(wAnd, wSel, wAndReg, preg);

            AscendC::Reg::AddrReg aregWeightB8Out = AscendC::Reg::CreateAddrReg<uint8_t>(
                loopKIdx, params.loopKDstStride, innerLoopIdx, params.innerDstStride);
            AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(
                (__ubuf__ uint8_t*&)params.weight8BitPhyAddr, (AscendC::Reg::RegTensor<uint8_t>&)wAnd, aregWeightB8Out,
                preg);
        }
    }
}

} // namespace Tile
