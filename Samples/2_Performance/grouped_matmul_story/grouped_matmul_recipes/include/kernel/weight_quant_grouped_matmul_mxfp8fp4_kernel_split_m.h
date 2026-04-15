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
 * \file weight_quant_grouped_matmul_mxfp8fp4_kernel_split_m.h
 * \brief Split-M kernel glue for weight-quant grouped matmul.
 */
#pragma once

#include "include/tensor.h"
#include "kernel_operator_list_tensor_intf.h"

namespace Kernel {

// Macro aliases keep long template specializations readable in declarations/definitions.
#define GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM \
    template <class ProblemShape, class BlockMmad, class BlockScheduler, class BlockEpilogue, class BlockPrologue>

#define GROUPED_MATMUL_RESPLIT_KERNEL_CLASS \
    WeightQuantGroupedMatmulMxfp8fp4Kernel<ProblemShape, BlockMmad, BlockScheduler, BlockEpilogue, BlockPrologue>

/*!
 * \brief Group-level orchestrator for the MXFP8(M1 input) + FP4(weight input) grouped matmul path.
 *
 * Design reason:
 * - The MMAD compute stage requires A/B to use the same compute representation.
 * - In this recipe, B is stored as FP4E2M1 in GM, but AIC cannot directly convert FP4E2M1 to FP8E4M3 inside MMAD.
 * - Therefore AIV runs prologue first to convert packed FP4E2M1 weight tiles into FP8E4M3-compatible B', then AIC
 *   consumes A, B', scaleA and scaleB for accumulation.
 *
 * Specific flow:
 * - Outer loop is group-based, and groupList[g] provides dynamic m_g for each group.
 * - For each group g:
 *   C_g = MatmulMX(A_g, B'_g, scaleA_g, scaleB_g)
 *   where A_g/B'_g are consumed by MMAD with a consistent compute type, and scaleA_g/scaleB_g are MX scales.
 * - AIC branch: tile scheduling + block MMAD.
 * - AIV branch: tile prologue (FP4E2M1 -> FP8E4M3) for B.
 *
 * Key constraints:
 * 1) groupList length and groupNum must match, and each entry is one group m_g.
 * 2) BlockMmad and BlockPrologue must use the same dispatch policy so they agree on tile boundaries and sync points.
 *
 * When to use:
 * - Use this kernel for the weight-quant grouped matmul path where weights are stored in FP4E2M1 and must be consumed
 *   by MMAD after prologue conversion to an A-compatible compute type.
 */
GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
class WeightQuantGroupedMatmulMxfp8fp4Kernel {
public:
    struct Params {
        ProblemShape problemShape;
        typename BlockMmad::Params mmad;
        typename BlockScheduler::Params scheduler;
        typename BlockPrologue::Params prologue;
        GM_ADDR ptrGroupList;
    };

    __aicore__ inline WeightQuantGroupedMatmulMxfp8fp4Kernel() = default;
    __aicore__ inline void operator()(const Params& params);

private:
    __gm__ typename BlockMmad::AType* xGm_;
    __gm__ typename BlockMmad::ScaleBType* antiquantScaleGm_;
    __gm__ typename BlockMmad::CType* yGm_;
    __gm__ typename BlockMmad::ScaleAType* perTokenScaleGm_;

    __gm__ typename BlockPrologue::InType* weightGm_;

    __gm__ int64_t* groupListGm_;

    using TensorLayoutGroupList = typename AscendC::Te::NDLayoutFormat<int64_t>;
};

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
__aicore__ inline void GROUPED_MATMUL_RESPLIT_KERNEL_CLASS::operator()(const Params& params)
{
    // AIC branch consumes A / scales and performs MMAD accumulation.
    if ASCEND_IS_AIC {
        xGm_ = reinterpret_cast<__gm__ typename BlockMmad::AType*>(params.mmad.ptrA);
        antiquantScaleGm_ = reinterpret_cast<__gm__ typename BlockMmad::ScaleBType*>(params.mmad.ptrScaleB);
        perTokenScaleGm_ = reinterpret_cast<__gm__ typename BlockMmad::ScaleAType*>(params.mmad.ptrScaleA);
        yGm_ = reinterpret_cast<__gm__ typename BlockMmad::CType*>(params.mmad.ptrC);
    }
    // AIV branch prepares transformed weights through the prologue path.
    if ASCEND_IS_AIV {
        weightGm_ = reinterpret_cast<__gm__ typename BlockPrologue::InType*>(params.prologue.ptrB);
    }
    BlockScheduler scheduler(params.scheduler);
    groupListGm_ = reinterpret_cast<__gm__ int64_t*>(params.ptrGroupList);
    const uint64_t kSize = AscendC::Std::get<1>(params.problemShape);
    const uint64_t nSize = AscendC::Std::get<2>(params.problemShape);
    const uint64_t groupNum = AscendC::Std::get<3>(params.problemShape);
    auto tensorGroupListGm =
        AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(groupListGm_), TensorLayoutGroupList{}(1, groupNum));
    using TensorLayoutB = typename BlockMmad::LayoutB;
    if ASCEND_IS_AIC {
        using TensorLayoutA = typename BlockMmad::LayoutA;
        using TensorLayoutC = typename BlockMmad::LayoutC;
        using TensorLayoutScaleA = typename BlockMmad::LayoutScaleA;
        using TensorLayoutScaleB = typename BlockMmad::LayoutScaleB;
        const uint64_t scaleKSize = CeilDiv(kSize, static_cast<uint64_t>(64)) * 2;
        BlockMmad blockMmad{};
        for (uint32_t groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
            uint64_t mSize = static_cast<uint64_t>(tensorGroupListGm[groupIdx]);
            if (mSize > 0 && nSize > 0) {
                auto tensorAGm =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(xGm_), TensorLayoutA{}(mSize, kSize));
                auto tensorScaleAGm = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeGMmemPtr(perTokenScaleGm_), TensorLayoutScaleA{}(mSize, scaleKSize));
                auto tensorScaleBGm = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeGMmemPtr(antiquantScaleGm_), TensorLayoutScaleB{}(scaleKSize, nSize));
                auto tensorCGm =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(yGm_), TensorLayoutC{}(mSize, nSize));

                scheduler.UpdateNextProblem(AscendC::Te::MakeShape(mSize, nSize, kSize));
                typename BlockScheduler::BlockCoord blockCoord;
                while (scheduler.GetTileIdx(blockCoord)) {
                    auto blockShape = scheduler.GetBlockShape(blockCoord);
                    auto mOffset = AscendC::Std::get<0>(blockCoord);
                    auto nOffset = AscendC::Std::get<1>(blockCoord);
                    auto mL1Size = AscendC::Std::get<0>(blockShape);
                    auto nL1Size = AscendC::Std::get<1>(blockShape);

                    auto tensorBlockAGm =
                        tensorAGm(AscendC::Te::MakeCoord(mOffset, 0), AscendC::Te::MakeShape(mL1Size, kSize));
                    auto tensorBlockScaleAGm =
                        tensorScaleAGm(AscendC::Te::MakeCoord(mOffset, 0), AscendC::Te::MakeShape(mL1Size, scaleKSize));
                    auto tensorBlockScaleBGm =
                        tensorScaleBGm(AscendC::Te::MakeCoord(0, nOffset), AscendC::Te::MakeShape(scaleKSize, nL1Size));
                    auto tensorBlockCGm =
                        tensorCGm(AscendC::Te::MakeCoord(mOffset, nOffset), AscendC::Te::MakeShape(mL1Size, nL1Size));
                    blockMmad(tensorBlockAGm, tensorBlockScaleAGm, tensorBlockScaleBGm, tensorBlockCGm);
                }
            }
            xGm_ += mSize * kSize;
            antiquantScaleGm_ += nSize * scaleKSize;
            perTokenScaleGm_ += mSize * scaleKSize;
            yGm_ += mSize * nSize;
        }
    } else {
        const uint64_t nAlign = AscendC::CeilAlign(nSize, static_cast<uint64_t>(BLOCK_CUBE));
        BlockPrologue blockPrologue;
        for (uint32_t groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
            uint64_t mSize = static_cast<uint64_t>(tensorGroupListGm[groupIdx]);
            if (mSize > 0 && nSize > 0) {
                scheduler.UpdateNextProblem(AscendC::Te::MakeShape(mSize, nSize, kSize));
                typename BlockScheduler::BlockCoord blockCoord;
                auto weightGmTensor = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeGMmemPtr(weightGm_),
                    TensorLayoutB{}(static_cast<int64_t>(kSize), static_cast<int64_t>(nAlign)));
                while (scheduler.GetTileIdx(blockCoord)) {
                    auto blockShape = scheduler.GetBlockShape(blockCoord);
                    auto nOffset = AscendC::Std::get<1>(blockCoord);
                    auto mL1Size = AscendC::Std::get<0>(blockShape);
                    auto nL1Size = AscendC::Std::get<1>(blockShape);
                    blockPrologue(weightGmTensor, mL1Size, kSize, nL1Size, nOffset, nAlign);
                }
            }
            // B4 is packed as two elements per byte, so address offset is in bytes.
            weightGm_ += (nSize * kSize) >> 1;
        }
    }
}
#undef GROUPED_MATMUL_RESPLIT_KERNEL_CLASS
#undef GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
} // namespace Kernel
