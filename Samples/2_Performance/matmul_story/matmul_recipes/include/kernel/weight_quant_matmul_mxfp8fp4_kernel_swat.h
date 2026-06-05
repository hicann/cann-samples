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
 * \file weight_quant_matmul_mxfp8fp4_kernel_swat.h
 * \brief Kernel orchestration for MXFP8 input and packed MXFP4 weight matmul.
 */
#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "include/tensor_api/tensor.h"
#include "kernel_utils/common_utils.h"
#include "../policy/dispatch_policy.h"
#include "../tile/copy_weight_gm_to_ub.h"
#include "../tile/copy_weight_ub_to_l1.h"
#include "../tile/shift_w4_to_w8.h"
#include "../utils/constant.h"
#include "../utils/layout_utils.h"

namespace Kernel {

template <typename T, uint64_t InnerStride>
struct Weight8BitUbLayout {
    __aicore__ inline decltype(auto) operator()(int64_t kSize, int64_t nSize)
    {
        static constexpr int64_t C0 = FP8_C0_SIZE;
        static constexpr int64_t VecRegElem = 256;
        static constexpr int64_t N0 = VecRegElem / C0;
        int64_t k1 = CeilDiv(kSize, C0);
        int64_t n1 = CeilDiv(nSize, N0);

        auto shape = AscendC::Te::MakeShape(
            AscendC::Te::MakeShape(AscendC::Std::Int<C0>{}, k1), AscendC::Te::MakeShape(AscendC::Std::Int<N0>{}, n1));
        auto stride = AscendC::Te::MakeStride(
            AscendC::Te::MakeStride(AscendC::Std::Int<1>{}, n1 * AscendC::Std::Int<InnerStride>{}),
            AscendC::Te::MakeStride(AscendC::Std::Int<C0>{}, AscendC::Std::Int<InnerStride>{}));
        return AscendC::Te::MakeLayout(shape, stride);
    }
};

template <class DispatchPolicy_, typename OutType_, typename InType_>
class WeightQuantMatmulMxfp8Fp4BlockPrologue {
public:
    using OutType = OutType_;
    using InType = InType_;
    using DispatchPolicy = DispatchPolicy_;
    static constexpr uint64_t L1_BUFFER_NUM = DispatchPolicy::l1BufNum;
    static constexpr uint64_t L1_BUFFER_MASK = L1_BUFFER_NUM - 1U;
    static constexpr bool USE_COMPACT_L1_LAYOUT = L1_BUFFER_NUM != L1_FOUR_BUFFER;
    static_assert(
        L1_BUFFER_NUM == DB_SIZE || L1_BUFFER_NUM == L1_FOUR_BUFFER,
        "MXFP8FP4 targets support only 2 or 4 L1 buffers.");

    struct Params {
        GM_ADDR bGmAddr{nullptr};
        uint64_t baseN{0};
        uint64_t kL1Size{0};
        // UB split is K-only: nUbSize is capacity, while the active N comes from the sliced GM tensor.
        uint64_t kUbSize{0};
        uint64_t nUbSize{0};
    };

    __aicore__ inline explicit WeightQuantMatmulMxfp8Fp4BlockPrologue(const Params& params)
    {
        l1BaseN_ = params.baseN;
        kL1Size_ = params.kL1Size;
        nUbSize_ = params.nUbSize;
        kUbSize_ = params.kUbSize;
        vecWeightInLenBytes_ = (L1_BUFFER_NUM * nUbSize_ * kUbSize_) >> INT4_DTYPE_PARAM;
        vecWeightOutLenBytes_ = VECTOR_WEIGHT_BUFFER_NUM * AscendC::CeilAlign(nUbSize_, CUBE_BLOCK) *
                                AscendC::CeilAlign(kUbSize_, MXFP_DIVISOR_SIZE);
        weightInBaseOffset_ = vecWeightOutLenBytes_;
        singleWeightInLenBytes_ = vecWeightInLenBytes_ / L1_BUFFER_NUM;
    }

    __aicore__ inline ~WeightQuantMatmulMxfp8Fp4BlockPrologue()
    {
        // Drain outstanding vector copies before releasing the final AIC/AIV handshakes.
        int64_t buffNum = Min(idx_ + 1, static_cast<int64_t>(L1_BUFFER_NUM));
        for (int64_t index = 0; index < buffNum; ++index) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(index);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(index);
        }
#pragma unroll
        for (int8_t index = 0; index < static_cast<int8_t>(L1_BUFFER_NUM); ++index) {
            WaitWeightFlag<AIC_SYNC_AIV_FLAG>();
        }
    }

    template <typename GMWeightTensor>
    __aicore__ inline void operator()(const GMWeightTensor& gmWeightTensor)
    {
        const auto& layout = gmWeightTensor.Layout();
        kSize_ = static_cast<uint64_t>(AscendC::Te::GetTotalRowShape(layout));
        nL1Len_ = static_cast<int32_t>(AscendC::Te::GetTotalColumnShape(layout));
        nL1PadLen_ = AscendC::CeilAlign(static_cast<uint64_t>(nL1Len_), CUBE_BLOCK);
        uint64_t kL1PadSize = AscendC::CeilAlign(kL1Size_, MXFP_DIVISOR_SIZE);
        // Buffer stride stays fixed at baseN so AIV producer and AIC consumer use the same L1 slot base.
        bL1BufferSize_ = l1BaseN_ * kL1PadSize;
        InitL1BufferOffsets();
        const uint64_t kTileCount = CeilDiv(kSize_, kL1Size_);
        for (uint64_t kLoopIdx = 0; kLoopIdx < kTileCount; ++kLoopIdx) {
            kGmOffset_ = kLoopIdx * kL1Size_;
            kL1Len_ = static_cast<int32_t>(Min(kSize_ - kGmOffset_, kL1Size_));
            // Current prologue does not split N; only K may be divided between the two AIV subblocks.
            nUbLen_ = nL1Len_;
            kUbLen_ = kL1Len_;
            nL1Offset_ = 0;
            kL1Offset_ = 0;
            if (UpdateUbSliceForCurrentSubBlock()) {
                ConvertCurrentWeightSlice(gmWeightTensor);
            } else {
                NotifyCurrentWeightSliceReady();
            }
            l1BufIdx_ = (l1BufIdx_ + 1) & L1_BUFFER_MASK;
        }
    }

private:
    template <uint64_t FLAG>
    __aicore__ inline void WaitWeightFlag() const
    {
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(FLAG);
    }

    template <uint64_t FLAG>
    __aicore__ inline void SetWeightFlag() const
    {
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(FLAG);
    }

    __aicore__ inline void InitL1BufferOffsets()
    {
        for (uint16_t bufIdx = 0; bufIdx < L1_BUFFER_NUM; ++bufIdx) {
            if constexpr (USE_COMPACT_L1_LAYOUT) {
                l1BufferWeightOffset_[bufIdx] = bufIdx * bL1BufferSize_;
            } else {
                // Four-buffer target partitions L1 into two halves, each holding a double-buffer pair.
                l1BufferWeightOffset_[bufIdx] = (bufIdx & 1U) * L1_HALF_SIZE + (bufIdx >> 1U) * bL1BufferSize_;
            }
        }
    }

    __aicore__ inline bool UpdateUbSliceForCurrentSubBlock()
    {
        // For short K, only AIV subblock 0 performs the conversion; subblock 1 still participates in sync.
        if (kL1Len_ <= static_cast<int32_t>(kUbSize_)) {
            if (AscendC::GetSubBlockIdx() == 0) {
                return true;
            }
            kUbLen_ = 0;
            return false;
        }
        kUbLen_ = static_cast<int32_t>(kUbSize_);
        if (AscendC::GetSubBlockIdx() == 1) {
            kL1Offset_ = kUbLen_;
            kUbLen_ = kL1Len_ - kUbLen_;
        }
        return kUbLen_ > 0;
    }

    template <typename GMWeightTensor>
    __aicore__ inline void ConvertCurrentWeightSlice(const GMWeightTensor& gmWeightTensor)
    {
        WaitWeightFlag<AIC_SYNC_AIV_FLAG>();
        CopyConvertStoreWeight(gmWeightTensor, GetL1WeightOffset());
        SetWeightFlag<AIV_SYNC_AIC_FLAG>();
    }

    __aicore__ inline void NotifyCurrentWeightSliceReady()
    {
        // Keep the AIC wait count balanced when this AIV subblock has no K slice to convert.
        WaitWeightFlag<AIC_SYNC_AIV_FLAG>();
        SetWeightFlag<AIV_SYNC_AIC_FLAG>();
    }

    __aicore__ inline int64_t GetL1WeightOffset() const
    {
        uint64_t sliceOffset = nL1Offset_ * FP8_C0_SIZE + kL1Offset_ * nL1PadLen_;
        return static_cast<int64_t>(l1BufferWeightOffset_[l1BufIdx_] + sliceOffset);
    }

    template <typename GMWeightTensor>
    __aicore__ inline void CopyConvertStoreWeight(const GMWeightTensor& gmWeightTensor, int64_t l1Offset)
    {
        idx_ += 1;
        ubBufIdx_ = static_cast<uint64_t>(idx_) & L1_BUFFER_MASK;
        if (idx_ >= static_cast<int64_t>(L1_BUFFER_NUM)) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubBufIdx_);
        }
        auto weight4BitTensor = MakeWeight4BitTensor();
        CopyPackedWeightGmToUb(gmWeightTensor, weight4BitTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ubBufIdx_);
        if (idx_ >= static_cast<int64_t>(L1_BUFFER_NUM)) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubBufIdx_);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ubBufIdx_);
        auto weight8BitTensor = MakeWeight8BitTensor();
        ::Tile::ShiftW4ToW8<OutType, InType>(weight4BitTensor, weight8BitTensor);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubBufIdx_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubBufIdx_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubBufIdx_);
        auto l1Tensor = MakeL1WeightTensor(l1Offset);
        auto copyUB2L1 = AscendC::Te::MakeCopy(::Tile::CopyUB2L1Weight{});
        AscendC::Te::Copy(copyUB2L1, l1Tensor, weight8BitTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubBufIdx_);
    }

    __aicore__ inline auto MakeWeight4BitTensor()
    {
        uint64_t offset = weightInBaseOffset_ + ubBufIdx_ * singleWeightInLenBytes_;
        return AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, InType>(offset),
            MatmulRecipe::Weight4BitNzLayout<OutType>{}(kUbLen_, nUbLen_));
    }

    __aicore__ inline auto MakeWeight8BitTensor()
    {
        uint64_t offset = ubBufIdx_ * VEC_MAX_ELEM_B8;
        return AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, OutType>(offset),
            Weight8BitUbLayout<OutType, VEC_MAX_ELEM_B8 * VECTOR_WEIGHT_BUFFER_NUM>{}(kUbLen_, nUbLen_));
    }

    __aicore__ inline auto MakeL1WeightTensor(int64_t l1Offset)
    {
        auto layout =
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<FP8_C0_SIZE>>(kUbLen_, nUbLen_);
        return AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, OutType>(static_cast<uint64_t>(l1Offset)), layout);
    }

    template <typename GMWeightTensor, typename Weight4BitTensor>
    __aicore__ inline void CopyPackedWeightGmToUb(
        const GMWeightTensor& gmWeightTensor, const Weight4BitTensor& weight4BitTensor)
    {
        if (unlikely(kUbLen_ <= 0 || nUbLen_ <= 0)) {
            return;
        }
        auto gmSlice = gmWeightTensor.Slice(
            AscendC::Te::MakeCoord(static_cast<int64_t>(kGmOffset_ + kL1Offset_), static_cast<int64_t>(nL1Offset_)),
            AscendC::Te::MakeShape(static_cast<int64_t>(kUbLen_), static_cast<int64_t>(nUbLen_)));
        auto copyGM2UB = AscendC::Te::MakeCopy(::Tile::CopyGM2UBPackedWeight{});
        AscendC::Te::Copy(copyGM2UB, weight4BitTensor, gmSlice);
    }

private:
    uint64_t kSize_{0};
    uint64_t nUbSize_{0};
    uint64_t kUbSize_{0};
    int32_t nUbLen_{0};
    int32_t kUbLen_{0};
    uint64_t kL1Size_{0};
    uint64_t l1BaseN_{0};
    uint64_t kGmOffset_{0};
    int32_t nL1Len_{0};
    uint64_t nL1PadLen_{0};
    int32_t kL1Len_{0};
    uint64_t bL1BufferSize_{0};
    uint64_t l1BufferWeightOffset_[L1_BUFFER_NUM] = {0UL};
    uint64_t vecWeightOutLenBytes_{0};
    uint64_t vecWeightInLenBytes_{0};
    uint64_t singleWeightInLenBytes_{0};
    uint64_t weightInBaseOffset_{0};
    uint64_t ubBufIdx_{0};
    uint64_t l1BufIdx_{0};
    int64_t idx_{-1};
    uint64_t nL1Offset_{0};
    uint64_t kL1Offset_{0};

    static constexpr uint64_t VECTOR_WEIGHT_BUFFER_NUM = L1_BUFFER_NUM;
    static constexpr uint64_t INT4_DTYPE_PARAM = 1;
    static constexpr int32_t VECTOR_REG_BYTES = 256;
    static constexpr int32_t VEC_MAX_ELEM_B8 = VECTOR_REG_BYTES / sizeof(OutType);
};

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
class GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<BlockEpilogue, void> &&
        AscendC::Std::is_same_v<KernelMixMmadWithScaleMx, typename BlockMmad::DispatchPolicy::ScheduleType>>> {
public:
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using ScaleAType = typename BlockMmad::ScaleAType;
    using ScaleBType = typename BlockMmad::ScaleBType;
    using CType = typename BlockMmad::CType;

    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutScaleA = typename BlockMmad::LayoutScaleA;
    using LayoutScaleB = typename BlockMmad::LayoutScaleB;
    using BlockPrologue = WeightQuantMatmulMxfp8Fp4BlockPrologue<typename BlockMmad::DispatchPolicy, AType, BType>;

    struct Params {
        ProblemShape problemShape;
        typename BlockMmad::Params mmad;
        typename BlockScheduler::Params scheduler;
    };

    GemmUniversal() = delete;

    __aicore__ inline static void Run(const Params& params)
    {
        BlockScheduler scheduler(params.scheduler);
        if ASCEND_IS_AIV {
            RunAiv(params, scheduler);
        }
        if ASCEND_IS_AIC {
            RunAic(params, scheduler);
        }
    }

private:
    __aicore__ inline static void RunAiv(const Params& params, const BlockScheduler& scheduler)
    {
        auto gmWeight = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(reinterpret_cast<__gm__ BType*>(params.mmad.bGmAddr)),
            LayoutB{}(static_cast<int64_t>(params.problemShape.k), static_cast<int64_t>(params.problemShape.n)));
        uint64_t tileNum = scheduler.GetTileNum();
        uint64_t curBlockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        typename BlockPrologue::Params prologueParams{
            params.mmad.bGmAddr, static_cast<uint64_t>(AscendC::Std::get<1>(params.mmad.l1TileShape)),
            static_cast<uint64_t>(AscendC::Std::get<2>(params.mmad.l1TileShape)), params.mmad.kBubSize,
            params.mmad.nBubSize};
        BlockPrologue blockPrologue(prologueParams);
        for (uint64_t loopIdx = curBlockIdx; loopIdx < tileNum; loopIdx += AscendC::GetBlockNum()) {
            auto blockCoord = scheduler.GetBlockCoord(loopIdx);
            auto blockShape = scheduler.GetBlockShape(blockCoord);
            uint64_t nOffset = static_cast<uint64_t>(AscendC::Std::get<1>(blockCoord));
            uint64_t kSize = static_cast<uint64_t>(AscendC::Std::get<2>(blockShape));
            uint64_t nL1Size = static_cast<uint64_t>(AscendC::Std::get<1>(blockShape));
            auto gmBlockWeight = gmWeight.Slice(
                AscendC::Te::MakeCoord(0, static_cast<int64_t>(nOffset)),
                AscendC::Te::MakeShape(static_cast<int64_t>(kSize), static_cast<int64_t>(nL1Size)));
            blockPrologue(gmBlockWeight);
        }
    }

    __aicore__ inline static void RunAic(const Params& params, const BlockScheduler& scheduler)
    {
        uint64_t scaleKSize =
            CeilDiv(AscendC::CeilAlign(params.problemShape.k, TILING_MXFP_DIVISOR_SIZE), MX_GROUP_SIZE);
        auto gmA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(reinterpret_cast<__gm__ AType*>(params.mmad.aGmAddr)),
            LayoutA{}(static_cast<int64_t>(params.problemShape.m), static_cast<int64_t>(params.problemShape.k)));
        auto gmScaleA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ ScaleAType*>(params.mmad.scaleAGmAddr)),
            LayoutScaleA{}(static_cast<int64_t>(params.problemShape.m), static_cast<int64_t>(scaleKSize)));
        auto gmScaleB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ ScaleBType*>(params.mmad.scaleBGmAddr)),
            LayoutScaleB{}(static_cast<int64_t>(scaleKSize), static_cast<int64_t>(params.problemShape.n)));
        auto gmC = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(reinterpret_cast<__gm__ CType*>(params.mmad.cGmAddr)),
            LayoutC{}(static_cast<int64_t>(params.problemShape.m), static_cast<int64_t>(params.problemShape.n)));

        uint64_t tileNum = scheduler.GetTileNum();
        BlockMmad blockMmad(params.mmad);
        for (uint64_t loopIdx = AscendC::GetBlockIdx(); loopIdx < tileNum; loopIdx += AscendC::GetBlockNum()) {
            auto blockCoord = scheduler.GetBlockCoord(loopIdx);
            auto blockShape = scheduler.GetBlockShape(blockCoord);
            uint64_t mOffset = static_cast<uint64_t>(AscendC::Std::get<0>(blockCoord));
            uint64_t nOffset = static_cast<uint64_t>(AscendC::Std::get<1>(blockCoord));
            uint64_t mL1Size = static_cast<uint64_t>(AscendC::Std::get<0>(blockShape));
            uint64_t nL1Size = static_cast<uint64_t>(AscendC::Std::get<1>(blockShape));
            uint64_t kSize = static_cast<uint64_t>(AscendC::Std::get<2>(blockShape));

            auto gmBlockA = gmA.Slice(
                AscendC::Te::MakeCoord(static_cast<int64_t>(mOffset), 0),
                AscendC::Te::MakeShape(static_cast<int64_t>(mL1Size), static_cast<int64_t>(kSize)));
            auto gmBlockScaleA = gmScaleA.Slice(
                AscendC::Te::MakeCoord(static_cast<int64_t>(mOffset), 0),
                AscendC::Te::MakeShape(static_cast<int64_t>(mL1Size), static_cast<int64_t>(scaleKSize)));
            auto gmBlockScaleB = gmScaleB.Slice(
                AscendC::Te::MakeCoord(0, static_cast<int64_t>(nOffset)),
                AscendC::Te::MakeShape(static_cast<int64_t>(scaleKSize), static_cast<int64_t>(nL1Size)));
            auto gmBlockC = gmC.Slice(
                AscendC::Te::MakeCoord(static_cast<int64_t>(mOffset), static_cast<int64_t>(nOffset)),
                AscendC::Te::MakeShape(static_cast<int64_t>(mL1Size), static_cast<int64_t>(nL1Size)));
            blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, gmBlockC);
        }
    }
};

} // namespace Kernel
