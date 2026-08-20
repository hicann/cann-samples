/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../kimi_delta_attn_lite_common.h"

#include <basic_api/reg_compute/kernel_reg_compute_utils.h>
#include <kernel_operator.h>

namespace KDALite {

// CANN 9.0 的 Host 预处理阶段不暴露 AscendC::MutexID, 设备侧底层类型为 uint8_t.
using MutexId = uint8_t;

// 上板时使用 mode2 同步同一 Mix 组的 1 个 AIC 和 2 个 AIV. 仿真模式改用
// mode4 分别同步两路 AIV, 规避 CANNsim 对多轮 mode2 支持不完整的问题.
constexpr uint8_t GROUP_CROSS_MODE = 2;
#ifdef SIM_COMPATIBLE
constexpr uint8_t PAIR_CROSS_MODE = 4;
constexpr uint16_t AIV1_FLAG_OFFSET = 16;
#endif

constexpr uint32_t DB_SLOT_NUM = 2;
constexpr uint32_t L0C_QUEUE_DEPTH = 4;
constexpr uint32_t STATE_PIPELINE_MAX_LANE_NUM = 4;
constexpr uint32_t STATE_R_MAX_QUEUE_DEPTH = STATE_PIPELINE_MAX_LANE_NUM - 1;

// StateOutput 的多 lane 路径使用 state, U/prediction, R 和 V2 阶段四组 CrossCore 交接.
// V2 阶段先成组交接 history/delta, 再交接 local. state 按物理 lane 分配 4 个 FlagID,
// R 队列最多使用 3 个 FlagID. Ascend 950 的逻辑 FlagID 只能取 0..10.
constexpr uint16_t FLAG_STATE_HANDOFF_BASE = 0;
constexpr uint16_t FLAG_U_PRED_HANDOFF_BASE = FLAG_STATE_HANDOFF_BASE + STATE_PIPELINE_MAX_LANE_NUM;
constexpr uint16_t FLAG_R_HANDOFF_BASE = FLAG_U_PRED_HANDOFF_BASE + DB_SLOT_NUM;
constexpr uint16_t FLAG_V2_PHASE_HANDOFF_BASE = FLAG_R_HANDOFF_BASE + STATE_R_MAX_QUEUE_DEPTH;
// wave 结束前 state lane 0 通道已排空, 可复用其 FlagID 传递 wave 完成信号.
constexpr uint16_t FLAG_STATE_WAVE_DONE = FLAG_STATE_HANDOFF_BASE;
static_assert(FLAG_V2_PHASE_HANDOFF_BASE + DB_SLOT_NUM - 1 <= 10, "StateOutput multi-lane CrossCore FlagID exceeds 10");

// 单 lane 路径使用独立的 delta 和 history/local 交接协议.
// 单 lane 与多 lane 是互斥的运行时分支, 因此其他通道可复用多 lane 的 FlagID.
constexpr uint16_t FLAG_SINGLE_LANE_R_HANDOFF = 2;
constexpr uint16_t FLAG_SINGLE_LANE_DELTA_HANDOFF_BASE = 6;
constexpr uint16_t FLAG_SINGLE_LANE_HISTORY_LOCAL_HANDOFF_BASE = 8;
static_assert(
    FLAG_SINGLE_LANE_HISTORY_LOCAL_HANDOFF_BASE + DB_SLOT_NUM - 1 <= 10,
    "StateOutput single-lane CrossCore FlagID exceeds 10");

constexpr uint32_t C0_BYTES = 32;
constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t DV_TILE_COUNT = VALUE_DIM / DV_TILE;
constexpr uint32_t CHUNK_D_ELEMS = CHUNK_SIZE * HEAD_DIM;
constexpr uint32_t CHUNK_C_ELEMS = CHUNK_SIZE * CHUNK_SIZE;
constexpr uint32_t STATE_TILE_ELEMS = HEAD_DIM * DV_TILE;
constexpr uint32_t CHUNK_DV_TILE_ELEMS = CHUNK_SIZE * DV_TILE;
constexpr uint32_t STATE_HALF_ELEMS = HEAD_DIM * AIV_DV_TILE;
constexpr uint32_t CHUNK_DV_HALF_ELEMS = CHUNK_SIZE * AIV_DV_TILE;
constexpr uint32_t AIV_USABLE_UB_BYTES = 248 * 1024;

// Prepare 用两个 CV 时间槽滚动发射 VP/C/VS. 每个时间槽含两路 AIV 子槽,
// 每个子槽保存一个 chunk 的 QFactor/KFactor/KInvFactor.
constexpr uint32_t PREP_CV_SLOT_NUM = 2;
constexpr uint32_t PREP_SUB_AIV_NUM = 2;
constexpr uint32_t PREP_K_FACTOR_L1_ADDR = 0;
constexpr uint32_t PREP_Q_FACTOR_L1_ADDR = PREP_K_FACTOR_L1_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_K_INV_FACTOR_L1_ADDR = PREP_Q_FACTOR_L1_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_L1_SLOT_BYTES = PREP_K_INV_FACTOR_L1_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_L1_CV_SLOT_BYTES = PREP_L1_SLOT_BYTES * PREP_SUB_AIV_NUM;
// AIV 写入三组 factor 后交给 AIC; AIC 的 MTE1 读完后归还该 L1 时间槽.
constexpr uint16_t FLAG_PREP_L1_HANDOFF_BASE = 0;
// AIC 通过 Fixpipe 写出 Pair/Araw, AIV 消费后归还对应的结果槽.
constexpr uint16_t FLAG_PREP_PAIR_ARAW_HANDOFF_BASE = FLAG_PREP_L1_HANDOFF_BASE + PREP_CV_SLOT_NUM;
static_assert(PREP_L1_CV_SLOT_BYTES * PREP_CV_SLOT_NUM <= 512 * 1024, "ChunkPrepare L1 allocation exceeds 512 KiB");
static_assert(FLAG_PREP_PAIR_ARAW_HANDOFF_BASE + PREP_CV_SLOT_NUM - 1 <= 10, "Prepare CrossCore FlagID exceeds 10");

constexpr uint32_t PREP_Q_BF16_SLOT_ADDR = 0;
constexpr uint32_t PREP_K_BF16_SLOT_ADDR = PREP_Q_BF16_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_G_FP32_SLOT_ADDR = PREP_K_BF16_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_BETA_BF16_SLOT_ADDR = PREP_G_FP32_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(float);
constexpr uint32_t PREP_K_PLUS_BF16_SLOT_ADDR = PREP_BETA_BF16_SLOT_ADDR + CHUNK_SIZE * sizeof(bfloat16_t);
constexpr uint32_t PREP_Q_FACTOR_BF16_SLOT_ADDR = PREP_K_PLUS_BF16_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_K_FACTOR_BF16_SLOT_ADDR = PREP_Q_FACTOR_BF16_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_K_INV_FACTOR_BF16_SLOT_ADDR = PREP_K_FACTOR_BF16_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_K_TAIL_BF16_SLOT_ADDR = PREP_K_INV_FACTOR_BF16_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_STATE_DECAY_FP32_SLOT_ADDR = PREP_K_TAIL_BF16_SLOT_ADDR + CHUNK_D_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_PAIR_FP32_UB_ADDR = PREP_STATE_DECAY_FP32_SLOT_ADDR + HEAD_DIM * sizeof(float);
constexpr uint32_t PREP_A_RAW_FP32_UB_ADDR = PREP_PAIR_FP32_UB_ADDR + CHUNK_C_ELEMS * sizeof(float);
constexpr uint32_t PREP_M_FP32_SLOT_ADDR = PREP_A_RAW_FP32_UB_ADDR + CHUNK_C_ELEMS * sizeof(float);
constexpr uint32_t PREP_M_BF16_SLOT_ADDR = PREP_M_FP32_SLOT_ADDR + CHUNK_C_ELEMS * sizeof(float);
constexpr uint32_t PREP_A_BF16_SLOT_ADDR = PREP_M_BF16_SLOT_ADDR + CHUNK_C_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_SLOT_BYTES = PREP_A_BF16_SLOT_ADDR + CHUNK_C_ELEMS * sizeof(bfloat16_t);
static_assert(PREP_SLOT_BYTES * PREP_CV_SLOT_NUM <= AIV_USABLE_UB_BYTES, "ChunkPrepare UB allocation exceeds 248 KiB");

// Fixpipe 结果位于每路 AIV 的本地 UB. U/prediction 和 V2 阶段分别使用双槽,
// 允许相邻 item 的 CrossCore 交接同时在途.
constexpr uint32_t PRED_UB_ADDR = 0;
constexpr uint32_t HISTORY_UB_ADDR = PRED_UB_ADDR + CHUNK_DV_HALF_ELEMS * sizeof(float) * DB_SLOT_NUM;
constexpr uint32_t DELTA_UB_ADDR = HISTORY_UB_ADDR + CHUNK_DV_HALF_ELEMS * sizeof(float) * DB_SLOT_NUM;
constexpr uint32_t LOCAL_UB_ADDR = DELTA_UB_ADDR + STATE_HALF_ELEMS * sizeof(float) * DB_SLOT_NUM;
constexpr uint32_t U_UB_ADDR = LOCAL_UB_ADDR + CHUNK_DV_HALF_ELEMS * sizeof(float) * DB_SLOT_NUM;
constexpr uint32_t HANDOFF_UB_END_ADDR = U_UB_ADDR + CHUNK_DV_HALF_ELEMS * sizeof(float) * DB_SLOT_NUM;

constexpr AscendC::FixpipeConfig KDA_FIXPIPE_CFG_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

__aicore__ inline uint16_t SlotFlagId(uint16_t baseFlagId, uint32_t slot)
{
    return baseFlagId + static_cast<uint16_t>(slot);
}

template <pipe_t PIPE>
__aicore__ inline void SetAicToAiv(uint16_t flagId)
{
    using namespace AscendC;
#ifdef SIM_COMPATIBLE
    CrossCoreSetFlag<PAIR_CROSS_MODE, PIPE>(flagId);
    CrossCoreSetFlag<PAIR_CROSS_MODE, PIPE>(flagId + AIV1_FLAG_OFFSET);
#else
    CrossCoreSetFlag<GROUP_CROSS_MODE, PIPE>(flagId);
#endif
}

template <pipe_t PIPE>
__aicore__ inline void WaitAicToAiv(uint16_t flagId)
{
    using namespace AscendC;
#ifdef SIM_COMPATIBLE
    CrossCoreWaitFlag<PAIR_CROSS_MODE, PIPE>(flagId);
#else
    CrossCoreWaitFlag<GROUP_CROSS_MODE, PIPE>(flagId);
#endif
}

template <pipe_t PIPE>
__aicore__ inline void SetAivToAic(uint16_t flagId)
{
    using namespace AscendC;
#ifdef SIM_COMPATIBLE
    CrossCoreSetFlag<PAIR_CROSS_MODE, PIPE>(flagId);
#else
    CrossCoreSetFlag<GROUP_CROSS_MODE, PIPE>(flagId);
#endif
}

template <pipe_t PIPE>
__aicore__ inline void WaitAivToAic(uint16_t flagId)
{
    using namespace AscendC;
#ifdef SIM_COMPATIBLE
    CrossCoreWaitFlag<PAIR_CROSS_MODE, PIPE>(flagId);
    CrossCoreWaitFlag<PAIR_CROSS_MODE, PIPE>(flagId + AIV1_FLAG_OFFSET);
#else
    CrossCoreWaitFlag<GROUP_CROSS_MODE, PIPE>(flagId);
#endif
}

template <typename R, typename T1, typename T2>
__aicore__ constexpr R CeilDiv(T1 x, T2 y)
{
    return static_cast<R>((x + y - 1) / y);
}

template <typename R, typename T1, typename T2>
__aicore__ constexpr R CeilAlign(T1 x, T2 align)
{
    return static_cast<R>(CeilDiv<R>(x, align) * align);
}

template <typename T>
__aicore__ constexpr uint32_t C0ElemNum()
{
    return C0_BYTES / sizeof(T);
}

template <typename T>
__aicore__ inline void CopyGmToL1(
    const AscendC::LocalTensor<T>& dstL1Local, const AscendC::GlobalTensor<T>& srcGlobal, uint32_t tileRows,
    uint32_t tileCols, uint32_t gmRowStride, uint32_t l1RowStride)
{
    using namespace AscendC;
    Nd2NzParams params;
    params.ndNum = 1;
    params.nValue = tileRows;
    params.dValue = tileCols;
    params.srcNdMatrixStride = 1;
    params.srcDValue = gmRowStride;
    params.dstNzC0Stride = CeilAlign<uint16_t>(l1RowStride, CUBE_BLOCK);
    params.dstNzNStride = 1;
    params.dstNzMatrixStride = 1;
    DataCopy(dstL1Local, srcGlobal, params);
}

template <typename T>
__aicore__ inline void CopyGmToL1(
    const AscendC::LocalTensor<T>& dstL1Local, const AscendC::GlobalTensor<T>& srcGlobal, uint32_t tileRows,
    uint32_t tileCols, uint32_t gmRowStride)
{
    CopyGmToL1(dstL1Local, srcGlobal, tileRows, tileCols, gmRowStride, tileRows);
}

template <typename T>
__aicore__ inline void CopyGmToUbRows(
    const AscendC::LocalTensor<T>& dstUBLocal, const AscendC::GlobalTensor<T>& srcGlobal, uint32_t rows, uint32_t cols,
    uint32_t gmRowStride)
{
    using namespace AscendC;
    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(rows);
    params.blockLen = cols * sizeof(T);
    params.srcStride = (gmRowStride - cols) * sizeof(T);
    params.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = static_cast<T>(0);
    DataCopyPad(dstUBLocal, srcGlobal, params, padParams);
}

template <typename T>
__aicore__ inline void CopyUbToGmRows(
    const AscendC::GlobalTensor<T>& dstGlobal, const AscendC::LocalTensor<T>& srcUBLocal, uint32_t rows, uint32_t cols,
    uint32_t ubRowStride, uint32_t gmRowStride)
{
    using namespace AscendC;
    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(rows);
    params.blockLen = cols * sizeof(T);
    params.srcStride = (ubRowStride - cols) * sizeof(T) / C0_BYTES;
    params.dstStride = (gmRowStride - cols) * sizeof(T);
    DataCopyPad(dstGlobal, srcUBLocal, params);
}

template <typename T>
__aicore__ inline void CopyL1ToL0A(
    const AscendC::LocalTensor<T>& dstL0ALocal, const AscendC::LocalTensor<T>& srcL1Local, uint32_t l1Rows,
    uint32_t l0Rows, uint32_t l0Cols, bool transpose = false)
{
    using namespace AscendC;
    LoadData2DParamsV2 params;
    params.mStartPosition = 0;
    params.kStartPosition = 0;
    params.srcStride = CeilDiv<int32_t>(l1Rows, CUBE_BLOCK);
    params.ifTranspose = transpose;
    if (transpose) {
        params.mStep = CeilDiv<uint16_t>(l0Cols, CUBE_BLOCK);
        params.kStep = CeilDiv<uint16_t>(l0Rows, C0ElemNum<T>());
        params.dstStride = CeilDiv<uint16_t>(l0Rows, CUBE_BLOCK);
    } else {
        params.mStep = CeilDiv<uint16_t>(l0Rows, CUBE_BLOCK);
        params.kStep = CeilDiv<uint16_t>(l0Cols, C0ElemNum<T>());
        params.dstStride = params.mStep;
    }
    LoadData(dstL0ALocal, srcL1Local, params);
}

template <typename T>
__aicore__ inline void CopyL1ToL0B(
    const AscendC::LocalTensor<T>& dstL0BLocal, const AscendC::LocalTensor<T>& srcL1Local, uint32_t l1Rows,
    uint32_t l0Rows, uint32_t l0Cols, bool transpose = true)
{
    using namespace AscendC;
    LoadData2DParamsV2 params;
    params.mStartPosition = 0;
    params.kStartPosition = 0;
    params.mStep = CeilDiv<uint16_t>(l0Rows, CUBE_BLOCK);
    params.kStep = CeilDiv<uint16_t>(l0Cols, C0ElemNum<T>());
    params.srcStride = CeilDiv<int32_t>(l1Rows, CUBE_BLOCK);
    params.ifTranspose = transpose;
    // 非转置模式用于已经按 [N,K] 组织的右矩阵, L0B 的连续方向是 N;
    // 转置模式沿用 [K,N] 输入的 ZN 目标跨度.
    params.dstStride = transpose ? params.kStep : params.mStep;
    LoadData(dstL0BLocal, srcL1Local, params);
}

template <typename DstT, typename AT, typename BT>
__aicore__ inline void CubeMmad(
    const AscendC::LocalTensor<DstT>& dstL0CLocal, const AscendC::LocalTensor<AT>& aL0ALocal,
    const AscendC::LocalTensor<BT>& bL0BLocal, uint32_t m, uint32_t n, uint32_t k)
{
    using namespace AscendC;
    MmadParams params;
    params.m = m;
    params.n = n;
    params.k = k;
    params.cmatrixSource = false;
    params.cmatrixInitVal = true;
    params.unitFlag = 0;
    Mmad(dstL0CLocal, aL0ALocal, bL0BLocal, params);
}

template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeToVecUB(
    const AscendC::LocalTensor<DstT>& dstUBLocal, const AscendC::LocalTensor<SrcT>& srcL0CLocal, uint32_t m, uint32_t n)
{
    using namespace AscendC;
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> params;
    params.nSize = n;
    params.mSize = m;
    params.srcStride = CeilAlign<uint32_t>(m, CUBE_BLOCK);
    params.dstStride = n / 2;
    params.dualDstCtl = 0b10;
    params.params.ndNum = 1;
    params.params.srcNdStride = 0;
    params.params.dstNdStride = 0;
    Fixpipe<DstT, SrcT, KDA_FIXPIPE_CFG_UB>(dstUBLocal, srcL0CLocal, params);
}

} // namespace KDALite
