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

constexpr uint16_t FLAG_PREP_INPUT_READY = 0;
// M/K_plus 写入共享 L1 后由 AIV 置位; AIC 完成两次 MTE1 读取后释放 L1.
constexpr uint16_t FLAG_PREP_L1_FREE = 1;

// prediction 和 state delta 复用 AIV 的 resultUB. MM7 Fix 覆写 resultUB 前,
// PRED_CONSUMED 必须确认 prediction 已读完.
constexpr uint16_t FLAG_STATE_INPUT_READY = 0;
constexpr uint16_t FLAG_STATE_PRED_READY = 1;
constexpr uint16_t FLAG_STATE_PRED_CONSUMED = 2;
constexpr uint16_t FLAG_STATE_R_READY = 3;
constexpr uint16_t FLAG_STATE_DELTA_READY = 4;

// LocalOutput 的 resultUB 为单槽. OUTPUT_DONE 防止下一个 task 的 Fix 提前覆写.
constexpr uint16_t FLAG_OUTPUT_LOCAL_READY = 0;
constexpr uint16_t FLAG_OUTPUT_DONE = 1;

constexpr uint32_t C0_BYTES = 32;
constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t DV_TILE_COUNT = HEAD_DIM / DV_TILE;

constexpr uint32_t CHUNK_D_ELEMS = CHUNK_SIZE * HEAD_DIM;
constexpr uint32_t CHUNK_C_ELEMS = CHUNK_SIZE * CHUNK_SIZE;
constexpr uint32_t STATE_TILE_ELEMS = HEAD_DIM * DV_TILE;
constexpr uint32_t CHUNK_DV_TILE_ELEMS = CHUNK_SIZE * DV_TILE;

// 三个 Kernel 分别规划自己的 L1, 均从地址 0 开始. A1/B1 对应同一块物理 L1.
constexpr uint32_t PREP_M_L1_ADDR = 0;
constexpr uint32_t PREP_M_L1_ELEMS = CHUNK_C_ELEMS;
constexpr uint32_t PREP_K_PLUS_L1_ADDR = PREP_M_L1_ADDR + PREP_M_L1_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_K_PLUS_L1_ELEMS = CHUNK_D_ELEMS;
constexpr uint32_t PREP_V_L1_ADDR = PREP_K_PLUS_L1_ADDR + PREP_K_PLUS_L1_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t PREP_V_L1_ELEMS = CHUNK_D_ELEMS;

constexpr uint32_t STATE_LHS_L1_ADDR = 0;
constexpr uint32_t STATE_LHS_L1_ELEMS = CHUNK_D_ELEMS;
constexpr uint32_t STATE_STATE_L1_ADDR = STATE_LHS_L1_ADDR + STATE_LHS_L1_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t STATE_STATE_L1_ELEMS = STATE_TILE_ELEMS;
constexpr uint32_t STATE_R_L1_ADDR = STATE_STATE_L1_ADDR + STATE_STATE_L1_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t STATE_R_L1_ELEMS = CHUNK_DV_TILE_ELEMS;

constexpr uint32_t OUTPUT_A_L1_ADDR = 0;
constexpr uint32_t OUTPUT_A_L1_ELEMS = CHUNK_C_ELEMS;
constexpr uint32_t OUTPUT_R_L1_ADDR = OUTPUT_A_L1_ADDR + OUTPUT_A_L1_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t OUTPUT_R_L1_ELEMS = CHUNK_DV_TILE_ELEMS;

// L0A/L0B/L0C 属于不同物理空间, 各 Kernel 均使用单槽并从地址 0 开始.
constexpr uint32_t PREP_L0A_ELEMS = CHUNK_C_ELEMS;
constexpr uint32_t PREP_L0B_ELEMS = CHUNK_D_ELEMS;
constexpr uint32_t PREP_L0C_ELEMS = CHUNK_D_ELEMS;
constexpr uint32_t STATE_L0A_ELEMS = CHUNK_D_ELEMS;
constexpr uint32_t STATE_L0B_ELEMS = STATE_TILE_ELEMS;
constexpr uint32_t STATE_L0C_ELEMS = STATE_TILE_ELEMS;
constexpr uint32_t STATE_RESULT_UB_ADDR = 0;
constexpr uint32_t STATE_RESULT_UB_ELEMS = HEAD_DIM * AIV_DV_TILE;

constexpr uint32_t OUTPUT_L0A_ELEMS = CHUNK_C_ELEMS;
constexpr uint32_t OUTPUT_L0B_ELEMS = CHUNK_DV_TILE_ELEMS;
constexpr uint32_t OUTPUT_L0C_ELEMS = CHUNK_DV_TILE_ELEMS;
constexpr uint32_t OUTPUT_RESULT_UB_ADDR = 0;
constexpr uint32_t OUTPUT_RESULT_UB_ELEMS = CHUNK_SIZE * AIV_DV_TILE;

constexpr AscendC::FixpipeConfig KDA_FIXPIPE_CFG_UB = {AscendC::CO2Layout::ROW_MAJOR, true};
constexpr AscendC::FixpipeConfig KDA_FIXPIPE_CFG_GM = {AscendC::CO2Layout::ROW_MAJOR, false};

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
    // UB srcStride 以 32B DataBlock 为单位; GM dstStride 以字节为单位.
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
        // LoadData 的转置模式交换 M/K 的搬运方向. 方阵无法暴露这一差异,
        // K_tail.T[128,C] 要按源 K=C, 源 M=128 配置分形.
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
    params.kStep = CeilDiv<uint16_t>(l0Cols, CUBE_BLOCK);
    params.srcStride = CeilDiv<int32_t>(l1Rows, CUBE_BLOCK);
    params.dstStride = CeilDiv<uint16_t>(l0Cols, CUBE_BLOCK);
    params.ifTranspose = transpose;
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

template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeToGm(
    const AscendC::GlobalTensor<DstT>& dstGlobal, const AscendC::LocalTensor<SrcT>& srcL0CLocal, uint32_t m, uint32_t n,
    uint32_t gmRowStride)
{
    using namespace AscendC;
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> params;
    params.mSize = m;
    params.nSize = n;
    params.srcStride = CeilAlign<uint32_t>(m, CUBE_BLOCK);
    params.dstStride = gmRowStride;
    params.dualDstCtl = 0;
    params.params.ndNum = 1;
    params.params.srcNdStride = 0;
    params.params.dstNdStride = 0;
    params.quantPre = IsSameType<DstT, bfloat16_t>::value ? QuantMode_t::F322BF16 : QuantMode_t::NoQuant;
    Fixpipe<DstT, SrcT, KDA_FIXPIPE_CFG_GM>(dstGlobal, srcL0CLocal, params);
}

template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeToGm(
    const AscendC::GlobalTensor<DstT>& dstGlobal, const AscendC::LocalTensor<SrcT>& srcL0CLocal, uint32_t m, uint32_t n)
{
    FixpipeToGm<DstT, SrcT>(dstGlobal, srcL0CLocal, m, n, n);
}

} // namespace KDALite
