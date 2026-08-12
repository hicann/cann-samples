/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_RELU_CV_STORY_KERNEL_COMMON_H
#define MATMUL_RELU_CV_STORY_KERNEL_COMMON_H

#include "kernel_operator.h"

// Aligned with asc-devkit matmul_gelu_high_performance (Ascend 950PR / dav-3510)
constexpr uint32_t MATMUL_RELU_M = 8192;
constexpr uint32_t MATMUL_RELU_K = 8192;
constexpr uint32_t MATMUL_RELU_N = 8192;

constexpr uint32_t MATMUL_RELU_BASE_M = 256;
constexpr uint32_t MATMUL_RELU_BASE_N = 256;
constexpr uint32_t MATMUL_RELU_BASE_K = 64;

// singleCore 1024×1024 (vs gelu 2048×1024): 8×8=64 MN blocks
// (strided: most cores 2 blocks, some 3) while keeping n-outer/m-inner B locality inside a block.
constexpr uint32_t MATMUL_RELU_SINGLE_CORE_M = 1024;
constexpr uint32_t MATMUL_RELU_SINGLE_CORE_N = 1024;
constexpr uint32_t MATMUL_RELU_SINGLE_CORE_K = 8192;

// L1 multi-baseK pack size (gelu 3510: stepKa=stepKb=4)
constexpr uint32_t MATMUL_RELU_STEP_KA = 4;
constexpr uint32_t MATMUL_RELU_STEP_KB = 4;

static_assert(MATMUL_RELU_M % MATMUL_RELU_SINGLE_CORE_M == 0, "M must be multiple of singleCoreM");
static_assert(MATMUL_RELU_N % MATMUL_RELU_SINGLE_CORE_N == 0, "N must be multiple of singleCoreN");
static_assert(MATMUL_RELU_K % MATMUL_RELU_SINGLE_CORE_K == 0, "K must be multiple of singleCoreK");
static_assert(MATMUL_RELU_SINGLE_CORE_M % MATMUL_RELU_BASE_M == 0, "singleCoreM must be multiple of baseM");
static_assert(MATMUL_RELU_SINGLE_CORE_N % MATMUL_RELU_BASE_N == 0, "singleCoreN must be multiple of baseN");
static_assert(MATMUL_RELU_SINGLE_CORE_K % MATMUL_RELU_BASE_K == 0, "singleCoreK must be multiple of baseK");
static_assert(MATMUL_RELU_SINGLE_CORE_K % (MATMUL_RELU_BASE_K * MATMUL_RELU_STEP_KA) == 0, "K pack A");
static_assert(MATMUL_RELU_SINGLE_CORE_K % (MATMUL_RELU_BASE_K * MATMUL_RELU_STEP_KB) == 0, "K pack B");
static_assert(MATMUL_RELU_BASE_M % 2 == 0, "baseM must be even for dualDstCtl=0b01");

constexpr uint32_t MATMUL_RELU_K_ROUND = MATMUL_RELU_SINGLE_CORE_K / MATMUL_RELU_BASE_K; // 128
constexpr uint32_t MATMUL_RELU_M_ITER = MATMUL_RELU_M / MATMUL_RELU_SINGLE_CORE_M;       // 8
constexpr uint32_t MATMUL_RELU_N_ITER = MATMUL_RELU_N / MATMUL_RELU_SINGLE_CORE_N;       // 8
constexpr uint32_t MATMUL_RELU_MN_BLOCKS = MATMUL_RELU_M_ITER * MATMUL_RELU_N_ITER;      // 64
constexpr uint32_t MATMUL_RELU_M_TILES_PER_CORE = MATMUL_RELU_SINGLE_CORE_M / MATMUL_RELU_BASE_M; // 4
constexpr uint32_t MATMUL_RELU_N_TILES_PER_CORE = MATMUL_RELU_SINGLE_CORE_N / MATMUL_RELU_BASE_N; // 4

// Launch size = platform Aicore Count (A310-50 / Ascend 950PR SKU = 28).
// 64 singleCore blocks are owned with stride NUM_BLOCKS (cores get 2 or 3 blocks).
constexpr uint32_t MATMUL_RELU_NUM_BLOCKS = 28;
static_assert(MATMUL_RELU_NUM_BLOCKS > 0, "numBlocks must be positive");
static_assert(MATMUL_RELU_NUM_BLOCKS <= MATMUL_RELU_MN_BLOCKS, "numBlocks cannot exceed MN block count");
static_assert(MATMUL_RELU_M_ITER > 0, "M_ITER must be positive");

constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024;
constexpr uint32_t L1_PINGPONG_BYTES = 256 * 1024;

constexpr uint16_t AIC_SYNC_AIV_FLAG = 0x8;

constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};
constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_GM = {AscendC::CO2Layout::ROW_MAJOR, false};

__aicore__ inline uint32_t DivCeilU32(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint32_t GetLogicCoreId()
{
    if ASCEND_IS_AIC {
        return AscendC::GetBlockIdx();
    } else {
        return AscendC::GetBlockIdx() / 2;
    }
}

__aicore__ inline void DecodeMnBlock(uint32_t blockId, uint32_t& mIterIdx, uint32_t& nIterIdx)
{
    mIterIdx = blockId % MATMUL_RELU_M_ITER;
    nIterIdx = blockId / MATMUL_RELU_M_ITER;
}

__aicore__ inline uint32_t GlobalMTileIdx(uint32_t mIterIdx, uint32_t mBlockIdx)
{
    return mIterIdx * MATMUL_RELU_M_TILES_PER_CORE + mBlockIdx;
}

__aicore__ inline uint32_t GlobalNTileIdx(uint32_t nIterIdx, uint32_t nBlockIdx)
{
    return nIterIdx * MATMUL_RELU_N_TILES_PER_CORE + nBlockIdx;
}

/**
 * High-performance Cube pipeline (gelu-aligned):
 * L1/L0 ping-pong + stepKa/stepKb packed CopyIn + reverse HardEvent + unitFlag.
 * A is ND [M,K]; B is host-transposed ND [N,K] (same as matmul_gelu_high_performance).
 */
template <uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepKa, uint32_t stepKb>
class MatmulCubePipeline {
public:
    __aicore__ inline MatmulCubePipeline() {}

    __aicore__ inline void Init(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c)
    {
        aGM.SetGlobalBuffer((__gm__ half*)a);
        bGM.SetGlobalBuffer((__gm__ half*)b);
        cGM.SetGlobalBuffer((__gm__ float*)c);
        mte1DBFlag = 0;
    }

    __aicore__ inline void InitAicSyncFlags()
    {
        l1APingMutex = AscendC::AllocMutexID();
        l1APongMutex = AscendC::AllocMutexID();
        l1BPingMutex = AscendC::AllocMutexID();
        l1BPongMutex = AscendC::AllocMutexID();
        l0PingMutex = AscendC::AllocMutexID();
        l0PongMutex = AscendC::AllocMutexID();
    }

    __aicore__ inline void WaitAicSyncFlags()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::ReleaseMutexID(l1APingMutex);
        AscendC::ReleaseMutexID(l1APongMutex);
        AscendC::ReleaseMutexID(l1BPingMutex);
        AscendC::ReleaseMutexID(l1BPongMutex);
        AscendC::ReleaseMutexID(l0PingMutex);
        AscendC::ReleaseMutexID(l0PongMutex);
    }

    __aicore__ inline void RunMatmul(
        AscendC::LocalTensor<float>& cLocal, uint32_t mTileIdx, uint32_t nTileIdx)
    {
        CubeL1L0Buf buf;
        AllocL1L0Buf(buf);

        uint32_t a1NextKChunkIdx = 0;
        uint32_t b1NextKChunkIdx = 0;
        uint8_t a1CopyInIdx = 0;
        uint8_t b1CopyInIdx = 0;
        PrefetchFirstL1Packs(
            buf, mTileIdx, nTileIdx, a1NextKChunkIdx, b1NextKChunkIdx, a1CopyInIdx, b1CopyInIdx);

        constexpr uint32_t kLoopCount = MATMUL_RELU_K_ROUND;
        for (uint32_t kBlockIdx = 0; kBlockIdx < kLoopCount; kBlockIdx++) {
            ProcessOneKBlock(
                buf, cLocal, mTileIdx, nTileIdx, kBlockIdx, kLoopCount,
                a1NextKChunkIdx, b1NextKChunkIdx, a1CopyInIdx, b1CopyInIdx);
        }
    }

    __aicore__ inline void FixpipeToGm(
        AscendC::LocalTensor<float>& cLocal, uint32_t mTileIdx, uint32_t nTileIdx)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.mSize = baseM;
        fixpipeParams.nSize = baseN;
        fixpipeParams.srcStride = baseM;
        fixpipeParams.dstStride = MATMUL_RELU_N;
        fixpipeParams.unitFlag = 3;
        uint32_t gmOffset = mTileIdx * baseM * MATMUL_RELU_N + nTileIdx * baseN;
        AscendC::Fixpipe<float, float, CFG_ROW_MAJOR_GM>(cGM[gmOffset], cLocal, fixpipeParams);
    }

    __aicore__ inline void FixpipeToUbDualM(
        AscendC::LocalTensor<float>& xUB, AscendC::LocalTensor<float>& cLocal)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.mSize = DivCeilU32(baseM, 2) * 2;
        fixpipeParams.nSize = baseN;
        fixpipeParams.srcStride = baseM;
        fixpipeParams.dstStride = baseN;
        fixpipeParams.dualDstCtl = 0b01;
        fixpipeParams.unitFlag = 3;
        AscendC::Fixpipe<float, float, CFG_ROW_MAJOR_UB>(xUB, cLocal, fixpipeParams);
    }

    AscendC::GlobalTensor<float> cGM;

private:
    struct CubeL1L0Buf {
        AscendC::LocalTensor<half> a1Ping;
        AscendC::LocalTensor<half> a1Pong;
        AscendC::LocalTensor<half> a2Ping;
        AscendC::LocalTensor<half> a2Pong;
        AscendC::LocalTensor<half> b1Ping;
        AscendC::LocalTensor<half> b1Pong;
        AscendC::LocalTensor<half> b2Ping;
        AscendC::LocalTensor<half> b2Pong;
    };

    __aicore__ inline void AllocL1L0Buf(CubeL1L0Buf& buf)
    {
        constexpr uint32_t a1PingpongSize = baseM * baseK * stepKa;
        constexpr uint32_t b1PingpongSize = baseK * baseN * stepKb;
        constexpr uint32_t a2PingpongSize = baseM * baseK;
        constexpr uint32_t b2PingpongSize = baseK * baseN;
        buf.a1Ping = AscendC::LocalTensor<half>(AscendC::TPosition::A1, 0, a1PingpongSize);
        buf.a1Pong = AscendC::LocalTensor<half>(
            AscendC::TPosition::A1, a1PingpongSize * sizeof(half), a1PingpongSize);
        buf.a2Ping = AscendC::LocalTensor<half>(AscendC::TPosition::A2, 0, a2PingpongSize);
        buf.a2Pong = AscendC::LocalTensor<half>(AscendC::TPosition::A2, L0_PINGPONG_BYTES, a2PingpongSize);
        buf.b1Ping = AscendC::LocalTensor<half>(AscendC::TPosition::B1, L1_PINGPONG_BYTES, b1PingpongSize);
        buf.b1Pong = AscendC::LocalTensor<half>(
            AscendC::TPosition::B1, L1_PINGPONG_BYTES + b1PingpongSize * sizeof(half), b1PingpongSize);
        buf.b2Ping = AscendC::LocalTensor<half>(AscendC::TPosition::B2, 0, b2PingpongSize);
        buf.b2Pong = AscendC::LocalTensor<half>(AscendC::TPosition::B2, L0_PINGPONG_BYTES, b2PingpongSize);
    }

    __aicore__ inline void PrefetchFirstL1Packs(
        CubeL1L0Buf& buf, uint32_t mTileIdx, uint32_t nTileIdx,
        uint32_t& a1NextKChunkIdx, uint32_t& b1NextKChunkIdx, uint8_t& a1CopyInIdx, uint8_t& b1CopyInIdx)
    {
        AscendC::Mutex::Lock<PIPE_MTE2>(l1APingMutex);
        DataCopyInA(buf.a1Ping, a1NextKChunkIdx, mTileIdx);
        AscendC::Mutex::Unlock<PIPE_MTE2>(l1APingMutex);
        a1NextKChunkIdx += stepKa;
        a1CopyInIdx ^= 1;

        AscendC::Mutex::Lock<PIPE_MTE2>(l1BPingMutex);
        DataCopyInB(buf.b1Ping, b1NextKChunkIdx, nTileIdx);
        AscendC::Mutex::Unlock<PIPE_MTE2>(l1BPingMutex);
        b1NextKChunkIdx += stepKb;
        b1CopyInIdx ^= 1;
    }

    __aicore__ inline void WaitL1Ready(uint32_t kOffsetInChunkA, uint32_t kOffsetInChunkB,
        uint32_t a1ReadIdx, uint32_t b1ReadIdx)
    {
        if (kOffsetInChunkA == 0) {
            AscendC::Mutex::Lock<PIPE_MTE1>((a1ReadIdx == 0) ? l1APingMutex : l1APongMutex);
        }
        if (kOffsetInChunkB == 0) {
            AscendC::Mutex::Lock<PIPE_MTE1>((b1ReadIdx == 0) ? l1BPingMutex : l1BPongMutex);
        }
    }

    __aicore__ inline void ReleaseL1AfterLoad(uint32_t kOffsetInChunkA, uint32_t kOffsetInChunkB,
        uint32_t a1ReadIdx, uint32_t b1ReadIdx)
    {
        if ((kOffsetInChunkA + 1) == stepKa) {
            AscendC::Mutex::Unlock<PIPE_MTE1>((a1ReadIdx == 0) ? l1APingMutex : l1APongMutex);
        }
        if ((kOffsetInChunkB + 1) == stepKb) {
            AscendC::Mutex::Unlock<PIPE_MTE1>((b1ReadIdx == 0) ? l1BPingMutex : l1BPongMutex);
        }
    }

    __aicore__ inline void PrefetchNextL1Pack(
        CubeL1L0Buf& buf, uint32_t mTileIdx, uint32_t nTileIdx, uint32_t kBlockIdx,
        uint32_t kOffsetInChunkA, uint32_t kOffsetInChunkB, uint32_t kLoopCount,
        uint32_t& a1NextKChunkIdx, uint32_t& b1NextKChunkIdx, uint8_t& a1CopyInIdx, uint8_t& b1CopyInIdx)
    {
        if (((kBlockIdx == 0) || ((kOffsetInChunkB + 1) == stepKb)) && b1NextKChunkIdx < kLoopCount) {
            AscendC::LocalTensor<half> b1WriteBuf = (b1CopyInIdx == 0) ? buf.b1Ping : buf.b1Pong;
            AscendC::Mutex::Lock<PIPE_MTE2>((b1CopyInIdx == 0) ? l1BPingMutex : l1BPongMutex);
            DataCopyInB(b1WriteBuf, b1NextKChunkIdx, nTileIdx);
            AscendC::Mutex::Unlock<PIPE_MTE2>((b1CopyInIdx == 0) ? l1BPingMutex : l1BPongMutex);
            b1NextKChunkIdx += stepKb;
            b1CopyInIdx ^= 1;
        }
        if (((kBlockIdx == 0) || ((kOffsetInChunkA + 1) == stepKa)) && a1NextKChunkIdx < kLoopCount) {
            AscendC::LocalTensor<half> a1WriteBuf = (a1CopyInIdx == 0) ? buf.a1Ping : buf.a1Pong;
            AscendC::Mutex::Lock<PIPE_MTE2>((a1CopyInIdx == 0) ? l1APingMutex : l1APongMutex);
            DataCopyInA(a1WriteBuf, a1NextKChunkIdx, mTileIdx);
            AscendC::Mutex::Unlock<PIPE_MTE2>((a1CopyInIdx == 0) ? l1APingMutex : l1APongMutex);
            a1NextKChunkIdx += stepKa;
            a1CopyInIdx ^= 1;
        }
    }

    __aicore__ inline void ProcessOneKBlock(
        CubeL1L0Buf& buf, AscendC::LocalTensor<float>& cLocal, uint32_t mTileIdx, uint32_t nTileIdx,
        uint32_t kBlockIdx, uint32_t kLoopCount,
        uint32_t& a1NextKChunkIdx, uint32_t& b1NextKChunkIdx, uint8_t& a1CopyInIdx, uint8_t& b1CopyInIdx)
    {
        // stepKa/stepKb are compile-time non-zero (static_assert on K packing).
        uint32_t a1ReadIdx = (kBlockIdx / stepKa) % 2;
        uint32_t b1ReadIdx = (kBlockIdx / stepKb) % 2;
        uint32_t kOffsetInChunkA = kBlockIdx % stepKa;
        uint32_t kOffsetInChunkB = kBlockIdx % stepKb;

        AscendC::LocalTensor<half> a1ReadBuf = (a1ReadIdx == 0) ? buf.a1Ping : buf.a1Pong;
        AscendC::LocalTensor<half> b1ReadBuf = (b1ReadIdx == 0) ? buf.b1Ping : buf.b1Pong;
        AscendC::LocalTensor<half> a2Local = (mte1DBFlag == 0) ? buf.a2Ping : buf.a2Pong;
        AscendC::LocalTensor<half> b2Local = (mte1DBFlag == 0) ? buf.b2Ping : buf.b2Pong;

        AscendC::Mutex::Lock<PIPE_MTE1>((mte1DBFlag == 0) ? l0PingMutex : l0PongMutex);
        WaitL1Ready(kOffsetInChunkA, kOffsetInChunkB, a1ReadIdx, b1ReadIdx);
        DataLoadA(a1ReadBuf, a2Local, kOffsetInChunkA);
        DataLoadB(b1ReadBuf, b2Local, kOffsetInChunkB);
        ReleaseL1AfterLoad(kOffsetInChunkA, kOffsetInChunkB, a1ReadIdx, b1ReadIdx);
        Compute(cLocal, a2Local, b2Local, kBlockIdx, kLoopCount);
        PrefetchNextL1Pack(
            buf, mTileIdx, nTileIdx, kBlockIdx, kOffsetInChunkA, kOffsetInChunkB, kLoopCount,
            a1NextKChunkIdx, b1NextKChunkIdx, a1CopyInIdx, b1CopyInIdx);
    }

    __aicore__ inline void DataCopyInA(
        AscendC::LocalTensor<half> a1Local, uint32_t kChunkIdx, uint32_t mTileIdx)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = baseM;
        nd2nzParams.dValue = baseK * stepKa;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = MATMUL_RELU_K;
        nd2nzParams.dstNzC0Stride = baseM;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        uint32_t aOffset = mTileIdx * baseM * MATMUL_RELU_K + kChunkIdx * baseK;
        AscendC::DataCopy(a1Local, aGM[aOffset], nd2nzParams);
    }

    __aicore__ inline void DataCopyInB(
        AscendC::LocalTensor<half> b1Local, uint32_t kChunkIdx, uint32_t nTileIdx)
    {
        // B host-transposed [N,K]
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = baseN;
        nd2nzParams.dValue = baseK * stepKb;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = MATMUL_RELU_K;
        nd2nzParams.dstNzC0Stride = baseN;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        uint32_t bOffset = nTileIdx * baseN * MATMUL_RELU_K + kChunkIdx * baseK;
        AscendC::DataCopy(b1Local, bGM[bOffset], nd2nzParams);
    }

    __aicore__ inline void DataLoadA(
        AscendC::LocalTensor<half> a1Local, AscendC::LocalTensor<half> a2Local, uint32_t kOffsetInChunkA)
    {
        uint32_t srcAddr = kOffsetInChunkA * baseK * baseM;
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = DivCeilU32(baseM, CUBE_BLOCK);
        loadDataParams.kStep = DivCeilU32(baseK, CUBE_BLOCK);
        loadDataParams.srcStride = DivCeilU32(baseM, CUBE_BLOCK);
        loadDataParams.dstStride = DivCeilU32(baseM, CUBE_BLOCK);
        loadDataParams.sid = 0;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local[srcAddr], loadDataParams);
    }

    __aicore__ inline void DataLoadB(
        AscendC::LocalTensor<half> b1Local, AscendC::LocalTensor<half> b2Local, uint32_t kOffsetInChunkB)
    {
        // B transposed [N,K]: no LoadData transpose
        uint32_t srcAddr = kOffsetInChunkB * baseK * baseN;
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = DivCeilU32(baseN, CUBE_BLOCK);
        loadDataParams.kStep = DivCeilU32(baseK * sizeof(half), 32);
        loadDataParams.srcStride = DivCeilU32(baseN, CUBE_BLOCK);
        loadDataParams.dstStride = DivCeilU32(baseN, CUBE_BLOCK);
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(b2Local, b1Local[srcAddr], loadDataParams);
    }

    __aicore__ inline void Compute(
        AscendC::LocalTensor<float> cLocal, AscendC::LocalTensor<half> a2Local,
        AscendC::LocalTensor<half> b2Local, uint32_t kBlockIdx, uint32_t kLoopCount)
    {
        uint8_t curL0Mutex = (mte1DBFlag == 0) ? l0PingMutex : l0PongMutex;
        AscendC::Mutex::Unlock<PIPE_MTE1>(curL0Mutex);
        AscendC::Mutex::Lock<PIPE_M>(curL0Mutex);
        AscendC::MmadParams mmadParams;
        mmadParams.m = baseM;
        mmadParams.n = baseN;
        mmadParams.k = baseK;
        mmadParams.cmatrixInitVal = (kBlockIdx == 0);
        mmadParams.unitFlag = (kBlockIdx != kLoopCount - 1) ? 2 : 3;
        AscendC::Mmad(cLocal, a2Local, b2Local, mmadParams);
        AscendC::Mutex::Unlock<PIPE_M>(curL0Mutex);
        mte1DBFlag ^= 1;
    }

    AscendC::GlobalTensor<half> aGM;
    AscendC::GlobalTensor<half> bGM;
    uint8_t mte1DBFlag = 0;
    uint8_t l1APingMutex = 0;
    uint8_t l1APongMutex = 0;
    uint8_t l1BPingMutex = 0;
    uint8_t l1BPongMutex = 0;
    uint8_t l0PingMutex = 0;
    uint8_t l0PongMutex = 0;
};

using CubePipe = MatmulCubePipeline<
    MATMUL_RELU_BASE_M, MATMUL_RELU_BASE_K, MATMUL_RELU_BASE_N,
    MATMUL_RELU_STEP_KA, MATMUL_RELU_STEP_KB>;

#endif // MATMUL_RELU_CV_STORY_KERNEL_COMMON_H
