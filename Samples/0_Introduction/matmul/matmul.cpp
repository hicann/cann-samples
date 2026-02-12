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
 * \file matmul.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <random>
#include "acl/acl.h"

#include "tiling/platform/platform_ascendc.h"
#include "matmul_golden.h"
#include "matmul_utils.h"

namespace matmul {

// 左矩阵GM->L1, 使用ND2NZ
template <typename T>
__aicore__ inline void CopyInA1(const AscendC::GlobalTensor<T>& aGlobal, const AscendC::LocalTensor<T>& al1Local,
                                uint64_t curML1, uint64_t curKL1, uint64_t k)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    uint64_t nDim = curML1;
    uint64_t dDim = curKL1;

    nd2nzParams.nValue = nDim;
    nd2nzParams.dValue = dDim;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = k;
    nd2nzParams.dstNzC0Stride = CeilAlign(nDim, AscendC::BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
}

// 右矩阵GM->L1, 使用ND2NZ
template <typename T>
__aicore__ inline void CopyInB1(const AscendC::GlobalTensor<T>& bGlobal, const AscendC::LocalTensor<T>& bl1Local,
                                uint64_t curNL1, uint64_t curKL1, uint64_t n)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    uint64_t nDim = curKL1;
    uint64_t dDim = curNL1;

    nd2nzParams.nValue = nDim;
    nd2nzParams.dValue = dDim;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = n;
    nd2nzParams.dstNzC0Stride = CeilAlign(nDim, AscendC::BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
}

// 左矩阵L1->L0A, 使用LoadData2D
template <typename T>
__aicore__ inline void CopyInA2(const AscendC::LocalTensor<T>& al0Local, const AscendC::LocalTensor<T>& al1Local,
                                uint64_t curML1, uint64_t curKL1, uint64_t mL0, uint64_t kL0)
{
    AscendC::LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = 0;

    loadDataParams.mStep = CeilDiv(mL0, AscendC::BLOCK_CUBE);
    if constexpr (AscendC::IsSameType<T, half>::value || AscendC::IsSameType<T, bfloat16_t>::value) {
        loadDataParams.kStep = CeilDiv(kL0, AscendC::BLOCK_CUBE);
    } else {
        loadDataParams.kStep = CeilDiv(kL0, GetC0Size<T>());
    }
    loadDataParams.srcStride = CeilDiv(curML1, AscendC::BLOCK_CUBE);
    loadDataParams.dstStride = loadDataParams.mStep;
    loadDataParams.ifTranspose = false;
    AscendC::LoadData<T>(al0Local, al1Local, loadDataParams);
}

// 右矩阵L1->L0B, 使用LoadData2D
template <typename T>
__aicore__ inline void CopyInB2(const AscendC::LocalTensor<T>& bl0Local, const AscendC::LocalTensor<T>& bl1Local,
                                uint64_t curNL1, uint64_t curKL1, uint64_t nL0, uint64_t kL0)
{
    AscendC::LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = 0;

    loadDataParams.mStep = CeilDiv(kL0, AscendC::BLOCK_CUBE);
    if constexpr (AscendC::IsSameType<T, half>::value || AscendC::IsSameType<T, bfloat16_t>::value) {
        loadDataParams.kStep = CeilDiv(nL0, AscendC::BLOCK_CUBE);
        loadDataParams.dstStride = loadDataParams.kStep;
    } else {
        loadDataParams.kStep = CeilDiv(nL0, AscendC::BLOCK_CUBE) * TWO_ALIGN;
        loadDataParams.dstStride = loadDataParams.kStep >> 1;
    }
    loadDataParams.srcStride = CeilDiv(curKL1, AscendC::BLOCK_CUBE);
    loadDataParams.ifTranspose = true;
    AscendC::LoadData<T>(bl0Local, bl1Local, loadDataParams);
}

// 矩阵计算
template <typename T>
__aicore__ inline void Mmad(const AscendC::LocalTensor<float>& c1Local, const AscendC::LocalTensor<T>& al0Local,
                            const AscendC::LocalTensor<T>& bl0Local, uint64_t m, uint64_t n, uint64_t k,
                            bool isFirstLoop)
{
    AscendC::MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.disableGemv = true;
    mmadParams.cmatrixSource = false;
    // 识别首次计算采用数据覆盖，非首次计算采用数据累加
    mmadParams.cmatrixInitVal = isFirstLoop;
    mmadParams.unitFlag = 0;
    AscendC::Mmad(c1Local, al0Local, bl0Local, mmadParams);
}

// L0C->GM, 使用NZ2ND
template <typename T>
__aicore__ inline void CopyOut(const AscendC::GlobalTensor<T>& cGlobal, const AscendC::LocalTensor<float>& c1Local,
                               uint64_t baseM, uint64_t baseN, uint64_t n)
{
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.nSize = baseN;
    intriParams.mSize = baseM;
    intriParams.dstStride = n;
    intriParams.srcStride = CeilAlign(baseM, AscendC::BLOCK_CUBE);
    // 根据输出Dtype设置不同的量化模式
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        intriParams.quantPre = QuantMode_t::F322BF16;
    } else if (AscendC::IsSameType<T, half>::value) {
        intriParams.quantPre = QuantMode_t::F322F16;
    } else if (AscendC::IsSameType<T, float>::value) {
        intriParams.quantPre = QuantMode_t::NoQuant;
    }
    intriParams.reluPre = 0;
    intriParams.nz2ndEn = true;
    intriParams.unitFlag = 0;
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    AscendC::DataCopy(cGlobal, c1Local, intriParams);
}

// 矩阵乘Kernel的简单样例
template <typename T>
__global__ __aicore__ void MatmulKernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, uint32_t m, uint32_t k, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    // 构造GM/L1/L0A/L0B/L0C Tensor
    AscendC::GlobalTensor<T> aGlobal;
    AscendC::GlobalTensor<T> bGlobal;
    AscendC::GlobalTensor<T> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(aGm));
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bGm));
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cGm));
    AscendC::LocalTensor<T> al0Local{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<T> bl0Local{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> l0cLocal{AscendC::TPosition::CO1, 0, L0C_SIZE};
    AscendC::LocalTensor<T> l1Local{AscendC::TPosition::A1, 0, L1_SIZE};

    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT / sizeof(T);

    // 设置基本块大小
    uint64_t baseM = 256;
    uint64_t baseN = 256;
    uint64_t baseK = 128 / sizeof(T);
    uint64_t kL1 = 512 / sizeof(T);
    uint64_t mTileNum = CeilDiv(m, baseM);
    uint64_t nTileNum = CeilDiv(n, baseN);
    uint64_t tileNum = mTileNum * nTileNum;
    uint64_t tailBaseM = m - (mTileNum - 1) * baseM;
    uint64_t tailBaseN = n - (nTileNum - 1) * baseN;
    uint64_t kL1TileNum = CeilDiv(k, kL1);
    uint64_t tailKL1 = k - (kL1TileNum - 1) * kL1;

    // DoubleBuffer同步标记
    uint64_t l1PingPong = 0;
    uint64_t l0PingPong = 0;

    uint64_t curBlockIdx = AscendC::GetBlockIdx();
    uint64_t blockNum = AscendC::GetBlockNum();

    // 用于匹配Kernel首次执行的Waitflag
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);

    // 单核多轮基本块计算循环
    for (uint64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
        // 获取当前处理的基本块index
        uint64_t mTileIdx = tileIdx / nTileNum;
        uint64_t nTileIdx = tileIdx % nTileNum;
        uint64_t mL1 = mTileIdx == (mTileNum - 1) ? tailBaseM : baseM;
        uint64_t nL1 = nTileIdx == (nTileNum - 1) ? tailBaseN : baseN;
        uint64_t mL0 = mL1;
        uint64_t nL0 = nL1;
        uint64_t mOffset = mTileIdx * baseM;
        uint64_t nOffset = nTileIdx * baseN;
        uint64_t offsetA = mOffset * k;
        uint64_t offsetB = nOffset;
        uint64_t offsetC = mOffset * n + nOffset;

        // 单核多轮计算之间，用于同步当前首次MMAD与上一个循环FIXP搬运，避免L0C数据被覆盖
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);

        for (uint64_t iter0 = 0; iter0 < kL1TileNum; ++iter0) {
            uint64_t curKL1 = (iter0 + 1 == kL1TileNum) ? tailKL1 : kL1;
            // 开启L1 DoubleBuffer，区分pingpong流水的同步id
            uint64_t l1BufId = l1PingPong & 0x1;

            // 搬运左右矩阵到L1 Buffer
            // 设置同步等待上一轮的MTE1搬运
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            uint64_t offsetAL1 = baseM * kL1 * l1BufId;
            CopyInA1<T>(aGlobal[offsetA], l1Local[offsetAL1], mL1, curKL1, k);
            offsetA += curKL1;
            uint64_t offsetBL1 = baseM * kL1 * 2 + baseN * kL1 * l1BufId;
            CopyInB1<T>(bGlobal[offsetB], l1Local[offsetBL1], nL1, curKL1, n);
            offsetB += curKL1 * n;

            // 设置同步MTE1等待MTE2完成搬运
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            uint64_t kL0TileNum = CeilDiv(curKL1, baseK);
            uint64_t tailKL0 = curKL1 - (kL0TileNum - 1) * baseK;
            for (uint64_t iter1 = 0; iter1 < kL0TileNum; ++iter1) {
                uint64_t curKL0 = (iter1 + 1 == kL0TileNum) ? tailKL0 : baseK;
                // 开启L0 DoubleBuffer，区分pingpong流水的同步id
                uint64_t l0BufId = l0PingPong & 0x1;

                // 搬运左右矩阵到L0 Buffer
                // 设置同步等待上一轮的MMAD计算
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
                uint64_t l0Offset = HALF_L0_SIZE * l0BufId;
                CopyInA2<T>(al0Local[l0Offset], l1Local[offsetAL1], mL1, curKL1, mL0, curKL0);
                offsetAL1 += CeilAlign(mL1, AscendC::BLOCK_CUBE) * baseK;
                CopyInB2<T>(bl0Local[l0Offset], l1Local[offsetBL1], nL1, curKL1, nL0, curKL0);
                offsetBL1 += baseK * GetC0Size<T>();

                // 设置同步MMAD等待MTE1完成搬运
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);

                // 矩阵乘计算
                bool isFirstLoop = iter0 == 0 && iter1 == 0;
                Mmad(l0cLocal, al0Local[l0Offset], bl0Local[l0Offset], mL0, nL0, curKL0, isFirstLoop);

                // 设置同步让下一轮的MTE1等待本轮MMAD完成
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
                l0PingPong++;
            }
            // 设置同步让下一轮的MTE2等待本轮MTE1完成
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            l1PingPong++;
        }
        // 设置同步FIXP等待MMAD完成计算
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);

        // 搬运输出到GM
        CopyOut(cGlobal[offsetC], l0cLocal, mL0, nL0, n);

        // 设置同步让下一轮的MMAD等待本轮FIXP完成
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
    }

    // 用于匹配Kernel最后设置的SetFlag
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
}

} // namespace matmul

#define CHECK_COND(cond, message, return_expr)                                                                         \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::cerr << "ERROR: " << message << std::endl;                                                            \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

// 打印使用说明
void printUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName << " m k n" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "Example: " << programName << " 100 50 200" << std::endl;
}

// 解析命令行参数
void parseArguments(int argc, char* argv[], int& m, int& k, int& n)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        printUsage(argv[0]);
        exit(1);
    }
    if (argc < 4) {
        throw std::invalid_argument("ERROR: Lacks Arguments");
    }
    try {
        m = std::stoi(argv[1]);
        k = std::stoi(argv[2]);
        n = std::stoi(argv[3]);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument("ERROR: m k n must be Integer");
    }

    if (m <= 0 || k <= 0 || n <= 0) {
        throw std::invalid_argument("ERROR: m k n must be positive");
    }
}

int main(int argc, char* argv[])
{
    // 获取输入Shape
    int m, k, n;
    try {
        parseArguments(argc, argv, m, k, n);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // 资源初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclInit(nullptr);
    CHECK_COND(ret == ACL_SUCCESS, "aclInit failed.", return 1);
    ret = aclrtSetDevice(deviceId);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtSetDevice failed.", return 1);
    ret = aclrtCreateStream(&stream);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtCreateStream failed.", return 1);

    // 构造Host侧输入输出
    std::vector<float> hostInput(m * k, 0);
    std::vector<float> hostWeight(k * n, 0);
    std::vector<float> hostOutput(m * n, 0);
    std::vector<float> goldenOutput(m * n, 0);
    matmul::FillRandomData<float>(hostInput, -2.0f, 2.0f);
    matmul::FillRandomData<float>(hostWeight, -2.0f, 2.0f);

    // 申请Device侧地址
    GM_ADDR deviceInput = nullptr;
    GM_ADDR deviceWeight = nullptr;
    GM_ADDR deviceOutput = nullptr;
    auto sizeInput = hostInput.size() * sizeof(float);
    auto sizeWeight = hostWeight.size() * sizeof(float);
    auto sizeOutput = hostOutput.size() * sizeof(float);
    ret = aclrtMalloc((void**)&deviceInput, sizeInput, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceInput failed.", return 1);
    ret = aclrtMalloc((void**)&deviceWeight, sizeWeight, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceWeight failed.", return 1);
    ret = aclrtMalloc((void**)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceOutput failed.", return 1);

    // 输入数据Host To Device
    ret = aclrtMemcpy(deviceInput, sizeInput, hostInput.data(), sizeInput, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceInput failed.", return 1);
    ret = aclrtMemcpy(deviceWeight, sizeWeight, hostWeight.data(), sizeWeight, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceWeight failed.", return 1);

    // 调用算子Kernel
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    CHECK_COND(ascendcPlatform != nullptr, "get ascendcPlatform failed.", return 1);
    uint32_t numBlocks = ascendcPlatform->GetCoreNumAic();
    matmul::MatmulKernel<float><<<numBlocks, nullptr, stream>>>(deviceInput, deviceWeight, deviceOutput, m, k, n);

    // 同步等待算子执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed.", return 1);

    // 输出数据Device To Host
    ret = aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceOutput failed.", return 1);

    // 计算golden，对比精度
    matmul::ComputeGolden<float>(m, k, n, hostInput, hostWeight, goldenOutput);
    std::vector<uint64_t> errorIndices = matmul::Compare<float>(hostOutput, goldenOutput);
    if (errorIndices.size() == 0) {
        std::cout << "matmul run successfully!" << std::endl;
    } else {
        for (uint64_t i : errorIndices) {
            uint64_t errIdx = errorIndices[i];
            std::cout << "error index: " << errIdx << ", output: " << hostOutput[errIdx]
                      << ", golden: " << goldenOutput[errIdx] << std::endl;
        }
        std::cout << "matmul run failed!" << std::endl;
    }

    // 资源释放
    std::unique_ptr<void, aclError (*)(void*)> DeviceInputAddr(deviceInput, aclrtFree);
    std::unique_ptr<void, aclError (*)(void*)> DeviceWeightAddr(deviceWeight, aclrtFree);
    std::unique_ptr<void, aclError (*)(void*)> DeviceOutputAddr(deviceOutput, aclrtFree);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}