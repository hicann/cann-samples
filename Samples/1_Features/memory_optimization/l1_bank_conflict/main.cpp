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
 * \file main.cpp
 * \brief Main implementation file for Ascend matrix multiplication kernel
 *        This file contains the kernel implementation and host-side setup code
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <random>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cstdint>

#include "acl/acl.h"
#include "kernel_basic_intf.h"
#include "tiling/platform/platform_ascendc.h"
#include "include/tensor.h"

namespace AscendC::Te {
// Layout definitions for matrices A and B (NZ format by default, can be transposed to ZN)
static constexpr bool transA = false;
static constexpr bool transB = false;
template <typename T>
using MakeLayoutAL1 =
    AscendC::Std::conditional_t<transA, AscendC::Te::ZnLayoutFormat<T>, AscendC::Te::NzLayoutFormat<T>>;
template <typename T>
using MakeLayoutBL1 =
    AscendC::Std::conditional_t<transB, AscendC::Te::ZnLayoutFormat<T>, AscendC::Te::NzLayoutFormat<T>>;
} // namespace AscendC::Te

namespace tool {
// Memory and buffer configuration constants
constexpr static uint64_t DOUBLE_BUFFER_COUNT = 2;    // Double buffering for ping-pong operation
constexpr static int64_t L0A_SIZE = 64 * 1024;        // L0A buffer size (64KB)
constexpr static int64_t TOTAL_L0C_SIZE = 256 * 1024; // Total L0C buffer size (256KB)
constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;        // Half L0A for ping-pong

// Synchronization flag values
constexpr static uint16_t ZERO_FLAG = 0;  // First flag value
constexpr static uint16_t FIRST_FLAG = 1; // Second flag value

// Helper function declarations for host-side operations
template <typename T>
void FillRandomData(std::vector<T>& data, T min, T max); // Fill vector with random data
float Bf16ToFloat(uint16_t h);
uint16_t FloatToBf16(float f);
template <typename T>
void ComputeGolden(    // Compute reference result on CPU
    int m, int k, int n, std::vector<T>& hostInput, std::vector<T>& hostWeight, std::vector<T>& goldenOutput);
template <typename T>
std::vector<uint64_t> Compare(std::vector<T>& hostOutput, std::vector<T>& goldenOutput); // Compare results
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b);                              // Ceiling division
} // namespace tool

namespace matmul {
/**
 * @brief Matrix multiplication kernel for Ascend AI processor
 *
 * This kernel implements C = A * B using optimized memory hierarchy:
 * - Double buffering between GM -> L1 and L1 -> L0
 * - Tiled computation to fit in on-chip memory
 * - Multi-core parallelization
 *
 * @tparam T Data type (float in this implementation)
 * @param aGm Global memory pointer to matrix A (size m*k)
 * @param bGm Global memory pointer to matrix B (size k*n)
 * @param cGm Global memory pointer to output matrix C (size m*n)
 * @param m Rows of A and C
 * @param k Columns of A, rows of B
 * @param n Columns of B and C
 */
template <typename T>
__global__ __aicore__ void MatmulKernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, uint32_t m, uint32_t k, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY); // Specify AI Core only task type

    // Initialize tiling parameters for memory hierarchy
    uint64_t baseM = 256;
    uint64_t baseN = 256;
    uint64_t baseK = 128 / sizeof(T);
    uint64_t kL1 = 512 / sizeof(T);
    uint64_t mTileNum = tool::CeilDiv(m, baseM);
    uint64_t nTileNum = tool::CeilDiv(n, baseN);
    uint64_t tileNum = mTileNum * nTileNum;
    uint64_t kL1TileNum = tool::CeilDiv(k, kL1);
    uint64_t tailKL1 = k - (kL1TileNum - 1) * kL1;
    uint64_t tailBaseM = m - (mTileNum - 1) * baseM;
    uint64_t tailBaseN = n - (nTileNum - 1) * baseN;
    uint64_t l0cOffset = 0; // L0C buffer offset

    uint64_t curBlockIdx = AscendC::GetBlockIdx();
    uint64_t blockNum = AscendC::GetBlockNum();

    // Double buffering indices
    uint64_t l0PingPong_ = 0;            // L0 buffer ping-pong index
    uint64_t l1PingPong = 0;             // L1 buffer ping-pong index
    uint64_t l1BufferAOffset[2] = {0UL}; // L1 buffer offsets for matrix A (ping/pong)
    uint64_t l1BufferBOffset[2] = {0UL}; // L1 buffer offsets for matrix B (ping/pong)

    // Construct GM tensors with appropriate layouts
    auto layoutA = AscendC::Te::MakeNDLayout<T>(m, k);
    auto layoutB = AscendC::Te::MakeNDLayout<T>(k, n);
    auto layoutC = AscendC::Te::MakeNDLayout<T>(m, n);

    auto tensorAgm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ T*>(aGm)), layoutA);
    auto tensorBgm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ T*>(bGm)), layoutB);
    auto tensorCgm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ T*>(cGm)), layoutC);

    // Initialize hardware event flags for synchronization
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(tool::ZERO_FLAG);  // MTE1->MTE2 sync
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(tool::FIRST_FLAG); // Second sync flag
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(tool::ZERO_FLAG);     // M->MTE1 sync
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(tool::FIRST_FLAG);    // Second M sync flag
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(tool::ZERO_FLAG);

    // Multi-core tile processing loop - each core processes different tiles
    for (uint64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
        uint64_t mTileIdx = tileIdx / nTileNum;
        uint64_t nTileIdx = tileIdx % nTileNum;
        int64_t curM = mTileIdx == (mTileNum - 1) ? tailBaseM : baseM;
        int64_t curN = nTileIdx == (nTileNum - 1) ? tailBaseN : baseN;

        // Extract current tile from global memory
        auto tensorAGmBlock = tensorAgm(AscendC::Te::MakeCoord(mTileIdx * baseM, 0L), AscendC::Te::MakeShape(curM, k));
        auto tensorBGmBlock = tensorBgm(AscendC::Te::MakeCoord(0L, nTileIdx * baseN), AscendC::Te::MakeShape(k, curN));
        auto tensorCGmBlock =
            tensorCgm(AscendC::Te::MakeCoord(mTileIdx * baseM, nTileIdx * baseN), AscendC::Te::MakeShape(curM, curN));

        // Setup L0C tensor for accumulation
        auto layoutL0C = AscendC::Te::MakeL0CLayout(curM, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeL0CmemPtr<float>(l0cOffset), layoutL0C);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(tool::ZERO_FLAG);
        // Loop over K dimension tiles for L1 buffer
        for (uint64_t iter0 = 0; iter0 < kL1TileNum; ++iter0) {
            uint64_t l1BufId = l1PingPong & 1;                         // Current L1 buffer ID
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId); // Wait for previous transfer

            // Determine current K tile sizes
            auto curGmBKL1 = (iter0 + 1 == kL1TileNum) ? (k - iter0 * kL1) : kL1;
            auto curGmAKL1 = curGmBKL1;

            // Copy GM to L1 buffers with double buffering
            uint64_t l1Offset = (AscendC::TOTAL_L1_SIZE >> 1) * l1BufId; // Half L1 for each buffer
            l1BufferAOffset[l1BufId] = l1Offset;
            l1BufferBOffset[l1BufId] = l1Offset + baseM * kL1 * sizeof(T);

            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto layoutAL1 = AscendC::Te::MakeLayoutAL1<T>{}(curM, curGmAKL1);
            auto tensorAL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<T>(l1BufferAOffset[l1BufId]), layoutAL1);
            // Copy A tile from GM to L1
            auto tensorAGmTile =
                tensorAGmBlock(AscendC::Te::MakeCoord(0, iter0 * kL1), AscendC::Te::MakeShape(curM, curGmAKL1));
            AscendC::Te::Copy(copyGM2L1, tensorAL1, tensorAGmTile);

            auto layoutBL1 = AscendC::Te::MakeLayoutBL1<T>{}(curGmBKL1, curN);
            auto tensorBL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<T>(l1BufferBOffset[l1BufId]), layoutBL1);
            // Copy B tile from GM to L1
            auto tensorBGmTile =
                tensorBGmBlock(AscendC::Te::MakeCoord(iter0 * kL1, 0), AscendC::Te::MakeShape(curGmBKL1, curN));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, tensorBGmTile);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            // Further tile K dimension for L0 buffer
            uint64_t kL0IterNum = tool::CeilDiv(curGmBKL1, baseK);
            uint64_t tailKL0 = curGmBKL1 - (kL0IterNum - 1) * baseK;

            for (uint16_t iter1 = 0; iter1 < kL0IterNum; ++iter1) {
                uint64_t l0BufId = l0PingPong_ & 1;               // Current L0 buffer ID
                uint64_t l0Offset = tool::HALF_L0_SIZE * l0BufId; // Offset in L0
                uint64_t curKL0 = (iter1 + 1 == kL0IterNum) ? tailKL0 : baseK;

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);

                // Copy L1 to L0 buffers
                auto copyL12L0 = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0{});
                auto layoutAL0 = AscendC::Te::MakeNzLayout<T>(curM, curKL0);
                auto tensorAL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<T>(l0Offset), layoutAL0);
                // Copy A sub-tile from L1 to L0
                auto tensorAL1Tile =
                    tensorAL1(AscendC::Te::MakeCoord(0, iter1 * baseK), AscendC::Te::MakeShape(curM, curKL0));
                AscendC::Te::Copy(copyL12L0, tensorAL0, tensorAL1Tile);

                auto layoutBL0 = AscendC::Te::MakeZnLayout<T>(curKL0, curN);
                auto tensorBL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<T>(l0Offset), layoutBL0);
                // Copy B sub-tile from L1 to L0
                auto tensorBL1Tile =
                    tensorBL1(AscendC::Te::MakeCoord(iter1 * baseK, 0), AscendC::Te::MakeShape(curKL0, curN));
                AscendC::Te::Copy(copyL12L0, tensorBL0, tensorBL1Tile);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);

                // Execute matrix multiplication on cube unit
                AscendC::MmadParams para;
                para.cmatrixInitVal = (iter1 == 0 && iter0 == 0);
                para.m = curM;
                para.n = curN;
                para.k = curKL0;

                // Perform MAD (Multiply-Add) operation: C += A * B
                auto MadOp = AscendC::Te::MakeMad(AscendC::Te::MmadOperation{}, AscendC::Te::MmadTraitDefault{});
                AscendC::Te::Mad(MadOp, tensorL0C, tensorAL0, tensorBL0, para);

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
                l0PingPong_++; // Toggle L0 buffer
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            l1PingPong++; // Toggle L1 buffer
        }

        // Wait for computation to complete
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(tool::ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(tool::ZERO_FLAG);

        // Copy result from L0C to global memory
        auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(copyL0C2GM, tensorCGmBlock, tensorL0C);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(tool::ZERO_FLAG);
    }

    // Final synchronization waits
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(tool::ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(tool::ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(tool::FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(tool::FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(tool::ZERO_FLAG);
}

} // namespace matmul

// Utility macro for condition checking with error message
#define CHECK_COND(cond, message, return_expr)              \
    do {                                                    \
        if (!(cond)) {                                      \
            std::cerr << "ERROR: " << message << std::endl; \
            return_expr;                                    \
        }                                                   \
    } while (0)

// Print command-line usage help
void printUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName << " m k n" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "Example: " << programName << " 100 50 200" << std::endl;
}

// Brief parses and validates command-line arguments
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

/**
 * @brief Main function - host-side setup and execution
 *
 * This function:
 * 1. Parses command line arguments
 * 2. Initializes Ascend Computing Language (ACL) resources
 * 3. Allocates and initializes host/device memory
 * 4. Launches the kernel
 * 5. Verifies results against CPU reference
 * 6. Cleans up resources
 */
int main(int argc, char* argv[])
{
    using namespace tool;
    int m, k, n;
    try {
        parseArguments(argc, argv, m, k, n);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Initialize ACL resources
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclInit(nullptr);
    CHECK_COND(ret == ACL_SUCCESS, "aclInit failed.", return 1);
    ret = aclrtSetDevice(deviceId);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtSetDevice failed.", return 1);
    ret = aclrtCreateStream(&stream);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtCreateStream failed.", return 1);

    // Allocate host memory and fill with random data
    std::vector<float> hostInput(m * k, 0);
    std::vector<float> hostWeight(k * n, 0);
    std::vector<float> hostOutput(m * n, 0);
    std::vector<float> goldenOutput(m * n, 0);
    FillRandomData<float>(hostInput, -2.0f, 2.0f);
    FillRandomData<float>(hostWeight, -2.0f, 2.0f);

    auto toBf16 = [](const std::vector<float>& src, std::vector<uint16_t>& dst) {
        std::transform(src.begin(), src.end(), dst.begin(), FloatToBf16);
    };
    std::vector<uint16_t> hostInputBf16(m * k, 0);
    std::vector<uint16_t> hostWeightBf16(k * n, 0);
    std::vector<uint16_t> hostOutputBf16(m * n, 0);
    std::vector<uint16_t> goldenOutputBf16(m * n, 0);
    toBf16(hostInput, hostInputBf16);
    toBf16(hostWeight, hostWeightBf16);
    toBf16(hostOutput, hostOutputBf16);
    toBf16(goldenOutput, goldenOutputBf16);

    // Allocate device memory
    GM_ADDR deviceInput = nullptr;
    GM_ADDR deviceWeight = nullptr;
    GM_ADDR deviceOutput = nullptr;
    auto sizeInput = hostInputBf16.size() * sizeof(uint16_t);
    auto sizeWeight = hostWeightBf16.size() * sizeof(uint16_t);
    auto sizeOutput = hostOutputBf16.size() * sizeof(uint16_t);
    std::unique_ptr<void, aclError (*)(void*)> deviceInputPtr(deviceInput, aclrtFree);
    ret = aclrtMalloc((void**)&deviceInput, sizeInput, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceInput failed.", return 1);

    std::unique_ptr<void, aclError (*)(void*)> deviceWeightPtr(deviceWeight, aclrtFree);
    ret = aclrtMalloc((void**)&deviceWeight, sizeWeight, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceWeight failed.", return 1);

    std::unique_ptr<void, aclError (*)(void*)> deviceOutputPtr(deviceOutput, aclrtFree);
    ret = aclrtMalloc((void**)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceOutput failed.", return 1);

    // Copy data from host to device
    ret = aclrtMemcpy(deviceInput, sizeInput, hostInputBf16.data(), sizeInput, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceInput failed.", return 1);
    ret = aclrtMemcpy(deviceWeight, sizeWeight, hostWeightBf16.data(), sizeWeight, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceWeight failed.", return 1);

    // Get platform instance to access AI core information and setup kernel timing events
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    CHECK_COND(ascendcPlatform != nullptr, "get ascendcPlatform failed.", return 1);
    uint32_t numBlocks = ascendcPlatform->GetCoreNumAic(); // Number of AI cores

    // Launch kernel on all available AI cores
    matmul::MatmulKernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(deviceInput, deviceWeight, deviceOutput, m, k, n);

    // Wait for kernel completion
    ret = aclrtSynchronizeStream(stream);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed.", return 1);

    // Copy result back from device to host
    ret = aclrtMemcpy(hostOutputBf16.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceOutput failed.", return 1);

    // Verify results against CPU golden reference
    auto toFloat = [](const std::vector<uint16_t>& src, std::vector<float>& dst) {
        std::transform(src.begin(), src.end(), dst.begin(), Bf16ToFloat);
    };

    std::vector<float> hostInputFloat(m * k, 0);
    std::vector<float> hostWeightFloat(k * n, 0);
    std::vector<float> hostOutputFloat(m * n, 0);
    std::vector<float> goldenOutputFloat(m * n, 0);
    toFloat(hostInputBf16, hostInputFloat);
    toFloat(hostWeightBf16, hostWeightFloat);
    toFloat(hostOutputBf16, hostOutputFloat);
    toFloat(goldenOutputBf16, goldenOutputFloat);

    ComputeGolden<float>(m, k, n, hostInputFloat, hostWeightFloat, goldenOutputFloat);
    std::vector<uint64_t> errorIndices = Compare<float>(hostOutputFloat, goldenOutputFloat);
    if (errorIndices.size() == 0) {
        std::cout << "matmul run successfully!" << std::endl;
    } else {
        for (uint64_t i : errorIndices) {
            std::cout << "error index: " << i << ", output: " << hostOutputFloat[i]
                      << ", golden: " << goldenOutputFloat[i] << std::endl;
        }
        std::cout << "matmul run failed!" << std::endl;
    }
    // Cleanup resources (using RAII with unique_ptr)
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

namespace tool {
/**
 * @brief Fill a vector with random data
 *
 * @tparam T Data type (integral or floating point)
 * @param data Vector to fill
 * @param min Minimum value
 * @param max Maximum value
 */
template <typename T>
void FillRandomData(std::vector<T>& data, T min, T max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(min, max);
        for (auto& elem : data)
            elem = dist(gen);
    } else if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dist(min, max);
        for (auto& elem : data)
            elem = dist(gen);
    }
}

/**
 * @brief Compute matrix multiplication on CPU as golden reference
 *
 * @tparam T Data type
 * @param m Rows of A
 * @param k Columns of A / Rows of B
 * @param n Columns of B
 * @param hostInput Matrix A
 * @param hostWeight Matrix B
 * @param goldenOutput Output matrix C (reference)
 */
template <typename T>
void ComputeGolden(
    int m, int k, int n, std::vector<T>& hostInput, std::vector<T>& hostWeight, std::vector<T>& goldenOutput)
{
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            size_t offsetGolden = row * n + col;
            T sum = 0;
            for (uint32_t iter = 0; iter < k; ++iter) {
                size_t offsetInput = row * k + iter;
                size_t offsetWeight = iter * n + col;
                sum += hostInput[offsetInput] * hostWeight[offsetWeight];
            }
            goldenOutput[offsetGolden] = sum;
        }
    }
}

/**
 * @brief Compare kernel output with golden reference
 *
 * @tparam T Data type
 * @param hostOutput Kernel output
 * @param goldenOutput CPU reference
 * @return std::vector<uint64_t> Indices where values differ beyond tolerance
 */
template <typename T>
std::vector<uint64_t> Compare(std::vector<T>& hostOutput, std::vector<T>& goldenOutput)
{
    std::vector<uint64_t> errorIndices;
    const float rtol = 1.0f / 256; // Relative tolerance for float comparison
    for (uint64_t i = 0; i < hostOutput.size(); ++i) {
        T actualValue = hostOutput[i];
        T expectValue = goldenOutput[i];
        T diff = std::fabs(actualValue - expectValue);
        if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
            errorIndices.push_back(i);
        }
    }
    return errorIndices;
}

/**
 * @brief Ceiling division for integer arithmetic
 */
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

/**
 * @brief Convert a 16-bit brain floating-point (bfloat16) value to a 32-bit float
 * @param h 16-bit bfloat16 value stored in uint16_t format
 * @return The converted 32-bit floating-point value
 */
float Bf16ToFloat(uint16_t h)
{
    uint32_t sign = (h & 0x8000U) ? 0x80000000U : 0x00000000U;
    uint32_t exponent = (h >> 7) & 0x00FFU;
    uint32_t mantissa = h & 0x007FU;
    uint32_t f_bits = sign | (exponent << 23) | (mantissa << (23 - 7));
    return *reinterpret_cast<float*>(&f_bits);
}

/**
 * @brief Convert a 32-bit float to a 16-bit brain floating-point (bfloat16) value
 * @param f 32-bit floating-point value to convert
 * @return The converted 16-bit bfloat16 value stored in uint16_t format (truncated rounding)
 */
uint16_t FloatToBf16(float f)
{
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(f_bits));

    // Extract the high 16 bits (simple truncation)
    return static_cast<uint16_t>(f_bits >> 16);
}

} // namespace tool