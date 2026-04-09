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
 * \brief Implementation of matrix multiplication kernel for Ascend processors
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

namespace tool {
constexpr static uint16_t ZERO_FLAG = 0;                    // Zero flag value for synchronization
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b); // Helper function for ceiling division
float Bf16ToFloat(uint16_t h);
uint16_t FloatToBf16(float f);
template <typename T>
void FillRandomData(std::vector<T>& data, T min, T max); // Fill vector with random data
template <typename T>
void ComputeGolden(
    int m, int k, int n, std::vector<T>& hostInput, std::vector<T>& hostWeight,
    std::vector<T>& goldenOutput); // Compute reference result on CPU
template <typename T>
std::vector<uint64_t> Compare(
    std::vector<T>& hostOutput, std::vector<T>& goldenOutput); // Compare device output with golden reference
} // namespace tool

namespace AscendC::Te {
static constexpr bool transA = false; // Whether to transpose matrix A
static constexpr bool transB = false; // Whether to transpose matrix B
template <typename T>
using MakeLayoutAL1 =
    AscendC::Std::conditional_t<transA, AscendC::Te::ZnLayoutFormat<T>, AscendC::Te::NzLayoutFormat<T>>; // Layout for A
                                                                                                         // matrix in L1
template <typename T>
using MakeLayoutBL1 =
    AscendC::Std::conditional_t<transB, AscendC::Te::ZnLayoutFormat<T>, AscendC::Te::NzLayoutFormat<T>>; // Layout for B
                                                                                                         // matrix in L1
} // namespace AscendC::Te

namespace matmul {
/**
 * @brief Simple kernel example for matrix multiplication on Ascend processors
 * @tparam T Data type of matrices (typically float)
 * @param aGm Global memory address of input matrix A (size m×k)
 * @param bGm Global memory address of input matrix B (size k×n)
 * @param cGm Global memory address of output matrix C (size m×n)
 * @param m Number of rows in matrix A
 * @param k Number of columns in matrix A / rows in matrix B
 * @param n Number of columns in matrix B
 */
template <typename T>
__global__ __aicore__ void MatmulKernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, uint32_t m, uint32_t k, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY); // Specify kernel type as AI Core only

    // Initialize tiling parameters for efficient memory hierarchy utilization
    uint64_t baseM = 256;             // Base tile size for M dimension in L0
    uint64_t baseN = 256;             // Base tile size for N dimension in L0
    uint64_t baseK = 256 / sizeof(T); // Base tile size for K dimension in L0
    uint64_t kL1 = 1024 / sizeof(T);  // Tile size for K dimension in L1
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

    // Construct GM tensors with ND layout
    auto layoutA = AscendC::Te::MakeNDLayout<T>(m, k);
    auto layoutB = AscendC::Te::MakeNDLayout<T>(k, n);
    auto layoutC = AscendC::Te::MakeNDLayout<T>(m, n);

    auto tensorAgm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ T*>(aGm)), layoutA);
    auto tensorBgm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ T*>(bGm)), layoutB);
    auto tensorCgm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ T*>(cGm)), layoutC);

    // Initialize synchronization flags for pipeline control
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(tool::ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(tool::ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(tool::ZERO_FLAG);

    // Multi-core tile processing loop - distribute tiles across available cores
    for (uint64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
        uint64_t mTileIdx = tileIdx / nTileNum;
        uint64_t nTileIdx = tileIdx % nTileNum;
        int64_t curM = mTileIdx == (mTileNum - 1) ? tailBaseM : baseM;
        int64_t curN = nTileIdx == (nTileNum - 1) ? tailBaseN : baseN;

        // Slice GM to L1 - prepare memory pointers for this tile
        auto tensorAGmBlock = tensorAgm(AscendC::Te::MakeCoord(mTileIdx * baseM, 0L), AscendC::Te::MakeShape(curM, k));
        auto tensorBGmBlock = tensorBgm(AscendC::Te::MakeCoord(0L, nTileIdx * baseN), AscendC::Te::MakeShape(k, curN));

        auto tensorCGmBlock =
            tensorCgm(AscendC::Te::MakeCoord(mTileIdx * baseM, nTileIdx * baseN), AscendC::Te::MakeShape(curM, curN));
        auto layoutL0C = AscendC::Te::MakeL0CLayout(curM, curN); // L0C layout for output
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeL0CmemPtr<float>(l0cOffset), layoutL0C);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(tool::ZERO_FLAG);
        // Process K dimension in multiple L1 tiles
        for (uint64_t iter0 = 0; iter0 < kL1TileNum; ++iter0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(tool::ZERO_FLAG); // Wait for previous MTE1-MTE2 operations

            auto curGmBKL1 = (iter0 + 1 == kL1TileNum) ? (k - iter0 * kL1) : kL1;
            auto curGmAKL1 = curGmBKL1;

            // Copy GM to L1 buffers
            uint64_t l1BufferAOffset = 0;
            uint64_t l1BufferBOffset = baseM * kL1 * sizeof(T);

            // GM to L1 copy operation
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto layoutAL1 = AscendC::Te::MakeLayoutAL1<T>{}(curM, curGmAKL1); // Layout for A in L1
            auto tensorAL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<T>(l1BufferAOffset), layoutAL1);
            auto tensorAGmTile =
                tensorAGmBlock(AscendC::Te::MakeCoord(0, iter0 * kL1), AscendC::Te::MakeShape(curM, curGmAKL1));
            AscendC::Te::Copy(copyGM2L1, tensorAL1, tensorAGmTile); // Copy A tile from GM to L1

            auto layoutBL1 = AscendC::Te::MakeLayoutBL1<T>{}(curGmBKL1, curN); // Layout for B in L1
            auto tensorBL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<T>(l1BufferBOffset), layoutBL1);
            auto tensorBGmTile =
                tensorBGmBlock(AscendC::Te::MakeCoord(iter0 * kL1, 0), AscendC::Te::MakeShape(curGmBKL1, curN));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, tensorBGmTile); // Copy B tile from GM to L1

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(tool::ZERO_FLAG);  // Signal MTE2-MTE1 completion
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(tool::ZERO_FLAG); // Wait for MTE2-MTE1 operations

            // Process K dimension in multiple L0 tiles
            uint64_t kL0IterNum = tool::CeilDiv(curGmBKL1, baseK);
            uint64_t tailKL0 = curGmBKL1 - (kL0IterNum - 1) * baseK;
            for (uint16_t iter1 = 0; iter1 < kL0IterNum; ++iter1) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(tool::ZERO_FLAG); // Wait for M-MTE1 operations

                uint64_t curKL0 = (iter1 + 1 == kL0IterNum) ? tailKL0 : baseK;

                // Copy L1 to L0 buffers
                auto copyL12L0 = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0{}); // L1 to L0 copy operation
                auto layoutAL0 = AscendC::Te::MakeNzLayout<T>(curM, curKL0);      // NZ layout for A in L0
                auto tensorAL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<T>(0), layoutAL0);
                auto tensorAL1Tile =
                    tensorAL1(AscendC::Te::MakeCoord(0, iter1 * baseK), AscendC::Te::MakeShape(curM, curKL0));
                AscendC::Te::Copy(copyL12L0, tensorAL0, tensorAL1Tile); // Copy A tile from L1 to L0

                auto layoutBL0 = AscendC::Te::MakeZnLayout<T>(curKL0, curN); // ZN layout for B in L0
                auto tensorBL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<T>(0), layoutBL0);
                auto tensorBL1Tile =
                    tensorBL1(AscendC::Te::MakeCoord(iter1 * baseK, 0), AscendC::Te::MakeShape(curKL0, curN));
                AscendC::Te::Copy(copyL12L0, tensorBL0, tensorBL1Tile); // Copy B tile from L1 to L0

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(tool::ZERO_FLAG);  // Signal MTE1-M completion
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(tool::ZERO_FLAG); // Wait for MTE1-M operations

                // Execute M-MAD operation (matrix multiplication)
                AscendC::MmadParams para;
                para.cmatrixInitVal = (iter1 == 0 && iter0 == 0); // Initialize C matrix on first iteration
                para.m = curM;
                para.n = curN;
                para.k = curKL0;

                auto MadOp = AscendC::Te::MakeMad(AscendC::Te::MmadOperation{}, AscendC::Te::MmadTraitDefault{});
                AscendC::Te::Mad(MadOp, tensorL0C, tensorAL0, tensorBL0, para);

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(tool::ZERO_FLAG);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(tool::ZERO_FLAG);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(tool::ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(tool::ZERO_FLAG);

        // Copy result from L0C to GM
        auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{}); // L0C to GM copy operation
        AscendC::Te::Copy(copyL0C2GM, tensorCGmBlock, tensorL0C);           // Copy computed tile to global memory

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(tool::ZERO_FLAG);
    }
    // Final synchronization for all pipeline stages
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(tool::ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(tool::ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(tool::ZERO_FLAG);
}

} // namespace matmul

// Macro for condition checking with error message
#define CHECK_COND(cond, message, return_expr)              \
    do {                                                    \
        if (!(cond)) {                                      \
            std::cerr << "ERROR: " << message << std::endl; \
            return_expr;                                    \
        }                                                   \
    } while (0)

/**
 * @brief Print command-line usage help
 * @param programName Name of the executable
 */
void printUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName << " m k n" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "Example: " << programName << " 100 50 200" << std::endl;
}

/**
 * @brief Parse and validate command-line arguments
 * @param argc Argument count
 * @param argv Argument vector
 * @param m Output parameter for M dimension
 * @param k Output parameter for K dimension
 * @param n Output parameter for N dimension
 * @throws std::invalid_argument on invalid input
 */
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
 * @brief Main function - host code for matrix multiplication
 * @param argc Argument count
 * @param argv Argument vector
 * @return 0 on success, non-zero on failure
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

    // Initialize ACL (Ascend Computing Language) resources
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclInit(nullptr);
    CHECK_COND(ret == ACL_SUCCESS, "aclInit failed.", return 1);
    ret = aclrtSetDevice(deviceId);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtSetDevice failed.", return 1);
    ret = aclrtCreateStream(&stream);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtCreateStream failed.", return 1);

    // Allocate host memory and fill with random data
    std::vector<float> hostInput(m * k, 0);    // Input matrix A
    std::vector<float> hostWeight(k * n, 0);   // Input matrix B
    std::vector<float> hostOutput(m * n, 0);   // Output matrix C from device
    std::vector<float> goldenOutput(m * n, 0); // Reference output from CPU
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
    GM_ADDR deviceInput = nullptr;  // Device pointer for matrix A
    GM_ADDR deviceWeight = nullptr; // Device pointer for matrix B
    GM_ADDR deviceOutput = nullptr; // Device pointer for matrix C
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
    uint32_t numBlocks = ascendcPlatform->GetCoreNumAic(); // Get number of AI cores

    // Launch kernel on device
    matmul::MatmulKernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(deviceInput, deviceWeight, deviceOutput, m, k, n);

    // Wait for kernel completion
    ret = aclrtSynchronizeStream(stream);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed.", return 1);

    // Copy result from device back to host
    ret = aclrtMemcpy(hostOutputBf16.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceOutput failed.", return 1);

    // Verify results against CPU reference
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

    // Cleanup resources
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

namespace tool {
/**
 * @brief Compute ceiling division of two integers
 * @param a Numerator
 * @param b Denominator
 * @return Ceiling of a/b
 */
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

/**
 * @brief Fill a vector with random data within specified range
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
 * @brief Compute reference matrix multiplication result on CPU
 * @tparam T Data type
 * @param m Number of rows in matrix A
 * @param k Number of columns in matrix A / rows in matrix B
 * @param n Number of columns in matrix B
 * @param hostInput Input matrix A (size m×k)
 * @param hostWeight Input matrix B (size k×n)
 * @param goldenOutput Output reference matrix C (size m×n)
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
 * @brief Compare device output with golden reference
 * @tparam T Data type
 * @param hostOutput Output from device
 * @param goldenOutput Reference output from CPU
 * @return Vector of indices where values differ beyond tolerance
 */
template <typename T>
std::vector<uint64_t> Compare(std::vector<T>& hostOutput, std::vector<T>& goldenOutput)
{
    std::vector<uint64_t> errorIndices;
    const float rtol = 1.0f / 256; // Relative tolerance (1/256 ≈ 0.0039)
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