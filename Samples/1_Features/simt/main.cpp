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
 * \brief
 */

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <limits>

typedef int32_t dataType;
typedef int32_t indicesType;

template <uint32_t MAX_THREADNUM, typename DATA_TYPE, typename INDICES_TYPE, typename INDEX_SIZE_TYPE>
inline __simt_vf__ [aicore] __launch_bounds__(MAX_THREADNUM) void gather_function(__gm__ DATA_TYPE *x, __gm__ INDICES_TYPE *indices, __gm__ DATA_TYPE *y, 
    INDEX_SIZE_TYPE gatherDimSize, INDEX_SIZE_TYPE indicesDimSize, INDEX_SIZE_TYPE innerDimSize, INDEX_SIZE_TYPE outNum) {
    for (INDEX_SIZE_TYPE idx = threadIdx.x + blockIdx.x * blockDim.x; idx < outNum; 
        idx += block_num * blockDim.x) {
        INDEX_SIZE_TYPE outerI = idx / (gatherDimSize * innerDimSize);
        INDEX_SIZE_TYPE tmpI = idx - outerI * (gatherDimSize * innerDimSize);
        INDEX_SIZE_TYPE gatherI = tmpI / innerDimSize;
        INDEX_SIZE_TYPE innerI = tmpI - gatherI * innerDimSize;
        INDICES_TYPE indicesValue = indices[gatherI];
        INDEX_SIZE_TYPE indicesValueI = static_cast<INDEX_SIZE_TYPE>(indicesValue);
        INDEX_SIZE_TYPE xIndex = outerI * gatherDimSize * innerDimSize + indicesValueI * innerDimSize + innerI;
        // indices overflow
        bool indexOutOfBound = indicesValue < 0 || indicesValue >= gatherDimSize;
        y[idx] = indexOutOfBound ? 0 : x[xIndex];
    }
}

template <typename DATA_TYPE, typename INDICES_TYPE>
__global__ [aicore] __attribute__ ((aiv)) void gather(__gm__ DATA_TYPE *x, __gm__ INDICES_TYPE *indices, __gm__ DATA_TYPE *y, 
    size_t outerDimSize, size_t gatherDimSize, size_t innerDimSize, size_t indicesDimSize) {
    constexpr uint64_t INT32_MAX_SIZE = std::numeric_limits<int32_t>::max();
    if ((outerDimSize * gatherDimSize * innerDimSize < INT32_MAX_SIZE) && (outerDimSize * indicesDimSize * innerDimSize < INT32_MAX_SIZE)) {
        asc_vf_call<gather_function<2048, DATA_TYPE, INDICES_TYPE, int32_t>>(dim3(2048), x, indices, y, gatherDimSize, indicesDimSize, innerDimSize, indicesDimSize * innerDimSize);
    } else {
        asc_vf_call<gather_function<2048, DATA_TYPE, INDICES_TYPE, int64_t>>(dim3(2048), x, indices, y, gatherDimSize, indicesDimSize, innerDimSize, indicesDimSize * innerDimSize);
    }

}

void CHECK_ACL(aclError __ret) {
    if (__ret != ACL_ERROR_NONE) 
        std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl;
}

size_t segmentProduct(const std::vector<size_t>& vec, size_t i, size_t j) {
    if (i < 0 || j > vec.size() || i > j) {
        std::cerr << "Invalid indices" << std::endl;
        return 0;
    }

    size_t product = 1;
    for (size_t k = i; k < j; ++k) {
        product *= vec[k];
    }
    return product;
}

template <typename T>
void genIndiceData(size_t dimSize, size_t numSamples, std::vector<T>& res) {
    std::vector<T> sequenceData(dimSize);
    std::iota(sequenceData.begin(), sequenceData.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, sequenceData.size() - 1);
    for (size_t i = 0; i < numSamples; ++i) {
        size_t randomIndex = dis(gen);
        res.push_back(sequenceData[randomIndex]);
    }
}

template <typename T>
void genInputData(size_t totalSize, std::vector<T>& res) {
    res.resize(totalSize);
    std::iota(res.begin(), res.end(), 0);
}

int32_t main() {
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    std::vector<size_t> inputShape = {8, 5, 3, 4};
    std::vector<size_t> indicesShape = {3, };
    std::vector<size_t> outputShape = {8, 3, 3, 4};

    size_t gatherDim = 1;
    size_t inputEleNum = segmentProduct(inputShape, 0, inputShape.size());
    size_t indicesEleNum = segmentProduct(indicesShape, 0, indicesShape.size());
    size_t outputEleNum = segmentProduct(outputShape, 0, outputShape.size());
    size_t inputSize = inputEleNum * sizeof(dataType);
    size_t indicesSize = indicesEleNum * sizeof(indicesType);
    size_t outputSize = outputEleNum * sizeof(dataType);

    size_t outerDimSize = segmentProduct(inputShape, 0, gatherDim);
    size_t gatherDimSize = inputShape[gatherDim];
    size_t indicesDimSize = segmentProduct(indicesShape, 0, indicesShape.size());
    size_t innerDimSize = segmentProduct(inputShape, gatherDim + 1, outputShape.size());

    size_t count = outputEleNum; 

    size_t showNum = 3 * innerDimSize;
    std::vector<dataType> inputData;
    genInputData(inputEleNum, inputData);

    std::vector<indicesType> indicesData;
    genIndiceData(gatherDimSize, indicesDimSize, indicesData);

    dataType *xDevice;
    indicesType *indicesDevice;
    dataType *yHost;
    dataType *yDevice;

    uint32_t threadNum = 2048; 
    uint64_t maxCoreNum = 4; // just an example, actual max_core_num should be based on hardware information.
    uint64_t tempUsedBlockNum = (static_cast<uint64_t>(outerDimSize) + threadNum - 1) / threadNum;
    uint64_t numBlock = std::min(tempUsedBlockNum, maxCoreNum);

    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&indicesDevice, indicesSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost((void **)&yHost, outputSize));
    CHECK_ACL(aclrtMemcpy(xDevice, inputSize, inputData.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(indicesDevice, indicesSize, indicesData.data(), indicesSize, ACL_MEMCPY_HOST_TO_DEVICE));
    gather<dataType, indicesType><<<numBlock, 0, stream>>>(xDevice, indicesDevice, yDevice, outerDimSize, gatherDimSize, innerDimSize, indicesDimSize);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(yHost, outputSize, yDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    std::cout << "The " << showNum << " elements of input are as followed: ";
    for (auto i =0; i < showNum; i++) {
        std::cout << inputData[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "The " << indicesDimSize << " elements of indices are as followed: ";
    for (auto i =0; i < indicesDimSize; i++) {
        std::cout << indicesData[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "The " << showNum << " elements of output are as followed: ";
    for (auto i =0; i < showNum; i++) {
        std::cout << *((dataType *)(yHost) + i) << " ";
    }
    std::cout << std::endl;

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(indicesDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFreeHost(yHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}
