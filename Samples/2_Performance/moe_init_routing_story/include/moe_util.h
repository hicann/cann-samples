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
 * \file moe_util.h
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint> 

#define CHECK_ACL(x) do { \
    aclError err = (x); \
    if (err != ACL_SUCCESS) { \
        printf("ACL Error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

const static int64_t UB_BLOCK_SIZE = 32;
const static int64_t MAX_BUFFER_NUM = 8;

template <typename T>
bool GetDataFromBin(const std::string &fileName, std::vector<T> &data)
{
    std::ifstream file(fileName, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << fileName << std::endl;
        return false;
    }

    data.clear();

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize == 0) {
        std::cerr << "Warning: File is empty" << std::endl;
        file.close();
        return false;
    }

    size_t numElements = fileSize / sizeof(T);
    size_t remainder = fileSize % sizeof(T);

    if (remainder != 0) {
        std::cerr << "Warning: File size (" << fileSize << " bytes) is not a multiple of element size (" << sizeof(T)
                  << " bytes)" << std::endl;
        std::cerr << "Ignoring last " << remainder << " bytes of incomplete data" << std::endl;
    }

    if (numElements > 0) {
        data.resize(numElements);

        file.read(reinterpret_cast<char *>(data.data()), numElements * sizeof(T));

        std::streamsize bytesRead = file.gcount();
        if (bytesRead != static_cast<std::streamsize>(numElements * sizeof(T))) {
            std::cerr << "Warning: Actual bytes read (" << bytesRead << ") does not match expected ("
                      << numElements * sizeof(T) << ")" << std::endl;

            size_t actualElements = bytesRead / sizeof(T);
            data.resize(actualElements);
        }
    }

    file.close();
    return data.size() > 0;
}

template <typename T>
bool WriteDataToBin(const std::string &fileName, const T *data, size_t numElements)
{
    std::ofstream file(fileName, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << fileName << std::endl;
        return false;
    }

    file.write(reinterpret_cast<const char *>(data), numElements * sizeof(T));
    
    if (file.fail()) {
        std::cerr << "Error: Failed to write data to file: " << fileName << std::endl;
        file.close();
        return false;
    }
    
    file.close();
    return true;
}

std::string GetExeDir()
{
    char path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len != -1) {
        path[len] = '\0';
        return std::string(dirname(path));
    }
    return ".";
}

int64_t CeilLog4(int64_t x)
{
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(4)));
}

int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

int64_t CeilAlign(int64_t x, int64_t y)
{
    if (y > 0) {
        return (x + y - 1) / y * y;
    }
    return 0;
};

int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE / bytes;
}

int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE;
}