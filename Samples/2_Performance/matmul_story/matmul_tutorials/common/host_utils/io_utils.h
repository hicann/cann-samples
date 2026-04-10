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
 * \file io_utils.h
 * \brief
 */

#ifndef IO_UTILS_H
#define IO_UTILS_H

#include <fcntl.h>
#include <fstream>
#include <limits.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "common_utils.h"

// Flat install: gen_data.py next to the executable. Source tree: under matmul_tutorials/scripts/.
inline std::string ResolveMatmulTutorialWorkspaceDir()
{
    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (len <= 0) {
        return ".";
    }
    exePath[len] = '\0';
    std::string path(exePath);
    size_t slash = path.find_last_of('/');
    if (slash != std::string::npos) {
        path.resize(slash);
    }
    const std::string exeDir = path;

    auto hasGenData = [](const std::string& dir) {
        return access((dir + "/gen_data.py").c_str(), F_OK) == 0;
    };

    if (hasGenData(exeDir)) {
        return exeDir;
    }
    if (hasGenData(exeDir + "/scripts")) {
        return exeDir + "/scripts";
    }
    slash = exeDir.find_last_of('/');
    if (slash != std::string::npos && slash > 0) {
        const std::string parent = exeDir.substr(0, slash);
        if (hasGenData(parent + "/scripts")) {
            return parent + "/scripts";
        }
        if (hasGenData(parent)) {
            return parent;
        }
        return parent + "/scripts";
    }
    return exeDir + "/scripts";
}

inline bool ReadFile(const std::string& filePath, size_t& fileSize, void* buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file");
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf* buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char*>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

/**
 * @brief Write data to file
 * @param [in] filePath: file path
 * @param [in] buffer: data to write to file
 * @param [in] size: size to write
 * @return write result
 */
inline bool WriteFile(const std::string& filePath, const void* buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t writeSize = write(fd, buffer, size);
    (void)close(fd);
    if (writeSize != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

#endif // IO_UTILS_H
