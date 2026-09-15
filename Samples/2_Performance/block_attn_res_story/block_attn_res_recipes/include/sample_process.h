/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_STORY_SAMPLE_PROCESS_H_
#define BLOCK_ATTN_RES_STORY_SAMPLE_PROCESS_H_

#include <cerrno>
#include <cstring>
#include <iostream>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace BlockAttnResStory {

inline int RunPython(const std::vector<std::string>& arguments)
{
    std::vector<std::string> command = {"python3"};
    command.insert(command.end(), arguments.begin(), arguments.end());
    std::vector<char*> argv;
    for (auto& argument : command) {
        argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);

    // Keep CANN's library path out of Python without changing the parent's environment.
    std::vector<char*> environment;
    for (char** entry = ::environ; *entry != nullptr; ++entry) {
        if (std::strncmp(*entry, "LD_LIBRARY_PATH=", sizeof("LD_LIBRARY_PATH=") - 1) != 0) {
            environment.push_back(*entry);
        }
    }
    environment.push_back(nullptr);

    pid_t child;
    const int error = posix_spawnp(&child, "python3", nullptr, nullptr, argv.data(), environment.data());
    if (error != 0) {
        std::cerr << "cannot start python3: " << std::strerror(error) << std::endl;
        return -1;
    }
    int status = 0;
    pid_t waited;
    do {
        waited = waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited < 0) {
        std::cerr << "cannot wait for python3: " << std::strerror(errno) << std::endl;
        return -1;
    }
    if (WIFEXITED(status)) {
        const int exitCode = WEXITSTATUS(status);
        if (exitCode != 0) {
            std::cerr << "python3 failed, exit code=" << exitCode << std::endl;
        }
        return exitCode;
    }
    if (WIFSIGNALED(status)) {
        std::cerr << "python3 terminated by signal=" << WTERMSIG(status) << std::endl;
    }
    return -1;
}

} // namespace BlockAttnResStory

#endif // BLOCK_ATTN_RES_STORY_SAMPLE_PROCESS_H_
