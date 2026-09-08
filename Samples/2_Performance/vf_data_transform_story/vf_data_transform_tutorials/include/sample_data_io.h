/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VF_DATA_TRANSFORM_STORY_SAMPLE_DATA_IO_H
#define VF_DATA_TRANSFORM_STORY_SAMPLE_DATA_IO_H

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#ifndef VF_DATA_TRANSFORM_SOURCE_DIR
#define VF_DATA_TRANSFORM_SOURCE_DIR "."
#endif

namespace VfDataTransformSample {

struct ArtifactPaths {
    std::filesystem::path root;
    std::filesystem::path input;
    std::filesystem::path outputInit;
    std::filesystem::path golden;
    std::filesystem::path goldenStorage;
    std::filesystem::path npuOutput;
    std::filesystem::path npuStorage;
};

inline std::filesystem::path GetExecutableDirectory()
{
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error && executable.has_parent_path()) {
        return executable.parent_path();
    }
    return std::filesystem::current_path();
}

inline std::filesystem::path FindScript(const char* scriptName)
{
    const auto executableDirectory = GetExecutableDirectory();
    const std::vector<std::filesystem::path> candidates = {
        std::filesystem::path(executableDirectory).append(scriptName),
        std::filesystem::path(executableDirectory).append("scripts").append(scriptName),
        std::filesystem::path(executableDirectory).append("..").append("scripts").append(scriptName),
        std::filesystem::path(executableDirectory).append("..").append("..").append("scripts").append(scriptName),
        std::filesystem::path(VF_DATA_TRANSFORM_SOURCE_DIR).append("scripts").append(scriptName),
    };
    for (const auto& candidate : candidates) {
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error) && !error) {
            return candidate;
        }
    }
    return candidates.back();
}

inline std::string ShellQuote(const std::string& value)
{
    std::string quoted("'");
    for (char character : value) {
        if (character == '\'') {
            quoted += "'\\''";
        } else {
            quoted += character;
        }
    }
    quoted += '\'';
    return quoted;
}

inline bool RunCommand(const std::string& command, const char* operation, const std::filesystem::path& logPath)
{
    const std::string redirectedCommand = command + " > " + ShellQuote(logPath.string()) + " 2>&1";
    const int status = std::system(redirectedCommand.c_str());
    if (status == 0) {
        return true;
    }
    std::cerr << "[HOST][ERROR] " << operation << " failed with status " << status << ": " << command << '\n';
    std::ifstream log(logPath);
    if (log.is_open()) {
        std::cerr << log.rdbuf();
    }
    return false;
}

inline ArtifactPaths MakeArtifactPaths(
    const std::filesystem::path& base, const std::string& task, const std::string& caseName)
{
    const auto root = std::filesystem::path(base).append(task).append(caseName);
    return {
        root,
        std::filesystem::path(root).append("input").append("input.bin"),
        std::filesystem::path(root).append("input").append("output_init.bin"),
        std::filesystem::path(root).append("output").append("golden.bin"),
        std::filesystem::path(root).append("output").append("golden_storage.bin"),
        std::filesystem::path(root).append("output").append("npu_out.bin"),
        std::filesystem::path(root).append("output").append("npu_storage.bin"),
    };
}

inline bool GenerateData(
    const std::string& task, const std::string& caseName, uint32_t m, uint32_t n, uint64_t inputCount,
    uint64_t outputStorageCount, uint64_t outputOffset, uint64_t outputCount, ArtifactPaths& paths)
{
    const char* configuredArtifactDirectory = std::getenv("VF_DATA_TRANSFORM_ARTIFACT_DIR");
    const bool hasConfiguredArtifactDirectory =
        configuredArtifactDirectory != nullptr && configuredArtifactDirectory[0] != '\0';
    const auto defaultBase = GetExecutableDirectory().append("artifacts");
    paths = MakeArtifactPaths(
        hasConfiguredArtifactDirectory ? std::filesystem::path(configuredArtifactDirectory) : defaultBase, task,
        caseName);
    std::error_code error;
    std::filesystem::create_directories(paths.root, error);
    if (error && !hasConfiguredArtifactDirectory) {
        error.clear();
        paths = MakeArtifactPaths(std::filesystem::current_path().append("artifacts"), task, caseName);
        std::filesystem::create_directories(paths.root, error);
    }
    if (error) {
        std::cerr << "[HOST][ERROR] Cannot create artifact directory " << paths.root << ": " << error.message() << '\n';
        return false;
    }
    std::ostringstream command;
    command << "env -u LD_LIBRARY_PATH -u LD_PRELOAD python3 " << ShellQuote(FindScript("gen_data.py").string())
            << " --task " << ShellQuote(task) << " --case " << ShellQuote(caseName) << " --m " << m << " --n " << n
            << " --input-count " << inputCount << " --output-storage-count " << outputStorageCount
            << " --output-offset " << outputOffset << " --output-count " << outputCount << " --output-dir "
            << ShellQuote(paths.root.string());
    return RunCommand(
        command.str(), "Python data generation", std::filesystem::path(paths.root).append("generate.log"));
}

template <typename T>
inline bool ReadBinaryExact(const std::filesystem::path& path, uint64_t expectedCount, std::vector<T>& data)
{
    if (expectedCount > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(T))) {
        std::cerr << "[HOST][ERROR] Binary element count is too large: " << expectedCount << '\n';
        return false;
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[HOST][ERROR] Cannot open binary file: " << path << '\n';
        return false;
    }
    const auto fileSize = file.tellg();
    const uint64_t expectedBytes = expectedCount * sizeof(T);
    if (fileSize < 0 || static_cast<uint64_t>(fileSize) != expectedBytes) {
        std::cerr << "[HOST][ERROR] Unexpected binary size for " << path << ": expected " << expectedBytes
                  << ", actual " << fileSize << '\n';
        return false;
    }
    data.resize(static_cast<size_t>(expectedCount));
    file.seekg(0, std::ios::beg);
    if (expectedBytes != 0) {
        file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(expectedBytes));
    }
    if (!file) {
        std::cerr << "[HOST][ERROR] Failed to read binary file: " << path << '\n';
        return false;
    }
    return true;
}

inline bool WriteBinary(const std::filesystem::path& path, const void* data, uint64_t bytes)
{
    if (data == nullptr && bytes != 0) {
        std::cerr << "[HOST][ERROR] Cannot write a null buffer to " << path << '\n';
        return false;
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "[HOST][ERROR] Cannot create binary file: " << path << '\n';
        return false;
    }
    if (bytes != 0) {
        file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    }
    if (!file) {
        std::cerr << "[HOST][ERROR] Failed to write binary file: " << path << '\n';
        return false;
    }
    return true;
}

template <typename T>
inline bool VerifyOutput(
    const ArtifactPaths& paths, const std::vector<T>& outputStorage, uint64_t outputOffset, uint64_t outputCount)
{
    if (outputOffset > outputStorage.size() || outputCount > outputStorage.size() - outputOffset) {
        std::cerr << "[HOST][ERROR] Output payload exceeds output storage\n";
        return false;
    }
    if (!WriteBinary(paths.npuStorage, outputStorage.data(), outputStorage.size() * sizeof(T)) ||
        !WriteBinary(paths.npuOutput, outputStorage.data() + outputOffset, outputCount * sizeof(T))) {
        return false;
    }
    std::ostringstream command;
    command << "env -u LD_LIBRARY_PATH -u LD_PRELOAD python3 " << ShellQuote(FindScript("verify_result.py").string())
            << " --data-dir " << ShellQuote(paths.root.string());
    return RunCommand(
        command.str(), "Python binary verification", std::filesystem::path(paths.root).append("verify.log"));
}

} // namespace VfDataTransformSample

#endif // VF_DATA_TRANSFORM_STORY_SAMPLE_DATA_IO_H
