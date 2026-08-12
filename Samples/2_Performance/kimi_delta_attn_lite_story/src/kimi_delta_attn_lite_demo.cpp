/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "kimi_delta_attn_lite.h"

#define CHECK_ACL(call)                                                                                    \
    do {                                                                                                   \
        aclError err = (call);                                                                             \
        if (err != ACL_SUCCESS) {                                                                          \
            std::fprintf(stderr, "ACL 错误 %d，位置：%s:%d\n", static_cast<int>(err), __FILE__, __LINE__); \
            return 1;                                                                                      \
        }                                                                                                  \
    } while (0)

#ifndef KDALITE_VERSION_ID
#define KDALITE_VERSION_ID 0
#endif

namespace {

constexpr uint32_t DEFAULT_B = 1;
constexpr uint32_t DEFAULT_S = 16;
constexpr uint32_t HEAD_DIM = 128;

struct AclrtFreeDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

struct AclFinalizeGuard {
    ~AclFinalizeGuard()
    {
        const aclError error = aclFinalize();
        if (error != ACL_SUCCESS) {
            std::fprintf(stderr, "ACL 清理失败：aclFinalize 返回 %d\n", static_cast<int>(error));
        }
    }
};

struct AclDeviceGuard {
    explicit AclDeviceGuard(int32_t deviceId) : deviceId(deviceId)
    {}

    ~AclDeviceGuard()
    {
        const aclError error = aclrtResetDevice(deviceId);
        if (error != ACL_SUCCESS) {
            std::fprintf(stderr, "ACL 清理失败：aclrtResetDevice 返回 %d\n", static_cast<int>(error));
        }
    }

    int32_t deviceId;
};

struct AclStreamGuard {
    explicit AclStreamGuard(aclrtStream stream) : stream(stream)
    {}

    ~AclStreamGuard()
    {
        if (!synchronized) {
            const aclError syncError = aclrtSynchronizeStream(stream);
            if (syncError != ACL_SUCCESS) {
                std::fprintf(stderr, "ACL 清理失败：aclrtSynchronizeStream 返回 %d\n", static_cast<int>(syncError));
            }
        }
        const aclError destroyError = aclrtDestroyStream(stream);
        if (destroyError != ACL_SUCCESS) {
            std::fprintf(stderr, "ACL 清理失败：aclrtDestroyStream 返回 %d\n", static_cast<int>(destroyError));
        }
    }

    void MarkSynchronized()
    {
        synchronized = true;
    }

    aclrtStream stream;
    bool synchronized = false;
};

std::string GetExeDir()
{
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return ".";
    }
    buf[n] = '\0';
    std::string path(buf);
    const size_t pos = path.find_last_of('/');
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

int RunCmd(const std::string& cmd)
{
    std::printf("  $ %s\n", cmd.c_str());
    return std::system(cmd.c_str());
}

bool ReadBin(const std::string& path, size_t expectedBytes, std::vector<uint8_t>& out)
{
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        std::fprintf(stderr, "读取失败：%s\n", path.c_str());
        return false;
    }
    std::fseek(file, 0, SEEK_END);
    const long fileBytes = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if (fileBytes < 0 || static_cast<uint64_t>(fileBytes) != expectedBytes) {
        std::fprintf(stderr, "文件字节数不符：%s，得到 %ld，期望 %zu\n", path.c_str(), fileBytes, expectedBytes);
        std::fclose(file);
        return false;
    }
    out.resize(expectedBytes);
    const size_t readBytes = std::fread(out.data(), 1, expectedBytes, file);
    std::fclose(file);
    return readBytes == expectedBytes;
}

bool WriteBin(const std::string& path, const void* data, size_t bytes)
{
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        std::fprintf(stderr, "写入失败：%s\n", path.c_str());
        return false;
    }
    const size_t writtenBytes = std::fwrite(data, 1, bytes, file);
    std::fclose(file);
    return writtenBytes == bytes;
}

bool ParseU32(const char* text, uint32_t& value)
{
    try {
        size_t pos = 0;
        const unsigned long long parsed = std::stoull(text, &pos);
        if (pos != std::string(text).size() || parsed > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        value = static_cast<uint32_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

void PrintUsage(const char* program)
{
    std::fprintf(
        stderr,
        "用法：%s [--size <B> <S>] [--core-num <n>] [--dry-run]  "
        "(N=1，Dk=Dv=128)\n"
        "  --core-num：指定最多使用的正整数个 Mix 组，不能超过本卡 Mix 组数；纯 AIV kernel 最多可使用 2*n 个 AIV。\n"
        "  --dry-run：真实执行 kernel、同步、回读并落盘输出，仅跳过 Golden 与比对。\n",
        program);
}

bool ParseArgs(
    int argc, char** argv, uint32_t& batchSize, uint32_t& seqLen, uint32_t& requestedMixCoreNum, bool& dryRun)
{
    batchSize = DEFAULT_B;
    seqLen = DEFAULT_S;
    requestedMixCoreNum = 0;
    dryRun = false;
    bool hasSize = false;
    for (int i = 1; i < argc;) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        }
        if (arg == "--dry-run" && !dryRun) {
            dryRun = true;
            ++i;
            continue;
        }
        if (arg == "--size" && !hasSize && i + 2 < argc && ParseU32(argv[i + 1], batchSize) &&
            ParseU32(argv[i + 2], seqLen) && batchSize > 0 && seqLen > 0) {
            hasSize = true;
            i += 3;
            continue;
        }
        if (arg == "--core-num" && requestedMixCoreNum == 0 && i + 1 < argc &&
            ParseU32(argv[i + 1], requestedMixCoreNum) && requestedMixCoreNum > 0) {
            i += 2;
            continue;
        }
        PrintUsage(argv[0]);
        return false;
    }
    return true;
}

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    uint32_t batchSize = 0;
    uint32_t seqLen = 0;
    uint32_t requestedMixCoreNum = 0;
    bool dryRun = false;
    if (!ParseArgs(argc, argv, batchSize, seqLen, requestedMixCoreNum, dryRun)) {
        return 1;
    }

    uint64_t tokenCount = 0;
    uint64_t qkvElements = 0;
    uint64_t qkvBytes64 = 0;
    uint64_t logDecayBytes64 = 0;
    uint64_t betaBytes64 = 0;
    uint64_t stateElements = 0;
    uint64_t stateBytes64 = 0;
    if (!CheckedMul(batchSize, seqLen, tokenCount) || !CheckedMul(tokenCount, HEAD_DIM, qkvElements) ||
        !CheckedMul(qkvElements, sizeof(uint16_t), qkvBytes64) ||
        !CheckedMul(qkvElements, sizeof(float), logDecayBytes64) ||
        !CheckedMul(tokenCount, sizeof(uint16_t), betaBytes64) ||
        !CheckedMul(batchSize, static_cast<uint64_t>(HEAD_DIM) * HEAD_DIM, stateElements) ||
        !CheckedMul(stateElements, sizeof(float), stateBytes64)) {
        std::fprintf(stderr, "输入或输出字节数超出 uint64_t 表示范围\n");
        return 1;
    }

    const size_t qkvBytes = static_cast<size_t>(qkvBytes64);
    const size_t logDecayBytes = static_cast<size_t>(logDecayBytes64);
    const size_t betaBytes = static_cast<size_t>(betaBytes64);
    const size_t stateBytes = static_cast<size_t>(stateBytes64);

    uint64_t workspaceBytes64 = 0;
    if (!GetKimiDeltaAttnLiteWorkspaceSize(batchSize, seqLen, workspaceBytes64)) {
        return 1;
    }
    const size_t workspaceBytes = static_cast<size_t>(workspaceBytes64);

    const std::string exeDir = GetExeDir();
    const std::string dataDir = exeDir + "/data/kdalite_v" + std::to_string(KDALITE_VERSION_ID);
    std::printf("kdalite: B=%u，S=%u，workspace=%zu bytes\n", batchSize, seqLen, workspaceBytes);
    if (RunCmd("rm -rf '" + dataDir + "'") != 0 || RunCmd("mkdir -p '" + dataDir + "'") != 0) {
        std::fprintf(stderr, "重建数据目录失败：%s\n", dataDir.c_str());
        return 1;
    }

    const std::string scriptArgs = "'" + dataDir + "' " + std::to_string(batchSize) + " " + std::to_string(seqLen) +
                                   " " + std::to_string(HEAD_DIM);
    if (RunCmd("python3 '" + exeDir + "/kimi_delta_attn_lite_gendata.py' " + scriptArgs) != 0) {
        std::fprintf(stderr, "生成数据失败，请确认 Python 依赖和构建目录内脚本完整\n");
        return 1;
    }

    std::vector<uint8_t> hostQ;
    std::vector<uint8_t> hostK;
    std::vector<uint8_t> hostV;
    std::vector<uint8_t> hostLogDecay;
    std::vector<uint8_t> hostBeta;
    if (!ReadBin(dataDir + "/q.bin", qkvBytes, hostQ) || !ReadBin(dataDir + "/k.bin", qkvBytes, hostK) ||
        !ReadBin(dataDir + "/v.bin", qkvBytes, hostV) ||
        !ReadBin(dataDir + "/log_decay.bin", logDecayBytes, hostLogDecay) ||
        !ReadBin(dataDir + "/beta.bin", betaBytes, hostBeta)) {
        return 1;
    }

    bool outputWriteSucceeded = true;
    {
        CHECK_ACL(aclInit(nullptr));
        AclFinalizeGuard aclGuard;
        uint32_t deviceCount = 0;
        CHECK_ACL(aclrtGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            std::fprintf(stderr, "未发现 ACL 设备\n");
            return 1;
        }
        constexpr int32_t deviceId = 0;
        CHECK_ACL(aclrtSetDevice(deviceId));
        AclDeviceGuard deviceGuard(deviceId);

        // StreamGuard 后声明, 先析构, 保证异常路径先等待在途 kernel,
        // 再由这些 unique_ptr 释放设备内存.
        std::unique_ptr<void, AclrtFreeDeleter> devQGuard;
        std::unique_ptr<void, AclrtFreeDeleter> devKGuard;
        std::unique_ptr<void, AclrtFreeDeleter> devVGuard;
        std::unique_ptr<void, AclrtFreeDeleter> devLogDecayGuard;
        std::unique_ptr<void, AclrtFreeDeleter> devBetaGuard;
        std::unique_ptr<void, AclrtFreeDeleter> devOGuard;
        std::unique_ptr<void, AclrtFreeDeleter> devFinalStateGuard;
        std::unique_ptr<void, AclrtFreeDeleter> devWorkspaceGuard;

        aclrtStream stream = nullptr;
        CHECK_ACL(aclrtCreateStream(&stream));
        AclStreamGuard streamGuard(stream);

        void* devQ = nullptr;
        void* devK = nullptr;
        void* devV = nullptr;
        void* devLogDecay = nullptr;
        void* devBeta = nullptr;
        void* devO = nullptr;
        void* devFinalState = nullptr;
        void* devWorkspace = nullptr;
        CHECK_ACL(aclrtMalloc(&devQ, qkvBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        devQGuard.reset(devQ);
        CHECK_ACL(aclrtMalloc(&devK, qkvBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        devKGuard.reset(devK);
        CHECK_ACL(aclrtMalloc(&devV, qkvBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        devVGuard.reset(devV);
        CHECK_ACL(aclrtMalloc(&devLogDecay, logDecayBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        devLogDecayGuard.reset(devLogDecay);
        CHECK_ACL(aclrtMalloc(&devBeta, betaBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        devBetaGuard.reset(devBeta);
        CHECK_ACL(aclrtMalloc(&devO, qkvBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        devOGuard.reset(devO);
        CHECK_ACL(aclrtMalloc(&devFinalState, stateBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        devFinalStateGuard.reset(devFinalState);
        CHECK_ACL(aclrtMalloc(&devWorkspace, workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        devWorkspaceGuard.reset(devWorkspace);

        CHECK_ACL(aclrtMemcpy(devQ, qkvBytes, hostQ.data(), qkvBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(devK, qkvBytes, hostK.data(), qkvBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(devV, qkvBytes, hostV.data(), qkvBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(
            aclrtMemcpy(devLogDecay, logDecayBytes, hostLogDecay.data(), logDecayBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(devBeta, betaBytes, hostBeta.data(), betaBytes, ACL_MEMCPY_HOST_TO_DEVICE));

        if (!KimiDeltaAttnLiteNPU(
                reinterpret_cast<uint8_t*>(devQ), reinterpret_cast<uint8_t*>(devK), reinterpret_cast<uint8_t*>(devV),
                reinterpret_cast<uint8_t*>(devLogDecay), reinterpret_cast<uint8_t*>(devBeta),
                reinterpret_cast<uint8_t*>(devO), reinterpret_cast<uint8_t*>(devFinalState),
                reinterpret_cast<uint8_t*>(devWorkspace), workspaceBytes64, batchSize, seqLen, requestedMixCoreNum,
                stream)) {
            return 1;
        }
        CHECK_ACL(aclrtSynchronizeStream(stream));
        streamGuard.MarkSynchronized();

        std::vector<uint8_t> hostFinalState(stateBytes);
        CHECK_ACL(aclrtMemcpy(hostFinalState.data(), stateBytes, devFinalState, stateBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        const std::string statePath = dataDir + "/npuout_final_state.bin";
        if (!WriteBin(statePath, hostFinalState.data(), hostFinalState.size())) {
            outputWriteSucceeded = false;
        } else {
            std::printf("kdalite: 已落盘 %s (%zu bytes)\n", statePath.c_str(), hostFinalState.size());
        }

        std::vector<uint8_t> hostO(qkvBytes);
        CHECK_ACL(aclrtMemcpy(hostO.data(), qkvBytes, devO, qkvBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        const std::string outputPath = dataDir + "/npuout_o.bin";
        if (!WriteBin(outputPath, hostO.data(), hostO.size())) {
            outputWriteSucceeded = false;
        } else {
            std::printf("kdalite: 已落盘 %s (%zu bytes)\n", outputPath.c_str(), hostO.size());
        }
    }

    if (!outputWriteSucceeded) {
        return 1;
    }
    if (dryRun) {
        std::printf("kdalite: --dry-run 已真实执行并同步 kernel、回读并落盘 NPU 输出，跳过 Golden 计算与结果比对。\n");
        return 0;
    }
    const int verifyStatus = RunCmd("python3 '" + exeDir + "/kimi_delta_attn_lite_verify.py' " + scriptArgs);
    if (verifyStatus != 0) {
        std::fprintf(stderr, "比对失败（kimi_delta_attn_lite_verify.py 退出码 %d）。详见上方报告。\n", verifyStatus);
        return 1;
    }
    return 0;
}
