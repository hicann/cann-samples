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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "flash_attn_lite.h"

#define CHECK_ACL(call)                                                                                    \
    do {                                                                                                   \
        aclError err = (call);                                                                             \
        if (err != ACL_SUCCESS) {                                                                          \
            std::fprintf(stderr, "ACL 错误 %d，位置：%s:%d\n", static_cast<int>(err), __FILE__, __LINE__); \
            return 1;                                                                                      \
        }                                                                                                  \
    } while (0)

namespace {

constexpr uint32_t DEFAULT_B = 1;
constexpr uint32_t DEFAULT_S = 256; // S 必须是 128 的整数倍.

struct AclrtFreeDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

std::string GetExeDir()
{
    char buf[4096];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return ".";
    }
    buf[n] = '\0';
    std::string path(buf);
    size_t pos = path.find_last_of('/');
    return (pos == std::string::npos) ? "." : path.substr(0, pos);
}

int RunCmd(const std::string& cmd)
{
    std::printf("  $ %s\n", cmd.c_str());
    return std::system(cmd.c_str());
}

bool ReadBin(const std::string& path, std::vector<uint16_t>& out)
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "读取失败：%s\n", path.c_str());
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (bytes < 0 || bytes % static_cast<long>(sizeof(uint16_t)) != 0) {
        std::fclose(f);
        return false;
    }
    out.resize(static_cast<size_t>(bytes) / sizeof(uint16_t));
    size_t rd = std::fread(out.data(), 1, static_cast<size_t>(bytes), f);
    std::fclose(f);
    return rd == static_cast<size_t>(bytes);
}

bool WriteBin(const std::string& path, const std::vector<uint16_t>& in)
{
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "写入失败：%s\n", path.c_str());
        return false;
    }
    size_t wr = std::fwrite(in.data(), sizeof(uint16_t), in.size(), f);
    std::fclose(f);
    return wr == in.size();
}

void PrintUsage(const char* prog)
{
    std::fprintf(
        stderr,
        "用法：%s [--size <B> <S>] [--core-num <n>] [--dry-run]  "
        "(D 固定 128)\n"
        "  --core-num：指定正整数个 AIC，不能超过本卡 AIC 核数。\n"
        "  --dry-run：真实执行 kernel 并落盘输出，仅跳过 Golden "
        "与比对。\n",
        prog);
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

bool ParseArgs(int argc, char** argv, uint32_t& B, uint32_t& S, uint32_t& requestedAicCoreNum, bool& dryRun)
{
    B = DEFAULT_B;
    S = DEFAULT_S;
    requestedAicCoreNum = 0;
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
        if (arg == "--size" && !hasSize && i + 2 < argc && ParseU32(argv[i + 1], B) && ParseU32(argv[i + 2], S) &&
            B > 0 && S > 0) {
            hasSize = true;
            i += 3;
            continue;
        }
        if (arg == "--core-num" && requestedAicCoreNum == 0 && i + 1 < argc &&
            ParseU32(argv[i + 1], requestedAicCoreNum) && requestedAicCoreNum > 0) {
            i += 2;
            continue;
        }
        PrintUsage(argv[0]);
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    uint32_t B = 0;
    uint32_t S = 0;
    uint32_t requestedAicCoreNum = 0;
    bool dryRun = false;
    if (!ParseArgs(argc, argv, B, S, requestedAicCoreNum, dryRun)) {
        return 1;
    }
    constexpr uint32_t D = 128; // 输入固定为 BF16, D=128.
    const float softmaxScale = 1.0f / std::sqrt(static_cast<float>(D));

    const std::string exeDir = GetExeDir();
    std::printf("falite: exe 目录=%s\n", exeDir.c_str());

    const std::string dataDir = exeDir + "/data";
    std::printf("falite: 重建数据目录 %s\n", dataDir.c_str());
    RunCmd("rm -rf '" + dataDir + "'");
    RunCmd("mkdir -p '" + dataDir + "'");

    std::printf("falite: 生成输入数据\n");
    const std::string gendataArgs =
        "'" + dataDir + "' " + std::to_string(B) + " " + std::to_string(S) + " " + std::to_string(D);
    if (RunCmd("python3 '" + exeDir + "/flash_attn_lite_gendata.py' " + gendataArgs) != 0) {
        std::fprintf(stderr, "生成数据失败，请确认 python3+numpy 可用且脚本已随构建拷贝到 %s\n", exeDir.c_str());
        return 1;
    }

    // Kernel 直接计算 K x Qᵀ, host 不预转置 K.
    std::vector<uint16_t> hostQ;
    std::vector<uint16_t> hostV;
    std::vector<uint16_t> hostK;
    if (!ReadBin(dataDir + "/q.bin", hostQ) || !ReadBin(dataDir + "/v.bin", hostV) ||
        !ReadBin(dataDir + "/k.bin", hostK)) {
        return 1;
    }
    const uint64_t expectedElements = static_cast<uint64_t>(B) * S * D;
    if (hostQ.size() != expectedElements || hostK.size() != expectedElements || hostV.size() != expectedElements) {
        std::fprintf(
            stderr,
            "输入文件元素数与 Tiling 不一致：期望=%llu，Q=%zu K=%zu "
            "V=%zu\n",
            static_cast<unsigned long long>(expectedElements), hostQ.size(), hostK.size(), hostV.size());
        return 1;
    }
    const size_t bytes = hostQ.size() * sizeof(uint16_t);
    std::printf("falite: 读入 Q/K/V，每个 %zu bytes (%zu ele)\n", bytes, hostQ.size());

    CHECK_ACL(aclInit(nullptr));
    uint32_t deviceCount = 0;
    CHECK_ACL(aclrtGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::fprintf(stderr, "未发现 ACL 设备\n");
        aclFinalize();
        return 1;
    }
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void* devQ = nullptr;
    void* devK = nullptr;
    void* devV = nullptr;
    void* devO = nullptr;
    CHECK_ACL(aclrtMalloc(&devQ, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> devQGuard(devQ);
    CHECK_ACL(aclrtMalloc(&devK, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> devKGuard(devK);
    CHECK_ACL(aclrtMalloc(&devV, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> devVGuard(devV);
    CHECK_ACL(aclrtMalloc(&devO, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> devOGuard(devO);
    auto releaseDeviceMemory = [&]() {
        devQGuard.reset();
        devKGuard.reset();
        devVGuard.reset();
        devOGuard.reset();
    };
    CHECK_ACL(aclrtMemcpy(devQ, bytes, hostQ.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(devK, bytes, hostK.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(devV, bytes, hostV.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    const bool launched = FlashAttnLiteNPU(
        reinterpret_cast<uint8_t*>(devQ), reinterpret_cast<uint8_t*>(devK), reinterpret_cast<uint8_t*>(devV),
        reinterpret_cast<uint8_t*>(devO), B, S, softmaxScale, requestedAicCoreNum, stream);
    if (!launched) {
        releaseDeviceMemory();
        CHECK_ACL(aclrtDestroyStream(stream));
        CHECK_ACL(aclrtResetDevice(deviceId));
        CHECK_ACL(aclFinalize());
        return 1;
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<uint16_t> hostO(hostQ.size(), 0);
    CHECK_ACL(aclrtMemcpy(hostO.data(), bytes, devO, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    std::printf("falite: kernel 执行完成，回读 O (%zu ele)\n", hostO.size());

    releaseDeviceMemory();
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    // npuout_o.bin 保存行主序 raw BF16 数据, 供 _verify.py 比对.
    const std::string npuOutPath = dataDir + "/npuout_o.bin";
    if (!WriteBin(npuOutPath, hostO)) {
        std::fprintf(stderr, "落盘 %s 失败\n", npuOutPath.c_str());
        return 1;
    }
    std::printf("已落盘：%s (%zu ele)\n", npuOutPath.c_str(), hostO.size());

    if (dryRun) {
        std::printf(
            "falite: --dry-run 已真实执行并同步 kernel、回读并落盘 "
            "NPU 输出，跳过 Golden 计算与结果比对。\n");
        return 0;
    }

    // _verify.py 生成 golden_o.bin 并比对; 非 0 退出码表示失败.
    const std::string verifyArgs =
        "'" + dataDir + "' " + std::to_string(B) + " " + std::to_string(S) + " " + std::to_string(D);
    const int verifyStatus = RunCmd("python3 '" + exeDir + "/flash_attn_lite_verify.py' " + verifyArgs);
    if (verifyStatus != 0) {
        std::fprintf(stderr, "比对失败（_verify.py 退出码 %d）。详见上方报告。\n", verifyStatus);
        return 1;
    }
    return 0;
}
