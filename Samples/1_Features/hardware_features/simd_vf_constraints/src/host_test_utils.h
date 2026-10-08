/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "acl/acl.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

inline void CheckAclStatus(aclError status, const char* operation)
{
    if (status != ACL_SUCCESS) {
        std::fprintf(stderr, "[HOST][ERROR] %s failed with ACL error %d\n", operation, static_cast<int>(status));
        std::abort();
    }
}

inline void CheckHostCondition(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "[HOST][ERROR] %s\n", message);
        std::abort();
    }
}

class AclRuntimeGuard {
public:
    explicit AclRuntimeGuard(int32_t deviceId = 0) : deviceId_(deviceId), stream_(nullptr)
    {
        CheckAclStatus(aclInit(nullptr), "aclInit");
        CheckAclStatus(aclrtSetDevice(deviceId_), "aclrtSetDevice");
        CheckAclStatus(aclrtCreateStream(&stream_), "aclrtCreateStream");
    }

    ~AclRuntimeGuard()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
        }
        aclrtResetDevice(deviceId_);
        aclFinalize();
    }

    aclrtStream Stream() const
    {
        return stream_;
    }

private:
    int32_t deviceId_;
    aclrtStream stream_;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t elemNum) : ptr_(nullptr), elemNum_(elemNum), bytes_(elemNum * sizeof(T))
    {
        CheckAclStatus(aclrtMalloc(reinterpret_cast<void**>(&ptr_), bytes_, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }

    ~DeviceBuffer()
    {
        if (ptr_ != nullptr) {
            aclrtFree(ptr_);
        }
    }

    T* Get() const
    {
        return ptr_;
    }

    size_t Bytes() const
    {
        return bytes_;
    }

    void CopyFromHost(const std::vector<T>& host)
    {
        CheckHostCondition(host.size() == elemNum_, "CopyFromHost requires the allocated element count");
        CheckAclStatus(
            aclrtMemcpy(ptr_, bytes_, host.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE), "host-to-device aclrtMemcpy");
    }

    void CopyToHost(std::vector<T>& host) const
    {
        CheckHostCondition(host.size() == elemNum_, "CopyToHost requires the allocated element count");
        CheckAclStatus(
            aclrtMemcpy(host.data(), bytes_, ptr_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST), "device-to-host aclrtMemcpy");
    }

private:
    T* ptr_;
    size_t elemNum_;
    size_t bytes_;
};

template <typename T>
void FillIndex(std::vector<T>& data)
{
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<T>(i);
    }
}

template <typename T>
void FillModulo(std::vector<T>& data, int mod)
{
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<T>(static_cast<int>(i) % mod);
    }
}

template <typename T>
void FillValue(std::vector<T>& data, T value)
{
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = value;
    }
}

template <typename T>
double ToReportValue(T value)
{
    return static_cast<double>(value);
}

template <typename T>
bool ValuesMatch(T actual, T expected, double tolerance = 0.001)
{
    if (std::is_floating_point<T>::value) {
        return std::fabs(static_cast<double>(actual) - static_cast<double>(expected)) <= tolerance;
    }
    return actual == expected;
}

template <typename T, typename ExpectedFn>
int CountMismatches(const char* caseName, const std::vector<T>& actual, ExpectedFn expectedFn, double tolerance = 0.001)
{
    int failed = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        T expected = expectedFn(static_cast<int>(i));
        if (!ValuesMatch(actual[i], expected, tolerance)) {
            if (failed < 4) {
                printf(
                    "[HOST][%s][MISMATCH] idx=%zu expected=%.6f actual=%.6f\n", caseName, i, ToReportValue(expected),
                    ToReportValue(actual[i]));
            }
            ++failed;
        }
    }
    return failed;
}

template <typename T, typename ExpectedFn>
int CountSampleMismatches(
    const std::vector<T>& actual, const std::vector<int>& samples, ExpectedFn expectedFn, double tolerance = 0.001)
{
    int failed = 0;
    for (int idx : samples) {
        CheckHostCondition(
            idx >= 0 && static_cast<size_t>(idx) < actual.size(),
            "CountSampleMismatches requires in-range sample indices");
        T expected = expectedFn(idx);
        if (!ValuesMatch(actual[idx], expected, tolerance)) {
            ++failed;
        }
    }
    return failed;
}

template <typename T, typename ExpectedFn>
void PrintCaseResult(
    const char* caseName, const char* status, int n, const std::vector<T>& actual, ExpectedFn expectedFn,
    const std::vector<int>& samples, int mismatches = 0)
{
    printf("[HOST][%s] status=%s N=%d mismatches=%d samples={", caseName, status, n, mismatches);
    for (size_t i = 0; i < samples.size(); ++i) {
        int idx = samples[i];
        CheckHostCondition(
            idx >= 0 && static_cast<size_t>(idx) < actual.size(), "PrintCaseResult requires in-range sample indices");
        T expected = expectedFn(idx);
        printf("%s%d:%.6f/%.6f", i == 0 ? "" : ",", idx, ToReportValue(actual[idx]), ToReportValue(expected));
    }
    printf("}\n");
}
