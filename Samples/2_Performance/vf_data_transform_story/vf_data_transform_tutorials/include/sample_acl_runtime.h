/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VF_DATA_TRANSFORM_STORY_SAMPLE_ACL_RUNTIME_H
#define VF_DATA_TRANSFORM_STORY_SAMPLE_ACL_RUNTIME_H

#include "acl/acl.h"

#include <cstdint>
#include <iostream>
#include <string>

inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0U) {
        return value;
    }
    return (value + alignment - 1U) / alignment * alignment;
}

inline bool CheckAcl(aclError error, const char* operation)
{
    if (error == ACL_SUCCESS) {
        return true;
    }
    std::cerr << "[ACL][ERROR] " << operation << " failed with " << error << '\n';
    return false;
}

class AclRuntime {
public:
    bool Init()
    {
        if (!CheckAcl(aclInit(nullptr), "aclInit")) {
            return false;
        }
        initialized_ = true;
        if (!CheckAcl(aclrtSetDevice(deviceId_), "aclrtSetDevice")) {
            return false;
        }
        deviceSet_ = true;
        if (!CheckAcl(aclrtCreateStream(&stream_), "aclrtCreateStream")) {
            return false;
        }
        return true;
    }

    ~AclRuntime()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
        }
        if (initialized_) {
            aclFinalize();
        }
    }

    aclrtStream Stream() const
    {
        return stream_;
    }

private:
    int32_t deviceId_{0};
    aclrtStream stream_{nullptr};
    bool initialized_{false};
    bool deviceSet_{false};
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;

    ~DeviceBuffer()
    {
        if (data_ != nullptr) {
            aclrtFree(data_);
        }
    }

    bool Allocate(size_t bytes)
    {
        bytes_ = bytes;
        return CheckAcl(aclrtMalloc(&data_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }

    bool CopyFromHost(const void* source) const
    {
        return CheckAcl(aclrtMemcpy(data_, bytes_, source, bytes_, ACL_MEMCPY_HOST_TO_DEVICE), "H2D aclrtMemcpy");
    }

    bool CopyToHost(void* destination) const
    {
        return CheckAcl(aclrtMemcpy(destination, bytes_, data_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST), "D2H aclrtMemcpy");
    }

    uint8_t* Get() const
    {
        return static_cast<uint8_t*>(data_);
    }

private:
    void* data_{nullptr};
    size_t bytes_{0};
};

struct CaseSpec {
    const char* name;
    uint32_t m;
    uint32_t n;
};

inline bool IsSelected(const std::string& selected, const char* caseName)
{
    return selected == "all" || selected == caseName;
}

#endif // VF_DATA_TRANSFORM_STORY_SAMPLE_ACL_RUNTIME_H
