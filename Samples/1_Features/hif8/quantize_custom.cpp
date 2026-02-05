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
 * \file quantize_custom.cpp
 * \brief
 */


#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "data_utils.h"

constexpr uint32_t TOTAL_LENGTH = 2048;

class KernelQuantize {
public:
    __aicore__ inline KernelQuantize() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y)
    {
        xGm.SetGlobalBuffer((__gm__ float *)x, TOTAL_LENGTH);
        scaleGm.SetGlobalBuffer((__gm__ float *)scale, TOTAL_LENGTH);
        offsetGm.SetGlobalBuffer((__gm__ float *)offset, TOTAL_LENGTH);
        yGm.SetGlobalBuffer((__gm__ hifloat8_t *)y, TOTAL_LENGTH);

        pipe.InitBuffer(inQueueX, 1, TOTAL_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueScale, 1, TOTAL_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueOffset, 1, TOTAL_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, TOTAL_LENGTH * sizeof(hifloat8_t));
        pipe.InitBuffer(tmpCalc, TOTAL_LENGTH * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> scaleLocal = inQueueScale.AllocTensor<float>();
        AscendC::LocalTensor<float> offsetLocal = inQueueOffset.AllocTensor<float>();

        AscendC::DataCopy(xLocal, xGm, TOTAL_LENGTH);
        AscendC::DataCopy(scaleLocal, scaleGm, TOTAL_LENGTH);
        AscendC::DataCopy(offsetLocal, offsetGm, TOTAL_LENGTH);

        inQueueX.EnQue(xLocal);
        inQueueScale.EnQue(scaleLocal);
        inQueueOffset.EnQue(offsetLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> scaleLocal = inQueueScale.DeQue<float>();
        AscendC::LocalTensor<float> offsetLocal = inQueueOffset.DeQue<float>();
        AscendC::LocalTensor<hifloat8_t> yLocal = outQueueY.AllocTensor<hifloat8_t>();

        AscendC::LocalTensor<float> tmpLocal = tmpCalc.Get<float>();

        AscendC::Div(tmpLocal, xLocal, scaleLocal, TOTAL_LENGTH);
        AscendC::Add(tmpLocal, tmpLocal, offsetLocal, TOTAL_LENGTH);
        AscendC::Cast(yLocal, tmpLocal, AscendC::RoundMode::CAST_ROUND, TOTAL_LENGTH);

        outQueueY.EnQue<hifloat8_t>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueScale.FreeTensor(scaleLocal);
        inQueueOffset.FreeTensor(offsetLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<hifloat8_t> yLocal = outQueueY.DeQue<hifloat8_t>();
        AscendC::DataCopy(yGm, yLocal, TOTAL_LENGTH);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueScale;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueOffset;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpCalc;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> scaleGm;
    AscendC::GlobalTensor<float> offsetGm;
    AscendC::GlobalTensor<hifloat8_t> yGm;
};
    
__global__ __aicore__ void quantize_custom(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelQuantize op;
    op.Init(x, scale, offset, y);
    op.Process();
}

int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = 1;
    size_t inputByteSize = static_cast<size_t>(1) * 2048 * sizeof(uint32_t);
    size_t scaleByteSize = static_cast<size_t>(1) * 2048 * sizeof(uint32_t);
    size_t offsetByteSize = static_cast<size_t>(1) * 2048 * sizeof(uint32_t);
    size_t outputByteSize = static_cast<size_t>(1) * 2048 * sizeof(uint8_t);
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    uint8_t *xHost, *scaleHost, *offsetHost, *yHost;
    uint8_t *xDevice, *scaleDevice, *offsetDevice, *yDevice;

    aclrtMallocHost((void **)(&xHost), inputByteSize);
    aclrtMallocHost((void **)(&scaleHost), scaleByteSize);
    aclrtMallocHost((void **)(&offsetHost), offsetByteSize);
    aclrtMallocHost((void **)(&yHost), outputByteSize);

    aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scaleDevice, scaleByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&offsetDevice, offsetByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
    ReadFile("./input/input_scale.bin", scaleByteSize, scaleHost, inputByteSize);
    ReadFile("./input/input_offset.bin", offsetByteSize, offsetHost, inputByteSize);

    aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(scaleDevice, inputByteSize, scaleHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(offsetDevice, inputByteSize, offsetHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    quantize_custom<<<blockDim, nullptr, stream>>>(xDevice, scaleDevice, offsetDevice, yDevice);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(yHost, outputByteSize, yDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_y.bin", yHost, outputByteSize);

    aclrtFree(xDevice);
    aclrtFree(scaleDevice);
    aclrtFree(offsetDevice);
    aclrtFree(yDevice);
    aclrtFreeHost(xHost);
    aclrtFreeHost(scaleHost);
    aclrtFreeHost(offsetHost);
    aclrtFreeHost(yHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}