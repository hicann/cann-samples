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
 * \file head_axis_common.h
 * \brief
 */

#pragma once

#include "broadcast_common.h"

static constexpr uint32_t HEAD_BRC_M = 17;
static constexpr uint32_t HEAD_BRC_N = VL_B32 * 10 + 13; // 653
static constexpr int HEAD_BRC_M_HOST = static_cast<int>(HEAD_BRC_M);
static constexpr int HEAD_BRC_N_HOST = static_cast<int>(HEAD_BRC_N);
static constexpr int HEAD_BRC_X_ELEMS = HEAD_BRC_M_HOST * HEAD_BRC_N_HOST;

struct BroadcastHeadAxisTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t nElemsAligned;
    uint32_t xElemsAligned;
    uint32_t aElemsAligned;
    uint32_t xBytesAligned;
    uint32_t aBytesAligned;
};

using HeadAxisKernelLauncher =
    void (*)(aclrtStream stream, float* x, float* a, float* y, BroadcastHeadAxisTilingData tiling);

inline float HeadAxisExpected(const std::vector<float>& x, const std::vector<float>& a, int idx)
{
    int n = idx % HEAD_BRC_N_HOST;
    return x[idx] + a[n];
}

inline int RunHeadAxisBroadcastCase(const char* caseName, HeadAxisKernelLauncher launcher)
{
    AclRuntimeGuard runtime;
    aclrtStream stream = runtime.Stream();
    std::vector<float> x(HEAD_BRC_X_ELEMS);
    std::vector<float> a(HEAD_BRC_N_HOST);
    std::vector<float> y(HEAD_BRC_X_ELEMS, 0.0f);
    FillIndex(x);
    FillIndex(a);

    DeviceBuffer<float> xDev(HEAD_BRC_X_ELEMS);
    DeviceBuffer<float> aDev(HEAD_BRC_N_HOST);
    DeviceBuffer<float> yDev(HEAD_BRC_X_ELEMS);
    xDev.CopyFromHost(x);
    aDev.CopyFromHost(a);
    yDev.CopyFromHost(y);

    uint32_t nElemsAligned = ElemAlignedForB32(HEAD_BRC_N);
    uint32_t xElemsAligned = HEAD_BRC_M * nElemsAligned;
    uint32_t aElemsAligned = nElemsAligned;
    uint32_t xBytesAligned = static_cast<uint32_t>(xElemsAligned * sizeof(float));
    uint32_t aBytesAligned = static_cast<uint32_t>(aElemsAligned * sizeof(float));
    BroadcastHeadAxisTilingData tiling{
        HEAD_BRC_M,
        HEAD_BRC_N,
        nElemsAligned,
        xElemsAligned,
        aElemsAligned,
        xBytesAligned,
        aBytesAligned,
    };
    launcher(stream, xDev.Get(), aDev.Get(), yDev.Get(), tiling);
    aclrtSynchronizeStream(stream);
    yDev.CopyToHost(y);

    auto expected = [&](int idx) -> float {
        return HeadAxisExpected(x, a, idx);
    };
    int failed = CountMismatches(caseName, y, expected);
    PrintCaseResult(caseName, failed == 0 ? "OK" : "MISMATCH", HEAD_BRC_X_ELEMS, y, expected,
        {0, HEAD_BRC_N_HOST, HEAD_BRC_X_ELEMS - 1}, failed);
    assert(failed == 0);
    return 0;
}
