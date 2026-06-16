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
 * \file tail_axis_common.h
 * \brief
 */

#pragma once

#include "broadcast_common.h"

static constexpr uint32_t TAIL_BRC_M = 17;
static constexpr uint32_t TAIL_BRC_N = VL_B32 * 10 + 13; // 653
static constexpr int TAIL_BRC_M_HOST = static_cast<int>(TAIL_BRC_M);
static constexpr int TAIL_BRC_N_HOST = static_cast<int>(TAIL_BRC_N);
static constexpr int TAIL_BRC_X_ELEMS = TAIL_BRC_M_HOST * TAIL_BRC_N_HOST;

struct BroadcastTailAxisTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t nElemsAligned;
    uint32_t xElemsAligned;
    uint32_t aElemsAligned;
    uint32_t xBytesAligned;
    uint32_t aBytesAligned;
};

using TailAxisKernelLauncher = void (*)(aclrtStream stream, float* x, float* a, float* y,
    BroadcastTailAxisTilingData tiling);

inline float TailAxisExpected(const std::vector<float>& x, const std::vector<float>& a, int idx)
{
    int m = idx / TAIL_BRC_N_HOST;
    return x[idx] + a[m];
}

inline int RunTailAxisBroadcastCase(const char* caseName, TailAxisKernelLauncher launcher)
{
    AclRuntimeGuard runtime;
    aclrtStream stream = runtime.Stream();
    std::vector<float> x(TAIL_BRC_X_ELEMS);
    std::vector<float> a(TAIL_BRC_M_HOST);
    std::vector<float> y(TAIL_BRC_X_ELEMS, 0.0f);
    FillIndex(x);
    FillIndex(a);

    DeviceBuffer<float> xDev(TAIL_BRC_X_ELEMS);
    DeviceBuffer<float> aDev(TAIL_BRC_M_HOST);
    DeviceBuffer<float> yDev(TAIL_BRC_X_ELEMS);
    xDev.CopyFromHost(x);
    aDev.CopyFromHost(a);
    yDev.CopyFromHost(y);

    uint32_t nElemsAligned = ElemAlignedForB32(TAIL_BRC_N);
    uint32_t xElemsAligned = TAIL_BRC_M * nElemsAligned;
    uint32_t aElemsAligned = ElemAlignedForB32(TAIL_BRC_M);
    uint32_t xBytesAligned = static_cast<uint32_t>(xElemsAligned * sizeof(float));
    uint32_t aBytesAligned = static_cast<uint32_t>(aElemsAligned * sizeof(float));
    BroadcastTailAxisTilingData tiling{
        TAIL_BRC_M,
        TAIL_BRC_N,
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
        return TailAxisExpected(x, a, idx);
    };
    int failed = CountMismatches(caseName, y, expected);
    PrintCaseResult(caseName, failed == 0 ? "OK" : "MISMATCH", TAIL_BRC_X_ELEMS, y, expected,
        {0, TAIL_BRC_N_HOST, TAIL_BRC_X_ELEMS - 1}, failed);
    assert(failed == 0);
    return 0;
}
