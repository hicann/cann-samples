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
 * \file coord_utils.h
 * \brief
 */

#pragma once

#include "kernel_utils/common_utils.h"
#include "quant_matmul_constant.h"

constexpr int IDX_M_BASE_NORM_CNT = 0;
constexpr int IDX_M_BASE_TAIL_MAIN = 1;
constexpr int IDX_N_BASE_NORM_CNT = 2;
constexpr int IDX_N_BASE_TAIL_MAIN = 3;

using TupleL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

template <bool isTransA_, bool isTransB_, CubeFormat layoutA_, CubeFormat layoutB_, CubeFormat layoutC_>
class Coordinate {
public:
    __aicore__ inline Coordinate(int64_t m, int64_t n, int64_t k, int64_t l1M, int64_t l1N, int64_t l1K)
        : m(m), n(n), k(k), l1M(l1M), l1N(l1N), l1K(l1K)
    {}

    static constexpr bool isTransA = isTransA_;
    static constexpr bool isTransB = isTransB_;
    static constexpr CubeFormat layoutB = layoutB_;

    template <bool enableLoadBalance = false>
    __aicore__ inline AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> GetQuantOffset(
        int64_t mTileIdx, int64_t nTileIdx, int64_t mSplitOffset = 0, int64_t nSplitOffset = 0,
        const AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>& loadBalanceParam = {0u, 0u, 0u, 0u})
    {
        int64_t mOffset = mTileIdx * l1M + mSplitOffset;
        int64_t nOffset = nTileIdx * l1N + nSplitOffset;
        if constexpr (enableLoadBalance) {
            if constexpr (!isTransA) {
                if (mTileIdx > Get<IDX_M_BASE_NORM_CNT>(loadBalanceParam)) {
                    mOffset -= (mTileIdx - Get<IDX_M_BASE_NORM_CNT>(loadBalanceParam)) *
                               (l1M - Get<IDX_M_BASE_TAIL_MAIN>(loadBalanceParam));
                }
            }
            if constexpr (isTransB) {
                if (nTileIdx > Get<IDX_N_BASE_NORM_CNT>(loadBalanceParam)) {
                    nOffset -= (nTileIdx - Get<IDX_N_BASE_NORM_CNT>(loadBalanceParam)) *
                               (l1N - Get<IDX_N_BASE_TAIL_MAIN>(loadBalanceParam));
                }
            }
        }
        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> offset{0, 0, 0, 0, 0};
        if constexpr (isTransA) {
            Get<IDX_A_OFFSET>(offset) = mOffset;
        } else {
            Get<IDX_A_OFFSET>(offset) = mOffset * k;
        }
        if constexpr (isTransB) {
            Get<IDX_B_OFFSET>(offset) = nOffset * k;
        } else {
            Get<IDX_B_OFFSET>(offset) = nOffset;
        }

        Get<IDX_C_OFFSET>(offset) = mOffset * n + nOffset; // 4: idx of y
        if constexpr (isTransA) {
            Get<IDX_X1SCALE_OFFSET>(offset) = mOffset * MXFP_MULTI_BASE_SIZE; // 2: idx of x1Scale
        } else {
            Get<IDX_X1SCALE_OFFSET>(offset) = mOffset * CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE; // 2: idx of x1Scale
        }
        if constexpr (isTransB) {
            Get<IDX_X2SCALE_OFFSET>(offset) = nOffset * CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE; // 3: idx of x2Scale
        } else {
            Get<IDX_X2SCALE_OFFSET>(offset) = nOffset * MXFP_MULTI_BASE_SIZE; // 3: idx of x2Scale
        }
        return offset;
    }

    int64_t m{0};
    int64_t n{0};
    int64_t k{0};
    int64_t l1M{0};
    int64_t l1N{0};
    int64_t l1K{0};
};
