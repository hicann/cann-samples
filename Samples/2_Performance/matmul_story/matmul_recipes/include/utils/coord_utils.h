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

#ifndef UTILS_COORD_UTILS_H
#define UTILS_COORD_UTILS_H

#include "kernel_utils/common_utils.h"
#include "quant_matmul_constant.h"

constexpr int IDX_M_BASE_NORM_CNT = 0;
constexpr int IDX_M_BASE_TAIL_MAIN = 1;
constexpr int IDX_N_BASE_NORM_CNT = 2;
constexpr int IDX_N_BASE_TAIL_MAIN = 3;

using TupleL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

// Convert scheduler tile coordinates into linear GM offsets.
//
// This helper sits between the scheduler and the compute pipeline:
// - the scheduler decides *which* logical tile a block should process;
// - `Coordinate` decides *where* that tile starts in GM for A/B/scale/bias/C.
//
// It also hides layout-dependent details:
// - Row-major vs. transposed inputs change whether the major stride is K or 1.
// - Tail load-balance corrections change the mapping for the last few tiles.
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
    __aicore__ inline AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> GetQuantOffset(
        int64_t mTileIdx, int64_t nTileIdx, int64_t mSplitOffset = 0, int64_t nSplitOffset = 0,
        const AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>& loadBalanceParam = {0u, 0u, 0u, 0u})
    {
        // Convert tile indices (and optional tail-split offsets) into element offsets in GM buffers.
        //
        // Offset tuple fields:
        //   0: A, 1: B, 2: scaleA, 3: scaleB, 4: bias, 5: C.
        int64_t mOffset = mTileIdx * l1M + mSplitOffset;
        int64_t nOffset = nTileIdx * l1N + nSplitOffset;
        if constexpr (enableLoadBalance) {
            // When tail blocks are merged/split, linear `tileIdx * base` mapping needs correction.
            //
            // Example:
            //   if the last M tiles are smaller than `l1M`, then the start offset of
            //   tile i is not simply `i * l1M` anymore. We subtract the accumulated
            //   shrinkage introduced by those smaller tail tiles.
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
        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> offset{0, 0, 0, 0, 0, 0};
        if constexpr (isTransA) {
            // Transposed A is indexed as A[K, M], so moving on M only advances by 1.
            Get<0>(offset) = mOffset;
        } else {
            // Non-transposed A is indexed as A[M, K], so moving on M skips an entire K row.
            Get<0>(offset) = mOffset * k;
        }
        if constexpr (isTransB) {
            // Transposed B is indexed as B[N, K].
            Get<1>(offset) = nOffset * k;
        } else {
            // Non-transposed B is indexed as B[K, N].
            Get<1>(offset) = nOffset;
        }

        // C is always addressed as a row-major [M, N] matrix in this sample.
        Get<5>(offset) = mOffset * n + nOffset; // 5: idx of y
        if constexpr (isTransA) {
            // scaleA follows the leading dimension of the logical M axis.
            Get<2>(offset) = mOffset * MXFP_MULTI_BASE_SIZE; // 2: idx of x1Scale
        } else {
            // Each M row owns CeilDiv(K, divisor) groups of scale values.
            Get<2>(offset) = mOffset * CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE; // 2: idx of x1Scale
        }
        if constexpr (isTransB) {
            // In the transposed case, moving on N advances over a full scale row.
            Get<3>(offset) = nOffset * CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE; // 3: idx of x2Scale
        } else {
            // In the non-transposed case, scaleB is already organized by the logical N axis.
            Get<3>(offset) = nOffset * MXFP_MULTI_BASE_SIZE; // 3: idx of x2Scale
        }
        // Bias is one fp32 value per output column.
        Get<4>(offset) = nOffset; // 4: idx of bias
        return offset;
    }

    int64_t m{0};
    int64_t n{0};
    int64_t k{0};
    int64_t l1M{0};
    int64_t l1N{0};
    int64_t l1K{0};
};
#endif