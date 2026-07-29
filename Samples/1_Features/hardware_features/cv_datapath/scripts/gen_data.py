#!/usr/bin/python3
# coding=utf-8
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Generate MatMul+ReLU inputs and golden for cv_datapath."""

import os
import sys
import numpy as np

np.random.seed(9)

# Must match include/kernel_common.h (Cube params gelu-aligned; singleCore sized for 28-core SKU)
M = 8192
K = 8192
N = 8192
BASE_M = 256
BASE_N = 256
BASE_K = 64
SINGLE_CORE_M = 1024
SINGLE_CORE_N = 1024
SINGLE_CORE_K = 8192


def _check_tiling_divisible():
    if M % SINGLE_CORE_M != 0 or N % SINGLE_CORE_N != 0 or K % SINGLE_CORE_K != 0:
        raise ValueError(
            f"M/N/K must be multiples of singleCore: "
            f"M={M}, N={N}, K={K}, singleCore=({SINGLE_CORE_M},{SINGLE_CORE_N},{SINGLE_CORE_K})"
        )
    if SINGLE_CORE_M % BASE_M != 0 or SINGLE_CORE_N % BASE_N != 0 or SINGLE_CORE_K % BASE_K != 0:
        raise ValueError(
            f"singleCore must be multiples of base: "
            f"singleCore=({SINGLE_CORE_M},{SINGLE_CORE_N},{SINGLE_CORE_K}), "
            f"base=({BASE_M},{BASE_N},{BASE_K})"
        )


def gen_golden_data():
    _check_tiling_divisible()

    input_type = np.dtype("float16")
    output_type = np.dtype("float32")

    x1_gm = np.random.uniform(-1, 1, [M, K]).astype(input_type)
    x2_gm = np.random.uniform(-1, 1, [K, N]).astype(input_type)

    golden = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32)).astype(np.float32)
    golden = np.maximum(golden, 0.0).astype(output_type)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    x1_gm.astype(input_type).tofile("./input/x1_gm.bin")
    # Match matmul_gelu_high_performance: store B as [N,K] for IS_B_TRANSPOSE Cube path.
    x2_gm.transpose().astype(input_type).tofile("./input/x2_gm.bin")
    golden.astype(output_type).tofile("./output/golden.bin")
    print(
        f"Generated data: M={M}, K={K}, N={N}, "
        f"base=({BASE_M},{BASE_N},{BASE_K}), "
        f"singleCore=({SINGLE_CORE_M},{SINGLE_CORE_N},{SINGLE_CORE_K}), "
        f"B_layout=transposed[N,K], elements={M * N}"
    )


if __name__ == "__main__":
    try:
        gen_golden_data()
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
