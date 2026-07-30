#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# Generate input and golden files for simt_histogram_story.
#
# Shape / config:
# - x: [INPUT_ELEMS] float32
# - min: scalar float32
# - max: scalar float32
# - y (golden): [BINS] int32 — histogram counts
import argparse
import os

import numpy as np


INPUT_ELEMS = 1_000_000
BINS = 100


def make_uniform_data(rng):
    """Generate uniformly distributed input data in [min_val, max_val]."""
    min_val = np.float32(-10.0)
    max_val = np.float32(10.0)
    x = rng.uniform(min_val, max_val, size=INPUT_ELEMS).astype(np.float32)
    return x, np.array([min_val], dtype=np.float32), np.array([max_val], dtype=np.float32)


def compute_histogram(x, min_val, max_val, bins):
    """Compute histogram golden using float32 arithmetic, matching the C++ kernel.

    The C++ kernel formula:
        idx = (int32_t)((float)(val - minVal) * bins / minMaxRange)

    """
    golden = np.zeros(bins, dtype=np.int32)
    min_val = np.float32(min_val)
    max_val = np.float32(max_val)

    if min_val == max_val:
        min_val = np.float32(min_val - np.float32(1.0))
        max_val = np.float32(max_val + np.float32(1.0))

    min_max_range = np.float32(max_val - min_val)
    bins_f32 = np.float32(bins)

    for val in x.flat:
        v = np.float32(val)
        if v >= min_val and v <= max_val:
            idx = int(np.float32(np.float32(v - min_val) * bins_f32) / min_max_range)
            if idx == bins:
                idx = bins - 1
            golden[idx] += 1
    return golden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".", help="directory to write input/ and output/")
    parser.add_argument("--seed", type=int, default=42, help="numpy random seed (default: 42)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x, min_val, max_val = make_uniform_data(rng)
    golden = compute_histogram(x, min_val[0], max_val[0], BINS)

    input_dir = os.path.join(args.output, "input")
    golden_dir = os.path.join(args.output, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    x.astype(np.float32).tofile(os.path.join(input_dir, "x.bin"))
    min_val.astype(np.float32).tofile(os.path.join(input_dir, "min.bin"))
    max_val.astype(np.float32).tofile(os.path.join(input_dir, "max.bin"))
    golden.astype(np.int32).tofile(os.path.join(golden_dir, "golden.bin"))

    print(f"Generated histogram data: {INPUT_ELEMS} elements, {BINS} bins")
    print(f"  input/x.bin, input/min.bin, input/max.bin → output/golden.bin")


if __name__ == "__main__":
    main()
