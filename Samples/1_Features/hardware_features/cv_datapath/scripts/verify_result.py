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

"""Verify MatMul+ReLU output. Strategy aligned with matmul_story a16w16."""

import sys
import numpy as np

# Align with matmul_story:
# - NaN/Inf always fail
# - Severe point: rel > POINT_ERROR_TOL AND abs > RATIO_POINT_ERROR_TOL (avoid false fail on ~0 golden)
# - Soft mismatch ratio: fraction with abs > RATIO_POINT_ERROR_TOL must be <= ERROR_RATIO_TOL
# ERROR_RATIO_TOL < one base-tile / total so a fully wrong tile cannot pass.
POINT_ERROR_TOL = 1e-1
RATIO_POINT_ERROR_TOL = 1e-3
ERROR_RATIO_TOL = 5e-4

M = 8192
N = 8192
BASE_M = 256
BASE_N = 256
TILE_ERROR_RATIO_TOL = 1e-2


def _rel_diff(abs_diff, golden):
    abs_g = np.abs(golden)
    rel = np.empty_like(abs_diff)
    nonzero = abs_g > 0
    rel[nonzero] = abs_diff[nonzero] / abs_g[nonzero]
    zero_g = ~nonzero
    rel[zero_g] = np.where(abs_diff[zero_g] == 0, 0.0, np.inf)
    return rel


def _check_tiles(output_2d, golden_2d):
    """Fail if any base tile is largely wrong (missing write / wrong offset)."""
    m_tiles = M // BASE_M
    n_tiles = N // BASE_N
    for tm in range(m_tiles):
        for tn in range(n_tiles):
            o = output_2d[tm * BASE_M: (tm + 1) * BASE_M, tn * BASE_N: (tn + 1) * BASE_N]
            g = golden_2d[tm * BASE_M: (tm + 1) * BASE_M, tn * BASE_N: (tn + 1) * BASE_N]
            abs_diff = np.abs(o - g)
            non_finite = ~(np.isfinite(o) & np.isfinite(g) & np.isfinite(abs_diff))
            bad = (abs_diff > RATIO_POINT_ERROR_TOL) | non_finite
            tile_ratio = float(np.count_nonzero(bad)) / bad.size
            if tile_ratio > TILE_ERROR_RATIO_TOL:
                print(
                    f"[ERROR] tile ({tm},{tn}) mismatch ratio {tile_ratio:.6f} "
                    f"> {TILE_ERROR_RATIO_TOL}"
                )
                return False
    return True


def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=np.float32)
    golden = np.fromfile(golden_path, dtype=np.float32)
    expected = M * N
    if output.size != expected or golden.size != expected:
        print(
            f"[ERROR] size mismatch: output={output.size}, golden={golden.size}, expected={expected}"
        )
        return False

    abs_diff = np.abs(output - golden)
    non_finite = ~(np.isfinite(output) & np.isfinite(golden) & np.isfinite(abs_diff))
    if np.any(non_finite):
        print(f"[ERROR] non-finite values: {int(np.count_nonzero(non_finite))}")
        return False

    rel_diff = _rel_diff(abs_diff, golden)
    point_error_mask = (rel_diff > POINT_ERROR_TOL) & (abs_diff > RATIO_POINT_ERROR_TOL)
    ratio_error_mask = abs_diff > RATIO_POINT_ERROR_TOL
    point_error_count = int(np.count_nonzero(point_error_mask))
    error_count = int(np.count_nonzero(ratio_error_mask))
    error_ratio = error_count / expected

    bad_idx = np.flatnonzero(point_error_mask)
    for index in range(min(len(bad_idx), 100)):
        i = int(bad_idx[index])
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, abs: %-.6f, rel: %-.6f"
            % (i, golden[i], output[i], abs_diff[i], rel_diff[i])
        )

    print(f"max abs diff: {float(abs_diff.max()) if expected else 0.0}")
    print(f"point error count(rel>{POINT_ERROR_TOL} & abs>{RATIO_POINT_ERROR_TOL}): "
          f"{point_error_count}/{expected}")
    print(
        f"ratio error count(abs>{RATIO_POINT_ERROR_TOL}): {error_count}/{expected}, "
        f"error ratio: {error_ratio:.6f}"
    )

    if point_error_count != 0 or error_ratio > ERROR_RATIO_TOL:
        return False

    output_2d = output.reshape(M, N)
    golden_2d = golden.reshape(M, N)
    return _check_tiles(output_2d, golden_2d)


def main():
    if len(sys.argv) < 3:
        print("usage: verify_result.py <output.bin> <golden.bin>", file=sys.stderr)
        return 1
    if verify_result(sys.argv[1], sys.argv[2]):
        print("test pass!")
        return 0
    print("[ERROR] result error", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
