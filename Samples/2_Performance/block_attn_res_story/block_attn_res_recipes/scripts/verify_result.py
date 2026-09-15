#!/usr/bin/env python3
# coding=utf-8

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import os
import sys

import numpy as np


def bf16_to_fp32(value: np.ndarray) -> np.ndarray:
    return (value.astype(np.uint32) << np.uint32(16)).view(np.float32)


# Use the operator ST CSV isclose policy by default. stat_rel_err remains optional.
# TTK 151886a704e2 CSV pairs are (rtol, ptol), NOT (rtol, atol).
DEFAULT_COMPARE = "isclose"
STAT_THRESHOLDS = {"float32": 2**-13, "bfloat16": 2**-7}
# Source: ops-transformer/attention/block_attn_res_{prepare,update}/tests/st/arch35.
# Each entry is (rtol, allowed mismatch fraction, absolute_precision).
CLOSE_TOLERANCES = {"prepare": (0.05, 0.05, 0.05), "update": (0.001, 0.001, 0.0)}


def compare(name: str, actual: np.ndarray, expected: np.ndarray,
            dtype: str = "float32", method: str = DEFAULT_COMPARE) -> bool:
    if actual.size != expected.size:
        print(f"{name}: FAILED, size {actual.size} != {expected.size}")
        return False
    if actual.size == 0:
        print(f"{name}: PASSED, empty outputs")
        return True
    # TTK promotes BF16 to FP32, keeping FP32 arithmetic for these outputs.
    actual, expected = actual.astype(np.float32), expected.astype(np.float32)
    finite = np.isfinite(actual) & np.isfinite(expected)
    same_special = ((np.isnan(actual) & np.isnan(expected)) |
                    (np.isinf(actual) & np.isinf(expected) & (np.sign(actual) == np.sign(expected))))
    special_mismatch = ~finite & ~same_special
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        error = np.abs(actual - expected)
        relative = error / (np.abs(expected) + 1e-7)
    mere = float(np.mean(relative[finite])) if np.any(finite) else 0.0
    mare = float(np.max(relative[finite])) if np.any(finite) else 0.0
    if method == "stat_rel_err":
        threshold = STAT_THRESHOLDS[dtype]
        passed = not np.any(special_mismatch) and mere < threshold and mare < 10 * threshold
        bad = np.flatnonzero(special_mismatch | (finite & (relative >= 10 * threshold)))
        limits = f"mean_rel<{threshold:g}, max_rel<{10 * threshold:g}"
    elif method == "isclose":
        rtol, ptol, atol = CLOSE_TOLERANCES[name.split("/")[0]]
        with np.errstate(invalid="ignore", over="ignore"):
            close = np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
        bad = np.flatnonzero(~close)
        # Match TTK's precision calculation, including its boundary behavior.
        precision = (actual.size - bad.size) / actual.size
        passed = (1 - precision) <= ptol
        limits = f"rtol={rtol:g}, atol={atol:g}, ptol={ptol:g}"
    else:
        raise ValueError(f"Unsupported comparison: {method}")
    max_abs = float(np.max(error[finite])) if np.any(finite) else 0.0
    print(f"{name}: {'PASSED' if passed else 'FAILED'}, standard={method}, "
          f"max_abs_error={max_abs:.6g}, mean_rel_error={mere:.6g}, max_rel_error={mare:.6g}, "
          f"outliers={bad.size}/{actual.size} ({bad.size / actual.size:.2%}), "
          f"nonfinite_mismatches={np.count_nonzero(special_mismatch)}, {limits}")
    # ULP is diagnostic only; it must not impose a different acceptance standard.
    if dtype == "bfloat16" and np.any(finite):
        def ordered(value):
            bits = value.view(np.uint32) >> np.uint32(16)
            magnitude = (bits & 0x7FFF).astype(np.int32)
            return np.where((bits & 0x8000) != 0, -magnitude, magnitude)
        ulps = np.abs(ordered(actual[finite]) - ordered(expected[finite]))
        print(f"  max_bf16_ulp={np.max(ulps)} (diagnostic only)")
    if not passed:
        # A mean-error failure can occur without any max-error outlier.
        worst = np.flatnonzero(finite)[np.argsort(-relative[finite])]
        indices = np.unique(np.concatenate((bad[:10], worst[:10])))[:10]
        for index in indices:
            print(f"  [{index}] expected={expected[index]:.8g}, actual={actual[index]:.8g}")
    return bool(passed)


def load(root: str, name: str, dtype: np.dtype) -> np.ndarray:
    return np.fromfile(os.path.join(root, "output", name), dtype=dtype)


def verify_prepare(root: str, method: str = DEFAULT_COMPARE) -> bool:
    return all([
        compare("prepare/numerator", load(root, "npu_prepare_numerator.bin", np.float32),
                load(root, "golden_prepare_numerator.bin", np.float32), method=method),
        compare("prepare/logit_max", load(root, "npu_prepare_logit_max.bin", np.float32),
                load(root, "golden_prepare_logit_max.bin", np.float32), method=method),
        compare("prepare/exp_sum", load(root, "npu_prepare_exp_sum.bin", np.float32),
                load(root, "golden_prepare_exp_sum.bin", np.float32), method=method),
    ])


def verify_update(root: str, method: str = DEFAULT_COMPARE) -> bool:
    return all([
        compare("update/partial", load(root, "npu_update_partial.bin", np.float32),
                load(root, "golden_update_partial.bin", np.float32), method=method),
        compare("update/h", bf16_to_fp32(load(root, "npu_update_h.bin", np.uint16)),
                bf16_to_fp32(load(root, "golden_update_h.bin", np.uint16)), dtype="bfloat16", method=method),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--mode", choices=("prepare", "update", "e2e"), required=True)
    parser.add_argument("--t", type=int, required=True)
    parser.add_argument("--s", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--compare", choices=("stat_rel_err", "isclose"), default=DEFAULT_COMPARE)
    args = parser.parse_args()
    ok = True
    if args.mode in ("prepare", "e2e"):
        ok = verify_prepare(args.root, args.compare) and ok
    if args.mode in ("update", "e2e"):
        ok = verify_update(args.root, args.compare) and ok
    print(f"block_attn_res_{args.mode}: {'PASSED' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
