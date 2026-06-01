#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import sys

import numpy as np
import torch

POINT_ERROR_TOL = 1e-1
RATIO_POINT_ERROR_TOL = 1e-3
ERROR_RATIO_TOL = 1e-3
DATA_TYPE = np.float16


def load_group_m_list(group_num: int):
    group_list = np.fromfile("./input/input_groupList.bin", dtype=np.int64)
    if group_list.size != group_num:
        raise ValueError("input_groupList.bin size does not match group_num")
    if np.any(group_list < 0):
        raise ValueError("input_groupList.bin contains negative group size")
    return group_list.astype(np.int64).tolist()


def verify_result(group_m_list, m: int, n: int) -> bool:
    output = np.fromfile("./output/npu_out.bin", dtype=DATA_TYPE)
    golden = np.fromfile("./output/cpu_output.bin", dtype=DATA_TYPE)
    if output.size != golden.size:
        raise ValueError("npu output size != cpu output size")
    if output.size != m * n:
        raise ValueError(f"output element count {output.size} does not match expected size={m * n}")

    sum_group_m = sum(group_m_list)
    golden_cmp = torch.from_numpy(golden).view(torch.bfloat16).reshape(m, n)[:sum_group_m]
    npu_cmp = torch.from_numpy(output).view(torch.bfloat16).reshape(m, n)[:sum_group_m]
    print("\ncpu golden (sum(group_m_list) rows):\n", golden_cmp)
    print("npu output (sum(group_m_list) rows):\n", npu_cmp)

    golden_f32 = golden_cmp.to(torch.float32)
    npu_f32 = npu_cmp.to(torch.float32)
    abs_diff = torch.abs(golden_f32 - npu_f32)
    non_finite_mask = ~(torch.isfinite(golden_f32) & torch.isfinite(npu_f32) & torch.isfinite(abs_diff))
    abs_golden = torch.abs(golden_f32)
    rel_diff = torch.where(
        abs_golden > 0,
        abs_diff / abs_golden,
        torch.where(abs_diff == 0, torch.zeros_like(abs_diff), torch.full_like(abs_diff, float("inf"))),
    )
    point_error_mask = (rel_diff > POINT_ERROR_TOL) | non_finite_mask
    ratio_error_mask = (abs_diff > RATIO_POINT_ERROR_TOL) | non_finite_mask
    point_error_count = int(point_error_mask.sum().item())
    error_count = int(ratio_error_mask.sum().item())
    total_count = ratio_error_mask.numel()
    error_ratio = error_count / total_count if total_count else 0.0
    print(f"max abs diff: {abs_diff.max().item() if total_count else 0.0}")
    print(f"point error count(>{POINT_ERROR_TOL}): {point_error_count}/{total_count}")
    print(
        f"ratio error count(>{RATIO_POINT_ERROR_TOL}): {error_count}/{total_count}, "
        f"error ratio: {error_ratio:.6f}"
    )
    return point_error_count == 0 and error_ratio <= ERROR_RATIO_TOL


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 verify_result.py group_num m k n")
        sys.exit(1)
    try:
        group_num = int(sys.argv[1])
        m = int(sys.argv[2])
        n = int(sys.argv[4])
        group_m_list = load_group_m_list(group_num)
        if sum(group_m_list) > m:
            raise ValueError("sum(group_m_list) must be less than or equal to m")
        if not verify_result(group_m_list, m, n):
            raise ValueError("[ERROR] NPU results differ from CPU.")
        print("[PASS] NPU results are consistent with CPU.\n")
    except Exception as err:
        print(err)
        sys.exit(1)
