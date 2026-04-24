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
from typing import List

import numpy as np
import torch

POINT_ERROR_TOL = 1e-1
RATIO_POINT_ERROR_TOL = 1e-3
ERROR_RATIO_TOL = 1e-3
DATA_TYPE = np.uint16


def parse_group_m_list(arg: str) -> List[int]:
    values = []
    for item in arg.split(","):
        item = item.strip()
        if not item:
            raise ValueError("group_m_list contains an empty item")
        value = int(item)
        if value < 0:
            raise ValueError("Each group M value must be greater than or equal to 0")
        values.append(value)
    if not values:
        raise ValueError("group_m_list must not be empty")
    return values


def load_group_m_list(group_num: int) -> List[int]:
    group_list = np.fromfile("./input/input_groupList.bin", dtype=np.int64)
    if group_list.size != group_num:
        raise ValueError("input_groupList.bin size does not match group_num")
    if np.any(group_list < 0):
        raise ValueError("input_groupList.bin contains negative group size")
    return group_list.astype(np.int64).tolist()


def compute_error_metrics(golden_f32: torch.Tensor, npu_f32: torch.Tensor):
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
    return abs_diff, rel_diff, point_error_mask, ratio_error_mask


def print_point_error_details(
    point_error_mask: torch.Tensor,
    golden_f32: torch.Tensor,
    npu_f32: torch.Tensor,
    abs_diff: torch.Tensor,
    rel_diff: torch.Tensor,
) -> None:
    point_error_count = int(point_error_mask.sum().item())
    if point_error_count == 0:
        return
    point_error_indices = torch.nonzero(point_error_mask, as_tuple=False)
    print(f"point error details(rel diff > {POINT_ERROR_TOL} or non-finite):")
    for idx in point_error_indices:
        row = int(idx[0].item())
        col = int(idx[1].item())
        print(
            f"  (row={row}, col={col}) "
            f"golden={float(golden_f32[row, col].item())}, "
            f"npu={float(npu_f32[row, col].item())}, "
            f"abs_diff={float(abs_diff[row, col].item())}, "
            f"rel_diff={float(rel_diff[row, col].item())}"
        )


def verify_result(group_m_list: List[int], m: int, n: int):
    sum_group_m = sum(group_m_list)
    if sum_group_m > m:
        raise ValueError("sum(group_m_list) must be less than or equal to m")
    output = np.fromfile("./output/npu_out.bin", dtype=DATA_TYPE)
    golden = np.fromfile("./output/cpu_output.bin", dtype=DATA_TYPE)
    if output.size != golden.size:
        raise ValueError("npu output size != cpu output size")
    if output.size != m * n:
        raise ValueError(f"output element count {output.size} does not match m*n={m * n}")

    golden_cmp = torch.from_numpy(golden).view(torch.bfloat16).reshape(m, n)[:sum_group_m]
    npu_cmp = torch.from_numpy(output).view(torch.bfloat16).reshape(m, n)[:sum_group_m]
    print("\ncpu golden (sum(group_m_list) rows):\n", golden_cmp)
    print("npu output (sum(group_m_list) rows):\n", npu_cmp)
    golden_cmp_f32 = golden_cmp.to(torch.float32)
    npu_cmp_f32 = npu_cmp.to(torch.float32)
    abs_diff, rel_diff, point_error_mask, ratio_error_mask = compute_error_metrics(golden_cmp_f32, npu_cmp_f32)
    point_error_count = int(point_error_mask.sum().item())
    error_count = int(ratio_error_mask.sum().item())
    total_count = ratio_error_mask.numel()
    error_ratio = error_count / total_count if total_count else 0.0
    print(f"max abs diff: {abs_diff.max().item() if total_count else 0.0}")
    print(f"point error count(>{POINT_ERROR_TOL}): {point_error_count}/{total_count}")
    print_point_error_details(point_error_mask, golden_cmp_f32, npu_cmp_f32, abs_diff, rel_diff)
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
        k = int(sys.argv[3])
        n = int(sys.argv[4])
        if group_num <= 0:
            raise ValueError("group_num must be greater than 0")
        if m < 0:
            raise ValueError("m must be greater than or equal to 0")
        if k <= 0:
            raise ValueError("k must be greater than 0")
        if n <= 0:
            raise ValueError("n must be greater than 0")
        group_m_list = load_group_m_list(group_num)
        res = verify_result(group_m_list, m, n)
        if not res:
            raise ValueError(
                f"[ERROR] NPU results differ from CPU. "
                f"Single-point relative error (abs_diff/abs(golden)) must be <= {POINT_ERROR_TOL}, "
                f"and the ratio of points with absolute error > {RATIO_POINT_ERROR_TOL} "
                f"must be <= {ERROR_RATIO_TOL}.\n"
            )
        print("[PASS] NPU results are consistent with CPU.\n")
    except Exception as e:
        print(e)
        sys.exit(1)
