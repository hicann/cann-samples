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

import argparse
import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

ERROR_TOL = 1e-3
DATA_TYPE = np.uint16
DEFAULT_MAX_DETAIL_ROWS = 50
DATA_TYPE_BYTES = np.dtype(DATA_TYPE).itemsize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify NPU output against CPU golden for weight_quant_grouped_matmul_mxfp8fp4."
    )
    parser.add_argument("group_num", type=int, help="Group number.")
    parser.add_argument("m", type=int, help="Rows in output tensor.")
    parser.add_argument("k", type=int, help="K size (must be multiple of 64).")
    parser.add_argument("n", type=int, help="Cols in output tensor.")
    parser.add_argument(
        "--max-detail-rows",
        type=int,
        default=DEFAULT_MAX_DETAIL_ROWS,
        help=(
            "Maximum mismatch rows to print in table. "
            "Use -1 to print all mismatch rows. Default: %(default)s."
        ),
    )
    return parser.parse_args()


def validate_args(group_num: int, m: int, k: int, n: int, max_detail_rows: int) -> None:
    if group_num <= 0:
        raise ValueError("group_num must be greater than 0")
    if m < 0:
        raise ValueError("m must be greater than or equal to 0")
    if k <= 0 or k % 64 != 0:
        raise ValueError("k must be a positive multiple of 64")
    if n <= 0:
        raise ValueError("n must be greater than 0")
    if max_detail_rows < -1:
        raise ValueError("max_detail_rows must be -1 or >= 0")


def load_group_m_list(group_num: int) -> np.ndarray:
    group_list = np.fromfile("./input/input_groupList.bin", dtype=np.int64)
    if group_list.size != group_num:
        raise ValueError("input_groupList.bin size does not match group_num")
    if np.any(group_list < 0):
        raise ValueError("input_groupList.bin contains negative group size")
    return group_list.astype(np.int64, copy=False)


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.7g}"


def bf16_u16_to_fp32(values_u16: np.ndarray) -> np.ndarray:
    return (values_u16.astype(np.uint32) << 16).view(np.float32)


def compress_indices(indices: np.ndarray) -> List[Tuple[int, int]]:
    if indices.size == 0:
        return []
    ranges: List[Tuple[int, int]] = []
    start = int(indices[0])
    prev = start
    for idx in indices[1:]:
        curr = int(idx)
        if curr == prev + 1:
            prev = curr
            continue
        ranges.append((start, prev))
        start = curr
        prev = curr
    ranges.append((start, prev))
    return ranges


def render_index_ranges(ranges: Sequence[Tuple[int, int]], wrap: int = 120) -> str:
    if not ranges:
        return "[]"
    tokens: List[str] = []
    for start, end in ranges:
        tokens.append(str(start) if start == end else f"{start}-{end}")
    lines: List[str] = []
    current = ""
    for token in tokens:
        next_chunk = token if not current else f"{current}, {token}"
        if len(next_chunk) > wrap:
            lines.append(current)
            current = token
        else:
            current = next_chunk
    if current:
        lines.append(current)
    return "\n".join(lines)


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "(no rows)"
    widths = [len(head) for head in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def _row(cells: Sequence[str]) -> str:
        return "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells))) + " |"

    parts: List[str] = [_line(), _row(headers), _line()]
    parts.extend(_row(row) for row in rows)
    parts.append(_line())
    return "\n".join(parts)


def verify_result(group_m_list: np.ndarray, m: int, n: int, max_detail_rows: int) -> bool:
    sum_group_m = int(group_m_list.sum(dtype=np.int64))
    if sum_group_m > m:
        raise ValueError("sum(group_m_list) must be less than or equal to m")

    total_elem_count = m * n
    compare_elem_count = sum_group_m * n
    expected_bytes = total_elem_count * DATA_TYPE_BYTES
    output_path = Path("./output/output_npu.bin")
    golden_path = Path("./output/output_cpu.bin")

    output_bytes = output_path.stat().st_size
    golden_bytes = golden_path.stat().st_size
    if output_bytes != expected_bytes:
        raise ValueError(f"output_npu.bin size mismatch: got {output_bytes}, expected {expected_bytes}")
    if golden_bytes != expected_bytes:
        raise ValueError(f"output_cpu.bin size mismatch: got {golden_bytes}, expected {expected_bytes}")

    output = np.fromfile(output_path, dtype=DATA_TYPE, count=compare_elem_count)
    golden = np.fromfile(golden_path, dtype=DATA_TYPE, count=compare_elem_count)

    if output.size != golden.size:
        raise ValueError("npu output size != cpu output size")
    if output.size != compare_elem_count:
        raise ValueError(
            f"output element count {output.size} does not match compare rows*n={compare_elem_count}"
        )

    # Fast path: exact bitwise equal elements need no floating conversion.
    candidate_idx = np.flatnonzero(output != golden)
    if candidate_idx.size == 0:
        print(
            f"[PASS] NPU results are consistent with CPU. "
            f"(checked_rows={sum_group_m}, checked_elements={compare_elem_count}, mismatch_count=0)"
        )
        return True

    output_candidate = output[candidate_idx]
    golden_candidate = golden[candidate_idx]
    output_candidate_fp32 = bf16_u16_to_fp32(output_candidate)
    golden_candidate_fp32 = bf16_u16_to_fp32(golden_candidate)

    close_mask = np.isclose(
        golden_candidate_fp32,
        output_candidate_fp32,
        rtol=ERROR_TOL,
        atol=ERROR_TOL,
        equal_nan=True,
    )
    mismatch_mask = ~close_mask
    mismatch_idx = candidate_idx[mismatch_mask]

    mismatch_count = int(mismatch_idx.size)
    if mismatch_count == 0:
        print(
            f"[PASS] NPU results are consistent with CPU. "
            f"(checked_rows={sum_group_m}, checked_elements={compare_elem_count}, mismatch_count=0)"
        )
        return True

    total_checked = int(compare_elem_count)
    mismatch_ratio = mismatch_count / max(1, total_checked)
    print("[FAIL] NPU results differ from CPU.")
    print(
        f"summary: checked_rows={sum_group_m}, checked_elements={total_checked}, "
        f"mismatch_count={mismatch_count}, mismatch_ratio={mismatch_ratio:.6%}"
    )

    range_text = render_index_ranges(compress_indices(mismatch_idx))
    print("mismatch_flat_indices(compressed_ranges):")
    print(range_text)

    mismatch_output_u16 = output[mismatch_idx]
    mismatch_golden_u16 = golden[mismatch_idx]
    mismatch_output_fp32 = bf16_u16_to_fp32(mismatch_output_u16)
    mismatch_golden_fp32 = bf16_u16_to_fp32(mismatch_golden_u16)

    abs_err = np.abs(mismatch_output_fp32 - mismatch_golden_fp32)
    denom = np.maximum(np.abs(mismatch_golden_fp32), 1e-12)
    rel_err = abs_err / denom

    if max_detail_rows == -1:
        show_count = mismatch_count
    else:
        show_count = min(mismatch_count, max_detail_rows)

    detail_rows: List[List[str]] = []
    for i in range(show_count):
        flat_idx = int(mismatch_idx[i])
        row = flat_idx // n
        col = flat_idx % n
        detail_rows.append(
            [
                str(i),
                str(flat_idx),
                str(row),
                str(col),
                format_float(float(mismatch_golden_fp32[i])),
                format_float(float(mismatch_output_fp32[i])),
                format_float(float(abs_err[i])),
                format_float(float(rel_err[i])),
                f"0x{int(mismatch_golden_u16[i]):04x}",
                f"0x{int(mismatch_output_u16[i]):04x}",
            ]
        )

    headers = [
        "#",
        "flat_idx",
        "row",
        "col",
        "expected",
        "actual",
        "abs_err",
        "rel_err",
        "expected_u16",
        "actual_u16",
    ]
    print("mismatch_details:")
    print(render_table(headers, detail_rows))
    if show_count < mismatch_count:
        print(
            f"detail rows truncated: showing {show_count}/{mismatch_count}. "
            "Use --max-detail-rows -1 to print all."
        )
    return False


if __name__ == "__main__":
    try:
        args = parse_args()
        validate_args(args.group_num, args.m, args.k, args.n, args.max_detail_rows)
        group_m_list = load_group_m_list(args.group_num)
        res = verify_result(group_m_list, args.m, args.n, args.max_detail_rows)
        if not res:
            raise ValueError("[ERROR] NPU results differ from CPU.")
    except Exception as e:
        print(e)
        sys.exit(1)
