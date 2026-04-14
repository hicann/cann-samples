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

import math
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from en_dtypes import float8_e8m0
from ml_dtypes import float8_e4m3fn


GROUP_LIST_MODE = "group_list"
EXPECT_M_PER_GROUP_MODE = "expect_m_per_group"


def _recipe_example_root() -> str:
    """Directory that holds input/ and output/ for this example (parent of scripts/ when applicable)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(here) == "scripts":
        return os.path.dirname(here)
    return here


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


def build_random_group_m_list(group_num: int, expect_m_per_group: int, m: int) -> List[int]:
    if group_num <= 0:
        raise ValueError("group_num must be greater than 0")
    if expect_m_per_group < 0:
        raise ValueError("expect_m_per_group must be greater than or equal to 0")
    if m < 0:
        raise ValueError("m must be greater than or equal to 0")

    low = int(math.floor(expect_m_per_group * 0.7))
    high = int(math.ceil(expect_m_per_group * 1.3))
    low = max(0, low)
    high = max(low, high)

    min_total_m = group_num * low
    if m < min_total_m:
        raise ValueError(
            f"m must be greater than or equal to group_num * floor(0.7 * expect_m_per_group)={min_total_m}"
        )

    if high == 0:
        return [0] * group_num

    # Keep each group in [0.7, 1.3] * expect_m_per_group and ensure sum <= m.
    for _ in range(200):
        group_m_list = np.random.randint(low, high + 1, size=group_num).astype(int).tolist()
        if sum(group_m_list) <= m:
            return group_m_list

    group_m_list = [low] * group_num
    remaining = m - sum(group_m_list)
    if remaining <= 0:
        return group_m_list

    capacities = [high - low for _ in range(group_num)]
    order = np.random.permutation(group_num).tolist()
    while remaining > 0:
        progressed = False
        for idx in order:
            if capacities[idx] <= 0:
                continue
            group_m_list[idx] += 1
            capacities[idx] -= 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return group_m_list


def parse_cli_args(argv: List[str]) -> Tuple[List[int], int, int, int]:
    if len(argv) == 6 and argv[1] == GROUP_LIST_MODE:
        group_m_list = parse_group_m_list(argv[2])
        m = int(argv[3])
        k = int(argv[4])
        n = int(argv[5])
        if m < sum(group_m_list):
            raise ValueError(f"m must be greater than or equal to sum(group_m_list)={sum(group_m_list)}")
        return group_m_list, m, k, n

    if len(argv) == 7 and argv[1] == EXPECT_M_PER_GROUP_MODE:
        group_num = int(argv[2])
        expect_m_per_group = int(argv[3])
        m = int(argv[4])
        k = int(argv[5])
        n = int(argv[6])
        group_m_list = build_random_group_m_list(group_num, expect_m_per_group, m)
        return group_m_list, m, k, n

    raise ValueError(
        "Usage:\n"
        "  python3 gen_data.py group_list group_m_list m k n\n"
        "  python3 gen_data.py expect_m_per_group group_num expect_m_per_group m k n"
    )


def build_group_list(group_m_list: List[int]) -> np.ndarray:
    return np.array(group_m_list, dtype=np.int64)


def write_artifacts(base_dir, a_fp8, b_fp8, a_scale, b_scale, group_list, out):
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    a_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_a.bin"))
    b_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_b.bin"))
    a_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    b_scale.tofile(os.path.join(input_dir, "input_scaleB.bin"))
    group_list.tofile(os.path.join(input_dir, "input_groupList.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data_simple(group_m_list: List[int], m: int, k: int, n: int):
    group_num = len(group_m_list)

    # X and scales are sized to the declared M budget (m), not sum(group_m_list), so inputs stay well-defined when
    # some experts have zero rows. Golden output is padded to (m, n) with zeros below the computed rows.
    a_ori = np.random.uniform(1, 8, (m, k)).astype(float8_e4m3fn)

    # Keep each weight group in (N, K) order so it matches the sample's
    # column-major/filter-major device-side interpretation.
    b_ori = np.random.uniform(1, 8, (group_num, n, k)).astype(float8_e4m3fn)

    scale_k = math.ceil(k / 64)
    a_scale = np.random.uniform(1, 8, size=(m, scale_k, 2)).astype(float8_e8m0)
    b_scale = np.random.uniform(1, 8, size=(group_num, n, scale_k, 2)).astype(float8_e8m0)
    group_list = build_group_list(group_m_list)

    a_scale_reshape = a_scale.reshape(m, scale_k * 2)
    a_scale_broadcast = np.repeat(a_scale_reshape, 32, axis=-1)[..., :k]

    outputs = []
    m_offset = 0
    for group_idx, group_m in enumerate(group_m_list):
        if group_m == 0:
            continue
        a_group = a_ori[m_offset : m_offset + group_m]
        a_group_scale = a_scale_broadcast[m_offset : m_offset + group_m]
        b_group = b_ori[group_idx]
        b_group_transpose = np.swapaxes(b_group, -1, -2)
        b_group_scale = b_scale[group_idx].reshape(n, -1)
        b_group_scale_broadcast = np.repeat(b_group_scale, 32, axis=-1)[..., :k]
        b_group_scale_broadcast_transpose = np.swapaxes(b_group_scale_broadcast, -1, -2)

        a_dequant = a_group.astype(np.float32) * a_group_scale.astype(np.float32)
        b_dequant = b_group_transpose.astype(np.float32) * b_group_scale_broadcast_transpose.astype(np.float32)
        outputs.append(torch.matmul(torch.from_numpy(a_dequant), torch.from_numpy(b_dequant)).to(torch.bfloat16))
        m_offset += group_m

    out = torch.cat(outputs, dim=0) if outputs else torch.empty((0, n), dtype=torch.bfloat16)
    if out.shape[0] < m:
        pad = torch.zeros((m - out.shape[0], n), dtype=torch.bfloat16)
        out = torch.cat([out, pad], dim=0)
    elif out.shape[0] > m:
        raise ValueError("internal error: computed rows exceed m")

    base = _recipe_example_root()
    write_artifacts(base, a_ori, b_ori, a_scale, b_scale, group_list, out)


if __name__ == "__main__":
    try:
        group_m_list, m, k, n = parse_cli_args(sys.argv)
    except ValueError as error:
        print(error)
        sys.exit(1)

    if m < 0:
        print("m must be greater than or equal to 0")
        sys.exit(1)
    if k <= 0 or n <= 0:
        print("k and n must be greater than 0")
        sys.exit(1)

    sum_group_m = sum(group_m_list)
    if m < sum_group_m:
        print(f"m must be greater than or equal to sum(group_m_list)={sum_group_m}")
        sys.exit(1)

    print(f"group_m_list={','.join(str(value) for value in group_m_list)}")
    print(f"m={m}, sum(group_m_list)={sum_group_m}, k={k}, n={n}")
    gen_golden_data_simple(group_m_list, m, k, n)
