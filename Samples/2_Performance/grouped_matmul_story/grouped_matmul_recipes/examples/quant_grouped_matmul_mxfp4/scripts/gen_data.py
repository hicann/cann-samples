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

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import numpy as np
import torch
from en_dtypes import float8_e8m0
from ml_dtypes import float4_e2m1fn


GROUP_LIST_MODE = "group_list"
EXPECT_M_PER_GROUP_MODE = "expect_m_per_group"
DEFAULT_TRANS_A = False
DEFAULT_TRANS_B = True


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
            raise ValueError("group_value_list contains an empty item")
        value = int(item)
        if value < 0:
            raise ValueError("Each group M value must be greater than or equal to 0")
        values.append(value)
    if not values:
        raise ValueError("group_value_list must not be empty")
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
        group_value_list = np.random.randint(low, high + 1, size=group_num).astype(int).tolist()
        if sum(group_value_list) <= m:
            return group_value_list

    # Fallback: start from lower bound and distribute remaining budget
    # without exceeding the upper bound.
    group_value_list = [low] * group_num
    remaining = m - sum(group_value_list)
    if remaining <= 0:
        return group_value_list

    capacities = [high - low for _ in range(group_num)]
    order = np.random.permutation(group_num).tolist()
    while remaining > 0:
        progressed = False
        for idx in order:
            if capacities[idx] <= 0:
                continue
            group_value_list[idx] += 1
            capacities[idx] -= 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return group_value_list


def parse_bool_arg(arg: str, name: str) -> bool:
    value = arg.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"{name} must be 0/1/true/false")


def parse_cli_args(argv: List[str]) -> Tuple[List[int], int, int, int, bool, bool]:
    if len(argv) in {6, 8} and argv[1] == GROUP_LIST_MODE:
        group_value_list = parse_group_m_list(argv[2])
        m = int(argv[3])
        k = int(argv[4])
        n = int(argv[5])
        if m < sum(group_value_list):
            raise ValueError(f"m must be greater than or equal to sum(group_value_list)={sum(group_value_list)}")
        trans_a = DEFAULT_TRANS_A
        trans_b = DEFAULT_TRANS_B
        if len(argv) == 8:
            trans_a = parse_bool_arg(argv[6], "transA")
            trans_b = parse_bool_arg(argv[7], "transB")
        return group_value_list, m, k, n, trans_a, trans_b

    if len(argv) in {7, 9} and argv[1] == EXPECT_M_PER_GROUP_MODE:
        group_num = int(argv[2])
        expect_m_per_group = int(argv[3])
        m = int(argv[4])
        k = int(argv[5])
        n = int(argv[6])
        group_value_list = build_random_group_m_list(group_num, expect_m_per_group, m)
        trans_a = DEFAULT_TRANS_A
        trans_b = DEFAULT_TRANS_B
        if len(argv) == 9:
            trans_a = parse_bool_arg(argv[7], "transA")
            trans_b = parse_bool_arg(argv[8], "transB")
        return group_value_list, m, k, n, trans_a, trans_b

    raise ValueError(
        "Usage:\n"
        "  python3 gen_data.py group_list group_value_list m k n [transA transB]\n"
        "  python3 gen_data.py expect_m_per_group group_num expect_m_per_group m k n [transA transB]"
    )


def pack_b4_to_b8(b4_data: np.ndarray):
    # Two fp4 values are packed into one byte before writing sample inputs.
    packed_shape = list(b4_data.shape[:-1]) + [int(b4_data.shape[-1] / 2)]
    pack_size = 2
    shift = np.array([0, 4], dtype=np.int8)
    if b4_data.size % pack_size != 0:
        b4_data = np.pad(b4_data.flatten(), (0, pack_size - b4_data.size % pack_size), "constant")
    b4_data = b4_data.reshape(-1, 2).view(np.int8)
    return np.sum(np.bitwise_and(b4_data, 0b00001111) << shift, axis=1, dtype=np.int8).reshape(packed_shape)


def build_group_list(group_value_list: List[int]) -> np.ndarray:
    return np.array(group_value_list, dtype=np.int64)


def write_artifacts(base_dir, a_pack_int8, b_pack_int8, a_scale, b_scale, group_list, out):
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    a_pack_int8.tofile(os.path.join(input_dir, "input_a.bin"))
    b_pack_int8.tofile(os.path.join(input_dir, "input_b.bin"))
    a_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    b_scale.tofile(os.path.join(input_dir, "input_scaleB.bin"))
    group_list.tofile(os.path.join(input_dir, "input_groupList.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data_simple(group_value_list: List[int], m: int, k: int, n: int, trans_a: bool, trans_b: bool):
    group_num = len(group_value_list)
    if trans_a:
        raise ValueError("transA=true is not supported in grouped quant matmul samples")

    # X and scales are sized to the declared M budget (m), not sum(group_value_list), so inputs stay well-defined when
    # some experts have zero rows. Golden output is padded to (m, n) with zeros below the computed rows.
    a_ori = np.random.uniform(1, 8, (m, k)).astype(float4_e2m1fn)
    a_pack_int8 = pack_b4_to_b8(a_ori)

    b_shape = (group_num, n, k) if trans_b else (group_num, k, n)
    b_ori = np.random.uniform(1, 8, b_shape).astype(float4_e2m1fn)
    b_pack_int8 = pack_b4_to_b8(b_ori)

    scale_k = math.ceil(k / 64)
    a_scale = np.random.uniform(1, 8, size=(m, scale_k, 2)).astype(float8_e8m0)
    b_scale_shape = (group_num, n, scale_k, 2) if trans_b else (group_num, scale_k, n, 2)
    b_scale = np.random.uniform(1, 8, size=b_scale_shape).astype(float8_e8m0)
    group_list = build_group_list(group_value_list)

    a_scale_reshape = a_scale.reshape(m, scale_k * 2)
    a_scale_broadcast = np.repeat(a_scale_reshape, 32, axis=-1)[..., :k]

    outputs = []
    m_offset = 0
    for group_idx, group_m in enumerate(group_value_list):
        if group_m == 0:
            continue
        a_group = a_ori[m_offset : m_offset + group_m]
        a_group_scale = a_scale_broadcast[m_offset : m_offset + group_m]
        b_group = b_ori[group_idx]
        if trans_b:
            b_dequant_base = np.swapaxes(b_group, -1, -2)
            b_group_scale = b_scale[group_idx].reshape(n, -1)
            b_group_scale_broadcast = np.repeat(b_group_scale, 32, axis=-1)[..., :k]
            b_scale_dequant = np.swapaxes(b_group_scale_broadcast, -1, -2)
        else:
            b_dequant_base = b_group
            b_group_scale = np.swapaxes(b_scale[group_idx], 0, 1).reshape(n, -1)
            b_group_scale_broadcast = np.repeat(b_group_scale, 32, axis=-1)[..., :k]
            b_scale_dequant = np.swapaxes(b_group_scale_broadcast, -1, -2)

        a_dequant = a_group.astype(np.float32) * a_group_scale.astype(np.float32)
        b_dequant = b_dequant_base.astype(np.float32) * b_scale_dequant.astype(np.float32)
        outputs.append(torch.matmul(torch.from_numpy(a_dequant), torch.from_numpy(b_dequant)).to(torch.bfloat16))
        m_offset += group_m

    out = torch.cat(outputs, dim=0) if outputs else torch.empty((0, n), dtype=torch.bfloat16)
    if out.shape[0] < m:
        pad = torch.zeros((m - out.shape[0], n), dtype=torch.bfloat16)
        out = torch.cat([out, pad], dim=0)
    elif out.shape[0] > m:
        raise ValueError("internal error: computed rows exceed m")

    base = _recipe_example_root()
    write_artifacts(base, a_pack_int8, b_pack_int8, a_scale, b_scale, group_list, out)


if __name__ == "__main__":
    try:
        group_value_list, m, k, n, trans_a, trans_b = parse_cli_args(sys.argv)
    except ValueError as error:
        print(error)
        sys.exit(1)

    if m < 0:
        print("m must be greater than or equal to 0")
        sys.exit(1)
    if k <= 0 or n <= 0:
        print("k and n must be greater than 0")
        sys.exit(1)
    sum_group_m = sum(group_value_list)
    if m < sum_group_m:
        print(f"m must be greater than or equal to sum(group_value_list)={sum_group_m}")
        sys.exit(1)
    if trans_a:
        print("transA=true is not supported")
        sys.exit(1)
    if k % 2 != 0:
        print("k must be a multiple of 2")
        sys.exit(1)
    if not trans_b and n % 2 != 0:
        print("n must be a multiple of 2 when transB=false")
        sys.exit(1)

    print(f"group_value_list={','.join(str(value) for value in group_value_list)}")
    print(f"m={m}, sum(group_value_list)={sum_group_m}, k={k}, n={n}, transA={trans_a}, transB={trans_b}")
    gen_golden_data_simple(group_value_list, m, k, n, trans_a, trans_b)
