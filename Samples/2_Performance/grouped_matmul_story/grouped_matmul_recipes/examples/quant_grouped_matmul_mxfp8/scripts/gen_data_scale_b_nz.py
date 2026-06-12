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
DEFAULT_TRANS_A = False
DEFAULT_TRANS_B = True
MX_DIVISOR_SIZE = 64
SCALE_C0 = 2
BLOCK_CUBE = 16


def _recipe_example_root() -> str:
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


def parse_bool_arg(arg: str, name: str) -> bool:
    value = arg.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"{name} must be 0/1/true/false")


def parse_cli_args(argv: List[str]) -> Tuple[List[int], int, int, int, bool, bool]:
    if len(argv) in {6, 8} and argv[1] == GROUP_LIST_MODE:
        group_m_list = parse_group_m_list(argv[2])
        m = int(argv[3])
        k = int(argv[4])
        n = int(argv[5])
        if m < sum(group_m_list):
            raise ValueError(f"m must be greater than or equal to sum(group_m_list)={sum(group_m_list)}")
        trans_a = DEFAULT_TRANS_A
        trans_b = DEFAULT_TRANS_B
        if len(argv) == 8:
            trans_a = parse_bool_arg(argv[6], "transA")
            trans_b = parse_bool_arg(argv[7], "transB")
        return group_m_list, m, k, n, trans_a, trans_b

    if len(argv) in {7, 9} and argv[1] == EXPECT_M_PER_GROUP_MODE:
        group_num = int(argv[2])
        expect_m_per_group = int(argv[3])
        m = int(argv[4])
        k = int(argv[5])
        n = int(argv[6])
        group_m_list = build_random_group_m_list(group_num, expect_m_per_group, m)
        trans_a = DEFAULT_TRANS_A
        trans_b = DEFAULT_TRANS_B
        if len(argv) == 9:
            trans_a = parse_bool_arg(argv[7], "transA")
            trans_b = parse_bool_arg(argv[8], "transB")
        return group_m_list, m, k, n, trans_a, trans_b

    raise ValueError(
        "Usage:\n"
        "  python3 gen_data_scale_b_nz.py group_list group_m_list m k n [transA transB]\n"
        "  python3 gen_data_scale_b_nz.py expect_m_per_group group_num expect_m_per_group m k n [transA transB]"
    )


def build_group_list(group_m_list: List[int]) -> np.ndarray:
    return np.array(group_m_list, dtype=np.int64)


def pack_scale_to_scaleB(scale_knm: np.ndarray) -> np.ndarray:
    """Pack one (Kg, N, SCALE_C0) fp8_e8m0 ScaleB tensor to scaleB private format.

    scaleB format: (N1, Kg, N0=16, SCALE_C0=2)
    where N1=ceil(N/16), Kg=ceil(K/64).
    scaleB[n1_idx, kg_idx, n0_idx, mx_idx] = scale_knm[kg_idx, n_idx, mx_idx]
    """
    if scale_knm.ndim != 3:
        raise ValueError("scale_knm must be 3D (Kg, N, SCALE_C0)")
    kg_raw, n, mx = scale_knm.shape
    assert mx == SCALE_C0, f"Last dimension must be SCALE_C0={SCALE_C0}, got {mx}"
    n1 = math.ceil(n / BLOCK_CUBE)
    padded_n = n1 * BLOCK_CUBE

    raw = scale_knm.view(np.uint8)
    if n < padded_n:
        buf = np.zeros((kg_raw, padded_n, SCALE_C0), dtype=np.uint8)
        buf[:, :n, :] = raw
        raw = buf

    scaleB = raw.reshape(kg_raw, n1, BLOCK_CUBE, SCALE_C0).transpose(1, 0, 2, 3)
    return scaleB.reshape(-1)


def pack_group_scale_to_scaleB(scale_gknm: np.ndarray) -> np.ndarray:
    """Pack all groups' (G, Kg, N, SCALE_C0) ScaleB to scaleB format.

    Always uses (N1, Kg, 16, 2) regardless of transB.
    """
    if scale_gknm.ndim != 4:
        raise ValueError("scale_gknm must be 4D (G, Kg, N, SCALE_C0)")
    group_num = scale_gknm.shape[0]
    packed_groups = [pack_scale_to_scaleB(scale_gknm[g]) for g in range(group_num)]
    return np.concatenate(packed_groups, axis=0) if packed_groups else np.empty((0,), dtype=np.uint8)


def write_artifacts(base_dir, a_fp8, b_fp8, a_scale, b_scale_scaleB, group_list, out):
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    a_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_a.bin"))
    b_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_b.bin"))
    a_scale.view(np.uint8).reshape(-1).tofile(os.path.join(input_dir, "input_scaleA.bin"))
    b_scale_scaleB.tofile(os.path.join(input_dir, "input_scaleB.bin"))
    group_list.tofile(os.path.join(input_dir, "input_groupList.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data_simple(group_m_list: List[int], m: int, k: int, n: int, trans_a: bool, trans_b: bool):
    group_num = len(group_m_list)
    if trans_a:
        raise ValueError("transA=true is not supported in grouped quant matmul scale_b_nz samples")

    a_ori = np.random.uniform(1, 8, (m, k)).astype(float8_e4m3fn)
    b_shape = (group_num, n, k) if trans_b else (group_num, k, n)
    b_ori = np.random.uniform(1, 8, b_shape).astype(float8_e4m3fn)

    scale_k = math.ceil(k / MX_DIVISOR_SIZE)
    a_scale = np.random.uniform(1, 8, size=(m, scale_k, SCALE_C0)).astype(float8_e8m0)
    b_scale_shape = (group_num, n, scale_k, SCALE_C0) if trans_b else (group_num, scale_k, n, SCALE_C0)
    b_scale = np.random.uniform(1, 8, size=b_scale_shape).astype(float8_e8m0)

    if trans_b:
        b_scale_for_scaleB = np.swapaxes(b_scale, 1, 2)  # (G, N, Kg, 2) -> (G, Kg, N, 2)
    else:
        b_scale_for_scaleB = b_scale  # (G, Kg, N, 2) already
    b_scale_scaleB = pack_group_scale_to_scaleB(b_scale_for_scaleB)

    group_list = build_group_list(group_m_list)

    a_scale_reshape = a_scale.reshape(m, scale_k * SCALE_C0)
    a_scale_broadcast = np.repeat(a_scale_reshape, 32, axis=-1)[..., :k]

    outputs = []
    m_offset = 0
    for group_idx, group_m in enumerate(group_m_list):
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
    write_artifacts(base, a_ori, b_ori, a_scale, b_scale_scaleB, group_list, out)


if __name__ == "__main__":
    try:
        group_m_list, m, k, n, trans_a, trans_b = parse_cli_args(sys.argv)
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
    if trans_a:
        print("transA=true is not supported")
        sys.exit(1)

    print(f"group_m_list={','.join(str(value) for value in group_m_list)}")
    print(f"m={m}, sum(group_m_list)={sum_group_m}, k={k}, n={n}, transA={trans_a}, transB={trans_b}")
    gen_golden_data_simple(group_m_list, m, k, n, trans_a, trans_b)