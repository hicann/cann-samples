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
from typing import List

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import numpy as np
import torch
from en_dtypes import hifloat8
from ml_dtypes import bfloat16

GROUP_LIST_MODE = "group_list"
EXPECT_M_PER_GROUP_MODE = "expect_m_per_group"
DEFAULT_TRANS_A = False
DEFAULT_TRANS_B = True


def example_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here) if os.path.basename(here) == "scripts" else here


def parse_bool_arg(arg: str, name: str) -> bool:
    value = arg.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"{name} must be 0/1/true/false")


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
    low = max(0, int(math.floor(expect_m_per_group * 0.7)))
    high = max(low, int(math.ceil(expect_m_per_group * 1.3)))
    if group_num <= 0:
        raise ValueError("group_num must be greater than 0")
    if m < group_num * low:
        raise ValueError("m is too small for the requested group distribution")
    for _ in range(200):
        values = np.random.randint(low, high + 1, size=group_num).astype(int).tolist()
        if sum(values) <= m:
            return values
    values = [low] * group_num
    remaining = m - sum(values)
    for idx in np.random.permutation(group_num).tolist():
        add = min(remaining, high - values[idx])
        values[idx] += add
        remaining -= add
        if remaining == 0:
            break
    return values


def parse_cli_args(argv):
    if len(argv) in {6, 8} and argv[1] == GROUP_LIST_MODE:
        group_m_list = parse_group_m_list(argv[2])
        m, k, n = int(argv[3]), int(argv[4]), int(argv[5])
        trans_a, trans_b = DEFAULT_TRANS_A, DEFAULT_TRANS_B
        if len(argv) == 8:
            trans_a = parse_bool_arg(argv[6], "transA")
            trans_b = parse_bool_arg(argv[7], "transB")
        return group_m_list, m, k, n, trans_a, trans_b
    if len(argv) in {7, 9} and argv[1] == EXPECT_M_PER_GROUP_MODE:
        group_num = int(argv[2])
        expect_m_per_group = int(argv[3])
        m, k, n = int(argv[4]), int(argv[5]), int(argv[6])
        trans_a, trans_b = DEFAULT_TRANS_A, DEFAULT_TRANS_B
        if len(argv) == 9:
            trans_a = parse_bool_arg(argv[7], "transA")
            trans_b = parse_bool_arg(argv[8], "transB")
        return build_random_group_m_list(group_num, expect_m_per_group, m), m, k, n, trans_a, trans_b
    raise ValueError(
        "Usage:\n"
        "  python3 gen_data_tt.py group_list group_m_list m k n [transA transB]\n"
        "  python3 gen_data_tt.py expect_m_per_group group_num expect_m_per_group m k n [transA transB]"
    )


def scale_generate(value):
    arr = np.atleast_1d(np.array(value, dtype=np.float32, copy=True))
    bits = arr.view(np.uint32)
    bits &= np.uint32(0xFFFFE000)
    return bits.view(np.float32)


def write_artifacts(base_dir, x1_u8, x2_u8, scale_a, scale_b, group_list, out):
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    x1_u8.tofile(os.path.join(input_dir, "x1.bin"))
    x2_u8.tofile(os.path.join(input_dir, "x2.bin"))
    scale_a.tofile(os.path.join(input_dir, "pertoken_scale.bin"))
    scale_b.tofile(os.path.join(input_dir, "scale.bin"))
    group_list.tofile(os.path.join(input_dir, "input_groupList.bin"))
    out.to(torch.float32).numpy().astype(bfloat16).tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_data(group_m_list: List[int], m: int, k: int, n: int, trans_a: bool, trans_b: bool):
    if trans_a:
        raise ValueError("Grouped HiFloat8 sample supports split-M only: transA must be false")
    if sum(group_m_list) > m:
        raise ValueError("m must be greater than or equal to sum(group_m_list)")
    group_num = len(group_m_list)
    x1_u8 = np.random.randint(8, 30, size=(m, k), dtype=np.uint8)
    x2_shape = (group_num, n, k) if trans_b else (group_num, k, n)
    x2_u8 = np.random.randint(8, 30, size=x2_shape, dtype=np.uint8)
    x1 = x1_u8.view(hifloat8).astype(np.float32)
    x2 = x2_u8.view(hifloat8).astype(np.float32)
    pertoken_scale = np.random.uniform(1, 10, size=(group_num,)).astype(np.float32)
    scale = np.random.uniform(1, 10, size=(group_num,)).astype(np.float32)

    outputs = []
    m_offset = 0
    for group_idx, group_m in enumerate(group_m_list):
        if group_m == 0:
            continue
        a_group = torch.from_numpy(x1[m_offset : m_offset + group_m])
        b_group = x2[group_idx].T if trans_b else x2[group_idx]
        out = torch.matmul(a_group, torch.from_numpy(b_group))
        out *= torch.from_numpy(scale_generate(pertoken_scale[group_idx] * scale[group_idx]))[0]
        outputs.append(out.to(torch.bfloat16))
        m_offset += group_m
    out = torch.cat(outputs, dim=0) if outputs else torch.empty((0, n), dtype=torch.bfloat16)
    if out.shape[0] < m:
        out = torch.cat([out, torch.zeros((m - out.shape[0], n), dtype=torch.bfloat16)], dim=0)
    write_artifacts(example_root(), x1_u8, x2_u8, pertoken_scale, scale, np.array(group_m_list, np.int64), out)


if __name__ == "__main__":
    try:
        parsed = parse_cli_args(sys.argv)
        print(f"group_m_list={','.join(str(v) for v in parsed[0])}")
        gen_data(*parsed)
        print("Generate done (grouped HiFloat8 TT).")
    except Exception as err:
        print(err)
        sys.exit(1)
