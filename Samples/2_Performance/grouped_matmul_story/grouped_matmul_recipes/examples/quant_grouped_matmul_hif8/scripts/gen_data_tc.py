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

import os
import sys

import numpy as np
import torch

from gen_data_tt import example_root, parse_cli_args
from en_dtypes import hifloat8
from ml_dtypes import bfloat16


def write_artifacts(base_dir, x1_u8, x2_u8, scale, group_list, out):
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    x1_u8.tofile(os.path.join(input_dir, "x1.bin"))
    x2_u8.tofile(os.path.join(input_dir, "x2.bin"))
    scale.tofile(os.path.join(input_dir, "scale.bin"))
    group_list.tofile(os.path.join(input_dir, "input_groupList.bin"))
    out.to(torch.float32).numpy().astype(bfloat16).tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_data(group_m_list, m: int, k: int, n: int, trans_a: bool, trans_b: bool):
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

    scale_fp32 = np.random.uniform(0.01, 2.0, size=(group_num, n)).astype(np.float32)
    scale_u32 = scale_fp32.view(np.uint32)
    scale_u32 &= np.uint32(0xFFFFE000)
    scale_high19 = scale_u32.view(np.float32)
    scale_u64 = scale_u32.astype(np.uint64) | (np.uint64(1) << np.uint64(46))

    outputs = []
    m_offset = 0
    for group_idx, group_m in enumerate(group_m_list):
        if group_m == 0:
            continue
        a_group = torch.from_numpy(x1[m_offset : m_offset + group_m])
        b_group = x2[group_idx].T if trans_b else x2[group_idx]
        out = torch.matmul(a_group, torch.from_numpy(b_group))
        out *= torch.from_numpy(scale_high19[group_idx]).reshape(1, n)
        outputs.append(out.to(torch.bfloat16))
        m_offset += group_m
    out = torch.cat(outputs, dim=0) if outputs else torch.empty((0, n), dtype=torch.bfloat16)
    if out.shape[0] < m:
        out = torch.cat([out, torch.zeros((m - out.shape[0], n), dtype=torch.bfloat16)], dim=0)
    write_artifacts(example_root(), x1_u8, x2_u8, scale_u64, np.array(group_m_list, np.int64), out)


if __name__ == "__main__":
    try:
        parsed = parse_cli_args(sys.argv)
        print(f"group_m_list={','.join(str(v) for v in parsed[0])}")
        gen_data(*parsed)
        print("Generate done (grouped HiFloat8 TC).")
    except Exception as err:
        print(err)
        sys.exit(1)
