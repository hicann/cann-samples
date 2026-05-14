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

import numpy as np
import torch
from en_dtypes import float8_e8m0
from ml_dtypes import float8_e4m3fn

TILING_MXFP_DIVISOR_SIZE = 64
MX_GROUP_SIZE = 32
CUBE_BLOCK = 16
FP8_C0_SIZE = 32


def to_weight_nz_layout(fp8_input):
    """Convert 2D [row, col] to weight-NZ GM bytes layout [ceil(col/32), ceil(row/16), 16, 32]."""
    row, col = fp8_input.shape
    row_tiles = math.ceil(row / CUBE_BLOCK)
    col_tiles = math.ceil(col / FP8_C0_SIZE)
    padded_row = row_tiles * CUBE_BLOCK
    padded_col = col_tiles * FP8_C0_SIZE

    padded = np.zeros((padded_row, padded_col), dtype=fp8_input.dtype)
    padded[:row, :col] = fp8_input

    blocked = padded.reshape(row_tiles, CUBE_BLOCK, col_tiles, FP8_C0_SIZE)
    return blocked.transpose(2, 0, 1, 3)


def write_artifacts(base_dir, a_fp8, b_fp8, a_scale, b_scale, out):
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    b_to_write = to_weight_nz_layout(b_fp8)

    a_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_a.bin"))
    b_to_write.view(np.uint8).tofile(os.path.join(input_dir, "input_b.bin"))
    a_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    b_scale.tofile(os.path.join(input_dir, "input_scaleB.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def build_scale_broadcast(scale, target_shape, chunk_axis):
    # Repeat on the last dim (size=2), then unfold along ceil(dim/64).
    scale_repeat = np.repeat(scale.astype(np.float32), MX_GROUP_SIZE, axis=-1)

    if chunk_axis == 1:
        # scale: [row, ceil(col/64), 2] -> [row, col]
        scale_broadcast = scale_repeat.reshape(scale.shape[0], -1)[..., : target_shape[1]]
    elif chunk_axis == 0:
        # scale: [ceil(row/64), col, 2] -> [row, col]
        scale_broadcast = np.transpose(scale_repeat, (0, 2, 1)).reshape(-1, scale.shape[1])[: target_shape[0], ...]
    else:
        raise ValueError(f"Invalid chunk_axis={chunk_axis}, expected 0 or 1.")

    return scale_broadcast


def dequant_mxfp8(fp8_input, scale, chunk_axis):
    scale_broadcast = build_scale_broadcast(scale, fp8_input.shape, chunk_axis)
    return fp8_input.astype(np.float32) * scale_broadcast


def gen_golden_data_simple(m, k, n, trans_a=False, trans_b=True):
    a_shape = (k, m) if trans_a else (m, k)
    a_ori = np.random.uniform(1, 8, a_shape).astype(float8_e4m3fn)

    b_shape = (n, k) if trans_b else (k, n)
    b_ori = np.random.uniform(1, 8, b_shape).astype(float8_e4m3fn)

    div = TILING_MXFP_DIVISOR_SIZE
    a_scale_shape = (math.ceil(k / div), m, 2) if trans_a else (m, math.ceil(k / div), 2)
    b_scale_shape = (n, math.ceil(k / div), 2) if trans_b else (math.ceil(k / div), n, 2)
    a_scale = np.random.uniform(1, 8, size=a_scale_shape).astype(float8_e8m0)
    b_scale = np.random.uniform(1, 8, size=b_scale_shape).astype(float8_e8m0)

    a_chunk_axis = 0 if trans_a else 1
    b_chunk_axis = 1 if trans_b else 0

    a_dequant_input = dequant_mxfp8(a_ori, a_scale, a_chunk_axis)
    b_dequant_input = dequant_mxfp8(b_ori, b_scale, b_chunk_axis)

    a_matmul = np.swapaxes(a_dequant_input, -1, -2) if trans_a else a_dequant_input
    b_matmul = np.swapaxes(b_dequant_input, -1, -2) if trans_b else b_dequant_input

    a_cpu = torch.from_numpy(a_matmul)
    b_cpu = torch.from_numpy(b_matmul)
    out = torch.matmul(a_cpu, b_cpu).to(torch.bfloat16)

    current_dir = os.getcwd()
    write_artifacts(current_dir, a_ori, b_ori, a_scale, b_scale, out)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The script may be called from either the source tree or the installed
    # sample directory. When those differ, emit artifacts to both locations.
    if os.path.normcase(os.path.abspath(script_dir)) != os.path.normcase(os.path.abspath(current_dir)):
        write_artifacts(script_dir, a_ori, b_ori, a_scale, b_scale, out)


def parse_bool_arg(value, name):
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "t"):
        return True
    if normalized in ("0", "false", "f"):
        return False
    raise ValueError(f"Invalid {name}: {value}. Expected one of 0/1/true/false.")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 6):
        print("Usage: python3 gen_data_weight_nz.py m k n [transA transB]")
        print("Example: python3 gen_data_weight_nz.py 257 258 259 0 1")
        print("MXFP8: one byte per element; layout matches gen_data.py / sample README.")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])

    if len(sys.argv) >= 6:
        trans_a = parse_bool_arg(sys.argv[4], "transA")
        trans_b = parse_bool_arg(sys.argv[5], "transB")
    else:
        trans_a = False
        trans_b = True

    try:
        gen_golden_data_simple(m, k, n, trans_a, trans_b)
    except ValueError as e:
        print(str(e))
        sys.exit(1)
