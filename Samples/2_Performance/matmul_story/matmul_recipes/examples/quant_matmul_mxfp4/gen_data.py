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
from ml_dtypes import float4_e2m1fn


def pack_b4_to_b8(b4_data: np.ndarray):
    # The sample stores two fp4 values inside one byte, so pack the last
    # dimension in pairs before writing the binary inputs consumed by the kernel.
    packed_shape = [b4_data.shape[0], int(b4_data.shape[1] / 2)]
    pack_size = 2
    shift = np.array([0, 4], dtype=np.int8)
    if b4_data.size % pack_size != 0:
        b4_data = np.pad(b4_data.flatten(), (0, pack_size - b4_data.size % pack_size), "constant")
    b4_data = b4_data.reshape(-1, 2).view(np.int8)
    return np.sum(np.bitwise_and(b4_data, 0b00001111) << shift, axis=1, dtype=np.int8).reshape(packed_shape)


def write_artifacts(base_dir, a_pack_int8, b_pack_int8, a_scale, b_scale, out):
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    a_pack_int8.tofile(os.path.join(input_dir, "input_a.bin"))
    b_pack_int8.tofile(os.path.join(input_dir, "input_b.bin"))
    a_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    b_scale.tofile(os.path.join(input_dir, "input_scaleB.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data_simple(m, k, n):
    M = m
    K = k
    N = n

    # Generate data.
    # When the result of the computation is very large, the precision check may fail.
    a_ori = np.random.uniform(1, 8, (M, K)).astype(float4_e2m1fn)
    a_pack_int8 = pack_b4_to_b8(a_ori)
    # Keep B in (N, K) order so the packed file matches the sample's
    # column-major/filter-major interpretation on the device side.
    b_ori = np.random.uniform(1, 8, (N, K)).astype(float4_e2m1fn)
    b_pack_int8 = pack_b4_to_b8(b_ori)
    a_scale = np.random.uniform(1, 8, size=(M, math.ceil(K / 64), 2)).astype(float8_e8m0)
    b_scale = np.random.uniform(1, 8, size=(N, math.ceil(K / 64), 2)).astype(float8_e8m0)

    # Expand A-side scales directly in [M, K] order, then transpose the B-side
    # activations and scales so the CPU reference matches the sample's B^T view.
    a_scale_reshape = a_scale.reshape(M, -1)
    a_scale_broadcast = np.repeat(a_scale_reshape, 32, axis=-1)[..., :K]

    b_ori_transpose = np.swapaxes(b_ori, -1, -2)
    b_scale_reshape = b_scale.reshape(N, -1)
    b_scale_broadcast = np.repeat(b_scale_reshape, 32, axis=-1)[..., :K]
    b_scale_broadcast_transpose = np.swapaxes(b_scale_broadcast, -1, -2)

    # Dequantize inputs.
    a_dequant = a_ori.astype(np.float32) * a_scale_broadcast.astype(np.float32)
    b_dequant = b_ori_transpose.astype(np.float32) * b_scale_broadcast_transpose.astype(np.float32)

    a_cpu = torch.from_numpy(a_dequant)
    b_cpu = torch.from_numpy(b_dequant)
    out = torch.matmul(a_cpu, b_cpu).to(torch.bfloat16)

    current_dir = os.getcwd()
    write_artifacts(current_dir, a_pack_int8, b_pack_int8, a_scale, b_scale, out)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The script may be called from either the source tree or the installed
    # sample directory. When those differ, emit artifacts to both locations.
    if os.path.normcase(os.path.abspath(script_dir)) != os.path.normcase(os.path.abspath(current_dir)):
        write_artifacts(script_dir, a_pack_int8, b_pack_int8, a_scale, b_scale, out)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 gen_data.py m k n")
        sys.exit(1)

    # Parse command-line arguments.
    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    # The only remaining layout constraint is fp4 packing granularity: two B4
    # values are packed into one B8 value, so k must be even.
    if k % 2 != 0:
        print("k must be a multiple of 2")
        sys.exit(1)

    gen_golden_data_simple(m, k, n)
