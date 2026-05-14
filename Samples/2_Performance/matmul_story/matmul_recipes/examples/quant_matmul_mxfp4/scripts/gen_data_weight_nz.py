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


TILING_MXFP_DIVISOR_SIZE = 64
MX_GROUP_SIZE = 32
CUBE_BLOCK = 16
FP4_C0_SIZE = 64
FP8_C0_SIZE = 32


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


def pack_weight_to_zn_fp4(weight_nk_packed: np.ndarray) -> np.ndarray:
    """Logical B (N, K) packed as (N, K//2) uint8 -> GM Zn bytes.

    Shape (ceil(K/FP4_C0), ceil(N/CUBE), CUBE, FP8_C0) uint8; two FP4 per byte along K.
    """
    if weight_nk_packed.ndim != 2:
        raise ValueError("weight_nk_packed must be 2D (N, K//2)")
    n, k_half = weight_nk_packed.shape
    k = k_half * 2
    n1 = math.ceil(n / CUBE_BLOCK)
    k1 = math.ceil(k / FP4_C0_SIZE)
    nz = np.zeros((k1, n1, CUBE_BLOCK, FP4_C0_SIZE // 2), dtype=np.uint8)
    raw = weight_nk_packed.astype(np.uint8, copy=False)
    for n_idx in range(n):
        n1_idx = n_idx // CUBE_BLOCK
        n0_idx = n_idx % CUBE_BLOCK
        for k_idx in range(k):
            k1_idx = k_idx // FP4_C0_SIZE
            k0_idx = k_idx % FP4_C0_SIZE
            k0_pack = k0_idx // 2
            nz[k1_idx, n1_idx, n0_idx, k0_pack] = raw[n_idx, k_idx // 2]
    return nz.reshape(-1)


def pack_weight_to_nz_fp4(weight_kn_packed: np.ndarray) -> np.ndarray:
    if weight_kn_packed.ndim != 2:
        raise ValueError("weight_kn_packed must be 2D (K, N//2)")
    k, n_half = weight_kn_packed.shape
    n0 = FP4_C0_SIZE
    n0_packed = n0 // 2
    k0 = CUBE_BLOCK
    n = n_half * 2
    n1 = math.ceil(n / n0)
    k1 = math.ceil(k / k0)
    nz = np.zeros((n1, k1, k0, n0_packed), dtype=np.uint8)
    raw = weight_kn_packed.astype(np.uint8, copy=False)
    for k_idx in range(k):
        k1_idx = k_idx // k0
        k0_idx = k_idx % k0
        for n_idx in range(n):
            n1_idx = n_idx // n0
            n0_base = n_idx % n0
            n0_pack_idx = n0_base // 2
            nz[n1_idx, k1_idx, k0_idx, n0_pack_idx] = raw[k_idx, n_idx // 2]
    return nz.reshape(-1)


def fp4_logical_b_to_weight_gm_bytes(b_ori: np.ndarray, trans_b: bool) -> np.ndarray:
    """Pack logical B fp4 matrix to GM weight-NZ bytes (MXFP4 Zn or Nz fractal)."""
    packed = pack_b4_to_b8(b_ori).astype(np.uint8, copy=False)
    if trans_b:
        return pack_weight_to_zn_fp4(packed)
    return pack_weight_to_nz_fp4(packed)


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


def dequant_mxfp4(fp4_input, scale, chunk_axis):
    scale_broadcast = build_scale_broadcast(scale, fp4_input.shape, chunk_axis)
    return fp4_input.astype(np.float32) * scale_broadcast


def validate_fp4_pack_shapes(m, k, n, trans_a, trans_b):
    # pack_b4_to_b8 pairs along the last dimension of the 2D logical tensor.
    if not trans_a:
        if k % 2 != 0:
            raise ValueError("transA=0 requires k even (fp4 packing along K).")
    else:
        if m % 2 != 0:
            raise ValueError("transA=1 requires m even (fp4 packing along M).")
    if trans_b:
        if k % 2 != 0:
            raise ValueError("transB=1 requires k even (fp4 packing along K).")
    else:
        if n % 2 != 0:
            raise ValueError("transB=0 requires n even (fp4 packing along N).")


def gen_golden_data_simple(m, k, n, trans_a=False, trans_b=True):
    validate_fp4_pack_shapes(m, k, n, trans_a, trans_b)

    a_shape = (k, m) if trans_a else (m, k)
    a_ori = np.random.uniform(1, 8, a_shape).astype(float4_e2m1fn)

    b_shape = (n, k) if trans_b else (k, n)
    b_ori = np.random.uniform(1, 8, b_shape).astype(float4_e2m1fn)

    div = TILING_MXFP_DIVISOR_SIZE
    a_scale_shape = (math.ceil(k / div), m, 2) if trans_a else (m, math.ceil(k / div), 2)
    b_scale_shape = (n, math.ceil(k / div), 2) if trans_b else (math.ceil(k / div), n, 2)
    a_scale = np.random.uniform(1, 8, size=a_scale_shape).astype(float8_e8m0)
    b_scale = np.random.uniform(1, 8, size=b_scale_shape).astype(float8_e8m0)

    a_chunk_axis = 0 if trans_a else 1
    b_chunk_axis = 1 if trans_b else 0

    a_dequant_input = dequant_mxfp4(a_ori, a_scale, a_chunk_axis)
    b_dequant_input = dequant_mxfp4(b_ori, b_scale, b_chunk_axis)

    a_matmul = np.swapaxes(a_dequant_input, -1, -2) if trans_a else a_dequant_input
    b_matmul = np.swapaxes(b_dequant_input, -1, -2) if trans_b else b_dequant_input

    a_pack_int8 = pack_b4_to_b8(a_ori)
    b_pack_int8 = fp4_logical_b_to_weight_gm_bytes(b_ori, trans_b).astype(np.int8, copy=False)

    a_cpu = torch.from_numpy(a_matmul)
    b_cpu = torch.from_numpy(b_matmul)
    out = torch.matmul(a_cpu, b_cpu).to(torch.bfloat16)

    current_dir = os.getcwd()
    write_artifacts(current_dir, a_pack_int8, b_pack_int8, a_scale, b_scale, out)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The script may be called from either the source tree or the installed
    # sample directory. When those differ, emit artifacts to both locations.
    if os.path.normcase(os.path.abspath(script_dir)) != os.path.normcase(os.path.abspath(current_dir)):
        write_artifacts(script_dir, a_pack_int8, b_pack_int8, a_scale, b_scale, out)


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
        print("Example: python3 gen_data_weight_nz.py 256 258 260 0 1")
        print(
            "FP4 packing: transA=0 needs k even; transA=1 needs m even; "
            "transB=1 needs k even; transB=0 needs n even."
        )
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
