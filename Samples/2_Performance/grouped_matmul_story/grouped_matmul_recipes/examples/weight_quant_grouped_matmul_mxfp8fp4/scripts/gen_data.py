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
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import numpy as np
from en_dtypes import float8_e8m0
from ml_dtypes import float4_e2m1fn, float8_e4m3fn


GROUP_LIST_MODE = "group_list"
EXPECT_M_PER_GROUP_MODE = "expect_m_per_group"
FP4_E2M1FN_TO_F32_LUT = np.arange(16, dtype=np.int8).view(float4_e2m1fn).astype(np.float32)
RANDOM_GENERATOR = np.random.default_rng()
TORCH_BACKEND_MIN_WEIGHT_ELEMS = 64 * 1024 * 1024
FP4_PARALLEL_GEN_MIN_ELEMS = 8 * 1024 * 1024


def _recipe_example_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(here) == "scripts":
        return os.path.dirname(here)
    return here


def parse_group_m_list(arg: str) -> List[int]:
    values: List[int] = []
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
        group_m_arr = RANDOM_GENERATOR.integers(low, high + 1, size=group_num)
        if int(group_m_arr.sum()) <= m:
            return group_m_arr.astype(int).tolist()

    group_m_list = [low] * group_num
    remaining = m - sum(group_m_list)
    if remaining <= 0:
        return group_m_list

    capacities = [high - low for _ in range(group_num)]
    order = RANDOM_GENERATOR.permutation(group_num).tolist()
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


def pack_b4_to_b8(b4_data: np.ndarray):
    # Two fp4 values are packed into one byte: low nibble + high nibble.
    b4_i8 = b4_data.view(np.int8)
    if b4_i8.shape[-1] % 2 == 0:
        low = np.bitwise_and(b4_i8[..., 0::2], 0x0F)
        high = np.bitwise_and(b4_i8[..., 1::2], 0x0F)
    else:
        pad_width = [(0, 0)] * b4_i8.ndim
        pad_width[-1] = (0, 1)
        padded = np.pad(b4_i8, pad_width, mode="constant")
        low = np.bitwise_and(padded[..., 0::2], 0x0F)
        high = np.bitwise_and(padded[..., 1::2], 0x0F)
    return np.bitwise_or(low, np.left_shift(high, 4)).astype(np.int8, copy=False)


def fp4_to_float32(data: np.ndarray) -> np.ndarray:
    return FP4_E2M1FN_TO_F32_LUT[np.bitwise_and(data.view(np.int8), 0x0F)]


def random_uniform_float32(low: float, high: float, size) -> np.ndarray:
    return RANDOM_GENERATOR.random(size, dtype=np.float32) * (high - low) + low


def random_uniform_fp4_parallel(group_num: int, n_size: int, k_size: int, low: float, high: float) -> np.ndarray:
    total_elems = group_num * n_size * k_size
    max_workers = os.cpu_count() or 1
    worker_num = min(group_num, max(1, min(10, max_workers)))
    if total_elems < FP4_PARALLEL_GEN_MIN_ELEMS or worker_num <= 1:
        return random_uniform_float32(low=low, high=high, size=(group_num, n_size, k_size)).astype(float4_e2m1fn)

    chunk_sizes = [group_num // worker_num] * worker_num
    for i in range(group_num % worker_num):
        chunk_sizes[i] += 1

    seeds = RANDOM_GENERATOR.integers(0, np.iinfo(np.int64).max, size=worker_num, dtype=np.int64)
    scale = high - low

    def _worker(seed_and_size: Tuple[int, int]) -> np.ndarray:
        seed, chunk_group_num = seed_and_size
        local_rng = np.random.default_rng(seed)
        return (local_rng.random((chunk_group_num, n_size, k_size), dtype=np.float32) * scale + low).astype(
            float4_e2m1fn
        )

    with ThreadPoolExecutor(max_workers=worker_num) as executor:
        parts = list(executor.map(_worker, zip(seeds.tolist(), chunk_sizes)))
    return np.concatenate(parts, axis=0)


def trans_nd2nz(input_data):
    g, n_pad, k_pad = input_data.shape
    return input_data.reshape(g, n_pad // 16, 16, k_pad // 32, 32).transpose(0, 3, 1, 2, 4)


def trans_nz2nd(input_data):
    return input_data.transpose(0, 2, 3, 1, 4).reshape(
        input_data.shape[0], input_data.shape[2] * input_data.shape[3], input_data.shape[1] * input_data.shape[4]
    )


def expand_scale_ceil_k64(scale_data, k_size):
    scale_flat = scale_data.astype(np.float32).reshape(scale_data.shape[0], -1)
    scale_k = np.repeat(scale_flat, 32, axis=-1)
    return scale_k[:, :k_size]


def expand_weight_scale_ceil_k64(scale_data, k_size):
    scale_flat = scale_data.astype(np.float32).reshape(scale_data.shape[0], scale_data.shape[1], -1)
    scale_k = np.repeat(scale_flat, 32, axis=-1)
    return scale_k[:, :, :k_size]


def build_group_list(group_m_list: List[int]) -> np.ndarray:
    return np.array(group_m_list, dtype=np.int64)


def write_artifacts(base_dir, x, weight, x_scale, weight_scale, group_list, out):
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    x.tofile(os.path.join(input_dir, "input_a.bin"))
    weight.tofile(os.path.join(input_dir, "input_b.bin"))
    x_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    weight_scale.tofile(os.path.join(input_dir, "input_scaleB.bin"))
    group_list.tofile(os.path.join(input_dir, "input_groupList.bin"))
    out.tofile(os.path.join(output_dir, "output_cpu.bin"))


def float32_to_bf16_u16(data: np.ndarray) -> np.ndarray:
    data_u32 = data.astype(np.float32, copy=False).view(np.uint32)
    lsb = (data_u32 >> 16) & 1
    rounded = data_u32 + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype(np.uint16, copy=False)


def _cal_weight_quant_mx_numpy(x1, x2_nd, antiquant_scale, pertoken_scale, group_list, y_dtype, trans_b, n):
    x2_valid = x2_nd[:, :n, :]
    shape_k = x2_valid.shape[-1]
    antiquant_scale_broadcast = expand_weight_scale_ceil_k64(antiquant_scale, shape_k)
    pertoken_scale_broadcast = expand_scale_ceil_k64(pertoken_scale, shape_k)
    x1_dequant = x1.astype(np.float32) * pertoken_scale_broadcast

    out_f32 = np.zeros((x1.shape[0], n), dtype=np.float32)
    m_offset = 0
    for i, group_m in enumerate(group_list):
        if group_m == 0:
            continue
        end_m = m_offset + group_m
        x1_group = x1_dequant[m_offset:end_m]
        x2_group_f32 = fp4_to_float32(x2_valid[i])
        scaled_weight = x2_group_f32 * antiquant_scale_broadcast[i]
        if trans_b:
            group_out = x1_group @ scaled_weight.transpose(1, 0)
        else:
            group_out = x1_group @ scaled_weight
        out_f32[m_offset:end_m] = group_out
        m_offset = end_m

    if y_dtype == "bfloat16":
        return float32_to_bf16_u16(out_f32)
    return out_f32.astype(np.float16)


def _cal_weight_quant_mx_torch(x1, x2_nd, antiquant_scale, pertoken_scale, group_list, y_dtype, trans_b, n):
    import torch

    x1_t = torch.from_numpy(x1.astype(np.float32))
    x2 = fp4_to_float32(x2_nd[:, :n, :])
    x2_t = torch.from_numpy(x2)

    shape_k = x2_t.shape[-1]
    antiquant_scale_broadcast = torch.from_numpy(expand_weight_scale_ceil_k64(antiquant_scale, shape_k))
    pertoken_scale_broadcast = torch.from_numpy(expand_scale_ceil_k64(pertoken_scale, shape_k))
    x1_dequant = x1_t * pertoken_scale_broadcast

    weight_dequant = x2_t * antiquant_scale_broadcast
    if trans_b:
        weight_dequant = weight_dequant.transpose(1, 2)

    if y_dtype == "bfloat16":
        out_t = torch.zeros((x1.shape[0], n), dtype=torch.bfloat16)
    else:
        out_t = torch.zeros((x1.shape[0], n), dtype=torch.float16)
    m_offset = 0
    for i, group_m in enumerate(group_list):
        if group_m == 0:
            continue
        end_m = m_offset + group_m
        x1_group = x1_dequant[m_offset:end_m, :]
        weight_group = weight_dequant[i]
        group_out = torch.matmul(x1_group, weight_group)
        if y_dtype == "bfloat16":
            out_t[m_offset:end_m] = group_out.to(torch.bfloat16)
        else:
            out_t[m_offset:end_m] = group_out.to(torch.float16)
        m_offset = end_m

    if y_dtype == "bfloat16":
        return out_t.view(torch.uint16).cpu().numpy()
    return out_t.cpu().numpy().astype(np.float16)


def cal_weight_quant_mx(x1, x2_nd, antiquant_scale, pertoken_scale, group_list, y_dtype, trans_b, n):
    weight_elems = x2_nd.shape[0] * n * x2_nd.shape[-1]
    if weight_elems >= TORCH_BACKEND_MIN_WEIGHT_ELEMS:
        try:
            return _cal_weight_quant_mx_torch(
                x1, x2_nd, antiquant_scale, pertoken_scale, group_list, y_dtype, trans_b, n
            )
        except ModuleNotFoundError:
            # Fall back to NumPy path in environments without available torch runtime.
            return _cal_weight_quant_mx_numpy(
                x1, x2_nd, antiquant_scale, pertoken_scale, group_list, y_dtype, trans_b, n
            )
    return _cal_weight_quant_mx_numpy(x1, x2_nd, antiquant_scale, pertoken_scale, group_list, y_dtype, trans_b, n)


def gen_data(group_list: List[int], m: int, k: int, n: int):
    g = len(group_list)
    x1_dtype = "float8_e4m3fn"
    x2_dtype = "float4_e2m1fn"
    y_dtype = "bfloat16"
    trans_b = True
    group_size = 32

    if x1_dtype != "float8_e4m3fn":
        raise ValueError("Only x1_dtype=float8_e4m3fn is supported")
    if x2_dtype != "float4_e2m1fn":
        raise ValueError("Only x2_dtype=float4_e2m1fn is supported")
    if k % (group_size * 2) != 0:
        raise ValueError("k must be a multiple of group_size * 2 for fp4 packed data")

    n1 = math.ceil(n / 16)
    k1 = math.ceil(k / 32)
    k_scale = math.ceil(k / 64)
    n_pad = n1 * 16
    k_pad = k1 * 32

    x1 = random_uniform_float32(low=-3.0, high=3.0, size=(m, k)).astype(float8_e4m3fn)
    x2_valid = random_uniform_fp4_parallel(group_num=g, n_size=n, k_size=k, low=-3.0, high=3.0)
    if n_pad == n and k_pad == k:
        x2_nd = x2_valid
    else:
        x2_nd = np.zeros((g, n_pad, k_pad), dtype=float4_e2m1fn)
        x2_nd[:, :n, :k] = x2_valid
    x2_nz = trans_nd2nz(x2_nd)
    x2 = pack_b4_to_b8(x2_nz)
    antiquant_scale = random_uniform_float32(low=0.5, high=2.0, size=(g, n, k_scale, 2)).astype(float8_e8m0)
    pertoken_scale = random_uniform_float32(low=0.5, high=2.0, size=(m, k_scale, 2)).astype(float8_e8m0)

    out = cal_weight_quant_mx(x1, x2_nd, antiquant_scale, pertoken_scale, group_list, y_dtype, trans_b, n)

    base_dir = _recipe_example_root()
    write_artifacts(base_dir, x1, x2, pertoken_scale, antiquant_scale, build_group_list(group_list), out)


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
    if k % 64 != 0:
        print("k must be a multiple of 64")
        sys.exit(1)

    sum_group_m = sum(group_m_list)
    if m < sum_group_m:
        print(f"m must be greater than or equal to sum(group_m_list)={sum_group_m}")
        sys.exit(1)

    print(f"group_m_list={','.join(str(value) for value in group_m_list)}")
    print(f"m={m}, sum(group_m_list)={sum_group_m}, k={k}, n={n}")
    gen_data(group_m_list, m, k, n)
