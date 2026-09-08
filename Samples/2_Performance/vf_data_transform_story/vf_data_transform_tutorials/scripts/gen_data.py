#!/usr/bin/python3
# coding=utf-8

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import json
import os

import ml_dtypes
import numpy as np


def make_linear_storage(input_count, input_dtype, output_storage_count, output_dtype, sentinel):
    input_data = np.zeros(input_count, dtype=input_dtype)
    output_init = np.full(output_storage_count, sentinel, dtype=output_dtype)
    return input_data, output_init


def generate_half_to_int8(args):
    total = args.m * args.n
    input_data, output_init = make_linear_storage(
        args.input_count, np.float16, args.output_storage_count, np.int8, -91
    )
    index = np.arange(total, dtype=np.uint64)
    fractions = np.array([-0.75, -0.5, -0.49, 0.0, 0.49, 0.5, 0.75], dtype=np.float32)
    integral = (index % 241).astype(np.int32) - 120
    values = (integral.astype(np.float32) + fractions[(index // 241) % fractions.size]).astype(np.float16)
    input_data[:total] = values
    golden = np.rint(values).astype(np.int8)
    return input_data, output_init, golden


def generate_fp32_to_fp8(args):
    total = args.m * args.n
    input_data, output_init = make_linear_storage(
        args.input_count, np.float32, args.output_storage_count, np.uint8, 0xA5
    )
    values = np.array(
        [0.0, 0.5, -0.5, 1.0, -1.0, 1.0625, 1.1875, 2.25, -3.5, 15.5, 32.0, -64.0, 240.0],
        dtype=np.float32,
    )
    logical_input = values[np.arange(total, dtype=np.uint64) % values.size]
    input_data[:total] = logical_input
    golden = logical_input.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return input_data, output_init, golden


def generate_int8_to_half(args):
    total = args.m * args.n
    input_data, output_init = make_linear_storage(
        args.input_count, np.int8, args.output_storage_count, np.uint16, 0x7BFF
    )
    index = np.arange(total, dtype=np.uint64)
    lane = index % 256
    group_phase = ((index // 256) * 73) % 256
    values = ((lane + group_phase) % 256).astype(np.int16) - 128
    logical_input = values.astype(np.int8)
    input_data[:total] = logical_input
    golden = logical_input.astype(np.float16).view(np.uint16)
    return input_data, output_init, golden


def pack_nibbles(codes, input_count, padding):
    packed = np.full(input_count, padding, dtype=np.uint8)
    low = codes[0::2]
    high = codes[1::2]
    packed[: low.size] = (packed[: low.size] & np.uint8(0xF0)) | low
    packed[: high.size] = (packed[: high.size] & np.uint8(0x0F)) | (high << np.uint8(4))
    return packed


def generate_int4_to_bf16(args):
    total = args.m * args.n
    index = np.arange(total, dtype=np.uint64)
    lane = index % 128
    group_phase = ((index // 128) * 7) % 16
    values = ((lane * 5 + (lane // 16) * 3 + group_phase) % 16).astype(np.int16) - 8
    codes = (values & 0x0F).astype(np.uint8)
    input_data = pack_nibbles(codes, args.input_count, 0x55)
    output_init = np.full(args.output_storage_count, 0x7FC1, dtype=np.uint16)
    value_bits = values.astype(np.float32).view(np.uint32)
    golden = (value_bits >> np.uint32(16)).astype(np.uint16)
    return input_data, output_init, golden


def generate_fp4_to_fp8(args):
    total = args.m * args.n
    index = np.arange(total, dtype=np.uint64)
    lane = index % 256
    group_phase = ((index // 256) * 7) % 16
    codes = ((lane + group_phase) % 16).astype(np.uint8)
    input_data = pack_nibbles(codes, args.input_count, 0x00)
    output_init = np.full(args.output_storage_count, 0x7F, dtype=np.uint8)
    lookup = np.array(
        [
            0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C,
            0x80, 0x84, 0x88, 0x8C, 0x90, 0x94, 0x98, 0x9C,
        ],
        dtype=np.uint8,
    )
    golden = lookup[codes]
    return input_data, output_init, golden


def generate_bf16_input(args):
    total = args.m * args.n
    input_data = np.zeros(args.input_count, dtype=np.uint16)
    rows = np.arange(args.m, dtype=np.uint32).reshape(-1, 1)
    columns = np.arange(args.n, dtype=np.uint32).reshape(1, -1)
    mixed = rows * np.uint32(131) + columns * np.uint32(17) + (rows // np.uint32(7)) * np.uint32(29)
    sign = ((mixed & np.uint32(1)) << np.uint32(15)).astype(np.uint16)
    exponent = ((np.uint32(120) + mixed % np.uint32(15)) << np.uint32(7)).astype(np.uint16)
    mantissa = ((mixed >> np.uint32(4)) & np.uint32(0x7F)).astype(np.uint16)
    logical_input = sign | exponent | mantissa
    input_data[:total] = logical_input.reshape(-1)
    return input_data, logical_input


def fill_rope_padding(golden_nz, out_nd_bf16, m):
    tile_rows = 96
    remainder = m % tile_rows
    if remainder == 0:
        return
    copy_rows = (remainder + 15) // 16 * 16
    if copy_rows <= remainder:
        return
    tile_start = (m // tile_rows) * tile_rows
    for r in range(remainder, copy_rows):
        for q in range(8):
            golden_nz[q, tile_start + r, :] = out_nd_bf16[tile_start + r, q * 16:q * 16 + 16]


def generate_bf16_nd2nz(args):
    if args.n != 128:
        raise ValueError("bf16_nd2nz requires n=128")
    input_data, logical_input = generate_bf16_input(args)

    output_init = np.full(args.output_storage_count, 0x7FC1, dtype=np.uint16)
    aligned_rows = ((args.m + 15) // 16) * 16
    golden_nz = np.zeros((8, aligned_rows, 16), dtype=np.uint16)
    golden_nz[:, :args.m, :] = logical_input.reshape(args.m, 8, 16).transpose(1, 0, 2)
    golden = golden_nz.reshape(-1)
    return input_data, output_init, golden


def fp32_to_bf16_bits(fp32_arr):
    u32 = fp32_arr.astype(np.float32).view(np.uint32).reshape(-1)
    rounding_bias = ((u32 >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u32 + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def compute_rope_output(logical_input, args, padded_rows):
    positions = np.arange(args.m, dtype=np.float32).reshape(-1, 1)
    dim_indices = np.arange(args.n // 2, dtype=np.float32).reshape(1, -1)
    inv_freq = np.power(10000.0, -dim_indices / np.float32(args.n // 2))
    angles = positions * inv_freq
    cos_half = np.cos(angles).astype(np.float32)
    sin_half = np.sin(angles).astype(np.float32)

    cos_padded = np.zeros((padded_rows, args.n // 2), dtype=np.float32)
    cos_padded[:args.m, :] = cos_half[:args.m, :]
    sin_padded = np.zeros((padded_rows, args.n // 2), dtype=np.float32)
    sin_padded[:args.m, :] = sin_half[:args.m, :]

    x_bf16 = np.zeros((padded_rows, args.n), dtype=np.float32)
    x_bf16[:args.m, :] = (logical_input.reshape(args.m, args.n).astype(np.uint32) << np.uint32(16)).view(np.float32)
    out_first = x_bf16[:, :args.n // 2] * cos_padded - x_bf16[:, args.n // 2:] * sin_padded
    out_second = x_bf16[:, args.n // 2:] * cos_padded + x_bf16[:, :args.n // 2] * sin_padded

    out_nd_bf16 = np.zeros((padded_rows, args.n), dtype=np.uint16)
    out_nd_bf16[:, :args.n // 2] = fp32_to_bf16_bits(out_first).reshape(padded_rows, -1)
    out_nd_bf16[:, args.n // 2:] = fp32_to_bf16_bits(out_second).reshape(padded_rows, -1)
    return out_nd_bf16, sin_padded, cos_padded


def generate_bf16_rope_nd2nz(args):
    if args.n != 128:
        raise ValueError("bf16_rope_nd2nz requires n=128")

    input_data, logical_input = generate_bf16_input(args)
    padded_rows = args.input_count // args.n
    out_nd_bf16, sin_padded, cos_padded = compute_rope_output(logical_input, args, padded_rows)

    output_init = np.full(args.output_storage_count, 0x7FC1, dtype=np.uint16)
    aligned_rows = ((args.m + 15) // 16) * 16
    golden_nz = np.full((8, aligned_rows, 16), 0x7FC1, dtype=np.uint16)
    golden_nz[:, :args.m, :] = out_nd_bf16[:args.m, :].reshape(args.m, 8, 16).transpose(1, 0, 2)
    fill_rope_padding(golden_nz, out_nd_bf16, args.m)
    golden = golden_nz.reshape(-1).copy()

    extra_inputs = {
        "sin": sin_padded.reshape(-1).copy(),
        "cos": cos_padded.reshape(-1).copy(),
    }
    return input_data, output_init, golden, extra_inputs


GENERATORS = {
    "half_to_int8": generate_half_to_int8,
    "fp32_to_fp8": generate_fp32_to_fp8,
    "int8_to_half": generate_int8_to_half,
    "int4_to_bf16": generate_int4_to_bf16,
    "fp4_to_fp8": generate_fp4_to_fp8,
    "bf16_nd2nz": generate_bf16_nd2nz,
    "bf16_rope_nd2nz": generate_bf16_rope_nd2nz,
}


def write_artifacts(args):
    result = GENERATORS[args.task](args)
    extra_inputs = {}
    if len(result) == 4:
        input_data, output_init, golden, extra_inputs = result
    else:
        input_data, output_init, golden = result
    if input_data.size != args.input_count:
        raise ValueError(f"input count mismatch: expected {args.input_count}, generated {input_data.size}")
    if output_init.size != args.output_storage_count:
        raise ValueError(
            f"output init count mismatch: expected {args.output_storage_count}, generated {output_init.size}"
        )
    if golden.size != args.output_count:
        raise ValueError(f"golden count mismatch: expected {args.output_count}, generated {golden.size}")
    if args.output_offset + args.output_count > args.output_storage_count:
        raise ValueError("output payload exceeds output storage")
    golden_storage = output_init.copy()
    golden_storage[args.output_offset:args.output_offset + args.output_count] = golden

    input_dir = os.path.join(args.output_dir, "input")
    output_dir = os.path.join(args.output_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    input_data.tofile(os.path.join(input_dir, "input.bin"))
    output_init.tofile(os.path.join(input_dir, "output_init.bin"))
    for name, data in extra_inputs.items():
        data.tofile(os.path.join(input_dir, f"{name}.bin"))
    golden.tofile(os.path.join(output_dir, "golden.bin"))
    golden_storage.tofile(os.path.join(output_dir, "golden_storage.bin"))
    metadata = {
        "task": args.task,
        "case": args.case_name,
        "m": args.m,
        "n": args.n,
        "input_count": args.input_count,
        "output_storage_count": args.output_storage_count,
        "output_offset": args.output_offset,
        "output_count": args.output_count,
        "output_dtype": str(golden.dtype),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")
    extra_str = f", extra_inputs={list(extra_inputs.keys())}" if extra_inputs else ""
    print(
        f"[GENERATE][{args.task}/{args.case_name}] input={input_data.size} {input_data.dtype}, "
        f"golden={golden.size} {golden.dtype}, output_dir={args.output_dir}{extra_str}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate input and bit-exact golden binaries.")
    parser.add_argument("--task", choices=sorted(GENERATORS), required=True)
    parser.add_argument("--case", dest="case_name", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--input-count", type=int, required=True)
    parser.add_argument("--output-storage-count", type=int, required=True)
    parser.add_argument("--output-offset", type=int, default=0)
    parser.add_argument("--output-count", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    counts = [args.m, args.n, args.input_count, args.output_storage_count, args.output_count]
    if any(value <= 0 for value in counts) or args.output_offset < 0:
        parser.error("shapes and storage counts must be positive; output offset must be non-negative")
    return args


if __name__ == "__main__":
    write_artifacts(parse_args())
