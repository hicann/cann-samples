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

import argparse
import os

import numpy as np


def float32_to_bf16(values: np.ndarray) -> np.ndarray:
    src = values.astype(np.float32, copy=False).view(np.uint32)
    lsb = (src >> 16) & 1
    rounded = src + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype(np.uint16)


def bf16_to_float32(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.uint32) << 16).view(np.float32)


def write_bin(path: str, data: np.ndarray):
    data.tofile(path)


def build_golden(kv_bf16, gamma_bf16, cos_bf16, sin_bf16, index, batch, seq, dv, dk, epsilon):
    kv = bf16_to_float32(kv_bf16).reshape(batch, 1, seq, dv + dk)
    gamma = bf16_to_float32(gamma_bf16).reshape(dv)
    cos = bf16_to_float32(cos_bf16).reshape(batch, 1, seq, dk)
    sin = bf16_to_float32(sin_bf16).reshape(batch, 1, seq, dk)

    rms_x = kv[..., :dv]
    rope_x = kv[..., dv:]

    mean_square = np.mean(rms_x * rms_x, axis=-1, keepdims=True)
    rms = np.sqrt(mean_square + epsilon)
    v_out = (rms_x / rms) * gamma

    real = rope_x[..., 0::2]
    imag = rope_x[..., 1::2]
    part1 = np.concatenate([real, imag], axis=-1)
    part2 = np.concatenate([-imag, real], axis=-1)
    k_out = part1 * cos + part2 * sin

    k_out_bf16 = float32_to_bf16(k_out)
    v_out_bf16 = float32_to_bf16(v_out)

    # Norm cache: [B, N, Scache, D], index is [B, S]. Initialize with zeros.
    k_cache = np.zeros((batch, 1, seq, dk), dtype=np.uint16)
    v_cache = np.zeros((batch, 1, seq, dv), dtype=np.uint16)
    flat_index = index.reshape(batch, seq)
    for b in range(batch):
        for s in range(seq):
            cache_idx = flat_index[b, s]
            if cache_idx >= 0:
                k_cache[b, 0, cache_idx, :] = k_out_bf16[b, 0, s, :]
                v_cache[b, 0, cache_idx, :] = v_out_bf16[b, 0, s, :]

    return k_cache, v_cache, k_out_bf16, v_out_bf16


def generate(args):
    if args.dk % 2 != 0:
        raise ValueError("dk must be even.")

    rng = np.random.default_rng(args.seed)
    input_dir = os.path.join(args.output, "input")
    output_dir = os.path.join(args.output, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    kv = rng.uniform(-0.7, 0.7, size=(args.batch, 1, args.seq, args.dv + args.dk)).astype(np.float32)
    gamma = rng.uniform(0.4, 1.2, size=(args.dv,)).astype(np.float32)
    cos = rng.uniform(-1.0, 1.0, size=(args.batch, 1, args.seq, args.dk)).astype(np.float32)
    sin = rng.uniform(-1.0, 1.0, size=(args.batch, 1, args.seq, args.dk)).astype(np.float32)
    index = np.tile(np.arange(args.seq, dtype=np.int64), args.batch)

    kv_bf16 = float32_to_bf16(kv)
    gamma_bf16 = float32_to_bf16(gamma)
    cos_bf16 = float32_to_bf16(cos)
    sin_bf16 = float32_to_bf16(sin)

    k_cache_init = np.zeros((args.batch, 1, args.seq, args.dk), dtype=np.uint16)
    v_cache_init = np.zeros((args.batch, 1, args.seq, args.dv), dtype=np.uint16)
    k_cache_golden, v_cache_golden, k_out_golden, v_out_golden = build_golden(
        kv_bf16, gamma_bf16, cos_bf16, sin_bf16, index, args.batch, args.seq, args.dv, args.dk, args.epsilon
    )

    write_bin(os.path.join(input_dir, "kv.bin"), kv_bf16.reshape(-1))
    write_bin(os.path.join(input_dir, "gamma.bin"), gamma_bf16.reshape(-1))
    write_bin(os.path.join(input_dir, "cos.bin"), cos_bf16.reshape(-1))
    write_bin(os.path.join(input_dir, "sin.bin"), sin_bf16.reshape(-1))
    write_bin(os.path.join(input_dir, "index.bin"), index.reshape(-1))
    write_bin(os.path.join(input_dir, "k_cache.bin"), k_cache_init.reshape(-1))
    write_bin(os.path.join(input_dir, "v_cache.bin"), v_cache_init.reshape(-1))

    write_bin(os.path.join(output_dir, "k_cache_golden.bin"), k_cache_golden.reshape(-1))
    write_bin(os.path.join(output_dir, "v_cache_golden.bin"), v_cache_golden.reshape(-1))
    write_bin(os.path.join(output_dir, "k_out_golden.bin"), k_out_golden.reshape(-1))
    write_bin(os.path.join(output_dir, "v_out_golden.bin"), v_out_golden.reshape(-1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--dv", type=int, required=True)
    parser.add_argument("--dk", type=int, required=True)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
