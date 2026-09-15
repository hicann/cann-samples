#!/usr/bin/env python3
# coding=utf-8

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate generalized BlockAttnRes inputs and FP32/BF16 golden outputs."""

import argparse
import os

import numpy as np


FP32_LOWEST = np.float32(-3.4028234663852886e38)


def fp32_to_bf16(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def bf16_to_fp32(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def save(path: str, value: np.ndarray) -> None:
    np.ascontiguousarray(value).tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--t", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--s", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--valid-blocks", type=int, required=True)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.t <= 0 or args.s <= 0:
        raise ValueError("requires T,S>0")
    if not 1 <= args.n <= 64:
        raise ValueError("requires 1<=N<=64")
    if not 1 <= args.d <= 8192:
        raise ValueError("requires 1<=D<=8192")
    if not 0 <= args.valid_blocks <= np.iinfo(np.uint64).max:
        raise ValueError("valid_blocks must fit uint64")
    if not 0 <= args.slot < args.s:
        raise ValueError("requires 0<=slot<S")
    if not np.isfinite(args.eps) or args.eps <= 0:
        raise ValueError("eps must be finite and positive")

    input_dir = os.path.join(args.output, "input")
    output_dir = os.path.join(args.output, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    block_res = rng.normal(0.0, 0.25, (args.t, args.n, args.d)).astype(np.float32)
    pseudo_query = rng.normal(0.0, 0.20, (args.s, args.d)).astype(np.float32)
    partial_block = rng.normal(0.0, 0.30, (args.t, args.d)).astype(np.float32)
    delta = fp32_to_bf16(rng.normal(0.0, 0.05, (args.t, args.d)).astype(np.float32))
    valid_blocks = np.array([args.valid_blocks], dtype=np.uint64)
    valid_n = min(args.valid_blocks, args.n)

    if valid_n == 0:
        numerator = np.zeros((args.s, args.t, args.d), dtype=np.float32)
        logit_max = np.full((args.s, args.t), FP32_LOWEST, dtype=np.float32)
        exp_sum = np.zeros((args.s, args.t), dtype=np.float32)
    else:
        valid_res = block_res[:, :valid_n, :]
        inv_rms = np.float32(1.0) / np.sqrt(
            np.mean(valid_res * valid_res, axis=-1, dtype=np.float32) + np.float32(args.eps))
        logits = np.einsum("sd,tnd->stn", pseudo_query, valid_res, dtype=np.float32) * inv_rms[None, :, :]
        logit_max = np.max(logits, axis=-1).astype(np.float32)
        weights = np.exp(logits - logit_max[..., None]).astype(np.float32)
        exp_sum = np.sum(weights, axis=-1, dtype=np.float32)
        numerator = np.einsum("stn,tnd->std", weights, valid_res, dtype=np.float32).astype(np.float32)

    updated_partial = (partial_block + bf16_to_fp32(delta)).astype(np.float32)
    if valid_n == 0:
        h = np.zeros((args.t, args.d), dtype=np.uint16)
    else:
        # Match BlockAttnResUpdateTestSpec.golden's FP32 reduction order.
        square_sum = np.sum(updated_partial * updated_partial, axis=-1, dtype=np.float32)
        update_rms = np.sqrt(square_sum * np.float32(1.0 / args.d) + np.float32(args.eps))
        dot_sum = np.sum(updated_partial * pseudo_query[args.slot], axis=-1, dtype=np.float32)
        score = dot_sum / update_rms
        current_max = np.maximum(logit_max[args.slot], score)
        alpha = np.exp(logit_max[args.slot] - current_max).astype(np.float32)
        beta = np.exp(score - current_max).astype(np.float32)
        denominator = exp_sum[args.slot] * alpha + beta
        h_fp32 = numerator[args.slot] * (alpha / denominator)[:, None] + updated_partial * (beta / denominator)[:, None]
        h = fp32_to_bf16(h_fp32)

    save(os.path.join(input_dir, "block_res.bin"), block_res)
    save(os.path.join(input_dir, "valid_blocks.bin"), valid_blocks)
    save(os.path.join(input_dir, "pseudo_query.bin"), pseudo_query)
    save(os.path.join(input_dir, "partial_block.bin"), partial_block)
    save(os.path.join(input_dir, "delta.bin"), delta)
    save(os.path.join(output_dir, "golden_prepare_numerator.bin"), numerator)
    save(os.path.join(output_dir, "golden_prepare_logit_max.bin"), logit_max)
    save(os.path.join(output_dir, "golden_prepare_exp_sum.bin"), exp_sum)
    save(os.path.join(output_dir, "golden_update_partial.bin"), updated_partial)
    save(os.path.join(output_dir, "golden_update_h.bin"), h)


if __name__ == "__main__":
    main()
