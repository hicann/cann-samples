#!/usr/bin/env python3
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

"""Kimi Delta Attention Lite 输入数据生成器.

使用 NumPy 生成 Q/K/V、log_decay 和 beta. Q/K 会先完成 L2Norm,
Q 额外乘 1/sqrt(D). BF16 转换优先使用 ml_dtypes, 否则使用
round-to-nearest-even 位运算.

用法:
    python3 kimi_delta_attn_lite_gendata.py <data_dir> <B> <S> <D>

产出文件不保存固定为 1 的 Head 轴:
    q/k/v.bin       BF16 [B,S,D]
    log_decay.bin   FP32 [B,S,D]
    beta.bin        BF16 [B,S]

环境变量:
    KDA_DATA_CASE = random|beta_zero|beta_one|no_decay|strong_decay|mixed_decay
"""

import os
import sys
import time

from kdalite_thread_limit import configure_python_threads

try:
    configure_python_threads()
except ValueError as error:
    print(f"错误: {error}", file=sys.stderr)
    sys.exit(1)

import numpy as np

try:
    import ml_dtypes

    _HAS_ML_DTYPES = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_ML_DTYPES = False


RANDOM_SEED = 20260805
SUPPORTED_CASES = ("random", "beta_zero", "beta_one", "no_decay", "strong_decay", "mixed_decay")


def fp32_to_bf16(value: np.ndarray) -> np.ndarray:
    """Convert FP32 to little-endian BF16 with round-to-nearest-even."""
    value = np.ascontiguousarray(value, dtype=np.float32)
    if _HAS_ML_DTYPES:
        return value.astype(ml_dtypes.bfloat16)
    bits = value.view(np.uint32).copy()
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = (bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)
    return rounded.astype("<u2")


def normalize_qk(value: np.ndarray, q_scale: float = 1.0) -> np.ndarray:
    """Normalize with sum(x*x)+1e-6 and optionally scale Q."""
    value = np.ascontiguousarray(value, dtype=np.float32)
    norm = np.einsum("...d,...d->...", value, value, dtype=np.float32, optimize=False)[..., None]
    norm += np.float32(1.0e-6)
    np.sqrt(norm, out=norm)
    np.reciprocal(norm, out=norm)
    value *= norm
    value *= np.float32(q_scale)
    return value


def write_bf16(path: str, value: np.ndarray) -> None:
    fp32_to_bf16(value).tofile(path)


def write_fp32(path: str, value: np.ndarray) -> None:
    np.ascontiguousarray(value, dtype="<f4").tofile(path)


def main() -> int:
    if len(sys.argv) != 5:
        print(f"用法: {sys.argv[0]} <data_dir> <B> <S> <D>", file=sys.stderr)
        return 1

    try:
        batch_size = int(sys.argv[2])
        seq_len = int(sys.argv[3])
        dim = int(sys.argv[4])
    except ValueError:
        print("错误: B/S/D 必须是整数", file=sys.stderr)
        return 1
    if batch_size <= 0 or seq_len <= 0 or dim <= 0:
        print(
            f"错误: B/S/D 必须为正整数，得到 B={batch_size} S={seq_len} D={dim}",
            file=sys.stderr,
        )
        return 1
    if dim != 128:
        print(f"错误: KDALite 固定 D=128，得到 D={dim}", file=sys.stderr)
        return 1

    data_case = os.environ.get("KDA_DATA_CASE", "random").strip().lower()
    if data_case not in SUPPORTED_CASES:
        print(
            f"错误: KDA_DATA_CASE={data_case!r}，可选值为 {', '.join(SUPPORTED_CASES)}",
            file=sys.stderr,
        )
        return 1

    if not _HAS_ML_DTYPES:
        print("gendata: 未装 ml_dtypes，bf16 走位运算回退", file=sys.stderr)

    data_dir = sys.argv[1]
    rng = np.random.default_rng(RANDOM_SEED)
    shape = (batch_size, seq_len, dim)
    start = time.perf_counter()
    try:
        os.makedirs(data_dir, exist_ok=True)

        # Generate and write large tensors one by one to limit peak host memory.
        raw_q = rng.standard_normal(shape, dtype=np.float32)
        q = normalize_qk(raw_q, 1.0 / np.sqrt(float(dim)))
        write_bf16(os.path.join(data_dir, "q.bin"), q)
        del raw_q, q

        raw_k = rng.standard_normal(shape, dtype=np.float32)
        k = normalize_qk(raw_k)
        write_bf16(os.path.join(data_dir, "k.bin"), k)
        del raw_k, k

        value = rng.standard_normal(shape, dtype=np.float32)
        write_bf16(os.path.join(data_dir, "v.bin"), value)
        del value

        # Kimi K3 log decay is non-positive and independent across Dk channels.
        log_decay = rng.random(shape, dtype=np.float32)
        log_decay *= np.float32(-0.049)
        log_decay -= np.float32(0.001)
        if data_case == "no_decay":
            log_decay.fill(0)
        elif data_case == "strong_decay":
            # -5 is the post-gate lower bound used by Kimi K3.
            log_decay.fill(np.float32(-5.0))
        elif data_case == "mixed_decay":
            token_index = np.arange(seq_len, dtype=np.int32)[None, :, None]
            channel_index = np.arange(dim, dtype=np.int32)[None, None, :]
            log_decay = np.where((token_index + channel_index) % 2 == 0, -5.0, 0.0).astype(np.float32)
            log_decay = np.broadcast_to(log_decay, shape).copy()
        write_fp32(os.path.join(data_dir, "log_decay.bin"), log_decay)
        del log_decay

        beta_logits = rng.standard_normal((batch_size, seq_len), dtype=np.float32)
        beta = np.reciprocal(np.float32(1.0) + np.exp(-beta_logits))
        if data_case == "beta_zero":
            beta.fill(0)
        elif data_case == "beta_one":
            beta.fill(1)
        write_bf16(os.path.join(data_dir, "beta.bin"), beta)
    except (OSError, ValueError, MemoryError) as error:
        print(f"生成输入失败：{error}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - start
    bf16_backend = "ml_dtypes" if _HAS_ML_DTYPES else "位运算回退"
    total_bytes = batch_size * seq_len * (10 * dim + 2)
    print(
        f"gendata: B={batch_size} S={seq_len} D={dim} case={data_case} -> "
        f"{{q,k,v,log_decay,beta}}.bin "
        f"(bf16/{bf16_backend}, seed={RANDOM_SEED}, total={total_bytes} bytes) {elapsed:.3f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
