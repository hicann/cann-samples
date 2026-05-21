# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Generate HiFloat8 TC inputs and CPU golden under ./input and ./output (per-channel FP32 scale on disk).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from en_dtypes import hifloat8


def _golden_matmul_hifloat8(m: int, k: int, n: int, transpose_a: bool, transpose_b: bool):
    """
    Storage matches matmul_a16w16/scripts/gen_data.py:
    x1 disk: (K, M) if transpose_a else (M, K); x2 disk: (N, K) if transpose_b else (K, N).
    Computes matmul(x1_eff, x2_eff), x1_eff (M, K), x2_eff (K, N).
    """
    x1_shape = (k, m) if transpose_a else (m, k)
    x2_shape = (n, k) if transpose_b else (k, n)

    x1_u8 = np.random.randint(8, 30, size=x1_shape, dtype=np.uint8)
    x2_u8 = np.random.randint(8, 30, size=x2_shape, dtype=np.uint8)
    x1_hif8 = x1_u8.view(hifloat8)
    x2_hif8 = x2_u8.view(hifloat8)
    x1 = x1_hif8.astype(np.float32)
    x2 = x2_hif8.astype(np.float32)

    x1_t = torch.from_numpy(x1).t() if transpose_a else torch.from_numpy(x1)
    x2_t = torch.from_numpy(x2).t() if transpose_b else torch.from_numpy(x2)
    out_base = torch.matmul(x1_t, x2_t)
    return x1_u8, x2_u8, out_base


def _dirs_next_to_this_script():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    output_dir = os.path.join(script_dir, "output")
    for d in (input_dir, output_dir):
        os.makedirs(d, exist_ok=True)
    return input_dir, output_dir


def _add_mkn_transpose_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("m", type=int, help="M dimension (logical rows of A / result)")
    parser.add_argument("k", type=int, help="K dimension (inner)")
    parser.add_argument("n", type=int, help="N dimension (logical cols of B / result)")
    parser.add_argument(
        "transA",
        nargs="?",
        default=None,
        choices=("true", "false", "True", "False"),
        help="transpose A on device (omit with transB: default false)",
    )
    parser.add_argument(
        "transB",
        nargs="?",
        default=None,
        choices=("true", "false", "True", "False"),
        help="transpose B on device (omit with transA: default true)",
    )


def _resolve_transpose(args: argparse.Namespace) -> tuple[bool, bool]:
    if (args.transA is None) ^ (args.transB is None):
        raise ValueError("transA and transB must be both omitted or both provided.")
    if args.transA is None:
        return False, True
    return args.transA.lower() == "true", args.transB.lower() == "true"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate hifloat8 TC golden/input bin files (per-channel FP32 scale in scale.bin). "
            "Writes to ./input and ./output next to this script. "
            "transA/transB must be both omitted or both provided; "
            "if omitted, defaults are transA=false and transB=true (same as matmul_a16w16 gen_data.py)."
        ),
        usage="%(prog)s m k n [transA transB]",
    )
    _add_mkn_transpose_args(parser)
    args = parser.parse_args()
    try:
        transpose_a, transpose_b = _resolve_transpose(args)
    except ValueError as exc:
        parser.error(str(exc))

    x1_u8, x2_u8, out_base = _golden_matmul_hifloat8(
        args.m, args.k, args.n, transpose_a, transpose_b
    )

    n = args.n
    scale_fp32 = np.random.uniform(0.01, 2.0, size=(n, 1)).astype(np.float32)
    scale_u32 = scale_fp32.view(np.uint32)
    scale_u32 &= np.uint32(0xFFFFE000)
    scale_fp32_high19 = scale_u32.view(np.float32)
    out_tc = out_base * torch.from_numpy(scale_fp32_high19.T)

    input_dir, output_dir = _dirs_next_to_this_script()

    x1_u8.tofile(f"{input_dir}/x1.bin")
    x2_u8.tofile(f"{input_dir}/x2.bin")
    scale_fp32.tofile(f"{input_dir}/scale.bin")
    out_tc_bf16 = out_tc.to(torch.bfloat16)
    out_tc_bf16.view(torch.uint16).numpy().tofile(f"{output_dir}/cpu_output.bin")

if __name__ == "__main__":
    main()
