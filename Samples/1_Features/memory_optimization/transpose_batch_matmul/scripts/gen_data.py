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
from dataclasses import dataclass

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import numpy as np
import torch
from ml_dtypes import bfloat16

# If calculations use float16, please change this to torch.float16 here
DATA_TYPE = torch.bfloat16


@dataclass(frozen=True)
class MatmulShape:
    """Encapsulates the correlated matmul shape and transpose/split parameters."""

    m: int
    k: int
    n: int
    batch: int
    perm_x1: int
    perm_x2: int
    batch_split_factor: int

    @classmethod
    def from_argv(cls, argv: list[str]) -> "MatmulShape":
        if len(argv) != 8:
            print("Usage: python3 gen_data.py m k n batch perm_x1 perm_x2 batch_split_factor")
            print("   perm_x1: 0=[0, 1, 2], 1=[1, 0, 2]")
            print("   perm_x2: 0=[0, 1, 2], 1=[0, 2, 1]")
            print("Example: python3 gen_data.py 32 512 128 16 1 0 1")
            sys.exit(1)
        return cls(int(argv[1]), int(argv[2]), int(argv[3]), int(argv[4]),
                   int(argv[5]), int(argv[6]), int(argv[7]))


def write_artifacts(base_dir, a_data, b_data, out):
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    a_data.view(torch.uint16).numpy().tofile(os.path.join(input_dir, "input_a.bin"))
    b_data.view(torch.uint16).numpy().tofile(os.path.join(input_dir, "input_b.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data(shape: MatmulShape):
    m, k, n, batch = shape.m, shape.k, shape.n, shape.batch
    perm_x1, perm_x2, batch_split_factor = shape.perm_x1, shape.perm_x2, shape.batch_split_factor

    if perm_x1 == 0:
        a = torch.from_numpy(np.random.uniform(1, 8, (batch, m, k)).astype(np.float32)).to(DATA_TYPE)
        a_bmk = a
    else:
        a = torch.from_numpy(np.random.uniform(1, 8, (m, batch, k)).astype(np.float32)).to(DATA_TYPE)
        a_bmk = a.permute(1, 0, 2).contiguous()

    if perm_x2 == 0:
        b = torch.from_numpy(np.random.uniform(1, 8, (batch, k, n)).astype(np.float32)).to(DATA_TYPE)
        b_bkn = b
    else:
        b = torch.from_numpy(np.random.uniform(1, 8, (batch, n, k)).astype(np.float32)).to(DATA_TYPE)
        b_bkn = b.permute(0, 2, 1).contiguous()

    c_bmn = torch.bmm(a_bmk, b_bkn).to(DATA_TYPE)

    c_mbn = c_bmn.permute(1, 0, 2).contiguous()

    if batch_split_factor > 1:
        inner_batch = batch // batch_split_factor
        c_out = c_mbn.reshape(m, batch_split_factor, inner_batch, n) \
                     .permute(1, 0, 2, 3) \
                     .reshape(batch_split_factor, m * inner_batch * n) \
                     .contiguous()
    else:
        c_out = c_mbn.reshape(m, batch * n).contiguous()

    current_dir = os.getcwd()
    write_artifacts(current_dir,
                    a.contiguous().view(DATA_TYPE),
                    b.contiguous().view(DATA_TYPE),
                    c_out.view(DATA_TYPE))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.normcase(os.path.abspath(script_dir)) != os.path.normcase(os.path.abspath(current_dir)):
        write_artifacts(script_dir,
                    a.contiguous().view(DATA_TYPE),
                    b.contiguous().view(DATA_TYPE),
                    c_out.view(DATA_TYPE))

    print("Data generated successfully!")


if __name__ == "__main__":
    shape = MatmulShape.from_argv(sys.argv)
    gen_golden_data(shape)
