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

import numpy as np
import torch
from ml_dtypes import bfloat16

# If calculations use float16, please change this to torch.float16 here
DATA_TYPE = torch.bfloat16

def write_artifacts(base_dir, a_data, b_data, out):
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    a_data.view(torch.uint16).numpy().tofile(os.path.join(input_dir, "input_a.bin"))
    b_data.view(torch.uint16).numpy().tofile(os.path.join(input_dir, "input_b.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data_simple(m, k, n, transpose_a, transpose_b):
    M = m
    K = k
    N = n

    a_ori = (np.random.uniform(1, 8, (K, M)).astype(np.float32) if transpose_a
             else np.random.uniform(1, 8, (M, K)).astype(np.float32))
    b_ori = (np.random.uniform(1, 8, (N, K)).astype(np.float32) if transpose_b
             else np.random.uniform(1, 8, (K, N)).astype(np.float32))

    a_cpu = torch.from_numpy(a_ori).to(DATA_TYPE)
    b_cpu = torch.from_numpy(b_ori).to(DATA_TYPE)

    a_cpu_t = a_cpu.t() if transpose_a else a_cpu
    b_cpu_t = b_cpu.t() if transpose_b else b_cpu

    out = torch.matmul(a_cpu_t, b_cpu_t).to(DATA_TYPE)

    current_dir = os.getcwd()
    write_artifacts(current_dir, a_cpu, b_cpu, out)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.normcase(os.path.abspath(script_dir)) != os.path.normcase(os.path.abspath(current_dir)):
        write_artifacts(script_dir, a_cpu, b_cpu, out)

    print("Data generated successfully!")


if __name__ == "__main__":
    if len(sys.argv) != 4 and len(sys.argv) != 6:
        print("Usage: python3 gen_data.py m k n")
        print("Or")
        print("Usage: python3 gen_data.py m k n transA transB")
        print("Example1: python3 gen_data.py 100 50 200")
        print("Example2: python3 gen_data.py 100 50 200 false true")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    if len(sys.argv) == 6:
        transpose_a = sys.argv[4].lower() == "true"
        transpose_b = sys.argv[5].lower() == "true"
    else:
        transpose_a = False
        transpose_b = True

    gen_golden_data_simple(m, k, n, transpose_a, transpose_b)
