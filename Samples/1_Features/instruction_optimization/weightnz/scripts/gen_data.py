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

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import numpy as np
import torch
from ml_dtypes import bfloat16

# If calculations use float16, please change this to torch.float16 here
DATA_TYPE = torch.bfloat16

def index_nd_to_nz_torch(matrix):
    """PyTorch 实现的 NZ 布局转换"""
    # 获取矩阵维度和数据类型
    matrix = matrix.t()
    if len(matrix.shape) == 2:
        k, n = matrix.shape
    else:
        raise ValueError(f"Expected 2D tensor, got shape {matrix.shape}")
    
    # 获取元素大小（字节）
    element_size = matrix.element_size()
    cube_size = 32 // element_size
    
    # Padding dimensions
    ceil_k = ((k + cube_size - 1) // cube_size) * cube_size
    ceil_n = ((n + 15) // 16) * 16
    
    # 创建零填充矩阵
    padded = torch.zeros((ceil_k, ceil_n), dtype=matrix.dtype, device=matrix.device)
    padded[:k, :n] = matrix
    
    num_k_tiles = ceil_k // cube_size
    num_n_tiles = ceil_n // 16
    
    # Reshape to [cube_size, num_k_tiles, 16, num_n_tiles]
    reshaped = padded.reshape(num_k_tiles, cube_size, num_n_tiles, 16)
    
    # Permute to [16, cube_size, num_k_tiles, num_n_tiles]
    shuffled = reshaped.permute(2, 0, 1, 3)
    
    return shuffled.flatten()

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

    b_cpu_nz = index_nd_to_nz_torch(b_cpu)

    current_dir = os.getcwd()
    write_artifacts(current_dir, a_cpu, b_cpu_nz, out)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.normcase(os.path.abspath(script_dir)) != os.path.normcase(os.path.abspath(current_dir)):
        write_artifacts(script_dir, a_cpu, b_cpu_nz, out)

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
