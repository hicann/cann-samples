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
import math
import numpy as np
import torch
from ml_dtypes import float4_e2m1fn
from en_dtypes import float8_e8m0


def pack_b4_to_b8(b4_data: np.ndarray):

    # pack b4 numpy array to int8 numpy array
    packed_shape = [b4_data.shape[0], int(b4_data.shape[1] / 2)]
    pack_size = 2
    shift = np.array([0, 4], dtype=np.int8)
    if b4_data.size % pack_size != 0:
        b4_data = np.pad(b4_data.flatten(), (0, pack_size - b4_data.size % pack_size), 'constant')
    b4_data = b4_data.reshape(-1, 2).view(np.int8)
    return np.sum(np.bitwise_and(b4_data, 0b00001111) << shift, axis=1, dtype=np.int8).reshape(packed_shape)


def gen_golden_data_simple(m, k, n):
    M = m
    K = k
    N = n

    # Generate data
    a_ori = np.random.uniform(0, 34, (M, K)).astype(float4_e2m1fn)
    a_pack_int8 = pack_b4_to_b8(a_ori)
    b_ori = np.random.uniform(0, 34, (N, K)).astype(float4_e2m1fn)
    b_pack_int8 = pack_b4_to_b8(b_ori)
    a_scale = np.random.uniform(1, 32, size=(M, math.ceil(K / 64), 2)).astype(float8_e8m0)
    b_scale = np.random.uniform(1, 32, size=(N, math.ceil(K / 64), 2)).astype(float8_e8m0)

    # false true transpose and broadcast
    a_scale_reshape = a_scale.reshape(M, -1)
    a_scale_broadcast = np.repeat(a_scale_reshape, 32, axis=-1)

    b_ori_transpose = np.swapaxes(b_ori, -1, -2)
    b_scale_reshape = b_scale.reshape(N, -1)
    b_scale_broadcast = np.repeat(b_scale_reshape, 32, axis=-1)
    b_scale_broadcast_transpose = np.swapaxes(b_scale_broadcast, -1, -2)

    # dequant
    a_dequant = a_ori.astype(np.float32) * a_scale_broadcast.astype(np.float32)
    b_dequant = b_ori_transpose.astype(np.float32) * b_scale_broadcast_transpose.astype(np.float32)

    a_cpu = torch.from_numpy(a_dequant)
    b_cpu = torch.from_numpy(b_dequant)

    out = torch.matmul(a_cpu, b_cpu)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    a_pack_int8.tofile("./input/input_a.bin")
    b_pack_int8.tofile("./input/input_b.bin")
    a_scale.tofile("./input/input_scaleA.bin")
    b_scale.tofile("./input/input_scaleB.bin")
    out.numpy().tofile("./output/golden_out.bin")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 gen_data.py m k n")
        sys.exit(1)

    # 获取参数
    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    gen_golden_data_simple(m, k, n)