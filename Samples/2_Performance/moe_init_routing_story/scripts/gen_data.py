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
import random
import argparse
import numpy


def _find_project_root():
    path = os.path.dirname(os.path.abspath(__file__))
    while path != os.path.dirname(path):
        if os.path.isdir(os.path.join(path, "cmake")):
            return path
        path = os.path.dirname(path)
    return os.path.dirname(os.path.abspath(__file__))


_PROJECT_ROOT = _find_project_root()
_DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "build", "Samples",
                                    "2_Performance", "moe_init_routing_story")


def moe_init_routing_numpy(x, expert_idx, k):
    expert_start = 0
    expert_end = 8
    expert_idx_in = expert_idx.copy().reshape(-1)
    actual_expert_total_num = numpy.sum((expert_idx_in >= expert_start) & (expert_idx_in < expert_end))

    # sort
    expert_idx_in[(expert_idx_in < expert_start)] = numpy.int32(numpy.iinfo(numpy.int32).max)
    sorted_expert_indices = numpy.argsort(expert_idx_in, axis=-1, kind="stable")
    sorted_expert_idx = expert_idx_in[sorted_expert_indices]
    
    # scatter
    expanded_row_idx = sorted_expert_indices

    # gather
    expanded_scale = None
    expaned_x = x[sorted_expert_indices[:actual_expert_total_num] // k, :]

    # count
    expert_token_count = numpy.bincount(sorted_expert_idx[:actual_expert_total_num] - expert_start)
    expert_token_count = numpy.concatenate([expert_token_count, 
        numpy.zeros((expert_end - expert_start) - len(expert_token_count)).astype(numpy.int64)])

    return expaned_x, expanded_row_idx, expert_token_count


def gen_input_data(n, k, h):
    output_dir = _DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    x_shape = (n, h)

    x = numpy.random.uniform(low=-10, high=10, size=x_shape).astype(numpy.float32)
    x.tofile(os.path.join(output_dir, "x.bin"))

    rng = numpy.random.default_rng()
    expert_idx_array = []
    for _ in range(n):
        nums = numpy.arange(0, 8)
        rng.shuffle(nums)
        row = nums[:k]
        expert_idx_array.append(row)
    expert_idx = numpy.array(expert_idx_array, dtype=numpy.int32)
    expert_idx.tofile(os.path.join(output_dir, "expert_idx.bin"))

    expaned_x, expanded_row_idx, expert_token_count = moe_init_routing_numpy(x, expert_idx, k)

    expaned_x = expaned_x.astype(numpy.float32)
    expanded_row_idx = expanded_row_idx.astype(numpy.int32)
    expert_token_count = expert_token_count.astype(numpy.int64)

    expaned_x.tofile(os.path.join(output_dir, "expaned_x.bin"))
    expanded_row_idx.tofile(os.path.join(output_dir, "expanded_row_idx.bin"))
    expert_token_count.tofile(os.path.join(output_dir, "expert_token_count.bin"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成 MoeInitRouting 测试数据')
    parser.add_argument('-n', '--ndim', type=int, required=True)
    parser.add_argument('-k', '--kdim', type=int, required=True)
    parser.add_argument('-c', '--coldim', type=int, required=True)
    args = parser.parse_args()
    gen_input_data(args.ndim, args.kdim, args.coldim)
