#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# Generate input and golden files for simt_scatter_story.
#
# Shape:
# - base/golden: [4096, 8] int32
# - unique updates: [4096, 8], unique destination rows
# - conflict updates: [8192, 8], sorted by (destination row, original update position)
import argparse
import os

import numpy as np


DST_ROWS = 4096
INNER_DIM = 8
UNIQUE_UPDATES = 4096
CONFLICT_UPDATES = 8192


def make_base():
    data = np.arange(DST_ROWS * INNER_DIM, dtype=np.int32)
    return (data.reshape(DST_ROWS, INNER_DIM) * 3 - 17).astype(np.int32)


def make_updates(row_count, start):
    row_id = np.arange(row_count, dtype=np.int32).reshape(row_count, 1)
    col_id = np.arange(INNER_DIM, dtype=np.int32).reshape(1, INNER_DIM)
    return (start + row_id * 13 + col_id).astype(np.int32)


def write_case(output_dir, prefix, base, indices, updates):
    golden = base.copy()
    for row, dst in enumerate(indices):
        golden[int(dst), :] = updates[row, :]

    input_dir = os.path.join(output_dir, "input")
    golden_dir = os.path.join(output_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)
    indices.astype(np.int32).tofile(os.path.join(input_dir, f"{prefix}_indices.bin"))
    updates.astype(np.int32).tofile(os.path.join(input_dir, f"{prefix}_updates.bin"))
    golden.astype(np.int32).tofile(os.path.join(golden_dir, f"{prefix}_golden.bin"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".", help="directory to write input/ and output/")
    parser.add_argument("--seed", type=int, default=42, help="numpy random seed (default: 42)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    base = make_base()

    input_dir = os.path.join(args.output, "input")
    os.makedirs(input_dir, exist_ok=True)
    base.tofile(os.path.join(input_dir, "base.bin"))

    unique_indices = rng.permutation(DST_ROWS).astype(np.int32)[:UNIQUE_UPDATES]
    unique_updates = make_updates(UNIQUE_UPDATES, 100000)
    write_case(args.output, "unique", base, unique_indices, unique_updates)

    raw_indices = rng.integers(0, DST_ROWS, size=CONFLICT_UPDATES, dtype=np.int32)
    hot_count = CONFLICT_UPDATES // 2
    raw_indices[:hot_count] = rng.integers(0, 512, size=hot_count, dtype=np.int32)
    raw_updates = make_updates(CONFLICT_UPDATES, 200000)

    original_pos = np.arange(CONFLICT_UPDATES, dtype=np.int32)
    order = np.lexsort((original_pos, raw_indices))
    conflict_indices = raw_indices[order]
    conflict_updates = raw_updates[order]

    conflict_golden = base.copy()
    for row, dst in enumerate(raw_indices):
        conflict_golden[int(dst), :] = raw_updates[row, :]

    conflict_indices.astype(np.int32).tofile(os.path.join(input_dir, "conflict_indices.bin"))
    conflict_updates.astype(np.int32).tofile(os.path.join(input_dir, "conflict_updates.bin"))

    golden_dir = os.path.join(args.output, "output")
    os.makedirs(golden_dir, exist_ok=True)
    conflict_golden.astype(np.int32).tofile(os.path.join(golden_dir, "conflict_golden.bin"))


if __name__ == "__main__":
    main()
