#!/usr/bin/python3
# coding=utf-8

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import json
import os
import sys

import numpy as np


MAX_MISMATCHES_TO_PRINT = 8


def format_value(value, dtype):
    width = dtype.itemsize * 2
    unsigned = int(np.asarray(value).view(np.dtype(f"uint{dtype.itemsize * 8}")))
    return f"0x{unsigned:0{width}x}"


def compare_binary(golden_path, actual_path, dtype, expected_count, label):
    expected_bytes = expected_count * dtype.itemsize
    golden_bytes = os.path.getsize(golden_path)
    actual_bytes = os.path.getsize(actual_path)
    if golden_bytes != expected_bytes or actual_bytes != expected_bytes:
        raise ValueError(
            f"{label} binary size mismatch: expected={expected_bytes}, "
            f"golden={golden_bytes}, actual={actual_bytes}"
        )

    golden = np.fromfile(golden_path, dtype=dtype)
    actual = np.fromfile(actual_path, dtype=dtype)
    mismatch_indices = np.flatnonzero(golden != actual)
    if mismatch_indices.size != 0:
        for index in mismatch_indices[:MAX_MISMATCHES_TO_PRINT]:
            print(
                f"[VERIFY][{label}][MISMATCH] index={int(index)} "
                f"expected={format_value(golden[index], dtype)} actual={format_value(actual[index], dtype)}"
            )
        raise ValueError(
            f"{label} binary data mismatch: {mismatch_indices.size}/{expected_count} elements differ"
        )
    return expected_bytes


def verify(data_dir):
    metadata_path = os.path.join(data_dir, "metadata.json")
    golden_path = os.path.join(data_dir, "output", "golden.bin")
    actual_path = os.path.join(data_dir, "output", "npu_out.bin")
    golden_storage_path = os.path.join(data_dir, "output", "golden_storage.bin")
    actual_storage_path = os.path.join(data_dir, "output", "npu_storage.bin")
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    dtype = np.dtype(metadata["output_dtype"])
    expected_count = int(metadata["output_count"])
    storage_count = int(metadata["output_storage_count"])
    expected_bytes = compare_binary(golden_path, actual_path, dtype, expected_count, "payload")
    storage_bytes = compare_binary(
        golden_storage_path, actual_storage_path, dtype, storage_count, "storage"
    )

    print(
        f"[VERIFY][{metadata['task']}/{metadata['case']}] status=PASS "
        f"payload_elements={expected_count} payload_bytes={expected_bytes} "
        f"storage_elements={storage_count} storage_bytes={storage_bytes} mismatches=0"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Compare NPU output and Python golden binaries exactly.")
    parser.add_argument("--data-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        verify(parse_args().data_dir)
    except Exception as error:
        print(f"[VERIFY][ERROR] {error}", file=sys.stderr)
        sys.exit(1)
