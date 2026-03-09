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

import sys
import numpy as np
import torch

ERROR_TOL = 1e-3
DATA_TYPE = np.float32


def verify_result():

    output = np.fromfile("./output/npu_out.bin", dtype=DATA_TYPE)
    golden = np.fromfile("./output/golden_out.bin", dtype=DATA_TYPE)

    if output.size != golden.size:
        raise ValueError("output size != golden size")

    # ------------------------------
    # NaN mask
    # ------------------------------

    output_nan = np.isnan(output)
    golden_nan = np.isnan(golden)

    both_nan = output_nan & golden_nan
    nan_mismatch = output_nan ^ golden_nan

    # ------------------------------
    # 数值误差
    # ------------------------------

    diff = np.abs(output - golden)

    # 误差位置
    diff_mask = diff > 1

    # 合并错误
    error_mask = (diff_mask | nan_mismatch) & (~both_nan)

    diff_indices = np.where(error_mask)[0]

    # ------------------------------
    # 打印前100个错误
    # ------------------------------

    max_print = min(100, diff_indices.size)

    for i in range(max_print):

        idx = diff_indices[i]

        golden_val = golden[idx]
        output_val = output[idx]

        denom = max(abs(golden_val), 1e-12)

        rdiff = abs(output_val - golden_val) / denom

        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f"
            % (idx, golden_val, output_val, rdiff)
        )

    # ------------------------------
    # error ratio
    # ------------------------------

    error_ratio = diff_indices.size / golden.size

    print("error count:", diff_indices.size)
    print("total count:", golden.size)
    print("error ratio: %.6f, tolerance: %.6f" % (error_ratio, ERROR_TOL))

    return error_ratio <= ERROR_TOL

if __name__ == "__main__":

    try:
        res = verify_result()
        if not res:
            raise ValueError("[ERROR] result error")
        print("test pass")

    except Exception as e:
        print(e)
        sys.exit(1)
        