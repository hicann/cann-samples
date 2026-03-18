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
import en_dtypes

ERROR_TOL = 1e-3
data_type = en_dtypes.hifloat8

def verify_result(output, golden):
    # 1ulp对比方式
    output = np.fromfile(output, dtype=data_type).view(np.int8)
    golden = np.fromfile(golden, dtype=data_type).view(np.int8)
    diff_results = np.abs(np.subtract(output, golden))
    diff_indices = np.where(diff_results > 1)[0]

    npu_nan, golden_nan = np.isnan(output), np.isnan(golden)
    diff_nan = np.logical_and(npu_nan, golden_nan)
    both_nan_idx = np.where(diff_nan)
    diff_indices = np.setdiff1d(diff_indices, both_nan_idx)

    for index in range(len(diff_indices)):
        real_index = diff_indices[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
            (real_index, golden_data, output_data,
            abs(output_data - golden_data) / golden_data))
        if index == 100:
            break
    print("golden_data : ", golden)
    print("output : ", output)
    error_ratio = float(diff_indices.size) / golden.size
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, ERROR_TOL))
    return error_ratio <= ERROR_TOL


if __name__ == '__main__':
    try:
        res = verify_result(sys.argv[1], sys.argv[2])
        if not res:
            raise ValueError("[ERROR] result error")
        else:
            print("test pass")
    except Exception as e:
        print(e)
        sys.exit(1)
