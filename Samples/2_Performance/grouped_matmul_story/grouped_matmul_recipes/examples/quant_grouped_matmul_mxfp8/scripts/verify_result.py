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
from typing import List

import numpy as np
import torch

ERROR_TOL = 1e-3
DATA_TYPE = np.uint16


def parse_group_m_list(arg: str) -> List[int]:
    values = []
    for item in arg.split(","):
        item = item.strip()
        if not item:
            raise ValueError("group_m_list contains an empty item")
        value = int(item)
        if value < 0:
            raise ValueError("Each group M value must be greater than or equal to 0")
        values.append(value)
    if not values:
        raise ValueError("group_m_list must not be empty")
    return values


def load_group_m_list(group_num: int) -> List[int]:
    group_list = np.fromfile("./input/input_groupList.bin", dtype=np.int64)
    if group_list.size != group_num:
        raise ValueError("input_groupList.bin size does not match group_num")
    if np.any(group_list < 0):
        raise ValueError("input_groupList.bin contains negative group size")
    return group_list.astype(np.int64).tolist()


def verify_result(group_m_list: List[int], m: int, n: int):
    sum_group_m = sum(group_m_list)
    if sum_group_m > m:
        raise ValueError("sum(group_m_list) must be less than or equal to m")
    output = np.fromfile("./output/npu_out.bin", dtype=DATA_TYPE)
    golden = np.fromfile("./output/cpu_output.bin", dtype=DATA_TYPE)

    if output.size != golden.size:
        raise ValueError("npu output size != cpu output size")
    if output.size != m * n:
        raise ValueError(f"output element count {output.size} does not match m*n={m * n}")

    npu_output_tensor = torch.from_numpy(output).view(torch.bfloat16).reshape(m, n)
    golden_tensor = torch.from_numpy(golden).view(torch.bfloat16).reshape(m, n)
    golden_cmp = golden_tensor[:sum_group_m]
    npu_cmp = npu_output_tensor[:sum_group_m]
    print("\ncpu golden (sum(group_m_list) rows):\n", golden_cmp)
    print("npu output (sum(group_m_list) rows):\n", npu_cmp)

    return torch.allclose(golden_cmp, npu_cmp, rtol=ERROR_TOL, atol=ERROR_TOL, equal_nan=True)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 verify_result.py group_num m k n")
        sys.exit(1)

    try:
        group_num = int(sys.argv[1])
        m = int(sys.argv[2])
        k = int(sys.argv[3])
        n = int(sys.argv[4])
        if group_num <= 0:
            raise ValueError("group_num must be greater than 0")
        if m < 0:
            raise ValueError("m must be greater than or equal to 0")
        if k <= 0:
            raise ValueError("k must be greater than 0")
        if n <= 0:
            raise ValueError("n must be greater than 0")
        group_m_list = load_group_m_list(group_num)
        res = verify_result(group_m_list, m, n)
        if not res:
            raise ValueError("[ERROR] NPU results differ from CPU.\n")
        print("[PASS] NPU results are consistent with CPU.\n")
    except Exception as e:
        print(e)
        sys.exit(1)
