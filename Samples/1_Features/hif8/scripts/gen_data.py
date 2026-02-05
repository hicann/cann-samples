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
import numpy as np
import en_dtypes


def gen_golden_data_simple():
    input_shape_x = [1, 2048]
    input_shape_scale = [1, 2048]
    dtype = np.float32
    y_dtype = en_dtypes.hifloat8
    input_x = np.random.uniform(-10, 10, input_shape_x).astype(dtype)
    input_scale = np.random.uniform(-10, 10, input_shape_scale).astype(dtype)
    input_offset = np.random.uniform(-10, 10, input_shape_scale).astype(dtype)
    y_golden = (input_x / input_scale + input_offset).astype(y_dtype)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    input_scale.tofile("./input/input_scale.bin")
    input_offset.tofile("./input/input_offset.bin")
    y_golden.tofile("./output/golden_y.bin")


if __name__ == "__main__":
    gen_golden_data_simple()