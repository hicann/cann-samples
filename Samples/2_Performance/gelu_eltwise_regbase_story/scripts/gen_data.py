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
# Generate input_x.bin and golden.bin for gelu_eltwise_regbase_story.
#
# - Input shape: [8192, 8192] float32, uniform(-3, 3), seed=42
# - Golden: y = exp(-0.5 * (GELU(x) + 1)^2 + 2)
# - Accuracy check is done in C++ (sample_common.h), tol=1e-3 (absolute)
import argparse
import os
import numpy as np


def gelu(x):
    coeff = -1.595769
    exponent = coeff * (x + 0.044715 * (x ** 3))
    return x / (1.0 + np.exp(exponent))


def gelu_eltwise(x):
    g = gelu(x)
    return np.exp(-0.5 * (g + 1.0) ** 2 + 2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".", help="directory to write input/ and output/")
    parser.add_argument("--seed", type=int, default=42, help="numpy random seed (default: 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    x = np.random.uniform(-3, 3, [8192, 8192]).astype(np.float32)
    y = gelu_eltwise(x)

    out_dir = args.output
    input_dir = os.path.join(out_dir, "input")
    output_dir = os.path.join(out_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    x.tofile(os.path.join(input_dir, "input_x.bin"))
    y.tofile(os.path.join(output_dir, "golden.bin"))


if __name__ == "__main__":
    main()
