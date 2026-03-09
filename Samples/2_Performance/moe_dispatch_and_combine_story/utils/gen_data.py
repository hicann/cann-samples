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
import logging
import numpy as np
import en_dtypes
import torch
import torch_npu



def gen_golden_data_simple(bs, h, k):
    print("please edit code")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 gen_data.py bs, h, k")
        sys.exit(1)

    # 获取参数
    bs = int(sys.argv[1])
    h = int(sys.argv[2])
    k = int(sys.argv[3])
    # 需要补充生成golden的代码
    gen_golden_data_simple(bs, h, k)