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

import numpy as np
import en_dtypes
import torch
import torch_npu
import os



def gen_golden_data_simple():
    b = 1
    n1 = 1
    n2 = 1
    s1 = 8192
    s2 = 8192
    d = 128

    input_layout = 'BNSD'
    scale_value = 0.088388

    q_tensor = torch.randint(-127, 127, (b, n1, s1, d), dtype=torch.int8)
    k_tensor = torch.randint(-127, 127, (b, n2, s2, d), dtype=torch.int8)
    v_tensor = torch.randint(-127, 127, (b, n2, s2, d), dtype=torch.int8)
    key_antiquant_scale = torch.rand((1, 1, 32, 1), dtype=torch.float32)
    value_antiquant_scale = torch.rand((1, 1, 32, 1), dtype=torch.float32)
    dequant_scale_query = torch.rand((1, 1, 64, 1), dtype=torch.float32)

    q_tensor_tmp = q_tensor.numpy()
    k_tensor_tmp = k_tensor.numpy()
    v_tensor_tmp = v_tensor.numpy()

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    q_tensor_tmp.tofile("./input/input_0.bin")
    k_tensor_tmp.tofile("./input/input_1.bin")
    v_tensor_tmp.tofile("./input/input_2.bin")
    key_antiquant_scale.numpy().tofile("./input/input_15.bin")
    value_antiquant_scale.numpy().tofile("./input/input_17.bin")
    dequant_scale_query.numpy().tofile("./input/input_27.bin")

    q_tensor = q_tensor.npu()
    k_tensor = k_tensor.npu()
    v_tensor = v_tensor.npu()
    key_antiquant_scale = key_antiquant_scale.npu()
    value_antiquant_scale = value_antiquant_scale.npu()
    dequant_scale_query = dequant_scale_query.npu()

    npu_out = torch.ops.npu.npu_fused_infer_attention_score_v2(q_tensor, k_tensor, v_tensor,
                                                            dequant_scale_key=key_antiquant_scale,
                                                            dequant_scale_value=value_antiquant_scale,
                                                            dequant_scale_query=dequant_scale_query, num_query_heads=n1,
                                                            softmax_scale=scale_value, input_layout=input_layout,
                                                            num_key_value_heads=n2, query_quant_mode=7,
                                                            key_quant_mode=7, value_quant_mode=7,
                                                            inner_precise=0, return_softmax_lse=0,
                                                            query_dtype=torch_npu.float8_e4m3fn,
                                                            key_dtype=torch_npu.float8_e4m3fn,
                                                            value_dtype=torch_npu.float8_e4m3fn)
    npu_out[0].cpu().numpy().tofile("./output/golden_out.bin")


if __name__ == "__main__":
    gen_golden_data_simple()