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


def compare_result(actual_data, golden_data, name, verbose=False):
    error_count = 0
    element_num = len(actual_data)
    
    for i in range(element_num):
        if abs(actual_data[i] - golden_data[i]) > 0:
            error_count += 1
            if verbose:
                print(f"Index: {error_count:04d} RealIndex: {i:04d} "
                      f"Expected: {int(golden_data[i]):3d} Actual: {int(actual_data[i]):3d}")
    
    accuracy = (element_num - error_count) / element_num * 100 if element_num > 0 else 0
    print(f"{name} Precision is {accuracy:.4f}%")
    return error_count == 0


def verify_results(verbose=False):
    data_dir = _DEFAULT_OUTPUT_DIR
    golden_files = {
        'expanded_x': os.path.join(data_dir, 'expaned_x.bin'),
        'expanded_row_idx': os.path.join(data_dir, 'expanded_row_idx.bin'),
        'expert_token_count': os.path.join(data_dir, 'expert_token_count.bin')
    }
    
    result_files = {
        'expanded_x': os.path.join(data_dir, 'result_expanded_x.bin'),
        'expanded_row_idx': os.path.join(data_dir, 'result_expanded_row_idx.bin'),
        'expert_token_count': os.path.join(data_dir, 'result_expert_token_count.bin')
    }
    
    all_passed = True
    
    # verify expanded_x
    if os.path.exists(golden_files['expanded_x']) and os.path.exists(result_files['expanded_x']):
        golden = numpy.fromfile(golden_files['expanded_x'], dtype=numpy.float32)
        actual = numpy.fromfile(result_files['expanded_x'], dtype=numpy.float32)
        if not compare_result(actual, golden, 'ExpandedX', verbose):
            all_passed = False
    else:
        print("ExpandedX files not found, skipping...")
        all_passed = False
    
    # verify expanded_row_idx
    if os.path.exists(golden_files['expanded_row_idx']) and os.path.exists(result_files['expanded_row_idx']):
        golden = numpy.fromfile(golden_files['expanded_row_idx'], dtype=numpy.int32)
        actual = numpy.fromfile(result_files['expanded_row_idx'], dtype=numpy.int32)
        if not compare_result(actual, golden, 'ExpandedRowIdx', verbose):
            all_passed = False
    else:
        print("ExpandedRowIdx files not found, skipping...")
        all_passed = False
    
    # verify expert_token_count
    if os.path.exists(golden_files['expert_token_count']) and os.path.exists(result_files['expert_token_count']):
        golden = numpy.fromfile(golden_files['expert_token_count'], dtype=numpy.int64)
        actual = numpy.fromfile(result_files['expert_token_count'], dtype=numpy.int64)
        if not compare_result(actual, golden, 'TokenCount', verbose):
            all_passed = False
    else:
        print("TokenCount files not found, skipping...")
        all_passed = False
    
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='验证 MOE Init Routing 结果')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细错误信息')
    
    args = parser.parse_args()
    
    passed = verify_results(args.verbose)
    exit(0 if passed else 1)