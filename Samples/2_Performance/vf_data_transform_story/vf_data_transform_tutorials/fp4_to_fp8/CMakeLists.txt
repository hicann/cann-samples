# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_subdirectory(0_raw_field_mapping)
add_subdirectory(1_unpack_packed_byte)
add_subdirectory(2_shift_select_and)
add_subdirectory(3_hoist_constants)
add_subdirectory(4_fold_scale)

add_custom_target(vf_data_transform_fp4_to_fp8_tutorial DEPENDS
    vf_data_transform_fp4_to_fp8_0_raw_field_mapping
    vf_data_transform_fp4_to_fp8_1_unpack_packed_byte
    vf_data_transform_fp4_to_fp8_2_shift_select_and
    vf_data_transform_fp4_to_fp8_3_hoist_constants
    vf_data_transform_fp4_to_fp8
)
