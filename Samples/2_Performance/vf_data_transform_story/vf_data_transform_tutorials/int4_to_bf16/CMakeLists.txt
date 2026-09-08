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

add_subdirectory(0_unpack4_load)
add_subdirectory(1_single_cast)
add_subdirectory(2_separate_views)
add_subdirectory(3_b8_predicate)

add_custom_target(vf_data_transform_int4_to_bf16_tutorial DEPENDS
    vf_data_transform_int4_to_bf16_0_unpack4_load
    vf_data_transform_int4_to_bf16_1_single_cast
    vf_data_transform_int4_to_bf16_2_separate_views
    vf_data_transform_int4_to_bf16
)
