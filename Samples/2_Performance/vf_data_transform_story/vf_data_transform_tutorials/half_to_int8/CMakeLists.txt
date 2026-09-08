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

add_subdirectory(0_dintlv_load)
add_subdirectory(1_complementary_cast)
add_subdirectory(2_or_merge)
add_subdirectory(3_contiguous_store)

add_custom_target(vf_data_transform_half_to_int8_tutorial DEPENDS
    vf_data_transform_half_to_int8_0_dintlv_load
    vf_data_transform_half_to_int8_1_complementary_cast
    vf_data_transform_half_to_int8_2_or_merge
    vf_data_transform_half_to_int8
)
