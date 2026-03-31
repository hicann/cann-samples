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

add_subdirectory(0_naive)
add_subdirectory(1_pingpong)
add_subdirectory(2_block_swat)
add_subdirectory(3_last_round_tile_balance)
add_subdirectory(4_unit_flag)
add_subdirectory(5_halfl1_ping_halfl1_pong)
add_subdirectory(6_scale_memory_access_coalescing)
add_subdirectory(7_fullload)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/
    DESTINATION ${MATMUL_INSTALL_PREFIX}/matmul_tutorials
    FILES_MATCHING PATTERN "*.py"
)
