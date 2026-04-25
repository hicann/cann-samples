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

set(MATMUL_STORY_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MATMUL_INSTALL_PREFIX "2_Performance/matmul_story")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MATMUL_ASCENDC_COMPILE_OPTS
    "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>"
    "$<$<COMPILE_LANGUAGE:ASC>:-O3>"
)

add_subdirectory(matmul_recipes)
add_subdirectory(matmul_tutorials)
