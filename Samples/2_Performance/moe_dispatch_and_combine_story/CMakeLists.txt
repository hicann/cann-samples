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

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})

# moe_dispatch_and_combine_story
add_executable(moe_dispatch_and_combine_story src/dispatch_and_combine_final.cpp)
target_include_directories(moe_dispatch_and_combine_story PRIVATE ${ASCEND_INCLUDE_DIRS})
target_compile_definitions(moe_dispatch_and_combine_story PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_directories(moe_dispatch_and_combine_story PRIVATE ${ASCEND_DIR}/lib64)
target_compile_options(moe_dispatch_and_combine_story PRIVATE
    "-xasc"
    "--npu-arch=dav-3101"
    "-w"
    "-O3"
)
target_link_libraries(moe_dispatch_and_combine_story
    PRIVATE
    m
    dl
    platform
    tiling_api
    ascendcl
    runtime
    stdc++
)
install(TARGETS moe_dispatch_and_combine_story
    RUNTIME DESTINATION 2_Performance/moe_dispatch_and_combine_story
)