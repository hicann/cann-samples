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

# gelu_without_vf
add_executable(gelu_without_vf gelu_without_vf.cpp gelu_cpu.cpp)
target_include_directories(gelu_without_vf PRIVATE ${ASCEND_INCLUDE_DIRS})
target_link_directories(gelu_without_vf PRIVATE ${ASCEND_DIR}/lib64)
target_compile_options(gelu_without_vf PRIVATE
    "-xasc"
    "--npu-arch=dav-3101"
    "-w"
    "-O1"
)
target_link_libraries(gelu_without_vf
    PRIVATE
    m
    dl
    platform
    tiling_api
)
install(TARGETS gelu_without_vf
    RUNTIME DESTINATION 1_Features/vector_function
)

# gelu_with_vf
add_executable(gelu_with_vf gelu_with_vf.cpp gelu_cpu.cpp)
target_include_directories(gelu_with_vf PRIVATE ${ASCEND_INCLUDE_DIRS})
target_link_directories(gelu_with_vf PRIVATE ${ASCEND_DIR}/lib64)
target_compile_options(gelu_with_vf PRIVATE
    "-xasc"
    "--npu-arch=dav-3101"
    "-w"
    "-O1"
)
target_link_libraries(gelu_with_vf
    PRIVATE
    m
    dl
    platform
    tiling_api
)
install(TARGETS gelu_with_vf
    RUNTIME DESTINATION 1_Features/vector_function
)
