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

set(SUPPORTED_NPU_ARCHS dav-3510)
file(RELATIVE_PATH SAMPLE_PATH ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
if(NOT "${NPU_ARCH}" IN_LIST SUPPORTED_NPU_ARCHS)
    message(STATUS "Skip sample ${SAMPLE_PATH}: NPU_ARCH=${NPU_ARCH} is not supported")
    return()
endif()

# unit_flag
add_executable(unit_flag main.asc)
target_link_directories(unit_flag PRIVATE ${ASCEND_DIR}/lib64)
target_compile_options(unit_flag PRIVATE
    "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
    "$<$<COMPILE_LANGUAGE:ASC>:-O3>"
)
target_link_libraries(unit_flag
    m
    dl
    platform
    tiling_api
    cann_samples::tensor_api
)

add_custom_command(
    TARGET unit_flag POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/
        ${CMAKE_CURRENT_BINARY_DIR}
)

set(DIST_PATH
    1_Features/instruction_optimization/unit_flag
)

install(TARGETS unit_flag
    RUNTIME DESTINATION ${DIST_PATH}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/
    DESTINATION ${DIST_PATH}
    FILES_MATCHING PATTERN "*"
)
