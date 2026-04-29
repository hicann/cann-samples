# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(TARGET cann_samples_tensor_api)
    return()
endif()

find_package(Git QUIET)

set(TENSOR_API_PATH "${PROJECT_SOURCE_DIR}/third_party/tensor_api")

if(NOT EXISTS "${TENSOR_API_PATH}/CMakeLists.txt" AND NOT GIT_FOUND)
    message(FATAL_ERROR
        "Git is required to initialize third_party/tensor_api automatically."
    )
endif()

set(TENSOR_API_PREPARE_COMMANDS)
if(GIT_FOUND)
    list(APPEND TENSOR_API_PREPARE_COMMANDS
        COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive third_party/tensor_api
    )
endif()

add_custom_target(cann_samples_tensor_api_dependencies
    ${TENSOR_API_PREPARE_COMMANDS}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Initializing third_party/tensor_api"
    VERBATIM
)

add_library(cann_samples_tensor_api INTERFACE)
add_library(cann_samples::tensor_api ALIAS cann_samples_tensor_api)
add_dependencies(cann_samples_tensor_api cann_samples_tensor_api_dependencies)

target_include_directories(cann_samples_tensor_api INTERFACE
    "${TENSOR_API_PATH}/include/tensor_api"
    "${ASCEND_DIR}/asc"
)
