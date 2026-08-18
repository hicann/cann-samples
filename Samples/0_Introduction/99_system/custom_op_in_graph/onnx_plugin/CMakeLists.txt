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

aux_source_directory(. SRCS)

if("x${SRCS}" STREQUAL "x")
    message(WARNING "onnx_plugin: no source files found, skip")
    return()
endif()

set(TARGET_NAME ${ONNX_PLUGIN_TARGET})
set(LIBRARY_OUTPUT_PATH ${ONNX_PLUGIN_TARGET_OUT_DIR})

add_library(${TARGET_NAME} SHARED ${SRCS})

target_compile_definitions(${TARGET_NAME} PRIVATE
    google=ascend_private
    _GLIBCXX_USE_CXX11_ABI=0
)

set(NLOHMANN_JSON_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/json")
set(NLOHMANN_JSON_HEADER "${NLOHMANN_JSON_DIR}/include/nlohmann/json.hpp")
if(NOT EXISTS "${NLOHMANN_JSON_HEADER}")
    file(DOWNLOAD
        "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip"
        "${NLOHMANN_JSON_DIR}/include.zip"
        STATUS download_status)
    list(GET download_status 0 download_code)
    if(NOT download_code EQUAL 0)
        message(FATAL_ERROR "Failed to download nlohmann/json: ${download_status}")
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf "${NLOHMANN_JSON_DIR}/include.zip" --format=zip
        WORKING_DIRECTORY "${NLOHMANN_JSON_DIR}"
        RESULT_VARIABLE extract_code)
    if(NOT extract_code EQUAL 0)
        message(FATAL_ERROR "Failed to extract nlohmann/json")
    endif()
endif()

target_include_directories(${TARGET_NAME} PRIVATE
    ${ASCEND_INC}
    ${ASCEND_DIR}/opp/built-in/op_graph/inc/
    "${NLOHMANN_JSON_DIR}/include"
)

target_link_libraries(${TARGET_NAME} PRIVATE
    ${ASCEND_DIR}/lib64/libgraph.so
    ${ASCEND_DIR}/lib64/libregister.so
    stdc++
)
