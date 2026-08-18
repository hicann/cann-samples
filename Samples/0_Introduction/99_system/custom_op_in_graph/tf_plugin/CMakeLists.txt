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
    message(WARNING "tf_plugin: no source files found, skip")
    return()
endif()

set(TARGET_NAME ${TF_PLUGIN_TARGET})
set(LIBRARY_OUTPUT_PATH ${TF_PLUGIN_TARGET_OUT_DIR})

add_library(${TARGET_NAME} SHARED ${SRCS})

target_compile_definitions(${TARGET_NAME} PRIVATE
    google=ascend_private
    _GLIBCXX_USE_CXX11_ABI=0
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${ASCEND_INC}
)

target_link_libraries(${TARGET_NAME} PRIVATE
    ${ASCEND_DIR}/lib64/libgraph.so
    ${ASCEND_DIR}/lib64/libregister.so
    stdc++
)
