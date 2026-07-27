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

cann_sample_check_arch(dav-3510)

# 每个 src/*.asc 编一个独立可执行（bw_read/bw_rw/bw_rcw）；新增 .asc 会被自动纳入。
file(GLOB ASC_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.asc)

set(BW_TARGETS "")
foreach(ASC_FILE ${ASC_FILES})
    get_filename_component(TARGET_NAME ${ASC_FILE} NAME_WE)
    add_executable(${TARGET_NAME} ${ASC_FILE})
    target_compile_options(${TARGET_NAME} PRIVATE
        "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
        "$<$<COMPILE_LANGUAGE:ASC>:-O3>"
    )
    target_link_libraries(${TARGET_NAME}
        PRIVATE
        m
        dl
        platform
        tiling_api
        ascendcl
        runtime
        stdc++
    )
    list(APPEND BW_TARGETS ${TARGET_NAME})
    install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION 1_Features/hardware_features/mem_bandwidth
    )
endforeach()

add_custom_target(mem_bandwidth DEPENDS ${BW_TARGETS})
