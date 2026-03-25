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

set(CPP_FILES
    src/1_multi_core.cpp
    src/2_double_buffer.cpp
    src/3_bandwidth_utilization.cpp
    src/4_ub_utilization.cpp
    src/5_simt.cpp
)
set(INSTALL_TARGETS "")
foreach(CPP_FILE ${CPP_FILES})
    get_filename_component(TARGET_NAME ${CPP_FILE} NAME_WE)
    add_executable(${TARGET_NAME} ${CPP_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_compile_definitions(${TARGET_NAME} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${TARGET_NAME} PRIVATE
        "-xasc"
        "--npu-arch=dav-3510"
        "-w"
        "-O3"
    )
    target_link_libraries(${TARGET_NAME}
        PRIVATE
        cann_samples::ascend_base
        ascendcl
        runtime
        stdc++
    )
    list(APPEND INSTALL_TARGETS ${TARGET_NAME})
endforeach()

add_custom_target(moe_init_routing_story DEPENDS ${INSTALL_TARGETS})
install(TARGETS ${INSTALL_TARGETS}
    RUNTIME DESTINATION 2_Performance/moe_init_routing_story
)