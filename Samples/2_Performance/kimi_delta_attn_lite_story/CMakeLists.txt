# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cann_sample_check_arch(dav-3510)

#! CANNsim 对 mode2 多代复用支持不完整. 仅仿真时解除下一行注释,
#! 使全部 AIC/AIV CrossCore 同步切换为 mode4; 真机默认使用 mode2.
# set(SIM_COMPATIBLE ON)

set(KDALITE_V0_CHUNK_SIZE_DEFINITION "")
if(DEFINED KDALITE_V0_CHUNK_SIZE)
    if(NOT "${KDALITE_V0_CHUNK_SIZE}" MATCHES "^(16|32|64)$")
        message(FATAL_ERROR "KDALITE_V0_CHUNK_SIZE 只支持 16、32 或 64")
    endif()
    set(KDALITE_V0_CHUNK_SIZE_DEFINITION
        KDALITE_V0_CHUNK_SIZE=${KDALITE_V0_CHUNK_SIZE})
endif()

set(KDALITE_V1_CHUNK_SIZE_DEFINITION "")
if(DEFINED KDALITE_V1_CHUNK_SIZE)
    if(NOT "${KDALITE_V1_CHUNK_SIZE}" MATCHES "^(16|32|64)$")
        message(FATAL_ERROR "KDALITE_V1_CHUNK_SIZE 只支持 16、32 或 64")
    endif()
    set(KDALITE_V1_CHUNK_SIZE_DEFINITION
        KDALITE_V1_CHUNK_SIZE=${KDALITE_V1_CHUNK_SIZE})
endif()

set(KDALITE_V2_CHUNK_SIZE_DEFINITION "")
if(DEFINED KDALITE_V2_CHUNK_SIZE)
    if(NOT "${KDALITE_V2_CHUNK_SIZE}" MATCHES "^(16|32)$")
        message(FATAL_ERROR "KDALITE_V2_CHUNK_SIZE 只支持 16 或 32")
    endif()
    set(KDALITE_V2_CHUNK_SIZE_DEFINITION
        KDALITE_V2_CHUNK_SIZE=${KDALITE_V2_CHUNK_SIZE})
endif()

set(KDALITE_V3_CHUNK_SIZE_DEFINITION "")
if(DEFINED KDALITE_V3_CHUNK_SIZE)
    if(NOT "${KDALITE_V3_CHUNK_SIZE}" MATCHES "^(16|32)$")
        message(FATAL_ERROR "KDALITE_V3_CHUNK_SIZE 只支持 16 或 32")
    endif()
    set(KDALITE_V3_CHUNK_SIZE_DEFINITION
        KDALITE_V3_CHUNK_SIZE=${KDALITE_V3_CHUNK_SIZE})
endif()

set(KDALITE_SCRIPTS
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/kimi_delta_attn_lite_gendata.py
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/kimi_delta_attn_lite_verify.py
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/kdalite_thread_limit.py
)

set(KDALITE_SHARED_DEMO
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kimi_delta_attn_lite_demo.cpp
)

# Demo 根据自身路径查找脚本, 因此脚本与所有版本的二进制放在同一目录.
foreach(SCRIPT_FILE ${KDALITE_SCRIPTS})
    get_filename_component(SCRIPT_NAME ${SCRIPT_FILE} NAME)
    configure_file(${SCRIPT_FILE} ${SCRIPT_NAME} COPYONLY)
endforeach()

set(KDALITE_VERSION_TARGETS "")

function(add_kdalite_version VERSION_NAME)
    set(VERSION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/${VERSION_NAME})
    set(TARGET_NAME kdalite_${VERSION_NAME})
    string(SUBSTRING "${VERSION_NAME}" 1 -1 VERSION_ID)
    set(VERSION_SOURCES
        ${KDALITE_SHARED_DEMO}
        ${VERSION_DIR}/host/kimi_delta_attn_lite_host.cpp
        ${VERSION_DIR}/kernel/kimi_delta_attn_lite_kernel.asc
    )
    set(VERSION_DEFINITIONS
        KDALITE_VERSION_ID=${VERSION_ID}
    )
    if(VERSION_NAME STREQUAL "v0" AND KDALITE_V0_CHUNK_SIZE_DEFINITION)
        list(APPEND VERSION_DEFINITIONS
            ${KDALITE_V0_CHUNK_SIZE_DEFINITION})
    elseif(VERSION_NAME STREQUAL "v1" AND KDALITE_V1_CHUNK_SIZE_DEFINITION)
        list(APPEND VERSION_DEFINITIONS
            ${KDALITE_V1_CHUNK_SIZE_DEFINITION})
    elseif(VERSION_NAME STREQUAL "v2" AND KDALITE_V2_CHUNK_SIZE_DEFINITION)
        list(APPEND VERSION_DEFINITIONS
            ${KDALITE_V2_CHUNK_SIZE_DEFINITION})
    elseif(VERSION_NAME STREQUAL "v3" AND KDALITE_V3_CHUNK_SIZE_DEFINITION)
        list(APPEND VERSION_DEFINITIONS
            ${KDALITE_V3_CHUNK_SIZE_DEFINITION})
    endif()

    foreach(SOURCE_FILE ${VERSION_SOURCES})
        if(NOT EXISTS ${SOURCE_FILE})
            message(FATAL_ERROR "${TARGET_NAME} 缺少源码: ${SOURCE_FILE}")
        endif()
    endforeach()

    add_executable(${TARGET_NAME} ${VERSION_SOURCES})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${ASCEND_DIR}/${SYSTEM_PREFIX}/include
        ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(${TARGET_NAME} PRIVATE
        ${VERSION_DEFINITIONS}
    )
    target_compile_options(${TARGET_NAME} PRIVATE
        "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
        "$<$<COMPILE_LANGUAGE:ASC>:-O3>"
        "$<$<COMPILE_LANGUAGE:ASC>:-DENABLE_CV_COMM_VIA_SSBUF=true>"
        "$<$<COMPILE_LANGUAGE:ASC>:--cce-auto-sync=off>"
    )
    if(SIM_COMPATIBLE)
        target_compile_definitions(${TARGET_NAME} PRIVATE
            "$<$<COMPILE_LANGUAGE:ASC>:SIM_COMPATIBLE>"
        )
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE
        m
        dl
        platform
        tiling_api
        ascendcl
        runtime
        stdc++
    )

    set(KDALITE_VERSION_TARGETS
        ${KDALITE_VERSION_TARGETS} ${TARGET_NAME}
        PARENT_SCOPE
    )
endfunction()

# 复制 src/v(n-1) 为 src/vn 后重新配置, 即可自动生成 kdalite_vn.
file(GLOB VERSION_DIRS CONFIGURE_DEPENDS LIST_DIRECTORIES true
    ${CMAKE_CURRENT_SOURCE_DIR}/src/v*
)
foreach(VERSION_DIR ${VERSION_DIRS})
    if(IS_DIRECTORY ${VERSION_DIR})
        get_filename_component(VERSION_NAME ${VERSION_DIR} NAME)
        if(VERSION_NAME MATCHES "^v[0-9]+$")
            add_kdalite_version(${VERSION_NAME})
        endif()
    endif()
endforeach()

if(NOT KDALITE_VERSION_TARGETS)
    message(FATAL_ERROR "${_cann_sample_path} 未发现 src/vN 版本目录")
endif()

add_custom_target(kdalite DEPENDS ${KDALITE_VERSION_TARGETS})
add_custom_target(kimi_delta_attn_lite_story DEPENDS ${KDALITE_VERSION_TARGETS})

install(TARGETS ${KDALITE_VERSION_TARGETS}
    RUNTIME DESTINATION 2_Performance/kimi_delta_attn_lite_story
)
install(FILES ${KDALITE_SCRIPTS}
    DESTINATION 2_Performance/kimi_delta_attn_lite_story
)
