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

set(FALITE_SCRIPTS
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/flash_attn_lite_gendata.py
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/flash_attn_lite_verify.py
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/thread_limit.py
)

set(FALITE_SHARED_DEMO
    ${CMAKE_CURRENT_SOURCE_DIR}/src/flash_attn_lite_demo.cpp
)

# 程序根据自身路径查找脚本, 因此脚本与所有版本的二进制放在同一目录.
foreach(SCRIPT_FILE ${FALITE_SCRIPTS})
    get_filename_component(SCRIPT_NAME ${SCRIPT_FILE} NAME)
    configure_file(${SCRIPT_FILE} ${SCRIPT_NAME} COPYONLY)
endforeach()

set(FALITE_VERSION_TARGETS "")

function(add_falite_version VERSION_NAME)
    set(VERSION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/${VERSION_NAME})
    set(TARGET_NAME falite_${VERSION_NAME})
    set(VERSION_SOURCES
        ${FALITE_SHARED_DEMO}
        ${VERSION_DIR}/host/flash_attn_lite_host.cpp
        ${VERSION_DIR}/kernel/flash_attn_lite_kernel.asc
    )

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

    set(FALITE_VERSION_TARGETS
        ${FALITE_VERSION_TARGETS} ${TARGET_NAME}
        PARENT_SCOPE
    )
endfunction()

# 复制 src/v(n-1) 为 src/vn 后重新配置, 即可自动生成 falite_vn.
file(GLOB VERSION_DIRS CONFIGURE_DEPENDS LIST_DIRECTORIES true
    ${CMAKE_CURRENT_SOURCE_DIR}/src/v*
)
foreach(VERSION_DIR ${VERSION_DIRS})
    if(IS_DIRECTORY ${VERSION_DIR})
        get_filename_component(VERSION_NAME ${VERSION_DIR} NAME)
        if(VERSION_NAME MATCHES "^v[0-9]+$")
            add_falite_version(${VERSION_NAME})
        endif()
    endif()
endforeach()

if(NOT FALITE_VERSION_TARGETS)
    message(FATAL_ERROR "${_cann_sample_path} 未发现 src/vN 版本目录")
endif()

add_custom_target(falite DEPENDS ${FALITE_VERSION_TARGETS})
add_custom_target(flash_attn_lite_story DEPENDS ${FALITE_VERSION_TARGETS})

install(TARGETS ${FALITE_VERSION_TARGETS}
    RUNTIME DESTINATION 2_Performance/flash_attn_lite_story
)
install(FILES ${FALITE_SCRIPTS}
    DESTINATION 2_Performance/flash_attn_lite_story
)
