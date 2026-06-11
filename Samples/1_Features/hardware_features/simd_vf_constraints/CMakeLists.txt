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

file(GLOB ASC_FILES CONFIGURE_DEPENDS src/*.asc)

set(INSTALL_TARGETS "")
foreach(ASC_FILE ${ASC_FILES})
    get_filename_component(TARGET_SUFFIX ${ASC_FILE} NAME_WE)
    set(TARGET_NAME "simd_vf_constraints_${TARGET_SUFFIX}")

    add_executable(${TARGET_NAME} ${ASC_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
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
    list(APPEND INSTALL_TARGETS ${TARGET_NAME})
endforeach()

add_custom_target(simd_vf_constraints DEPENDS ${INSTALL_TARGETS})

install(TARGETS ${INSTALL_TARGETS}
    RUNTIME DESTINATION 1_Features/hardware_features/simd_vf_constraints
)
