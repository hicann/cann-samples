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

function(add_simd_vf_story_case CASE_NAME)
    file(GLOB ASC_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.asc)

    set(CASE_TARGETS "")
    foreach(ASC_FILE ${ASC_FILES})
        get_filename_component(TARGET_SUFFIX ${ASC_FILE} NAME_WE)
        set(TARGET_NAME "simd_vf_story_${CASE_NAME}_${TARGET_SUFFIX}")

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
        list(APPEND CASE_TARGETS ${TARGET_NAME})
    endforeach()

    set(CASE_AGGREGATE_TARGET "simd_vf_story_${CASE_NAME}")
    add_custom_target(${CASE_AGGREGATE_TARGET} DEPENDS ${CASE_TARGETS})
    set_property(GLOBAL APPEND PROPERTY SIMD_VF_STORY_CASE_TARGETS ${CASE_AGGREGATE_TARGET})

    if(CASE_TARGETS)
        install(TARGETS ${CASE_TARGETS}
            RUNTIME DESTINATION 2_Performance/simd_vf_story/${CASE_NAME}
        )
    endif()
endfunction()

set_property(GLOBAL PROPERTY SIMD_VF_STORY_CASE_TARGETS "")
file(GLOB CHILD_CMAKES CONFIGURE_DEPENDS */CMakeLists.txt)
foreach(CHILD_CMAKE ${CHILD_CMAKES})
    get_filename_component(CHILD_DIR ${CHILD_CMAKE} DIRECTORY)
    add_subdirectory(${CHILD_DIR})
endforeach()

get_property(SIMD_VF_STORY_CASE_TARGETS GLOBAL PROPERTY SIMD_VF_STORY_CASE_TARGETS)
add_custom_target(simd_vf_story DEPENDS ${SIMD_VF_STORY_CASE_TARGETS})
