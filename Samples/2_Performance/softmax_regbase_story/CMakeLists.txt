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

set(CPP_FILES
    src/0_membase.asc
    src/1_vf_fused.asc
    src/2_binary_fold.asc
    src/3_multi_row.asc
    src/4_pipeline.asc
    src/5_bigtile.asc
    src/6_merged_vf.asc
    src/7_merged_vf_direct.asc
)

configure_file(scripts/gen_data.py gen_data.py COPYONLY)

set(INSTALL_TARGETS "")
foreach(CPP_FILE ${CPP_FILES})
    get_filename_component(TARGET_NAME ${CPP_FILE} NAME_WE)
    set(TARGET_NAME "softmax_regbase_${TARGET_NAME}")
    add_executable(${TARGET_NAME} ${CPP_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(${TARGET_NAME} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    set(VF_FUSION true)
    if(TARGET_NAME STREQUAL "softmax_regbase_0_membase")
        set(VF_FUSION false)
    endif()
    target_compile_options(${TARGET_NAME} PRIVATE
        "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
        "$<$<COMPILE_LANGUAGE:ASC>:-O3>"
        "$<$<COMPILE_LANGUAGE:ASC>:--cce-simd-vf-fusion=${VF_FUSION}>"
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

add_custom_target(softmax_regbase_story DEPENDS ${INSTALL_TARGETS})

install(TARGETS ${INSTALL_TARGETS}
    RUNTIME DESTINATION 2_Performance/softmax_regbase_story
)

install(FILES scripts/gen_data.py
    DESTINATION 2_Performance/softmax_regbase_story
)
