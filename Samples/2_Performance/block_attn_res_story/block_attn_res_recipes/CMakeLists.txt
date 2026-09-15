# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(BLOCK_ATTN_RES_RECIPE_SOURCES
    examples/block_attn_res/block_attn_res_prepare.asc
    examples/block_attn_res/block_attn_res_update.asc
    examples/block_attn_res/block_attn_res_e2e.asc
)

configure_file(scripts/gen_data.py gen_data.py COPYONLY)
configure_file(scripts/verify_result.py verify_result.py COPYONLY)
configure_file(scripts/run.sh run.sh COPYONLY)

set(BLOCK_ATTN_RES_RECIPE_TARGETS "")
foreach(SOURCE_FILE ${BLOCK_ATTN_RES_RECIPE_SOURCES})
    get_filename_component(TARGET_NAME ${SOURCE_FILE} NAME_WE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(${TARGET_NAME} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${TARGET_NAME} PRIVATE
        "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
        "$<$<COMPILE_LANGUAGE:ASC>:-O3>"
        "$<$<COMPILE_LANGUAGE:ASC>:--cce-simd-vf-fusion=true>"
    )
    target_link_libraries(${TARGET_NAME} PRIVATE
        m
        dl
        platform
        tiling_api
        ascendcl
        runtime
        stdc++
        cann_samples::tensor_api
    )
    list(APPEND BLOCK_ATTN_RES_RECIPE_TARGETS ${TARGET_NAME})
endforeach()

add_custom_target(block_attn_res_recipes DEPENDS ${BLOCK_ATTN_RES_RECIPE_TARGETS})

install(TARGETS ${BLOCK_ATTN_RES_RECIPE_TARGETS}
    RUNTIME DESTINATION ${BLOCK_ATTN_RES_INSTALL_PREFIX}/block_attn_res_recipes
)

install(FILES scripts/gen_data.py scripts/verify_result.py scripts/run.sh
    DESTINATION ${BLOCK_ATTN_RES_INSTALL_PREFIX}/block_attn_res_recipes
)
