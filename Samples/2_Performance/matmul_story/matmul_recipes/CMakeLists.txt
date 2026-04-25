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

function(add_matmul_recipe target_name source variant)
    add_executable(${target_name} ${CMAKE_CURRENT_SOURCE_DIR}/examples/${source})
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_options(${target_name} PRIVATE ${MATMUL_ASCENDC_COMPILE_OPTS})
    target_link_libraries(${target_name} PRIVATE
        cann_samples::ascend_base
        cann_samples::tensor_api
    )
    install(TARGETS ${target_name}
        RUNTIME DESTINATION ${MATMUL_INSTALL_PREFIX}/matmul_recipes/${variant}
    )
endfunction()

add_matmul_recipe(quant_matmul_mxfp4_swat
    quant_matmul_mxfp4/quant_matmul_mxfp4_swat.asc quant_matmul_mxfp4)
add_matmul_recipe(quant_matmul_mxfp4_a_full_load
    quant_matmul_mxfp4/quant_matmul_mxfp4_a_full_load.asc quant_matmul_mxfp4)
add_matmul_recipe(matmul_a16w16_swat
    matmul_a16w16/matmul_a16w16_swat.asc matmul_a16w16)
add_matmul_recipe(matmul_a16w16_streamk
    matmul_a16w16/matmul_a16w16_streamk.asc matmul_a16w16)

add_matmul_recipe(quant_matmul_mxfp8_swat
    quant_matmul_mxfp8/quant_matmul_mxfp8_swat.asc quant_matmul_mxfp8)
add_matmul_recipe(quant_matmul_mxfp8_a_full_load
    quant_matmul_mxfp8/quant_matmul_mxfp8_a_full_load.asc quant_matmul_mxfp8)

# Install helper scripts into the same directory as the executables. Host code
# uses the executable directory as baseDir and runs `python3 verify_result.py`
# there; gen_data.py is also documented as run from that cwd.
set(RECIPE_VARIANTS quant_matmul_mxfp4 quant_matmul_mxfp8 matmul_a16w16)
foreach(variant IN LISTS RECIPE_VARIANTS)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples/${variant}/scripts/
        DESTINATION ${MATMUL_INSTALL_PREFIX}/matmul_recipes/${variant}
        FILES_MATCHING
        PATTERN "*.py"
        PATTERN "*.sh"
    )
endforeach()
