## ----------------------------------------------------------------------------
## This program is free software, you can redistribute it and/or modify it.
## Copyright (c) 2026 Huawei Technologies Co., Ltd.
## This file is a part of the CANN Open Software.
## Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
## Please refer to the License for details. You may not use this file except in compliance with the License.
## You may obtain a copy of the License at
## https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850
## ----------------------------------------------------------------------------

function(add_grouped_matmul_recipe target_name source variant)
    add_executable(${target_name}
        ${GROUPED_MATMUL_STORY_DIR}/grouped_matmul_recipes/examples/${source}
    )
    target_include_directories(${target_name} PRIVATE
        ${GROUPED_MATMUL_STORY_DIR}/grouped_matmul_recipes/include
        ${GROUPED_MATMUL_STORY_DIR}/grouped_matmul_recipes/common
    )
    target_compile_options(${target_name} PRIVATE ${GMM_STORY_ASCENDC_COMPILE_OPTS})
    target_link_libraries(${target_name} PRIVATE
        cann_samples::ascend_base
        cann_samples::tensor_api
    )
    install(TARGETS ${target_name}
        RUNTIME DESTINATION ${GROUPED_MATMUL_INSTALL_PREFIX}/grouped_matmul_recipes/${variant}
    )
endfunction()

add_grouped_matmul_recipe(
    quant_grouped_matmul_mxfp4
    quant_grouped_matmul_mxfp4/quant_grouped_matmul_mxfp4_split_m.asc
    quant_grouped_matmul_mxfp4
)

add_grouped_matmul_recipe(
    quant_grouped_matmul_mxfp8
    quant_grouped_matmul_mxfp8/quant_grouped_matmul_mxfp8_split_m.asc
    quant_grouped_matmul_mxfp8
)

add_grouped_matmul_recipe(
    weight_quant_grouped_matmul_mxfp8fp4
    weight_quant_grouped_matmul_mxfp8fp4/weight_quant_grouped_matmul_mxfp8fp4_split_m.asc
    weight_quant_grouped_matmul_mxfp8fp4
)

# Remove legacy install layout from old versions.
install(CODE
    "file(REMOVE_RECURSE \"\${CMAKE_INSTALL_PREFIX}/${GROUPED_MATMUL_INSTALL_PREFIX}/grouped_matmul_recipes/examples\")"
)

# Install helper scripts into the same directory as the executables. Host code
# uses the executable directory as baseDir and runs `python3 verify_result.py`
# there; gen_data.py is also documented as run from that cwd.
set(RECIPE_VARIANTS quant_grouped_matmul_mxfp4 quant_grouped_matmul_mxfp8 weight_quant_grouped_matmul_mxfp8fp4)
foreach(variant IN LISTS RECIPE_VARIANTS)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples/${variant}/scripts/
        DESTINATION ${GROUPED_MATMUL_INSTALL_PREFIX}/grouped_matmul_recipes/${variant}
        FILES_MATCHING
        PATTERN "*.py"
        PATTERN "*.sh"
    )
endforeach()
