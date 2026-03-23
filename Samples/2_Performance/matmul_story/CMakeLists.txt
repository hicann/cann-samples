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

# Canonical source root and install prefix shared by every recipe/tutorial
# target in this sample bundle.
set(MATMUL_STORY_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MATMUL_INSTALL_PREFIX "2_Performance/matmul_story")

# Keep the sample targets on a consistent language/runtime baseline.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Honor an externally provided Bisheng path when this sample bundle is
# configured in an environment that expects explicit compiler overrides.
if(BISHENG)
    set(CMAKE_C_COMPILER ${BISHENG})
    set(CMAKE_CXX_COMPILER ${BISHENG})
    set(CMAKE_LINKER ${BISHENG})
endif()

# Shared Ascend C compile flags for every recipe and tutorial executable.
set(MATMUL_ASCENDC_COMPILE_OPTS "-xasc" "--npu-arch=dav-3510" "-w" "-O3")

function(configure_matmul_recipe_target target_name)
    # All recipe executables share the same common utilities and recipe headers.
    target_include_directories(${target_name} PRIVATE
        ${MATMUL_STORY_DIR}/common
        ${MATMUL_STORY_DIR}/matmul_recipes/include
    )
    target_compile_options(${target_name} PRIVATE ${MATMUL_ASCENDC_COMPILE_OPTS})

    if(TARGET cann_samples::ascend_base)
        # Preferred in-tree linkage path when the umbrella sample targets are
        # already available in the current build graph.
        target_link_libraries(${target_name} PRIVATE
            cann_samples::ascend_base
            cann_samples::tensor_api
        )
    else()
        # Fallback linkage path for standalone builds against an installed
        # Ascend environment.
        target_include_directories(${target_name} PRIVATE ${ASCEND_INCLUDE_DIRS})
        target_link_directories(${target_name} PRIVATE ${ASCEND_DIR}/lib64)
        target_link_libraries(${target_name} PRIVATE
            m
            dl
            platform
            tiling_api
        )
    endif()
endfunction()

function(add_matmul_recipe target_name source_under_examples group_name)
    # Build one recipe launcher from the examples tree and install it next to
    # the helper scripts used by that recipe family.
    add_executable(${target_name}
        ${MATMUL_STORY_DIR}/matmul_recipes/examples/${source_under_examples}
    )
    configure_matmul_recipe_target(${target_name})
    install(TARGETS ${target_name}
        RUNTIME DESTINATION ${MATMUL_INSTALL_PREFIX}/matmul_recipes/examples/${group_name}
    )
endfunction()

add_matmul_recipe(
    quant_matmul_mxfp4_swat
    quant_matmul_mxfp4/quant_matmul_mxfp4_swat.cpp
    quant_matmul_mxfp4
)
add_matmul_recipe(
    quant_matmul_mxfp4_a_full_load
    quant_matmul_mxfp4/quant_matmul_mxfp4_a_full_load.cpp
    quant_matmul_mxfp4
)

function(add_matmul_tutorial target_name source_under_tutorials subdir)
    # Tutorials keep their own include roots, but otherwise share the same
    # Ascend C compile options as the recipe executables.
    add_executable(${target_name}
        ${MATMUL_STORY_DIR}/matmul_tutorials/${source_under_tutorials}
    )
    target_include_directories(${target_name} PRIVATE
        ${MATMUL_STORY_DIR}/common
        ${MATMUL_STORY_DIR}/matmul_tutorials/${subdir}
    )
    target_compile_options(${target_name} PRIVATE ${MATMUL_ASCENDC_COMPILE_OPTS})
    target_link_libraries(${target_name} PRIVATE
        cann_samples::ascend_base
    )
    install(TARGETS ${target_name}
        RUNTIME DESTINATION ${MATMUL_INSTALL_PREFIX}/matmul_tutorials/${subdir}
    )
endfunction()

add_matmul_tutorial(matmul_tutorial_mxfp4_base 0_naive/matmul_tutorial_mxfp4_base.cpp 0_naive)
add_matmul_tutorial(matmul_tutorial_mxfp4_swat  2_block_swat/matmul_tutorial_mxfp4_swat.cpp  2_block_swat)
add_matmul_tutorial(matmul_tutorial_mxfp4_swat_balance  3_last_round_tile_balance/matmul_tutorial_mxfp4_swat_balance.cpp  3_last_round_tile_balance)
add_matmul_tutorial(matmul_tutorial_mxfp4_swat_unitflag  4_unit_flag/matmul_tutorial_mxfp4_swat_unitflag.cpp  4_unit_flag)

# ============================================================================
# Install layout
# ============================================================================
# Keep the example-side helper scripts colocated with the executables, and
# also mirror them into `common/golden` for workflows that still expect the
# legacy sample golden-data layout.

set(RECIPE_GROUPS quant_matmul_mxfp4)
foreach(group IN LISTS RECIPE_GROUPS)
    install(FILES
        ${MATMUL_STORY_DIR}/matmul_recipes/examples/${group}/gen_data.py
        ${MATMUL_STORY_DIR}/matmul_recipes/examples/${group}/verify_result.py
        DESTINATION ${MATMUL_INSTALL_PREFIX}/matmul_recipes/examples/${group}
    )
    # Keep a compatibility copy of the generators under the historical
    # `common/golden` install tree.
    install(FILES
        ${MATMUL_STORY_DIR}/matmul_recipes/examples/${group}/gen_data.py
        ${MATMUL_STORY_DIR}/matmul_recipes/examples/${group}/verify_result.py
        DESTINATION ${MATMUL_INSTALL_PREFIX}/common/golden/${group}
    )
endforeach()

# The recommendation helper is installed next to the quantized recipe
# executables so it can discover them and reuse the colocated generator.
install(FILES
    ${MATMUL_STORY_DIR}/matmul_recipes/examples/quant_matmul_mxfp4/quant_matmul_mxfp4_algorithm_recommend.py
    DESTINATION ${MATMUL_INSTALL_PREFIX}/matmul_recipes/examples/quant_matmul_mxfp4
)

# Tutorial golden scripts stay under the tutorial-specific install tree because
# the tutorial launcher resolves them relative to that directory.
install(DIRECTORY
    ${MATMUL_STORY_DIR}/matmul_tutorials/golden/
    DESTINATION ${MATMUL_INSTALL_PREFIX}/matmul_tutorials/golden
    FILES_MATCHING
    PATTERN "*.py"
)
