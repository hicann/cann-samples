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

set(MATMUL_STORY_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MATMUL_INSTALL_PREFIX "2_Performance/matmul_story")

# ----------------------------------------------------------------------------
# 公共编译/链接配置（AscendC 可执行文件）
# ----------------------------------------------------------------------------
set(MATMUL_ASCENDC_COMPILE_OPTS "-xasc" "--npu-arch=dav-3510" "-w" "-O3")

# ----------------------------------------------------------------------------
# matmul_recipes 示例：add_matmul_recipe(目标名 源文件相对 examples/ 的路径 组名)
# 组名用于安装路径及该组下的 gen_data.py/verify_result.py
# ----------------------------------------------------------------------------
function(add_matmul_recipe target_name source_under_examples group_name)
    add_executable(${target_name}
        ${MATMUL_STORY_DIR}/matmul_recipes/examples/${source_under_examples}
    )
    target_include_directories(${target_name} PRIVATE
        ${MATMUL_STORY_DIR}/common
        ${MATMUL_STORY_DIR}/matmul_recipes/include
    )
    target_compile_options(${target_name} PRIVATE ${MATMUL_ASCENDC_COMPILE_OPTS})
    target_link_libraries(${target_name} PRIVATE
        cann_samples::ascend_base
        cann_samples::tensor_api
    )
    install(TARGETS ${target_name}
        RUNTIME DESTINATION ${MATMUL_INSTALL_PREFIX}/matmul_recipes/${group_name}
    )
endfunction()

# 声明 recipes 目标（新增示例时在此追加）
add_matmul_recipe(quant_matmul_mxfp4_swat   quant_matmul_mxfp4/quant_matmul_mxfp4_swat.cpp   quant_matmul_mxfp4)

# ----------------------------------------------------------------------------
# matmul_tutorials 教程：add_matmul_tutorial(目标名 源文件相对 matmul_tutorials/ 的路径 子目录名)
# 子目录名即 0_naive、1_db，用于 include 与安装路径
# ----------------------------------------------------------------------------
function(add_matmul_tutorial target_name source_under_tutorials subdir)
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

# 声明 tutorials 目标（新增教程时在此追加）
add_matmul_tutorial(matmul_tutorial_mxfp4_base  0_naive/matmul_tutorial_mxfp4_base.cpp  0_naive)

# ============================================================================
# Install 规则说明
# ============================================================================
# 执行 cmake --install build --prefix ./build_out 后，安装目录与源码对应结构:
#
#   <install_prefix>/
#   └── 2_Performance/
#       └── matmul_story/
#           ├── matmul_recipes/
#           │   └── quant_matmul_mxfp4/
#           │       ├── quant_matmul_mxfp4_swat
#           │       ├── gen_data.py
#           │       └── verify_result.py
#           └── matmul_tutorials/
#               ├── 0_naive/
#               │   └── matmul_tutorial_mxfp4_base
#               └── golden/
#                   ├── gen_data.py
#                   └── verify_result.py
#
# 使用方式：gen_data.py 生成数据后，直接运行可执行文件即可完成计算并调用同级目录下的 verify_result.py 做校验。
# 扩展：新增 recipe 在 add_matmul_recipe 处加一行；新增 tutorial 在 add_matmul_tutorial 处加一行；
# 新增 recipe 组时在 RECIPE_GROUPS 中追加组名并保证该组目录下存在 gen_data.py。
# ============================================================================

# Install: 各 recipe 组的 gen_data.py、verify_result.py（按组名列表，每组安装到 matmul_recipes/<group>/）
set(RECIPE_GROUPS quant_matmul_mxfp4)  # 新增组时追加，如 quant_matmul_mxfp8
foreach(group IN LISTS RECIPE_GROUPS)
    install(FILES
        ${MATMUL_STORY_DIR}/matmul_recipes/examples/${group}/gen_data.py
        ${MATMUL_STORY_DIR}/matmul_recipes/examples/${group}/verify_result.py
        DESTINATION ${MATMUL_INSTALL_PREFIX}/matmul_recipes/${group}
    )
endforeach()

# Install: tutorials 共用 golden 脚本
install(DIRECTORY
    ${MATMUL_STORY_DIR}/matmul_tutorials/golden/
    DESTINATION ${MATMUL_INSTALL_PREFIX}/matmul_tutorials/golden
    FILES_MATCHING
    PATTERN "*.py"
)
