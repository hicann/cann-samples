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

add_executable(full_quant_fused_infer_attention_score
${CMAKE_CURRENT_SOURCE_DIR}/examples/full_quant_fused_infer_attention_score_template.asc)

set(COMMON_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/examples
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/block
    ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/include/memcopy
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vf
    ${ASCEND_HOME}/include
    ${ASCEND_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include/basic_api/impl
    ${ASCEND_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include/basic_api
    ${ASCEND_HOME}/include/ascendc
    ${ASCEND_HOME}/include/ascendc/basic_api/interface
    ${ASCEND_HOME}/include/ascendc/basic_api
    ${ASCEND_HOME}/pkg_inc/op_common
    ${ASCEND_HOME}/pkg_inc/base
    ${ASCEND_HOME}/pkg_inc
)

target_include_directories(full_quant_fused_infer_attention_score PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${COMMON_INCLUDE_DIRS}
)

target_compile_options(full_quant_fused_infer_attention_score PRIVATE
    "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>"
    "$<$<COMPILE_LANGUAGE:ASC>:-O3>"
)

target_link_libraries(full_quant_fused_infer_attention_score
    PRIVATE
    cann_samples::ascend_base
)

# ============================================================================
# Install 规则说明
# ============================================================================
# 执行 cmake --install build --prefix ./build_out 后的目录结构:
#
#   build_out/
#   └── 2_Performance/
#       └── full_quant_fused_infer_attention_score_story/
#           ├── full_quant_fused_infer_attention_score    <- 可执行文件
#           ├── gen_data.py                   <- 生成测试数据的脚本
#           └── verify_result.py              <- 验证计算结果的脚本
#
# 使用方式:
#   1. 运行 gen_data.py 生成测试输入数据和期望输出(golden)
#   2. 运行 quant_matmul_template_aswt 执行算子计算
#   3. 运行 verify_result.py 对比实际输出与期望输出，验证正确性
# ============================================================================

# Install 规则 1: 安装可执行文件
# 将编译生成的 quant_matmul_template_aswt 可执行文件安装到:
#   <install_prefix>/2_Performance/full_quant_fused_infer_attention_score_story/full_quant_fused_infer_attention_score
install(TARGETS full_quant_fused_infer_attention_score
    RUNTIME DESTINATION 2_Performance/full_quant_fused_infer_attention_score_story
)

# Install 规则 2: 安装 golden 验证脚本
# 将 common/golden/ 目录下的所有 .py 文件安装到:
#   <install_prefix>/2_Performance/full_quant_fused_infer_attention_score_story/
# FILES_MATCHING PATTERN "*.py": 只复制 .py 文件，忽略其他文件
install(
    FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/common/golden/verify_result.py 
        ${CMAKE_CURRENT_SOURCE_DIR}/common/golden/gen_data.py
    DESTINATION 
        DESTINATION 2_Performance/full_quant_fused_infer_attention_score_story
)
