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

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})

add_executable(quant_matmul_template_aswt ${CMAKE_CURRENT_SOURCE_DIR}/examples/quant_matmul_template_aswt.cpp)

target_include_directories(quant_matmul_template_aswt PRIVATE
    ${ASCEND_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_directories(quant_matmul_template_aswt PRIVATE ${ASCEND_DIR}/lib64)

target_compile_options(quant_matmul_template_aswt PRIVATE
    "-xasc"
    "--npu-arch=dav-3101"
    "-w"
    "-O3"
)

target_link_libraries(quant_matmul_template_aswt
    PRIVATE
    m
    dl
    platform
    tiling_api
)

# ============================================================================
# Install 规则说明
# ============================================================================
# 执行 cmake --install build --prefix ./build_out 后的目录结构:
#
#   build_out/
#   └── 2_Performance/
#       └── matmul_story/
#           ├── quant_matmul_template_aswt    <- 可执行文件
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
#   <install_prefix>/2_Performance/matmul_story/quant_matmul_template_aswt
install(TARGETS quant_matmul_template_aswt
    RUNTIME DESTINATION 2_Performance/matmul_story
)

# Install 规则 2: 安装 golden 验证脚本
# 将 common/golden/quant_matmul_mxfp4/ 目录下的所有 .py 文件安装到:
#   <install_prefix>/2_Performance/matmul_story/
# FILES_MATCHING PATTERN "*.py": 只复制 .py 文件，忽略其他文件
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/common/golden/quant_matmul_mxfp4/
    DESTINATION 2_Performance/matmul_story
    FILES_MATCHING
    PATTERN "*.py"
)
