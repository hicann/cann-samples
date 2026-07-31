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


cmake_minimum_required(VERSION 3.5.1)
project(cust_framework_plugins LANGUAGES CXX)

# ---- 编译器：必须用 g++，这是 Host 侧代码 ----
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# ---- CANN SDK include 路径 ----
if(NOT DEFINED ASCEND_INC)
    if(DEFINED ENV{ASCEND_TENSOR_COMPILER_INCLUDE})
        set(ASCEND_INC $ENV{ASCEND_TENSOR_COMPILER_INCLUDE})
    elseif(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_INC $ENV{ASCEND_HOME_PATH}/include)
    else()
        message(FATAL_ERROR "请设置 ASCEND_TENSOR_COMPILER_INCLUDE 环境变量，或通过 -DASCEND_INC=... 指定 CANN include 路径")
    endif()
endif()
message(STATUS "ASCEND_INC=${ASCEND_INC}")
include_directories(${ASCEND_INC})

# ---- target 名称与输出目录 ----
set(OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/makepkg/packages/vendors/customize)
set(ONNX_PLUGIN_TARGET          cust_onnx_parsers)
set(ONNX_PLUGIN_TARGET_OUT_DIR  ${OUT_DIR}/framework/onnx)
set(TF_PLUGIN_TARGET            cust_tf_parsers)
set(TF_PLUGIN_TARGET_OUT_DIR    ${OUT_DIR}/framework/tensorflow)

# ---- 子目录 ----
add_subdirectory(onnx_plugin)
add_subdirectory(tf_plugin)
