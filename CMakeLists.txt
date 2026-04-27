# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

set(VALID_NPU_ARCHS dav-3510 dav-2201)
if(NOT DEFINED NPU_ARCH OR NPU_ARCH STREQUAL "")
    message(FATAL_ERROR "NPU_ARCH is required. Example: cmake -S . -B build -DNPU_ARCH=dav-3510")
endif()
if(NOT "${NPU_ARCH}" IN_LIST VALID_NPU_ARCHS)
    message(FATAL_ERROR "Unsupported NPU_ARCH: ${NPU_ARCH}. Supported values: dav-3510, dav-2201")
endif()

include(cmake/ascend.cmake)
find_package(ASC)
project(cann-samples LANGUAGES C CXX ASC)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/tensor_api.cmake)
add_subdirectory(Samples)
