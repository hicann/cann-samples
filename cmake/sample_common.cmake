# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

macro(cann_sample_check_arch)
    set(_cann_sample_expected_archs ${ARGN})
    file(RELATIVE_PATH _cann_sample_path ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    if(NOT "${NPU_ARCH}" IN_LIST _cann_sample_expected_archs)
        message(STATUS "Skip sample ${_cann_sample_path}: NPU_ARCH=${NPU_ARCH} is not supported")
        return()
    endif()
endmacro()
