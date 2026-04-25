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

add_executable(matmul_tutorial_mxfp4_swat_unitflag matmul_tutorial_mxfp4_swat_unitflag.asc)
target_include_directories(matmul_tutorial_mxfp4_swat_unitflag PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(matmul_tutorial_mxfp4_swat_unitflag PRIVATE ${MATMUL_ASCENDC_COMPILE_OPTS})
target_link_libraries(matmul_tutorial_mxfp4_swat_unitflag PRIVATE
    cann_samples::ascend_base
    cann_samples::tensor_api
)
install(TARGETS matmul_tutorial_mxfp4_swat_unitflag
    RUNTIME DESTINATION ${MATMUL_INSTALL_PREFIX}/matmul_tutorials
)
