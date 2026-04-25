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

add_executable(scale_cache main.asc)

target_compile_options(scale_cache PRIVATE
    "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>"
    "$<$<COMPILE_LANGUAGE:ASC>:-O3>"
)

target_link_libraries(scale_cache
    cann_samples::ascend_base
    cann_samples::tensor_api
)

install(TARGETS scale_cache
    RUNTIME DESTINATION 1_Features/memory_optimization/scale_cache
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/
    DESTINATION 1_Features/memory_optimization/scale_cache
    FILES_MATCHING PATTERN "*"
)
