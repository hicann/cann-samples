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

target_include_directories(full_quant_fused_infer_attention_score PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/examples
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/block
    ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/include/memcopy
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vf
)

target_compile_options(full_quant_fused_infer_attention_score PRIVATE
    "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>"
    "$<$<COMPILE_LANGUAGE:ASC>:-O3>"
)

target_link_libraries(full_quant_fused_infer_attention_score
    PRIVATE
    m
    dl
    platform
    tiling_api
)

install(TARGETS full_quant_fused_infer_attention_score
    RUNTIME DESTINATION 2_Performance/full_quant_fused_infer_attention_score_story
)

install(
    FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/common/golden/verify_result.py 
        ${CMAKE_CURRENT_SOURCE_DIR}/common/golden/gen_data.py
    DESTINATION 2_Performance/full_quant_fused_infer_attention_score_story
)
