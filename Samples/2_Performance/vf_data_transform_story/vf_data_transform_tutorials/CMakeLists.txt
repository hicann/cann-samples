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

function(vf_data_transform_add_stage TARGET_NAME ASC_FILE INSTALL_SUBDIR)
    add_executable(${TARGET_NAME} ${ASC_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${VF_DATA_TRANSFORM_TUTORIALS_DIR}/include
    )
    target_compile_definitions(${TARGET_NAME} PRIVATE
        VF_DATA_TRANSFORM_SOURCE_DIR="${VF_DATA_TRANSFORM_TUTORIALS_DIR}"
    )
    target_compile_options(${TARGET_NAME} PRIVATE ${VF_DATA_TRANSFORM_ASCENDC_COMPILE_OPTS})
    target_link_libraries(${TARGET_NAME} PRIVATE
        m
        dl
        platform
        tiling_api
        cann_samples::tensor_api
    )
    install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${VF_DATA_TRANSFORM_INSTALL_PREFIX}/${INSTALL_SUBDIR}
    )
endfunction()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/scripts)
configure_file(scripts/gen_data.py scripts/gen_data.py COPYONLY)
configure_file(scripts/verify_result.py scripts/verify_result.py COPYONLY)

add_subdirectory(half_to_int8)
add_subdirectory(fp32_to_fp8)
add_subdirectory(int8_to_half)
add_subdirectory(int4_to_bf16)
add_subdirectory(fp4_to_fp8)
add_subdirectory(bf16_nd2nz)
add_subdirectory(bf16_rope_nd2nz)

add_custom_target(vf_data_transform_tutorials DEPENDS
    vf_data_transform_half_to_int8_tutorial
    vf_data_transform_fp32_to_fp8_tutorial
    vf_data_transform_int8_to_half_tutorial
    vf_data_transform_int4_to_bf16_tutorial
    vf_data_transform_fp4_to_fp8_tutorial
    vf_data_transform_bf16_nd2nz_tutorial
    vf_data_transform_bf16_rope_nd2nz_tutorial
)

install(DIRECTORY scripts/
    DESTINATION ${VF_DATA_TRANSFORM_INSTALL_PREFIX}/scripts
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.sh"
)
