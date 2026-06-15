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

set(SUPPORTED_NPU_ARCHS dav-3510)
file(RELATIVE_PATH SAMPLE_PATH ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
if(NOT "${NPU_ARCH}" IN_LIST SUPPORTED_NPU_ARCHS)
    message(STATUS "Skip sample ${SAMPLE_PATH}: NPU_ARCH=${NPU_ARCH} is not supported")
    return()
endif()

set(REG_DM_SCENES
    scene1_dist_norm
    scene2_unpack_b16
    scene3_pack_b32
    scene4_first_element
    scene5_brc_b32
    scene6_block_copy
    scene7_gather_arange
    scene8_dintlv_intlv
    scene9_mask_dist
    scene10_unalign
)

foreach(scene ${REG_DM_SCENES})
    add_executable(${scene} src/${scene}.asc)
    target_link_directories(${scene} PRIVATE ${ASCEND_DIR}/lib64)
    target_compile_options(${scene} PRIVATE
        "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
        "$<$<COMPILE_LANGUAGE:ASC>:-O2>"
    )
    target_link_libraries(${scene}
        m
        dl
        platform
        tiling_api
        cann_samples::tensor_api
    )
    install(TARGETS ${scene}
        RUNTIME DESTINATION 1_Features/memory_optimization/reg_data_movement
    )
endforeach()
