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

configure_file(scripts/gen_data.py gen_data.py COPYONLY)

# Generate test data once per build, shared by all cases
set(HISTOGRAM_DATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/data)
add_custom_command(
    OUTPUT
        ${HISTOGRAM_DATA_DIR}/input/x.bin
        ${HISTOGRAM_DATA_DIR}/input/min.bin
        ${HISTOGRAM_DATA_DIR}/input/max.bin
        ${HISTOGRAM_DATA_DIR}/output/golden.bin
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HISTOGRAM_DATA_DIR}/input
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HISTOGRAM_DATA_DIR}/output
    COMMAND env -u LD_LIBRARY_PATH python3 ${CMAKE_CURRENT_BINARY_DIR}/gen_data.py --output ${HISTOGRAM_DATA_DIR}
    COMMENT "Generating histogram test data"
)
add_custom_target(histogram_test_data DEPENDS
    ${HISTOGRAM_DATA_DIR}/input/x.bin
    ${HISTOGRAM_DATA_DIR}/input/min.bin
    ${HISTOGRAM_DATA_DIR}/input/max.bin
    ${HISTOGRAM_DATA_DIR}/output/golden.bin
)

function(add_histogram_case SRC_FILE ENABLE_SIMT)
    get_filename_component(TARGET_NAME ${SRC_FILE} NAME_WE)
    set(TARGET_NAME "simt_histogram_${TARGET_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(${TARGET_NAME} PRIVATE DATA_DIR="${HISTOGRAM_DATA_DIR}")
    set(COMPILE_OPTS "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
                     "$<$<COMPILE_LANGUAGE:ASC>:-O3>")
    if(ENABLE_SIMT)
        list(APPEND COMPILE_OPTS "$<$<COMPILE_LANGUAGE:ASC>:--enable-simt>")
    endif()
    target_compile_options(${TARGET_NAME} PRIVATE ${COMPILE_OPTS})
    target_link_libraries(${TARGET_NAME} PRIVATE m dl platform tiling_api ascendcl runtime stdc++)
    add_dependencies(${TARGET_NAME} histogram_test_data)
    set(INSTALL_TARGETS ${INSTALL_TARGETS} ${TARGET_NAME} PARENT_SCOPE)
endfunction()

set(INSTALL_TARGETS "")

add_histogram_case(src/0_serial_scalar.asc OFF)    # Case 0: MTE+Vector baseline
add_histogram_case(src/1_basic_simt.asc ON)        # Case 1: Basic SIMT
add_histogram_case(src/2_mem_optimized.asc ON)     # Case 2: float4 vectorization
add_histogram_case(src/3_advanced_tuning.asc ON)   # Case 3: launch_bounds + GridDim

add_custom_target(simt_histogram_story DEPENDS ${INSTALL_TARGETS})

install(TARGETS ${INSTALL_TARGETS}
    RUNTIME DESTINATION 2_Performance/simt_histogram_story
)
install(FILES scripts/gen_data.py
    DESTINATION 2_Performance/simt_histogram_story
)
