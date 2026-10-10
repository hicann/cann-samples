# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

foreach(REQUIRED_VARIABLE IN ITEMS SOURCE_DIR GIT_EXECUTABLE SUBMODULE_PATH)
    if(NOT DEFINED ${REQUIRED_VARIABLE} OR "${${REQUIRED_VARIABLE}}" STREQUAL "")
        message(FATAL_ERROR "${REQUIRED_VARIABLE} is required to update a Git submodule")
    endif()
endforeach()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --git-common-dir
    WORKING_DIRECTORY "${SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_COMMON_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GIT_DIR_RESULT
)
if(NOT "${GIT_DIR_RESULT}" STREQUAL "0")
    message(FATAL_ERROR "Cannot locate the Git common directory for ${SOURCE_DIR}")
endif()
get_filename_component(GIT_COMMON_DIR "${GIT_COMMON_DIR}" ABSOLUTE BASE_DIR "${SOURCE_DIR}")

# Independent dependency targets share the superproject's Git config, also
# when invoked from different build directories or a linked Git worktree.
file(LOCK "${GIT_COMMON_DIR}/cann-samples-submodule-update.lock"
    GUARD PROCESS TIMEOUT 600 RESULT_VARIABLE LOCK_RESULT
)
if(NOT "${LOCK_RESULT}" STREQUAL "0")
    message(FATAL_ERROR "Cannot acquire the Git submodule update lock: ${LOCK_RESULT}")
endif()

set(UPDATE_ARGUMENTS submodule update --init)
if(RECURSIVE)
    list(APPEND UPDATE_ARGUMENTS --recursive)
endif()
list(APPEND UPDATE_ARGUMENTS -- "${SUBMODULE_PATH}")
execute_process(
    COMMAND "${GIT_EXECUTABLE}" ${UPDATE_ARGUMENTS}
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE UPDATE_RESULT
)
if(NOT "${UPDATE_RESULT}" STREQUAL "0")
    message(FATAL_ERROR "Failed to update Git submodule ${SUBMODULE_PATH}: ${UPDATE_RESULT}")
endif()
