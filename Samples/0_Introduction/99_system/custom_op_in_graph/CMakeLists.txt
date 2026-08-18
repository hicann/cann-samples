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

set(ASCEND_INC ${ASCEND_DIR}/include)

set(OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build_out/makepkg/packages/vendors/customize)
set(ONNX_PLUGIN_TARGET          cust_onnx_parsers)
set(ONNX_PLUGIN_TARGET_OUT_DIR  ${OUT_DIR}/framework/onnx)
set(TF_PLUGIN_TARGET            cust_tf_parsers)
set(TF_PLUGIN_TARGET_OUT_DIR    ${OUT_DIR}/framework/tensorflow)

file(MAKE_DIRECTORY ${ONNX_PLUGIN_TARGET_OUT_DIR} ${TF_PLUGIN_TARGET_OUT_DIR})

add_subdirectory(onnx_plugin)
add_subdirectory(tf_plugin)

file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/build_out/makepkg/set_env.bash [[#!/bin/bash
vendor_path=$(cd "$(dirname "${BASH_SOURCE[0]}")/packages/vendors/customize"; pwd)
export ASCEND_CUSTOM_OPP_PATH=${vendor_path}:${ASCEND_CUSTOM_OPP_PATH}
]])
