#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

project_path=$(cd "$(dirname "$0")"; pwd)
repo_root=$(cd "${project_path}/../../../.."; pwd)

if [ "x$1" == "xclean" ] 2>/dev/null; then
  rm -rf ${project_path}/build_out
  echo "[INFO] Clean successfully."
  exit 0
fi

log() {
  cur_date=`date +"%Y-%m-%d %H:%M:%S"`
  echo "[$cur_date] "$1
}

NPU_ARCH=${NPU_ARCH:-dav-3510}

log "[INFO] Cmake configure begin."
cmake -S ${repo_root} -B ${repo_root}/build -DNPU_ARCH=${NPU_ARCH}
if [ $? -ne 0 ]; then
  log "[ERROR] Please check cmake configure result."
  exit 1
fi

log "[INFO] Build begin."
cmake --build ${repo_root}/build --target cust_onnx_parsers cust_tf_parsers --parallel
if [ $? -ne 0 ]; then
  log "[ERROR] Please check build result."
  exit 1
fi

log "[INFO] Build successfully."
log "[INFO] Output in ${project_path}/build_out/makepkg/."
log "[INFO] Deploy: source ${project_path}/build_out/makepkg/set_env.bash"
