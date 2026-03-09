#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#
# CI Build Script for ops-samples
# 功能: 清理、配置、编译、安装 ops-samples 项目
# 用法: bash .ci/build.sh

set -e  # 任何命令失败时立即退出脚本
set -x  # 打印执行的命令，便于 CI 日志调试

# 定位项目根目录（.ci 的上级目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 清理旧的构建产物
rm -rf build
rm -rf build_out

# CMake 配置
cmake -S . -B build

# 并行编译
cmake --build build --parallel

# 安装到 build_out 目录
cmake --install build --prefix ./build_out
