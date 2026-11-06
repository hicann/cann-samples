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
# Thin wrapper for manifest-driven CI functional tests.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MANIFEST_PATH="${MANIFEST_PATH:-tests/ci_functional_test.yaml}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-test_artifacts/ci_functional}"

bash .ci/build.sh "${NPU_ARCH:-dav-3510}"

python3 .ci/run_ci_functional.py \
    --manifest "${MANIFEST_PATH}" \
    --artifacts-dir "${ARTIFACTS_DIR}" \
    "$@"
