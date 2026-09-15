#!/usr/bin/env bash

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-block_attn_res_e2e}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${TARGET}" in
    block_attn_res_prepare|block_attn_res_update|block_attn_res_e2e) ;;
    *)
        echo "Usage: bash run.sh [block_attn_res_prepare|block_attn_res_update|block_attn_res_e2e]"
        exit 1
        ;;
esac

if [[ $# -lt 4 ]]; then
    echo "Usage: bash run.sh <target> T N S D [options]"
    echo "Example: bash run.sh block_attn_res_e2e 128 8 32 512 --template auto --repeat 10"
    exit 1
fi

if [[ ! -x "${SCRIPT_DIR}/${TARGET}" ]]; then
    echo "ERROR: ${SCRIPT_DIR}/${TARGET} does not exist; build and install the sample first."
    exit 1
fi

"${SCRIPT_DIR}/${TARGET}" "$@"
