# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_repo_root() {
    local dir="$SCRIPT_DIR"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/.ci/build.sh" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

REPO_ROOT="$(find_repo_root || true)"
if [[ -z "$REPO_ROOT" ]]; then
    echo "ERROR: cannot locate repo root containing .ci/build.sh"
    exit 1
fi

INSTALL_DIR="${REPO_ROOT}/build_out/2_Performance/matmul_story/matmul_recipes/quant_matmul_mxfp8"
TARGET=""
SKIP_BUILD=false
M=""
K=""
N=""
TRANSA=""
TRANSB=""

usage() {
    cat <<'EOF'
Usage: bash run.sh [OPTIONS] m k n [transA transB]

Options:
  --target <name>    Specify executable name to run.
  --skip-build       Skip build/install stage.
  -h, --help         Show this help.

When --target is omitted, run.sh auto-selects target via:
  python3 quant_matmul_mxfp8_algorithm_recommend.py m k n [transA transB] --print-target

Defaults:
  transA=false (0), transB=true (1)
EOF
}

normalize_transpose_flag() {
    local raw="$1"
    local name="$2"
    local normalized="${raw,,}"
    case "$normalized" in
        0|false|f) echo "0" ;;
        1|true|t) echo "1" ;;
        *)
            echo "ERROR: ${name} must be one of 0/1/true/false/t/f"
            usage
            exit 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            [[ -z "${2:-}" ]] && { echo "ERROR: --target needs a value"; exit 1; }
            TARGET="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "ERROR: unknown option: $1"
            usage
            exit 1
            ;;
        *)
            if [[ -z "$M" ]]; then
                M="$1"
            elif [[ -z "$K" ]]; then
                K="$1"
            elif [[ -z "$N" ]]; then
                N="$1"
            elif [[ -z "$TRANSA" ]]; then
                TRANSA="$1"
            elif [[ -z "$TRANSB" ]]; then
                TRANSB="$1"
            else
                echo "ERROR: unexpected argument: $1"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$M" || -z "$K" || -z "$N" ]]; then
    echo "ERROR: m k n are required"
    usage
    exit 1
fi

if [[ -n "$TRANSA" && -z "$TRANSB" ]]; then
    echo "ERROR: transA/transB must be both provided or both omitted"
    usage
    exit 1
fi

if [[ -z "$TRANSA" && -z "$TRANSB" ]]; then
    TRANSA="0"
    TRANSB="1"
else
    TRANSA="$(normalize_transpose_flag "$TRANSA" "transA")"
    TRANSB="$(normalize_transpose_flag "$TRANSB" "transB")"
fi

if [[ "$SKIP_BUILD" != true ]]; then
    bash "${REPO_ROOT}/.ci/build.sh"
fi

if [[ ! -d "$INSTALL_DIR" ]]; then
    echo "ERROR: install dir not found: $INSTALL_DIR"
    echo "Hint: remove --skip-build for full build/install."
    exit 1
fi

cd "$INSTALL_DIR"

python3 gen_data.py "$M" "$K" "$N" "$TRANSA" "$TRANSB"

if [[ -z "$TARGET" ]]; then
    TARGET="$(python3 quant_matmul_mxfp8_algorithm_recommend.py "$M" "$K" "$N" "$TRANSA" "$TRANSB" --print-target)"
fi

if [[ -z "$TARGET" ]]; then
    echo "ERROR: failed to select a target executable"
    exit 1
fi

if [[ ! -x "./$TARGET" ]]; then
    echo "ERROR: executable not found: $INSTALL_DIR/$TARGET"
    exit 1
fi

"./$TARGET" "$M" "$K" "$N" "$TRANSA" "$TRANSB"
