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

INSTALL_DIR="${REPO_ROOT}/build_out/2_Performance/matmul_story/matmul_recipes/quant_matmul_hifp8"
TARGET=""
TARGET_FROM_CLI=false
MODE=""
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
  --mode tt|tc       TT: pertensor scales (gen_data_tt.py, quant_matmul_hifp8_tt).
                     TC: per-channel scale (gen_data_tc.py, quant_matmul_hifp8_tc).
                     Default: tt when --target is omitted.
  --target <name>    Specify executable name (quant_matmul_hifp8_tt or quant_matmul_hifp8_tc). Overrides --mode.
  --skip-build       Skip build/install stage.
  -h, --help         Show this help.

Defaults:
  transA=false (0), transB=true (1)
  Python gen_data_*.py expects literal true/false; this script converts from 0/1|true/false flags.

Install dir contains quant_matmul_hifp8_tt and quant_matmul_hifp8_tc.
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

transpose01_to_py_bool() {
    case "$1" in
        0) echo "false" ;;
        1) echo "true" ;;
        *)
            echo "ERROR: internal: expected trans flag 0 or 1, got $1"
            exit 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            [[ -z "${2:-}" ]] && { echo "ERROR: --target needs a value"; exit 1; }
            TARGET="$2"
            TARGET_FROM_CLI=true
            shift 2
            ;;
        --mode)
            [[ -z "${2:-}" ]] && { echo "ERROR: --mode needs a value (tt or tc)"; exit 1; }
            MODE="${2,,}"
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
    bash "${REPO_ROOT}/.ci/build.sh" dav-3510
fi

if [[ ! -d "$INSTALL_DIR" ]]; then
    echo "ERROR: install dir not found: $INSTALL_DIR"
    echo "Hint: remove --skip-build for full build/install."
    exit 1
fi

cd "$INSTALL_DIR"

PY_TRANSA="$(transpose01_to_py_bool "$TRANSA")"
PY_TRANSB="$(transpose01_to_py_bool "$TRANSB")"

if [[ -z "$TARGET" ]]; then
    case "$MODE" in
        ""|tt)
            TARGET="quant_matmul_hifp8_tt"
            ;;
        tc)
            TARGET="quant_matmul_hifp8_tc"
            ;;
        *)
            echo "ERROR: --mode must be tt or tc (got: ${MODE})"
            usage
            exit 1
            ;;
    esac
fi

case "$TARGET" in
    quant_matmul_hifp8_tt)
        GEN_DATA_SCRIPT="gen_data_tt.py"
        VARIANT_LABEL="TT (pertensor scales)"
        ;;
    quant_matmul_hifp8_tc)
        GEN_DATA_SCRIPT="gen_data_tc.py"
        VARIANT_LABEL="TC (per-channel scale)"
        ;;
    *)
        echo "ERROR: unsupported target: ${TARGET} (expected quant_matmul_hifp8_tt or quant_matmul_hifp8_tc)"
        exit 1
        ;;
esac

if [[ ! -f "./${GEN_DATA_SCRIPT}" ]]; then
    echo "ERROR: data script not found: $INSTALL_DIR/${GEN_DATA_SCRIPT}"
    exit 1
fi

if [[ ! -x "./$TARGET" ]]; then
    echo "ERROR: executable not found: $INSTALL_DIR/$TARGET"
    exit 1
fi

echo ""
if [[ "${TARGET_FROM_CLI}" == true ]]; then
    echo "[run.sh] Running user-specified executable (--target): ${TARGET} (${VARIANT_LABEL})"
else
    echo "[run.sh] Running ${VARIANT_LABEL}: ${TARGET}"
fi
echo ""

python3 "${GEN_DATA_SCRIPT}" "$M" "$K" "$N" "$PY_TRANSA" "$PY_TRANSB"

"./$TARGET" "$M" "$K" "$N" "$TRANSA" "$TRANSB"
