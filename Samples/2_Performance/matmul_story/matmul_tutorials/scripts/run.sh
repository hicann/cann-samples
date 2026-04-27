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
INSTALL_DIR="${REPO_ROOT}/build_out/2_Performance/matmul_story/matmul_tutorials"
DEFAULT_TUTORIAL_TARGET="matmul_tutorial_mxfp4_base"

TARGET=""
SKIP_BUILD=false
M=""
K=""
N=""

usage() {
    cat <<'EOF'
Usage: bash run.sh [OPTIONS] m k n

One-stop script: build, generate data, run, and verify a matmul tutorial step.

Positional arguments:
  m   Row count of matrix A
  k   Shared dimension of A and B (must be even)
  n   Column count of matrix B

Options:
  --target <name>    Tutorial executable to run (e.g. matmul_tutorial_mxfp4_swat).
                     When omitted, defaults to Step 0: matmul_tutorial_mxfp4_base.
  --skip-build       Skip the build/install phase (reuse existing build_out).
  -h, --help         Show this help message and exit.

Available tutorial executables (Step 0 – Step 7):
  matmul_tutorial_mxfp4_base                        (Step 0)
  matmul_tutorial_mxfp4_pingpong                    (Step 1)
  matmul_tutorial_mxfp4_swat                        (Step 2)
  matmul_tutorial_mxfp4_swat_balance                (Step 3)
  matmul_tutorial_mxfp4_swat_unitflag               (Step 4)
  matmul_tutorial_mxfp4_half1l1_ping_halfl1_pong    (Step 5)
  matmul_tutorial_mxfp4_memery_access_coalescing    (Step 6)
  matmul_tutorial_mxfp4_a_fullload                  (Step 7)

Examples:
  # Run with explicit target
  bash scripts/run.sh --target matmul_tutorial_mxfp4_swat 256 256 256

  # Default Step 0 + skip rebuild
  bash scripts/run.sh --skip-build 256 256 256

EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            [[ -z "${2:-}" ]] && { echo "ERROR: --target requires a value"; usage; exit 1; }
            TARGET="$2"; shift 2 ;;
        --skip-build)
            SKIP_BUILD=true; shift ;;
        -h|--help)
            usage; exit 0 ;;
        -*)
            echo "ERROR: unknown option: $1"; usage; exit 1 ;;
        *)
            if   [[ -z "$M" ]]; then M="$1"
            elif [[ -z "$K" ]]; then K="$1"
            elif [[ -z "$N" ]]; then N="$1"
            else echo "ERROR: unexpected argument: $1"; usage; exit 1
            fi
            shift ;;
    esac
done

if [[ -z "$M" || -z "$K" || -z "$N" ]]; then
    echo "ERROR: m, k, n are required."
    usage
    exit 1
fi

# ── 1. Build ────────────────────────────────────────────────────────────────
if [[ "$SKIP_BUILD" != true ]]; then
    echo "=== Building project (via .ci/build.sh) ==="
    bash "${REPO_ROOT}/.ci/build.sh" dav-3510
fi

if [[ ! -d "$INSTALL_DIR" ]]; then
    echo "ERROR: Install directory not found: $INSTALL_DIR"
    echo "       Run without --skip-build to perform a full build first."
    exit 1
fi
cd "$INSTALL_DIR"

# ── 2. Generate test data ──────────────────────────────────────────────────
echo "=== Generating test data (m=$M k=$K n=$N) ==="
python3 gen_data.py "$M" "$K" "$N"

# ── 3. Determine target executable ─────────────────────────────────────────
if [[ -z "$TARGET" ]]; then
    TARGET="${DEFAULT_TUTORIAL_TARGET}"
    echo "    Using default Step 0 target: $TARGET"
fi

if [[ ! -x "./$TARGET" ]]; then
    echo "ERROR: Executable not found or not executable: $INSTALL_DIR/$TARGET"
    exit 1
fi

# ── 4. Run ──────────────────────────────────────────────────────────────────
echo "=== Running ./$TARGET $M $K $N ==="
"./$TARGET" "$M" "$K" "$N"
