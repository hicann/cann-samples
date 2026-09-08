#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: $0 <cannsim|msopprof> <tutorial> <build-dir> <output-dir> [trace|full]" >&2
    exit 2
fi

profile_mode=$1
tutorial=$2
build_dir=$3
output_dir=$4
cannsim_case=${5:-trace}
binary_root="${build_dir}/Samples/2_Performance/vf_data_transform_story/vf_data_transform_tutorials"

case "${tutorial}" in
    half_to_int8)
        stage_names=(0_dintlv_load 1_complementary_cast 2_or_merge 3_contiguous_store)
        stage_binaries=(
            vf_data_transform_half_to_int8_0_dintlv_load
            vf_data_transform_half_to_int8_1_complementary_cast
            vf_data_transform_half_to_int8_2_or_merge
            vf_data_transform_half_to_int8
        )
        ;;
    fp32_to_fp8)
        stage_names=(0_dintlv_b32_load 1_complementary_cast 2_pack_b16_store 3_shared_b8_mask)
        stage_binaries=(
            vf_data_transform_fp32_to_fp8_0_dintlv_b32_load
            vf_data_transform_fp32_to_fp8_1_complementary_cast
            vf_data_transform_fp32_to_fp8_2_pack_b16_store
            vf_data_transform_fp32_to_fp8
        )
        ;;
    int8_to_half)
        stage_names=(0_compact_b8_load 1_complementary_cast 2_intlv_b16_store 3_b8_cast_mask)
        stage_binaries=(
            vf_data_transform_int8_to_half_0_compact_b8_load
            vf_data_transform_int8_to_half_1_complementary_cast
            vf_data_transform_int8_to_half_2_intlv_b16_store
            vf_data_transform_int8_to_half
        )
        ;;
    int4_to_bf16)
        stage_names=(0_unpack4_load 1_single_cast 2_separate_views 3_b8_predicate)
        stage_binaries=(
            vf_data_transform_int4_to_bf16_0_unpack4_load
            vf_data_transform_int4_to_bf16_1_single_cast
            vf_data_transform_int4_to_bf16_2_separate_views
            vf_data_transform_int4_to_bf16
        )
        ;;
    fp4_to_fp8)
        stage_names=(0_raw_field_mapping 1_unpack_packed_byte 2_shift_select_and 3_hoist_constants 4_fold_scale)
        stage_binaries=(
            vf_data_transform_fp4_to_fp8_0_raw_field_mapping
            vf_data_transform_fp4_to_fp8_1_unpack_packed_byte
            vf_data_transform_fp4_to_fp8_2_shift_select_and
            vf_data_transform_fp4_to_fp8_3_hoist_constants
            vf_data_transform_fp4_to_fp8
        )
        ;;
    bf16_nd2nz)
        stage_names=(0_data_block_copy 1_conflict_padding)
        stage_binaries=(
            vf_data_transform_bf16_nd2nz_0_data_block_copy
            vf_data_transform_bf16_nd2nz
        )
        ;;
    bf16_rope_nd2nz)
        stage_names=(0_pack_or_fusion 1_conflict_padding)
        stage_binaries=(
            vf_data_transform_bf16_rope_nd2nz_0_pack_or_fusion
            vf_data_transform_bf16_rope_nd2nz
        )
        ;;
    *)
        echo "Unknown tutorial: ${tutorial}" >&2
        exit 2
        ;;
esac

mkdir -p "${output_dir}"

case "${profile_mode}" in
    cannsim)
        if [[ "${cannsim_case}" != "trace" && "${cannsim_case}" != "full" ]]; then
            echo "CANNsim case must be trace or full: ${cannsim_case}" >&2
            exit 2
        fi
        simulator_command=cannsim
        if command -v npusim >/dev/null 2>&1; then
            simulator_command=npusim
        fi
        for stage_index in "${!stage_names[@]}"; do
            stage_name=${stage_names[stage_index]}
            binary_path="${binary_root}/${tutorial}/${stage_name}/${stage_binaries[stage_index]}"
            "${simulator_command}" record "${binary_path}" -s Ascend950 -g vf -n 0 \
                -u "--case ${cannsim_case}" -o "${output_dir}/${stage_name}"
        done
        ;;
    msopprof)
        if [[ $# -ne 4 ]]; then
            echo "msopprof mode does not accept a CANNsim case" >&2
            exit 2
        fi
        for run_index in 1 2 3; do
            for stage_index in "${!stage_names[@]}"; do
                stage_name=${stage_names[stage_index]}
                binary_path="${binary_root}/${tutorial}/${stage_name}/${stage_binaries[stage_index]}"
                msopprof --warm-up=5 --launch-count=1 --launch-skip-before-match=1 \
                    --aic-metrics=BasicInfo --replay-mode=kernel \
                    --output="${output_dir}/${stage_name}/run${run_index}" \
                    "${binary_path}" --case perf
            done
        done
        ;;
    *)
        echo "Unknown profiling mode: ${profile_mode}" >&2
        exit 2
        ;;
esac
