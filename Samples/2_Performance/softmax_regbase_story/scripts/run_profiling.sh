#!/bin/bash
# Run msprof for each softmax_regbase case, compute average task_time.
# Usage: bash run_profiling.sh [build_dir] [rounds]
#
# Requires: CANN env sourced, executables already built.

BUILD_DIR="${1:-build}"
ROUNDS="${2:-3}"
SAMPLE_DIR="Samples/2_Performance/softmax_regbase_story"
PROF_BASE="./prof_data"

CASES=(
    "0_membase:softmax_regbase_0_membase"
    "1_vf_fused:softmax_regbase_1_vf_fused"
    "2_binary_fold:softmax_regbase_2_binary_fold"
    "3_multi_row:softmax_regbase_3_multi_row"
    "4_pipeline:softmax_regbase_4_pipeline"
    "5_bigtile:softmax_regbase_5_bigtile"
    "6_merged_vf:softmax_regbase_6_merged_vf"
    "7_merged_vf_direct:softmax_regbase_7_merged_vf_direct"
)

# Build header
HEADER="Case"
for r in $(seq 1 $ROUNDS); do
    HEADER="${HEADER},Run${r}"
done
HEADER="${HEADER},Avg(us)"

echo "=========================================="
echo " Softmax Regbase Profiling (${ROUNDS} rounds)"
echo "=========================================="
printf "%-22s" "Case"
for r in $(seq 1 $ROUNDS); do
    printf " %10s" "Run${r}"
done
printf " %10s\n" "Avg(us)"
echo "------------------------------------------"

for entry in "${CASES[@]}"; do
    name="${entry%%:*}"
    target="${entry##*:}"
    exe="${BUILD_DIR}/${SAMPLE_DIR}/${target}"

    if [ ! -f "$exe" ]; then
        printf "%-22s %10s\n" "$name" "NOT FOUND"
        continue
    fi

    times=()
    for r in $(seq 1 $ROUNDS); do
        prof_dir="${PROF_BASE}/${name}_r${r}"
        rm -rf "$prof_dir"
        msprof --application="./${exe}" --output="${prof_dir}" > /dev/null 2>&1

        t=$(find "${prof_dir}" -name "task_time_*.csv" -exec grep "AI_VECTOR_CORE" {} \; | head -1 | cut -d',' -f6)
        if [ -z "$t" ]; then
            t="N/A"
        fi
        times+=("$t")
    done

    # Compute average
    sum=0
    valid=0
    for t in "${times[@]}"; do
        if [[ "$t" =~ ^[0-9.]+$ ]]; then
            sum=$(echo "$sum + $t" | bc -l)
            valid=$((valid + 1))
        fi
    done

    if [ $valid -gt 0 ]; then
        avg=$(echo "scale=3; $sum / $valid" | bc -l)
    else
        avg="N/A"
    fi

    # Print row
    printf "%-22s" "$name"
    for t in "${times[@]}"; do
        printf " %10s" "$t"
    done
    printf " %10s\n" "$avg"
done

echo "=========================================="
echo "Done. Raw data in ${PROF_BASE}/"
