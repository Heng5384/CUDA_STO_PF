#!/usr/bin/env bash
set -euo pipefail

repo=${1:-"$(pwd)"}
binary=${2:-"$repo/main_cuda"}
params=${3:-"$repo/task1_pf_only.params"}
output_root=${4:-"/tmp/codex_pf_task1_replay"}

run_mode() {
    local tag=$1
    local mode=$2
    local root="${output_root}_${tag}"
    rm -rf "$root"
    PF_COMPOSITION_TRACE_ONE_STEP=1 \
    PF_COMPOSITION_TRACE_STEP=1 \
    CUDA_STO_RESULTS_ROOT="$root" \
    "$binary" 8 8 8 0.0001 1 1 1 0 \
        --pf-param-file "$params" \
        --mode dynamics \
        --radius 2 \
        --pf-composition-mode legacy \
        --pf-y-update-mode "$mode" \
        --y-update-mass-projection-enabled 0 \
        --diagnostic-rsmd-enabled 0 \
        --elastic 0 \
        --enable-dynamics-mass-diagnostics \
        --dynamics-mass-diag-interval 1 \
        --init-case-tag "task1_${tag}"
}

run_mode L lagged_rhs
run_mode X x_transport_projection_split
run_mode Q q_transport_projection_split
