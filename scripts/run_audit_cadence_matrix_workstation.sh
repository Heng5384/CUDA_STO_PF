#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/home/zhiheng/PF/CUDA_STO_PF_high_throughput_v1}
base_params=${2:-$repo/runs/high_throughput_pf_v1/stage0_small3d/N32_B187/runtime.params}
root=$repo/runs/high_throughput_pf_v1/audit_cadence_matrix

mkdir -p "$root"
for cadence in 1 10 50 100; do
    case_dir=$root/cadence_${cadence}
    mkdir -p "$case_dir"
    (
        cd "$case_dir"
        /usr/bin/time -f 'wall=%e maxrss_kib=%M' \
            "$repo/main_cuda" 32 32 32 \
            --pf-param-file "$base_params" \
            --nsteps 20 --out-every 20 --csv-out-every 20 \
            --radius 8 --temperature-C 400 \
            --ctot-full-audit-cadence "$cadence" \
            --init-case-tag "audit_cadence_${cadence}" \
            > run.log 2>&1
    )
done
