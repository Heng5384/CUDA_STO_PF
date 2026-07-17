#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/home/zhiheng/PF/CUDA_STO_PF_high_throughput_v1}
root=${2:-$repo/runs/high_throughput_pf_v1/benchmark_400cube}
case_timeout_s=${3:-1800}
base=$repo/reports/high_throughput_pf_v1/benchmark_inputs/stage0_strict_dt16.params
variable_base=$repo/reports/high_throughput_pf_v1/accuracy_inputs/variable_G9_dt16_to_dt4.params
template_meta=$repo/runs/high_throughput_pf_v1/stage0_current_B187_8000_raw/Results/ch_T400_cuda_512x1x1_dt0.000195_steps8000_xB0.030/stage0_current_B187_8000_raw/ctot_checkpoint_step008000_meta.json
planar=$root/planar_input

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo 'workstation_gpu_busy=true' >&2
    exit 75
fi

if [[ -e $root ]]; then
    echo "refusing_to_overwrite=$root" >&2
    exit 73
fi
mkdir -p "$root" "$planar"
python3 "$repo/scripts/prepare_high_throughput_planar_cube.py" \
    --out-dir "$planar" --size 400 --dx-nm 1 \
    --interface-half-width-nm 2 --slab-half-width-nm 100 --matrix-xB 0.03 \
    --template-meta "$template_meta" \
    > "$planar/generation.log"

make_params() {
    local candidate=$1 output=$2
    if [[ $candidate == variable_G9 ]]; then
        cp "$variable_base" "$output"
    else
        cp "$base" "$output"
    fi
    cat >> "$output" <<EOF

# 400-cube high-throughput benchmark numerical overlay
ctot_memory_perf_features=187
ctot_diagnostics_enabled=0
ctot_benchmark_suppress_field_output=1
ctot_performance_profile_enabled=1
ctot_full_audit_cadence=100
ctot_transport_gate_trajectory_diagnostics=0
ctot_transport_defect_v2_diagnostics=0
EOF
    if [[ $candidate == fixed_G9_dt4 ]]; then
        cat >> "$output" <<EOF
dt=7.81250000000000043e-04
ctot_residual_abs_tol=1.00000000000000006e-09
ctot_residual_rel_tol=1.00000000000000004e-10
ctot_numerics_contract=ctot_jichen_imex_bdf2_active_manifold_v1
ctot_automatic_dt_growth=0
ctot_step_max_retries=0
EOF
    fi
}

run_case() {
    local geometry=$1 candidate=$2
    local tag=${geometry}_${candidate}
    local run=$root/$tag
    mkdir -p "$run"
    make_params "$candidate" "$run/runtime.params"
    local init_args=()
    if [[ $geometry == P ]]; then
        init_args=(
            --init-mode raw_fields
            --init-phi-raw "$planar/phi_init.raw"
            --init-xB-raw "$planar/xB_init.raw"
            --init-Ctot-raw "$planar/Ctot_init.raw"
            --init-meta "$planar/init_meta.json"
        )
    else
        init_args=(--radius-phys-nm 100)
    fi
    (
        while true; do
            nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
                --format=csv,noheader,nounits || true
            sleep 1
        done
    ) > "$run/gpu_memory_samples.csv" &
    local monitor=$!
    set +e
    (
        cd "$run"
        /usr/bin/time -f 'wall=%e maxrss_kib=%M' -o time.txt \
            timeout "$case_timeout_s" \
            "$repo/main_cuda" 400 400 400 --pf-param-file "$run/runtime.params" \
            --nsteps 8 --out-every 1000 --csv-out-every 1000 \
            --ctot-full-audit-cadence 100 --init-case-tag "$tag" \
            "${init_args[@]}" > run.log 2>&1
    )
    local rc=$?
    set -e
    kill "$monitor" 2>/dev/null || true
    wait "$monitor" 2>/dev/null || true
    echo "$rc" > "$run/exit_code.txt"
    if (( rc != 0 )); then
        echo "benchmark_failed=$tag" >&2
        return "$rc"
    fi
}

for geometry in P M; do
    for candidate in strict_dt16 fixed_G9_dt4 variable_G9; do
        run_case "$geometry" "$candidate"
    done
done

echo 'benchmark_400cube_complete=true'
