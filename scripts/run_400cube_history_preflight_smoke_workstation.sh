#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/home/zhiheng/PF/CUDA_STO_PF_high_throughput_v1}
root=${2:-$repo/runs/high_throughput_pf_v1/history_mass_fix_400cube_smoke_v1}
base=$repo/reports/high_throughput_pf_v1/benchmark_inputs/stage0_strict_dt16.params
planar=$repo/runs/high_throughput_pf_v1/benchmark_400cube_v3/planar_input

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo 'workstation_gpu_busy=true' >&2
    exit 75
fi
if [[ -e $root ]]; then
    echo "refusing_to_overwrite=$root" >&2
    exit 73
fi
mkdir -p "$root"
cp "$base" "$root/runtime.params"
cat >> "$root/runtime.params" <<'EOF'

# History preflight scale-equivalence smoke; accepted-step hard gates unchanged.
ctot_memory_perf_features=187
ctot_diagnostics_enabled=0
ctot_benchmark_suppress_field_output=1
ctot_performance_profile_enabled=1
ctot_full_audit_cadence=100
dt=7.81250000000000043e-04
ctot_residual_abs_tol=1.00000000000000006e-09
ctot_residual_rel_tol=1.00000000000000004e-10
ctot_numerics_contract=ctot_jichen_imex_bdf2_active_manifold_v1
ctot_automatic_dt_growth=0
ctot_step_max_retries=0
EOF

sha256sum "$repo/main_cuda" "$repo/main_cuda.cu" \
    "$repo/bdf2_history_mass_utils.h" "$planar"/*.raw "$planar/init_meta.json" \
    > "$root/input_hashes.txt"

(
    while true; do
        nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits || true
        sleep 1
    done
) > "$root/gpu_memory_samples.csv" &
monitor=$!
set +e
(
    cd "$root"
    /usr/bin/time -f 'wall=%e maxrss_kib=%M' -o time.txt \
        timeout 900 "$repo/main_cuda" 400 400 400 \
        --pf-param-file "$root/runtime.params" --nsteps 4 \
        --out-every 1000 --csv-out-every 1000 --ctot-full-audit-cadence 100 \
        --init-case-tag history_mass_fix_400cube_smoke \
        --init-mode raw_fields \
        --init-phi-raw "$planar/phi_init.raw" \
        --init-xB-raw "$planar/xB_init.raw" \
        --init-Ctot-raw "$planar/Ctot_init.raw" \
        --init-meta "$planar/init_meta.json" > run.log 2>&1
)
rc=$?
set -e
kill "$monitor" 2>/dev/null || true
wait "$monitor" 2>/dev/null || true
echo "$rc" > "$root/exit_code.txt"

python3 "$repo/scripts/analyze_history_mass_fix_400cube_smoke.py" \
    --run-dir "$root" --exit-code "$rc"

echo 'history_mass_fix_400cube_smoke_pass=true'
