#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/home/zhiheng/PF/CUDA_STO_PF_high_throughput_v1}
source_root=${2:-/home/zhiheng/PF/CUDA_STO_PF_lie_be_v2_20260716/runs/T400_longtime_v1/coarse4_runs/growth_N512_shift0_dt16_pre_event_freeze/run/Results/ch_T400_cuda_512x1x1_dt0.000195_steps20000_xB0.030/growth_N512_shift0_dt16_pre_event_freeze}
root=${3:-$repo/runs/high_throughput_pf_v1/variable_long_accuracy}
variable_base=$repo/reports/high_throughput_pf_v1/accuracy_inputs/variable_G9_dt16_to_dt4.params
fixed_base=$repo/reports/transport_residual_gate_v3/workstation_runs/V3_G9_dt4_6th_window_holdout/input/runtime.params
strict_base=$repo/reports/transport_residual_gate_v3/workstation_runs/V3_G12_dt32_6th_window_holdout/input/runtime.params

if [[ -e $root ]]; then echo "refusing_to_overwrite=$root" >&2; exit 73; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo 'workstation_gpu_busy=true' >&2; exit 75
fi
mkdir -p "$root"

common=(
    --init-mode raw_fields
    --init-phi-raw "$source_root/ctot_checkpoint_step005482_phi.raw"
    --init-xB-raw "$source_root/ctot_checkpoint_step005482_xB_alpha.raw"
    --init-Ctot-raw "$source_root/ctot_checkpoint_step005482_Ctot.raw"
    --init-Ctot-nm1-raw "$source_root/ctot_checkpoint_step005482_Ctot_nm1.raw"
    --init-phi-nm1-raw "$source_root/ctot_checkpoint_step005482_phi_nm1.raw"
    --init-meta "$source_root/ctot_checkpoint_step005482_meta.json"
    --ctot-full-audit-cadence 100
)

prepare_params() {
    local source=$1 output=$2
    cp "$source" "$output"
    cat >> "$output" <<EOF
ctot_memory_perf_features=187
ctot_diagnostics_enabled=1
ctot_benchmark_suppress_field_output=0
ctot_performance_profile_enabled=0
ctot_full_audit_cadence=100
ctot_transport_gate_trajectory_diagnostics=0
ctot_transport_defect_v2_diagnostics=0
EOF
}

run_case() {
    local tag=$1 base=$2 steps=$3
    local case=$root/$tag
    mkdir -p "$case"
    prepare_params "$base" "$case/runtime.params"
    (
        cd "$case"
        /usr/bin/time -f 'wall=%e maxrss_kib=%M' -o time.txt \
            "$repo/main_cuda" 512 1 1 --pf-param-file "$case/runtime.params" \
            --nsteps "$steps" --out-every "$steps" --csv-out-every 100 \
            --init-case-tag "$tag" "${common[@]}" > run.log 2>&1
    )
}

run_case variable_long_8000 "$variable_base" 8000
variable_meta=$(find "$root/variable_long_8000" -name 'ctot_checkpoint_step008000_meta.json' -print -quit)
delta_time=$(python3 - "$variable_meta" "$source_root/ctot_checkpoint_step005482_meta.json" <<'PY'
import json,sys
def read(path): return json.loads(open(path).read().replace(': nan', ': null'))
print(read(sys.argv[1])['bdf2_time_code'] - read(sys.argv[2])['bdf2_time_code'])
PY
)
fixed_steps=$(python3 - "$delta_time" <<'PY'
import sys
print(round(float(sys.argv[1])/7.8125e-4))
PY
)
strict_steps=$(python3 - "$delta_time" <<'PY'
import sys
print(round(float(sys.argv[1])/9.765625e-5))
PY
)
run_case fixed_G9_dt4_long "$fixed_base" "$fixed_steps"
run_case strict_G12_dt32_long "$strict_base" "$strict_steps"
printf 'variable_delta_time_code=%s\nfixed_steps=%s\nstrict_steps=%s\n' \
    "$delta_time" "$fixed_steps" "$strict_steps" > "$root/equal_time_plan.txt"
cat "$root/equal_time_plan.txt"
