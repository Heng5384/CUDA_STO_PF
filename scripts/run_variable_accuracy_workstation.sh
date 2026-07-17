#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/home/zhiheng/PF/CUDA_STO_PF_high_throughput_v1}
source_root=${2:-/home/zhiheng/PF/CUDA_STO_PF_lie_be_v2_20260716/runs/T400_longtime_v1/coarse4_runs/growth_N512_shift0_dt16_pre_event_freeze/run/Results/ch_T400_cuda_512x1x1_dt0.000195_steps20000_xB0.030/growth_N512_shift0_dt16_pre_event_freeze}
root=$repo/runs/high_throughput_pf_v1/variable_accuracy
variable_params=$repo/reports/high_throughput_pf_v1/accuracy_inputs/variable_G9_dt16_to_dt4.params
fixed_params=$repo/reports/transport_residual_gate_v3/workstation_runs/V3_G9_dt4_6th_window_holdout/input/runtime.params
strict_params=$repo/reports/transport_residual_gate_v3/workstation_runs/V3_G12_dt32_6th_window_holdout/input/runtime.params

common=(
    --init-mode raw_fields
    --init-phi-raw "$source_root/ctot_checkpoint_step005482_phi.raw"
    --init-xB-raw "$source_root/ctot_checkpoint_step005482_xB_alpha.raw"
    --init-Ctot-raw "$source_root/ctot_checkpoint_step005482_Ctot.raw"
    --init-Ctot-nm1-raw "$source_root/ctot_checkpoint_step005482_Ctot_nm1.raw"
    --init-phi-nm1-raw "$source_root/ctot_checkpoint_step005482_phi_nm1.raw"
    --init-meta "$source_root/ctot_checkpoint_step005482_meta.json"
    --out-every 2000 --csv-out-every 1
    --ctot-full-audit-cadence 100
)

mkdir -p "$root/variable"
(
    cd "$root/variable"
    /usr/bin/time -f 'wall=%e maxrss_kib=%M' \
        "$repo/main_cuda" 512 1 1 --pf-param-file "$variable_params" \
        --nsteps 200 --out-every 200 --init-case-tag variable_G9_200 \
        "${common[@]}" > run.log 2>&1
)

variable_meta=$(find "$root/variable" -name 'ctot_checkpoint_step000200_meta.json' -print -quit)
delta_time=$(python3 - "$variable_meta" "$source_root/ctot_checkpoint_step005482_meta.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    final = json.load(f)
with open(sys.argv[2], encoding="utf-8") as f:
    initial = json.load(f)
print(final["bdf2_time_code"] - initial["bdf2_time_code"])
PY
)

fixed_steps=$(python3 - "$delta_time" <<'PY'
import sys
print(max(1, round(float(sys.argv[1]) / 7.8125e-4)))
PY
)
strict_steps=$(python3 - "$delta_time" <<'PY'
import sys
print(max(1, round(float(sys.argv[1]) / 9.765625e-5)))
PY
)

mkdir -p "$root/fixed_G9_dt4" "$root/strict_G12_dt32"
(
    cd "$root/fixed_G9_dt4"
    /usr/bin/time -f 'wall=%e maxrss_kib=%M' \
        "$repo/main_cuda" 512 1 1 --pf-param-file "$fixed_params" \
        --nsteps "$fixed_steps" --out-every "$fixed_steps" \
        --init-case-tag fixed_G9_dt4_equal_time \
        "${common[@]}" > run.log 2>&1
)
(
    cd "$root/strict_G12_dt32"
    /usr/bin/time -f 'wall=%e maxrss_kib=%M' \
        "$repo/main_cuda" 512 1 1 --pf-param-file "$strict_params" \
        --nsteps "$strict_steps" --out-every "$strict_steps" \
        --init-case-tag strict_G12_dt32_equal_time \
        "${common[@]}" > run.log 2>&1
)

printf 'variable_delta_time_code=%s\nfixed_steps=%s\nstrict_steps=%s\n' \
    "$delta_time" "$fixed_steps" "$strict_steps"
