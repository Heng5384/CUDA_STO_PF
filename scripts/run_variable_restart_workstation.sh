#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/home/zhiheng/PF/CUDA_STO_PF_high_throughput_v1}
source_root=${2:-/home/zhiheng/PF/CUDA_STO_PF_lie_be_v2_20260716/runs/T400_longtime_v1/coarse4_runs/growth_N512_shift0_dt16_pre_event_freeze/run/Results/ch_T400_cuda_512x1x1_dt0.000195_steps20000_xB0.030/growth_N512_shift0_dt16_pre_event_freeze}
root=$repo/runs/high_throughput_pf_v1/variable_restart
params=$repo/reports/high_throughput_pf_v1/accuracy_inputs/variable_G9_dt16_to_dt4.params

rm -rf "$root"
mkdir -p "$root"

initial_args=(
    --init-mode raw_fields
    --init-phi-raw "$source_root/ctot_checkpoint_step005482_phi.raw"
    --init-xB-raw "$source_root/ctot_checkpoint_step005482_xB_alpha.raw"
    --init-Ctot-raw "$source_root/ctot_checkpoint_step005482_Ctot.raw"
    --init-Ctot-nm1-raw "$source_root/ctot_checkpoint_step005482_Ctot_nm1.raw"
    --init-phi-nm1-raw "$source_root/ctot_checkpoint_step005482_phi_nm1.raw"
    --init-meta "$source_root/ctot_checkpoint_step005482_meta.json"
)

run_initial_case() {
    local tag=$1 steps=$2
    local run=$root/$tag
    mkdir -p "$run"
    (
        cd "$run"
        "$repo/main_cuda" 512 1 1 --pf-param-file "$params" \
            --nsteps "$steps" --out-every "$steps" --csv-out-every 1 \
            --ctot-full-audit-cadence 100 --init-case-tag "$tag" \
            "${initial_args[@]}" > run.log 2>&1
    )
}

checkpoint_stem() {
    find "$1" -name 'ctot_checkpoint_step*_meta.json' -print | sort | tail -1 | sed 's/_meta.json$//'
}

run_initial_case continuous_40 40
run_initial_case split_first_20 20
first_stem=$(checkpoint_stem "$root/split_first_20")

mkdir -p "$root/split_second_20"
(
    cd "$root/split_second_20"
    "$repo/main_cuda" 512 1 1 --pf-param-file "$params" \
        --nsteps 20 --out-every 20 --csv-out-every 1 \
        --ctot-full-audit-cadence 100 --init-case-tag split_second_20 \
        --init-mode raw_fields \
        --init-phi-raw "${first_stem}_phi.raw" \
        --init-xB-raw "${first_stem}_xB_alpha.raw" \
        --init-Ctot-raw "${first_stem}_Ctot.raw" \
        --init-Ctot-nm1-raw "${first_stem}_Ctot_nm1.raw" \
        --init-phi-nm1-raw "${first_stem}_phi_nm1.raw" \
        --init-meta "${first_stem}_meta.json" > run.log 2>&1
)

continuous_stem=$(checkpoint_stem "$root/continuous_40")
restart_stem=$(checkpoint_stem "$root/split_second_20")
fields=(Ctot Ctot_nm1 phi phi_nm1 xB_alpha)
all_equal=true
{
    echo 'field,continuous_sha256,restart_sha256,bitwise_equal'
    for field in "${fields[@]}"; do
        a=$(sha256sum "${continuous_stem}_${field}.raw" | awk '{print $1}')
        b=$(sha256sum "${restart_stem}_${field}.raw" | awk '{print $1}')
        equal=false
        if [[ $a == "$b" ]]; then equal=true; else all_equal=false; fi
        printf '%s,%s,%s,%s\n' "$field" "$a" "$b" "$equal"
    done
} > "$root/restart_hash_comparison.csv"

python3 - "$continuous_stem" "$restart_stem" "$root/restart_summary.json" "$all_equal" <<'PY'
import json, sys
continuous = json.load(open(sys.argv[1] + '_meta.json'))
restart = json.load(open(sys.argv[2] + '_meta.json'))
keys = [
    'bdf2_time_code', 'bdf2_dt_n', 'bdf2_dt_nm1',
    'variable_controller_previous_error', 'variable_controller_next_dt',
    'bdf2_history_valid', 'variable_controller_valid',
]
summary = {
    'bitwise_equal': sys.argv[4] == 'true',
    'continuous': {key: continuous.get(key) for key in keys},
    'restart': {key: restart.get(key) for key in keys},
    'metadata_equal': all(continuous.get(key) == restart.get(key) for key in keys),
}
json.dump(summary, open(sys.argv[3], 'w'), indent=2)
open(sys.argv[3], 'a').write('\n')
print(json.dumps(summary, indent=2))
PY

test "$all_equal" = true
