#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 BUILD_ROOT REFERENCE_BINARY RUN_ROOT" >&2
    exit 2
fi

build_root=$1
reference_binary=$2
run_root=$3
binary="$build_root/main_cuda"

for required in \
    "$binary" \
    "$reference_binary" \
    "$build_root/prompt8_phase_kkt_retry.params" \
    "$build_root/prompt8_phase_kkt_direct_half_dt.params" \
    "$build_root/prompt8_phase_kkt_restart.params" \
    "$build_root/prompt8_phase_kkt_old_solver.params"; do
    [[ -e "$required" ]] || { echo "missing: $required" >&2; exit 2; }
done

if [[ -e "$run_root" ]]; then
    echo "refusing to overwrite existing run root: $run_root" >&2
    exit 2
fi
mkdir -p "$run_root"

run_standard() {
    local name=$1
    local executable=$2
    local params=$3
    local steps=$4
    local tag=$5
    local radius=${6:-3}
    mkdir -p "$run_root/$name"
    (
        cd "$run_root/$name"
        "$executable" 16 16 16 5e-5 "$steps" 1 1 0 \
            --pf-param-file "$params" --init-case-tag "$tag" \
            --radius "$radius" \
            > run.log 2>&1
    )
}

checkpoint_path() {
    local case_dir=$1
    local step=$2
    local field=$3
    local found
    found=$(find "$case_dir" -type f \
        -name "ctot_checkpoint_step${step}_${field}.raw" -print -quit)
    [[ -n "$found" ]] || {
        echo "missing checkpoint step=${step} field=${field} in ${case_dir}" >&2
        exit 2
    }
    printf '%s\n' "$found"
}

run_standard retry "$binary" \
    "$build_root/prompt8_phase_kkt_retry.params" 1 prompt8_retry
run_standard direct_half_dt "$binary" \
    "$build_root/prompt8_phase_kkt_direct_half_dt.params" 1 prompt8_direct_half_dt

run_standard restart_direct_two "$binary" \
    "$build_root/prompt8_phase_kkt_restart.params" 2 prompt8_restart_direct_two
run_standard restart_first "$binary" \
    "$build_root/prompt8_phase_kkt_restart.params" 1 prompt8_restart_first

restart_phi=$(checkpoint_path "$run_root/restart_first" 000001 phi)
restart_x=$(checkpoint_path "$run_root/restart_first" 000001 xB_alpha)
restart_C=$(checkpoint_path "$run_root/restart_first" 000001 Ctot)
restart_meta=${restart_phi%_phi.raw}_meta.json
mkdir -p "$run_root/restart_second"
(
    cd "$run_root/restart_second"
    "$binary" 16 16 16 5e-5 1 1 1 0 \
        --pf-param-file "$build_root/prompt8_phase_kkt_restart.params" \
        --init-case-tag prompt8_restart_second --init-mode raw_fields \
        --init-phi-raw "$restart_phi" --init-xB-raw "$restart_x" \
        --init-Ctot-raw "$restart_C" --init-meta "$restart_meta" \
        > run.log 2>&1
)

mkdir -p "$run_root/restart_solver_mismatch"
set +e
(
    cd "$run_root/restart_solver_mismatch"
    "$binary" 16 16 16 5e-5 1 1 1 0 \
        --pf-param-file "$build_root/prompt8_phase_kkt_old_solver.params" \
        --init-case-tag prompt8_restart_solver_mismatch --init-mode raw_fields \
        --init-phi-raw "$restart_phi" --init-xB-raw "$restart_x" \
        --init-Ctot-raw "$restart_C" --init-meta "$restart_meta" \
        > run.log 2>&1
)
mismatch_exit=$?
set -e
if [[ $mismatch_exit -eq 0 ]] || ! grep -q \
    'Ctot checkpoint phase-solver provenance missing or mismatched' \
    "$run_root/restart_solver_mismatch/run.log"; then
    echo "phase-solver restart mismatch did not fail fast" >&2
    exit 2
fi
printf '%s\n' "$mismatch_exit" > \
    "$run_root/restart_solver_mismatch/exit_code.txt"

run_standard old_solver_current "$binary" \
    "$build_root/prompt8_phase_kkt_old_solver.params" 1 prompt8_old_current 1
run_standard old_solver_reference "$reference_binary" \
    "$build_root/prompt8_phase_kkt_old_solver.params" 1 prompt8_old_reference 1

comparison_csv="$run_root/checkpoint_comparison.csv"
printf 'comparison,field,reference_path,candidate_path,bitwise_equal\n' \
    > "$comparison_csv"

compare_pair() {
    local comparison=$1
    local reference_dir=$2
    local reference_step=$3
    local candidate_dir=$4
    local candidate_step=$5
    local field reference candidate status
    for field in phi xB_alpha Ctot; do
        reference=$(checkpoint_path "$reference_dir" "$reference_step" "$field")
        candidate=$(checkpoint_path "$candidate_dir" "$candidate_step" "$field")
        status=false
        cmp -s "$reference" "$candidate" && status=true
        printf '%s,%s,%s,%s,%s\n' "$comparison" "$field" \
            "$reference" "$candidate" "$status" >> "$comparison_csv"
    done
}

compare_pair retry_vs_direct \
    "$run_root/direct_half_dt" 000001 "$run_root/retry" 000001
compare_pair direct_two_vs_restart \
    "$run_root/restart_direct_two" 000002 "$run_root/restart_second" 000001
compare_pair old_solver_reference_vs_current \
    "$run_root/old_solver_reference" 000001 "$run_root/old_solver_current" 000001

grep -hE 'CTOT_(COUPLED_STEP_RETRY|FV_BE_ACCEPT|PHASE_INNER|PHASE_SOLVER_RESTART_MIGRATION)' \
    "$run_root"/*/run.log > "$run_root/key_runtime_markers.log" || true

echo "prompt8_phase_kkt_stage1_workstation_tests_complete"
echo "comparison_csv=$comparison_csv"
