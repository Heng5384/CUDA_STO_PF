#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 BUILD_ROOT MOVING_INPUT_ROOT RUN_ROOT" >&2
    exit 2
fi

build_root=$1
input_root=$2
run_root=$3
binary="$build_root/main_cuda"
expected_binary_sha=7a6cf5a03c58f160e30933bcb6a69ee0a1fe7aa2695fb41bb9174476cbf27c2e

[[ -x "$binary" ]] || { echo "missing binary: $binary" >&2; exit 2; }
[[ -f "$input_root/next2_moving_manifest.json" ]] || {
    echo "missing moving manifest: $input_root" >&2
    exit 2
}
[[ ! -e "$run_root" ]] || {
    echo "refusing to overwrite existing run root: $run_root" >&2
    exit 2
}
actual_binary_sha=$(sha256sum "$binary" | awk '{print $1}')
[[ "$actual_binary_sha" == "$expected_binary_sha" ]] || {
    echo "P2 binary hash mismatch: $actual_binary_sha" >&2
    exit 2
}
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        grep -q '[0-9]'; then
    echo "GPU already has a compute process; refusing to displace it" >&2
    exit 3
fi

mkdir -p "$run_root"
sha256sum "$binary" > "$run_root/binary_hash.txt"
printf 'case,R_over_lambda,correction,dt_code,requested_steps,accepted_steps,retries,rejects,exit_code,status\n' \
    > "$run_root/case_status.csv"

mapfile -t cases < <(python3 - "$input_root/next2_moving_manifest.json" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1])):
    print(item["case"])
PY
)

overall=0
for label in "${cases[@]}"; do
    input="$input_root/$label/input"
    case_dir="$run_root/$label"
    mkdir -p "$case_dir"
    read -r nx ny nz dt steps output_every ratio correction < <(
        python3 - "$input/benchmark_manifest.json" <<'PY'
import json
import sys
m = json.load(open(sys.argv[1]))
print(*m["grid"], m["dt_code"], m["nsteps"], m["output_every"],
      m["R_over_lambda"], m["finite_interface_correction"])
PY
    )
    set +e
    (
        cd "$case_dir"
        CUDA_STO_RESULTS_ROOT="$case_dir/results" \
            "$binary" "$nx" "$ny" "$nz" "$dt" "$steps" \
                "$output_every" 1 0 --mode=dynamics \
                --pf-param-file "$input/benchmark.params" \
                --temperature-C 400 --init-case-tag "$label" \
                --init-mode raw_fields \
                --init-phi-raw "$input/phi_init.raw" \
                --init-xB-raw "$input/xB_init.raw" \
                --init-Ctot-raw "$input/Ctot_init.raw" \
                --init-meta "$input/init_meta.json" > run.log 2>&1
    )
    rc=$?
    set -e
    accepts=$(grep -c 'CTOT_MIMETIC_BE_ACCEPT' "$case_dir/run.log" || true)
    retries=$(grep -c 'CTOT_COUPLED_STEP_RETRY' "$case_dir/run.log" || true)
    rejects=$(grep -c 'CTOT_COUPLED_STEP_REJECT' "$case_dir/run.log" || true)
    status=PASS
    if [[ $rc -ne 0 || $accepts -ne $steps || $retries -ne 0 || $rejects -ne 0 ]]; then
        status=FAIL
        overall=1
    fi
    printf '%s,%s,%s,%.17g,%s,%s,%s,%s,%s,%s\n' \
        "$label" "$ratio" "$correction" "$dt" "$steps" "$accepts" \
        "$retries" "$rejects" "$rc" "$status" >> "$run_root/case_status.csv"
    echo "next2_case_complete=$label status=$status accepted=$accepts/$steps"
done

echo "next2_moving_cases_complete=${#cases[@]}"
echo "run_root=$run_root"
exit "$overall"
