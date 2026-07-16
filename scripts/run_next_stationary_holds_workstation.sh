#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 P1_BINARY ORACLE_ROOT RUN_ROOT" >&2
    exit 2
fi

binary=$1
oracle_root=$2
run_root=$3
expected_binary_sha=1bc5405ab12a6fd872480d5d6c373b98fc3b8940e5d7044cd7a50e6c3bf6c4b0

[[ -x "$binary" ]] || { echo "missing binary: $binary" >&2; exit 2; }
[[ ! -e "$run_root" ]] || {
    echo "refusing to overwrite existing run root: $run_root" >&2
    exit 2
}
actual_binary_sha=$(sha256sum "$binary" | awk '{print $1}')
[[ "$actual_binary_sha" == "$expected_binary_sha" ]] || {
    echo "P1 binary hash mismatch: $actual_binary_sha" >&2
    exit 2
}
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        grep -q '[0-9]'; then
    echo "GPU already has a compute process; refusing to displace it" >&2
    exit 3
fi

mkdir -p "$run_root"
sha256sum "$binary" > "$run_root/binary_hash.txt"

run_case() {
    local ratio=$1
    local nx=$2
    local input="$oracle_root/r$ratio"
    local case_dir="$run_root/r$ratio"
    for required in phi_init.raw xB_init.raw Ctot_init.raw init_meta.json \
                    benchmark.params benchmark_manifest.json; do
        [[ -f "$input/$required" ]] || {
            echo "missing oracle input: $input/$required" >&2
            exit 2
        }
    done
    mkdir -p "$case_dir"
    sha256sum "$input"/* > "$case_dir/input_hashes.txt"
    (
        cd "$case_dir"
        CUDA_STO_RESULTS_ROOT="$case_dir/results" \
            "$binary" "$nx" 2 "$nx" 6.25e-6 1000 100 1 0 \
            --mode=dynamics --pf-param-file "$input/benchmark.params" \
            --temperature-C 400 --init-case-tag "stationary_r${ratio}_hold" \
            --init-mode raw_fields --init-phi-raw "$input/phi_init.raw" \
            --init-xB-raw "$input/xB_init.raw" \
            --init-Ctot-raw "$input/Ctot_init.raw" \
            --init-meta "$input/init_meta.json" > run.log 2>&1
    )
    local accepts
    accepts=$(grep -c 'CTOT_MIMETIC_BE_ACCEPT' "$case_dir/run.log" || true)
    [[ "$accepts" -eq 1000 ]] || {
        echo "r$ratio accepted $accepts/1000 steps" >&2
        return 1
    }
}

run_case 5 320
run_case 8 320
run_case 10 336
run_case 15 456
run_case 20 576

echo "next_stationary_holds_workstation_complete"
echo "run_root=$run_root"
