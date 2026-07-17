#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/home/zhiheng/PF/CUDA_STO_PF_high_throughput_v1}
root=${2:-$repo/runs/high_throughput_pf_v1/history_mass_fix_regression_v1}
base_1d=$repo/runs/high_throughput_pf_v1/stage0_current_B187_8000_raw/runtime.params
base_small=$repo/runs/high_throughput_pf_v1/stage0_small3d/N32_B187/runtime.params
raw_root=/home/zhiheng/PF/CUDA_STO_PF_memory_perf_refactor_v1/baseline_inputs
dt=1.953125e-4

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo 'workstation_gpu_busy=true' >&2
    exit 75
fi
if [[ -e $root ]]; then
    echo "refusing_to_overwrite=$root" >&2
    exit 73
fi
mkdir -p "$root"
sha256sum "$repo/main_cuda" "$repo/main_cuda.cu" \
    "$repo/bdf2_history_mass_utils.h" > "$root/source_binary_hashes.txt"

checkpoint_stem() {
    find "$1/Results" -name 'ctot_checkpoint_step*_meta.json' -print \
        | sort | tail -1 | sed 's/_meta.json$//'
}

hash_checkpoint() {
    local stem=$1 output=$2
    sha256sum "${stem}_Ctot.raw" "${stem}_Ctot_nm1.raw" \
        "${stem}_phi.raw" "${stem}_phi_nm1.raw" \
        "${stem}_xB_alpha.raw" > "$output"
}

run_1d() {
    local run=$root/N512_B187_8000
    mkdir -p "$run"
    cp "$base_1d" "$run/runtime.params"
    (
        cd "$run"
        /usr/bin/time -f 'wall=%e maxrss_kib=%M' -o time.txt \
            "$repo/main_cuda" 512 1 1 "$dt" 8000 8000 8000 0 \
            --mode=dynamics --pf-param-file "$run/runtime.params" \
            --init-mode raw_fields \
            --init-phi-raw "$raw_root/phi_init.raw" \
            --init-xB-raw "$raw_root/xB_init.raw" \
            --init-Ctot-raw "$raw_root/Ctot_init.raw" \
            --init-meta "$raw_root/init_meta.json" \
            --init-case-tag history_mass_fix_N512_8000 > run.log 2>&1
    )
    hash_checkpoint "$(checkpoint_stem "$run")" "$run/hashes.txt"
}

run_small() {
    local n=$1 steps=$2 tag=$3
    local run=$root/$tag
    mkdir -p "$run"
    cp "$base_small" "$run/runtime.params"
    (
        cd "$run"
        "$repo/main_cuda" "$n" "$n" "$n" "$dt" "$steps" "$steps" "$steps" 0 \
            --mode=dynamics --pf-param-file "$run/runtime.params" \
            --radius-phys-nm "$((n / 4))" --init-case-tag "$tag" > run.log 2>&1
    )
}

run_1d
run_small 32 5 N32_B187_5
run_small 64 5 N64_B187_5
hash_checkpoint "$(checkpoint_stem "$root/N32_B187_5")" "$root/N32_B187_5/hashes.txt"
hash_checkpoint "$(checkpoint_stem "$root/N64_B187_5")" "$root/N64_B187_5/hashes.txt"

run_small 32 4 restart_continuous4
run_small 32 2 restart_first2
first_stem=$(checkpoint_stem "$root/restart_first2")
restart_run=$root/restart_second2
mkdir -p "$restart_run"
cp "$base_small" "$restart_run/runtime.params"
(
    cd "$restart_run"
    "$repo/main_cuda" 32 32 32 "$dt" 2 2 2 0 --mode=dynamics \
        --pf-param-file "$restart_run/runtime.params" \
        --init-mode raw_fields \
        --init-phi-raw "${first_stem}_phi.raw" \
        --init-xB-raw "${first_stem}_xB_alpha.raw" \
        --init-Ctot-raw "${first_stem}_Ctot.raw" \
        --init-Ctot-nm1-raw "${first_stem}_Ctot_nm1.raw" \
        --init-phi-nm1-raw "${first_stem}_phi_nm1.raw" \
        --init-meta "${first_stem}_meta.json" \
        --init-case-tag restart_second2 > run.log 2>&1
)
hash_checkpoint "$(checkpoint_stem "$root/restart_continuous4")" \
    "$root/restart_continuous4/hashes.txt"
hash_checkpoint "$(checkpoint_stem "$root/restart_second2")" \
    "$root/restart_second2/hashes.txt"

python3 - "$root" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = {
    "N512_B187_8000": [
        "c2789fa84d4b3017c2836b4c95ae4e60f02ce23b52532188b36086ef66b3d772",
        "2a81a5c6e3dacdfb0a768d0aa51adef0e23ce4175940142408d768416073c6e3",
        "0e48c8b51bfd5c84bf8c7403625fe8f6fcb2ae730259b26c371703db2714fcdb",
        "f0b568bb9dc27e9c82a3919a558bbdf50188057673dcd84792a22cf3a8aa0c78",
        "6d22a351a2f631a8a1e75e736ddb5075df4606cab83320543454d8b292fd1a23",
    ],
    "N32_B187_5": [
        "4453af114e34e19df26469a07e1c9248bd53e0c48ea8356d39ee79f71fc33e2b",
        "a5b976e72853027243531d4d5bcd7c3d792b9583502461914105e0823098135e",
        "3ce3b32c403051098a7bb5db0e4afb0ef0447fbd1ec124b86050bb5ad85e6d19",
        "14425179f1eb0b74743883d6cc97f564c2743327b1403a41d381a33c0c28b031",
        "bbc3e3dd9b894a303608a04667c778472642b33dc9fa4e84c01a952bc4f0de5e",
    ],
    "N64_B187_5": [
        "aff9ac99731c11d1f6f397c0e9c9fa0ea8ba9e1c94cffaa59885a0f253c2e114",
        "d9cc5cd14863791fede27e84bd4a8eca0e1596e0af2564cdacf126fa4b8134e3",
        "8641a8072fdfbb7bc5707623126eed3f72549acc4643b8c6c346581bfbc63eb4",
        "d97792e67f41d912c5b3223c7980d3fd83ed85a4f699a725429d72b2fa8d85cb",
        "8c501f5a57e110b01583e7c741493a512790e6072d21e28191438ef055c12954",
    ],
}

result = {"cases": {}, "restart_bitwise": False}
for case, want in expected.items():
    got = [line.split()[0] for line in (root / case / "hashes.txt").read_text().splitlines()]
    result["cases"][case] = {"expected": want, "actual": got, "bitwise": got == want}

continuous = [line.split()[0] for line in (root / "restart_continuous4" / "hashes.txt").read_text().splitlines()]
restart = [line.split()[0] for line in (root / "restart_second2" / "hashes.txt").read_text().splitlines()]
result["restart_bitwise"] = continuous == restart
result["all_pass"] = all(v["bitwise"] for v in result["cases"].values()) and result["restart_bitwise"]
(root / "regression_summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if not result["all_pass"]:
    raise SystemExit(1)
PY

echo 'history_mass_fix_regression_pass=true'
