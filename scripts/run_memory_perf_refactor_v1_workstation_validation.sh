#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/home/zhiheng/PF/CUDA_STO_PF_memory_perf_refactor_v1}
base="$repo/baseline_inputs/runtime.params"
root="$repo/runs/final_validation"
dt=1.953125e-4

mkdir -p "$root"
while pgrep -f 'main_cuda 400 400 400 .*PF_ONLY_N400_benchmark13' >/dev/null; do
    sleep 10
done

run_case() {
    local tag=$1 n=$2 features=$3 steps=$4 radius=$5 diagnostics=$6
    local run="$root/$tag"
    rm -rf "$run"
    mkdir -p "$run"
    cp "$base" "$run/runtime.params"
    printf '\nctot_memory_perf_features=%s\nctot_diagnostics_enabled=%s\nctot_benchmark_suppress_field_output=0\nctot_performance_profile_enabled=0\n' \
        "$features" "$diagnostics" >> "$run/runtime.params"
    (
        cd "$run"
        /usr/bin/time -f 'wall=%e maxrss_kib=%M' -o time.txt \
            "$repo/main_cuda" "$n" "$n" "$n" "$dt" "$steps" "$steps" "$steps" 0 \
            --mode=dynamics --pf-param-file "$run/runtime.params" \
            --radius-phys-nm "$radius" --init-case-tag "$tag" > run.log 2>&1
    )
}

checkpoint_stem() {
    find "$1/Results" -name 'ctot_checkpoint_step*_meta.json' -print | sort | tail -1 | sed 's/_meta.json$//'
}

hash_checkpoint() {
    local stem=$1
    sha256sum "${stem}_Ctot.raw" "${stem}_Ctot_nm1.raw" \
        "${stem}_phi.raw" "${stem}_phi_nm1.raw" "${stem}_xB_alpha.raw"
}

for n in 32 64; do
    run_case "N${n}_B0_5" "$n" 0 5 "$((n / 4))" 1
    run_case "N${n}_B187_5" "$n" 187 5 "$((n / 4))" 1
    hash_checkpoint "$(checkpoint_stem "$root/N${n}_B0_5")" > "$root/N${n}_B0_5/hashes.txt"
    hash_checkpoint "$(checkpoint_stem "$root/N${n}_B187_5")" > "$root/N${n}_B187_5/hashes.txt"
done

run_case restart_continuous4 32 187 4 8 1
run_case restart_first2 32 187 2 8 1
first_stem=$(checkpoint_stem "$root/restart_first2")
restart_run="$root/restart_second2"
rm -rf "$restart_run"
mkdir -p "$restart_run"
cp "$base" "$restart_run/runtime.params"
printf '\nctot_memory_perf_features=187\nctot_diagnostics_enabled=1\nctot_benchmark_suppress_field_output=0\nctot_performance_profile_enabled=0\n' >> "$restart_run/runtime.params"
(
    cd "$restart_run"
    /usr/bin/time -f 'wall=%e maxrss_kib=%M' -o time.txt \
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
hash_checkpoint "$(checkpoint_stem "$root/restart_continuous4")" > "$root/restart_continuous4/hashes.txt"
hash_checkpoint "$(checkpoint_stem "$root/restart_second2")" > "$root/restart_second2/hashes.txt"

profile_case() {
    local tag=$1 features=$2
    local run="$root/$tag"
    rm -rf "$run"
    mkdir -p "$run"
    cp "$base" "$run/runtime.params"
    printf '\nctot_memory_perf_features=%s\nctot_diagnostics_enabled=0\nctot_benchmark_suppress_field_output=1\nctot_performance_profile_enabled=1\n' \
        "$features" >> "$run/runtime.params"
    (
        cd "$run"
        /usr/local/bin/nsys profile --trace=cuda,nvtx,osrt --sample=none \
            --cpuctxsw=none --force-overwrite=true -o "$run/profile" \
            "$repo/main_cuda" 128 128 128 "$dt" 2 2 2 0 \
            --mode=dynamics --pf-param-file "$run/runtime.params" \
            --radius-phys-nm 32 --init-case-tag "$tag" > run.log 2>&1
        /usr/local/bin/nsys stats \
            --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,osrt_sum \
            --format csv "$run/profile.nsys-rep" > nsys_stats.txt
    )
}

profile_case N128_B0_nsys 0
profile_case N128_B187_nsys 187

{
    echo "validation_complete=true"
    for n in 32 64; do
        echo "N${n}_endpoint_bitwise=$(diff -q <(awk '{print $1}' "$root/N${n}_B0_5/hashes.txt") <(awk '{print $1}' "$root/N${n}_B187_5/hashes.txt") >/dev/null && echo true || echo false)"
    done
    echo "restart_bitwise=$(diff -q <(awk '{print $1}' "$root/restart_continuous4/hashes.txt") <(awk '{print $1}' "$root/restart_second2/hashes.txt") >/dev/null && echo true || echo false)"
} > "$root/validation_summary.txt"
cat "$root/validation_summary.txt"
