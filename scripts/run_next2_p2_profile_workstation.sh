#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "usage: $0 P1_BINARY P2_BINARY EXPECTED_P2_SHA BASE_PARAMS PROFILE_ROOT RUN_ROOT" >&2
    exit 2
fi

p1_binary=$1
p2_binary=$2
expected_p2_sha=$3
base_params=$4
profile_root=$5
run_root=$6
expected_p1_sha=1bc5405ab12a6fd872480d5d6c373b98fc3b8940e5d7044cd7a50e6c3bf6c4b0

for required in "$p1_binary" "$p2_binary" "$base_params"; do
    [[ -e "$required" ]] || { echo "missing input: $required" >&2; exit 2; }
done
[[ ! -e "$run_root" ]] || {
    echo "refusing to overwrite run root: $run_root" >&2
    exit 2
}
[[ "$(sha256sum "$p1_binary" | awk '{print $1}')" == "$expected_p1_sha" ]] || {
    echo "P1 binary hash mismatch" >&2
    exit 2
}
[[ "$(sha256sum "$p2_binary" | awk '{print $1}')" == "$expected_p2_sha" ]] || {
    echo "P2 binary hash mismatch" >&2
    exit 2
}
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        grep -q '[0-9]'; then
    echo "GPU already has a compute process; refusing to displace it" >&2
    exit 3
fi

mkdir -p "$run_root"
sha256sum "$p1_binary" "$p2_binary" "$base_params" > "$run_root/input_hashes.txt"
nvidia-smi -q > "$run_root/nvidia_smi_before.txt"

run_case() {
    local implementation=$1
    local binary=$2
    local size=$3
    local steps=$4
    local elastic=$5
    local case_dir="$run_root/${implementation}_cubic${size}_elastic_$([[ $elastic == 1 ]] && echo on || echo off)"
    local input="$profile_root/n$size"
    local params="$case_dir/profile.params"
    mkdir -p "$case_dir"
    for required in phi_init.raw xB_init.raw Ctot_init.raw init_meta.json; do
        [[ -f "$input/$required" ]] || {
            echo "missing profile input: $input/$required" >&2
            exit 2
        }
    done
    cp "$base_params" "$params"
    {
        echo
        echo '# next2 matched P1/P2 cubic profile overlay'
        echo 'composition_evolution_mode=ctot_mimetic_be'
        echo 'ctot_phase_semismooth_pdas_enabled=1'
        echo 'ctot_finite_interface_antitrapping_enabled=0'
        echo 'finite_interface_resolution_test_override=0'
        echo 'ctot_step_max_retries=0'
        echo 'ctot_automatic_dt_growth=0'
        echo 'dt=6.25e-6'
        echo 'dt_code=6.25e-6'
        echo 'ctot_performance_profile_enabled=1'
        echo "ctot_elastic_validation_enabled=$elastic"
        echo "elastic_enabled=$elastic"
        echo 'diagnostic_rsmd_enabled=0'
        echo 'gp_nuc_enabled=0'
        echo 'gp_literature_model_enabled=0'
        echo 'gp_initial_population_enabled=0'
        echo 'gp_stochastic_enabled=0'
        echo 'gp_to_beta_enabled=0'
        echo 'gp_growth_enabled=0'
        echo 'gp_radius_evolution_enabled=0'
        echo 'gp_inventory_growth_enabled=0'
    } >> "$params"
    (
        cd "$case_dir"
        CUDA_STO_RESULTS_ROOT="$case_dir/results" nsys profile \
            --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
            --cuda-event-trace=false --force-overwrite=false \
            --output="$case_dir/nsys_trace" \
            "$binary" "$size" "$size" "$size" 6.25e-6 "$steps" \
                "$steps" 1 "$elastic" --mode=dynamics \
                --pf-param-file "$params" --temperature-C 400 \
                --init-case-tag "${implementation}_cubic${size}_elastic${elastic}" \
                --init-mode raw_fields \
                --init-phi-raw "$input/phi_init.raw" \
                --init-xB-raw "$input/xB_init.raw" \
                --init-Ctot-raw "$input/Ctot_init.raw" \
                --init-meta "$input/init_meta.json" > run.log 2>&1 &
        local pid=$!
        printf 'timestamp_ms,pid,used_memory_MiB\n' > gpu_memory_samples.csv
        while kill -0 "$pid" 2>/dev/null; do
            local stamp
            stamp=$(date +%s%3N)
            nvidia-smi --query-compute-apps=pid,used_memory \
                --format=csv,noheader,nounits | awk -v t="$stamp" -F, \
                '{print t "," $1 "," $2}' >> gpu_memory_samples.csv || true
            sleep 0.05
        done
        wait "$pid"
        for report in cuda_api_sum cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum; do
            nsys stats --force-export=true --report "$report" --format csv \
                nsys_trace.nsys-rep > "nsys_${report}.csv"
        done
    )
    local accepted
    accepted=$(grep -Ec 'CTOT_(FV|MIMETIC)_BE_ACCEPT' "$case_dir/run.log" || true)
    [[ "$accepted" -eq "$steps" ]] || {
        echo "$case_dir accepted $accepted/$steps" >&2
        exit 1
    }
}

for implementation in p1 p2; do
    binary=$p1_binary
    [[ "$implementation" == p2 ]] && binary=$p2_binary
    run_case "$implementation" "$binary" 32 5 0
    run_case "$implementation" "$binary" 32 3 1
    run_case "$implementation" "$binary" 64 3 0
    run_case "$implementation" "$binary" 64 1 1
done

nvidia-smi -q > "$run_root/nvidia_smi_after.txt"
echo "next2_p2_profile_workstation_complete"
echo "run_root=$run_root"
