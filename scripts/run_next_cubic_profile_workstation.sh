#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 P1_BINARY BASE_PARAMS PROFILE_ROOT RUN_ROOT" >&2
    exit 2
fi

binary=$1
base_params=$2
profile_root=$3
run_root=$4
expected_binary_sha=1bc5405ab12a6fd872480d5d6c373b98fc3b8940e5d7044cd7a50e6c3bf6c4b0

[[ -x "$binary" ]] || { echo "missing binary: $binary" >&2; exit 2; }
[[ -f "$base_params" ]] || { echo "missing params: $base_params" >&2; exit 2; }
command -v nsys >/dev/null || {
    echo "nsys is required for launch/sync/copy/FFT counts" >&2
    exit 2
}
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
sha256sum "$binary" "$base_params" > "$run_root/input_hashes.txt"
nvidia-smi -q > "$run_root/nvidia_smi_before.txt"

run_case() {
    local name=$1
    local size=$2
    local steps=$3
    local elastic=$4
    local input="$profile_root/n$size"
    local case_dir="$run_root/$name"
    local params="$case_dir/profile.params"
    mkdir -p "$case_dir"
    for required in phi_init.raw xB_init.raw Ctot_init.raw init_meta.json; do
        [[ -f "$input/$required" ]] || {
            echo "missing stationary profile: $input/$required" >&2
            exit 2
        }
    done
    cp "$base_params" "$params"
    {
        echo
        echo '# next low-cost cubic profile overlay'
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
            --cuda-event-trace=false \
            --force-overwrite=false --output="$case_dir/nsys_trace" \
            "$binary" "$size" "$size" "$size" 6.25e-6 "$steps" \
                "$steps" 1 "$elastic" --mode=dynamics \
                --pf-param-file "$params" --temperature-C 400 \
                --init-case-tag "$name" --init-mode raw_fields \
                --init-phi-raw "$input/phi_init.raw" \
                --init-xB-raw "$input/xB_init.raw" \
                --init-Ctot-raw "$input/Ctot_init.raw" \
                --init-meta "$input/init_meta.json" \
                > run.log 2>&1 &
        local pid=$!
        printf 'timestamp_ms,pid,used_memory_MiB\n' > gpu_memory_samples.csv
        while kill -0 "$pid" 2>/dev/null; do
            local stamp
            stamp=$(date +%s%3N)
            nvidia-smi --query-compute-apps=pid,used_memory \
                --format=csv,noheader,nounits | awk -v t="$stamp" -F, \
                '{print t "," $1 "," $2}' \
                >> gpu_memory_samples.csv || true
            sleep 0.05
        done
        wait "$pid"
        nsys stats --force-export=true --report cuda_api_sum --format csv \
            nsys_trace.nsys-rep \
            > nsys_cuda_api_sum.csv
        nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv \
            nsys_trace.nsys-rep \
            > nsys_cuda_gpu_kern_sum.csv
        nsys stats --force-export=true --report cuda_gpu_mem_time_sum --format csv \
            nsys_trace.nsys-rep > nsys_cuda_gpu_mem_time_sum.csv
        nsys stats --force-export=true --report cuda_gpu_mem_size_sum --format csv \
            nsys_trace.nsys-rep > nsys_cuda_gpu_mem_size_sum.csv
    )
    grep -q 'CTOT_MIMETIC_BE_ACCEPT' "$case_dir/run.log"
}

run_case cubic32_elastic_off 32 5 0
run_case cubic32_elastic_on 32 3 1
run_case cubic64_elastic_off 64 3 0
run_case cubic64_elastic_on 64 1 1

nvidia-smi -q > "$run_root/nvidia_smi_after.txt"
echo "next_cubic_profile_workstation_complete"
echo "run_root=$run_root"
