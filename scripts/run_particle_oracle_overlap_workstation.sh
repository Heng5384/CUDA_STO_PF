#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/home/zhiheng/PF/CUDA_STO_PF_high_throughput_v1}
root=${2:-$repo/runs/high_throughput_pf_v1/particle_oracle_overlap}
dt_code=${3:-1.95312500000000011e-04}
base=$repo/reports/high_throughput_pf_v1/benchmark_inputs/stage0_strict_dt16.params
template_meta=$repo/runs/high_throughput_pf_v1/stage0_current_B187_8000_raw/Results/ch_T400_cuda_512x1x1_dt0.000195_steps8000_xB0.030/stage0_current_B187_8000_raw/ctot_checkpoint_step008000_meta.json

if [[ -e $root ]]; then echo "refusing_to_overwrite=$root" >&2; exit 73; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo 'workstation_gpu_busy=true' >&2; exit 75
fi
mkdir -p "$root"

run_case() {
    local tag=$1 steps=$2; shift 2
    local case=$root/$tag input=$root/$tag/input
    mkdir -p "$case" "$input"
    python3 "$repo/scripts/prepare_particle_overlap_fields.py" \
        --out-dir "$input" --template-meta "$template_meta" --size 64 \
        --dx-nm 1 --interface-half-width-nm 2 --matrix-xB 0.012 "$@" \
        > "$input/generation.log"
    cp "$base" "$case/runtime.params"
    cat >> "$case/runtime.params" <<EOF

# PF/particle short-overlap numerical overlay. The default is the qualified
# strict dt16 reference; callers may explicitly provide another fixed dt.
dt=$dt_code
ctot_residual_abs_tol=1.00000000000000006e-09
ctot_residual_rel_tol=1.00000000000000004e-10
ctot_numerics_contract=ctot_jichen_imex_bdf2_active_manifold_v1
ctot_automatic_dt_growth=0
ctot_step_max_retries=0
ctot_memory_perf_features=187
ctot_diagnostics_enabled=1
ctot_benchmark_suppress_field_output=0
ctot_performance_profile_enabled=0
ctot_full_audit_cadence=100
EOF
    (
        cd "$case"
        "$repo/main_cuda" 64 64 64 --pf-param-file "$case/runtime.params" \
            --nsteps "$steps" --out-every "$steps" --csv-out-every "$steps" \
            --ctot-full-audit-cadence 100 --init-case-tag "$tag" \
            --init-mode raw_fields --init-phi-raw "$input/phi_init.raw" \
            --init-xB-raw "$input/xB_init.raw" \
            --init-Ctot-raw "$input/Ctot_init.raw" \
            --init-meta "$input/init_meta.json" > run.log 2>&1
    )
    local final
    final=$(find "$case/Results" -name "ctot_checkpoint_step$(printf '%06d' "$steps")_meta.json" -print -quit | sed 's/_meta.json$//')
    python3 "$repo/scripts/pf_particle_handoff.py" \
        --Ctot "$input/Ctot_init.raw" --phi "$input/phi_init.raw" \
        --shape 64 64 64 --dx-nm 1 --output "$case/handoff_initial.json"
    python3 "$repo/scripts/pf_particle_handoff.py" \
        --Ctot "${final}_Ctot.raw" --phi "${final}_phi.raw" \
        --shape 64 64 64 --dx-nm 1 --output "$case/handoff_final.json"
}

run_case two_particle 200 \
    --particle 18,32,32,6 --particle 45,32,32,10
run_case three_particle_extinction 1200 \
    --particle 10,32,32,2 --particle 28,32,32,6 --particle 50,32,32,10

echo 'particle_oracle_overlap_pf_complete=true'
