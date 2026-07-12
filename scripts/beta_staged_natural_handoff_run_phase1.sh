#!/usr/bin/env bash
set -euo pipefail

repo="${1:-/home/zhiheng/PF/CUDA_STO_PF}"
cd "$repo"

out_root="tmp_codex_ops/beta_staged_natural_handoff_validation/phase1_s_window"
mkdir -p "$out_root"

temps=(380 400)
s_values=(0.5 0.3 0.2 0.1 0.05)

for T in "${temps[@]}"; do
  for S in "${s_values[@]}"; do
    s_tag="${S/./p}"
    param="params/beta_enabled_long_coupling_diagnostic/T${T}_beta_enabled_S${s_tag}_1000.params"
    case_tag="natural_staged_T${T}_S${s_tag}_1000"
    stdout="${out_root}/${case_tag}.stdout"
    stderr="${out_root}/${case_tag}.stderr"
    status="${out_root}/${case_tag}.status"
    if [[ ! -f "$param" ]]; then
      echo "MISSING_PARAM $param" | tee "$status"
      continue
    fi
    echo "RUN case=${case_tag} param=${param} S=${S}"
    set +e
    ./main_cuda \
      --Nx 128 --Ny 128 --Nz 128 \
      --nsteps 1000 \
      --pf-param-file "$param" \
      --init-case-tag "$case_tag" \
      --gp-site-S-factor "$S" \
      --beta_staged_accumulation_enabled 1 \
      --beta_staged_accumulation_interval_steps 10 \
      --beta_staged_accumulation_GP_capture_radius_nm 12 \
      --beta_staged_accumulation_matrix_draw_radius_nm 2 \
      --beta_staged_accumulation_max_fraction_per_step 0.1 \
      --beta_staged_insert_when_target_reached 1 \
      > "$stdout" 2> "$stderr"
    rc=$?
    set -e
    echo "EXIT ${rc}" | tee "$status"
    if [[ "$rc" -ne 0 ]]; then
      echo "FAILED case=${case_tag} rc=${rc}"
    fi
  done
done

echo "PHASE1_S_WINDOW_DONE"
