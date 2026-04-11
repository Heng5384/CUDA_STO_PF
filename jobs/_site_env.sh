#!/usr/bin/env bash

site_find_project_root() {
  local start_dir="${1:-$(pwd)}"
  start_dir="$(cd "$start_dir" && pwd)"

  while true; do
    if [[ -f "$start_dir/Makefile" && -f "$start_dir/main_cuda.cu" ]]; then
      printf '%s\n' "$start_dir"
      return 0
    fi

    local parent_dir
    parent_dir="$(dirname "$start_dir")"
    if [[ "$parent_dir" == "$start_dir" ]]; then
      return 1
    fi
    start_dir="$parent_dir"
  done
}

site_setup_project_root() {
  local caller_dir="${1:-$(pwd)}"

  if [[ -n "${PROJECT_ROOT:-}" && -f "${PROJECT_ROOT}/Makefile" && -f "${PROJECT_ROOT}/main_cuda.cu" ]]; then
    cd "${PROJECT_ROOT}"
    export PROJECT_ROOT="$(pwd)"
    return 0
  fi

  local root
  if ! root="$(site_find_project_root "$caller_dir")"; then
    echo "[fatal] Could not locate the project root from: $caller_dir" >&2
    return 1
  fi

  cd "$root"
  export PROJECT_ROOT="$(pwd)"
}

site_in_slurm() {
  [[ -n "${SLURM_JOB_ID:-}" ]]
}

site_try_enable_modules() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi
  for f in /etc/profile.d/modules.sh /usr/share/modules/init/bash; do
    if [[ -f "$f" ]]; then
      # shellcheck disable=SC1090
      source "$f" >/dev/null 2>&1 || true
    fi
  done
}

site_prepare_job_dirs() {
  mkdir -p "${PROJECT_ROOT}/jobs/logs"
  mkdir -p "${PROJECT_ROOT}/jobs/generated"
  mkdir -p "${PROJECT_ROOT}/Results"
}

site_redirect_slurm_logs() {
  if ! site_in_slurm; then
    return 0
  fi

  if [[ "${SITE_SLURM_LOG_REDIRECTED:-0}" == "1" ]]; then
    return 0
  fi

  local log_tag="${1:-${SLURM_JOB_NAME:-slurm_job}}"
  local safe_tag
  safe_tag="$(site_make_safe_tag "$log_tag")"

  site_prepare_job_dirs

  local stdout_log="${PROJECT_ROOT}/jobs/logs/${safe_tag}_${SLURM_JOB_ID}.out"
  local stderr_log="${PROJECT_ROOT}/jobs/logs/${safe_tag}_${SLURM_JOB_ID}.err"

  export SITE_SLURM_STDOUT_LOG="${stdout_log}"
  export SITE_SLURM_STDERR_LOG="${stderr_log}"
  export SITE_SLURM_LOG_REDIRECTED=1

  exec >>"${stdout_log}" 2>>"${stderr_log}"
}

site_resolve_physical_input_json() {
  local default_json="${PROJECT_ROOT}/physical_inputs.example.json"
  local path="${PHYSICAL_INPUT_JSON:-$default_json}"
  if [[ ! -f "$path" ]]; then
    echo "[fatal] Physical input JSON not found: $path" >&2
    return 20
  fi
  printf '%s\n' "$path"
}

site_make_safe_tag() {
  local raw="${1:-run}"
  raw="${raw// /_}"
  raw="${raw//\//_}"
  raw="${raw//:/_}"
  raw="${raw//=/__}"
  printf '%s\n' "$raw"
}

site_generate_pf_param_file() {
  local tag="${1:-run}"
  local safe_tag
  safe_tag="$(site_make_safe_tag "$tag")"
  local base_json
  base_json="$(site_resolve_physical_input_json)" || return $?

  local env_keys=(
    PHYSICAL_INPUT_JSON
    PHYSICAL_OVERRIDE_JSON
    PHYSICAL_OVERRIDE_FILE
    DT
    TEMP_C
    PHYS_TEMPERATURE_C
    DX_M
    PHYS_DX_M
    PF_DX_M
    PHYS_DX_REF_M
    GAMMA_JM2
    PHYS_GAMMA_JM2
    LAMBDA_SM_M
    PHYS_LAMBDA_SM_M
    V_A
    PHYS_V_A
    V_B
    PHYS_V_B
    VM_COMPOUND
    PHYS_VM_COMPOUND
    VM_ALPHA_0
    PHYS_VM_ALPHA_0
    D_RATIO
    PHYS_D_RATIO
    VF_INIT
    PHYS_VF_INIT
    VF_TARGET
    PHYS_VF_TARGET
    L_REF_FACTOR
    PHYS_L_REF_FACTOR
    GEL_SHIFT_JM3
    PHYS_GEL_SHIFT_JM3
    EPS_ISO
    PHYS_EPS_ISO
  )
  local env_key
  for env_key in "${env_keys[@]}"; do
    if [[ "${!env_key+x}" == "x" ]]; then
      export "${env_key}"
    fi
  done

  site_prepare_job_dirs

  local input_json="${PROJECT_ROOT}/jobs/generated/physical_inputs_${safe_tag}.json"
  local payload_json="${PROJECT_ROOT}/jobs/generated/pf_payload_${safe_tag}.json"
  local param_file="${PROJECT_ROOT}/jobs/generated/pf_params_${safe_tag}.params"

  BASE_PHYSICAL_JSON="$base_json" OUTPUT_PHYSICAL_JSON="$input_json" python3 - <<'PY'
import json
import os
from pathlib import Path

base_path = Path(os.environ["BASE_PHYSICAL_JSON"])
out_path = Path(os.environ["OUTPUT_PHYSICAL_JSON"])

with base_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

data = {k: v for k, v in data.items() if not k.startswith("_")}

env_map = {
    "DT": ("dt", float),
    "TEMP_C": ("temperature_C", float),
    "PHYS_TEMPERATURE_C": ("temperature_C", float),
    "DX_M": ("dx", float),
    "PHYS_DX_M": ("dx", float),
    "PF_DX_M": ("pf_dx", float),
    "PHYS_DX_REF_M": ("phys_dx_ref", float),
    "GAMMA_JM2": ("gamma", float),
    "PHYS_GAMMA_JM2": ("gamma", float),
    "LAMBDA_SM_M": ("lambda_sm", float),
    "PHYS_LAMBDA_SM_M": ("lambda_sm", float),
    "V_A": ("v_A", float),
    "PHYS_V_A": ("v_A", float),
    "V_B": ("v_B", float),
    "PHYS_V_B": ("v_B", float),
    "VM_COMPOUND": ("Vm_compound", float),
    "PHYS_VM_COMPOUND": ("Vm_compound", float),
    "VM_ALPHA_0": ("Vm_alpha_0", float),
    "PHYS_VM_ALPHA_0": ("Vm_alpha_0", float),
    "D_RATIO": ("D_ratio", float),
    "PHYS_D_RATIO": ("D_ratio", float),
    "VF_INIT": ("vf_init", float),
    "PHYS_VF_INIT": ("vf_init", float),
    "VF_TARGET": ("vf_target", float),
    "PHYS_VF_TARGET": ("vf_target", float),
    "L_REF_FACTOR": ("L_ref_factor", float),
    "PHYS_L_REF_FACTOR": ("L_ref_factor", float),
    "GEL_SHIFT_JM3": ("gel_shift_Jm3", float),
    "PHYS_GEL_SHIFT_JM3": ("gel_shift_Jm3", float),
    "EPS_ISO": ("eps_iso", float),
    "PHYS_EPS_ISO": ("eps_iso", float),
}

for env_key, (json_key, caster) in env_map.items():
    value = os.environ.get(env_key)
    if value is None or value == "":
        continue
    data[json_key] = caster(value)

override_json = os.environ.get("PHYSICAL_OVERRIDE_JSON")
if override_json:
    extra = json.loads(override_json)
    if not isinstance(extra, dict):
        raise SystemExit("PHYSICAL_OVERRIDE_JSON must be a JSON object")
    data.update(extra)

override_file = os.environ.get("PHYSICAL_OVERRIDE_FILE")
if override_file:
    with open(override_file, "r", encoding="utf-8") as f:
        extra = json.load(f)
    if not isinstance(extra, dict):
        raise SystemExit("PHYSICAL_OVERRIDE_FILE must contain a JSON object")
    data.update(extra)

out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
PY

  python3 "${PROJECT_ROOT}/Unit_Psedobinary.py" \
    --input-json "$input_json" \
    --output-json "$payload_json" \
    --output-pf-param-file "$param_file" \
    --no-summary >&2

  echo "[site-env] physical_input_json=${input_json}" >&2
  echo "[site-env] pf_payload_json=${payload_json}" >&2
  echo "[site-env] pf_param_file=${param_file}" >&2
  printf '%s\n' "$param_file"
}

site_print_env_banner() {
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  if [[ -n "${SITE_SLURM_STDOUT_LOG:-}" || -n "${SITE_SLURM_STDERR_LOG:-}" ]]; then
    echo "SITE_SLURM_STDOUT_LOG=${SITE_SLURM_STDOUT_LOG:-<unset>}"
    echo "SITE_SLURM_STDERR_LOG=${SITE_SLURM_STDERR_LOG:-<unset>}" >&2
  fi
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
}

site_prepare_cuda_env() {
  # 优先级：
  # 1. 用户显式传入 CUDA_ARCH
  # 2. Slurm 环境默认用较保守的集群架构
  # 3. 非 Slurm 环境默认用本地 workstation 架构
  if [[ -n "${CUDA_ARCH:-}" ]]; then
    export CUDA_ARCH
  elif site_in_slurm; then
    export CUDA_ARCH="${SITE_DEFAULT_CUDA_ARCH_SLURM:-sm_80}"
  else
    export CUDA_ARCH="${SITE_DEFAULT_CUDA_ARCH_LOCAL:-sm_120}"
  fi

  local mode="server"
  if site_in_slurm; then
    mode="slurm"
    site_try_enable_modules
    if [[ "${SITE_SKIP_MODULE_LOAD:-0}" != "1" ]] && command -v module >/dev/null 2>&1; then
      module purge >/dev/null 2>&1 || true
      module load "${CUDA_MODULE:-cuda/cuda-12.9}"
    fi
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo "[fatal] CUDA_VISIBLE_DEVICES is unset. This usually means you did not get a GPU allocation." >&2
      return 2
    fi
  fi

  local nvcc_path=""
  if [[ -n "${NVCC:-}" && -x "${NVCC}" ]]; then
    nvcc_path="${NVCC}"
  fi

  if [[ -z "$nvcc_path" && -n "${CUDA_ROOT:-}" && -x "${CUDA_ROOT}/bin/nvcc" ]]; then
    nvcc_path="${CUDA_ROOT}/bin/nvcc"
  fi

  if [[ -z "$nvcc_path" ]] && command -v nvcc >/dev/null 2>&1; then
    nvcc_path="$(command -v nvcc)"
  fi

  if [[ -z "$nvcc_path" ]]; then
    local candidates=()
    if site_in_slurm; then
      candidates=(
        "${CUDA_ROOT:-}"
        /usr/local/cuda-12.9
        /usr/local/cuda
        /usr/local/cuda-12.8
        /usr/local/cuda-12
        /opt/cuda
      )
    else
      candidates=(
        "${CUDA_ROOT:-}"
        /usr/local/cuda-12.9
        /usr/local/cuda
        /usr/local/cuda-12.8
        /usr/local/cuda-12
        /usr/local/cuda-11
        /opt/cuda
      )
    fi

    local candidate
    for candidate in "${candidates[@]}"; do
      [[ -n "$candidate" ]] || continue
      if [[ -x "$candidate/bin/nvcc" ]]; then
        nvcc_path="$candidate/bin/nvcc"
        break
      fi
    done
  fi

  if [[ -z "$nvcc_path" ]]; then
    return 1
  fi

  export NVCC="$nvcc_path"
  export CUDA_ROOT="$(cd "$(dirname "$nvcc_path")/.." && pwd)"
  export PATH="${CUDA_ROOT}/bin:${PATH}"
  if [[ -d "${CUDA_ROOT}/lib64" ]]; then
    export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
  fi

  echo "[site-env] mode=${mode}"
  echo "[site-env] PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}"
  echo "[site-env] CUDA_ROOT=${CUDA_ROOT}"
  echo "[site-env] NVCC=${NVCC}"
  echo "[site-env] CUDA_ARCH=${CUDA_ARCH}"
  return 0
}

site_require_cuda_env() {
  if site_prepare_cuda_env; then
    return 0
  fi
  echo "[fatal] Could not find a usable CUDA toolchain. Set CUDA_ROOT/NVCC or load the CUDA module first." >&2
  return 10
}

site_build_main_cuda() {
  echo "编译程序..."
  site_require_cuda_env || return $?
  site_prepare_job_dirs
  make clean
  make main_cuda "CUDA_ROOT=${CUDA_ROOT}" "CUDA_ARCH=${CUDA_ARCH}" "NVCC=${NVCC}"
}
