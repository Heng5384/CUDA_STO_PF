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
  mkdir -p "${PROJECT_ROOT}/Results"
}

site_print_env_banner() {
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
}

site_prepare_cuda_env() {
  export CUDA_ARCH="${CUDA_ARCH:-sm_120}"

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
