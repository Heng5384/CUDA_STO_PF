#!/usr/bin/env bash
set -euo pipefail

# 标准化脚本：本地单案例模板实例（dynamics）
#
# 用途：
# - 在本地或交互式 GPU 环境中运行 dynamics 模式
# - 适合快速 smoke test、参数调试，以及复跑现成二进制
#
# 可复制修改：
# - 改帮助信息
# - 改默认参数区
# - 改 PF 参数 tag 与 main_cuda flags

# ------------------------------------------------------------
# A. 启动与项目定位
# ------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_site_env.sh"
site_setup_project_root "$SCRIPT_DIR"
site_prepare_job_dirs
cd "$PROJECT_ROOT"

want_help=0
want_build=0
want_no_build=0
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "" ]]; then
  want_help=1
fi
if [[ "${1:-}" == "--build" ]]; then
  want_build=1
  shift
fi
if [[ "${1:-}" == "--no-build" ]]; then
  want_no_build=1
  shift
fi

if [[ $want_help -eq 1 ]]; then
  cat <<'EOF'
用法：
  ./run_dynamics_local.sh --build
  ./run_dynamics_local.sh --no-build Nx Ny Nz dt nsteps out_every csv_out_every elastic_enabled
  ./run_dynamics_local.sh Nx Ny Nz dt nsteps out_every csv_out_every elastic_enabled

作用：
  本地运行 dynamics 模式，输出会统一写入仓库根目录下的 Results/。

参数说明：
  Nx Ny Nz            网格大小
  dt                  时间步长（double）
  nsteps              总步数
  out_every           每 out_every 步输出一次 VTK
  csv_out_every       每 csv_out_every 步输出一次 CSV
  elastic_enabled     弹性开关：0=关闭，1=开启

示例：
  ./run_dynamics_local.sh 32 32 32 0.01 5 5 1 0
  ./run_dynamics_local.sh 64 64 64 0.01 100 50 10 1
EOF
  exit 0
fi

# ------------------------------------------------------------
# B. 编译函数
# ------------------------------------------------------------
do_build() {
  if ! site_prepare_cuda_env; then
    if [[ -x ./main_cuda ]]; then
      echo "[warn] 未找到 nvcc，无法重新编译；将直接运行现成的 ./main_cuda。"
      echo "       若要让源码改动生效，请设置 CUDA_ROOT/NVCC 或先加载 CUDA 模块。"
      return 0
    fi
    echo "[error] 未找到 nvcc，且当前目录没有可运行的 ./main_cuda"
    echo "        可选方案："
    echo "          export CUDA_ROOT=/usr/local/cuda-12.9"
    echo "          ./jobs/run_dynamics_local.sh --build"
    exit 10
  fi

  echo "编译程序..."
  echo "[info] build: make clean && make main_cuda CUDA_ROOT=$CUDA_ROOT CUDA_ARCH=$CUDA_ARCH NVCC=$NVCC"
  make clean
  make main_cuda "CUDA_ROOT=$CUDA_ROOT" "CUDA_ARCH=$CUDA_ARCH" "NVCC=$NVCC"
}

if [[ $want_build -eq 1 ]]; then
  do_build
  exit 0
fi

if [[ $# -ne 8 ]]; then
  echo "[error] 参数个数不对：需要 8 个参数，当前=$#"
  echo "        运行 ./jobs/run_dynamics_local.sh --help 查看用法"
  exit 2
fi

# ------------------------------------------------------------
# C. 参数区
# ------------------------------------------------------------
NX="$1"
NY="$2"
NZ="$3"
DT="$4"
NSTEPS="$5"
OUT_EVERY="$6"
CSV_OUT_EVERY="$7"
ELASTIC="$8"

if [[ $want_no_build -eq 0 ]]; then
  echo "[info] 每次运行前强制重新编译（可用 --no-build 跳过）..."
  do_build
else
  if [[ ! -e ./main_cuda ]]; then
    echo "[error] 你选择了 --no-build，但当前目录没有 ./main_cuda"
    exit 3
  fi
fi

if [[ -e ./main_cuda && ! -x ./main_cuda ]]; then
  echo "[warn] ./main_cuda 不可执行，尝试 chmod +x ..."
  chmod +x ./main_cuda || true
fi

if [[ ! -x ./main_cuda ]]; then
  echo "[error] ./main_cuda 不存在或不可执行，无法继续运行。"
  exit 3
fi

CUDA_RT_LIB="$(python3 - <<'PY'
import os, site
print(os.path.join(site.getusersitepackages(), "nvidia", "cuda_runtime", "lib"))
PY
)"
CUFFT_LIB="$(python3 - <<'PY'
import os, site
print(os.path.join(site.getusersitepackages(), "nvidia", "cufft", "lib"))
PY
)"
NVJITLINK_LIB="$(python3 - <<'PY'
import os, site
print(os.path.join(site.getusersitepackages(), "nvidia", "nvjitlink", "lib"))
PY
)"

for d in "$CUDA_RT_LIB" "$CUFFT_LIB" "$NVJITLINK_LIB"; do
  if [[ ! -d "$d" ]]; then
    echo "[error] 未找到 nvidia 动态库目录：$d"
    echo "        请先安装：python3 -m pip install --user --break-system-packages -U nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-nvjitlink-cu12"
    exit 4
  fi
done

export LD_LIBRARY_PATH="$CUDA_RT_LIB:$CUFFT_LIB:$NVJITLINK_LIB:${LD_LIBRARY_PATH:-}"

# ------------------------------------------------------------
# D. 生成 PF 参数并执行
# ------------------------------------------------------------
echo "[info] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[info] CUDA_ROOT=${CUDA_ROOT:-<not set>}"
echo "[info] CUDA_ARCH=${CUDA_ARCH:-<not set>}"
echo "[info] PROJECT_ROOT=$PROJECT_ROOT"
PF_PARAM_FILE="$(site_generate_pf_param_file "dynamics_${NX}x${NY}x${NZ}_T${TEMP_C:-base}")"
echo "[info] pf_param_file=$PF_PARAM_FILE"
echo "[info] run: ./main_cuda $NX $NY $NZ $DT $NSTEPS $OUT_EVERY $CSV_OUT_EVERY $ELASTIC --pf-param-file $PF_PARAM_FILE"
echo ""

./main_cuda "$NX" "$NY" "$NZ" "$DT" "$NSTEPS" "$OUT_EVERY" "$CSV_OUT_EVERY" "$ELASTIC" \
  --pf-param-file "$PF_PARAM_FILE"
