#!/usr/bin/env bash
set -euo pipefail

# 本地一键运行脚本（无需系统安装 CUDA Toolkit；依赖 pip 安装的 nvidia-cu12 动态库）
# 用法：
#   ./run_local.sh --help                   # 帮助
#   ./run_local.sh --build                  # 若有 nvcc，则重新编译 main_cuda
#   ./run_local.sh --no-build               # 运行时跳过编译（默认会先编译）
#   ./run_local.sh Nx Ny Nz dt nsteps out_every csv_out_every elastic_enabled
#
# 示例：
#   ./run_local.sh 64 64 64 0.01 100 50 10 0
#   ./run_local.sh 128 128 128 0.01 200 200 10 1
#
# 可选环境变量：
#   CUDA_VISIBLE_DEVICES=0   选择 GPU
#   CUDA_ROOT=/usr/local/cuda-12.9  可手动指定 CUDA 根目录
#   CUDA_ARCH=sm_80              默认适配当前服务器；可按 GPU 覆盖

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/lib/site_env.sh"
site_setup_project_root "$SCRIPT_DIR"
cd "$SCRIPT_DIR"

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
  ./run_local.sh --build
  ./run_local.sh --no-build Nx Ny Nz dt nsteps out_every csv_out_every elastic_enabled
  ./run_local.sh Nx Ny Nz dt nsteps out_every csv_out_every elastic_enabled

参数说明：
  Nx Ny Nz            网格大小
  dt                  时间步长（double）
  nsteps              总步数
  out_every           每 out_every 步输出一次 VTK
  csv_out_every       每 csv_out_every 步输出一次 CSV
  elastic_enabled     弹性开关：0=关闭，1=开启
  （诊断开关 diag_vtk_enabled / diag_elastic_bulk_penalty_enabled 不再从命令行控制，请在 params_default 中修改）

示例：
  ./run_local.sh 32 32 32 0.01 5 5 1 0
  ./run_local.sh 64 64 64 0.01 100 50 10 1

提示：
  - 如果你改了 .cu/.h 源码，需要重新编译 main_cuda；可用：
      ./run_local.sh --build
    但前提是你的环境里有 nvcc（CUDA Toolkit）。
  - 默认行为：每次运行前都会先重新编译（需要 nvcc）。若你只想用现成二进制运行，用 --no-build。
  - 若运行时报 “libcudart.so.12 / libcufft.so.11 not found”，先执行一次：
      python3 -m pip install --user --break-system-packages -U nvidia-cuda-runtime-cu12 nvidia-cufft-cu12
EOF
  exit 0
fi

do_build() {
  if ! site_prepare_cuda_env; then
    if [[ -x ./main_cuda ]]; then
      echo "[warn] 未找到 nvcc，无法重新编译；将直接运行现成的 ./main_cuda（你的源码改动不会生效）。"
      echo "       若要让源码改动生效：请设置 CUDA_ROOT、把 nvcc 加入 PATH，或先加载 CUDA 模块。"
      return 0
    fi
    echo "[error] 未找到 nvcc，且当前目录也没有可运行的 ./main_cuda"
    echo "        可选方案："
    echo "          export CUDA_ROOT=/usr/local/cuda-12.9"
    echo "          ./run_local.sh --build"
    echo "        或者先加载模块后再运行。"
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
  echo "        运行 ./run_local.sh --help 查看用法"
  exit 2
fi

NX="$1"; NY="$2"; NZ="$3"; DT="$4"; NSTEPS="$5"; OUT_EVERY="$6"; CSV_OUT_EVERY="$7"; ELASTIC="$8"
# 不再从命令行解析 DIAG_VTK / DIAG_ELASTIC_BULK，统一在 params_default 里设置

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

# 通过 python 定位 pip 安装的动态库目录（用户态，无需 sudo）
CUDA_RT_LIB="$(python3 - <<'PY'
import site, os
p = os.path.join(site.getusersitepackages(), "nvidia", "cuda_runtime", "lib")
print(p)
PY
)"
CUFFT_LIB="$(python3 - <<'PY'
import site, os
p = os.path.join(site.getusersitepackages(), "nvidia", "cufft", "lib")
print(p)
PY
)"
NVJITLINK_LIB="$(python3 - <<'PY'
import site, os
p = os.path.join(site.getusersitepackages(), "nvidia", "nvjitlink", "lib")
print(p)
PY
)"

for d in "$CUDA_RT_LIB" "$CUFFT_LIB" "$NVJITLINK_LIB"; do
  if [[ ! -d "$d" ]]; then
    echo "[error] 未找到 nvidia 动态库目录：$d"
    echo "        请先安装：python3 -m pip install --user --break-system-packages -U nvidia-cuda-runtime-cu12 nvidia-cufft-cu12"
    exit 4
  fi
done

export LD_LIBRARY_PATH="$CUDA_RT_LIB:$CUFFT_LIB:$NVJITLINK_LIB:${LD_LIBRARY_PATH:-}"

echo "[info] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[info] CUDA_ROOT=$CUDA_ROOT"
echo "[info] CUDA_ARCH=$CUDA_ARCH"
echo "[info] LD_LIBRARY_PATH 已设置（来自 ~/.local 的 nvidia wheel）"
echo "[info] run: ./main_cuda $NX $NY $NZ $DT $NSTEPS $OUT_EVERY $CSV_OUT_EVERY $ELASTIC"
echo ""

./main_cuda "$NX" "$NY" "$NZ" "$DT" "$NSTEPS" "$OUT_EVERY" "$CSV_OUT_EVERY" "$ELASTIC"

