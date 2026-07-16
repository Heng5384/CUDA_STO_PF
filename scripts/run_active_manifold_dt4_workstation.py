#!/usr/bin/env python3
"""Run the preregistered T400 dt/4 active-manifold qualification."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "reports" / "T400_longtime_v1" / "coarse4_runs" /
    "growth_N512_shift0_dt8_pre_event_freeze"
)
LOCAL_RUN_ROOT = ROOT / "reports" / "active_manifold_bdf2_v1" / "workstation_runs"
REMOTE_REPO = "/home/zhiheng/PF/CUDA_STO_PF_lie_be_v2_20260716"
REMOTE_RUN_ROOT = f"{REMOTE_REPO}/runs/active_manifold_bdf2_v1"
DT4 = 7.81250000000000043e-4


def execute(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="workstation-tail")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--label", default="fixed_qualification")
    args = parser.parse_args()
    busy = execute(["ssh", args.host, "pgrep -a -x main_cuda || true"])
    if busy.stdout.strip():
        raise RuntimeError("workstation main_cuda already active:\n" + busy.stdout)

    tag = f"dt4_active_manifold_{args.label}_{args.steps}"
    staging = LOCAL_RUN_ROOT / f"{tag}_input"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name in ("phi_init.raw", "xB_init.raw", "Ctot_init.raw", "init_meta.json"):
        shutil.copy2(SOURCE / name, staging / name)
    params = (SOURCE / "runtime.params").read_text(encoding="utf-8")
    params += f"""

# Active-manifold dt/4 qualification; physical inputs unchanged.
dt={DT4:.17e}
ctot_numerics_contract=ctot_jichen_imex_bdf2_active_manifold_v1
ctot_split_defect_policy=IMEX_BDF2_NO_POST_PHASE_POLISH
ctot_max_coupling_correctors=0
ctot_automatic_dt_growth=0
ctot_step_max_retries=0
BDF2_EVENT_PREFLIGHT_V1=1
BDF2_EVENT_BE_SUBCYCLING_V1=1
"""
    (staging / "runtime.params").write_text(params, encoding="utf-8")

    remote_out = f"{REMOTE_RUN_ROOT}/{tag}"
    execute(["ssh", args.host, "rm -rf " + shlex.quote(remote_out) +
             " && mkdir -p " + shlex.quote(remote_out)])
    execute(["rsync", "-az", str(staging) + "/", f"{args.host}:{remote_out}/"])
    command = f"""set -e
repo={shlex.quote(REMOTE_REPO)}
out={shlex.quote(remote_out)}
cd "$out"
export CUDA_STO_RESULTS_ROOT="$out/Results"
"$repo/main_cuda" 512 1 1 {DT4:.17e} {args.steps} {args.steps} {args.steps} 0 \
  --mode dynamics --pf-param-file "$out/runtime.params" --temperature-C 400 \
  --init-mode raw_fields --init-phi-raw "$out/phi_init.raw" \
  --init-xB-raw "$out/xB_init.raw" --init-Ctot-raw "$out/Ctot_init.raw" \
  --init-meta "$out/init_meta.json" --init-case-tag {shlex.quote(tag)} \
  > run.log 2>&1
"""
    started = time.monotonic()
    completed = execute(["ssh", args.host, command], check=False)
    wall = time.monotonic() - started
    local = LOCAL_RUN_ROOT / tag
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True)
    download = execute(
        ["rsync", "-az", f"{args.host}:{remote_out}/", str(local) + "/"],
        check=False,
    )
    status = {
        "case": "dt4", "steps": args.steps, "label": args.label,
        "returncode": completed.returncode,
        "download_returncode": download.returncode,
        "wall_seconds": wall, "remote_out": remote_out,
    }
    (local / "workstation_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))
    return 0 if completed.returncode == 0 and download.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
