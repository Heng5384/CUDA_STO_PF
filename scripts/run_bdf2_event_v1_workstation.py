#!/usr/bin/env python3
"""Run the frozen T400 BDF2 active-set crossings on workstation only."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/home/zhiheng/PF/CUDA_STO_PF_lie_be_v2_20260716"
REMOTE_RUN_ROOT = f"{REMOTE_REPO}/runs/bdf2_event_v1"
LOCAL_RUN_ROOT = ROOT / "reports" / "bdf2_event_v1" / "workstation_runs"

CASES = {
    "dt8": {
        "dt": 3.90625000000000022e-4,
        "step": 2957,
        "source": "growth_N512_shift0_dt8_pre_event_freeze",
        "result_parent": "ch_T400_cuda_512x1x1_dt0.000391_steps10000_xB0.030",
    },
    "dt16": {
        "dt": 1.95312500000000011e-4,
        "step": 5482,
        "source": "growth_N512_shift0_dt16_pre_event_freeze",
        "result_parent": "ch_T400_cuda_512x1x1_dt0.000195_steps20000_xB0.030",
    },
}


def execute(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def ensure_idle(host: str) -> None:
    probe = execute(
        [
            "ssh",
            host,
            "nvidia-smi --query-compute-apps=pid,process_name "
            "--format=csv,noheader; pgrep -x main_cuda || true",
        ]
    )
    busy = [line for line in probe.stdout.splitlines() if line.strip()]
    if busy:
        raise RuntimeError("workstation main_cuda process already active:\n" + "\n".join(busy))


def run_case(host: str, name: str, steps: int) -> dict[str, object]:
    case = CASES[name]
    ensure_idle(host)
    source = (
        f"{REMOTE_REPO}/runs/T400_longtime_v1/coarse4_runs/{case['source']}"
    )
    checkpoint = (
        f"{source}/run/Results/{case['result_parent']}/{case['source']}"
    )
    remote_out = f"{REMOTE_RUN_ROOT}/{name}_event_safe_{steps}"
    tag = f"{name}_event_safe_{steps}"
    step = int(case["step"])
    stem = f"ctot_checkpoint_step{step:06d}"
    command = f"""set -e
repo={shlex.quote(REMOTE_REPO)}
src={shlex.quote(source)}
ck={shlex.quote(checkpoint)}
out={shlex.quote(remote_out)}
rm -rf "$out"
mkdir -p "$out"
cp "$src/runtime.params" "$out/runtime.params"
printf '\nBDF2_EVENT_PREFLIGHT_V1=1\nBDF2_EVENT_BE_SUBCYCLING_V1=1\n' >> "$out/runtime.params"
cd "$out"
export CUDA_STO_RESULTS_ROOT="$out/Results"
"$repo/main_cuda" 512 1 1 {float(case['dt']):.17e} {steps} {steps} {steps} 0 \
  --mode dynamics --pf-param-file "$out/runtime.params" --temperature-C 400 \
  --init-mode raw_fields --init-phi-raw "$ck/{stem}_phi.raw" \
  --init-xB-raw "$ck/{stem}_xB_alpha.raw" \
  --init-Ctot-raw "$ck/{stem}_Ctot.raw" \
  --init-Ctot-nm1-raw "$ck/{stem}_Ctot_nm1.raw" \
  --init-phi-nm1-raw "$ck/{stem}_phi_nm1.raw" \
  --init-meta "$ck/{stem}_meta.json" --init-case-tag {shlex.quote(tag)} \
  > run.log 2>&1
"""
    started = time.monotonic()
    completed = execute(["ssh", host, command], check=False)
    wall = time.monotonic() - started
    local = LOCAL_RUN_ROOT / tag
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True, exist_ok=True)
    download = execute(
        ["rsync", "-az", f"{host}:{remote_out}/", str(local) + "/"],
        check=False,
    )
    status = {
        "case": name,
        "steps": steps,
        "returncode": completed.returncode,
        "download_returncode": download.returncode,
        "wall_seconds": wall,
        "remote_out": remote_out,
    }
    (local / "workstation_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    if completed.returncode != 0 or download.returncode != 0:
        raise RuntimeError(json.dumps(status, indent=2))
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="workstation-tail")
    parser.add_argument("--case", choices=["dt8", "dt16", "both"], default="both")
    parser.add_argument("--steps", type=int, default=502)
    args = parser.parse_args()
    selected = list(CASES) if args.case == "both" else [args.case]
    results = [run_case(args.host, name, args.steps) for name in selected]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
