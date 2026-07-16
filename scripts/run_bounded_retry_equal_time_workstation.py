#!/usr/bin/env python3
"""Run the common-state bounded-retry BDF2 equal-time matrix on workstation."""

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
LOCAL_ROOT = ROOT / "reports" / "bounded_retry_bdf2_v1" / "workstation_runs"
REMOTE_REPO = "/home/zhiheng/PF/CUDA_STO_PF_lie_be_v2_20260716"
REMOTE_ROOT = f"{REMOTE_REPO}/runs/bounded_retry_bdf2_v1/equal_time"
CONTRACT = "ACTIVE_MANIFOLD_BDF2_BOUNDED_RETRY_PRODUCTION_V1"
TIME_CODE = 1.5625
CASES = {
    "dt4": (7.81250000000000043e-4, 2000),
    "dt8": (3.90625000000000022e-4, 4000),
    "dt16": (1.95312500000000011e-4, 8000),
    "fine_dt32": (9.76562500000000054e-5, 16000),
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
    probe = execute(["ssh", host, "pgrep -a -x main_cuda || true"])
    if probe.stdout.strip():
        raise RuntimeError("workstation main_cuda already active:\n" + probe.stdout)


def stage_input(host: str) -> str:
    staging = LOCAL_ROOT / "equal_time_common_input"
    staging.mkdir(parents=True, exist_ok=True)
    for name in ("phi_init.raw", "xB_init.raw", "Ctot_init.raw", "init_meta.json"):
        shutil.copy2(SOURCE / name, staging / name)
    shutil.copy2(SOURCE / "runtime.params", staging / "runtime.base.params")
    remote = f"{REMOTE_ROOT}/common_input"
    execute(["ssh", host, f"mkdir -p {shlex.quote(remote)}"])
    execute(["rsync", "-az", str(staging) + "/", f"{host}:{remote}/"])
    return remote


def remote_complete(host: str, remote: str, steps: int) -> bool:
    marker = (
        f"test -f {shlex.quote(remote + '/run.log')} && "
        f"grep -q 'CTOT_BOUNDED_RETRY_SUMMARY.*macro_steps={steps}' "
        f"{shlex.quote(remote + '/run.log')}"
    )
    return execute(["ssh", host, marker], check=False).returncode == 0


def pull_evidence(host: str, remote: str, tag: str, steps: int) -> Path:
    local = LOCAL_ROOT / tag
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True)
    for name in ("run.log", "runtime.params", "workstation_status.json"):
        execute(
            ["rsync", "-az", f"{host}:{remote}/{name}", str(local) + "/"],
            check=False,
        )
    listing = execute([
        "ssh", host,
        "find " + shlex.quote(remote + "/Results") +
        " -type f \\( -name 'ctot_retry_attempts.csv' -o "
        "-name 'ctot_failed_cell_state.csv' -o "
        "-name 'ctot_acceptance_predicate.csv' -o "
        "-name 'ctot_energy_work.csv' -o "
        "-name 'ctot_phase_transaction.csv' -o "
        "-name 'ctot_split_step_metrics.csv' -o "
        f"-name 'ctot_checkpoint_step{steps:06d}_Ctot.raw' -o "
        f"-name 'ctot_checkpoint_step{steps:06d}_phi.raw' -o "
        f"-name 'ctot_checkpoint_step{steps:06d}_xB_alpha.raw' -o "
        f"-name 'ctot_checkpoint_step{steps:06d}_meta.json' \\) -print",
    ])
    for remote_file in filter(None, listing.stdout.splitlines()):
        execute(["rsync", "-az", f"{host}:{remote_file}", str(local) + "/"])
    return local


def run_case(host: str, common_input: str, name: str, force: bool) -> dict[str, object]:
    dt, steps = CASES[name]
    tag = f"{name}_common_state_{steps}"
    remote = f"{REMOTE_ROOT}/{tag}"
    if remote_complete(host, remote, steps) and not force:
        local = pull_evidence(host, remote, tag, steps)
        status = json.loads((local / "workstation_status.json").read_text())
        status["resumed_existing"] = True
        print(json.dumps(status), flush=True)
        return status

    ensure_idle(host)
    execute(["ssh", host, f"rm -rf {shlex.quote(remote)} && mkdir -p {shlex.quote(remote)}"])
    params = (SOURCE / "runtime.params").read_text(encoding="utf-8") + f"""

# Bounded-retry common-state equal-time qualification; physics unchanged.
dt={dt:.17e}
ctot_numerics_contract=ctot_jichen_imex_bdf2_active_manifold_v1
ctot_split_defect_policy=IMEX_BDF2_NO_POST_PHASE_POLISH
ctot_max_coupling_correctors=0
ctot_automatic_dt_growth=0
ctot_step_max_retries=0
BDF2_EVENT_PREFLIGHT_V1=1
BDF2_EVENT_BE_SUBCYCLING_V1=1
ctot_retry_acceptance_contract={CONTRACT}
"""
    local_params = LOCAL_ROOT / f".{tag}.runtime.params"
    local_params.parent.mkdir(parents=True, exist_ok=True)
    local_params.write_text(params, encoding="utf-8")
    execute(["rsync", "-az", str(local_params), f"{host}:{remote}/runtime.params"])
    local_params.unlink()

    command = f"""set -e
repo={shlex.quote(REMOTE_REPO)}
input={shlex.quote(common_input)}
out={shlex.quote(remote)}
cd "$out"
export CUDA_STO_RESULTS_ROOT="$out/Results"
started=$(date +%s.%N)
set +e
"$repo/main_cuda" 512 1 1 {dt:.17e} {steps} {steps} {steps} 0 \
  --mode dynamics --pf-param-file "$out/runtime.params" --temperature-C 400 \
  --init-mode raw_fields --init-phi-raw "$input/phi_init.raw" \
  --init-xB-raw "$input/xB_init.raw" --init-Ctot-raw "$input/Ctot_init.raw" \
  --init-meta "$input/init_meta.json" --init-case-tag {shlex.quote(tag)} \
  > run.log 2>&1
rc=$?
set -e
ended=$(date +%s.%N)
python3 - "$rc" "$started" "$ended" <<'PY'
import json, sys
rc, start, end = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
print(json.dumps({{"returncode": rc, "wall_seconds": end-start}}, indent=2))
PY
python3 - "$rc" "$started" "$ended" > workstation_status.json <<'PY'
import json, sys
rc, start, end = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
print(json.dumps({{"returncode": rc, "wall_seconds": end-start}}, indent=2))
PY
exit "$rc"
"""
    started = time.monotonic()
    completed = execute(["ssh", host, command], check=False)
    client_wall = time.monotonic() - started
    local = pull_evidence(host, remote, tag, steps)
    status_path = local / "workstation_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status.update({
        "case": name,
        "dt_code": dt,
        "steps": steps,
        "equal_time_code": dt * steps,
        "client_wall_seconds": client_wall,
        "remote_out": remote,
        "returncode": completed.returncode,
        "resumed_existing": False,
    })
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status), flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed: {json.dumps(status, indent=2)}")
    if abs(dt * steps - TIME_CODE) > 1.0e-14:
        raise RuntimeError(f"equal-time contract violated by {name}")
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="workstation-tail")
    parser.add_argument(
        "--case", choices=[*CASES, "all"], default="all",
        help="run one case or the full serial matrix",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    common_input = stage_input(args.host)
    selected = list(CASES) if args.case == "all" else [args.case]
    statuses = [run_case(args.host, common_input, name, args.force) for name in selected]
    print(json.dumps(statuses, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
