#!/usr/bin/env python3
"""Run the fixed-step transport residual-gate matrix on the workstation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "transport_residual_gate_v1"
LOCAL_RUN_ROOT = REPORT_ROOT / "workstation_runs"
COMMON_INPUT = (
    ROOT / "reports" / "bounded_retry_bdf2_v1" / "workstation_runs" /
    "equal_time_common_input"
)
REMOTE_REPO = "/home/zhiheng/PF/CUDA_STO_PF_transport_gate_v1"
REMOTE_RUN_ROOT = f"{REMOTE_REPO}/runs/transport_residual_gate_v1"
CONTRACT = "ACTIVE_MANIFOLD_BDF2_BOUNDED_RETRY_PRODUCTION_V1"
TIME_CODE = 1.5625
T_REAL_UNIT_S = 41.12958542455477

GATES = {
    "G12": 1.0e-12,
    "G10": 1.0e-10,
    "G9": 1.0e-9,
    "G8": 1.0e-8,
}
DT_CASES = {
    "dt16": (1.95312500000000011e-4, 8000),
    "dt8": (3.90625000000000022e-4, 4000),
    "dt4": (7.81250000000000043e-4, 2000),
    "dt2": (1.56250000000000009e-3, 1000),
    "dt1": (3.12500000000000017e-3, 500),
}


def execute(
    args: list[str], *, check: bool = True, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args, text=True, capture_output=True, timeout=timeout
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_idle(host: str) -> None:
    probe = execute(["ssh", host, "pgrep -a -x main_cuda || true"])
    if probe.stdout.strip():
        raise RuntimeError("workstation main_cuda already active:\n" + probe.stdout)


def stage_common_input(host: str) -> str:
    required = (
        "phi_init.raw",
        "xB_init.raw",
        "Ctot_init.raw",
        "init_meta.json",
        "runtime.base.params",
    )
    missing = [name for name in required if not (COMMON_INPUT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing common input assets: {missing}")
    remote = f"{REMOTE_RUN_ROOT}/common_input"
    execute(["ssh", host, f"mkdir -p {shlex.quote(remote)}"])
    execute(["rsync", "-az", str(COMMON_INPUT) + "/", f"{host}:{remote}/"])
    return remote


def build_params(gate: float, dt: float, diagnostics: bool = True) -> str:
    base = (COMMON_INPUT / "runtime.base.params").read_text(encoding="utf-8")
    return base + f"""

# CTOT_TRANSPORT_RESIDUAL_GATE_SWEEP_V1: only the absolute transport
# residual gate changes across candidates. Every physical and non-transport
# numerical parameter remains frozen by the baseline manifest.
dt={dt:.17e}
ctot_residual_abs_tol={gate:.17e}
ctot_residual_rel_tol=1.00000000000000004e-10
ctot_numerics_contract=ctot_jichen_imex_bdf2_active_manifold_v1
ctot_split_defect_policy=IMEX_BDF2_NO_POST_PHASE_POLISH
ctot_max_coupling_correctors=0
ctot_automatic_dt_growth=0
ctot_step_max_retries=0
BDF2_EVENT_PREFLIGHT_V1=1
BDF2_EVENT_BE_SUBCYCLING_V1=1
ctot_retry_acceptance_contract={CONTRACT}
ctot_debug_transport_floor_audit=0
ctot_transport_gate_trajectory_diagnostics={1 if diagnostics else 0}
"""


def local_case_dir(case_id: str) -> Path:
    return LOCAL_RUN_ROOT / case_id


def remote_case_dir(case_id: str) -> str:
    return f"{REMOTE_RUN_ROOT}/{case_id}"


def status_complete(status: dict[str, object], steps: int) -> bool:
    return (
        status.get("returncode") == 0
        and status.get("steps") == steps
        and status.get("completed_common_window") is True
    )


def pull_case(host: str, case_id: str) -> Path:
    local = local_case_dir(case_id)
    local.mkdir(parents=True, exist_ok=True)
    remote = remote_case_dir(case_id)
    execute(
        [
            "rsync", "-az", "--exclude=*.vtk",
            f"{host}:{remote}/", str(local) + "/",
        ],
        check=False,
    )
    return local


def run_case(
    host: str,
    remote_input: str,
    gate_id: str,
    dt_id: str,
    *,
    force: bool,
    diagnostics: bool = True,
    steps_override: int | None = None,
) -> dict[str, object]:
    gate = GATES[gate_id]
    dt, matrix_steps = DT_CASES[dt_id]
    steps = steps_override if steps_override is not None else matrix_steps
    case_id = (
        f"{gate_id}_{dt_id}_diag_{'on' if diagnostics else 'off'}_{steps}"
    )
    local = local_case_dir(case_id)
    status_path = local / "workstation_status.json"
    if status_path.is_file() and not force:
        prior = json.loads(status_path.read_text(encoding="utf-8"))
        if status_complete(prior, steps):
            prior["resumed_existing"] = True
            print(json.dumps(prior, sort_keys=True), flush=True)
            return prior

    ensure_idle(host)
    remote = remote_case_dir(case_id)
    execute([
        "ssh", host,
        f"rm -rf {shlex.quote(remote)} && mkdir -p {shlex.quote(remote)}",
    ])
    LOCAL_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    generated = LOCAL_RUN_ROOT / f".{case_id}.runtime.params"
    generated.write_text(
        build_params(gate, dt, diagnostics=diagnostics), encoding="utf-8"
    )
    params_hash = sha256(generated)
    execute(["rsync", "-az", str(generated), f"{host}:{remote}/runtime.params"])
    generated.unlink()

    tag = case_id
    command = f"""set -e
repo={shlex.quote(REMOTE_REPO)}
input={shlex.quote(remote_input)}
out={shlex.quote(remote)}
cd "$out"
export CUDA_STO_RESULTS_ROOT="$out/Results"
started=$(date +%s.%N)
set +e
"$repo/main_cuda" 512 1 1 {dt:.17e} {steps} {steps} {steps} 0 \\
  --mode dynamics --pf-param-file "$out/runtime.params" --temperature-C 400 \\
  --init-mode raw_fields --init-phi-raw "$input/phi_init.raw" \\
  --init-xB-raw "$input/xB_init.raw" --init-Ctot-raw "$input/Ctot_init.raw" \\
  --init-meta "$input/init_meta.json" --init-case-tag {shlex.quote(tag)} \\
  > run.log 2>&1
rc=$?
set -e
ended=$(date +%s.%N)
python3 - "$rc" "$started" "$ended" > workstation_status.remote.json <<'PY'
import json, sys
rc, start, end = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
print(json.dumps({{"returncode": rc, "wall_seconds": end-start}}, indent=2))
PY
exit 0
"""
    started = time.monotonic()
    completed = execute(["ssh", host, command], check=False)
    client_wall = time.monotonic() - started
    local = pull_case(host, case_id)
    remote_status_path = local / "workstation_status.remote.json"
    remote_status = (
        json.loads(remote_status_path.read_text(encoding="utf-8"))
        if remote_status_path.is_file()
        else {}
    )
    run_rc = int(remote_status.get("returncode", completed.returncode))
    log_text = (
        (local / "run.log").read_text(encoding="utf-8", errors="replace")
        if (local / "run.log").is_file()
        else ""
    )
    completed_window = (
        run_rc == 0
        and f"CTOT_BOUNDED_RETRY_SUMMARY contract={CONTRACT} "
            f"macro_steps={steps}" in log_text
    )
    status: dict[str, object] = {
        "case_id": case_id,
        "gate_id": gate_id,
        "requested_transport_gate": gate,
        "relative_gate": 1.0e-10,
        "dt_id": dt_id,
        "dt_code": dt,
        "dt_physical_s": dt * T_REAL_UNIT_S,
        "steps": steps,
        "equal_time_code": dt * steps,
        "equal_time_physical_s": dt * steps * T_REAL_UNIT_S,
        "diagnostics_enabled": diagnostics,
        "runtime_params_sha256": params_hash,
        "remote_binary": f"{REMOTE_REPO}/main_cuda",
        "remote_out": remote,
        "returncode": run_rc,
        "completed_common_window": completed_window,
        "client_wall_seconds": client_wall,
        "remote_wall_seconds": remote_status.get("wall_seconds"),
        "resumed_existing": False,
    }
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, sort_keys=True), flush=True)
    return status


def selected_pairs(gate: str, dt_case: str) -> list[tuple[str, str]]:
    gates = list(GATES) if gate == "all" else [gate]
    dts = list(DT_CASES) if dt_case == "all" else [dt_case]
    return [(gate_id, dt_id) for gate_id in gates for dt_id in dts]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="workstation-tail")
    parser.add_argument("--gate", choices=[*GATES, "all"], default="all")
    parser.add_argument("--dt-case", choices=[*DT_CASES, "all"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--diagnostics-off", action="store_true")
    parser.add_argument("--steps-override", type=int)
    args = parser.parse_args()

    remote_input = stage_common_input(args.host)
    statuses = []
    for gate_id, dt_id in selected_pairs(args.gate, args.dt_case):
        statuses.append(
            run_case(
                args.host,
                remote_input,
                gate_id,
                dt_id,
                force=args.force,
                diagnostics=not args.diagnostics_off,
                steps_override=args.steps_override,
            )
        )
    print(json.dumps(statuses, indent=2, sort_keys=True))
    return 0 if all(int(row["returncode"]) == 0 for row in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
