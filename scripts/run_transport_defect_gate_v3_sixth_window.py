#!/usr/bin/env python3
"""Run the two pre-registered V3 sixth-window continuations on workstation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "reports" / "transport_residual_gate_v2"
V3 = ROOT / "reports" / "transport_residual_gate_v3"
LOCAL_RUNS = V3 / "workstation_runs"
REMOTE_REPO = "/home/zhiheng/PF/CUDA_STO_PF_transport_defect_v2"
REMOTE_RUNS = f"{REMOTE_REPO}/v3_runs"
EXPECTED_BINARY_SHA256 = (
    "7d76e43e262a99bc452efc008a1ea0e767745f423e508555eb13aed15fffc16d"
)
WINDOW_TIME_CODE = 1.5625
T_REAL_UNIT_S = 41.12958542455477
CASES = {
    "strict": {
        "gate_id": "G12",
        "dt_id": "dt32",
        "dt": 9.76562500000000054e-5,
        "steps": 16000,
        "v2_case": "V2_G12_dt32_5windows_holdout",
        "checkpoint_step": 80000,
    },
    "candidate": {
        "gate_id": "G9",
        "dt_id": "dt4",
        "dt": 7.81250000000000043e-4,
        "steps": 2000,
        "v2_case": "V2_G9_dt4_5windows_holdout",
        "checkpoint_step": 10000,
    },
}


def execute(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, text=True, capture_output=True)
    if check and result.returncode:
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


def one(root: Path, pattern: str) -> Path:
    values = list(root.rglob(pattern))
    if len(values) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, found {values}")
    return values[0]


def ensure_idle(host: str) -> None:
    probe = execute(["ssh", host, "pgrep -a -x main_cuda || true"])
    if probe.stdout.strip():
        raise RuntimeError("workstation main_cuda already active:\n" + probe.stdout)


def verify_binary(host: str) -> None:
    result = execute(
        ["ssh", host, f"sha256sum {shlex.quote(REMOTE_REPO + '/main_cuda')}"]
    )
    observed = result.stdout.split()[0]
    if observed != EXPECTED_BINARY_SHA256:
        raise RuntimeError(
            f"V3 must reuse exact V2 binary: expected {EXPECTED_BINARY_SHA256}, got {observed}"
        )


def source_assets(case: dict[str, object]) -> dict[str, Path]:
    v2_root = V2 / "workstation_runs" / str(case["v2_case"])
    step = int(case["checkpoint_step"])
    stem = f"ctot_checkpoint_step{step:06d}"
    return {
        "Ctot_init.raw": one(v2_root, stem + "_Ctot.raw"),
        "Ctot_nm1.raw": one(v2_root, stem + "_Ctot_nm1.raw"),
        "phi_init.raw": one(v2_root, stem + "_phi.raw"),
        "phi_nm1.raw": one(v2_root, stem + "_phi_nm1.raw"),
        "xB_init.raw": one(v2_root, stem + "_xB_alpha.raw"),
        "init_meta.json": one(v2_root, stem + "_meta.json"),
        "runtime.params": v2_root / "runtime.params",
    }


def count_complete_windows(root: Path) -> int:
    metrics = list(root.rglob("ctot_transport_defect_v2_window_metrics.csv"))
    if len(metrics) != 1:
        return 0
    with metrics[0].open(newline="", encoding="utf-8") as stream:
        return sum(row["record_type"] == "WINDOW" for row in csv.DictReader(stream))


def run_case(host: str, label: str, case: dict[str, object]) -> dict[str, object]:
    gate_id, dt_id = str(case["gate_id"]), str(case["dt_id"])
    case_id = f"V3_{gate_id}_{dt_id}_6th_window_holdout"
    local = LOCAL_RUNS / case_id
    status_path = local / "workstation_status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("returncode") == 0 and status.get("complete_windows") == 1:
            return status

    ensure_idle(host)
    assets = source_assets(case)
    meta = json.loads(assets["init_meta.json"].read_text(encoding="utf-8"))
    if int(meta.get("bdf2_history_valid", 0)) != 1:
        raise RuntimeError(f"{case_id} source checkpoint lacks valid BDF2 history")
    if abs(float(meta.get("time_code", 0.0)) - 5.0 * WINDOW_TIME_CODE) > 1.0e-9:
        raise RuntimeError(f"{case_id} source checkpoint is not the 5x endpoint")

    remote = f"{REMOTE_RUNS}/{case_id}"
    remote_input = f"{remote}/input"
    execute(
        [
            "ssh",
            host,
            f"rm -rf {shlex.quote(remote)} && mkdir -p {shlex.quote(remote_input)}",
        ]
    )
    for name, path in assets.items():
        execute(["rsync", "-az", str(path), f"{host}:{remote_input}/{name}"])

    dt, steps = float(case["dt"]), int(case["steps"])
    command = f"""set -e
repo={shlex.quote(REMOTE_REPO)}
input={shlex.quote(remote_input)}
out={shlex.quote(remote)}
cd "$out"
export CUDA_STO_RESULTS_ROOT="$out/Results"
started=$(date +%s.%N)
set +e
"$repo/main_cuda" 512 1 1 {dt:.17e} {steps} {steps} {steps} 0 \\
  --mode dynamics --pf-param-file "$input/runtime.params" --temperature-C 400 \\
  --init-mode raw_fields --init-phi-raw "$input/phi_init.raw" \\
  --init-xB-raw "$input/xB_init.raw" --init-Ctot-raw "$input/Ctot_init.raw" \\
  --init-Ctot-nm1-raw "$input/Ctot_nm1.raw" \\
  --init-phi-nm1-raw "$input/phi_nm1.raw" \\
  --init-meta "$input/init_meta.json" --init-case-tag {shlex.quote(case_id)} \\
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
    execute(["ssh", host, command], check=False)
    client_wall = time.monotonic() - started
    local.mkdir(parents=True, exist_ok=True)
    execute(
        [
            "rsync",
            "-az",
            "--exclude=*.vtk",
            f"{host}:{remote}/",
            str(local) + "/",
        ],
        check=False,
    )
    remote_status_path = local / "workstation_status.remote.json"
    remote_status = (
        json.loads(remote_status_path.read_text(encoding="utf-8"))
        if remote_status_path.is_file()
        else {}
    )
    summary_path = local / "v3_run_summary.json"
    execute(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_transport_gate_long_case.py"),
            str(local),
            "--output",
            str(summary_path),
        ]
    )
    status = {
        "schema": "TRANSPORT_DEFECT_GATE_V3_SIXTH_WINDOW_RUN_V1",
        "label": label,
        "case_id": case_id,
        "gate_id": gate_id,
        "dt_id": dt_id,
        "dt_code": dt,
        "steps": steps,
        "window_index": 6,
        "initial_time_code": float(meta["time_code"]),
        "final_time_code_expected": float(meta["time_code"]) + dt * steps,
        "physical_duration_s": dt * steps * T_REAL_UNIT_S,
        "source_checkpoint_step": int(case["checkpoint_step"]),
        "source_bdf2_history_valid": int(meta["bdf2_history_valid"]),
        "source_bdf2_accepted_step": int(meta["bdf2_accepted_step"]),
        "source_asset_sha256": {name: sha256(path) for name, path in assets.items()},
        "runtime_params_sha256": sha256(assets["runtime.params"]),
        "remote_binary_sha256": EXPECTED_BINARY_SHA256,
        "returncode": int(remote_status.get("returncode", 255)),
        "remote_wall_seconds": remote_status.get("wall_seconds"),
        "client_wall_seconds": client_wall,
        "complete_windows": count_complete_windows(local),
        "physics_changed": False,
        "transport_solver_changed": False,
        "residual_gate_changed": False,
        "dt_changed": False,
    }
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="workstation-tail")
    args = parser.parse_args()
    V3.mkdir(parents=True, exist_ok=True)
    ensure_idle(args.host)
    verify_binary(args.host)
    statuses = [run_case(args.host, label, case) for label, case in CASES.items()]
    manifest = {
        "schema": "TRANSPORT_DEFECT_GATE_V3_SIXTH_WINDOW_MANIFEST_V1",
        "contract_frozen_before_holdout": True,
        "extra_main_simulations": len(statuses),
        "cluster_used": False,
        "binary_sha256": EXPECTED_BINARY_SHA256,
        "statuses": statuses,
    }
    (V3 / "sixth_window_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0 if all(
        row["returncode"] == 0 and row["complete_windows"] == 1
        for row in statuses
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
