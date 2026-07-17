#!/usr/bin/env python3
"""Run at most three V2 replay-plus-holdout simulations on the workstation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

try:
    from .run_transport_residual_gate_v1_workstation import build_params
except ImportError:  # Direct script execution keeps scripts/ on sys.path.
    from run_transport_residual_gate_v1_workstation import build_params


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "reports" / "transport_residual_gate_v1"
V2 = ROOT / "reports" / "transport_residual_gate_v2"
LOCAL_RUNS = V2 / "workstation_runs"
COMMON = (
    ROOT
    / "reports"
    / "bounded_retry_bdf2_v1"
    / "workstation_runs"
    / "equal_time_common_input"
)
REMOTE_REPO = "/home/zhiheng/PF/CUDA_STO_PF_transport_defect_v2"
REMOTE_RUNS = f"{REMOTE_REPO}/runs"
REMOTE_COMMON = f"{REMOTE_REPO}/common_input"
WINDOW_TIME_CODE = 1.5625
WINDOW_COUNT = 5
T_REAL_UNIT_S = 41.12958542455477
GATES = {"G12": 1.0e-12, "G10": 1.0e-10, "G9": 1.0e-9, "G8": 1.0e-8}
DT = {
    "dt32": (9.76562500000000054e-5, 16000),
    "dt16": (1.95312500000000011e-4, 8000),
    "dt8": (3.90625000000000022e-4, 4000),
    "dt4": (7.81250000000000043e-4, 2000),
    "dt2": (1.56250000000000009e-3, 1000),
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


def ensure_idle(host: str) -> None:
    probe = execute(["ssh", host, "pgrep -a -x main_cuda || true"])
    if probe.stdout.strip():
        raise RuntimeError("workstation main_cuda already active:\n" + probe.stdout)


def formal_queue_complete() -> None:
    path = V1 / "long_window_candidate_summary.csv"
    if not path.is_file():
        raise RuntimeError("V1 long-window summary is missing")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 13:
        raise RuntimeError(f"expected 13 V1 long candidates, found {len(rows)}")
    incomplete = [
        f"{row['gate_id']}+{row['dt_id']}:{row['long_window_status']}"
        for row in rows
        if row["long_window_status"] in ("", "PENDING")
    ]
    if incomplete:
        raise RuntimeError(
            "formal V1 long queue must finish before V2 holdout: "
            + ", ".join(incomplete)
        )


def auto_candidates() -> list[tuple[str, str]]:
    path = V1 / "long_window_candidate_summary.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    eligible = [
        row
        for row in rows
        if row.get("long_hard_gates_pass", "").lower() == "true"
        and row.get("long_retry_efficiency_gate_pass", "").lower() == "true"
        and row.get("all_registered_windows_accuracy_pass", "").lower()
        == "true"
        and row.get("signed_residual_bias_growth_pass", "").lower() == "true"
    ]
    if not eligible:
        raise RuntimeError("no candidate satisfies non-V1-peak production gates")
    eligible.sort(key=lambda row: float(row["physical_s_per_GPU_hour"]), reverse=True)
    selected = [(eligible[0]["gate_id"], eligible[0]["dt_id"])]
    if len(eligible) > 1:
        best = float(eligible[0]["physical_s_per_GPU_hour"])
        second = float(eligible[1]["physical_s_per_GPU_hour"])
        # Frozen before holdout: a runner-up is close only within 10% throughput.
        if second >= 0.9 * best:
            selected.append((eligible[1]["gate_id"], eligible[1]["dt_id"]))
    return selected


def source_files() -> list[Path]:
    names = [ROOT / "Makefile"]
    for pattern in ("*.cu", "*.h"):
        names.extend(sorted(ROOT.glob(pattern)))
    return [path for path in names if path.is_file()]


def stage_and_build(host: str) -> dict[str, object]:
    ensure_idle(host)
    execute(["ssh", host, f"mkdir -p {shlex.quote(REMOTE_REPO)}"])
    files = source_files()
    execute(["rsync", "-az", *map(str, files), f"{host}:{REMOTE_REPO}/"])
    command = (
        f"cd {shlex.quote(REMOTE_REPO)} && "
        "make main_cuda NVCC=/usr/local/cuda-12.9/bin/nvcc "
        "CUDA_ROOT=/usr/local/cuda-12.9 -j1"
    )
    built = execute(["ssh", host, command], check=False)
    if built.returncode:
        raise RuntimeError("V2 workstation build failed:\n" + built.stdout + built.stderr)
    binary_hash = execute(
        ["ssh", host, f"sha256sum {shlex.quote(REMOTE_REPO + '/main_cuda')}"]
    ).stdout.split()[0]
    source_hashes = {path.name: sha256(path) for path in files}
    report = {
        "remote_repo": REMOTE_REPO,
        "binary_sha256": binary_hash,
        "source_sha256": source_hashes,
        "build_stdout_tail": built.stdout[-4000:],
        "build_stderr_tail": built.stderr[-4000:],
        "physical_model_changed": False,
        "transport_solver_changed": False,
        "transport_gate_changed": False,
    }
    V2.mkdir(parents=True, exist_ok=True)
    (V2 / "v2_build_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def stage_common(host: str) -> None:
    required = (
        "phi_init.raw",
        "xB_init.raw",
        "Ctot_init.raw",
        "init_meta.json",
        "runtime.base.params",
    )
    missing = [name for name in required if not (COMMON / name).is_file()]
    if missing:
        raise RuntimeError(f"missing common input assets: {missing}")
    execute(
        [
            "ssh",
            host,
            f"mkdir -p {shlex.quote(REMOTE_COMMON)} {shlex.quote(REMOTE_RUNS)}",
        ]
    )
    execute(["rsync", "-az", str(COMMON) + "/", f"{host}:{REMOTE_COMMON}/"])


def params(gate: float, dt: float) -> str:
    return build_params(gate, dt, diagnostics=True) + f"""

# PHYSICALLY_NORMALIZED_TRANSPORT_DEFECT_GATE_V2 observer only.
ctot_transport_defect_v2_diagnostics=1
ctot_transport_defect_v2_window_time_code={WINDOW_TIME_CODE:.17e}
"""


def pull(host: str, remote: str, local: Path) -> None:
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


def run_case(host: str, gate_id: str, dt_id: str, binary_hash: str) -> dict[str, object]:
    ensure_idle(host)
    gate = GATES[gate_id]
    dt, steps_per_window = DT[dt_id]
    steps = WINDOW_COUNT * steps_per_window
    case_id = f"V2_{gate_id}_{dt_id}_{WINDOW_COUNT}windows_holdout"
    local = LOCAL_RUNS / case_id
    status_path = local / "workstation_status.json"
    if status_path.is_file():
        prior = json.loads(status_path.read_text(encoding="utf-8"))
        if prior.get("returncode") == 0 and prior.get("complete_windows") == WINDOW_COUNT:
            return prior

    remote = f"{REMOTE_RUNS}/{case_id}"
    execute(["ssh", host, f"rm -rf {shlex.quote(remote)} && mkdir -p {shlex.quote(remote)}"])
    generated = LOCAL_RUNS / f".{case_id}.runtime.params"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(params(gate, dt), encoding="utf-8")
    params_hash = sha256(generated)
    execute(["rsync", "-az", str(generated), f"{host}:{remote}/runtime.params"])
    generated.unlink()
    command = f"""set -e
repo={shlex.quote(REMOTE_REPO)}
input={shlex.quote(REMOTE_COMMON)}
out={shlex.quote(remote)}
cd "$out"
export CUDA_STO_RESULTS_ROOT="$out/Results"
started=$(date +%s.%N)
set +e
"$repo/main_cuda" 512 1 1 {dt:.17e} {steps} {steps_per_window} {steps_per_window} 0 \\
  --mode dynamics --pf-param-file "$out/runtime.params" --temperature-C 400 \\
  --init-mode raw_fields --init-phi-raw "$input/phi_init.raw" \\
  --init-xB-raw "$input/xB_init.raw" --init-Ctot-raw "$input/Ctot_init.raw" \\
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
    pull(host, remote, local)
    remote_status_path = local / "workstation_status.remote.json"
    remote_status = (
        json.loads(remote_status_path.read_text(encoding="utf-8"))
        if remote_status_path.is_file()
        else {}
    )
    metrics = list(local.rglob("ctot_transport_defect_v2_window_metrics.csv"))
    complete_windows = 0
    if len(metrics) == 1:
        with metrics[0].open(newline="", encoding="utf-8") as stream:
            complete_windows = sum(
                row["record_type"] == "WINDOW" for row in csv.DictReader(stream)
            )
    status = {
        "case_id": case_id,
        "gate_id": gate_id,
        "dt_id": dt_id,
        "requested_transport_gate": gate,
        "dt_code": dt,
        "steps_per_window": steps_per_window,
        "steps": steps,
        "window_count": WINDOW_COUNT,
        "holdout_window_index": WINDOW_COUNT,
        "physical_time_code": dt * steps,
        "physical_time_s": dt * steps * T_REAL_UNIT_S,
        "runtime_params_sha256": params_hash,
        "remote_binary_sha256": binary_hash,
        "returncode": int(remote_status.get("returncode", 255)),
        "remote_wall_seconds": remote_status.get("wall_seconds"),
        "client_wall_seconds": client_wall,
        "complete_windows": complete_windows,
        "replay_windows": 4,
        "new_holdout_windows": 1,
        "solver_state_modified_by_observer": False,
    }
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, sort_keys=True), flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="workstation-tail")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="explicit GATE:dt pair; otherwise frozen automatic selection",
    )
    args = parser.parse_args()
    formal_queue_complete()
    requested = [tuple(value.split(":", 1)) for value in args.candidate]
    candidates = requested or auto_candidates()
    pairs = [("G12", "dt32"), *candidates]
    pairs = list(dict.fromkeys(pairs))
    if len(pairs) > 3:
        raise RuntimeError(f"V2 permits at most three holdout simulations, got {pairs}")
    build = stage_and_build(args.host)
    stage_common(args.host)
    statuses = [
        run_case(args.host, gate_id, dt_id, str(build["binary_sha256"]))
        for gate_id, dt_id in pairs
    ]
    manifest = {
        "schema": "TRANSPORT_DEFECT_GATE_V2_HOLDOUT_MANIFEST_V1",
        "selection_contract": "BEST_NON_PEAK_HARD_GATE_THROUGHPUT_PLUS_RUNNER_WITHIN_10_PERCENT",
        "pairs": [list(pair) for pair in pairs],
        "extra_holdout_simulations": len(pairs),
        "existing_long_window_queue_preserved": True,
        "statuses": statuses,
    }
    V2.mkdir(parents=True, exist_ok=True)
    (V2 / "holdout_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if all(
        row["returncode"] == 0 and row["complete_windows"] == WINDOW_COUNT
        for row in statuses
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
