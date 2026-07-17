#!/usr/bin/env python3
"""Run the pre-registered 4x transport-gate accumulation window."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

from run_transport_residual_gate_v1_workstation import build_params


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "transport_residual_gate_v1"
LOCAL_RUNS = REPORT / "long_workstation_runs"
COMMON = ROOT / "reports" / "bounded_retry_bdf2_v1" / "workstation_runs" / "equal_time_common_input"
SHORT_RUNS = REPORT / "workstation_runs"
REFERENCE = ROOT / "reports" / "bounded_retry_bdf2_v1" / "workstation_runs" / "fine_dt32_common_state_16000"
REMOTE_REPO = "/home/zhiheng/PF/CUDA_STO_PF_transport_gate_v1"
REMOTE_RUNS = f"{REMOTE_REPO}/runs/transport_residual_gate_v1_long"
REMOTE_COMMON = f"{REMOTE_REPO}/runs/transport_residual_gate_v1/common_input"
T_REAL_UNIT_S = 41.12958542455477
TIME_CODE = 6.25
DT = {
    "dt32": (9.76562500000000054e-5, 64000),
    "dt16": (1.95312500000000011e-4, 32000),
    "dt8": (3.90625000000000022e-4, 16000),
    "dt4": (7.81250000000000043e-4, 8000),
    "dt2": (1.56250000000000009e-3, 4000),
    "dt1": (3.12500000000000017e-3, 2000),
}
COMMON_STEPS = {key: steps // 4 for key, (_, steps) in DT.items()}
GATES = {"G12": 1.0e-12, "G10": 1.0e-10, "G9": 1.0e-9, "G8": 1.0e-8}


def execute(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(f"command failed: {' '.join(args)}\n{result.stdout}\n{result.stderr}")
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


def stage(host: str) -> None:
    execute(["ssh", host, f"mkdir -p {shlex.quote(REMOTE_COMMON)} {shlex.quote(REMOTE_RUNS)}"])
    execute(["rsync", "-az", str(COMMON) + "/", f"{host}:{REMOTE_COMMON}/"])
    execute([
        "rsync", "-az", str(ROOT / "scripts" / "summarize_transport_gate_long_case.py"),
        f"{host}:{REMOTE_REPO}/scripts/",
    ])


def one(root: Path, pattern: str) -> Path:
    found = list(root.rglob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, got {found}")
    return found[0]


def restart_source(gate_id: str, dt_id: str) -> Path:
    if dt_id == "dt32":
        return REFERENCE
    return SHORT_RUNS / f"{gate_id}_{dt_id}_diag_on_{COMMON_STEPS[dt_id]}"


def stage_restart_input(
    host: str, gate_id: str, dt_id: str, remote: str
) -> tuple[str, dict[str, str]]:
    source = restart_source(gate_id, dt_id)
    common_steps = COMMON_STEPS[dt_id]
    source_paths = {
        "Ctot_init.raw": one(source, f"*step{common_steps:06d}_Ctot.raw"),
        "phi_init.raw": one(source, f"*step{common_steps:06d}_phi.raw"),
        "xB_init.raw": one(source, f"*step{common_steps:06d}_xB_alpha.raw"),
        "source_meta": one(source, f"*step{common_steps:06d}_meta.json"),
    }
    remote_input = f"{remote}/restart_input"
    execute(["ssh", host, f"mkdir -p {shlex.quote(remote_input)}"])
    for name in ("Ctot_init.raw", "phi_init.raw", "xB_init.raw"):
        execute(["rsync", "-az", str(source_paths[name]), f"{host}:{remote_input}/{name}"])

    source_meta = json.loads(source_paths["source_meta"].read_text(encoding="utf-8"))
    history = {
        "bdf2_Ctot_nm1_file": ("Ctot_nm1.raw", "Ctot_nm1"),
        "bdf2_phi_nm1_file": ("phi_nm1.raw", "phi_nm1"),
    }
    for key, (name, stem) in history.items():
        local_found = list(source.rglob(f"*step{common_steps:06d}_{stem}.raw"))
        if len(local_found) == 1:
            execute(["rsync", "-az", str(local_found[0]), f"{host}:{remote_input}/{name}"])
        elif source_meta.get(key):
            execute([
                "ssh", host,
                f"cp {shlex.quote(str(source_meta[key]))} {shlex.quote(remote_input + '/' + name)}",
            ])
        else:
            raise RuntimeError(f"missing restart history {key} for {gate_id}+{dt_id}")
        source_meta[key] = f"{remote_input}/{name}"

    generated_meta = LOCAL_RUNS / f".{gate_id}_{dt_id}.restart_meta.json"
    generated_meta.parent.mkdir(parents=True, exist_ok=True)
    generated_meta.write_text(json.dumps(source_meta, indent=2) + "\n", encoding="utf-8")
    execute(["rsync", "-az", str(generated_meta), f"{host}:{remote_input}/init_meta.json"])
    generated_meta.unlink()
    hashes = {
        name: sha256(path) for name, path in source_paths.items()
    }
    remote_hashes = execute([
        "ssh", host,
        "sha256sum "
        + " ".join(
            shlex.quote(f"{remote_input}/{name}")
            for name in ("Ctot_nm1.raw", "phi_nm1.raw", "init_meta.json")
        ),
    ]).stdout.splitlines()
    for line in remote_hashes:
        digest, path = line.split(maxsplit=1)
        hashes[f"staged_{Path(path).name}"] = digest
    return remote_input, hashes


def candidates() -> list[tuple[str, str]]:
    path = REPORT / "equal_time_metrics.csv"
    if not path.is_file():
        raise RuntimeError("run equal-time analyzer before the long window")
    selected: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("full_common_window_pass", "").lower() == "true":
                selected.append((row["gate_id"], row["dt_id"]))
    return selected


def pull(host: str, remote: str, local: Path) -> None:
    local.mkdir(parents=True, exist_ok=True)
    execute([
        "rsync", "-az", "--include=*/", "--include=*.raw", "--include=*.json",
        "--include=*.params", "--include=run.key.log", "--exclude=*",
        f"{host}:{remote}/", str(local) + "/",
    ], check=False)


def run_case(host: str, gate_id: str, dt_id: str, *, force: bool) -> dict[str, object]:
    gate = GATES[gate_id]
    dt, total_steps = DT[dt_id]
    common_steps = COMMON_STEPS[dt_id]
    steps = total_steps - common_steps
    case_id = f"LONG_{gate_id}_{dt_id}_{total_steps}"
    local = LOCAL_RUNS / case_id
    status_path = local / "workstation_status.json"
    if status_path.is_file() and not force:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("returncode") == 0 and status.get("completed_long_window") is True:
            print(json.dumps(status, sort_keys=True), flush=True)
            return status

    ensure_idle(host)
    remote = f"{REMOTE_RUNS}/{case_id}"
    execute(["ssh", host, f"rm -rf {shlex.quote(remote)} && mkdir -p {shlex.quote(remote)}"])
    remote_input, restart_hashes = stage_restart_input(
        host, gate_id, dt_id, remote
    )
    remote_binary_sha256 = execute([
        "ssh", host, f"sha256sum {shlex.quote(REMOTE_REPO + '/main_cuda')}",
    ]).stdout.split()[0]
    generated = LOCAL_RUNS / f".{case_id}.runtime.params"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(build_params(gate, dt, diagnostics=True), encoding="utf-8")
    params_hash = sha256(generated)
    execute(["rsync", "-az", str(generated), f"{host}:{remote}/runtime.params"])
    generated.unlink()
    interval = common_steps
    command = f"""set -e
repo={shlex.quote(REMOTE_REPO)}
input={shlex.quote(remote_input)}
out={shlex.quote(remote)}
cd \"$out\"
export CUDA_STO_RESULTS_ROOT=\"$out/Results\"
started=$(date +%s.%N)
set +e
\"$repo/main_cuda\" 512 1 1 {dt:.17e} {steps} {interval} {interval} 0 \\
  --mode dynamics --pf-param-file \"$out/runtime.params\" --temperature-C 400 \\
  --init-mode raw_fields --init-phi-raw \"$input/phi_init.raw\" \\
  --init-xB-raw \"$input/xB_init.raw\" --init-Ctot-raw \"$input/Ctot_init.raw\" \\
  --init-Ctot-nm1-raw \"$input/Ctot_nm1.raw\" \\
  --init-phi-nm1-raw \"$input/phi_nm1.raw\" \\
  --init-meta \"$input/init_meta.json\" --init-case-tag {shlex.quote(case_id)} \\
  > run.log 2>&1
rc=$?
set -e
ended=$(date +%s.%N)
python3 - \"$rc\" \"$started\" \"$ended\" > workstation_status.remote.json <<'PY'
import json, sys
rc, start, end = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
print(json.dumps({{"returncode": rc, "wall_seconds": end-start}}, indent=2))
PY
grep -E 'CTOT_BOUNDED_RETRY_SUMMARY|CTOT_TRANSPORT_GATE_TRAJECTORY|\\[fatal\\]|\\[reject\\]' run.log > run.key.log || true
python3 \"$repo/scripts/summarize_transport_gate_long_case.py\" \"$out\" --output \"$out/long_remote_summary.json\"
exit 0
"""
    started = time.monotonic()
    execute(["ssh", host, command], check=False)
    client_wall = time.monotonic() - started
    pull(host, remote, local)
    remote_status_path = local / "workstation_status.remote.json"
    remote_status = json.loads(remote_status_path.read_text(encoding="utf-8")) if remote_status_path.is_file() else {}
    remote_summary_path = local / "long_remote_summary.json"
    remote_summary = json.loads(remote_summary_path.read_text(encoding="utf-8")) if remote_summary_path.is_file() else {}
    bounded = remote_summary.get("bounded_retry_summary", {})
    run_rc = int(remote_status.get("returncode", 255))
    complete = run_rc == 0 and int(bounded.get("macro_steps", 0)) == steps
    status: dict[str, object] = {
        "case_id": case_id,
        "gate_id": gate_id,
        "requested_transport_gate": gate,
        "relative_gate": 1.0e-10,
        "dt_id": dt_id,
        "dt_code": dt,
        "dt_physical_s": dt * T_REAL_UNIT_S,
        "steps": steps,
        "total_steps_from_common_origin": total_steps,
        "common_steps_reused": common_steps,
        "continuation_steps": steps,
        "long_time_code": TIME_CODE,
        "long_time_physical_s": TIME_CODE * T_REAL_UNIT_S,
        "continuation_time_code": dt * steps,
        "continuation_time_physical_s": dt * steps * T_REAL_UNIT_S,
        "window_interval_steps": interval,
        "restart_source": str(restart_source(gate_id, dt_id)),
        "restart_input_sha256": restart_hashes,
        "runtime_params_sha256": params_hash,
        "remote_binary": f"{REMOTE_REPO}/main_cuda",
        "remote_binary_sha256": remote_binary_sha256,
        "remote_out": remote,
        "returncode": run_rc,
        "completed_long_window": complete,
        "client_wall_seconds": client_wall,
        "remote_wall_seconds": remote_status.get("wall_seconds"),
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, sort_keys=True), flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="workstation-tail")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--candidate", action="append", default=[])
    args = parser.parse_args()
    stage(args.host)
    pairs = [("G12", "dt32")]
    if not args.reference_only:
        pairs.extend(candidates())
    if args.candidate:
        requested = {tuple(value.split(":", 1)) for value in args.candidate}
        pairs = [pair for pair in pairs if pair == ("G12", "dt32") or pair in requested]
    statuses = [run_case(args.host, gate, dt_id, force=args.force) for gate, dt_id in pairs]
    print(json.dumps(statuses, indent=2, sort_keys=True))
    return 0 if all(row["returncode"] == 0 for row in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
