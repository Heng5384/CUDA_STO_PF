#!/usr/bin/env python3
"""Run the versioned low-memory transport ablation on the workstation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_transport_residual_gate_v1_workstation import build_params


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "low_memory_transport_v1" / "workstation_runs"
REMOTE_REPO = "/home/zhiheng/PF/CUDA_STO_PF_low_memory_transport_v1"
REMOTE_COMMON = "/home/zhiheng/PF/CUDA_STO_PF_transport_defect_v2/common_input"
REMOTE_RUNS = REMOTE_REPO + "/runs"
DEVELOPMENT_GATE = 1.0e-10

BOUNDED_RETRY_SUMMARY_RE = re.compile(
    r"CTOT_BOUNDED_RETRY_SUMMARY .*?"
    r"macro_steps=(?P<macro_steps>\d+) .*?"
    r"macro_hard_rejects=(?P<macro_hard_rejects>\d+) .*?"
    r"internal_trial_rejects=(?P<internal_trial_rejects>\d+) .*?"
    r"fallback_macros=(?P<fallback_macros>\d+)"
)

VERSIONS = {
    "V0": ("LEGACY_CURRENT", 0),
    "V1": ("LOW_MEMORY_GLOBALIZATION_V1", 1),
    "V2": ("LOW_MEMORY_GLOBALIZATION_V1", 3),
    "V3": ("LOW_MEMORY_GLOBALIZATION_V1", 7),
    "V4": ("LOW_MEMORY_GLOBALIZATION_V1", 15),
    "V5": ("LOW_MEMORY_GLOBALIZATION_V1", 31),
}

DT = {
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


def ensure_idle(host: str) -> None:
    result = execute(["ssh", host, "pgrep -a -x main_cuda || true"])
    if result.stdout.strip():
        raise RuntimeError("workstation main_cuda already active:\n" + result.stdout)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_params(version: str, dt: float) -> str:
    mode, features = VERSIONS[version]
    migration = 0 if version == "V0" else 1
    return build_params(DEVELOPMENT_GATE, dt, diagnostics=True) + f"""

# LOW_MEMORY_TRANSPORT_V1 solver-development diagnostic. G10 is not a
# production-qualified transport gate because V2 holdout failed.
ctot_transport_efficiency_mode={mode}
ctot_transport_efficiency_features={features}
ctot_transport_efficiency_restart_migration_allowed={migration}
ctot_transport_gate_trajectory_diagnostics=0
ctot_transport_defect_v2_diagnostics=0
ctot_performance_profile_enabled=1
ctot_automatic_dt_growth=0
ctot_step_max_retries=0
"""


def run_case(host: str, version: str, dt_id: str, steps: int) -> dict[str, object]:
    ensure_idle(host)
    dt, _ = DT[dt_id]
    case_id = f"{version}_{dt_id}_{steps}steps"
    local = OUT / case_id
    local.mkdir(parents=True, exist_ok=True)
    params = local / "runtime.params"
    params.write_text(runtime_params(version, dt), encoding="utf-8")
    params_hash = sha256(params)
    remote = f"{REMOTE_RUNS}/{case_id}"
    execute(["ssh", host, f"rm -rf {shlex.quote(remote)} && mkdir -p {shlex.quote(remote)}"])
    execute(["rsync", "-az", str(params), f"{host}:{remote}/runtime.params"])
    command = f"""set -e
repo={shlex.quote(REMOTE_REPO)}
input={shlex.quote(REMOTE_COMMON)}
out={shlex.quote(remote)}
cd "$out"
export CUDA_STO_RESULTS_ROOT="$out/Results"
start=$(date +%s.%N)
set +e
"$repo/main_cuda" 512 1 1 {dt:.17e} {steps} {steps} {steps} 0 \\
  --mode dynamics --pf-param-file "$out/runtime.params" --temperature-C 400 \\
  --init-mode raw_fields --init-phi-raw "$input/phi_init.raw" \\
  --init-xB-raw "$input/xB_init.raw" --init-Ctot-raw "$input/Ctot_init.raw" \\
  --init-meta "$input/init_meta.json" --init-case-tag {shlex.quote(case_id)} \\
  > run.log 2>&1
rc=$?
set -e
end=$(date +%s.%N)
python3 - "$rc" "$start" "$end" > workstation_status.remote.json <<'PY'
import json, sys
rc, start, end = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
print(json.dumps({{"returncode": rc, "wall_seconds": end-start}}, indent=2))
PY
exit 0
"""
    started = time.monotonic()
    execute(["ssh", host, command], check=False)
    client_wall = time.monotonic() - started
    execute([
        "rsync", "-az", "--exclude=Results", f"{host}:{remote}/", str(local) + "/"
    ])
    remote_status = json.loads((local / "workstation_status.remote.json").read_text())
    log = (local / "run.log").read_text(encoding="utf-8", errors="replace")
    retry_matches = list(BOUNDED_RETRY_SUMMARY_RE.finditer(log))
    retry_summary = retry_matches[-1].groupdict() if retry_matches else {}
    mode, features = VERSIONS[version]
    status = {
        "case_id": case_id,
        "version": version,
        "mode": mode,
        "features": features,
        "dt_id": dt_id,
        "dt_code": dt,
        "steps": steps,
        "development_gate": DEVELOPMENT_GATE,
        "production_gate_qualified": False,
        "runtime_params_sha256": params_hash,
        "returncode": int(remote_status["returncode"]),
        "remote_wall_seconds": remote_status["wall_seconds"],
        "client_wall_seconds": client_wall,
        "runtime_selector_seen": f"mode={mode} features={features}" in log,
        "low_memory_summary_rows": log.count("CTOT_LOW_MEMORY_TRANSPORT_SUMMARY"),
        "accepted_rows": log.count("CTOT_MIMETIC_BE_ACCEPT"),
        "plateau_rows": log.count("CTOT_TRANSPORT_PLATEAU_ESCALATION"),
        "macro_steps_reported": int(retry_summary.get("macro_steps", 0)),
        "macro_hard_rejects": int(
            retry_summary.get("macro_hard_rejects", 0)
        ),
        "internal_trial_rejects": int(
            retry_summary.get("internal_trial_rejects", 0)
        ),
        "fallback_macros": int(retry_summary.get("fallback_macros", 0)),
        "homotopy_triggers": log.count("CTOT_INTERNAL_ALGEBRAIC_HOMOTOPY_TRIGGER"),
        "homotopy_level_failures": len(re.findall(
            r"CTOT_INTERNAL_ALGEBRAIC_HOMOTOPY_LEVEL .*?status=FAIL", log
        )),
        "cluster_used": False,
    }
    (local / "workstation_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, sort_keys=True), flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="workstation-tail")
    parser.add_argument("--suite", choices=("smoke", "trajectory"), default="smoke")
    parser.add_argument("--version", action="append", choices=tuple(VERSIONS))
    parser.add_argument("--dt", action="append", choices=tuple(DT))
    args = parser.parse_args()
    versions = args.version or list(VERSIONS)
    dt_ids = args.dt or (["dt16"] if args.suite == "smoke" else list(DT))
    statuses: list[dict[str, object]] = []
    for version in versions:
        for dt_id in dt_ids:
            steps = 20 if args.suite == "smoke" else DT[dt_id][1]
            statuses.append(run_case(args.host, version, dt_id, steps))
    summary = OUT / f"{args.suite}_status.json"
    summary.write_text(json.dumps(statuses, indent=2, sort_keys=True) + "\n")
    return 0 if all(
        row["returncode"] == 0 and row["runtime_selector_seen"]
        for row in statuses
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
