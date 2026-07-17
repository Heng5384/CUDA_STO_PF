#!/usr/bin/env python3
"""Run the conditional low-cost T400 growth-entry restart smoke."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

import numpy as np

from run_transport_residual_gate_v1_workstation import build_params


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "transport_residual_gate_v1"
LOCAL = REPORT / "entry_smoke_workstation_runs"
COMMON = (
    ROOT / "reports" / "bounded_retry_bdf2_v1" / "workstation_runs"
    / "equal_time_common_input"
)
REMOTE_REPO = "/home/zhiheng/PF/CUDA_STO_PF_transport_gate_v1"
REMOTE = f"{REMOTE_REPO}/runs/transport_residual_gate_v1_entry_smoke"
T_REAL_UNIT_S = 41.12958542455477
CONTINUOUS_STEPS = 128
SPLIT_STEPS = 64


def execute(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(
            f"command failed: {' '.join(args)}\n{result.stdout}\n{result.stderr}"
        )
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one(root: Path, pattern: str) -> Path:
    found = list(root.rglob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, got {found}")
    return found[0]


def selected_candidate() -> tuple[str, str, float, float]:
    path = REPORT / "long_window_candidate_summary.csv"
    if not path.is_file():
        raise RuntimeError("long-window analysis must pass before entry smoke")
    with path.open(newline="", encoding="utf-8") as stream:
        passed = [
            row for row in csv.DictReader(stream)
            if row.get("long_window_status") == "PASS"
        ]
    if not passed:
        raise RuntimeError("no fully passed long-window candidate")
    selected = max(
        passed, key=lambda row: float(row["physical_s_per_GPU_hour"])
    )
    return (
        selected["gate_id"], selected["dt_id"],
        float(selected["requested_transport_gate"]),
        float(selected["dt_code"]),
    )


def ensure_idle(host: str) -> None:
    active = execute(["ssh", host, "pgrep -a -x main_cuda || true"]).stdout.strip()
    if active:
        raise RuntimeError("workstation main_cuda already active:\n" + active)


def stage(host: str) -> str:
    remote_common = f"{REMOTE}/common_input"
    execute(["ssh", host, f"mkdir -p {shlex.quote(remote_common)} {shlex.quote(REMOTE)}"])
    execute(["rsync", "-az", str(COMMON) + "/", f"{host}:{remote_common}/"])
    execute([
        "rsync", "-az", str(ROOT / "scripts" / "summarize_transport_gate_long_case.py"),
        f"{host}:{REMOTE_REPO}/scripts/",
    ])
    return remote_common


def pull(host: str, remote: str, local: Path) -> None:
    local.mkdir(parents=True, exist_ok=True)
    execute([
        "rsync", "-az", "--include=*/", "--include=*.raw", "--include=*.json",
        "--include=*.params", "--include=run.key.log", "--exclude=*",
        f"{host}:{remote}/", str(local) + "/",
    ])


def run_segment(
    host: str,
    label: str,
    gate: float,
    dt: float,
    steps: int,
    output_interval: int,
    init: dict[str, str],
) -> dict[str, Any]:
    ensure_idle(host)
    remote = f"{REMOTE}/{label}"
    execute(["ssh", host, f"rm -rf {shlex.quote(remote)} && mkdir -p {shlex.quote(remote)}"])
    generated = LOCAL / f".{label}.runtime.params"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(build_params(gate, dt, diagnostics=True), encoding="utf-8")
    params_hash = sha256(generated)
    execute(["rsync", "-az", str(generated), f"{host}:{remote}/runtime.params"])
    generated.unlink()

    history_args = ""
    if init.get("Ctot_nm1") and init.get("phi_nm1"):
        history_args = (
            f" --init-Ctot-nm1-raw {shlex.quote(init['Ctot_nm1'])}"
            f" --init-phi-nm1-raw {shlex.quote(init['phi_nm1'])}"
        )
    command = f"""set -e
repo={shlex.quote(REMOTE_REPO)}
out={shlex.quote(remote)}
cd "$out"
export CUDA_STO_RESULTS_ROOT="$out/Results"
started=$(date +%s.%N)
set +e
"$repo/main_cuda" 512 1 1 {dt:.17e} {steps} {output_interval} {output_interval} 0 \
  --mode dynamics --pf-param-file "$out/runtime.params" --temperature-C 400 \
  --init-mode raw_fields --init-phi-raw {shlex.quote(init['phi'])} \
  --init-xB-raw {shlex.quote(init['xB'])} \
  --init-Ctot-raw {shlex.quote(init['Ctot'])}{history_args} \
  --init-meta {shlex.quote(init['meta'])} --init-case-tag {shlex.quote(label)} \
  > run.log 2>&1 &
pid=$!
peak=0
while kill -0 "$pid" 2>/dev/null; do
  mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F, -v p="$pid" '$1+0==p {{gsub(/ /,"",$2); print $2; exit}}')
  if [ -n "$mem" ] && [ "$mem" -gt "$peak" ]; then peak=$mem; fi
  sleep 0.2
done
wait "$pid"
rc=$?
set -e
ended=$(date +%s.%N)
python3 - "$rc" "$started" "$ended" "$peak" > workstation_status.remote.json <<'PY'
import json, sys
rc, start, end, peak = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
print(json.dumps({{"returncode": rc, "wall_seconds": end-start, "peak_gpu_memory_MiB": peak}}, indent=2))
PY
grep -E 'CTOT_BOUNDED_RETRY_SUMMARY|CTOT_TRANSPORT_GATE_TRAJECTORY|\\[fatal\\]|\\[reject\\]' run.log > run.key.log || true
python3 "$repo/scripts/summarize_transport_gate_long_case.py" "$out" --output "$out/segment_summary.json" >/dev/null
exit 0
"""
    execute(["ssh", host, command], check=False)
    local = LOCAL / label
    pull(host, remote, local)
    status = json.loads((local / "workstation_status.remote.json").read_text())
    status.update({
        "label": label,
        "steps": steps,
        "dt_code": dt,
        "requested_transport_gate": gate,
        "runtime_params_sha256": params_hash,
        "remote_out": remote,
    })
    (local / "workstation_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return status


def checkpoint_init(remote: str, local: Path, step: int) -> dict[str, str]:
    names = {
        "Ctot": f"*step{step:06d}_Ctot.raw",
        "phi": f"*step{step:06d}_phi.raw",
        "xB": f"*step{step:06d}_xB_alpha.raw",
        "Ctot_nm1": f"*step{step:06d}_Ctot_nm1.raw",
        "phi_nm1": f"*step{step:06d}_phi_nm1.raw",
        "meta": f"*step{step:06d}_meta.json",
    }
    result: dict[str, str] = {}
    for key, pattern in names.items():
        relative = one(local, pattern).relative_to(local)
        result[key] = f"{remote}/{relative}"
    return result


def summary_pass(summary: dict[str, Any], steps: int, dt: float) -> bool:
    bounded = summary.get("bounded_retry_summary", {})
    trajectory = summary.get("trajectory_meta", {})
    budget = int(bounded.get("nonlinear_iteration_budget", 0))
    return bool(
        summary.get("acceptance_hard_pass")
        and summary.get("energy_pass")
        and summary.get("rollback_pass")
        and float(summary.get("max_mass_error", math.inf)) <= 1.0e-10
        and int(bounded.get("macro_hard_rejects", 1)) == 0
        and float(bounded.get("retry_fraction", math.inf)) <= 0.01
        and float(bounded.get("fallback_fraction", math.inf)) <= 0.01
        and float(bounded.get("reject_trial_wall_fraction", math.inf)) <= 0.05
        and int(bounded.get("max_consecutive_fallback_macros", 99)) <= 2
        and int(bounded.get("max_accepted_subcycle_depth", 99)) <= 2
        and not summary.get("persistent_failed_cells", [])
        and budget > 0
        and float(bounded.get("accepted_iteration_p99", math.inf)) < 0.8 * budget
        and trajectory.get("schema")
        == "CTOT_TRANSPORT_GATE_TRAJECTORY_V2_TRANSACTIONAL"
        and abs(float(trajectory.get("observed_time_code", math.nan)) - steps * dt)
        <= 1.0e-12
    )


def main() -> int:
    host = "workstation-tail"
    gate_id, dt_id, gate, dt = selected_candidate()
    remote_common = stage(host)
    initial = {
        "Ctot": f"{remote_common}/Ctot_init.raw",
        "phi": f"{remote_common}/phi_init.raw",
        "xB": f"{remote_common}/xB_init.raw",
        "meta": f"{remote_common}/init_meta.json",
    }
    prefix = f"ENTRY_{gate_id}_{dt_id}_prefix64"
    continuous = f"ENTRY_{gate_id}_{dt_id}_continuous128"
    suffix = f"ENTRY_{gate_id}_{dt_id}_suffix64"

    continuous_status = run_segment(
        host, continuous, gate, dt, CONTINUOUS_STEPS, SPLIT_STEPS, initial
    )
    prefix_status = run_segment(
        host, prefix, gate, dt, SPLIT_STEPS, SPLIT_STEPS, initial
    )
    prefix_remote = f"{REMOTE}/{prefix}"
    restart = checkpoint_init(prefix_remote, LOCAL / prefix, SPLIT_STEPS)
    suffix_status = run_segment(
        host, suffix, gate, dt, SPLIT_STEPS, SPLIT_STEPS, restart
    )

    continuous_root = LOCAL / continuous
    prefix_root = LOCAL / prefix
    suffix_root = LOCAL / suffix
    fields = {
        "Ctot": (f"*step{CONTINUOUS_STEPS:06d}_Ctot.raw", f"*step{SPLIT_STEPS:06d}_Ctot.raw"),
        "phi": (f"*step{CONTINUOUS_STEPS:06d}_phi.raw", f"*step{SPLIT_STEPS:06d}_phi.raw"),
        "xB_alpha": (f"*step{CONTINUOUS_STEPS:06d}_xB_alpha.raw", f"*step{SPLIT_STEPS:06d}_xB_alpha.raw"),
        "Ctot_nm1": (f"*step{CONTINUOUS_STEPS:06d}_Ctot_nm1.raw", f"*step{SPLIT_STEPS:06d}_Ctot_nm1.raw"),
        "phi_nm1": (f"*step{CONTINUOUS_STEPS:06d}_phi_nm1.raw", f"*step{SPLIT_STEPS:06d}_phi_nm1.raw"),
    }
    hashes: dict[str, dict[str, Any]] = {}
    for key, (continuous_pattern, suffix_pattern) in fields.items():
        a = one(continuous_root, continuous_pattern)
        b = one(suffix_root, suffix_pattern)
        hashes[key] = {
            "continuous_sha256": sha256(a),
            "restart_sha256": sha256(b),
            "bitwise_equal": a.read_bytes() == b.read_bytes(),
        }

    continuous_defect = np.fromfile(
        one(continuous_root, "ctot_transport_gate_cumulative_defect.raw"), np.float64
    )
    split_defect = (
        np.fromfile(one(prefix_root, "ctot_transport_gate_cumulative_defect.raw"), np.float64)
        + np.fromfile(one(suffix_root, "ctot_transport_gate_cumulative_defect.raw"), np.float64)
    )
    defect_linf = float(np.max(np.abs(continuous_defect - split_defect)))
    summaries = [
        json.loads((root / "segment_summary.json").read_text(encoding="utf-8"))
        for root in (continuous_root, prefix_root, suffix_root)
    ]
    hard_pass = all(
        summary_pass(summary, steps, dt)
        for summary, steps in zip(
            summaries, (CONTINUOUS_STEPS, SPLIT_STEPS, SPLIT_STEPS)
        )
    )
    post_gpu_processes = execute([
        "ssh", host,
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true",
    ]).stdout.strip()
    bitwise = all(item["bitwise_equal"] for item in hashes.values())
    status = "PASS_OPTIONAL_8NM_ENTRY_SMOKE" if (
        hard_pass and bitwise and defect_linf <= 1.0e-24
        and not post_gpu_processes
        and all(item["returncode"] == 0 for item in (
            continuous_status, prefix_status, suffix_status
        ))
    ) else "FAIL_OPTIONAL_8NM_ENTRY_SMOKE"
    result = {
        "schema": "CTOT_TRANSPORT_GATE_OPTIONAL_ENTRY_SMOKE_V1",
        "status": status,
        "gate_id": gate_id,
        "dt_id": dt_id,
        "transport_gate": gate,
        "dt_code": dt,
        "physical_time_s": CONTINUOUS_STEPS * dt * T_REAL_UNIT_S,
        "continuous_steps": CONTINUOUS_STEPS,
        "restart_split": [SPLIT_STEPS, SPLIT_STEPS],
        "all_segment_hard_gates_pass": hard_pass,
        "all_state_and_history_fields_bitwise_equal": bitwise,
        "field_hashes": hashes,
        "cumulative_defect_split_difference_Linf": defect_linf,
        "continuous_peak_gpu_memory_MiB": continuous_status["peak_gpu_memory_MiB"],
        "prefix_peak_gpu_memory_MiB": prefix_status["peak_gpu_memory_MiB"],
        "suffix_peak_gpu_memory_MiB": suffix_status["peak_gpu_memory_MiB"],
        "post_smoke_gpu_processes": post_gpu_processes,
        "continuous_wall_seconds": continuous_status["wall_seconds"],
        "continuous_physical_s_per_GPU_hour": (
            CONTINUOUS_STEPS * dt * T_REAL_UNIT_S
            / continuous_status["wall_seconds"] * 3600.0
        ),
        "formal_8nm_campaign_run": False,
    }
    (REPORT / "optional_8nm_entry_smoke.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Optional T400 8 nm entry smoke", "",
        f"`status={status}`", "",
        "This is a low-cost entry and restart test on the frozen T400 growth initial state; it is not a formal 8 nm displacement campaign.", "",
        f"- Candidate: `{gate_id}+{dt_id}`, gate `{gate:.3e}`, dt `{dt:.17e}`.",
        f"- Continuous vs 64+64 restart state/history bitwise equality: `{bitwise}`.",
        f"- Cumulative defect split difference Linf: `{defect_linf:.17e}`.",
        f"- All unchanged hard/retry gates: `{hard_pass}`.",
        f"- Peak GPU allocation: `{continuous_status['peak_gpu_memory_MiB']} MiB`.",
        f"- Entry throughput: `{result['continuous_physical_s_per_GPU_hour']:.3f}` physical s/GPU h.",
        "- Formal 8 nm campaign: `NOT_RUN`.", "",
    ]
    (REPORT / "optional_8nm_entry_smoke.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if status.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
