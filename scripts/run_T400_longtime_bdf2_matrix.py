#!/usr/bin/env python3
"""Prepare, run, and audit fixed-step T400 coarse4 BDF2 planar cases.

This runner is intentionally workstation-only.  It preserves the qualified
physics/numerics parameter file and varies only the preregistered grid, initial
state, fixed time step, and run horizon.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "T400_longtime_v1"
STATE_ROOT = REPORT_ROOT / "initial_states"
BASELINE = REPORT_ROOT / "baseline_manifest.json"
TEST_MANIFEST = ROOT / "examples" / "T400_longtime_v1_manifest.json"


def execute(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def remote_idle(host: str) -> None:
    probe = execute([
        "ssh", host,
        "p=$(pgrep -af main_cuda | grep -v 'pgrep -af' || true); "
        "g=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader "
        "2>/dev/null || true); printf 'P\\n%s\\nG\\n%s\\n' \"$p\" \"$g\"",
    ])
    if probe.returncode != 0:
        raise RuntimeError(probe.stderr)
    before, _, after = probe.stdout.partition("G\n")
    if before.replace("P\n", "").strip() or after.strip():
        raise RuntimeError("workstation is not idle; refusing to displace a task")


def parse_params(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def rewrite_params(source: str, replacements: dict[str, str]) -> str:
    output: list[str] = []
    seen: set[str] = set()
    for raw in source.splitlines():
        if "=" in raw and not raw.lstrip().startswith("#"):
            key = raw.split("=", 1)[0].strip()
            if key in replacements:
                output.append(f"{key}={replacements[key]}")
                seen.add(key)
                continue
        output.append(raw)
    output.extend(f"{key}={value}" for key, value in replacements.items() if key not in seen)
    return "\n".join(output) + "\n"


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def crossings(phi: np.ndarray, dx: float) -> list[float]:
    result: list[float] = []
    n = phi.size
    for i in range(n):
        j = (i + 1) % n
        left = float(phi[i] - 0.5)
        right = float(phi[j] - 0.5)
        if left == 0.0:
            result.append((i + 0.5) * dx)
        elif left * right < 0.0:
            fraction = -left / (right - left)
            result.append(((i + 0.5) + fraction) * dx % (n * dx))
    return sorted(result)


def crossing_half_width(phi: np.ndarray, dx: float) -> float:
    points = crossings(phi, dx)
    if len(points) != 2:
        return math.nan
    span = min(points[1] - points[0], phi.size * dx - (points[1] - points[0]))
    return 0.5 * span


def one(root: Path, pattern: str) -> Path:
    paths = list(root.rglob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, found {paths}")
    return paths[0]


def checkpoint_steps(run_root: Path) -> list[int]:
    values: set[int] = set()
    for path in run_root.rglob("ctot_checkpoint_step*_phi.raw"):
        match = re.search(r"step(\d+)_phi\.raw$", path.name)
        if match:
            values.add(int(match.group(1)))
    return sorted(values)


def prepare_case(
    host: str,
    remote_repo: str,
    run_id: str,
    state_id: str,
    dt_code: float,
    steps: int,
    checkpoint_every: int,
) -> tuple[Path, dict[str, Any]]:
    baseline = json.loads(BASELINE.read_text())
    test = json.loads(TEST_MANIFEST.read_text())
    state_row = next(row for row in test["states"] if row["case_id"] == state_id)
    source = STATE_ROOT / state_id
    case = REPORT_ROOT / "coarse4_runs" / run_id
    case.mkdir(parents=True, exist_ok=True)
    for name in ("phi_init.raw", "xB_init.raw", "Ctot_init.raw"):
        shutil.copy2(source / name, case / name)

    fetch = execute([
        "ssh", host,
        "cd " + shlex.quote(remote_repo) + " && find "
        "runs/bdf2_v1/fixed_dt_qualification/dt_div8/run "
        "-name pf_input.params -print -quit | xargs cat",
    ])
    if fetch.returncode != 0:
        raise RuntimeError("could not fetch qualified parameter file")
    params = fetch.stdout
    values = parse_params(params)
    if values.get("PF_RESEARCH_MODEL") != "pbte_ag2te_gp_coarse4_stoich_rd_v2":
        raise RuntimeError("qualified model mismatch")
    if values.get("ctot_numerics_contract") != "ctot_jichen_imex_bdf2_v1":
        raise RuntimeError("qualified BDF2 contract mismatch")
    replacements = {
        "dt": f"{dt_code:.17e}",
        "init_case_tag": run_id,
        "ctot_numerics_contract": "ctot_jichen_imex_bdf2_v1",
        "ctot_split_defect_policy": "IMEX_BDF2_NO_POST_PHASE_POLISH",
        "ctot_max_coupling_correctors": "0",
        "ctot_outer_acceleration": "OFF",
        "ctot_automatic_dt_growth": "0",
        "ctot_step_max_retries": "0",
        "elastic_enabled": "0",
        "diagnostic_rsmd_enabled": "0",
        "enable_gp_assisted_beta_nucleation": "0",
        "gp_growth_enabled": "0",
        "ctot_finite_interface_antitrapping_enabled": "0",
        "coarse_interface_mobility_a_M": "0.00000000000000000e+00",
        "y_update_mass_projection_enabled": "0",
    }
    (case / "runtime.params").write_text(rewrite_params(params, replacements))

    meta = dict(baseline["qualified_checkpoint_meta"])
    meta.update({
        "schema": "ctot_checkpoint_v1",
        "Nx": int(state_row["Nx"]), "Ny": 1, "Nz": 1,
        "dx_nm": float(state_row["dx_nm"]),
        "interface_width_nm": float(state_row["lambda_nm"]),
        "step": 0, "time_code": 0.0,
        "bdf2_history_valid": 0,
        "bdf2_restart_fallback_pending": 0,
        "bdf2_accepted_step": 0,
        "bdf2_time_code": 0.0,
        "bdf2_physical_time_s": 0.0,
        "bdf2_last_fallback_reason": "startup_history_unavailable",
        "reference_type": "T400_longtime_coarse4_BDF2",
    })
    meta.pop("bdf2_Ctot_nm1_file", None)
    meta.pop("bdf2_phi_nm1_file", None)
    (case / "init_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    manifest = {
        "schema": "T400_longtime_coarse4_case_v1",
        "run_id": run_id,
        "state_id": state_id,
        "direction": state_row["direction"],
        "grid": [int(state_row["Nx"]), 1, 1],
        "dx_nm": float(state_row["dx_nm"]),
        "lambda_nm": float(state_row["lambda_nm"]),
        "matrix_xB": float(state_row["matrix_xB"]),
        "shift_dx": float(state_row["interface_center_shift_dx"]),
        "dt_code": dt_code,
        "dt_physical_s": dt_code * (
            float(test["selected_dt_physical_s"])
            / float(test["selected_dt_code"])
        ),
        "steps": steps,
        "checkpoint_every": checkpoint_every,
        "initial_h_half_width_nm": 0.5 * float(np.sum(h(np.fromfile(case / "phi_init.raw", np.float64)))) * float(state_row["dx_nm"]),
        "cluster_used": False,
        "retry_allowed": False,
        "source_binary_sha256": baseline["binary_sha256"],
    }
    (case / "case_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return case, manifest


def run_case(host: str, remote_repo: str, case: Path, manifest: dict[str, Any]) -> None:
    remote_idle(host)
    remote_case = f"{remote_repo}/runs/T400_longtime_v1/coarse4_runs/{manifest['run_id']}"
    mkdir = execute(["ssh", host, "mkdir -p " + shlex.quote(remote_case)])
    if mkdir.returncode != 0:
        raise RuntimeError(mkdir.stderr)
    sync = execute(["rsync", "-az", str(case) + "/", f"{host}:{remote_case}/"])
    if sync.returncode != 0:
        raise RuntimeError(sync.stderr)
    nx = int(manifest["grid"][0])
    dt = float(manifest["dt_code"])
    steps = int(manifest["steps"])
    every = int(manifest["checkpoint_every"])
    command = (
        "set -e; repo=" + shlex.quote(remote_repo) + "; c=" + shlex.quote(remote_case)
        + "; rm -rf \"$c/run\"; mkdir -p \"$c/run\"; cd \"$c/run\"; "
        + "export CUDA_STO_RESULTS_ROOT=\"$c/run/Results\"; "
        + f"\"$repo/main_cuda\" {nx} 1 1 {dt:.17e} {steps} {every} {every} 0 "
        + "--mode dynamics --pf-param-file \"$c/runtime.params\" --temperature-C 400 "
        + "--init-mode raw_fields --init-phi-raw \"$c/phi_init.raw\" "
        + "--init-xB-raw \"$c/xB_init.raw\" --init-Ctot-raw \"$c/Ctot_init.raw\" "
        + "--init-meta \"$c/init_meta.json\" --init-case-tag "
        + shlex.quote(str(manifest["run_id"])) + " > run.log 2>&1"
    )
    start = time.monotonic()
    completed = execute(["ssh", host, command])
    wall = time.monotonic() - start
    downloaded = execute(["rsync", "-az", f"{host}:{remote_case}/run/", str(case / "run") + "/"])
    status = {"returncode": completed.returncode, "download_returncode": downloaded.returncode, "wall_seconds": wall}
    (case / "workstation_status.json").write_text(json.dumps(status, indent=2) + "\n")
    if completed.returncode != 0 or downloaded.returncode != 0:
        raise RuntimeError(f"runtime failed: {status}")


def analyze_case(case: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    run = case / "run"
    log = (run / "run.log").read_text(errors="replace")
    phi0 = np.fromfile(case / "phi_init.raw", np.float64)
    C0 = np.fromfile(case / "Ctot_init.raw", np.float64)
    initial_h = 0.5 * float(np.sum(h(phi0))) * float(manifest["dx_nm"])
    initial_cross = crossing_half_width(phi0, float(manifest["dx_nm"]))
    rows: list[dict[str, Any]] = []
    for step in checkpoint_steps(run):
        phi = np.fromfile(one(run, f"ctot_checkpoint_step{step:06d}_phi.raw"), np.float64)
        C = np.fromfile(one(run, f"ctot_checkpoint_step{step:06d}_Ctot.raw"), np.float64)
        x = np.fromfile(one(run, f"ctot_checkpoint_step{step:06d}_xB_alpha.raw"), np.float64)
        hh = 0.5 * float(np.sum(h(phi))) * float(manifest["dx_nm"])
        cross = crossing_half_width(phi, float(manifest["dx_nm"]))
        rows.append({
            "run_id": manifest["run_id"], "direction": manifest["direction"],
            "step": step, "time_code": step * float(manifest["dt_code"]),
            "time_s": step * float(manifest["dt_physical_s"]),
            "h_half_width_nm": hh, "h_displacement_nm": hh - initial_h,
            "crossing_half_width_nm": cross,
            "crossing_displacement_nm": cross - initial_cross,
            "total_C_integral": float(np.sum(C)) * float(manifest["dx_nm"]),
            "mass_error_rel": abs(float(np.sum(C) - np.sum(C0))) / max(abs(float(np.sum(C0))), 1.0),
            "matrix_xB_mean": float(np.mean(x)), "matrix_xB_min": float(np.min(x)),
            "matrix_xB_max": float(np.max(x)), "phi_min": float(np.min(phi)),
            "phi_max": float(np.max(phi)),
        })
    accepts = log.count("CTOT_MIMETIC_BE_ACCEPT")
    rejects = log.count("CTOT_MIMETIC_BE_REJECT")
    retries = log.count("CTOT_COUPLED_STEP_RETRY")
    nonfinite = any(int(value) for value in re.findall(r"\bnonfinite=(\d+)", log))
    energy_fail = len(re.findall(r"CTOT_IMEX_BDF2_ENERGY_WORK .*?pass=0", log))
    clipping = log.count("CLIPPING_APPLIED")
    projection = log.count("PHYSICAL_PROJECTION_APPLIED")
    final = rows[-1] if rows else {}
    summary = {
        **manifest,
        "accepted_steps": accepts, "reject_count": rejects, "retry_count": retries,
        "checkpoint_rows": len(rows), "final_h_displacement_nm": final.get("h_displacement_nm"),
        "final_crossing_displacement_nm": final.get("crossing_displacement_nm"),
        "max_mass_error_rel": max((row["mass_error_rel"] for row in rows), default=math.nan),
        "energy_fail_rows": energy_fail, "clipping_count": clipping,
        "physical_projection_count": projection, "nonfinite": nonfinite,
        "hard_gate_status": "PASS" if (
            accepts == int(manifest["steps"]) and rejects == 0 and retries == 0
            and rows and max(row["mass_error_rel"] for row in rows) <= 1.0e-10
            and energy_fail == 0 and clipping == 0 and projection == 0 and not nonfinite
        ) else "FAIL",
    }
    with (case / "coarse4_timeseries.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (case / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"run_id={manifest['run_id']}")
    print(f"hard_gate_status={summary['hard_gate_status']}")
    print(f"h_displacement_nm={summary['final_h_displacement_nm']}")
    print(f"crossing_displacement_nm={summary['final_crossing_displacement_nm']}")
    print(f"max_mass_error_rel={summary['max_mass_error_rel']}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="workstation-tail")
    parser.add_argument("--remote-repo", default="/home/zhiheng/PF/CUDA_STO_PF_lie_be_v2_20260716")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--dt-code", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    case, manifest = prepare_case(
        args.host, args.remote_repo, args.run_id, args.state_id,
        args.dt_code, args.steps, min(args.checkpoint_every, args.steps),
    )
    if args.analyze_only:
        analyze_case(case, manifest)
    elif not args.prepare_only:
        run_case(args.host, args.remote_repo, case, manifest)
        analyze_case(case, manifest)
    else:
        print(f"prepared={case}")
    print("cluster_used=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
