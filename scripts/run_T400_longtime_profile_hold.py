#!/usr/bin/env python3
"""Run and audit zero-driving profile holds on the workstation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "T400_longtime_v1"
HOLD_ROOT = REPORT_ROOT / "profile_hold"
BASELINE = REPORT_ROOT / "baseline_manifest.json"
MANIFEST = ROOT / "examples" / "T400_longtime_v1_manifest.json"


def execute(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def remote_idle(host: str) -> None:
    command = (
        "p=$(pgrep -af main_cuda | grep -v 'pgrep -af' || true); "
        "g=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader "
        "2>/dev/null || true); printf 'P\\n%s\\nG\\n%s\\n' \"$p\" \"$g\""
    )
    probe = execute(["ssh", host, command])
    if probe.returncode != 0:
        raise RuntimeError(probe.stderr)
    before, _, after = probe.stdout.partition("G\n")
    processes = before.replace("P\n", "").strip()
    if processes or after.strip():
        raise RuntimeError("workstation is not idle; refusing to displace a task")


def parse_params(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


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
    for key, value in replacements.items():
        if key not in seen:
            output.append(f"{key}={value}")
    return "\n".join(output) + "\n"


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def alpha(phi: np.ndarray) -> np.ndarray:
    value = 1.0 - phi
    return value**3 * (6.0 * value**2 - 15.0 * value + 10.0)


def crossings(phi: np.ndarray, dx: float) -> list[float]:
    values: list[float] = []
    n = phi.size
    for i in range(n):
        j = (i + 1) % n
        a = float(phi[i] - 0.5)
        b = float(phi[j] - 0.5)
        if a == 0.0:
            values.append((i + 0.5) * dx)
        elif a * b < 0.0:
            fraction = -a / (b - a)
            values.append(((i + 0.5) + fraction) * dx % (n * dx))
    return sorted(values)


def slab_half_width(phi: np.ndarray, dx: float) -> float:
    return 0.5 * float(np.sum(h(phi))) * dx


def crossing_half_width(phi: np.ndarray, dx: float) -> float:
    points = crossings(phi, dx)
    if len(points) != 2:
        raise RuntimeError(f"expected two crossings, found {points}")
    span = points[1] - points[0]
    span = min(span, phi.size * dx - span)
    return 0.5 * span


def only(root: Path, pattern: str) -> Path:
    values = list(root.rglob(pattern))
    if len(values) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, found {values}")
    return values[0]


def max_marker(text: str, pattern: str) -> float:
    values = [float(value) for value in re.findall(pattern, text)]
    return max(values, default=math.nan)


def prepare_cases(host: str, remote_repo: str, steps: int) -> dict[str, Any]:
    baseline = json.loads(BASELINE.read_text())
    manifest = json.loads(MANIFEST.read_text())
    params_command = (
        "cd " + shlex.quote(remote_repo) + " && "
        "find runs/bdf2_v1/fixed_dt_qualification/dt_div8/run "
        "-name pf_input.params -print -quit | xargs cat"
    )
    fetched = execute(["ssh", host, params_command])
    if fetched.returncode != 0:
        raise RuntimeError("could not fetch qualified parameter file")
    base_params = fetched.stdout
    base_values = parse_params(base_params)
    if base_values.get("PF_RESEARCH_MODEL") != "pbte_ag2te_gp_coarse4_stoich_rd_v2":
        raise RuntimeError("qualified parameter model mismatch")
    if base_values.get("ctot_numerics_contract") != "ctot_jichen_imex_bdf2_v1":
        raise RuntimeError("qualified parameter integrator mismatch")

    import sys
    sys.path.insert(0, str(ROOT))
    import Unit_Psedobinary as unit  # noqa: E402

    x_eq = float(unit.xAg2Te_eq_from_T(673.15))
    selected = [
        row for row in manifest["states"]
        if row["Nx"] == 512 and row["direction"] == "growth"
    ]
    cases: list[dict[str, Any]] = []
    HOLD_ROOT.mkdir(parents=True, exist_ok=True)
    (HOLD_ROOT / "qualified_pf_input.params").write_text(base_params)
    for state in selected:
        case_id = f"T400_stationary_hold_shift{str(state['interface_center_shift_dx']).replace('.', 'p')}"
        case = HOLD_ROOT / "cases" / case_id
        case.mkdir(parents=True, exist_ok=True)
        source = REPORT_ROOT / "initial_states" / state["case_id"]
        phi = np.fromfile(source / "phi_init.raw", dtype=np.float64)
        x = np.full_like(phi, x_eq)
        C = h(phi) + alpha(phi) * x
        phi.tofile(case / "phi_init.raw")
        x.tofile(case / "xB_init.raw")
        C.tofile(case / "Ctot_init.raw")

        meta = dict(baseline["qualified_checkpoint_meta"])
        meta.update({
            "schema": "ctot_checkpoint_v1",
            "Nx": 512, "Ny": 1, "Nz": 1,
            "step": 0, "time_code": 0.0,
            "bdf2_history_valid": 0,
            "bdf2_restart_fallback_pending": 0,
            "bdf2_accepted_step": 0,
            "bdf2_time_code": 0.0,
            "bdf2_physical_time_s": 0.0,
            "bdf2_last_fallback_reason": "startup_history_unavailable",
            "reference_type": "T400_zero_driving_profile_hold",
        })
        meta.pop("bdf2_Ctot_nm1_file", None)
        meta.pop("bdf2_phi_nm1_file", None)
        (case / "init_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

        replacements = {
            "dt": f"{float(manifest['selected_dt_code']):.17e}",
            "init_case_tag": case_id,
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
        (case / "runtime.params").write_text(rewrite_params(base_params, replacements))
        case_manifest = {
            "case_id": case_id,
            "shift_dx": state["interface_center_shift_dx"],
            "grid": [512, 1, 1],
            "steps": steps,
            "dt_code": manifest["selected_dt_code"],
            "xB_eq": x_eq,
            "initial_half_width_h_nm": slab_half_width(phi, 1.0),
            "initial_half_width_crossing_nm": crossing_half_width(phi, 1.0),
            "source_state": state["case_id"],
        }
        (case / "case_manifest.json").write_text(json.dumps(case_manifest, indent=2) + "\n")
        cases.append(case_manifest)
    result = {
        "schema": "T400_profile_hold_matrix_v1",
        "cases": cases,
        "baseline_binary_sha256": baseline["binary_sha256"],
        "cluster_used": False,
    }
    (HOLD_ROOT / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_cases(host: str, remote_repo: str, manifest: dict[str, Any]) -> None:
    remote_idle(host)
    remote_root = f"{remote_repo}/runs/T400_longtime_v1/profile_hold"
    mkdir = execute(["ssh", host, "mkdir -p " + shlex.quote(remote_root)])
    if mkdir.returncode != 0:
        raise RuntimeError(mkdir.stderr)
    sync = execute(["rsync", "-az", str(HOLD_ROOT) + "/", f"{host}:{remote_root}/"])
    if sync.returncode != 0:
        raise RuntimeError(sync.stderr)
    for row in manifest["cases"]:
        case_id = row["case_id"]
        case = HOLD_ROOT / "cases" / case_id
        remote_case = f"{remote_root}/cases/{case_id}"
        steps = int(row["steps"])
        dt = float(row["dt_code"])
        command = (
            "set -e; repo=" + shlex.quote(remote_repo) + "; "
            "c=" + shlex.quote(remote_case) + "; rm -rf \"$c/run\"; "
            "mkdir -p \"$c/run\"; cd \"$c/run\"; "
            "export CUDA_STO_RESULTS_ROOT=\"$c/run/Results\"; "
            f"\"$repo/main_cuda\" 512 1 1 {dt:.17e} {steps} {steps} {steps} 0 "
            "--mode dynamics --pf-param-file \"$c/runtime.params\" "
            "--temperature-C 400 --init-mode raw_fields "
            "--init-phi-raw \"$c/phi_init.raw\" "
            "--init-xB-raw \"$c/xB_init.raw\" "
            "--init-Ctot-raw \"$c/Ctot_init.raw\" "
            "--init-meta \"$c/init_meta.json\" "
            f"--init-case-tag {shlex.quote(case_id)} > run.log 2>&1"
        )
        started = time.monotonic()
        completed = execute(["ssh", host, command])
        wall = time.monotonic() - started
        downloaded = execute([
            "rsync", "-az", f"{host}:{remote_case}/run/", str(case / "run") + "/",
        ])
        status = {
            "returncode": completed.returncode,
            "download_returncode": downloaded.returncode,
            "wall_seconds": wall,
        }
        (case / "workstation_status.json").write_text(json.dumps(status, indent=2) + "\n")
        if completed.returncode != 0 or downloaded.returncode != 0:
            raise RuntimeError(f"profile hold failed: {case_id}: {status}")


def analyze(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_manifest in manifest["cases"]:
        case = HOLD_ROOT / "cases" / case_manifest["case_id"]
        log = (case / "run" / "run.log").read_text(errors="replace")
        step = int(case_manifest["steps"])
        phi0 = np.fromfile(case / "phi_init.raw", np.float64)
        C0 = np.fromfile(case / "Ctot_init.raw", np.float64)
        phi1 = np.fromfile(only(case / "run", f"ctot_checkpoint_step{step:06d}_phi.raw"), np.float64)
        C1 = np.fromfile(only(case / "run", f"ctot_checkpoint_step{step:06d}_Ctot.raw"), np.float64)
        x1 = np.fromfile(only(case / "run", f"ctot_checkpoint_step{step:06d}_xB_alpha.raw"), np.float64)
        accepts = log.count("CTOT_MIMETIC_BE_ACCEPT")
        rejects = log.count("CTOT_MIMETIC_BE_REJECT")
        retries = log.count("CTOT_COUPLED_STEP_RETRY")
        h0 = slab_half_width(phi0, 1.0)
        h1 = slab_half_width(phi1, 1.0)
        c0 = crossing_half_width(phi0, 1.0)
        c1 = crossing_half_width(phi1, 1.0)
        profile_l2 = float(np.sqrt(np.mean((phi1 - phi0) ** 2)))
        profile_linf = float(np.max(np.abs(phi1 - phi0)))
        mass_error = abs(float(np.sum(C1) - np.sum(C0))) / max(abs(float(np.sum(C0))), 1.0)
        row: dict[str, Any] = {
            "case_id": case_manifest["case_id"],
            "shift_dx": case_manifest["shift_dx"],
            "steps": step,
            "accepted_steps": accepts,
            "reject_count": rejects,
            "retry_count": retries,
            "initial_half_width_h_nm": h0,
            "final_half_width_h_nm": h1,
            "h_half_width_drift_nm": h1 - h0,
            "initial_half_width_crossing_nm": c0,
            "final_half_width_crossing_nm": c1,
            "crossing_half_width_drift_nm": c1 - c0,
            "phi_profile_L2_change": profile_l2,
            "phi_profile_Linf_change": profile_linf,
            "Ctot_mass_error_rel_recomputed": mass_error,
            "xB_min": float(np.min(x1)),
            "xB_max": float(np.max(x1)),
            "max_runtime_mass_error": max_marker(log, r"mass_error=([0-9.eE+-]+)"),
            "max_phase_KKT": max_marker(log, r"phase_KKT=([0-9.eE+-]+)"),
            "energy_work_fail_rows": len(re.findall(r"CTOT_IMEX_BDF2_ENERGY_WORK .*?pass=0", log)),
            "clipping_count": log.count("CLIPPING_APPLIED"),
            "physical_projection_count": log.count("PHYSICAL_PROJECTION_APPLIED"),
            "nan_or_inf": any(
                int(value) != 0
                for value in re.findall(r"\bnonfinite=(\d+)", log)
            ),
        }
        row["status"] = "PASS" if (
            accepts == step and rejects == 0 and retries == 0
            and abs(float(row["h_half_width_drift_nm"])) <= 0.01
            and abs(float(row["crossing_half_width_drift_nm"])) <= 0.01
            and profile_l2 <= 1.0e-3 and profile_linf <= 1.0e-2
            and mass_error <= 1.0e-10
            and row["energy_work_fail_rows"] == 0
            and row["clipping_count"] == 0
            and row["physical_projection_count"] == 0
            and not row["nan_or_inf"]
        ) else "FAIL"
        rows.append(row)
    fields = list(rows[0])
    with (HOLD_ROOT / "profile_hold_validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    status = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    table = "\n".join(
        f"| {row['case_id']} | {row['h_half_width_drift_nm']:.3e} | "
        f"{row['crossing_half_width_drift_nm']:.3e} | "
        f"{row['phi_profile_L2_change']:.3e} | "
        f"{row['Ctot_mass_error_rel_recomputed']:.3e} | {row['status']} |"
        for row in rows
    )
    (HOLD_ROOT / "profile_hold_validation.md").write_text(f"""# T400 Production-BDF2 Zero-Driving Profile Hold

The analytic planar profile was held at the T400 matrix equilibrium
composition for 1000 fixed BDF2 steps.  Automatic retry, clipping, physical
projection, GP/S3/source, finite-interface correction, and elasticity were off.

| case | h-half-width drift (nm) | crossing drift (nm) | phi L2 change | mass error | status |
|---|---:|---:|---:|---:|---|
{table}

Acceptance limits were 0.01 nm for each interface-position measure, `1e-3`
for profile L2, `1e-2` for profile Linf, and `1e-10` for source-free mass.

`profile_hold_status={status}`
""")
    if status != "PASS":
        raise RuntimeError(f"profile hold failed: {rows}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="workstation-tail")
    parser.add_argument(
        "--remote-repo",
        default="/home/zhiheng/PF/CUDA_STO_PF_lie_be_v2_20260716",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    manifest = prepare_cases(args.host, args.remote_repo, args.steps)
    if args.analyze_only:
        rows = analyze(manifest)
        print(f"profile_hold_cases={len(rows)}")
        print("profile_hold_status=PASS")
    elif not args.prepare_only:
        run_cases(args.host, args.remote_repo, manifest)
        rows = analyze(manifest)
        print(f"profile_hold_cases={len(rows)}")
        print("profile_hold_status=PASS")
    else:
        print(f"profile_hold_cases_prepared={len(manifest['cases'])}")
    print("cluster_used=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
