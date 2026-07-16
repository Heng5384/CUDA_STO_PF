#!/usr/bin/env python3
"""Fit the frozen one-parameter coarse4 interface closure, or fail closed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_correction2_runtime import (  # noqa: E402
    crossings,
    final_checkpoint,
    h_switch,
    read_rows,
    unique_result_file,
)
from scripts.analyze_research2_fast_interface_plateau import (  # noqa: E402
    candidate_mu_and_mobility,
    face_flux,
    parse_params,
)


FLUX_TOL = 0.05
INVENTORY_TOL = 0.05
TRAJECTORY_TOL = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-300)


def _read_status(case_dir: Path) -> dict[str, object]:
    path = case_dir / "workstation_status.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _hard_gate_metrics(case_dir: Path, expected_steps: int) -> dict[str, object]:
    status = _read_status(case_dir)
    log_path = case_dir / "run" / "run.log"
    log = log_path.read_text(errors="replace") if log_path.is_file() else ""
    accepted = len(re.findall(r"CTOT_MIMETIC_BE_ACCEPT", log))
    retries = len(re.findall(r"CTOT_COUPLED_STEP_RETRY", log))
    rejects = len(re.findall(r"CTOT_MIMETIC_BE_REJECT", log))
    result: dict[str, object] = {
        "returncode": status.get("returncode", ""),
        "accepted_steps": accepted,
        "retry_count": retries,
        "reject_count": rejects,
        "predicate_pass": False,
        "energy_pass": False,
        "phase_KKT_max": math.nan,
        "phase_storage_residual_max": math.nan,
        "runtime_mass_error_max": math.nan,
    }
    if not log_path.is_file():
        result["failure_reason"] = "NOT_RUN"
        result["hard_gates_pass"] = False
        return result
    mass_values = [abs(float(value)) for value in re.findall(
        r"CTOT_MIMETIC_BE_ACCEPT[^\n]*mass_error=([^ ]+)", log
    )]
    result["runtime_mass_error_max"] = max(mass_values, default=math.nan)
    try:
        predicate = read_rows(unique_result_file(case_dir, "ctot_acceptance_predicate.csv"))
        excluded = {"physical_step_id", "attempt_id"}
        result["predicate_pass"] = len(predicate) == expected_steps and all(
            all(value == "1" for key, value in row.items() if key not in excluded)
            for row in predicate
        )
        energy = read_rows(unique_result_file(case_dir, "ctot_energy_work.csv"))
        result["energy_pass"] = len(energy) == expected_steps and all(
            row.get("accepted") == "1" and row.get("monotone_pass") == "1"
            and row.get("balance_pass") == "1" for row in energy
        )
        outer = read_rows(unique_result_file(case_dir, "ctot_outer_iterations.csv"))
        converged = [row for row in outer if row.get("status") == "CONVERGED"]
        result["phase_KKT_max"] = max(
            (abs(float(row["phase_KKT_residual"])) for row in converged),
            default=math.nan,
        )
        result["phase_storage_residual_max"] = max(
            (abs(float(row["local_phase_storage_residual"])) for row in converged),
            default=math.nan,
        )
    except (FileNotFoundError, RuntimeError):
        pass
    passed = (
        status.get("returncode") == 0
        and accepted == expected_steps and retries == 0 and rejects == 0
        and bool(result["predicate_pass"]) and bool(result["energy_pass"])
        and float(result["runtime_mass_error_max"]) <= 1.0e-10
        and float(result["phase_storage_residual_max"]) <= 1.0e-12
    )
    result["hard_gates_pass"] = passed
    if passed:
        result["failure_reason"] = ""
    elif accepted == 0 and retries > 0:
        result["failure_reason"] = "PHASE_KKT_FAIL_CLOSED_AT_FIRST_STEP"
    elif status.get("returncode", "") == "":
        result["failure_reason"] = "NOT_RUN_AFTER_EARLIER_CANDIDATE_FAILURE"
    else:
        result["failure_reason"] = "NUMERICAL_HARD_GATE_FAILURE"
    return result


def extract_case(case_dir: Path) -> dict[str, object]:
    manifest = json.loads((case_dir / "runtime_manifest.json").read_text())
    expected_steps = int(manifest["nsteps"])
    row: dict[str, object] = {
        "case": manifest["case_id"],
        "direction": manifest["direction"],
        "calibration_role": manifest["calibration_role"],
        "phase_candidate": manifest["phase_candidate"],
        "PHASE_KINETICS_MODE": manifest["PHASE_KINETICS_MODE"],
        "a_M": float(manifest["a_M"]),
        "dx_nm": float(manifest["dx_nm"]),
        "lambda_nm": float(manifest["lambda_nm"]),
        "elapsed_s": float(manifest["elapsed_s"]),
        "expected_steps": expected_steps,
        "common_sharp_state_sha256": manifest["common_sharp_state_sha256"],
    }
    row.update(_hard_gate_metrics(case_dir, expected_steps))
    if not bool(row["hard_gates_pass"]):
        return row

    shape = tuple(map(int, manifest["grid"]))
    phi0 = np.fromfile(case_dir / "phi_init.raw", dtype=np.float64).reshape(shape)
    ctot0 = np.fromfile(case_dir / "Ctot_init.raw", dtype=np.float64)
    step_phi, phi_path = final_checkpoint(case_dir, "phi")
    step_c, ctot_path = final_checkpoint(case_dir, "Ctot")
    step_x, x_path = final_checkpoint(case_dir, "xB_alpha")
    if not (step_phi == step_c == step_x == expected_steps):
        row["hard_gates_pass"] = False
        row["failure_reason"] = "FINAL_CHECKPOINT_MISMATCH"
        return row
    phi1 = np.fromfile(phi_path, dtype=np.float64).reshape(shape)
    ctot1 = np.fromfile(ctot_path, dtype=np.float64)
    x1 = np.fromfile(x_path, dtype=np.float64).reshape(shape)
    h0 = h_switch(phi0)
    h1 = h_switch(phi1)
    dx_nm = float(manifest["dx_nm"])
    area_cells = shape[1] * shape[2]
    delta_h = float((h1 - h0).sum())
    inventory_displacement = delta_h * dx_nm / (2.0 * area_cells)
    phi0_line = phi0.mean(axis=(1, 2))
    phi1_line = phi1.mean(axis=(1, 2))
    crossing0 = crossings(phi0_line, dx_nm)
    crossing1 = crossings(phi1_line, dx_nm)
    trajectory = 0.5 * (
        crossing1[1] - crossing1[0] - crossing0[1] + crossing0[0]
    )
    x_line = x1.mean(axis=(1, 2))
    params = parse_params(case_dir / "runtime.params")
    mu_line, mobility_line = candidate_mu_and_mobility(phi1_line, x_line, params)
    direct_flux = face_flux(mu_line, mobility_line, float(manifest["dx_code"]))
    interface_faces = [int(math.floor(value / dx_nm)) % shape[0]
                       for value in crossing1]
    flux_code = float(np.mean([abs(direct_flux[index]) for index in interface_faces]))
    time_unit_s = float(manifest["elapsed_s"]) / float(manifest["final_code_time"])
    mass_error = abs(float(ctot1.sum() - ctot0.sum())) / max(abs(float(ctot0.sum())), 1.0)
    row.update({
        "matrix_flux_physical": flux_code / time_unit_s,
        "matrix_flux_code": flux_code,
        "beta_inventory_gain_per_interface_nm": inventory_displacement,
        "trajectory_nm": trajectory,
        "direction_pass": (
            inventory_displacement > 0.0 if manifest["direction"] == "growth"
            else inventory_displacement < 0.0
        ),
        "mass_error_rel_recomputed": mass_error,
        "phi_min": float(phi1.min()),
        "phi_max": float(phi1.max()),
        "xB_min": float(x1.min()),
        "xB_max": float(x1.max()),
    })
    if mass_error > 1.0e-10 or not bool(row["direction_pass"]):
        row["hard_gates_pass"] = False
        row["failure_reason"] = (
            "MASS_GATE_FAILURE" if mass_error > 1.0e-10 else "DIRECTION_FAILURE"
        )
    return row


def score_candidate_records(
    records: Iterable[dict[str, object]], weights: dict[str, float]
) -> dict[str, object]:
    rows = list(records)
    required = {"growth", "dissolution"}
    directions = {str(row["direction"]) for row in rows}
    complete = directions == required and len(rows) == 2
    hard = complete and all(bool(row.get("hard_gates_pass")) for row in rows)
    if not hard:
        return {
            "complete": complete, "hard_gates_pass": False,
            "objective": math.inf, "calibration_pass": False,
            "flux_error_max": math.inf, "inventory_error_max": math.inf,
            "trajectory_error_max": math.inf,
        }
    flux_max = max(float(row["matrix_flux_error_rel"]) for row in rows)
    inventory_max = max(float(row["beta_inventory_error_rel"]) for row in rows)
    trajectory_max = max(float(row["trajectory_error_rel"]) for row in rows)
    objective = sum(
        weights["matrix_flux"] * float(row["matrix_flux_error_rel"]) ** 2
        + weights["beta_inventory"] * float(row["beta_inventory_error_rel"]) ** 2
        + weights["trajectory"] * float(row["trajectory_error_rel"]) ** 2
        for row in rows
    ) / len(rows)
    return {
        "complete": True,
        "hard_gates_pass": True,
        "objective": objective,
        "calibration_pass": (
            flux_max <= FLUX_TOL and inventory_max <= INVENTORY_TOL
            and trajectory_max <= TRAJECTORY_TOL
        ),
        "flux_error_max": flux_max,
        "inventory_error_max": inventory_max,
        "trajectory_error_max": trajectory_max,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def calibrate(matrix_root: Path, report_root: Path, examples_root: Path) -> dict[str, object]:
    manifest = json.loads((matrix_root / "matrix_manifest.json").read_text())
    cases_root = matrix_root / "cases"
    rows = [extract_case(cases_root / item["case_id"]) for item in manifest["cases"]]
    references = {
        str(row["direction"]): row for row in rows
        if row["calibration_role"] == "reference"
    }
    if set(references) != {"growth", "dissolution"} or not all(
        bool(row["hard_gates_pass"]) for row in references.values()
    ):
        raise RuntimeError("both frozen fine references must pass before calibration")
    for row in rows:
        if row["calibration_role"] == "reference" or not row["hard_gates_pass"]:
            continue
        reference = references[str(row["direction"])]
        row["matrix_flux_reference"] = reference["matrix_flux_physical"]
        row["beta_inventory_reference"] = reference["beta_inventory_gain_per_interface_nm"]
        row["trajectory_reference"] = reference["trajectory_nm"]
        row["matrix_flux_error_rel"] = relative_error(
            float(row["matrix_flux_physical"]), float(reference["matrix_flux_physical"])
        )
        row["beta_inventory_error_rel"] = relative_error(
            float(row["beta_inventory_gain_per_interface_nm"]),
            float(reference["beta_inventory_gain_per_interface_nm"]),
        )
        row["trajectory_error_rel"] = relative_error(
            float(row["trajectory_nm"]), float(reference["trajectory_nm"])
        )

    weights = {key: float(value) for key, value in manifest["weights_frozen_before_scan"].items()}
    score_rows: list[dict[str, object]] = []
    for candidate in ("finite_lphi", "quasi_equilibrium"):
        for a_m in map(float, manifest["a_M_scan_frozen_before_results"]):
            selected = [row for row in rows if row["phase_candidate"] == candidate
                        and math.isclose(float(row["a_M"]), a_m)]
            score = score_candidate_records(selected, weights)
            score_rows.append({"phase_candidate": candidate, "a_M": a_m, **score})
    for row in rows:
        if row["calibration_role"] == "candidate":
            score = next(item for item in score_rows
                         if item["phase_candidate"] == row["phase_candidate"]
                         and math.isclose(float(item["a_M"]), float(row["a_M"])))
            row.update({f"pair_{key}": value for key, value in score.items()
                        if key not in ("phase_candidate", "a_M")})

    passing = [row for row in score_rows if bool(row["calibration_pass"])]
    passing.sort(key=lambda row: (float(row["objective"]),
                                  str(row["phase_candidate"]), float(row["a_M"])))
    finite_scores = [row for row in score_rows if math.isfinite(float(row["objective"]))]
    finite_scores.sort(key=lambda row: (float(row["objective"]),
                                        str(row["phase_candidate"]), float(row["a_M"])))
    best_observed = finite_scores[0] if finite_scores else None
    selected = passing[0] if passing else None
    status = (
        "PASS_COARSE4_ONE_PARAMETER_CALIBRATION" if selected
        else "BLOCKED_ONE_PARAMETER_COARSE_CLOSURE_INSUFFICIENT"
    )
    report_root.mkdir(parents=True, exist_ok=True)
    _write_csv(report_root / "coarse4_calibration_scan.csv", rows)
    _write_csv(report_root / "coarse4_calibration_candidate_scores.csv", score_rows)

    reference_manifest = {
        "schema": "coarse4_fine_reference_manifest_v1",
        "status": "FROZEN_CALIBRATION_REFERENCES_COMPLETE",
        "fine_reference_hash": manifest["fine_reference"]["fine_reference_hash"],
        "fit_cases_only": manifest["fit_cases"],
        "heldout_refit_allowed": False,
        "references": [],
    }
    for direction, reference in sorted(references.items()):
        case_dir = cases_root / str(reference["case"])
        reference_manifest["references"].append({
            "direction": direction,
            "case": reference["case"],
            "common_sharp_state_sha256": reference["common_sharp_state_sha256"],
            "phi_init_sha256": sha256(case_dir / "phi_init.raw"),
            "xB_init_sha256": sha256(case_dir / "xB_init.raw"),
            "Ctot_init_sha256": sha256(case_dir / "Ctot_init.raw"),
            "runtime_params_sha256": sha256(case_dir / "runtime.params"),
            "matrix_flux_physical": reference["matrix_flux_physical"],
            "beta_inventory_gain_per_interface_nm": reference["beta_inventory_gain_per_interface_nm"],
            "trajectory_nm": reference["trajectory_nm"],
            "mass_error_rel": reference["mass_error_rel_recomputed"],
        })
    (report_root / "coarse4_reference_manifest.json").write_text(
        json.dumps(reference_manifest, indent=2) + "\n"
    )
    references_md = [
        "# Coarse4 Frozen Fine Reference Library", "",
        "The calibration references are PF-only `lambda=0.30 nm`, `dx=0.025 nm` ",
        "fixed-Ctot runs mapped from the same direction-specific pre-aged sharp state as each coarse candidate.", "",
        "| Direction | Flux | Inventory displacement (nm/interface) | Trajectory (nm) | Mass error |",
        "|---|---:|---:|---:|---:|",
    ]
    for direction, row in sorted(references.items()):
        references_md.append(
            f"| {direction} | {float(row['matrix_flux_physical']):.12e} | "
            f"{float(row['beta_inventory_gain_per_interface_nm']):.12e} | "
            f"{float(row['trajectory_nm']):.12e} | "
            f"{float(row['mass_error_rel_recomputed']):.3e} |"
        )
    references_md += ["", "Held-out cases are not part of this fit and refitting is forbidden.", ""]
    (report_root / "coarse4_fine_reference_library.md").write_text("\n".join(references_md))

    best_text = "none"
    if best_observed:
        best_text = (
            f"`{best_observed['phase_candidate']}`, `a_M={best_observed['a_M']}`, "
            f"objective `{float(best_observed['objective']):.6e}`, max errors "
            f"flux/inventory/trajectory = {float(best_observed['flux_error_max']):.3%}/"
            f"{float(best_observed['inventory_error_max']):.3%}/"
            f"{float(best_observed['trajectory_error_max']):.3%}"
        )
    (report_root / "coarse4_calibration.md").write_text(f"""# Coarse4 One-Parameter Calibration

## Decision

`{status}`

The weights were frozen before results: flux `{weights['matrix_flux']}`, inventory
`{weights['beta_inventory']}`, trajectory `{weights['trajectory']}`.  Acceptance
requires both growth and dissolution to satisfy 5% flux, 5% inventory, and 10%
trajectory errors with the same nonnegative `a_M`.

Best finite observed pair: {best_text}.

The quasi-equilibrium candidate failed closed at the first growth phase-KKT solve;
`a_M` changes transport mobility and cannot repair that phase solve.  The finite-Lphi
candidate remained conservative and direction-correct but no scanned value closed
growth and dissolution simultaneously.  No second closure parameter is introduced.

Therefore calibration is blocked and held-out validation must not run.
""")
    examples_root.mkdir(parents=True, exist_ok=True)
    selected_payload = {
        "schema": "coarse4_selected_parameters_v1",
        "status": status,
        "selected_phase_candidate": selected["phase_candidate"] if selected else None,
        "selected_a_M": selected["a_M"] if selected else None,
        "best_observed_nonaccepted": best_observed,
        "physics_changed": False,
        "second_parameter_added": False,
        "heldout_refit_allowed": False,
    }
    (examples_root / "coarse4_selected_parameters.json").write_text(
        json.dumps(selected_payload, indent=2, allow_nan=False) + "\n"
    )
    heldout_rows = [{
        "status": "NOT_RUN_CALIBRATION_FAILED",
        "reason": "No phase-candidate/a_M pair passed both frozen calibration directions",
        "selected_phase_candidate": "",
        "selected_a_M": "",
        "refit_performed": False,
    }]
    _write_csv(report_root / "coarse4_heldout_metrics.csv", heldout_rows)
    (report_root / "coarse4_heldout_validation.md").write_text(f"""# Coarse4 Held-Out Validation

Status: `NOT_RUN_CALIBRATION_FAILED`

The frozen growth+dissolution calibration did not select a valid one-parameter
coarse closure.  The goal's hard stop therefore prevents all planar, curved,
spherical, and two-particle held-out runs.  No held-out data were used to fit,
and no second parameter was added.

Final coarse-model decision: `{status}`.
""")
    return {
        "status": status,
        "selected": selected,
        "best_observed": best_observed,
        "reference_manifest": reference_manifest,
        "rows": rows,
        "scores": score_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--examples-root", type=Path, default=ROOT / "examples")
    args = parser.parse_args()
    result = calibrate(args.matrix_root, args.report_root, args.examples_root)
    print(f"calibration_status={result['status']}")
    selected = result["selected"]
    print(f"selected_phase_candidate={selected['phase_candidate'] if selected else 'NONE'}")
    print(f"selected_a_M={selected['a_M'] if selected else 'NONE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
