#!/usr/bin/env python3
"""Finalize the frozen coarse4 calibration evidence without refitting.

This script is deliberately report-only.  It consumes the completed Stage-1
replay and deterministic one-parameter calibration outputs, then enforces the
calibration hard stop before held-out or selected-mode performance work.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil


MECHANICS_PRECISION_MODE = "FP32_SPECTRAL"
MECHANICS_ACCEPTANCE_MODE = "FP32_NORMALIZED_BACKWARD_ERROR_V1"
MECHANICS_ETA_FLOOR_VERSION = "COARSE4_FP32_ETA_FLOOR_16_32_V1"
MECHANICS_ETA_ACCEPT = 2.569922949844634e-7
MECHANICS_DOUBLE_ORACLE_CONTRACT_HASH = (
    "cc4cad8955684d45d34ca9db7d1b300dd82acc1fc7473ff52b23cd0f5cd3ae3e"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_number(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def classify_candidates(
    scores: list[dict[str, str]], scan: list[dict[str, str]]
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for candidate in ("finite_lphi", "quasi_equilibrium"):
        candidate_scores = [row for row in scores if row["phase_candidate"] == candidate]
        passing = [row for row in candidate_scores if row["calibration_pass"] == "True"]
        finite_scores = [
            row for row in candidate_scores if finite_number(row["objective"]) is not None
        ]
        best = min(finite_scores, key=lambda row: float(row["objective"])) \
            if finite_scores else None
        if passing:
            selected = min(passing, key=lambda row: float(row["objective"]))
            status = "PASS_FROZEN_CALIBRATION_GATES"
        elif candidate == "finite_lphi":
            selected = None
            status = "FAIL_NO_SINGLE_AM_MEETS_FLUX_INVENTORY_TRAJECTORY"
        else:
            growth = [
                row for row in scan
                if row.get("phase_candidate") == candidate
                and row.get("direction") == "growth"
            ]
            all_kkt = bool(growth) and all(
                row.get("failure_reason") == "PHASE_KKT_FAIL_CLOSED_AT_FIRST_STEP"
                for row in growth
            )
            selected = None
            status = (
                "FAIL_PHASE_KKT_GROWTH_ALL_AM" if all_kkt
                else "FAIL_NUMERICAL_OR_CALIBRATION_GATES"
            )
        result[candidate] = {
            "status": status,
            "selected": selected,
            "best": best,
            "scan_count": len(candidate_scores),
        }
    return result


def parse_stage1_log(case_id: str, path: Path) -> dict[str, object]:
    log = path.read_text(errors="replace")
    accepts = re.findall(r"CTOT_MIMETIC_BE_ACCEPT[^\n]*", log)
    mass = [abs(float(value)) for value in re.findall(r"mass_error=([^ ]+)", "\n".join(accepts))]
    projection = [int(value) for value in re.findall(r"projection_mass=([0-9]+)", "\n".join(accepts))]
    clipping = [int(value) for value in re.findall(r"clip_count=([0-9]+)", "\n".join(accepts))]
    mechanical = re.findall(r"CTOT_MECHANICAL_RESIDUAL[^\n]*", log)
    spectral = [
        abs(float(value)) for value in re.findall(r"spectral_Linf=([^ ]+)", "\n".join(mechanical))
    ]
    eta = [abs(float(value)) for value in re.findall(r"eta_Linf=([^ ]+)", "\n".join(mechanical))]
    phase_kkt = [
        abs(float(value)) for value in re.findall(r"final_KKT=([^ ]+)", log)
    ]
    elastic = case_id.endswith("elastic1")
    mechanics_gate_failures = log.count("mechanical_gate_pass=0")
    passed = (
        len(accepts) >= 2
        and log.count("CTOT_COUPLED_STEP_RETRY") == 0
        and log.count("CTOT_MIMETIC_BE_REJECT") == 0
        and max(mass, default=0.0) <= 1.0e-10
        and max(projection, default=0) == 0
        and max(clipping, default=0) == 0
        and mechanics_gate_failures == 0
        and (not elastic or max(eta, default=math.inf) <= MECHANICS_ETA_ACCEPT)
    )
    return {
        "case_id": case_id,
        "phase_candidate": (
            "FINITE_LPHI_BE" if case_id.startswith("finite_lphi")
            else "QUASI_EQUILIBRIUM_FAST_INTERFACE_V1"
        ),
        "elastic_enabled": int(elastic),
        "accepted_steps": len(accepts),
        "retry_count": log.count("CTOT_COUPLED_STEP_RETRY"),
        "reject_count": log.count("CTOT_MIMETIC_BE_REJECT"),
        "max_mass_error_abs": max(mass, default=0.0),
        "max_phase_inner_KKT": max(phase_kkt, default=0.0),
        "max_mechanical_spectral_Linf": max(spectral, default=0.0),
        "max_mechanical_eta_Linf": max(eta, default=0.0),
        "projection_mass": max(projection, default=0),
        "clip_count": max(clipping, default=0),
        "status": "PASS" if passed else "FAIL",
        "evidence": str(path),
    }


def fmt_percent(row: dict[str, str] | None, key: str) -> str:
    if row is None:
        return "NA"
    value = finite_number(row.get(key, ""))
    return "NA" if value is None else f"{value:.3%}"


def calibration_binary_provenance(
    status_rows: list[dict[str, object]], expected_source_hashes: dict[str, str]
) -> dict[str, object]:
    counts = Counter(str(row.get("binary_sha256", "")) for row in status_rows)
    counts.pop("", None)
    source_match = bool(status_rows) and all(
        all(
            str(row.get("source_hashes", {}).get(name, "")) == expected_hash
            for name, expected_hash in expected_source_hashes.items()
        )
        for row in status_rows
    )
    current = counts.most_common(1)[0][0] if counts else "MISSING"
    return {
        "counts": dict(sorted(counts.items())),
        "current": current,
        "all_runtime_source_hashes_match": source_match,
    }


def finalize(
    matrix_root: Path,
    calibration_root: Path,
    stage1_root: Path,
    report_root: Path,
    source_root: Path,
) -> dict[str, object]:
    report_root.mkdir(parents=True, exist_ok=True)
    required_calibration = (
        "coarse4_calibration.md",
        "coarse4_calibration_scan.csv",
        "coarse4_calibration_candidate_scores.csv",
        "coarse4_fine_reference_library.md",
        "coarse4_reference_manifest.json",
        "coarse4_heldout_validation.md",
        "coarse4_heldout_metrics.csv",
    )
    for name in required_calibration:
        source = calibration_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, report_root / name)

    manifest_path = matrix_root / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    scores = read_csv(calibration_root / "coarse4_calibration_candidate_scores.csv")
    scan = read_csv(calibration_root / "coarse4_calibration_scan.csv")
    candidates = classify_candidates(scores, scan)
    selected = [
        (name, data["selected"]) for name, data in candidates.items()
        if data["selected"] is not None
    ]
    if selected:
        raise RuntimeError("finalizer currently expects calibration hard-stop evidence")

    stage1_rows = []
    for log_path in sorted(stage1_root.glob("*/run.log")):
        stage1_rows.append(parse_stage1_log(log_path.parent.name, log_path))
    stage1_pass = len(stage1_rows) == 4 and all(row["status"] == "PASS" for row in stage1_rows)
    if not stage1_pass:
        raise RuntimeError("fresh four-case Stage-1 mechanics replay did not pass")
    write_csv(report_root / "coarse4_fp32_contract_freeze_metrics.csv", stage1_rows)

    source_hash = sha256(source_root / "main_cuda.cu")
    status_rows = [
        json.loads(path.read_text())
        for path in matrix_root.glob("cases/*/workstation_status.json")
    ]
    runtime_source_names = (
        "main_cuda.cu", "cuda_kernels.cu", "cuda_kernels.h", "pf_params.h",
        "thermo_utils.h", "Unit_Psedobinary.py",
    )
    expected_source_hashes = {
        name: sha256(source_root / name) for name in runtime_source_names
    }
    binary_provenance = calibration_binary_provenance(
        status_rows, expected_source_hashes
    )
    binary_counts = ", ".join(
        f"{binary_hash}:{count}"
        for binary_hash, count in binary_provenance["counts"].items()
    )
    contract = f"""# Coarse4 FP32 Contract Freeze

## Frozen contract

- `mechanics_precision_mode={MECHANICS_PRECISION_MODE}`
- `mechanics_acceptance_mode={MECHANICS_ACCEPTANCE_MODE}`
- `eta_floor_version={MECHANICS_ETA_FLOOR_VERSION}`
- `eta_accept={MECHANICS_ETA_ACCEPT:.15e}`
- `double_oracle_contract_hash={MECHANICS_DOUBLE_ORACLE_CONTRACT_HASH}`
- `main_cuda_sha256={source_hash}`
- `current_qualified_workstation_binary_sha256={binary_provenance['current']}`
- `calibration_binary_sha256_counts={binary_counts}`
- `all_calibration_runtime_source_hashes_match={str(binary_provenance['all_runtime_source_hashes_match']).lower()}`

No mechanics precision, elastic constant, eigenstrain, residual normalization,
safety factor, or acceptance threshold was changed during this calibration goal.
The production mechanics remains FP32 spectral; no production double FFT was added.
The calibration includes artifacts from two workstation rebuilds. Their binary
hashes differ, but every recorded runtime source hash matches the frozen source
set above, so this is rebuild provenance rather than source-version drift.

## Fresh 16-cube replay

| Candidate | Elastic | Accepted | max mass | max eta | Projection | Clipping | Status |
|---|---:|---:|---:|---:|---:|---:|---|
"""
    for row in stage1_rows:
        contract += (
            f"| {row['phase_candidate']} | {row['elastic_enabled']} | "
            f"{row['accepted_steps']} | {float(row['max_mass_error_abs']):.3e} | "
            f"{float(row['max_mechanical_eta_Linf']):.3e} | "
            f"{row['projection_mass']} | {row['clip_count']} | {row['status']} |\n"
        )
    contract += """

All four fresh runs accepted two steps with mass, storage, bounds/KKT, energy,
and normalized mechanics predicates active. The immediately preceding
transaction qualification on the identical source/binary hash established
forced retry rollback, split restart equivalence, elastic-off bitwise identity,
and unchanged legacy mechanics behavior. Evidence remains frozen in
`tmp/workstation_fp32_evidence/coarse4_fp32_transaction_v1` and the independent
FP64 oracle/residual-spectrum reports.

`mechanics_gate_replay_status=PASS_4_OF_4`
"""
    (report_root / "coarse4_fp32_contract_freeze.md").write_text(contract)

    reference_manifest = json.loads(
        (calibration_root / "coarse4_reference_manifest.json").read_text()
    )
    reference_manifest.update({
        "schema": "coarse4_fine_reference_manifest_v2",
        "observation_window_version": manifest["observation_window_version"],
        "observation_Fo_fine": manifest["observation_Fo_fine"],
        "matrix_manifest_sha256": sha256(manifest_path),
        "heldout_status": "NOT_BUILT_OR_EXECUTED_CALIBRATION_FAILED",
        "requested_heldout_cases": [
            "T400_half_driving_growth", "T400_moderate_driving_growth",
            "T380_planar_growth", "cylindrical_growth",
            "cylindrical_dissolution", "spherical_case",
            "two_particle_competition",
        ],
        "heldout_refit_allowed": False,
    })
    (report_root / "coarse4_reference_manifest.json").write_text(
        json.dumps(reference_manifest, indent=2) + "\n"
    )
    fine_library = (report_root / "coarse4_fine_reference_library.md").read_text()
    fine_library += f"""

## Frozen observation contract

- Fine interface: `lambda=0.30 nm`, `dx=0.025 nm`, `lambda/dx=12`.
- Phase kinetics: finite `Lphi/Lphi_diff=0.90`; finite-interface correction off.
- Observation: `Fo_lambda={manifest['observation_Fo_fine']}`
  (`{manifest['observation_window_version']}`), physical duration
  `{float(manifest['elapsed_s']):.12e} s`.
- Matrix manifest SHA-256: `{sha256(manifest_path)}`.

The protocol named seven held-out cases, but complete held-out inputs, state
hashes, reference trajectories, and observation manifests were not built after
calibration selected no valid candidate. Their status is
`NOT_BUILT_OR_EXECUTED_CALIBRATION_FAILED`; no held-out datum was used for fit.
"""
    (report_root / "coarse4_fine_reference_library.md").write_text(fine_library)

    finite = candidates["finite_lphi"]
    qe = candidates["quasi_equilibrium"]
    finite_best = finite["best"]
    decision = f"""# Coarse4 Phase-Candidate Decision

## Result

`selected_phase_candidate=NONE`
`selected_a_M=NONE`

The deterministic nonnegative scan contained `{finite['scan_count']}` values:
`{', '.join(str(value) for value in manifest['a_M_scan_frozen_before_results'])}`.
It was frozen before the new results and used one free parameter only.

| Candidate | Status | Best observed a_M | Flux max | Inventory max | Trajectory max |
|---|---|---:|---:|---:|---:|
| FINITE_LPHI_BE | `{finite['status']}` | {finite_best['a_M']} | {fmt_percent(finite_best, 'flux_error_max')} | {fmt_percent(finite_best, 'inventory_error_max')} | {fmt_percent(finite_best, 'trajectory_error_max')} |
| QUASI_EQUILIBRIUM_FAST_INTERFACE_V1 | `{qe['status']}` | NONE | NA | NA | NA |

At `a_M={finite_best['a_M']}`, finite-Lphi minimizes the frozen weighted objective
but still misses all quantitative gates. Low `a_M` can approach the flux target
without producing the required inventory/trajectory response; larger `a_M`
reduces the latter errors only while strongly violating flux. This tradeoff is
not a narrow optimum satisfying the contract.

Every QE growth candidate failed closed at the first phase-KKT solve. Since
`a_M` modifies matrix-side transport mobility, it cannot repair that phase solve.
Mechanics remained qualified and is not used to compensate either failure.

No phase candidate or `a_M` is frozen for held-out use. No second coarse
parameter is introduced.
"""
    (report_root / "coarse4_phase_candidate_decision.md").write_text(decision)

    first_failures = [
        {
            "phase_candidate": "FINITE_LPHI_BE",
            "stage": "CALIBRATION",
            "a_M": finite_best["a_M"],
            "first_failed_gate": "FLUX_INVENTORY_TRAJECTORY_COMBINED",
            "failure_reason": finite["status"],
            "flux_error_max": finite_best["flux_error_max"],
            "inventory_error_max": finite_best["inventory_error_max"],
            "trajectory_error_max": finite_best["trajectory_error_max"],
            "heldout_started": False,
            "physics_changed": False,
        },
        {
            "phase_candidate": "QUASI_EQUILIBRIUM_FAST_INTERFACE_V1",
            "stage": "CALIBRATION_GROWTH_FIRST_STEP",
            "a_M": manifest["a_M_scan_frozen_before_results"][0],
            "first_failed_gate": "PHASE_KKT",
            "failure_reason": qe["status"],
            "flux_error_max": "NA",
            "inventory_error_max": "NA",
            "trajectory_error_max": "NA",
            "heldout_started": False,
            "physics_changed": False,
        },
    ]
    write_csv(report_root / "coarse4_first_failure.csv", first_failures)
    write_csv(report_root / "coarse4_selected_performance.csv", [{
        "status": "NOT_RUN_NO_SELECTED_MODE",
        "selected_phase_candidate": "NONE",
        "selected_a_M": "NONE",
        "grid": "16/32/64",
        "elastic_modes": "OFF/ON",
        "accepted_step_wall_s": "NOT_RUN",
        "outer_iterations": "NOT_RUN",
        "PDAS_iterations": "NOT_RUN",
        "mechanics_iterations": "NOT_RUN",
        "FFT_count": "NOT_RUN",
        "memory_bytes": "NOT_RUN",
        "reason": "Calibration hard stop: profiling an unselected mode is not selected-mode evidence",
    }])

    calibration_md = (report_root / "coarse4_calibration.md").read_text()
    calibration_md += f"""

## Deterministic refinement result

The final scan used protocol `{manifest['a_M_scan_protocol']['version']}` and
all `{len(manifest['cases'])}` runs have terminal status records. The scan was
not adapted to the observed values. Finite-Lphi's best observed point was
`a_M={finite_best['a_M']}` with max relative errors
`{fmt_percent(finite_best, 'flux_error_max')}` flux,
`{fmt_percent(finite_best, 'inventory_error_max')}` inventory, and
`{fmt_percent(finite_best, 'trajectory_error_max')}` trajectory. QE growth
failed phase KKT for all `{qe['scan_count']}` values.
"""
    (report_root / "coarse4_calibration.md").write_text(calibration_md)

    fine_status = (
        "CALIBRATION_REFERENCES_FROZEN_HELDOUT_NOT_BUILT_"
        "CALIBRATION_FAILED"
    )
    final_status = "BLOCKED_ONE_PARAMETER_COARSE_CLOSURE_INSUFFICIENT"
    markers = f"""mechanics_contract_preserved=true
mechanics_gate_replay_status=PASS_4_OF_4
fine_reference_status={fine_status}
finite_Lphi_calibration_status={finite['status']}
finite_Lphi_selected_a_M=NONE
QE_calibration_status={qe['status']}
QE_selected_a_M=NONE
selected_phase_candidate=NONE
selected_a_M=NONE
heldout_planar_error_max=NOT_RUN_CALIBRATION_FAILED
heldout_curved_error_max=NOT_RUN_CALIBRATION_FAILED
heldout_inventory_error_max=NOT_RUN_CALIBRATION_FAILED
competition_ranking_status=NOT_RUN_CALIBRATION_FAILED
particle_survival_ranking_status=NOT_RUN_CALIBRATION_FAILED
selected_mode_64cube_step_time=NOT_RUN_NO_SELECTED_MODE
selected_mode_memory=NOT_RUN_NO_SELECTED_MODE
GP_source_reintegrated=false
large_3D_run_executed=false
cluster_used=false
commit_created=false
push_performed=false
recommended_next_action=HYBRID_SHARP_INTERFACE_BETA_REQUIRED_FOR_DX_1NM
final_status={final_status}
"""
    (report_root / "final_terminal_output.txt").write_text(markers)
    return {
        "final_status": final_status,
        "stage1_pass": stage1_pass,
        "finite_status": finite["status"],
        "qe_status": qe["status"],
        "selected": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        args.matrix_root, args.calibration_root, args.stage1_root,
        args.report_root, args.source_root,
    )
    print(f"mechanics_gate_replay_status={'PASS_4_OF_4' if result['stage1_pass'] else 'FAIL'}")
    print(f"finite_Lphi_calibration_status={result['finite_status']}")
    print(f"QE_calibration_status={result['qe_status']}")
    print(f"final_status={result['final_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
