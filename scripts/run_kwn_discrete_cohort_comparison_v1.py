#!/usr/bin/env python3
"""Adjudicate Eulerian smooth KWN and an exact discrete-cohort comparator.

The runner deliberately keeps the legacy six-particle Eulerian P5 failure
intact.  It reads that audit as immutable evidence, qualifies the Eulerian
backend only for its converged smooth-PSD domain, and uses an event-aware
cohort representation for the finite six-particle beta-only PF fixture.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.cohort_solver import Cohort, CohortSnapshot, CohortSolver, sphere_volume_m3  # noqa: E402
from kwn_mvp.diagnostics import discrete_wasserstein_distance  # noqa: E402
from kwn_mvp.radius_grid import RadiusGrid  # noqa: E402
from kwn_mvp.solver import KWNSolver  # noqa: E402


REPORT_ROOT = ROOT / "reports" / "kwn_discrete_cohort_comparison_v1"
OUTPUT_ROOT = ROOT / "outputs" / "kwn_discrete_cohort_comparison_v1"
FROZEN_PF_ROOT = ROOT / "outputs" / "kwn_pf_cuda_runtime_closure_v1"
LEGACY_REPORT_COMMIT = "2f67a34751af1312eaf63f635f4a5be9166bf03a"
PRIMARY_METRICS = (
    "N_m0_m3",
    "Rmean_m",
    "Rmean3_m3",
    "Sv_m_inv",
    "f_beta",
    "matrix_xB",
)
SMOOTH_TIMES_H = (0.0, 0.1, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0)
EXACT_TIMES_H = (0.0, 0.01, 0.1, 0.39317699499770825, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0)
PF_REQUESTED_TIMES_H = (0.0, 0.1, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0)
HUMAN_GATE_EXPECTED = {
    "LEGACY_SIX_PARTICLE_EULERIAN_P5": "FAIL_RETAINED",
    "SMOOTH_POPULATION_EULERIAN_P5": "ACCEPTED_AS_KWN_POPULATION_BACKEND_QUALIFICATION",
    "EXACT_SIX_PARTICLE_FIXTURE": "REQUIRES_DISCRETE_COHORT_CHARACTERISTIC_COMPARATOR",
    "TWO_PERCENT_GATE": "NOT_RELAXED",
    "EULERIAN_SIX_PARTICLE_OUTPUT": "NOT_AUTHORITY_FOR_BETA_ONLY_PF_COMPARISON",
    "LOCAL_GP_RELEASE": "NOT_AUTHORIZED_UNTIL_COHORT_COMPARISON_PASSES",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(name: str, title: str, body: str) -> None:
    path = REPORT_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), 1.0e-300)


def _sign(value: float, *, tolerance: float = 0.0) -> str:
    if value > tolerance:
        return "INCREASE"
    if value < -tolerance:
        return "DECREASE"
    return "UNCHANGED"


def _float(value: str | float | int | None) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def _load_radius_audit_module() -> Any:
    path = ROOT / "scripts" / "run_kwn_radius_grid_convergence_v1.py"
    specification = importlib.util.spec_from_file_location("radius_grid_audit_for_cohort_v1", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load radius-grid audit runner from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _require_human_gate() -> dict[str, Any]:
    decision_path = OUTPUT_ROOT / "human_gate_decision.json"
    report_path = REPORT_ROOT / "00_human_gate_decision.md"
    if not decision_path.is_file() or not report_path.is_file():
        raise RuntimeError("human gate decision must exist before code-driven adjudication")
    decision = _read_json(decision_path)
    observed = {key: decision.get(key) for key in HUMAN_GATE_EXPECTED}
    if observed != HUMAN_GATE_EXPECTED:
        raise RuntimeError(f"human gate decision mismatch: {observed!r}")
    return decision


def _legacy_input_paths(root: Path) -> list[Path]:
    paths = [
        root / "analysis_provenance.json",
        root / "baseline_reproduction.json",
        root / "final_acceptance.json",
        root / "grid_pair_errors.csv",
        root / "grid_trajectory_all.csv",
        root / "grid_psd_distances.csv",
        root / "kwn_requalification_summary.json",
        root / "initial_projection_audit.json",
        root / "dissolution_event_times.csv",
    ]
    for scenario in ("smooth", "fixture"):
        for bins in (100, 200, 400, 800, 1600, 3200):
            run = root / "runs" / f"ladder_{scenario}_{bins}_uniform"
            paths.extend((run / "manifest.json", run / "trajectory.csv", run / "psd_snapshots.npz"))
            if scenario == "fixture":
                paths.extend((run / "class_events.csv", run / "lower_boundary_flux.csv"))
    return paths


def _baseline_reproduction(legacy_root: Path) -> dict[str, Any]:
    missing = [str(path) for path in _legacy_input_paths(legacy_root) if not path.is_file()]
    if missing:
        result = {
            "status": "FAIL_BASELINE_REPRODUCTION",
            "reason": "immutable radius-grid inputs are missing",
            "missing": missing,
        }
        _write_json(OUTPUT_ROOT / "baseline_reproduction.json", result)
        _write_report(
            "01_baseline_reproduction.md",
            "Read-only baseline reproduction",
            "`FAIL_BASELINE_REPRODUCTION`: required immutable inputs are missing.\n\n"
            + "\n".join(f"- `{path}`" for path in missing),
        )
        return result

    final = _read_json(legacy_root / "final_acceptance.json")
    baseline = _read_json(legacy_root / "baseline_reproduction.json")
    provenance = _read_json(legacy_root / "analysis_provenance.json")
    pairs = _read_csv(legacy_root / "grid_pair_errors.csv")
    required = {
        "baseline": baseline.get("status") == "PASS_RADIUS_GRID_BASELINE_REPRODUCTION",
        "top_status": final.get("top_status") == "DISCRETE_EVENT_SENSITIVITY_GATE_REVIEW_REQUIRED",
        "smooth": final.get("smooth_ladder", {}).get("status") == "PASS_KWN_RADIUS_GRID_CONVERGENCE",
        "fixture": final.get("fixture_ladder", {}).get("status") == "FAIL_KWN_RADIUS_GRID_CONVERGENCE",
        "positivity": final.get("positivity_conservation", {}).get("status")
        == "PASS_KWN_POSITIVITY_CONSERVATION_48H",
    }
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for scenario in ("smooth", "fixture"):
        summary: dict[str, dict[str, float]] = {}
        final_rows = [
            row
            for row in pairs
            if row["scenario"] == scenario
            and row["left_bins"] == "1600"
            and row["right_bins"] == "3200"
            and math.isclose(float(row["target_time_h"]), 48.0, abs_tol=1.0e-12)
        ]
        for row in final_rows:
            summary[row["metric"]] = {
                "endpoint_relative_error": float(row["relative_error"]),
                "full_time_max_relative_error": float(row["full_time_max_relative_error"]),
            }
        summaries[scenario] = summary
    expected_fixture_failure = any(
        values["endpoint_relative_error"] > 0.02
        for values in summaries["fixture"].values()
    )
    expected_smooth_pass = bool(summaries["smooth"]) and all(
        values["endpoint_relative_error"] <= 0.02
        and values["full_time_max_relative_error"] <= 0.02
        for values in summaries["smooth"].values()
    )
    required["final_pair"] = expected_fixture_failure and expected_smooth_pass
    result = {
        "schema_version": "KWN_DISCRETE_COHORT_BASELINE_REPRODUCTION_V1",
        "status": "PASS_RADIUS_GRID_BASELINE_REPRODUCTION"
        if all(required.values())
        else "FAIL_BASELINE_REPRODUCTION",
        "legacy_radius_grid_output_root": str(legacy_root),
        "legacy_report_commit": LEGACY_REPORT_COMMIT,
        "legacy_runtime_analysis_git_head": provenance.get("git_head"),
        "legacy_runtime_analysis_binding": provenance.get("run_launch_source_binding"),
        "checks": required,
        "final_1600_vs_3200": summaries,
        "maximum_fixture_ledger_residual": final.get("positivity_conservation", {}).get(
            "maximum_uniform_fixture_ledger_residual"
        ),
        "input_sha256": {str(path.relative_to(legacy_root)): _sha256_file(path) for path in _legacy_input_paths(legacy_root)},
    }
    _write_json(OUTPUT_ROOT / "baseline_reproduction.json", result)
    body = (
        f"Read-only source: `{legacy_root}`.  The report-producing clean commit is "
        f"`{LEGACY_REPORT_COMMIT}`; the older `git_head` in runtime provenance is retained only as "
        "the disclosed launch-time baseline.\n\n"
        f"Status: `{result['status']}`.\n\n"
        "| Scenario | 1600→3200 N | Rmean | Rmean3 | Sv | fβ | matrix xB |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            "| "
            + scenario
            + " | "
            + " | ".join(
                f"{100.0 * summaries[scenario].get(metric, {}).get('endpoint_relative_error', float('nan')):.6g}%"
                for metric in PRIMARY_METRICS
            )
            + " |"
            for scenario in ("smooth", "fixture")
        )
        + "\n\n"
        f"The immutable 48 h positivity/conservation status remains "
        f"`{final.get('positivity_conservation', {}).get('status')}`, with maximum residual "
        f"`{result['maximum_fixture_ledger_residual']}`."
    )
    _write_report("01_baseline_reproduction.md", "Read-only baseline reproduction", body)
    return result


def _smooth_authority_selection(legacy_root: Path) -> dict[str, Any]:
    rows = _read_csv(legacy_root / "grid_pair_errors.csv")
    pair_metrics: dict[tuple[int, int], dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        if row["scenario"] != "smooth":
            continue
        key = (int(row["left_bins"]), int(row["right_bins"]))
        metric = row["metric"]
        if math.isclose(float(row["target_time_h"]), 48.0, abs_tol=1.0e-12):
            pair_metrics[key][metric] = {
                "endpoint": float(row["relative_error"]),
                "full_time": float(row["full_time_max_relative_error"]),
            }
    assessed: list[dict[str, Any]] = []
    for pair in sorted(pair_metrics):
        values = pair_metrics[pair]
        passed = set(values) == set(PRIMARY_METRICS) and all(
            values[metric]["endpoint"] <= 0.02 and values[metric]["full_time"] <= 0.02
            for metric in PRIMARY_METRICS
        )
        assessed.append(
            {"pair": f"{pair[0]}_vs_{pair[1]}", "left": pair[0], "right": pair[1], "pass": passed, "metrics": values}
        )
    passing = [item for item in assessed if item["pass"]]
    if not passing:
        return {"status": "FAIL_EULERIAN_SMOOTH_POPULATION_AUTHORITY", "pairs": assessed, "authority_grid": None}
    # The registered ladder ends at 3200; 1600→3200 is the available
    # next-finer confirmation.  800→1600 fails, so 1600 cannot be authority.
    final = passing[-1]
    previous = next((item for item in assessed if item["right"] == final["left"]), None)
    if previous is not None and previous["pass"]:
        authority = int(final["left"])
        rationale = "two consecutive passing pairs select the smallest doubly-confirmed grid"
    else:
        authority = int(final["right"])
        rationale = (
            "the first registered passing pair is 1600→3200; its finer member is retained as authority. "
            "No unregistered 6400-bin extension is permitted."
        )
    return {
        "status": "PENDING_REQUALIFICATION",
        "authority_grid": authority,
        "qualifying_pair": final["pair"],
        "selection_rationale": rationale,
        "pairs": assessed,
    }


def _kwn_snapshot(solver: KWNSolver) -> dict[str, float]:
    beta = solver.population("beta")
    moments = [beta.radius_moment(order, quadrature="fixed_pivot") for order in range(4)]
    m0, m1, m2, m3 = (float(value) for value in moments)
    ledger = solver.ledger.snapshot(
        matrix_xb=solver.matrix_xb,
        populations=solver.population_list(),
        beta_resolved_fraction=1.0,
    )
    return {
        "time_h": solver.time_s / 3600.0,
        "N_m0_m3": m0,
        "M0_m3": m0,
        "M1_m2": m1,
        "M2_m": m2,
        "M3_dimensionless": m3,
        "Rmean_m": 0.0 if m0 == 0.0 else m1 / m0,
        "Rmean3_m3": 0.0 if m0 == 0.0 else m3 / m0,
        "Sv_m_inv": 4.0 * math.pi * m2,
        "f_beta": 4.0 * math.pi * m3 / 3.0,
        "matrix_xB": solver.matrix_xb,
        "inventory_relative_residual": ledger.relative_residual,
        "minimum_bin_density_per_m4": float(np.min(beta.number_density_per_m4)),
        "roundoff_zeroed_bin_count": float(solver.roundoff_zeroed_bin_count),
    }


def _advance_kwn(solver: KWNSolver, target_s: float) -> dict[str, float]:
    maximum_residual = 0.0
    while solver.time_s < target_s:
        diagnostic = solver.advance_one(maximum_dt_s=target_s - solver.time_s)
        maximum_residual = max(maximum_residual, float(diagnostic.inventory.relative_residual))
        if solver.history:
            solver.history.pop()
    return {"maximum_inventory_relative_residual": maximum_residual}


def _run_smooth_kwn(
    audit: Any, *, bins: int, max_dt_factor: float, output_times_h: Sequence[float]
) -> tuple[KWNSolver, dict[str, Any], Any, Any, list[dict[str, float]], float]:
    solver, construction, contract, fixture = audit._build_smooth_solver(
        bins=bins, max_dt_factor=max_dt_factor
    )
    rows: list[dict[str, float]] = []
    maximum_residual = 0.0
    for target_h in output_times_h:
        maximum_residual = max(
            maximum_residual,
            _advance_kwn(solver, float(target_h) * 3600.0)["maximum_inventory_relative_residual"],
        )
        row = _kwn_snapshot(solver)
        row["target_time_h"] = float(target_h)
        rows.append(row)
    return solver, construction, contract, fixture, rows, maximum_residual


def _authority_requalification(legacy_root: Path) -> dict[str, Any]:
    selection = _smooth_authority_selection(legacy_root)
    if selection["status"] != "PENDING_REQUALIFICATION":
        _write_json(OUTPUT_ROOT / "eulerian_smooth_authority.json", selection)
        _write_report(
            "02_eulerian_smooth_authority.md",
            "Eulerian smooth-population authority",
            "`FAIL_EULERIAN_SMOOTH_POPULATION_AUTHORITY`: no registered smooth-grid pair satisfies the full-time 2% gate.",
        )
        return selection
    audit = _load_radius_audit_module()
    authority = int(selection["authority_grid"])
    continuous, construction, contract, fixture, rows, continuous_residual = _run_smooth_kwn(
        audit,
        bins=authority,
        max_dt_factor=1.0,
        output_times_h=SMOOTH_TIMES_H,
    )
    half, _, _, _, half_rows, half_residual = _run_smooth_kwn(
        audit,
        bins=authority,
        max_dt_factor=0.5,
        output_times_h=SMOOTH_TIMES_H,
    )
    endpoint = rows[-1]
    timestep_errors: dict[str, float] = {metric: 0.0 for metric in PRIMARY_METRICS}
    tolerance_rows: list[dict[str, Any]] = []
    for full_row, half_row in zip(rows, half_rows):
        if not math.isclose(
            float(full_row["target_time_h"]), float(half_row["target_time_h"]), abs_tol=1.0e-12
        ):
            raise RuntimeError("smooth timestep trajectories do not share the registered output schedule")
        for metric in PRIMARY_METRICS:
            error = _relative_error(float(full_row[metric]), float(half_row[metric]))
            timestep_errors[metric] = max(timestep_errors[metric], error)
            tolerance_rows.append(
                {
                    "scope": "smooth_authority_timestep",
                    "authority_grid": authority,
                    "target_time_h": full_row["target_time_h"],
                    "metric": metric,
                    "relative_error_max_dt_vs_half": error,
                    "threshold": 0.02,
                    "pass": error <= 0.02,
                }
            )
    timestep_pass = all(value <= 0.02 for value in timestep_errors.values())

    direct, _, _, _, _, direct_residual = _run_smooth_kwn(
        audit, bins=authority, max_dt_factor=1.0, output_times_h=(48.0,)
    )
    restart, _, _, _, _, restart_residual_a = _run_smooth_kwn(
        audit, bins=authority, max_dt_factor=1.0, output_times_h=(24.0,)
    )
    checkpoint = OUTPUT_ROOT / "smooth_authority_restart_24h.npz"
    restart.save_checkpoint(checkpoint)
    resumed = KWNSolver.load_checkpoint(config=restart.config, path=checkpoint)
    restart_residual_b = _advance_kwn(resumed, 48.0 * 3600.0)["maximum_inventory_relative_residual"]
    direct_arrays = direct.state_arrays()
    resumed_arrays = resumed.state_arrays()
    restart_differences = {
        key: float(np.max(np.abs(direct_arrays[key] - resumed_arrays[key])) if np.asarray(direct_arrays[key]).size else 0.0)
        for key in direct_arrays
    }
    restart_scale = {
        key: max(float(np.max(np.abs(np.asarray(value)))) if np.asarray(value).size else 0.0, 1.0e-300)
        for key, value in direct_arrays.items()
    }
    restart_relative = {key: restart_differences[key] / restart_scale[key] for key in restart_differences}
    restart_pass = all(value <= 1.0e-10 for value in restart_relative.values())
    max_residual = max(continuous_residual, half_residual, direct_residual, restart_residual_a, restart_residual_b)
    state_pass = (
        all(
            row["minimum_bin_density_per_m4"] >= 0.0
            and row["roundoff_zeroed_bin_count"] == 0.0
            for row in rows + half_rows
        )
        and max_residual <= 1.0e-10
    )
    status = (
        "PASS_EULERIAN_SMOOTH_POPULATION_AUTHORITY"
        if timestep_pass and restart_pass and state_pass
        else "FAIL_EULERIAN_SMOOTH_POPULATION_AUTHORITY"
    )
    restart_rows = [
        {
            "comparison": "continuous_0_48h_vs_restart_0_24_48h",
            "state_array": key,
            "maximum_absolute_difference": restart_differences[key],
            "maximum_relative_difference": restart_relative[key],
            "threshold": 1.0e-10,
            "pass": restart_relative[key] <= 1.0e-10,
        }
        for key in restart_differences
    ]
    _write_csv(OUTPUT_ROOT / "eulerian_smooth_authority_timestep.csv", tolerance_rows)
    _write_csv(OUTPUT_ROOT / "eulerian_smooth_restart_comparison.csv", restart_rows)
    result = {
        "schema_version": "KWN_EULERIAN_SMOOTH_AUTHORITY_V1",
        **{key: value for key, value in selection.items() if key != "status"},
        "status": status,
        "authority_config_hash": continuous.config.source_config_hash,
        "semantic_config_hash": _canonical_hash(construction["config_mapping"]),
        "contract_hash": contract.contract_hash,
        "fixture_hash": fixture.fixture_hash,
        "timestep_convergence": {"pass": timestep_pass, "errors": timestep_errors},
        "restart": {
            "status": "PASS" if restart_pass else "FAIL",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "relative_differences": restart_relative,
        },
        "state_gate": {
            "pass": state_pass,
            "maximum_inventory_relative_residual": max_residual,
            "minimum_bin_density_per_m4": min(
                row["minimum_bin_density_per_m4"] for row in rows + half_rows
            ),
            "roundoff_zeroed_bin_count": max(
                row["roundoff_zeroed_bin_count"] for row in rows + half_rows
            ),
        },
        "authority_trajectory": rows,
    }
    _write_json(OUTPUT_ROOT / "eulerian_smooth_authority.json", result)
    _write_report(
        "02_eulerian_smooth_authority.md",
        "Eulerian smooth-population authority",
        f"Status: `{status}`.  Authority grid: `{authority}`; config hash: "
        f"`{continuous.config.source_config_hash}`.\n\n"
        f"Selection: {selection['selection_rationale']}  The inherited 800→1600 smooth pair fails, "
        "so 1600 is not authority; registered 1600→3200 is the qualifying full-time pair.\n\n"
        f"Timestep gate: `{timestep_pass}`; restart gate: `{restart_pass}`; maximum inventory residual: `{max_residual:.6e}`.",
    )
    return result


def _frozen_pf_rows() -> tuple[dict[int, dict[str, str]], dict[int, list[dict[str, str]]]]:
    trajectory_path = FROZEN_PF_ROOT / "cuda_ae_trajectories.csv"
    component_path = FROZEN_PF_ROOT / "cuda_component_history.csv"
    if not trajectory_path.is_file() or not component_path.is_file():
        raise RuntimeError("frozen Case A PF files are unavailable")
    trajectory: dict[int, dict[str, str]] = {}
    for row in _read_csv(trajectory_path):
        if row["case"] == "A":
            trajectory[int(row["step"])] = row
    components: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(component_path):
        if row["case"] == "A":
            components[int(row["step"])].append(row)
    return trajectory, components


def _inherited_cuda_evidence() -> dict[str, Any]:
    """Read the frozen CUDA A--E closure record and make its reuse explicit."""

    audit_path = FROZEN_PF_ROOT / "cuda_ae_runtime_audit.json"
    binary_path = FROZEN_PF_ROOT / "build_manifest.json"
    audit = _read_json(audit_path)
    binary = _read_json(binary_path)
    case_e = audit.get("case_E_runtime_closure", {})
    trajectory_rows = _read_csv(FROZEN_PF_ROOT / "cuda_ae_trajectories.csv")
    case_a_residuals = [
        float(row["four_bucket_relative_residual"])
        for row in trajectory_rows
        if row["case"] == "A"
    ]
    checks = {
        "controlled_binary_provenance": binary.get("status")
        == "PASS_CONTROLLED_CUDA_BINARY_PROVENANCE_V1",
        "cuda_ae_smoke": audit.get("status") == "PASS_CUDA_AE_SMOKE",
        "runtime_failures_empty": audit.get("runtime_failures") == [],
        "four_bucket_no_double_count": case_e.get("double_count_status")
        == "PASS_FOUR_BUCKET_SUM_AND_FIXED_RESOLVED_INVENTORY_CHECKED_AT_EVERY_CHECKPOINT",
        "case_e_four_bucket_residual": float(
            case_e.get("max_four_bucket_relative_residual", float("inf"))
        ) <= 1.0e-10,
        "case_a_four_bucket_residual": bool(case_a_residuals)
        and max(case_a_residuals) <= 1.0e-10,
    }
    result = {
        "schema_version": "INHERITED_CUDA_AE_EVIDENCE_REUSE_V1",
        "status": "PASS_INHERITED_CUDA_AE_EVIDENCE" if all(checks.values()) else "FAIL_INHERITED_CUDA_AE_EVIDENCE",
        "checks": checks,
        "case_e_max_four_bucket_relative_residual": case_e.get("max_four_bucket_relative_residual"),
        "case_a_max_four_bucket_relative_residual": max(case_a_residuals) if case_a_residuals else None,
        "cuda_rerun": False,
        "pf_source_modified": False,
        "source_files": {
            "runtime_audit": str(audit_path),
            "runtime_audit_sha256": _sha256_file(audit_path),
            "build_manifest": str(binary_path),
            "build_manifest_sha256": _sha256_file(binary_path),
        },
    }
    _write_json(OUTPUT_ROOT / "inherited_cuda_evidence.json", result)
    return result


def _strict_fixture_cohorts(audit: Any) -> tuple[CohortSolver, dict[str, Any], Any, Any]:
    kwn, construction, contract, fixture = audit._build_fixture_solver(bins=3200)
    trajectory, components = _frozen_pf_rows()
    t0 = next((row for row in trajectory.values() if math.isclose(float(row["time_h"]), 0.0, abs_tol=1.0e-15)), None)
    if t0 is None:
        raise RuntimeError("frozen Case A lacks t=0 trajectory row")
    t0_components = components[int(t0["step"])]
    profiles = fixture.source_details["profiles"]
    profile_by_center = {
        # The frozen fixture records zero-based cell indices while the PF
        # component extractor reports physical cell centres in nm.  The
        # 96^3 fixture has a 1 nm cell size, hence centre = index + 0.5 nm.
        tuple(float(value) + 0.5 for value in profile["center_grid"]): str(profile["particle_id"])
        for profile in profiles
    }
    mapping: dict[str, str] = {}
    radii: dict[str, float] = {}
    for row in t0_components:
        center = tuple(float(row[key]) for key in ("centroid_x_nm", "centroid_y_nm", "centroid_z_nm"))
        matches = [
            fixture_id
            for expected, fixture_id in profile_by_center.items()
            if all(math.isclose(observed, wanted, rel_tol=0.0, abs_tol=1.0e-6) for observed, wanted in zip(center, expected))
        ]
        if len(matches) != 1:
            raise RuntimeError(f"cannot fail-closed-map frozen t=0 PF component at {center}")
        pf_id = str(row["particle_id"])
        mapping[matches[0]] = pf_id
        radii[matches[0]] = float(row["equivalent_radius_nm"]) * 1.0e-9
    if set(mapping) != set(profile_by_center.values()) or len(set(mapping.values())) != 6:
        raise RuntimeError("frozen PF initial identity mapping is unresolved")
    sharp_volume = math.fsum(sphere_volume_m3(radius) for radius in radii.values())
    target_beta_volume = float(t0["beta_volume_fraction"]) * fixture.box_volume_m3
    physical_weight_per_particle_m3 = 1.0 / fixture.box_volume_m3
    # PF's global resolved-beta bucket includes the diffuse interface whereas
    # the component extractor supplies sharp equivalent radii.  The common
    # factor below is therefore a derived inventory representation map: it is
    # fixed by t=0 PF volume closure, never fitted to later dynamics, and does
    # not alter a physical radius or any material parameter.
    common_weight_scale = target_beta_volume / sharp_volume
    weight = common_weight_scale * physical_weight_per_particle_m3
    cohorts = [Cohort(initial_id=fixture_id, radius_m=radii[fixture_id], weight_m3=weight) for fixture_id in sorted(radii)]
    solver = CohortSolver.from_kwn_solver(
        kwn_solver=kwn,
        cohorts=cohorts,
        rtol=1.0e-10,
        atol_m=1.0e-18,
        method="Radau",
    )
    snapshot = solver.snapshot()
    strict_matrix_xb = (
        float(t0["Q_B_matrix_mol"])
        * kwn.config.matrix_molar_volume_m3_mol
        / ((1.0 - target_beta_volume / fixture.box_volume_m3) * fixture.box_volume_m3)
    )
    checks = {
        "six_component_identity_resolved": len(mapping) == 6,
        "radii_exact_pf_t0_equivalent": all(
            math.isclose(radii[fixture_id], solver_row["radius_m"], rel_tol=0.0, abs_tol=0.0)
            for fixture_id, solver_row in {
                row["initial_id"]: row for row in solver.cohort_rows()
            }.items()
        ),
        "beta_inventory": _relative_error(
            snapshot.beta_inventory_mol_m3,
            float(t0["Q_B_beta_resolved_fixed_mol"]) / fixture.box_volume_m3,
        ) <= 1.0e-12,
        "matrix_xB": abs(snapshot.matrix_xB - strict_matrix_xb) <= 1.0e-12,
        "total_inventory": _relative_error(
            snapshot.total_inventory_mol_m3,
            float(t0["Q_B_total_mol"]) / fixture.box_volume_m3,
        ) <= 1.0e-12,
        "contract_hash": str(t0["validation_contract_hash"]) == contract.contract_hash,
        "fixture_hash": str(t0["fixture_manifest_sha256"]) == fixture.fixture_hash,
        "temperature": math.isclose(kwn.config.temperature_k, contract.temperature_K, abs_tol=1.0e-12),
        "resolved_equivalent_weight_formula": math.isclose(
            weight,
            physical_weight_per_particle_m3 * target_beta_volume / sharp_volume,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
    }
    identity = {
        "status": "PASS_BETA_ONLY_INITIAL_STATE_IDENTITY"
        if all(checks.values())
        else "FAIL_BETA_ONLY_INITIAL_STATE_IDENTITY",
        "checks": checks,
        "cohort_to_pf_particle_id": mapping,
        "cohort_radii_m": radii,
        "common_weight_scale": common_weight_scale,
        "physical_weight_per_particle_m3": physical_weight_per_particle_m3,
        "weight_per_cohort_m3": weight,
        "target_beta_volume_m3": target_beta_volume,
        "sharp_sphere_volume_m3": sharp_volume,
        "strict_pf_matrix_xB": strict_matrix_xb,
        "cohort_matrix_xB": snapshot.matrix_xB,
        "weight_representation": (
            "one physical particle per box is preserved as identity/count; "
            "cohort weight is the t=0 PF-resolved-volume-equivalent common number scale over box volume"
        ),
        "contract_hash": contract.contract_hash,
        "fixture_hash": fixture.fixture_hash,
    }
    return solver, identity, contract, fixture


def _new_fixture_cohort_solver(audit: Any, *, rtol: float, method: str) -> tuple[CohortSolver, dict[str, Any], Any, Any]:
    _, identity, contract, fixture = _strict_fixture_cohorts(audit)
    kwn, _, _, _ = audit._build_fixture_solver(bins=3200)
    cohorts = [
        Cohort(
            initial_id=identifier,
            radius_m=float(radius),
            weight_m3=float(identity["weight_per_cohort_m3"]),
        )
        for identifier, radius in sorted(identity["cohort_radii_m"].items())
    ]
    solver = CohortSolver.from_kwn_solver(
        kwn_solver=kwn, cohorts=cohorts, rtol=rtol, atol_m=1.0e-18, method=method
    )
    return solver, identity, contract, fixture


def _snapshot_row(snapshot: CohortSnapshot, *, target_time_h: float, cohort_count: int, record_type: str = "GLOBAL") -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_type": record_type,
        "target_time_h": target_time_h,
        "time_h": snapshot.time_s / 3600.0,
        "cohort_count": cohort_count,
    }
    row.update(asdict(snapshot))
    return row


def _run_cohort_trajectory(solver: CohortSolver, times_h: Sequence[float], *, cohort_count: int) -> tuple[dict[float, CohortSnapshot], list[dict[str, Any]], list[dict[str, Any]]]:
    snapshots: dict[float, CohortSnapshot] = {}
    global_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    for time_h in times_h:
        solver.advance_to(float(time_h) * 3600.0)
        snapshot = solver.snapshot()
        snapshots[float(time_h)] = snapshot
        global_rows.append(_snapshot_row(snapshot, target_time_h=float(time_h), cohort_count=cohort_count))
        for row in solver.cohort_rows():
            cohort_rows.append(
                {
                    "record_type": "COHORT",
                    "target_time_h": float(time_h),
                    "time_h": snapshot.time_s / 3600.0,
                    "cohort_count": cohort_count,
                    **row,
                    "active_cohort_count": snapshot.active_cohort_count,
                    "matrix_xB": snapshot.matrix_xB,
                    "ledger_relative_residual": snapshot.inventory_relative_residual,
                }
            )
    return snapshots, global_rows, cohort_rows


def _run_cohort_unit_tests() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    process = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.kwn.test_cohort_solver"],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {"returncode": process.returncode, "pass": process.returncode == 0, "output": process.stdout[-12000:]}


def _cohort_numerical_qualification() -> tuple[dict[str, Any], dict[float, CohortSnapshot], list[dict[str, Any]], dict[str, Any]]:
    audit = _load_radius_audit_module()
    unit_tests = _run_cohort_unit_tests()
    tolerance_snapshots: dict[float, CohortSnapshot] = {}
    tolerance_events: dict[float, dict[str, float | None]] = {}
    for rtol in (1.0e-6, 1.0e-8, 1.0e-10):
        solver, _, _, _ = _new_fixture_cohort_solver(audit, rtol=rtol, method="Radau")
        solver.advance_to(48.0 * 3600.0)
        tolerance_snapshots[rtol] = solver.snapshot()
        tolerance_events[rtol] = {
            row["initial_id"]: (
                None if row["dissolution_time_s"] is None else float(row["dissolution_time_s"])
            )
            for row in solver.cohort_rows()
        }
    fine = tolerance_snapshots[1.0e-10]
    tolerance_rows: list[dict[str, Any]] = []
    continuous_metrics = ("N_m0_m3", "Rmean_m", "Rmean3_m3", "Sv_m_inv", "f_beta", "matrix_xB")
    for rtol in (1.0e-6, 1.0e-8):
        for metric in continuous_metrics:
            observed = float(getattr(tolerance_snapshots[rtol], metric))
            reference = float(getattr(fine, metric))
            tolerance_rows.append(
                {
                    "record_type": "continuous_metric",
                    "rtol": rtol,
                    "reference_rtol": 1.0e-10,
                    "metric": metric,
                    "relative_error": _relative_error(observed, reference),
                    "threshold": 1.0e-3,
                    "pass": _relative_error(observed, reference) <= 1.0e-3,
                }
            )
        for initial_id, event_time in tolerance_events[rtol].items():
            reference = tolerance_events[1.0e-10][initial_id]
            if event_time is None and reference is None:
                absolute, relative, event_pass = 0.0, 0.0, True
            elif event_time is None or reference is None:
                absolute, relative, event_pass = float("inf"), float("inf"), False
            else:
                absolute = abs(event_time - reference)
                relative = absolute / max(abs(reference), 1.0e-300)
                event_pass = relative <= 1.0e-3
            tolerance_rows.append(
                {
                    "record_type": "dissolution_event",
                    "rtol": rtol,
                    "reference_rtol": 1.0e-10,
                    "initial_id": initial_id,
                    "event_time_s": event_time,
                    "reference_event_time_s": reference,
                    "absolute_difference_s": absolute,
                    "relative_difference": relative,
                    "threshold_relative": 1.0e-3,
                    "pass": event_pass,
                }
            )
    exact_solver, identity, _, _ = _new_fixture_cohort_solver(audit, rtol=1.0e-10, method="Radau")
    exact_snapshots, exact_global, exact_cohort_rows = _run_cohort_trajectory(
        exact_solver, EXACT_TIMES_H, cohort_count=6
    )
    continuous, _, _, _ = _new_fixture_cohort_solver(audit, rtol=1.0e-10, method="Radau")
    continuous.advance_to(48.0 * 3600.0)
    split, _, _, _ = _new_fixture_cohort_solver(audit, rtol=1.0e-10, method="Radau")
    split.advance_to(24.0 * 3600.0)
    checkpoint_path = OUTPUT_ROOT / "cohort_restart_24h.json"
    _write_json(checkpoint_path, split.checkpoint())
    resumed, _, _, _ = _new_fixture_cohort_solver(audit, rtol=1.0e-10, method="Radau")
    resumed.restore_checkpoint(_read_json(checkpoint_path))
    resumed.advance_to(48.0 * 3600.0)
    restart_rows: list[dict[str, Any]] = []
    for field, value in asdict(continuous.snapshot()).items():
        if isinstance(value, int):
            difference = abs(int(value) - int(asdict(resumed.snapshot())[field]))
            relative = float(difference)
        else:
            difference = abs(float(value) - float(asdict(resumed.snapshot())[field]))
            relative = difference / max(abs(float(value)), 1.0e-300)
        restart_rows.append(
            {
                "record_type": "aggregate",
                "field": field,
                "maximum_absolute_difference": difference,
                "relative_difference": relative,
                "threshold": 1.0e-3,
                "pass": relative <= 1.0e-3,
            }
        )
    for left, right in zip(continuous.cohort_rows(), resumed.cohort_rows()):
        radius_difference = abs(float(left["radius_m"]) - float(right["radius_m"]))
        radius_relative = radius_difference / max(abs(float(left["radius_m"])), 1.0e-300)
        event_left = left["dissolution_time_s"]
        event_right = right["dissolution_time_s"]
        event_difference = (
            0.0
            if event_left is None and event_right is None
            else abs(float(event_left) - float(event_right))
            if event_left is not None and event_right is not None
            else float("inf")
        )
        event_radius_left = left["radius_before_event_m"]
        event_radius_right = right["radius_before_event_m"]
        event_radius_difference = (
            0.0
            if event_radius_left is None and event_radius_right is None
            else abs(float(event_radius_left) - float(event_radius_right))
            if event_radius_left is not None and event_radius_right is not None
            else float("inf")
        )
        returned_difference = abs(
            float(left["returned_inventory_mol_m3"])
            - float(right["returned_inventory_mol_m3"])
        )
        returned_relative = returned_difference / max(
            abs(float(left["returned_inventory_mol_m3"])), 1.0e-300
        )
        event_ledger_left = left["post_event_ledger_relative_residual"]
        event_ledger_right = right["post_event_ledger_relative_residual"]
        event_ledger_difference = (
            0.0
            if event_ledger_left is None and event_ledger_right is None
            else abs(float(event_ledger_left) - float(event_ledger_right))
            if event_ledger_left is not None and event_ledger_right is not None
            else float("inf")
        )
        restart_rows.append(
            {
                "record_type": "cohort",
                "initial_id": left["initial_id"],
                "active_match": left["active"] == right["active"],
                "radius_absolute_difference_m": radius_difference,
                "radius_relative_difference": radius_relative,
                "event_time_absolute_difference_s": event_difference,
                "radius_before_event_absolute_difference_m": event_radius_difference,
                "returned_inventory_relative_difference": returned_relative,
                "post_event_ledger_relative_difference": event_ledger_difference,
                "threshold": 1.0e-3,
                "pass": left["active"] == right["active"]
                and radius_relative <= 1.0e-3
                and event_difference <= 1.0e-3
                and event_radius_difference <= 1.0e-18
                and returned_relative <= 1.0e-3
                and event_ledger_difference <= 1.0e-10,
            }
        )
    max_residual = max(snapshot.inventory_relative_residual for snapshot in exact_snapshots.values())
    tolerance_pass = all(bool(row["pass"]) for row in tolerance_rows)
    restart_pass = all(bool(row["pass"]) for row in restart_rows)
    status = (
        "PASS_DISCRETE_COHORT_NUMERICS"
        if unit_tests["pass"] and tolerance_pass and restart_pass and max_residual <= 1.0e-10
        else "FAIL_DISCRETE_COHORT_NUMERICS"
    )
    result = {
        "schema_version": "KWN_DISCRETE_COHORT_NUMERICAL_QUALIFICATION_V1",
        "status": status,
        "integrator": "RADAU_WITH_ONE_SIDED_RMIN_CHARACTERISTIC_EVENT_COORDINATE",
        "rtol_values": [1.0e-6, 1.0e-8, 1.0e-10],
        "atol_radius_m": 1.0e-18,
        "unit_tests": unit_tests,
        "tolerance_pass": tolerance_pass,
        "restart_pass": restart_pass,
        "max_inventory_relative_residual": max_residual,
        "initial_state_identity": identity,
        "event_times_s": tolerance_events,
        "restart_checkpoint": str(checkpoint_path),
        "restart_checkpoint_sha256": _sha256_file(checkpoint_path),
    }
    _write_csv(OUTPUT_ROOT / "cohort_tolerance_convergence.csv", tolerance_rows)
    _write_csv(OUTPUT_ROOT / "cohort_restart_comparison.csv", restart_rows)
    _write_json(OUTPUT_ROOT / "cohort_numerical_qualification.json", result)
    _write_report(
        "03_cohort_solver_design.md",
        "Discrete-cohort solver design",
        "Each cohort retains its initial ID, frozen PF t=0 equivalent radius, active state, "
        "dissolution time, returned inventory and a strictly derived current inventory.  No cohort is binned, "
        "projected, split or smoothed.\n\n"
        "The physical fixture count is one particle per box.  To close the frozen PF global resolved-beta bucket with "
        "the six sharp-equivalent component radii, the cohort inventory weight is the unique common t=0 "
        "resolved-volume-equivalent number scale divided by box volume.  It is derived from the PF t=0 bucket, not a "
        "fitted parameter, and leaves radii, D(T), gamma and thermodynamics unchanged.\n\n"
        "The RHS uses the existing `growth_rate_m_s`, exact validation-contract curvature equilibrium, D(T), "
        "molar volumes and lower edge.  Matrix xB is algebraically recovered from total inventory after every RHS "
        "evaluation.  A one-sided radius characteristic ends at the Rmin convention without evaluating an invalid "
        "sub-Rmin thermodynamic state; the remaining Rmin inventory is returned to the matrix at the accepted event.",
    )
    _write_report(
        "04_cohort_numerical_qualification.md",
        "Discrete-cohort numerical qualification",
        f"Status: `{status}`.  C1–C5 unit-test return code: `{unit_tests['returncode']}`; "
        f"C6 tolerance pass: `{tolerance_pass}`; C7 restart pass: `{restart_pass}`; "
        f"C8 maximum residual: `{max_residual:.6e}`.\n\n"
        f"The 1e-8 versus 1e-10 continuous metric gate is 0.1%; event-time absolute and relative differences are in "
        "`outputs/kwn_discrete_cohort_comparison_v1/cohort_tolerance_convergence.csv`.",
    )
    return result, exact_snapshots, exact_global, {"identity": identity, "cohort_rows": exact_cohort_rows}


def _smooth_quadrature_cohorts(
    audit: Any, count: int, *, authority_grid: int
) -> tuple[CohortSolver, dict[str, Any]]:
    kwn, construction, contract, fixture = audit._build_smooth_solver(bins=authority_grid)
    beta = kwn.population("beta")
    raw_radii = np.asarray(fixture.resolved_equivalent_radii_m, dtype=np.float64)
    median = float(np.exp(np.mean(np.log(raw_radii))))
    sigma = 0.14
    nodes, gauss_weights = np.polynomial.legendre.leggauss(count)
    low_log = math.log(median) - 12.0 * sigma
    high_log = math.log(median) + 12.0 * sigma
    log_radii = 0.5 * (high_log + low_log) + 0.5 * (high_log - low_log) * nodes
    radii = np.exp(log_radii)
    dlog = 0.5 * (high_log - low_log) * gauss_weights
    lognormal_per_m = np.exp(-0.5 * ((log_radii - math.log(median)) / sigma) ** 2)
    lognormal_per_m /= radii * sigma * math.sqrt(2.0 * math.pi)
    raw_weights = lognormal_per_m * radii * dlog
    target_m3 = beta.radius_moment(3, quadrature="fixed_pivot")
    scale = target_m3 / float(np.sum(raw_weights * radii**3))
    weights = raw_weights * scale
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise RuntimeError("smooth deterministic quadrature generated a non-positive cohort weight")
    cohorts = [
        Cohort(initial_id=f"Q{index + 1:04d}", radius_m=float(radius), weight_m3=float(weight))
        for index, (radius, weight) in enumerate(zip(radii, weights))
    ]
    solver = CohortSolver.from_kwn_solver(
        kwn_solver=kwn, cohorts=cohorts, rtol=1.0e-9, atol_m=1.0e-18, method="DOP853"
    )
    return solver, {
        "count": count,
        "quadrature": "DETERMINISTIC_GAUSS_LEGENDRE_IN_LOG_RADIUS_NO_BINS",
        "median_radius_m": median,
        "log_sigma": sigma,
        "log_radius_bounds_m": [math.exp(low_log), math.exp(high_log)],
        "target_M3_dimensionless": target_m3,
        "quadrature_M3_dimensionless": solver.snapshot().M3_dimensionless,
        "contract_hash": contract.contract_hash,
        "fixture_hash": fixture.fixture_hash,
        "source_config_hash": kwn.config.source_config_hash,
        "smooth_benchmark": construction["smooth_psd_benchmark"],
    }


def _legacy_smooth_row(legacy_root: Path, target_time_h: float, *, bins: int) -> dict[str, str]:
    rows = _read_csv(legacy_root / "runs" / f"ladder_smooth_{bins}_uniform" / "trajectory.csv")
    matches = [
        row
        for row in rows
        if math.isclose(float(row["target_time_h"]), target_time_h, rel_tol=0.0, abs_tol=1.0e-12)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"smooth {bins} legacy trajectory lacks unique {target_time_h} h row")
    return matches[0]


def _legacy_smooth_psd(legacy_root: Path, target_time_h: float, *, bins: int) -> tuple[np.ndarray, np.ndarray]:
    path = legacy_root / "runs" / f"ladder_smooth_{bins}_uniform" / "psd_snapshots.npz"
    with np.load(path) as archive:
        times = np.asarray(archive["target_time_h"], dtype=np.float64)
        matches = np.flatnonzero(np.isclose(times, target_time_h, rtol=0.0, atol=1.0e-12))
        if matches.size != 1:
            raise RuntimeError(f"smooth {bins} PSD lacks unique {target_time_h} h snapshot")
        grid = RadiusGrid(np.asarray(archive["radius_edges_m"], dtype=np.float64))
        density = np.asarray(archive["number_density_per_m4"], dtype=np.float64)[int(matches[0])]
    return grid.centres_m, density * grid.widths_m


def _cohort_eulerian_smooth_crosscheck(
    legacy_root: Path, authority: Mapping[str, Any]
) -> dict[str, Any]:
    audit = _load_radius_audit_module()
    authority_grid = int(authority["authority_grid"])
    trajectories: dict[int, dict[float, CohortSnapshot]] = {}
    metadata: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for count in (400, 800, 1600):
        solver, meta = _smooth_quadrature_cohorts(audit, count, authority_grid=authority_grid)
        snapshots, _, trajectory_cohort_rows = _run_cohort_trajectory(
            solver, SMOOTH_TIMES_H, cohort_count=count
        )
        trajectories[count] = snapshots
        metadata[count] = meta
        for time_h, snapshot in snapshots.items():
            eulerian = _legacy_smooth_row(legacy_root, time_h, bins=authority_grid)
            current_cohorts = [
                row
                for row in trajectory_cohort_rows
                if math.isclose(float(row["target_time_h"]), time_h, rel_tol=0.0, abs_tol=1.0e-12)
                and bool(row["active"])
            ]
            cohort_radii = np.asarray(
                [row["radius_m"] for row in current_cohorts], dtype=np.float64
            )
            cohort_weights = np.asarray(
                [row["weight_m3"] for row in current_cohorts], dtype=np.float64
            )
            eulerian_radii, eulerian_weights = _legacy_smooth_psd(
                legacy_root, time_h, bins=authority_grid
            )
            values = {
                "N_m0_m3": snapshot.N_m0_m3,
                "M0_m3": snapshot.M0_m3,
                "M1_m2": snapshot.M1_m2,
                "M2_m": snapshot.M2_m,
                "M3_dimensionless": snapshot.M3_dimensionless,
                "Rmean_m": snapshot.Rmean_m,
                "Rmean3_m3": snapshot.Rmean3_m3,
                "Sv_m_inv": snapshot.Sv_m_inv,
                "f_beta": snapshot.f_beta,
                "matrix_xB": snapshot.matrix_xB,
                "cumulative_dissolution_inventory_mol_m3": snapshot.cumulative_dissolution_inventory_mol_m3,
            }
            for metric, cohort_value in values.items():
                eulerian_key = metric
                if metric == "cumulative_dissolution_inventory_mol_m3":
                    eulerian_key = "cumulative_lower_boundary_beta_inventory_mol_m3"
                eulerian_value = float(eulerian[eulerian_key])
                rows.append(
                    {
                        "record_type": "cohort_vs_eulerian",
                        "cohort_count": count,
                        "target_time_h": time_h,
                        "metric": metric,
                        "cohort_value": cohort_value,
                        "eulerian_authority_value": eulerian_value,
                        "relative_error": _relative_error(cohort_value, eulerian_value),
                        "cohort_inventory_relative_residual": snapshot.inventory_relative_residual,
                    }
                )
            rows.append(
                {
                    "record_type": "PSD_WASSERSTEIN",
                    "cohort_count": count,
                    "target_time_h": time_h,
                    "metric": "W1_number_m",
                    "cohort_value": discrete_wasserstein_distance(
                        cohort_radii, cohort_weights, eulerian_radii, eulerian_weights
                    ),
                    "eulerian_authority_value": 0.0,
                    "relative_error": "",
                    "cohort_inventory_relative_residual": snapshot.inventory_relative_residual,
                }
            )
            rows.append(
                {
                    "record_type": "PSD_WASSERSTEIN",
                    "cohort_count": count,
                    "target_time_h": time_h,
                    "metric": "W1_volume_m",
                    "cohort_value": discrete_wasserstein_distance(
                        cohort_radii,
                        cohort_weights * cohort_radii**3,
                        eulerian_radii,
                        eulerian_weights * eulerian_radii**3,
                    ),
                    "eulerian_authority_value": 0.0,
                    "relative_error": "",
                    "cohort_inventory_relative_residual": snapshot.inventory_relative_residual,
                }
            )
    refinement_metrics = {
        "N_m0_m3", "M0_m3", "M1_m2", "M2_m", "M3_dimensionless",
        "Rmean_m", "Rmean3_m3", "Sv_m_inv", "f_beta", "matrix_xB",
    }
    for coarse_count, fine_count in ((400, 800), (800, 1600)):
        for time_h in SMOOTH_TIMES_H:
            coarse = trajectories[coarse_count][time_h]
            fine = trajectories[fine_count][time_h]
            for metric in refinement_metrics:
                rows.append(
                    {
                        "record_type": "cohort_quadrature_refinement",
                        "coarse_cohort_count": coarse_count,
                        "fine_cohort_count": fine_count,
                        "target_time_h": time_h,
                        "metric": metric,
                        "cohort_value": float(getattr(fine, metric)),
                        "coarse_cohort_value": float(getattr(coarse, metric)),
                        "relative_error": _relative_error(
                            float(getattr(coarse, metric)), float(getattr(fine, metric))
                        ),
                        "threshold": 0.02,
                    }
                )
    authority_rows = [row for row in rows if row["record_type"] == "cohort_vs_eulerian" and row["cohort_count"] == 1600]
    continuous_metrics = {"N_m0_m3", "M0_m3", "M1_m2", "M2_m", "M3_dimensionless", "Rmean_m", "Rmean3_m3", "Sv_m_inv", "f_beta", "matrix_xB"}
    refinement_rows = [
        row
        for row in rows
        if row["record_type"] == "cohort_quadrature_refinement"
        and row["coarse_cohort_count"] == 800
        and row["fine_cohort_count"] == 1600
    ]
    quadrature_pass = all(float(row["relative_error"]) <= 0.02 for row in refinement_rows)
    crosscheck_pass = all(
        float(row["relative_error"]) <= 0.02
        for row in authority_rows
        if row["metric"] in continuous_metrics
    ) and quadrature_pass and all(
        snapshot.inventory_relative_residual <= 1.0e-10 for snapshot in trajectories[1600].values()
    )
    directions = {}
    direction_metrics = {
        "N": ("N_m0_m3", "N_m0_m3"),
        "M0": ("M0_m3", "M0_m3"),
        "M1": ("M1_m2", "M1_m2"),
        "M2": ("M2_m", "M2_m"),
        "M3": ("M3_dimensionless", "M3_dimensionless"),
        "Rmean": ("Rmean_m", "Rmean_m"),
        "Rmean3": ("Rmean3_m3", "Rmean3_m3"),
        "Sv": ("Sv_m_inv", "Sv_m_inv"),
        "f_beta": ("f_beta", "f_beta"),
        "matrix_xB": ("matrix_xB", "matrix_xB"),
        "cumulative_dissolution_inventory": (
            "cumulative_dissolution_inventory_mol_m3",
            "cumulative_lower_boundary_beta_inventory_mol_m3",
        ),
    }
    for metric, (cohort_key, eulerian_key) in direction_metrics.items():
        cohort_direction = _sign(
            float(getattr(trajectories[1600][48.0], cohort_key))
            - float(getattr(trajectories[1600][0.0], cohort_key))
        )
        start = _legacy_smooth_row(legacy_root, 0.0, bins=authority_grid)
        end = _legacy_smooth_row(legacy_root, 48.0, bins=authority_grid)
        directions[metric] = {
            "cohort": cohort_direction,
            "eulerian": _sign(float(end[eulerian_key]) - float(start[eulerian_key])),
        }
    direction_pass = all(value["cohort"] == value["eulerian"] for value in directions.values())
    status = "PASS_COHORT_EULERIAN_SMOOTH_CROSSCHECK" if crosscheck_pass and direction_pass else "FAIL_COHORT_EULERIAN_PHYSICS_MISMATCH"
    result = {
        "schema_version": "KWN_COHORT_EULERIAN_SMOOTH_CROSSCHECK_V1",
        "status": status,
        "cohort_authority_count": 1600,
        "eulerian_authority_grid": authority_grid,
        "crosscheck_pass": crosscheck_pass,
        "quadrature_refinement_pass": quadrature_pass,
        "direction_pass": direction_pass,
        "directions": directions,
        "cohort_quadrature": metadata,
        "representation_note": "deterministic log-radius quadrature cohorts; no radius bins, projection, splitting, or smoothing after t=0",
    }
    _write_csv(OUTPUT_ROOT / "cohort_smooth_crosscheck.csv", rows)
    _write_json(OUTPUT_ROOT / "cohort_eulerian_smooth_crosscheck.json", result)
    _write_report(
        "06_cohort_eulerian_smooth_crosscheck.md",
        "Cohort–Eulerian smooth-PSD crosscheck",
        f"Status: `{status}`.  The 1600-node deterministic quadrature is the cohort comparison authority. "
        f"800→1600 quadrature refinement: `{quadrature_pass}`; global direction agreement: `{direction_pass}`; "
        f"2% continuous-metric gate: `{crosscheck_pass}`.\n\n"
        "All three cohort representations use the same frozen beta growth law, curvature equilibrium, D(T), "
        "molar volumes, total inventory and matrix inverse as Eulerian KWN.  Differences are therefore reported as "
        "representation differences, not physical retuning.",
    )
    return result


def _event_diagnosis(
    legacy_root: Path,
    exact_snapshots: Mapping[float, CohortSnapshot],
    exact_context: Mapping[str, Any],
) -> dict[str, Any]:
    cohort_rows = list(exact_context["cohort_rows"])
    event_rows = [
        row
        for row in cohort_rows
        if math.isclose(float(row["target_time_h"]), 48.0, abs_tol=1.0e-12)
    ]
    _write_csv(OUTPUT_ROOT / "six_particle_cohort_trajectory.csv", [
        _snapshot_row(snapshot, target_time_h=time_h, cohort_count=6)
        for time_h, snapshot in exact_snapshots.items()
    ] + cohort_rows)
    event_table = []
    for row in event_rows:
        event_table.append(
            {
                "initial_id": row["initial_id"],
                "initial_radius_m": row["initial_radius_m"],
                "weight_m3": row["weight_m3"],
                "initial_growth_sign": row["initial_growth_sign"],
                "dissolution_time_h": "" if row["dissolution_time_s"] is None else float(row["dissolution_time_s"]) / 3600.0,
                "radius_before_event_m": row["radius_before_event_m"],
                "returned_inventory_mol_m3": row["returned_inventory_mol_m3"],
                "active_at_48h": row["active"],
                "post_event_ledger_relative_residual": row["post_event_ledger_relative_residual"],
            }
        )
    _write_csv(OUTPUT_ROOT / "six_particle_event_table.csv", event_table)
    overlay_rows: list[dict[str, Any]] = []
    for bins in (100, 200, 400, 800, 1600, 3200):
        events_path = legacy_root / "runs" / f"ladder_fixture_{bins}_uniform" / "class_events.csv"
        by_time = _read_csv(events_path)
        for time_h in EXACT_TIMES_H:
            matches = [
                row for row in by_time
                if math.isclose(float(row["target_time_h"]), time_h, abs_tol=1.0e-12)
            ]
            if matches:
                for row in matches:
                    overlay_rows.append(
                        {
                            "scope": "NON_AUTHORITY_EULERIAN_DISCRETE_FIXTURE_DIAGNOSTIC",
                            "eulerian_bins": bins,
                            "target_time_h": time_h,
                            "class_label": row["class_label"],
                            "eulerian_effective_radius_m": row["effective_radius_m"],
                            "eulerian_survival_weight_fraction": row["survival_weight_fraction"],
                            "eulerian_first_lower_boundary_crossing_h": row["first_lower_boundary_crossing_time_h"],
                            "eulerian_returned_inventory_mol_m3": row["cumulative_returned_matrix_beta_inventory_mol_m3"],
                            "cohort_active_count": exact_snapshots[time_h].active_cohort_count,
                            "cohort_N_m0_m3": exact_snapshots[time_h].N_m0_m3,
                            "cohort_Rmean3_m3": exact_snapshots[time_h].Rmean3_m3,
                            "cohort_Sv_m_inv": exact_snapshots[time_h].Sv_m_inv,
                            "cohort_f_beta": exact_snapshots[time_h].f_beta,
                            "cohort_matrix_xB": exact_snapshots[time_h].matrix_xB,
                            "eulerian_available": True,
                        }
                    )
            else:
                overlay_rows.append(
                    {
                        "scope": "NON_AUTHORITY_EULERIAN_DISCRETE_FIXTURE_DIAGNOSTIC",
                        "eulerian_bins": bins,
                        "target_time_h": time_h,
                        "class_label": "",
                        "cohort_active_count": exact_snapshots[time_h].active_cohort_count,
                        "cohort_N_m0_m3": exact_snapshots[time_h].N_m0_m3,
                        "cohort_Rmean3_m3": exact_snapshots[time_h].Rmean3_m3,
                        "cohort_Sv_m_inv": exact_snapshots[time_h].Sv_m_inv,
                        "cohort_f_beta": exact_snapshots[time_h].f_beta,
                        "cohort_matrix_xB": exact_snapshots[time_h].matrix_xB,
                        "eulerian_available": False,
                    }
                )
    _write_csv(OUTPUT_ROOT / "eulerian_vs_cohort_event_overlay.csv", overlay_rows)
    final_by_grid = {}
    for bins in (1600, 3200):
        rows = _read_csv(legacy_root / "runs" / f"ladder_fixture_{bins}_uniform" / "class_events.csv")
        final_by_grid[bins] = {
            row["class_label"]: row
            for row in rows
            if math.isclose(float(row["target_time_h"]), 48.0, abs_tol=1.0e-12)
        }
    r105_1600 = final_by_grid[1600]["R10.5nm"]
    r105_3200 = final_by_grid[3200]["R10.5nm"]
    result = {
        "primary_discrete_event_root_cause": "IMPLICIT_UPWIND_NUMERICAL_DIFFUSION",
        "event_responsible_for_1600_3200_gap": "R10.5nm lower-bound numerical tail survival/leakage",
        "cohort_exact_events": event_table,
        "r10p5_final_survival": {
            "1600": float(r105_1600["survival_weight_fraction"]),
            "3200": float(r105_3200["survival_weight_fraction"]),
            "1600_first_lower_crossing_h": float(r105_1600["first_lower_boundary_crossing_time_h"]),
            "3200_first_lower_crossing_h": float(r105_3200["first_lower_boundary_crossing_time_h"]),
        },
        "interpretation": {
            "event_timing": "Eulerian lower-face tails start at grid-dependent accepted endpoints before the sharp cohort reaches its exact Rmin event.",
            "N_Rmean3_Sv": "At 48 h the only appreciable Eulerian surviving class is R10.5nm; its grid-dependent survival fraction therefore dominates N, Rmean3 and Sv differences.",
            "f_beta_matrix_xB": "M3-weighted beta inventory is far less sensitive to a low-radius number tail, and conservative lower-bound return keeps f_beta and matrix_xB inventory closed and comparatively convergent.",
            "envelope": "Exact cohort events are a no-bin characteristic reference; legacy Eulerian sequences remain non-authority diagnostics rather than an authority envelope or a replacement comparison.",
        },
    }
    _write_json(OUTPUT_ROOT / "discrete_event_diagnosis.json", result)
    _write_report(
        "05_discrete_event_diagnosis.md",
        "Discrete-event diagnosis",
        "The old Eulerian six-particle sequence is explicitly `NON-AUTHORITY EULERIAN DISCRETE-FIXTURE DIAGNOSTIC`.\n\n"
        f"Primary root cause: `{result['primary_discrete_event_root_cause']}`.  The 1600→3200 N gap is dominated at 48 h by "
        f"`{result['event_responsible_for_1600_3200_gap']}`: its survival is "
        f"{float(r105_1600['survival_weight_fraction']):.6g} (1600) versus "
        f"{float(r105_3200['survival_weight_fraction']):.6g} (3200).  The lower-tail first crossing is "
        f"{float(r105_1600['first_lower_boundary_crossing_time_h']):.6g} h versus "
        f"{float(r105_3200['first_lower_boundary_crossing_time_h']):.6g} h.\n\n"
        "The sharp cohorts record exact event times and return the Rmin remainder to the matrix.  Eulerian upwind tails "
        "advance/lag different class leakage paths under refinement; they do not license changing the P5 threshold.",
    )
    return result


def _pf_component_metrics(rows: Sequence[Mapping[str, str]], box_volume_m3: float) -> dict[str, Any]:
    unresolved = any(
        row[key] == "True"
        for row in rows
        for key in ("unresolved_merge", "unresolved_split", "unresolved_new_component")
    )
    if unresolved:
        raise RuntimeError("PF component identity is unresolved; comparison must fail closed")
    radii = np.asarray([float(row["equivalent_radius_nm"]) * 1.0e-9 for row in rows], dtype=np.float64)
    if radii.size == 0:
        return {"count": 0, "N_m0_m3": 0.0, "M0_m3": 0.0, "M1_m2": 0.0, "M2_m": 0.0, "M3_dimensionless": 0.0, "Rmean_m": 0.0, "Rmean3_m3": 0.0, "Sv_m_inv": 0.0, "radii_m": radii}
    weights = np.full(radii.size, 1.0 / box_volume_m3, dtype=np.float64)
    moments = [float(np.sum(weights * radii**order)) for order in range(4)]
    return {
        "count": int(radii.size),
        "N_m0_m3": moments[0],
        "M0_m3": moments[0],
        "M1_m2": moments[1],
        "M2_m": moments[2],
        "M3_dimensionless": moments[3],
        "Rmean_m": moments[1] / moments[0],
        "Rmean3_m3": moments[3] / moments[0],
        "Sv_m_inv": 4.0 * math.pi * moments[2],
        "radii_m": radii,
    }


def _pf_matrix_xb(row: Mapping[str, str], contract: Any, fixture: Any) -> float:
    """Convert the frozen PF matrix bucket to the same alpha composition basis."""

    beta_fraction = float(row["beta_volume_fraction"])
    matrix_fraction = 1.0 - beta_fraction
    if matrix_fraction <= 0.0:
        raise RuntimeError("frozen PF row has no matrix fraction")
    return (
        float(row["Q_B_matrix_mol"])
        * contract.vm_alpha_m3_mol
        / (matrix_fraction * fixture.box_volume_m3)
    )


def _validated_pf_component_metrics(
    pf_row: Mapping[str, str], component_rows: Sequence[Mapping[str, str]], box_volume_m3: float
) -> dict[str, Any]:
    """Fail closed if frozen component extraction disagrees with its checkpoint metadata."""

    expected_count = int(pf_row["connected_particle_count"])
    if len(component_rows) != expected_count:
        raise RuntimeError(
            "frozen PF component extraction count mismatch: "
            f"step={pf_row['step']} expected={expected_count} observed={len(component_rows)}"
        )
    metrics = _pf_component_metrics(component_rows, box_volume_m3)
    if metrics["count"] != expected_count:
        raise RuntimeError("PF component moment count does not match frozen checkpoint metadata")
    return metrics


def _pf_reference_steps(trajectory: Mapping[int, Mapping[str, str]]) -> list[tuple[float, dict[str, str]]]:
    rows = sorted((float(row["time_h"]), dict(row)) for row in trajectory.values())
    selected = []
    for requested in PF_REQUESTED_TIMES_H:
        candidates = sorted(rows, key=lambda item: abs(item[0] - requested))
        actual, row = candidates[0]
        if abs(actual - requested) > 2.0e-3:
            raise RuntimeError(f"frozen PF lacks an accepted checkpoint for requested {requested} h")
        selected.append((requested, row))
    return selected


def _cohort_pf_comparison(
    authority: Mapping[str, Any], numeric: Mapping[str, Any], crosscheck: Mapping[str, Any]
) -> dict[str, Any]:
    gate_pass = (
        authority.get("status") == "PASS_EULERIAN_SMOOTH_POPULATION_AUTHORITY"
        and numeric.get("status") == "PASS_DISCRETE_COHORT_NUMERICS"
        and crosscheck.get("status") == "PASS_COHORT_EULERIAN_SMOOTH_CROSSCHECK"
    )
    if not gate_pass:
        result = {
            "status": "BLOCKED_PREREQUISITE_GATE",
            "authority_status": authority.get("status"),
            "numeric_status": numeric.get("status"),
            "crosscheck_status": crosscheck.get("status"),
            "no_cuda_rerun": True,
        }
        _write_csv(OUTPUT_ROOT / "beta_only_cohort_pf_comparison.csv", [result])
        _write_json(OUTPUT_ROOT / "beta_only_cohort_pf_comparison.json", result)
        _write_report(
            "07_beta_only_cohort_pf_comparison.md",
            "Beta-only cohort–PF comparison",
            "`BLOCKED_PREREQUISITE_GATE`: PF was not compared because an upstream numerical gate did not pass.  No CUDA rerun was performed.",
        )
        return result
    audit = _load_radius_audit_module()
    solver, identity, contract, fixture = _new_fixture_cohort_solver(audit, rtol=1.0e-10, method="Radau")
    if identity["status"] != "PASS_BETA_ONLY_INITIAL_STATE_IDENTITY":
        result = {"status": "FAIL_BETA_ONLY_INITIAL_STATE_IDENTITY", "identity": identity, "no_cuda_rerun": True}
        _write_csv(OUTPUT_ROOT / "beta_only_cohort_pf_comparison.csv", [result])
        _write_json(OUTPUT_ROOT / "beta_only_cohort_pf_comparison.json", result)
        _write_report("07_beta_only_cohort_pf_comparison.md", "Beta-only cohort–PF comparison", "`FAIL_BETA_ONLY_INITIAL_STATE_IDENTITY`.")
        return result
    trajectory, components = _frozen_pf_rows()
    reference_rows = _pf_reference_steps(trajectory)
    cohort_times = [float(row["time_h"]) for _, row in reference_rows]
    snapshots, _, cohort_rows = _run_cohort_trajectory(solver, cohort_times, cohort_count=6)
    mappings = identity["cohort_to_pf_particle_id"]
    rows: list[dict[str, Any]] = []
    pf_presence: dict[str, list[tuple[float, bool, float | None]]] = defaultdict(list)
    pf_metrics_by_time: dict[float, dict[str, Any]] = {}
    pf_matrix_xb_by_time: dict[float, float] = {}
    for requested_h, pf_row in reference_rows:
        step = int(pf_row["step"])
        pf_metrics = _validated_pf_component_metrics(
            pf_row, components[step], fixture.box_volume_m3
        )
        pf_matrix_xb = _pf_matrix_xb(pf_row, contract, fixture)
        actual_time_h = float(pf_row["time_h"])
        pf_metrics_by_time[actual_time_h] = pf_metrics
        pf_matrix_xb_by_time[actual_time_h] = pf_matrix_xb
        snapshot = snapshots[float(pf_row["time_h"])]
        for metric, cohort_value, pf_value in (
            ("N_m0_m3", snapshot.N_m0_m3, pf_metrics["N_m0_m3"]),
            ("Rmean_m", snapshot.Rmean_m, pf_metrics["Rmean_m"]),
            ("Rmean3_m3", snapshot.Rmean3_m3, pf_metrics["Rmean3_m3"]),
            ("invN_m3", 0.0 if snapshot.N_m0_m3 == 0.0 else 1.0 / snapshot.N_m0_m3, 0.0 if pf_metrics["N_m0_m3"] == 0.0 else 1.0 / pf_metrics["N_m0_m3"]),
            ("Sv_m_inv", snapshot.Sv_m_inv, pf_metrics["Sv_m_inv"]),
            ("f_beta", snapshot.f_beta, float(pf_row["beta_volume_fraction"])),
            ("matrix_xB", snapshot.matrix_xB, pf_matrix_xb),
            ("M3_dimensionless", snapshot.M3_dimensionless, pf_metrics["M3_dimensionless"]),
        ):
            rows.append(
                {
                    "record_type": "global",
                    "requested_time_h": requested_h,
                    "pf_actual_time_h": float(pf_row["time_h"]),
                    "cohort_time_h": snapshot.time_s / 3600.0,
                    "metric": metric,
                    "cohort_value": cohort_value,
                    "pf_value": pf_value,
                    "relative_difference": _relative_error(cohort_value, pf_value),
                    "pf_component_count": pf_metrics["count"],
                    "cohort_active_count": snapshot.active_cohort_count,
                    "cohort_inventory_relative_residual": snapshot.inventory_relative_residual,
                    "pf_four_bucket_relative_residual": float(pf_row["four_bucket_relative_residual"]),
                }
            )
        pf_by_id = {str(row["particle_id"]): row for row in components[step]}
        cohort_by_id = {str(row["initial_id"]): row for row in cohort_rows if math.isclose(float(row["time_h"]), float(pf_row["time_h"]), abs_tol=1.0e-12)}
        for cohort_id, pf_id in mappings.items():
            component = pf_by_id.get(str(pf_id))
            cohort = cohort_by_id[cohort_id]
            present = component is not None
            radius = None if component is None else float(component["equivalent_radius_nm"]) * 1.0e-9
            pf_presence[cohort_id].append((float(pf_row["time_h"]), present, radius))
            rows.append(
                {
                    "record_type": "identity_class",
                    "requested_time_h": requested_h,
                    "pf_actual_time_h": float(pf_row["time_h"]),
                    "cohort_time_h": snapshot.time_s / 3600.0,
                    "cohort_initial_id": cohort_id,
                    "pf_particle_id": pf_id,
                    "initial_radius_m": cohort["initial_radius_m"],
                    "cohort_active": cohort["active"],
                    "cohort_radius_m": cohort["radius_m"],
                    "pf_present": present,
                    "pf_radius_m": "" if radius is None else radius,
                    "pf_unresolved_identity": False,
                }
            )
    initial = snapshots[cohort_times[0]]
    final = snapshots[cohort_times[-1]]
    first_pf = reference_rows[0][1]
    last_pf = reference_rows[-1][1]
    first_metrics = _validated_pf_component_metrics(
        first_pf, components[int(first_pf["step"])], fixture.box_volume_m3
    )
    final_metrics = _validated_pf_component_metrics(
        last_pf, components[int(last_pf["step"])], fixture.box_volume_m3
    )
    cohort_rows_final = {row["initial_id"]: row for row in solver.cohort_rows()}
    initial_radius = {row["initial_id"]: float(row["initial_radius_m"]) for row in solver.cohort_rows()}
    smallest = min(initial_radius.values())
    largest = max(initial_radius.values())
    small_ids = [identifier for identifier, radius in initial_radius.items() if math.isclose(radius, smallest, abs_tol=2.0e-13)]
    large_ids = [identifier for identifier, radius in initial_radius.items() if math.isclose(radius, largest, abs_tol=2.0e-13)]
    smallest_cohort_dissolves = all(not bool(cohort_rows_final[identifier]["active"]) for identifier in small_ids)
    smallest_pf_dissolves = all(not pf_presence[identifier][-1][1] for identifier in small_ids)
    largest_cohort_grows = any(
        bool(cohort_rows_final[identifier]["active"])
        and float(cohort_rows_final[identifier]["radius_m"]) > initial_radius[identifier]
        for identifier in large_ids
    )
    largest_pf_grows = any(
        pf_presence[identifier][-1][1]
        and float(pf_presence[identifier][-1][2]) > initial_radius[identifier]
        for identifier in large_ids
    )
    largest_cohort_grows_or_survives = any(
        bool(cohort_rows_final[identifier]["active"]) for identifier in large_ids
    )
    largest_pf_grows_or_survives = any(
        bool(pf_presence[identifier][-1][1]) for identifier in large_ids
    )
    first_pf_matrix_xb = _pf_matrix_xb(first_pf, contract, fixture)
    final_pf_matrix_xb = _pf_matrix_xb(last_pf, contract, fixture)
    directions = {
        "N": {"cohort": _sign(final.N_m0_m3 - initial.N_m0_m3), "pf": _sign(final_metrics["N_m0_m3"] - first_metrics["N_m0_m3"])},
        "Rmean": {"cohort": _sign(final.Rmean_m - initial.Rmean_m), "pf": _sign(final_metrics["Rmean_m"] - first_metrics["Rmean_m"])},
        "Sv": {"cohort": _sign(final.Sv_m_inv - initial.Sv_m_inv), "pf": _sign(final_metrics["Sv_m_inv"] - first_metrics["Sv_m_inv"])},
        "f_beta": {"cohort": _sign(final.f_beta - initial.f_beta), "pf": _sign(float(last_pf["beta_volume_fraction"]) - float(first_pf["beta_volume_fraction"]))},
        "matrix_xB": {
            "cohort": _sign(final.matrix_xB - initial.matrix_xB),
            "pf": _sign(final_pf_matrix_xb - first_pf_matrix_xb),
        },
    }
    global_direction_match = all(item["cohort"] == item["pf"] for item in directions.values())
    direction_pass = (
        smallest_cohort_dissolves
        and smallest_pf_dissolves
        and largest_cohort_grows_or_survives
        and largest_pf_grows_or_survives
        and final.N_m0_m3 <= initial.N_m0_m3
        and final.Rmean_m >= initial.Rmean_m
        and final.Sv_m_inv <= initial.Sv_m_inv
        and global_direction_match
    )
    cohort_event_times = {
        row["initial_id"]: row["dissolution_time_s"]
        for row in solver.cohort_rows()
    }
    pf_event_brackets: dict[str, dict[str, float | None]] = {}
    for identifier, history in pf_presence.items():
        present = [time for time, state, _ in history if state]
        absent = [time for time, state, _ in history if not state]
        pf_event_brackets[identifier] = {
            "last_present_h": max(present) if present else None,
            "first_absent_h": min(absent) if absent else None,
        }
    slope_cohort = np.polyfit(
        np.asarray(cohort_times) * 3600.0,
        np.asarray([snapshots[time].Rmean3_m3 for time in cohort_times]),
        1,
    )[0]
    slope_pf = np.polyfit(
        np.asarray([float(row["time_h"]) for _, row in reference_rows]) * 3600.0,
        np.asarray([
            pf_metrics_by_time[float(row["time_h"])]["Rmean3_m3"]
            for _, row in reference_rows
        ]),
        1,
    )[0]
    individual_gap = any(
        bool(cohort_rows_final[identifier]["active"]) != bool(pf_presence[identifier][-1][1])
        for identifier in initial_radius
    )
    event_ratio_rows: list[dict[str, Any]] = []
    checkpoint_bracketed_event = False
    for identifier, cohort_event_s in cohort_event_times.items():
        bracket = pf_event_brackets[identifier]
        cohort_event_h = None if cohort_event_s is None else float(cohort_event_s) / 3600.0
        last_present_h = bracket["last_present_h"]
        first_absent_h = bracket["first_absent_h"]
        comparison = "BOTH_SURVIVE"
        ratio_to_last_present = None
        ratio_to_first_absent = None
        if cohort_event_h is not None and first_absent_h is not None:
            comparison = "PF_CHECKPOINT_BRACKET_ONLY"
            checkpoint_bracketed_event = True
            if last_present_h is not None and last_present_h > 0.0:
                ratio_to_last_present = cohort_event_h / last_present_h
            if first_absent_h > 0.0:
                ratio_to_first_absent = cohort_event_h / first_absent_h
        elif cohort_event_h is None and first_absent_h is not None:
            comparison = "PF_DISSOLVES_COHORT_SURVIVES"
        elif cohort_event_h is not None:
            comparison = "COHORT_DISSOLVES_PF_SURVIVES"
        event_row = {
            "record_type": "dissolution_event_timescale",
            "cohort_initial_id": identifier,
            "cohort_event_time_h": cohort_event_h,
            "pf_last_present_h": last_present_h,
            "pf_first_absent_h": first_absent_h,
            "cohort_over_pf_last_present_ratio": ratio_to_last_present,
            "cohort_over_pf_first_absent_ratio": ratio_to_first_absent,
            "comparison": comparison,
        }
        event_ratio_rows.append(event_row)
        rows.append(event_row)

    duration_s = float(cohort_times[-1]) * 3600.0

    def endpoint_time_scale(value0: float, value1: float) -> float | None:
        change = abs(value1 - value0)
        if change == 0.0 or value0 == 0.0:
            return None
        return duration_s * abs(value0) / change

    sv_tau_cohort = endpoint_time_scale(initial.Sv_m_inv, final.Sv_m_inv)
    sv_tau_pf = endpoint_time_scale(first_metrics["Sv_m_inv"], final_metrics["Sv_m_inv"])
    matrix_tau_cohort = endpoint_time_scale(initial.matrix_xB, final.matrix_xB)
    matrix_tau_pf = endpoint_time_scale(first_pf_matrix_xb, final_pf_matrix_xb)
    timescale_metrics = {
        "Rmean3_slope_ratio_cohort_over_pf": (
            float(slope_cohort / slope_pf) if slope_pf != 0.0 else None
        ),
        "Sv_loss_timescale_ratio_cohort_over_pf": (
            None if sv_tau_cohort is None or sv_tau_pf is None else sv_tau_cohort / sv_tau_pf
        ),
        "matrix_relaxation_timescale_ratio_cohort_over_pf": (
            None
            if matrix_tau_cohort is None or matrix_tau_pf is None
            else matrix_tau_cohort / matrix_tau_pf
        ),
    }
    for metric, value in timescale_metrics.items():
        rows.append(
            {
                "record_type": "global_timescale",
                "metric": metric,
                "cohort_value": value,
                "pf_value": 1.0,
                "definition": "endpoint-defined characteristic time or least-squares slope; no fitted D_scale or pass threshold",
            }
        )
    endpoint_ratios: dict[str, float | None] = {}
    for metric, cohort_value, pf_value in (
        ("N_m0_m3", final.N_m0_m3, final_metrics["N_m0_m3"]),
        ("Rmean_m", final.Rmean_m, final_metrics["Rmean_m"]),
        ("Rmean3_m3", final.Rmean3_m3, final_metrics["Rmean3_m3"]),
        ("Sv_m_inv", final.Sv_m_inv, final_metrics["Sv_m_inv"]),
        ("f_beta", final.f_beta, float(last_pf["beta_volume_fraction"])),
        ("matrix_xB", final.matrix_xB, final_pf_matrix_xb),
        ("M3_dimensionless", final.M3_dimensionless, final_metrics["M3_dimensionless"]),
    ):
        ratio = None if pf_value == 0.0 else float(cohort_value / pf_value)
        endpoint_ratios[metric] = ratio
        rows.append(
            {
                "record_type": "endpoint_ratio_48h",
                "metric": metric,
                "cohort_value": cohort_value,
                "pf_value": pf_value,
                "cohort_over_pf_ratio": ratio,
            }
        )
    if not direction_pass:
        timescale_status = "TIMESCALE_NOT_COMPARABLE_DISCRETE_EVENTS"
    elif checkpoint_bracketed_event:
        timescale_status = "TIMESCALE_NOT_COMPARABLE_DISCRETE_EVENTS"
    elif individual_gap:
        timescale_status = "CONDITIONAL_MEAN_FIELD_TIMESCALE_GAP"
    else:
        timescale_status = "PASS_BETA_ONLY_TIMESCALE"
    mean_field_gap = (
        "CONDITIONAL_SPATIAL_ELASTIC_MEAN_FIELD_GAP"
        if direction_pass and (individual_gap or timescale_status != "PASS_BETA_ONLY_TIMESCALE")
        else "NONE"
        if direction_pass
        else "NOT_ASSESSED_AFTER_DIRECTION_FAILURE"
    )
    result = {
        "status": "PASS_BETA_ONLY_DIRECTION" if direction_pass else "FAIL_BETA_ONLY_DIRECTION",
        "initial_identity": identity,
        "directions": directions,
        "global_direction_match": global_direction_match,
        "smallest_class": {"cohort_dissolves": smallest_cohort_dissolves, "pf_dissolves": smallest_pf_dissolves},
        "largest_class": {
            "cohort_grows": largest_cohort_grows,
            "pf_grows": largest_pf_grows,
            "cohort_grows_or_survives": largest_cohort_grows_or_survives,
            "pf_grows_or_survives": largest_pf_grows_or_survives,
        },
        "dissolution_event_times_s": cohort_event_times,
        "pf_checkpoint_bounded_event_brackets_h": pf_event_brackets,
        "dissolution_time_ratio_records": event_ratio_rows,
        "timescale_metrics": timescale_metrics,
        "endpoint_ratios_48h": endpoint_ratios,
        "timescale_status": timescale_status,
        "mean_field_spatial_elastic_gap": mean_field_gap,
        "pf_reference": {
            "case": "A",
            "reason": "A and B have audited trajectory-identical physical fields; A is the baseline without handoff auxiliary metadata.",
            "trajectory_sha256": _sha256_file(FROZEN_PF_ROOT / "cuda_ae_trajectories.csv"),
            "component_history_sha256": _sha256_file(FROZEN_PF_ROOT / "cuda_component_history.csv"),
            "no_cuda_rerun": True,
        },
    }
    _write_csv(OUTPUT_ROOT / "beta_only_cohort_pf_comparison.csv", rows)
    _write_json(OUTPUT_ROOT / "beta_only_cohort_pf_comparison.json", result)
    _write_report(
        "07_beta_only_cohort_pf_comparison.md",
        "Beta-only cohort–PF same-contract comparison",
        f"Initial identity: `{identity['status']}`.  Direction gate: `{result['status']}`.  "
        f"Timescale classification: `{timescale_status}`.  Mean-field/spatial-elastic classification: `{mean_field_gap}`.\n\n"
        "Case A is reused because audited A/B physical trajectories are identical; Case A contains no handoff auxiliary metadata. "
        "PF points are accepted checkpoint times (recorded in the CSV), never interpolated, and CUDA was not rerun.",
    )
    return result


def _model_boundary_report(
    authority: Mapping[str, Any], numeric: Mapping[str, Any], crosscheck: Mapping[str, Any],
    pf: Mapping[str, Any], cuda_evidence: Mapping[str, Any]
) -> None:
    _write_report(
        "08_model_role_boundary.md",
        "Model role boundary",
        "Eulerian KWN is qualified here only for a smooth, high-number-density population under the frozen validation contract. "
        "It is not authority for the exact six-particle delta-like PF fixture, whose finite dissolution events are represented by the no-bin cohort comparator.\n\n"
        "The cohort model is spherical-equivalent and mean-field: it has no spatial competition, coherent elasticity, diffuse-interface topology or particle–particle elastic interaction. "
        "PF retains those effects.  Any individual fate difference with compatible global direction remains a conditional mean-field/spatial-elastic gap.\n\n"
        "Historical boundaries remain `HISTORICAL_AS_RUN_AUTHORITY_UNRECOVERED` and `HISTORICAL_12H_PSD_NOT_RECOVERED`.  This is a validation-contract six-particle code-level beta-only comparison, not historical 400³ validation, all-case validation, experimental GP-nucleation validation, GP release, GP→beta, or online coupling.\n\n"
        f"Current gates: inherited CUDA `{cuda_evidence.get('status')}`, Eulerian `{authority.get('status')}`, cohort numerics `{numeric.get('status')}`, smooth crosscheck `{crosscheck.get('status')}`, PF direction `{pf.get('status')}`.",
    )


def _final_status(
    baseline: Mapping[str, Any], authority: Mapping[str, Any], numeric: Mapping[str, Any],
    crosscheck: Mapping[str, Any], pf: Mapping[str, Any], cuda_evidence: Mapping[str, Any]
) -> tuple[str, str]:
    if baseline.get("status") != "PASS_RADIUS_GRID_BASELINE_REPRODUCTION":
        return "FAIL_BASELINE_REPRODUCTION", "LOCAL_GP_RELEASE_NOT_AUTHORIZED"
    if authority.get("status") != "PASS_EULERIAN_SMOOTH_POPULATION_AUTHORITY":
        return "FAIL_EULERIAN_SMOOTH_POPULATION_AUTHORITY", "LOCAL_GP_RELEASE_NOT_AUTHORIZED"
    if numeric.get("status") != "PASS_DISCRETE_COHORT_NUMERICS":
        return "FAIL_DISCRETE_COHORT_NUMERICS", "LOCAL_GP_RELEASE_NOT_AUTHORIZED"
    if crosscheck.get("status") != "PASS_COHORT_EULERIAN_SMOOTH_CROSSCHECK":
        return "FAIL_COHORT_EULERIAN_PHYSICS_MISMATCH", "LOCAL_GP_RELEASE_NOT_AUTHORIZED"
    if cuda_evidence.get("status") != "PASS_INHERITED_CUDA_AE_EVIDENCE":
        return "FAIL_CUDA_EVIDENCE_REUSE", "LOCAL_GP_RELEASE_NOT_AUTHORIZED"
    if pf.get("status") == "FAIL_BETA_ONLY_INITIAL_STATE_IDENTITY":
        return "FAIL_BETA_ONLY_INITIAL_STATE_IDENTITY", "LOCAL_GP_RELEASE_NOT_AUTHORIZED"
    if pf.get("status") != "PASS_BETA_ONLY_DIRECTION":
        return "FAIL_BETA_ONLY_DIRECTION", "LOCAL_GP_RELEASE_NOT_AUTHORIZED"
    local_gp = "ELIGIBLE_FOR_SEPARATELY_GATED_LOCAL_GP_RELEASE_PROTOTYPE"
    if pf.get("mean_field_spatial_elastic_gap") == "CONDITIONAL_SPATIAL_ELASTIC_MEAN_FIELD_GAP" or pf.get("timescale_status") != "PASS_BETA_ONLY_TIMESCALE":
        return "PASS_KWN_NUMERICS_CONDITIONAL_MEAN_FIELD_GAP", local_gp
    return "PASS_KWN_PF_ONE_WAY_STORAGE_COUPLING_V1", local_gp


def _analysis_provenance(legacy_root: Path, *, command: str) -> dict[str, Any]:
    source_files = [
        ROOT / "scripts" / "run_kwn_discrete_cohort_comparison_v1.py",
        ROOT / "scripts" / "run_kwn_radius_grid_convergence_v1.py",
        ROOT / "src" / "kwn_mvp" / "cohort_solver.py",
        ROOT / "src" / "kwn_mvp" / "growth.py",
        ROOT / "src" / "kwn_mvp" / "solver.py",
        ROOT / "src" / "kwn_mvp" / "thermo_adapter.py",
        ROOT / "tests" / "kwn" / "test_cohort_solver.py",
    ]
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, text=True, stdout=subprocess.PIPE, check=True
    ).stdout
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True, stdout=subprocess.PIPE, check=True
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stdout=subprocess.PIPE, check=True
    ).stdout.strip()
    return {
        "schema_version": "KWN_DISCRETE_COHORT_ANALYSIS_PROVENANCE_V1",
        "git_branch": branch,
        "git_head_at_launch": head,
        "working_tree_porcelain_at_analysis": status,
        "command": command,
        "legacy_report_commit": LEGACY_REPORT_COMMIT,
        "legacy_radius_grid_output_root": str(legacy_root),
        "legacy_input_sha256": {
            str(path.relative_to(legacy_root)): _sha256_file(path) for path in _legacy_input_paths(legacy_root)
        },
        "source_file_sha256": {str(path.relative_to(ROOT)): _sha256_file(path) for path in source_files},
        "frozen_cuda_pf": {
            "cuda_rerun": False,
            "trajectory_sha256": _sha256_file(FROZEN_PF_ROOT / "cuda_ae_trajectories.csv"),
            "component_history_sha256": _sha256_file(FROZEN_PF_ROOT / "cuda_component_history.csv"),
            "binary_provenance_sha256": _sha256_file(FROZEN_PF_ROOT / "binary_provenance.json"),
        },
        "pf_source_modified": False,
        "physical_retuning": False,
        "prohibitions_retained": [
            "legacy_six_particle_eulerian_P5_not_relaxed",
            "no_6400_or_12800_bins",
            "no_PF_source_modification",
            "no_CUDA_A_to_E_rerun",
            "no_validation_contract_D_gamma_or_thermodynamic_change",
            "no_D_scale_fit",
            "no_GP_release_or_online_coupling",
        ],
    }


def _write_final_report(
    *,
    baseline: Mapping[str, Any],
    authority: Mapping[str, Any],
    numeric: Mapping[str, Any],
    crosscheck: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    pf: Mapping[str, Any],
    cuda_evidence: Mapping[str, Any],
    top_status: str,
    local_gp: str,
) -> None:
    findings = [
        "Legacy six-particle Eulerian P5 remains FAIL; the 2% threshold was not relaxed.",
        f"Eulerian KWN authority is smooth-population grid {authority.get('authority_grid')} only.",
        "Exact six-particle evolution uses no-bin event-aware cohorts and strict algebraic inventory closure.",
        f"Frozen CUDA A--E evidence reuse is {cuda_evidence.get('status')} with no CUDA rerun.",
        f"Frozen Case A PF direction result is {pf.get('status')} with timescale {pf.get('timescale_status', 'NOT_RUN')}.",
        f"The discrete event diagnosis identifies {diagnosis.get('event_responsible_for_1600_3200_gap', 'NOT_RUN')}.",
    ]
    body = (
        f"Top-level status: `{top_status}`.\n\n"
        f"- Baseline: `{baseline.get('status')}`\n"
        f"- Smooth Eulerian authority: `{authority.get('status')}` at grid `{authority.get('authority_grid')}`\n"
        f"- Cohort numerics: `{numeric.get('status')}`\n"
        f"- Cohort/Eulerian smooth crosscheck: `{crosscheck.get('status')}`\n"
        f"- Inherited CUDA A--E evidence: `{cuda_evidence.get('status')}`\n"
        f"- Beta-only PF direction: `{pf.get('status')}`\n"
        f"- Local GP release: `{local_gp}`\n\n"
        "## Findings\n\n"
        + "\n".join(f"- {item}" for item in findings)
        + "\n\n## Boundaries\n\n"
        "CUDA A–E is frozen evidence reused without a rerun. PF source was not modified. No physical retuning, double counting, "
        "mass drift, GP release, GP→beta or online coupling was introduced. Historical authority remains unrecovered.\n\n"
        "## Next action\n\n"
        + (
            "If a GP prototype is desired, open a separately gated local-GP-release task; this task does not start it."
            if local_gp.startswith("ELIGIBLE")
            else "Do not start local GP release; resolve the failed gate above in a separate, scoped task."
        )
    )
    _write_report("09_final_acceptance_report.md", "Final acceptance", body)


def _write_reproduction_commands(legacy_root: Path) -> None:
    _write_report(
        "10_reproduction_commands.md",
        "Reproduction commands",
        "From this worktree:\n\n"
        "```bash\n"
        "PYTHONPATH=src python3 -m unittest tests.kwn.test_cohort_solver\n"
        f"python3 scripts/run_kwn_discrete_cohort_comparison_v1.py all --radius-grid-output-root {legacy_root}\n"
        "```\n\n"
        "The second command reads the previous radius-grid artifacts only; it does not rerun CUDA/PF A–E or change their sources.",
    )


def run_all(legacy_root: Path, *, command: str) -> int:
    figures = OUTPUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "README.md").write_text(
        "# Plot-ready data\n\n"
        "No standalone figure is substituted for the numerical evidence.  Use "
        "`../eulerian_vs_cohort_event_overlay.csv` for the event overlay and "
        "`../beta_only_cohort_pf_comparison.csv` for the frozen-PF comparison.\n",
        encoding="utf-8",
    )
    _require_human_gate()
    baseline = _baseline_reproduction(legacy_root)
    cuda_evidence = _inherited_cuda_evidence()
    if baseline["status"] != "PASS_RADIUS_GRID_BASELINE_REPRODUCTION":
        empty = {"status": "NOT_RUN_AFTER_BASELINE_FAILURE"}
        _model_boundary_report(empty, empty, empty, empty, cuda_evidence)
        _write_final_report(
            baseline=baseline,
            authority=empty,
            numeric=empty,
            crosscheck=empty,
            diagnosis=empty,
            pf=empty,
            cuda_evidence=cuda_evidence,
            top_status="FAIL_BASELINE_REPRODUCTION",
            local_gp="LOCAL_GP_RELEASE_NOT_AUTHORIZED",
        )
        _write_reproduction_commands(legacy_root)
        _write_json(OUTPUT_ROOT / "analysis_provenance.json", _analysis_provenance(legacy_root, command=command))
        return 2
    authority = _authority_requalification(legacy_root)
    numeric, exact_snapshots, _, exact_context = _cohort_numerical_qualification()
    crosscheck = _cohort_eulerian_smooth_crosscheck(legacy_root, authority)
    diagnosis = _event_diagnosis(legacy_root, exact_snapshots, exact_context)
    pf = _cohort_pf_comparison(authority, numeric, crosscheck)
    _model_boundary_report(authority, numeric, crosscheck, pf, cuda_evidence)
    top_status, local_gp = _final_status(
        baseline, authority, numeric, crosscheck, pf, cuda_evidence
    )
    _write_final_report(
        baseline=baseline,
        authority=authority,
        numeric=numeric,
        crosscheck=crosscheck,
        diagnosis=diagnosis,
        pf=pf,
        cuda_evidence=cuda_evidence,
        top_status=top_status,
        local_gp=local_gp,
    )
    _write_reproduction_commands(legacy_root)
    provenance = _analysis_provenance(legacy_root, command=command)
    provenance["top_status"] = top_status
    provenance["inherited_cuda_evidence"] = cuda_evidence
    _write_json(OUTPUT_ROOT / "analysis_provenance.json", provenance)
    print(json.dumps({"status": top_status, "local_gp_release": local_gp}, sort_keys=True))
    return 0 if top_status.startswith("PASS_") else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all",))
    parser.add_argument(
        "--radius-grid-output-root",
        type=Path,
        required=True,
        help="immutable outputs/kwn_radius_grid_convergence_v1 directory from the audit worktree",
    )
    arguments = parser.parse_args()
    return run_all(arguments.radius_grid_output_root.resolve(), command=" ".join(sys.argv))


if __name__ == "__main__":
    raise SystemExit(main())
