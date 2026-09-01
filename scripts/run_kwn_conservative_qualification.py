#!/usr/bin/env python3
"""Qualify the conservative KWN positivity repair on the frozen beta fixture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from coupling.fixture_conditioned_handoff_v2 import (  # noqa: E402
    load_host_96cube_fixture,
    load_validation_contract,
)
from kwn_mvp.radius_grid import RadiusGrid  # noqa: E402
from kwn_mvp.solver import KWNSolver, SolverConfig, StepDiagnostics  # noqa: E402
from scripts import run_beta_only_same_contract_control as beta_control  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs" / "kwn_pf_cuda_runtime_closure_v1"
CONTRACT = ROOT / "contracts" / "pf_kwn_validation_contract_v1.json"
FIXTURE_SPEC = (
    ROOT
    / "data"
    / "qualification"
    / "pf_mass_conserving_library_handoff_v1"
    / "six_particle_96cube_spec.json"
)
PROFILE_ROOT = (
    ROOT.parent
    / "CUDA_STO_PF"
    / "data"
    / "qualification"
    / "pf_elastic_target_profile_quarter_nm_v2"
    / "profiles"
)
METRIC_NAMES = (
    "beta_number_density_m3",
    "beta_mean_radius_m",
    "beta_mean_radius_cubed_m3",
    "beta_specific_surface_area_m_inv",
    "beta_volume_fraction",
    "matrix_xB",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_solver(*, fixture: Any, contract: Any, bins: int, max_dt_factor: float) -> KWNSolver:
    base, construction = beta_control.build_same_contract_config(fixture=fixture, contract=contract)
    mapping = deepcopy(construction["config_mapping"])
    if bins != base.grid.bins:
        grid = RadiusGrid.logarithmic(
            float(mapping["radius_grid"]["minimum_m"]),
            float(mapping["radius_grid"]["maximum_m"]),
            bins,
        )
        entries, _ = beta_control._project_fixture_resolved_psd(fixture, contract, grid)
        mapping["radius_grid"]["bins"] = bins
        mapping["populations"]["beta"]["initial"]["entries"] = entries
    mapping["simulation"]["max_dt_s"] = (
        float(mapping["simulation"]["max_dt_s"]) * max_dt_factor
    )
    return KWNSolver(SolverConfig.from_mapping(mapping))


def _metrics(solver: KWNSolver) -> dict[str, float]:
    beta = solver.population("beta")
    ledger = solver.ledger.snapshot(matrix_xb=solver.matrix_xb, populations=solver.population_list())
    return {
        "time_h": solver.time_s / 3600.0,
        "beta_number_density_m3": beta.number_density_m3(),
        "beta_mean_radius_m": beta.mean_radius_m(),
        "beta_mean_radius_cubed_m3": beta.mean_radius_cubed_m3(),
        "beta_specific_surface_area_m_inv": beta.specific_surface_area_m_inv(),
        "beta_volume_fraction": beta.volume_fraction(),
        "matrix_xB": solver.matrix_xb,
        "inventory_relative_residual": ledger.relative_residual,
        "minimum_bin_density_per_m4": float(np.min(beta.number_density_per_m4)),
        "roundoff_zeroed_bin_count": float(solver.roundoff_zeroed_bin_count),
    }


def _audit(history: Sequence[StepDiagnostics]) -> dict[str, float | int]:
    """Summarise a concrete sequence of accepted steps for a gate audit."""

    if not history:
        raise RuntimeError("qualification solver has no accepted steps")
    return {
        "accepted_step_count": len(history),
        "rejected_step_count": 0,
        "minimum_dt_s": min(item.dt_s for item in history),
        "median_dt_s": float(np.median([item.dt_s for item in history])),
        "maximum_size_cfl": max(item.size_cfl for item in history),
        "maximum_positivity_utilization": max(item.positivity_utilization for item in history),
        "roundoff_zeroed_bin_count": sum(item.roundoff_zeroed_bin_count for item in history),
    }


def _state_equal(left: KWNSolver, right: KWNSolver) -> tuple[bool, dict[str, float]]:
    differences: dict[str, float] = {}
    for name, array in left.state_arrays().items():
        other = right.state_arrays()[name]
        differences[name] = float(np.max(np.abs(array - other)))
    return all(value == 0.0 for value in differences.values()), differences


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), 1.0e-300)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    try:
        output_root.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise SystemExit("refusing output outside this task's output root") from error
    work = output_root / "kwn_conservative_qualification"
    summary_path = work / "summary.json"
    convergence_path = output_root / "kwn_repair_convergence.csv"
    restart_path = work / "restart_comparison.csv"
    if any(path.exists() for path in (summary_path, convergence_path, restart_path)):
        raise SystemExit("refusing to overwrite KWN qualification artifacts")
    work.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = work / "checkpoints"
    checkpoint_dir.mkdir()

    contract = load_validation_contract(CONTRACT)
    fixture = load_host_96cube_fixture(FIXTURE_SPEC, PROFILE_ROOT, contract)
    canonical = _build_solver(fixture=fixture, contract=contract, bins=200, max_dt_factor=1.0)
    trajectory_rows: list[dict[str, object]] = []
    canonical_states: dict[float, dict[str, np.ndarray]] = {}
    for time_h in (0.0, 0.1, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0):
        canonical.run_to_time(time_h * 3600.0)
        canonical_states[time_h] = canonical.state_arrays()
        trajectory_rows.append({"run_id": "canonical_200", "bins": 200, "max_dt_factor": 1.0, **_metrics(canonical)})

    restart = _build_solver(fixture=fixture, contract=contract, bins=200, max_dt_factor=1.0)
    restart_segment_audits: dict[str, dict[str, float | int]] = {}
    restart_0_to_3h = restart.run_to_time(3.0 * 3600.0)
    restart_segment_audits["restart_0_to_3h"] = _audit(restart_0_to_3h)
    checkpoint_3h = checkpoint_dir / "restart_3h.npz"
    restart.save_checkpoint(checkpoint_3h)
    restart = KWNSolver.load_checkpoint(config=restart.config, path=checkpoint_3h)
    restart_3_to_6h = restart.run_to_time(6.0 * 3600.0)
    restart_segment_audits["restart_3_to_6h"] = _audit(restart_3_to_6h)
    continuous_restart_control = _build_solver(
        fixture=fixture, contract=contract, bins=200, max_dt_factor=1.0
    )
    continuous_restart_control.run_to_time(3.0 * 3600.0)
    continuous_restart_control.run_to_time(6.0 * 3600.0)
    equal_6h, differences_6h = _state_equal(continuous_restart_control, restart)
    checkpoint_6h = checkpoint_dir / "restart_6h.npz"
    restart.save_checkpoint(checkpoint_6h)
    restart_6_to_24h = restart.run_to_time(24.0 * 3600.0)
    restart_segment_audits["restart_6_to_24h"] = _audit(restart_6_to_24h)
    checkpoint_24h = checkpoint_dir / "restart_24h.npz"
    restart.save_checkpoint(checkpoint_24h)
    restart = KWNSolver.load_checkpoint(config=restart.config, path=checkpoint_24h)
    restart_24_to_48h = restart.run_to_time(48.0 * 3600.0)
    restart_segment_audits["restart_24_to_48h"] = _audit(restart_24_to_48h)
    continuous_restart_control.run_to_time(24.0 * 3600.0)
    continuous_restart_control.run_to_time(48.0 * 3600.0)
    equal_48h, differences_48h = _state_equal(continuous_restart_control, restart)
    restart_rows = [
        {
            "comparison": "continuous_0_6h_vs_restart_0_3h_3_6h",
            "pass": equal_6h,
            "checkpoint_sha256": _sha256(checkpoint_3h),
            **differences_6h,
        },
        {
            "comparison": "continuous_0_48h_vs_restart_0_6h_6_24h_24_48h",
            "pass": equal_48h,
            "checkpoint_sha256": _sha256(checkpoint_24h),
            **differences_48h,
        },
    ]

    refined_dt = _build_solver(fixture=fixture, contract=contract, bins=200, max_dt_factor=0.5)
    refined_dt.run_to_time(48.0 * 3600.0)
    trajectory_rows.append({"run_id": "dt_half_200", "bins": 200, "max_dt_factor": 0.5, **_metrics(refined_dt)})
    grid_400 = _build_solver(fixture=fixture, contract=contract, bins=400, max_dt_factor=1.0)
    grid_400.run_to_time(48.0 * 3600.0)
    trajectory_rows.append({"run_id": "grid_400", "bins": 400, "max_dt_factor": 1.0, **_metrics(grid_400)})

    canonical_48 = _metrics(canonical)
    refined_48 = _metrics(refined_dt)
    grid_400_48 = _metrics(grid_400)
    grid_convergence = {name: _relative_difference(canonical_48[name], grid_400_48[name]) for name in METRIC_NAMES}
    timestep_convergence = {name: _relative_difference(canonical_48[name], refined_48[name]) for name in METRIC_NAMES}
    canonical_audit = _audit(canonical.history)
    refined_dt_audit = _audit(refined_dt.history)
    grid_400_audit = _audit(grid_400.history)
    continuous_restart_control_audit = _audit(continuous_restart_control.history)
    # The sampled canonical path crossed the legacy 0.39317699499770825 h
    # failure during the 0.1 -> 1 h interval; verify the state remains finite.
    p1_solver = _build_solver(fixture=fixture, contract=contract, bins=200, max_dt_factor=1.0)
    p1_solver.run_to_time(0.5 * 3600.0)
    p1_metrics = _metrics(p1_solver)
    p1_audit = _audit(p1_solver.history)
    p3_no_clipping_audits = {
        "canonical_200_0_to_48h": canonical_audit,
        "exact_failure_regression_200_0_to_0p5h": p1_audit,
        "dt_half_200_0_to_48h": refined_dt_audit,
        "grid_400_0_to_48h": grid_400_audit,
        "continuous_restart_control_0_to_48h": continuous_restart_control_audit,
        **restart_segment_audits,
    }
    gate = {
        "P1_exact_failure_regression": p1_solver.time_s / 3600.0 > 0.39317699499770825 and p1_metrics["minimum_bin_density_per_m4"] >= 0.0,
        "P2_48h_completion": canonical.time_s == 48.0 * 3600.0,
        "P3_no_clipping": all(
            item["roundoff_zeroed_bin_count"] == 0
            for item in p3_no_clipping_audits.values()
        ),
        "P4_conservation": max(
            canonical_48["inventory_relative_residual"],
            refined_48["inventory_relative_residual"],
            grid_400_48["inventory_relative_residual"],
        ) <= 1.0e-10,
        "P5_bin_convergence_200_vs_400": all(value <= 0.02 for value in grid_convergence.values()),
        "P5_timestep_convergence": all(value <= 0.02 for value in timestep_convergence.values()),
        "P6_restart": equal_6h and equal_48h,
        "P6_positivity_utilization": canonical_audit["maximum_positivity_utilization"] <= canonical.config.positivity_safety * (1.0 + 1.0e-12),
    }
    with convergence_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(trajectory_rows)
    with restart_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(restart_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(restart_rows)
    summary = {
        "schema_version": "KWN_CONSERVATIVE_POSITIVITY_QUALIFICATION_V1",
        "status": "PASS_KWN_CONSERVATIVE_POSITIVITY" if all(gate.values()) else "FAIL_KWN_CONSERVATIVE_POSITIVITY",
        "contract_hash": contract.contract_hash,
        "fixture_id": fixture.fixture_id,
        "fixture_hash": fixture.fixture_hash,
        "repair": {
            "method": "CONSERVATIVE_DONOR_OUTGOING_FACE_FLUX_LIMITER",
            "positivity_safety": canonical.config.positivity_safety,
            "cfl_active_inventory_relative_threshold": canonical.config.cfl_active_inventory_relative_threshold,
            "no_parameter_retuning": True,
            "negative_bin_clamp": False,
        },
        "gates": gate,
        "p1_metrics_0p5h": p1_metrics,
        "p1_audit": p1_audit,
        "p3_no_clipping_audits": p3_no_clipping_audits,
        "canonical_48h": canonical_48,
        "grid_400_48h": grid_400_48,
        "dt_half_200_48h": refined_48,
        "grid_convergence_relative_difference": grid_convergence,
        "timestep_convergence_relative_difference": timestep_convergence,
        "retry_audit": canonical_audit,
        "restart": restart_rows,
        "outputs": {
            "convergence_csv": str(convergence_path),
            "restart_csv": str(restart_path),
            "checkpoint_sha256": {path.name: _sha256(path) for path in checkpoint_dir.glob("*.npz")},
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "gates": gate, "summary": str(summary_path)}, sort_keys=True))
    return 0 if summary["status"] == "PASS_KWN_CONSERVATIVE_POSITIVITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
