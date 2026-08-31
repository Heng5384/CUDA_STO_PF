#!/usr/bin/env python3
"""Run prescribed-source and finite effective-CNT GP feasibility trajectories."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.composition_mapping import xb_to_ag_at_fraction  # noqa: E402
from kwn_mvp.config import load_config  # noqa: E402
from kwn_mvp.diagnostics import solver_observables, write_csv  # noqa: E402
from kwn_mvp.solver import KWNSolver, SolverConfig  # noqa: E402
from kwn_mvp.units import hours_to_seconds  # noqa: E402


TARGET_HOURS = (0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 16.0, 24.0, 48.0)


def _run_snapshots(config: Mapping[str, Any], label: str, parameter_set: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run one effective KWN state and return trajectory and PSD heatmap rows."""

    solver = KWNSolver(SolverConfig.from_mapping(config))
    trajectory: List[Dict[str, Any]] = []
    heatmap: List[Dict[str, Any]] = []
    for target_h in TARGET_HOURS:
        solver.run_to_time(hours_to_seconds(target_h))
        row: Dict[str, Any] = {
            "scenario": label,
            "parameter_set": parameter_set,
            **solver_observables(solver),
            "matrix_Ag_at_fraction": xb_to_ag_at_fraction(solver.matrix_xb),
        }
        trajectory.append(row)
        for population_name in ("g", "beta"):
            population = solver.population(population_name)
            for left, right, density in zip(
                population.grid.edges_m[:-1],
                population.grid.edges_m[1:],
                population.number_density_per_m4,
            ):
                heatmap.append(
                    {
                        "scenario": label,
                        "parameter_set": parameter_set,
                        "time_h": target_h,
                        "population": population_name,
                        "radius_left_m": left,
                        "radius_right_m": right,
                        "number_density_per_m4": density,
                    }
                )
    return trajectory, heatmap


def _latin_hypercube(seed: int, count: int, bounds: Sequence[Tuple[float, float]], log_scale: Sequence[bool]) -> np.ndarray:
    """Return a deterministic Latin hypercube without an external sampler dependency."""

    rng = np.random.default_rng(seed)
    values = np.empty((count, len(bounds)), dtype=np.float64)
    for column, ((lower, upper), is_log) in enumerate(zip(bounds, log_scale)):
        points = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        rng.shuffle(points)
        if is_log:
            values[:, column] = 10.0 ** (math.log10(lower) + points * (math.log10(upper) - math.log10(lower)))
        else:
            values[:, column] = lower + points * (upper - lower)
    return values


def _classification(rows: Sequence[Mapping[str, Any]]) -> str:
    """Assign the declared soft-constraint feasibility label for one CNT trajectory."""

    at_six = next(row for row in rows if row["time_h"] == 6.0)
    at_48 = next(row for row in rows if row["time_h"] == 48.0)
    peak = max(rows, key=lambda row: row["g_number_density_m3"])
    conditions = (
        at_six["g_number_density_m3"] >= 1.0e23,
        0.0058 <= at_six["matrix_Ag_at_fraction"] <= 0.0066,
        2.0 <= peak["time_h"] <= 10.0,
        at_48["g_number_density_m3"] < at_six["g_number_density_m3"],
        at_six["g_mean_radius_nm"] <= 10.0,
        max(row["inventory_relative_residual"] for row in rows) <= 1.0e-10,
    )
    return "FEASIBLE" if all(conditions) else "INFEASIBLE_SOFT_CONSTRAINTS"


def _effective_config(base: Mapping[str, Any], values: Sequence[float]) -> Dict[str, Any]:
    """Make one explicit effective-CNT parameter set from a bounded LHS sample."""

    gamma, x_b_g, xeq_g, site_density, attachment, d_scale, elastic_penalty = values
    config = copy.deepcopy(dict(base))
    g = config["populations"]["g"]
    g["gamma_j_m2"] = float(gamma)
    g["xB"] = float(x_b_g)
    g["xeq_infinity"] = float(xeq_g)
    g["diffusivity_m2_s"] = 8.073255954803963e-18 * float(d_scale)
    g["elastic_penalty_j_m3"] = float(elastic_penalty)
    g["nucleation"]["site_density_m3"] = float(site_density)
    g["nucleation"]["attachment_prefactor_s_inv"] = float(attachment)
    return config


def main() -> int:
    """Generate all requested effective-GP trajectories and feasibility outputs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--prescribed-config", default=str(ROOT / "configs/kwn/gp_prescribed_source_mvp.yaml"))
    parser.add_argument("--cnt-config", default=str(ROOT / "configs/kwn/gp_effective_cnt_sweep.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/kwn_pf_mvp_v1"))
    parser.add_argument("--report", default=str(ROOT / "reports/kwn_pf_mvp_v1/04_gp_feasibility.md"))
    args = parser.parse_args()
    prescribed_document = load_config(args.prescribed_config)
    cnt_document = load_config(args.cnt_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories, heatmap = _run_snapshots(prescribed_document.data, "prescribed_source", 0)
    prescribed_six = next(row for row in trajectories if row["time_h"] == 6.0)
    prescribed_48 = next(row for row in trajectories if row["time_h"] == 48.0)
    prescribed_peak = max(trajectories, key=lambda row: row["g_number_density_m3"])
    prescribed_g = prescribed_document.data["populations"]["g"]
    sweep_rows: List[Dict[str, Any]] = [
        {
            "parameter_set": 0,
            "mode": "prescribed_source",
            "status": "RUNNABLE_NONPREDICTIVE",
            "gamma_g_J_m2": prescribed_g["gamma_j_m2"],
            "xB_g": prescribed_g["xB"],
            "xeq_g_infinity": prescribed_g["xeq_infinity"],
            "site_density_g_m3": "",
            "attachment_prefactor_s_inv": "",
            "D_scale_g": "",
            "elastic_penalty_J_m3": prescribed_g["elastic_penalty_j_m3"],
            "N_g_6h_m3": prescribed_six["g_number_density_m3"],
            "N_g_48h_m3": prescribed_48["g_number_density_m3"],
            "matrix_Ag_6h": prescribed_six["matrix_Ag_at_fraction"],
            "peak_time_h": prescribed_peak["time_h"],
            "peak_N_g_m3": prescribed_peak["g_number_density_m3"],
            "failure_reason": "PRESCRIBED_SOURCE_IS_NOT_A_GP_NUCLEATION_PREDICTION",
            "config_sha256": prescribed_document.sha256,
        }
    ]
    count = int(cnt_document.data["simulation"].get("sweep_sets", 64))
    samples = _latin_hypercube(
        int(cnt_document.data["simulation"]["random_seed"]),
        count,
        bounds=(
            (0.001, 0.02),
            (0.01, 0.12),
            (0.0045, 0.0072),
            (1.0e26, 3.0e29),
            (1.0e-5, 3.0e1),
            (0.001, 0.03),
            (0.0, 1.0e7),
        ),
        log_scale=(True, False, False, True, True, True, False),
    )
    for index, values in enumerate(samples, start=1):
        parameter_config = _effective_config(cnt_document.data, values)
        try:
            run_rows, run_heatmap = _run_snapshots(parameter_config, "effective_cnt", index)
            trajectories.extend(run_rows)
            heatmap.extend(run_heatmap)
            status = _classification(run_rows)
            at_six = next(row for row in run_rows if row["time_h"] == 6.0)
            at_48 = next(row for row in run_rows if row["time_h"] == 48.0)
            peak = max(run_rows, key=lambda row: row["g_number_density_m3"])
            failure = ""
        except Exception as exc:  # Each failed effective-prior set is a retained result, not discarded.
            status = "INFEASIBLE_SOLVER_OR_INVENTORY"
            at_six = at_48 = peak = {}
            failure = f"{type(exc).__name__}: {exc}"
        sweep_rows.append(
            {
                "parameter_set": index,
                "mode": "effective_cnt",
                "status": status,
                "gamma_g_J_m2": values[0],
                "xB_g": values[1],
                "xeq_g_infinity": values[2],
                "site_density_g_m3": values[3],
                "attachment_prefactor_s_inv": values[4],
                "D_scale_g": values[5],
                "elastic_penalty_J_m3": values[6],
                "N_g_6h_m3": at_six.get("g_number_density_m3", float("nan")),
                "N_g_48h_m3": at_48.get("g_number_density_m3", float("nan")),
                "matrix_Ag_6h": at_six.get("matrix_Ag_at_fraction", float("nan")),
                "peak_time_h": peak.get("time_h", float("nan")),
                "peak_N_g_m3": peak.get("g_number_density_m3", float("nan")),
                "failure_reason": failure,
                "config_sha256": cnt_document.sha256,
            }
        )
    write_csv(output_dir / "gp_trajectories.csv", trajectories)
    write_csv(output_dir / "gp_psd_heatmap.csv", [row for row in heatmap if row["population"] == "g"])
    write_csv(output_dir / "beta_psd_heatmap.csv", [row for row in heatmap if row["population"] == "beta"])
    write_csv(output_dir / "gp_parameter_sweep.csv", sweep_rows)
    feasible = sum(row.get("status") == "FEASIBLE" for row in sweep_rows)
    infeasible_soft = sum(row.get("status") == "INFEASIBLE_SOFT_CONSTRAINTS" for row in sweep_rows)
    infeasible_solver = sum(row.get("status") == "INFEASIBLE_SOLVER_OR_INVENTORY" for row in sweep_rows)
    if feasible:
        conclusion = "EFFECTIVE_CNT_SOFT_CONSTRAINT_FEASIBLE_NOT_IDENTIFIED"
    elif infeasible_soft:
        conclusion = "EFFECTIVE_CNT_CONDITIONALLY_FEASIBLE"
    else:
        conclusion = "STANDARD_CNT_CANNOT_REPRODUCE_REQUIRED_POPULATION"
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Effective GP-KWN feasibility\n\n"
        f"Status: `{conclusion}`\n\n"
        "The prescribed-source route is numerically active solely to test two-population mass exchange and handoff. "
        "`PRESCRIBED_SOURCE_IS_NOT_A_GP_NUCLEATION_PREDICTION`.\n\n"
        f"The effective-CNT sweep retained all {count} deterministic Latin-hypercube parameter sets: {feasible} met all soft constraints, "
        f"{infeasible_soft} missed at least one soft constraint, and {infeasible_solver} hit an explicit solver/inventory boundary. "
        "`xB_g`, gamma_g, site density, attachment, diffusivity scale, and elastic penalty remain effective/exploratory "
        "parameters rather than identified GP thermodynamics. A soft-constraint hit is not a calibrated physical GP nucleation prediction. "
        "Yu 2024 is a holdout plausibility envelope, not a joint fit.\n\n"
        "Outputs: `gp_parameter_sweep.csv`, `gp_trajectories.csv`, and population PSD heatmaps.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
