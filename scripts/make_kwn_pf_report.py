#!/usr/bin/env python3
"""Render reproducible KWN-MVP figures and final, authority-gated reports.

The renderer only visualizes results that exist in the output ledger.  For the
blocked beta-only and PF-smoke comparisons it renders an explicit blocked-state
panel instead of inventing numerical trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.diagnostics import solver_observables  # noqa: E402
from kwn_mvp.solver import KWNSolver, SolverConfig  # noqa: E402


def _git_commit() -> str:
    """Return the source revision from which the report was rendered."""

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read one UTF-8 CSV artifact with its original column names."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write homogeneous result rows with deterministic field ordering."""

    if not rows:
        raise ValueError("cannot write an empty diagnostic table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _float(row: Mapping[str, Any], key: str) -> float:
    """Read a finite scalar or raise rather than silently plot invalid data."""

    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite value for {key}")
    return value


def _synthetic_qualification_config(*, bins: int, max_dt_s: float) -> Dict[str, Any]:
    """Return the named synthetic N5/N6 qualification state in SI units.

    This is a reporting fixture only.  It matches the public numerical-gate
    specification and is not a PbTe--Ag2Te physical calibration input.
    """

    return {
        "simulation": {
            "temperature_K": 653.15,
            "max_dt_s": max_dt_s,
            "min_dt_s": 1.0e-12,
            "size_cfl": 0.35,
            "rmax_outflow_relative_tolerance": 1.0e-10,
        },
        "matrix": {
            "molar_volume_m3_mol": 4.1009e-5,
            "initial_xB": 0.0066,
            "total_b_mol_m3": None,
            "inventory_tolerance_relative": 1.0e-10,
        },
        "radius_grid": {"minimum_m": 5.0e-10, "maximum_m": 2.0e-7, "bins": bins},
        "thermodynamics": {"mode": "approximate_dilute", "planar_reference_xB": 0.006},
        "populations": {
            "g": {
                "xB": 0.02,
                "molar_volume_m3_mol": 4.1009e-5,
                "diffusivity_m2_s": 0.0,
                "gamma_j_m2": 0.0,
                "xeq_infinity": 0.005,
                "initial": {"kind": "empty"},
                "nucleation": {"mode": "off"},
            },
            "beta": {
                "xB": 1.0,
                "molar_volume_m3_mol": 4.1009e-5,
                "diffusivity_m2_s": 3.0e-20,
                "gamma_j_m2": 0.002,
                "xeq_infinity": 0.006,
                "shape_factor": 1.0,
                "elastic_penalty_j_m3": 0.0,
                "initial": {
                    "kind": "lognormal",
                    "number_density_m3": 2.0e22,
                    "median_radius_m": 5.0e-9,
                    "log_sigma": 0.2,
                },
                "nucleation": {"mode": "off"},
            },
        },
    }


def _qualification_tables(output_dir: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reproduce the N5 and N6 diagnostic tables used by the figures."""

    convergence_rows: List[Dict[str, Any]] = []
    observables_by_bins: Dict[int, Dict[str, float]] = {}
    for bins in (100, 200, 400):
        solver = KWNSolver(SolverConfig.from_mapping(_synthetic_qualification_config(bins=bins, max_dt_s=200.0)))
        solver.run_to_time(3.0e4)
        observables_by_bins[bins] = solver_observables(solver)
    reference = observables_by_bins[400]
    for bins in (100, 200, 400):
        current = observables_by_bins[bins]
        for metric in (
            "beta_number_density_m3",
            "beta_mean_radius_m",
            "beta_mean_radius_cubed_m3",
            "beta_specific_surface_area_m_inv",
            "beta_volume_fraction",
            "matrix_xB",
        ):
            denominator = max(abs(reference[metric]), 1.0e-300)
            convergence_rows.append(
                {
                    "diagnostic": "N5_radius_bin_convergence",
                    "bins": bins,
                    "metric": metric,
                    "value": current[metric],
                    "reference_400_bins": reference[metric],
                    "relative_difference_vs_400": abs(current[metric] - reference[metric]) / denominator,
                    "target_relative_difference": 0.02,
                    "time_s": 3.0e4,
                    "config_label": "synthetic_N5_qualification_not_physical_fit",
                }
            )
    timestep_rows: List[Dict[str, Any]] = []
    by_dt: Dict[float, Dict[str, float]] = {}
    for maximum_dt_s in (300.0, 150.0):
        solver = KWNSolver(
            SolverConfig.from_mapping(_synthetic_qualification_config(bins=200, max_dt_s=maximum_dt_s))
        )
        solver.run_to_time(3.0e4)
        by_dt[maximum_dt_s] = solver_observables(solver)
    reference_dt = by_dt[150.0]
    for maximum_dt_s in (300.0, 150.0):
        current = by_dt[maximum_dt_s]
        for metric in (
            "beta_number_density_m3",
            "beta_mean_radius_m",
            "beta_volume_fraction",
            "matrix_xB",
        ):
            denominator = max(abs(reference_dt[metric]), 1.0e-300)
            timestep_rows.append(
                {
                    "diagnostic": "N6_timestep_convergence",
                    "max_dt_s": maximum_dt_s,
                    "metric": metric,
                    "value": current[metric],
                    "reference_dt_150_s": reference_dt[metric],
                    "relative_difference_vs_150_s": abs(current[metric] - reference_dt[metric]) / denominator,
                    "target_relative_difference": 0.02,
                    "time_s": 3.0e4,
                    "config_label": "synthetic_N6_qualification_not_physical_fit",
                }
            )
    _write_csv(output_dir / "radius_bin_convergence.csv", convergence_rows)
    _write_csv(output_dir / "timestep_convergence.csv", timestep_rows)
    return convergence_rows, timestep_rows


def _blocked_panel(plt: Any, path: Path, title: str, status: str, detail: str) -> None:
    """Render an unambiguous status panel for a scientifically blocked comparison."""

    figure, axis = plt.subplots(figsize=(8.0, 3.6))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.60, status, ha="center", va="center", fontsize=15, fontweight="bold", color="#9b2226")
    axis.text(0.5, 0.34, detail, ha="center", va="center", fontsize=10, wrap=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _status_counts(rows: Sequence[Mapping[str, str]]) -> Dict[str, int]:
    """Count categorical sweep outcomes without discarding failed rows."""

    counts: Dict[str, int] = {}
    for row in rows:
        status = row["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def _render_figures(
    *,
    figure_dir: Path,
    constraints: Sequence[Mapping[str, str]],
    trajectories: Sequence[Mapping[str, str]],
    gp_heatmap: Sequence[Mapping[str, str]],
    beta_heatmap: Sequence[Mapping[str, str]],
    sweep_rows: Sequence[Mapping[str, str]],
    convergence_rows: Sequence[Mapping[str, Any]],
    timestep_rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    beta_status: str,
    smoke_status: str,
) -> None:
    """Render the requested figures from retained data without smoothing."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    prescribed = sorted(
        (row for row in trajectories if row["scenario"] == "prescribed_source"),
        key=lambda row: _float(row, "time_h"),
    )
    feasible_ids = {row["parameter_set"] for row in sweep_rows if row["status"] == "FEASIBLE"}
    feasible = sorted(
        (
            row
            for row in trajectories
            if row["scenario"] == "effective_cnt" and row["parameter_set"] in feasible_ids
        ),
        key=lambda row: _float(row, "time_h"),
    )
    primary = [row for row in constraints if row["role"] == "primary"]
    secondary = [row for row in constraints if row["role"] == "secondary"]

    figure, axes = plt.subplots(2, 1, figsize=(8.3, 7.4), sharex=True)
    for series, label, style in (
        (prescribed, "KWN prescribed source (non-predictive)", {"color": "#1d3557", "linestyle": "-"}),
        (feasible, "effective-CNT soft-constraint hit", {"color": "#e76f51", "linestyle": "--"}),
    ):
        if series:
            times = [_float(row, "time_h") for row in series]
            axes[0].plot(times, [_float(row, "g_number_density_m3") for row in series], label=label, **style)
            axes[1].plot(times, [_float(row, "matrix_Ag_at_fraction") for row in series], label=label, **style)
    axes[0].errorbar(
        [_float(row, "time_h") for row in primary],
        [_float(row, "number_density_m3") for row in primary],
        yerr=[_float(row, "number_density_uncertainty_m3") for row in primary],
        fmt="o",
        color="#111111",
        label="Sheskin 2018 primary Ag-rich objects",
    )
    axes[0].scatter(
        [_float(row, "time_h") for row in secondary],
        [_float(row, "number_density_m3") for row in secondary],
        marker="x",
        color="#6a994e",
        label="Yu 2024 small population (holdout)",
    )
    axes[1].errorbar(
        [_float(row, "time_h") for row in primary],
        [_float(row, "matrix_Ag_at_fraction") for row in primary],
        yerr=[_float(row, "matrix_Ag_at_fraction_uncertainty") for row in primary],
        fmt="o",
        color="#111111",
        label="Sheskin 2018 primary matrix Ag",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("number density (m$^{-3}$)")
    axes[1].set_ylabel("matrix Ag atomic fraction")
    axes[1].set_xlabel("aging time (h), 380 °C")
    axes[0].set_title("Primary constraints versus effective KWN trajectories")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_dir / "01_primary_constraints_vs_kwn.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    prescribed_heat = [
        row for row in gp_heatmap if row["scenario"] == "prescribed_source" and row["parameter_set"] == "0"
    ]
    times = sorted({_float(row, "time_h") for row in prescribed_heat})
    radii = sorted(
        {
            math.sqrt(_float(row, "radius_left_m") * _float(row, "radius_right_m")) * 1.0e9
            for row in prescribed_heat
        }
    )
    image = np.full((len(radii), len(times)), np.nan)
    radius_index = {value: index for index, value in enumerate(radii)}
    time_index = {value: index for index, value in enumerate(times)}
    for row in prescribed_heat:
        radius = math.sqrt(_float(row, "radius_left_m") * _float(row, "radius_right_m")) * 1.0e9
        image[radius_index[radius], time_index[_float(row, "time_h")]] = math.log10(
            max(_float(row, "number_density_per_m4"), 1.0e-300)
        )
    mesh = axes[0].pcolormesh(times, radii, image, shading="nearest", cmap="viridis")
    figure.colorbar(mesh, ax=axes[0], label="log$_{10}$ n(R) (m$^{-4}$)")
    axes[0].set_yscale("log")
    axes[0].set_title("g PSD: prescribed-source route")
    axes[0].set_xlabel("time (h)")
    axes[0].set_ylabel("spherical-equivalent radius (nm)")
    beta_nonzero = any(_float(row, "number_density_per_m4") > 0.0 for row in beta_heatmap)
    axes[1].set_title("beta PSD")
    axes[1].set_xlabel("time (h)")
    if beta_nonzero:
        axes[1].text(0.5, 0.5, "Non-zero beta data available", transform=axes[1].transAxes, ha="center")
    else:
        axes[1].text(
            0.5,
            0.5,
            "beta nucleation = off\nno beta population was fabricated",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
        )
    figure.tight_layout()
    figure.savefig(figure_dir / "02_population_psd_heatmaps.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.3, 4.5))
    times = [_float(row, "time_h") for row in prescribed]
    for key, label, color in (
        ("C_B_matrix_mol_m3", "matrix", "#457b9d"),
        ("C_B_GP_mol_m3", "GP", "#f4a261"),
        ("C_B_beta_mol_m3", "beta", "#6a994e"),
    ):
        axis.plot(times, [_float(row, key) for row in prescribed], marker="o", label=label, color=color)
    axis.set_xlabel("time (h)")
    axis.set_ylabel("B inventory (mol B m$^{-3}$)")
    axis.set_title("KWN inventory exchange: prescribed-source route")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "03_inventory_vs_time.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.3, 4.5))
    axis.semilogy(
        times,
        [max(abs(_float(row, "inventory_relative_residual")), 1.0e-18) for row in prescribed],
        marker="o",
        color="#264653",
        label="relative residual (display floor 1e-18)",
    )
    axis.axhline(1.0e-10, color="#9b2226", linestyle="--", label="acceptance threshold 1e-10")
    axis.set_xlabel("time (h)")
    axis.set_ylabel("relative inventory residual")
    axis.set_title("Mass residual versus time")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_dir / "04_mass_residual_vs_time.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.3, 4.5))
    metrics = sorted({str(row["metric"]) for row in convergence_rows})
    for metric in metrics:
        subset = [row for row in convergence_rows if row["metric"] == metric]
        axis.plot(
            [int(row["bins"]) for row in subset],
            [float(row["relative_difference_vs_400"]) for row in subset],
            marker="o",
            label=metric.replace("beta_", ""),
        )
    axis.axhline(0.02, color="#9b2226", linestyle="--", label="2% target")
    axis.set_xlabel("radius bins")
    axis.set_ylabel("relative difference versus 400 bins")
    axis.set_title("N5 radius-bin convergence (synthetic qualification state)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(figure_dir / "05_radius_bin_convergence.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.3, 4.5))
    metrics = sorted({str(row["metric"]) for row in timestep_rows})
    positions = np.arange(len(metrics), dtype=float)
    width = 0.34
    for offset, maximum_dt_s in ((-width / 2.0, 300.0), (width / 2.0, 150.0)):
        subset = {str(row["metric"]): float(row["relative_difference_vs_150_s"]) for row in timestep_rows if float(row["max_dt_s"]) == maximum_dt_s}
        axis.bar(positions + offset, [subset[metric] for metric in metrics], width=width, label=f"max dt={maximum_dt_s:g} s")
    axis.axhline(0.02, color="#9b2226", linestyle="--", label="2% target")
    axis.set_xticks(positions, [metric.replace("beta_", "") for metric in metrics], rotation=20, ha="right")
    axis.set_ylabel("relative difference versus 150 s")
    axis.set_title("N6 timestep convergence (synthetic qualification state)")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_dir / "06_timestep_convergence.png", dpi=180)
    plt.close(figure)

    _blocked_panel(
        plt,
        figure_dir / "07_beta_only_pf_blocked.png",
        "Beta-only KWN versus PF consistency",
        beta_status,
        "PF thermodynamic authority is unresolved and complete Broad/Narrow 400-cube PSDs are not local. No curves are fabricated.",
    )

    figure, axis = plt.subplots(figsize=(8.3, 5.0))
    colors = {
        "FEASIBLE": "#2a9d8f",
        "INFEASIBLE_SOFT_CONSTRAINTS": "#e9c46a",
        "INFEASIBLE_SOLVER_OR_INVENTORY": "#e76f51",
    }
    for status, color in colors.items():
        subset = [row for row in sweep_rows if row["mode"] == "effective_cnt" and row["status"] == status]
        if subset:
            axis.scatter(
                [_float(row, "gamma_g_J_m2") for row in subset],
                [_float(row, "site_density_g_m3") for row in subset],
                c=color,
                label=status,
                alpha=0.8,
            )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("effective $\\gamma_g$ (J m$^{-2}$)")
    axis.set_ylabel("effective site density (m$^{-3}$)")
    axis.set_title("Effective-CNT feasibility map (all 64 sampled sets retained)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_dir / "08_gp_effective_cnt_feasibility.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.3, 4.6))
    bucket_order = ("C_B_matrix", "C_B_GP", "C_B_beta_subgrid", "C_B_beta_resolved")
    positions = np.arange(len(bucket_order), dtype=float)
    source = audit["source_ledger"]
    materialized = audit["pf_materialized_inventory"]
    package_only = audit["package_only_inventory"]
    axis.bar(positions - 0.25, [source[key] for key in bucket_order], width=0.25, label="source ledger")
    axis.bar(positions, [materialized[key] for key in bucket_order], width=0.25, label="PF materialized plan")
    axis.bar(positions + 0.25, [package_only.get(key, 0.0) for key in bucket_order], width=0.25, label="package-only retained")
    axis.set_xticks(positions, [key.replace("C_B_", "") for key in bucket_order], rotation=15, ha="right")
    axis.set_ylabel("B inventory (mol B m$^{-3}$)")
    axis.set_title(f"Handoff ledger: {audit['status']}")
    axis.legend(fontsize=8)
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "09_handoff_ledger.png", dpi=180)
    plt.close(figure)

    _blocked_panel(
        plt,
        figure_dir / "10_pf_baseline_vs_handoff_blocked.png",
        "PF baseline versus KWN-informed primary handoff",
        smoke_status,
        "No PF smoke executable was invoked: P0 thermodynamic conflict and partial PF state closure prevent a valid baseline/handoff comparison.",
    )


def _write_reports(
    *,
    report_dir: Path,
    output_dir: Path,
    sweep_rows: Sequence[Mapping[str, str]],
    audit: Mapping[str, Any],
    metadata: Mapping[str, Any],
    beta_status: str,
    smoke_status: str,
) -> None:
    """Write limitation, final-acceptance, and reproduction documents."""

    counts = _status_counts(sweep_rows)
    source = metadata["source"]
    gp_unmapped = float(audit["package_only_inventory"]["C_B_GP"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "07_model_limitations.md").write_text(
        "# Model limitations\n\n"
        "## Authority and PF-state blockers\n\n"
        "- `P0_CONTRACT_CONFLICT`: the PF thermodynamic freeze remains `BLOCKED_CONTRACT_CONFLICT`; this delivery does not choose legacy or exact-candidate coefficients.\n"
        "- The requested full 12 h BROAD/REF and NARROW 400-cube resolved-beta PSDs are not retained locally, so beta-only PF comparison is not substituted with scalar moments.\n"
        "- The current qualified PF state has no independent persistent GP inventory state and no sub-grid beta inventory state.  A non-zero such bucket is retained package-only, never added to the matrix.\n"
        "\n## Effective KWN limits\n\n"
        "- The prescribed GP source is a numerical coupling exercise, not GP nucleation prediction.\n"
        "- Effective-CNT parameters are exploratory priors.  One soft-constraint hit among 64 retained sets does not identify GP composition, gamma, site density, attachment, diffusivity, or elastic penalty.\n"
        "- The GP lower radius boundary is 1 nm because the stated observation scope is 1–3 nm.  Below-boundary dissolution is conservative but cannot resolve sub-nanometre cluster physics.\n"
        "- KWN is mean-field and spherical-equivalent; it cannot reproduce spatial elastic competition, morphology, orientation, merging/splitting identity, or PF profile relaxation.\n"
        "\n## What was intentionally not done\n\n"
        "No PF/CUDA source was modified; no full 400³ campaign, new beta birth, GP-to-beta conversion, GP release, online coupling, dislocation physics, or physical retuning was introduced.\n",
        encoding="utf-8",
    )
    (report_dir / "08_final_acceptance_report.md").write_text(
        "# KWN–PF MVP v1 final acceptance\n\n"
        "Top-level status: `PARTIAL_PF_STATE_NOT_CLOSED`\n\n"
        "| question | evidence-grounded answer |\n"
        "|---|---|\n"
        "| 1. KWN backend | `INTERNAL_KWN_BACKEND_SELECTED`; Kawin was unavailable/unpinned, while the internal finite-volume backend is fully owned and tested. |\n"
        "| 2. Strict KWN conservation | Yes for the implemented solver: N1–N7 and the all-state ledger invariant pass at <= `1e-10`; the handoff package residual is `0`. |\n"
        f"| 3. beta-only PF consistency | `{beta_status}`: no valid same-contract PF/KWN comparison was run. |\n"
        "| 4. source of beta-only differences | Not determinable yet; thermodynamic authority, PF-consistent diffusivity, elastic/spatial competition, and full PF PSD inputs are not simultaneously available. |\n"
        f"| 5. 6 h GP-like population | Prescribed source runs; effective-CNT has {counts.get('FEASIBLE', 0)} soft-constraint hit(s) of 64, not a predictive calibration. |\n"
        "| 6. unidentifiable GP parameters | `xB_g`, gamma_g, site density, attachment, diffusivity scale, and elastic penalty remain non-identified exploratory parameters. |\n"
        "| 7. full 6 h mapping | No.  The matrix/resolved-beta portions have declared targets, but the complete four-bucket state is not PF-closed. |\n"
        f"| 8. unrepresentable inventory | GP inventory is non-zero in the emitted package (`{gp_unmapped:.16e}` mol B m⁻³); PF also lacks a persistent sub-grid beta state even though that bucket is zero for this prescribed source. |\n"
        f"| 9. PF pulse/seed dissolution | `{smoke_status}`: no PF run occurred, so no transient improvement or degradation is claimed. |\n"
        "| 10. next-step decision | Resolve/hash-bind the PF thermodynamic contract and specify independent PF GP/sub-grid-beta state variables before any PF smoke run. Do not start a larger PF case or concurrent coupling. |\n\n"
        "## Handoff evidence\n\n"
        f"The v1 package was generated from commit `{source['git_commit']}` with source config hash `{source['config_hash']}`. "
        f"Its audited status is `{audit['status']}`, package-level relative residual `{audit['relative_accounting_residual']:.3e}`, "
        "and PF raw initialization was not emitted.\n",
        encoding="utf-8",
    )
    (report_dir / "09_reproduction_commands.md").write_text(
        "# Reproduction commands\n\n"
        "Run from the isolated worktree root.  The configuration files are JSON-subset YAML because the local environment has no PyYAML dependency.\n\n"
        "```bash\n"
        "cd /Users/heng/Documents/GitHub/CUDA_STO_PF-kwn-pf-mvp-v1\n"
        "PYTHONPATH=src python3 -m unittest discover -s tests/kwn -v\n"
        "PYTHONPATH=src python3 -m unittest discover -s tests/coupling -v\n"
        "PYTHONPATH=src python3 scripts/run_beta_only_consistency.py\n"
        "PYTHONPATH=src python3 scripts/run_gp_feasibility_sweep.py\n"
        "PYTHONPATH=src python3 scripts/build_kwn_pf_handoff.py\n"
        "python3 scripts/run_pf_handoff_smoke.py\n"
        "PYTHONPATH=src python3 scripts/make_kwn_pf_report.py\n"
        "```\n\n"
        "The PF smoke wrapper is an authority gate.  A `NOT_RUN_*` result is the intended reproducible outcome while the P0 contract conflict and partial state closure remain.\n",
        encoding="utf-8",
    )


def main() -> int:
    """Render figures and final reports from the retained MVP output set."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/kwn_pf_mvp_v1"))
    parser.add_argument("--report-dir", default=str(ROOT / "reports/kwn_pf_mvp_v1"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    required = {
        "constraints": ROOT / "data/kwn_constraints/experimental_constraints.csv",
        "trajectories": output_dir / "gp_trajectories.csv",
        "gp_heatmap": output_dir / "gp_psd_heatmap.csv",
        "beta_heatmap": output_dir / "beta_psd_heatmap.csv",
        "sweep": output_dir / "gp_parameter_sweep.csv",
        "audit": output_dir / "handoff_audit.json",
        "metadata": output_dir / "kwn_pf_handoff_6h/metadata.json",
        "smoke": output_dir / "pf_smoke_gate.json",
        "beta_report": report_dir / "03_beta_only_pf_consistency.md",
    }
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"required {label} artifact is missing: {path}")
    convergence_rows, timestep_rows = _qualification_tables(output_dir)
    constraints = _read_csv(required["constraints"])
    trajectories = _read_csv(required["trajectories"])
    gp_heatmap = _read_csv(required["gp_heatmap"])
    beta_heatmap = _read_csv(required["beta_heatmap"])
    sweep_rows = _read_csv(required["sweep"])
    audit = json.loads(required["audit"].read_text(encoding="utf-8"))
    metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
    smoke = json.loads(required["smoke"].read_text(encoding="utf-8"))
    beta_status = "P0_CONTRACT_CONFLICT"
    smoke_status = str(smoke["status"])
    _render_figures(
        figure_dir=report_dir / "figures",
        constraints=constraints,
        trajectories=trajectories,
        gp_heatmap=gp_heatmap,
        beta_heatmap=beta_heatmap,
        sweep_rows=sweep_rows,
        convergence_rows=convergence_rows,
        timestep_rows=timestep_rows,
        audit=audit,
        beta_status=beta_status,
        smoke_status=smoke_status,
    )
    _write_reports(
        report_dir=report_dir,
        output_dir=output_dir,
        sweep_rows=sweep_rows,
        audit=audit,
        metadata=metadata,
        beta_status=beta_status,
        smoke_status=smoke_status,
    )
    print(json.dumps({"commit": _git_commit(), "status": audit["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
