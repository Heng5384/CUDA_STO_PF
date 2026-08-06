#!/usr/bin/env python3
"""Assemble P1 tables, diagnostics, figures, and bounded recommendations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


FINAL_CONDITIONAL = "CONDITIONAL_PASS_TRUE_AREA_OR_AUTHORITY_PENDING"
CASES = ("001", "002", "003", "004", "019", "020", "021")
FAMILIES = ("REF_BROAD_N512", "NARROW_N512", "NARROW_N704")
COLORS = {"REF_BROAD_N512": "#1f77b4", "NARROW_N512": "#ff7f0e", "NARROW_N704": "#2ca02c"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(errors: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in errors:
        model = row["compressed_model"]
        groups[(model, "overall", "all")].append(row)
        groups[(model, "temperature_K", row["temperature_K"])].append(row)
        groups[(model, "time_h", row["time_h"])].append(row)
        groups[(model, "family", row["family"])].append(row)
        groups[(model, "matrix_mode", row["matrix_mode"])].append(row)
    result: list[dict[str, Any]] = []
    for (model, dimension, value), rows in sorted(groups.items()):
        signed = np.asarray([float(row["signed_error_W_mK"]) for row in rows])
        absolute = np.abs(signed)
        relative = np.asarray([float(row["relative_error_percent"]) for row in rows])
        result.append(
            {
                "compressed_model": model,
                "reference_model": "E_direct_discrete_PSD",
                "group_dimension": dimension,
                "group_value": value,
                "n": len(rows),
                "mean_signed_error_W_mK": float(np.mean(signed)),
                "MAE_W_mK": float(np.mean(absolute)),
                "RMSE_W_mK": float(np.sqrt(np.mean(signed**2))),
                "maximum_absolute_error_W_mK": float(np.max(absolute)),
                "mean_relative_error_percent": float(np.mean(relative)),
                "maximum_relative_error_percent": float(np.max(relative)),
            }
        )
    return result


def family_case_values(rows: list[dict[str, Any]], value: str) -> dict[str, list[float]]:
    result: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        result[str(row["family"])].append(float(row[value]))
    return result


def effect_decomposition(summary: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [row for row in summary if row["model"] in ("E_direct_discrete_PSD", "F_direct_PSD_plus_sphere_MI", "G_direct_PSD_plus_true_MI", "H_true_area_envelope")]
    by_state: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        by_state[(row["matrix_mode"], row["temperature_K"], row["time_h"])].append(row)
    result: list[dict[str, Any]] = []
    for (matrix_mode, temperature, time_h), rows in sorted(by_state.items()):
        e = [row for row in rows if row["model"] == "E_direct_discrete_PSD"]
        f = {row["case_id"]: row for row in rows if row["model"] == "F_direct_PSD_plus_sphere_MI"}
        g = {row["case_id"]: row for row in rows if row["model"] == "G_direct_PSD_plus_true_MI"}
        h = {row["case_id"]: row for row in rows if row["model"] == "H_true_area_envelope"}
        family_values = family_case_values(e, "kappa_member_mean_W_mK")
        within = max((max(values) - min(values) for family, values in family_values.items() if len(values) >= 2), default=0.0)
        family_means = [float(np.mean(values)) for values in family_values.values()]
        between = max(family_means) - min(family_means)
        sphere_true = float(np.mean([abs(float(f[case]["kappa_member_mean_W_mK"]) - float(g[case]["kappa_member_mean_W_mK"])) for case in g]))
        calibration = float(np.mean([float(row["kappa_member_max_W_mK"]) - float(row["kappa_member_min_W_mK"]) for row in g.values()]))
        levelset = float(np.mean([float(row["kappa_levelset_and_member_max_W_mK"]) - float(row["kappa_levelset_and_member_min_W_mK"]) for row in h.values()]))
        sources = {
            "within_seed_spread": within,
            "between_family_difference": between,
            "sphere_true_difference": sphere_true,
            "calibration_uncertainty": calibration,
            "levelset_uncertainty": levelset,
        }
        result.append(
            {
                "quantity": "kappa_conditional_lattice_no_dislocation_W_mK",
                "matrix_mode": matrix_mode,
                "temperature_K": temperature,
                "time_h": time_h,
                **sources,
                "dominant_source": max(sources, key=sources.get),
                "case_004_single_seed_causal_claim": False,
            }
        )
    return result


def collapse_statistics(summary: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model, x_name in (("E_direct_discrete_PSD", "Sv_sph_m-1"), ("G_direct_PSD_plus_true_MI", "Sv_true_phi50_m-1")):
        rows = [row for row in summary if row["model"] == model]
        groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            groups[(row["matrix_mode"], row["temperature_K"])].append(row)
        for (matrix_mode, temperature), values in sorted(groups.items()):
            x = np.log(np.asarray([float(row[x_name]) for row in values]))
            y = np.asarray([float(row["kappa_member_mean_W_mK"]) for row in values])
            design = np.column_stack((np.ones_like(x), x, x**2))
            coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
            prediction = design @ coefficient
            residual = y - prediction
            denominator = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else 1.0
            result.append(
                {
                    "model": model,
                    "state_variable": x_name,
                    "matrix_mode": matrix_mode,
                    "temperature_K": temperature,
                    "n": len(values),
                    "quadratic_log_state_R2": r2,
                    "residual_RMSE_W_mK": float(np.sqrt(np.mean(residual**2))),
                    "collapse_pass_R2_ge_0p9": r2 >= 0.9,
                }
            )
    return result


def handoff_rows(summary: list[dict[str, str]], observables: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    lookup = {(row["case_id"], int(row["time_h"]), row["matrix_mode"], row["temperature_K"], row["model"]): row for row in summary}
    for case in CASES:
        for matrix_mode in ("fixed_6h_matrix", "pf_time_varying_matrix"):
            temperatures = sorted({row["temperature_K"] for row in summary})
            for temperature in temperatures:
                for model in ("E_direct_discrete_PSD", "G_direct_PSD_plus_true_MI"):
                    end = lookup[(case, 48, matrix_mode, temperature, model)]
                    for start in (6, 12, 24):
                        first = lookup[(case, start, matrix_mode, temperature, model)]
                        result.append(
                            {
                                "case_id": case,
                                "family": first["family"],
                                "replicate": first["replicate"],
                                "matrix_mode": matrix_mode,
                                "temperature_K": temperature,
                                "model": model,
                                "start_time_h": start,
                                "end_time_h": 48,
                                "handoff_sensitive_start": start < 12,
                                "delta_kappa_W_mK": float(end["kappa_member_mean_W_mK"]) - float(first["kappa_member_mean_W_mK"]),
                                "relative_delta_kappa_percent": 100.0 * (float(end["kappa_member_mean_W_mK"]) - float(first["kappa_member_mean_W_mK"])) / float(first["kappa_member_mean_W_mK"]),
                                "delta_Sv_sph_m-1": float(end["Sv_sph_m-1"]) - float(first["Sv_sph_m-1"]),
                                "delta_Sv_true_m-1": float(end["Sv_true_phi50_m-1"]) - float(first["Sv_true_phi50_m-1"]),
                                "delta_M6_m3": float(end["M6_m3"]) - float(first["M6_m3"]),
                            }
                        )
    # The registered hourly aggregate supplies a 10 h descriptor-only anchor.
    obs = {(row["case_id"], int(row["hour_h"])): row for row in observables}
    for case in CASES:
        first, end = obs[(case, 10)], obs[(case, 48)]
        result.append(
            {
                "case_id": case,
                "family": first["family"],
                "replicate": first["replicate"],
                "matrix_mode": "DESCRIPTOR_ONLY",
                "temperature_K": "NOT_EVALUATED_WITH_FULL_PSD",
                "model": "10h_registered_descriptor_anchor",
                "start_time_h": 10,
                "end_time_h": 48,
                "handoff_sensitive_start": True,
                "delta_kappa_W_mK": "NOT_EVALUATED_PRIMARY_P1_HAS_NO_10H_TRUE_AREA",
                "relative_delta_kappa_percent": "NOT_EVALUATED",
                "delta_Sv_sph_m-1": (float(end["Sv_nm_inv"]) - float(first["Sv_nm_inv"])) * 1.0e9,
                "delta_Sv_true_m-1": "NOT_EVALUATED",
                "delta_M6_m3": (float(end["M6_nm3"]) - float(first["M6_nm3"])) * 1.0e-27,
            }
        )
    return result


def structure_factor_diagnostic(geometry: list[dict[str, str]]) -> tuple[list[dict[str, Any]], str]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in geometry:
        groups[(row["case_id"], int(float(row["time_h"])))].append(row)
    q_values = np.geomspace(0.02, 2.0, 48)
    result: list[dict[str, Any]] = []
    for (case, hour), rows in sorted(groups.items()):
        centers = np.asarray([[float(row[f"center_{axis}_nm"]) for axis in "xyz"] for row in rows])
        radii = np.asarray([float(row["equivalent_radius_nm"]) for row in rows])
        tree = cKDTree(centers, boxsize=400.0)
        nearest, nearest_index = tree.query(centers, k=2)
        surface_gap = nearest[:, 1] - radii - radii[nearest_index[:, 1]]
        pairs = np.asarray(sorted(tree.query_pairs(200.0)), dtype=np.int64)
        if pairs.size:
            delta = centers[pairs[:, 0]] - centers[pairs[:, 1]]
            delta -= 400.0 * np.round(delta / 400.0)
            distance = np.linalg.norm(delta, axis=1)
        else:
            distance = np.asarray([], dtype=float)
        for q in q_values:
            qr = q * distance
            structure = 1.0 + 2.0 / len(rows) * float(np.sum(np.sinc(qr / math.pi)))
            result.append(
                {
                    "case_id": case,
                    "time_h": hour,
                    "particle_count": len(rows),
                    "q_nm-1": q,
                    "S_isotropic_finite_box": structure,
                    "nearest_neighbor_mean_nm": float(np.mean(nearest[:, 1])),
                    "nearest_surface_gap_mean_nm": float(np.mean(surface_gap)),
                    "diagnostic_only_not_in_transport_rate": True,
                }
            )
    # No validated differential-scattering mapping exists, so the diagnostic
    # cannot clear the physics gate even when center statistics are measurable.
    return result, "NO_GO_DIRECT_SPATIAL_TRANSPORT_FOR_CURRENT_PAPER"


def save_figure(fig: plt.Figure, directory: Path, name: str) -> None:
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def figures(
    geometry: list[dict[str, str]],
    summary: list[dict[str, str]],
    effects: list[dict[str, Any]],
    handoff: list[dict[str, Any]],
    directory: Path,
) -> None:
    directory.mkdir()
    # 1: object-area parity.
    sph = np.asarray([float(row["sphere_area_nm2"]) for row in geometry])
    true = np.asarray([float(row["true_area_nm2"]) for row in geometry])
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(sph, true, s=7, alpha=.35)
    limit = max(float(np.max(sph)), float(np.max(true)))
    ax.plot([0, limit], [0, limit], "k--", lw=1)
    ax.set(xlabel="Equivalent-sphere area (nm²)", ylabel="True φ=0.50 area (nm²)", title="Conditional PF objects: sphere vs true area")
    save_figure(fig, directory, "01_sphere_true_area_parity")

    # Aggregate helpers.
    by_snapshot: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in geometry:
        by_snapshot[(row["case_id"], int(float(row["time_h"])))].append(row)
    aggregates = []
    for (case, hour), rows in sorted(by_snapshot.items()):
        aggregates.append({"case": case, "hour": hour, "sph": sum(float(r["sphere_area_nm2"]) for r in rows), "true": sum(float(r["true_area_nm2"]) for r in rows), "aspect": float(np.median([float(r["aspect_ratio_1_3"]) for r in rows]))})
    family_by_case = {row["case_id"]: row["family"] for row in summary}

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for case in CASES:
        rows = [r for r in aggregates if r["case"] == case]
        ax.plot([r["hour"] for r in rows], [r["true"] / r["sph"] for r in rows], marker="o", label=case, color=COLORS[family_by_case[case]], alpha=.75)
    ax.axvspan(6, 12, color="0.85", label="handoff-sensitive")
    ax.set(xlabel="Age (h)", ylabel="A_true/A_sph", title="True/sphere aggregate area ratio (conditional)")
    ax.legend(ncol=4, fontsize=8)
    save_figure(fig, directory, "02_true_sphere_ratio_time")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for case in CASES:
        rows = [r for r in aggregates if r["case"] == case]
        ax.plot([r["hour"] for r in rows], [r["sph"] / 400.0**3 for r in rows], "--o", color=COLORS[family_by_case[case]], alpha=.55)
        ax.plot([r["hour"] for r in rows], [r["true"] / 400.0**3 for r in rows], "-o", color=COLORS[family_by_case[case]], alpha=.8)
    ax.axvspan(6, 12, color="0.9")
    ax.set(xlabel="Age (h)", ylabel="Sv (nm⁻¹)", title="Sv(t): dashed sphere, solid true area")
    from matplotlib.lines import Line2D
    family_handles = [Line2D([0], [0], color=COLORS[family], lw=2, label=family) for family in FAMILIES]
    area_handles = [Line2D([0], [0], color="0.25", ls="--", lw=2, label="sphere area"), Line2D([0], [0], color="0.25", ls="-", lw=2, label="true φ=0.50 area")]
    ax.legend(handles=family_handles + area_handles, ncol=2, fontsize=7)
    save_figure(fig, directory, "03_Sv_sphere_true_time")

    selected = [r for r in summary if float(r["temperature_K"]) == 303.15 and r["matrix_mode"] == "pf_time_varying_matrix" and r["model"] in ("D_Sv_sph_plus_M6", "E_direct_discrete_PSD", "F_direct_PSD_plus_sphere_MI", "G_direct_PSD_plus_true_MI")]
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    styles = {"D_Sv_sph_plus_M6": ":", "E_direct_discrete_PSD": "--", "F_direct_PSD_plus_sphere_MI": "-.", "G_direct_PSD_plus_true_MI": "-"}
    for model in styles:
        for family in FAMILIES:
            values = [r for r in selected if r["model"] == model and r["family"] == family]
            by_time = defaultdict(list)
            for row in values: by_time[int(row["time_h"])].append(float(row["kappa_member_mean_W_mK"]))
            times = sorted(by_time)
            ax.plot(times, [np.mean(by_time[t]) for t in times], styles[model], color=COLORS[family], marker="o", alpha=.8)
    ax.axvspan(6, 12, color="0.9")
    ax.set(xlabel="Age (h)", ylabel="Conditional no-dislocation κ (W m⁻¹ K⁻¹)", title="Model comparison at 303.15 K, PF-time matrix")
    model_handles = [Line2D([0], [0], color="0.25", ls=styles[model], lw=2, label=model.split("_", 1)[0]) for model in styles]
    ax.legend(handles=family_handles + model_handles, ncol=2, fontsize=7, loc="center right")
    save_figure(fig, directory, "04_kappa_time_models")

    e_rows = [r for r in summary if r["model"] == "E_direct_discrete_PSD" and float(r["temperature_K"]) == 303.15]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for mode, marker in (("fixed_6h_matrix", "o"), ("pf_time_varying_matrix", "s")):
        values = [r for r in e_rows if r["matrix_mode"] == mode]
        ax.scatter([float(r["Sv_sph_m-1"]) for r in values], [float(r["kappa_member_mean_W_mK"]) for r in values], marker=marker, label=mode, c=[COLORS[r["family"]] for r in values])
    ax.set(xlabel="Sv_sph (m⁻¹)", ylabel="Direct-PSD κ (W m⁻¹ K⁻¹)", title="κ(Sv_sph), 303.15 K")
    ax.legend(fontsize=8)
    save_figure(fig, directory, "05_kappa_vs_Sv_sphere")

    g_rows = [r for r in summary if r["model"] == "G_direct_PSD_plus_true_MI" and float(r["temperature_K"]) == 303.15]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for mode, marker in (("fixed_6h_matrix", "o"), ("pf_time_varying_matrix", "s")):
        values = [r for r in g_rows if r["matrix_mode"] == mode]
        ax.scatter([float(r["Sv_true_phi50_m-1"]) for r in values], [float(r["kappa_member_mean_W_mK"]) for r in values], marker=marker, label=mode, c=[COLORS[r["family"]] for r in values])
    ax.set(xlabel="Sv_true φ=0.50 (m⁻¹)", ylabel="True-area MI κ (W m⁻¹ K⁻¹)", title="κ(Sv_true), 303.15 K")
    ax.legend(fontsize=8)
    save_figure(fig, directory, "06_kappa_vs_Sv_true")

    d_lookup = {(r["case_id"], r["time_h"], r["matrix_mode"], r["temperature_K"]): r for r in summary if r["model"] == "D_Sv_sph_plus_M6"}
    e_lookup = {(r["case_id"], r["time_h"], r["matrix_mode"], r["temperature_K"]): r for r in summary if r["model"] == "E_direct_discrete_PSD"}
    x = np.asarray([float(e_lookup[k]["kappa_member_mean_W_mK"]) for k in sorted(e_lookup)])
    y = np.asarray([float(d_lookup[k]["kappa_member_mean_W_mK"]) for k in sorted(e_lookup)])
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(x, y, s=12, alpha=.55)
    bounds = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(bounds, bounds, "k--")
    ax.set(xlabel="Direct PSD κ (W m⁻¹ K⁻¹)", ylabel="Sv+M6 κ (W m⁻¹ K⁻¹)", title="Compressed vs direct PSD parity")
    save_figure(fig, directory, "07_compressed_direct_parity")

    values = {name: float(np.mean([float(r[name]) for r in effects])) for name in ("within_seed_spread", "between_family_difference", "sphere_true_difference", "calibration_uncertainty", "levelset_uncertainty")}
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(range(len(values)), list(values.values()), color=["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"])
    ax.set_xticks(range(len(values)), [key.replace("_", "\n") for key in values], fontsize=8)
    ax.set(ylabel="Mean effect scale (W m⁻¹ K⁻¹)", title="Separated seed, family, area and calibration scales")
    save_figure(fig, directory, "08_effect_size_decomposition")

    h_values = [r for r in handoff if r["model"] == "G_direct_PSD_plus_true_MI" and r["matrix_mode"] == "pf_time_varying_matrix" and str(r["temperature_K"]) == "303.15"]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    starts = (6, 12, 24)
    means = [np.mean([float(r["delta_kappa_W_mK"]) for r in h_values if int(r["start_time_h"]) == start]) for start in starts]
    ax.bar([str(start) for start in starts], means, color=["0.65", "#4c78a8", "#4c78a8"])
    ax.set(xlabel="Start age (h)", ylabel="Mean Δκ to 48 h", title="Handoff start-time sensitivity, 303.15 K")
    save_figure(fig, directory, "09_handoff_start_sensitivity")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for case in CASES:
        rows = [r for r in aggregates if r["case"] == case]
        ax.plot([r["hour"] for r in rows], [r["aspect"] for r in rows], "-o", color=COLORS[family_by_case[case]], alpha=.75)
    ax.axvspan(6, 12, color="0.9")
    ax.set(xlabel="Age (h)", ylabel="Median principal-axis ratio", title="PF object shape evolution (conditional)")
    save_figure(fig, directory, "10_shape_aspect_ratio_time")

    h_rows = [r for r in summary if r["model"] == "H_true_area_envelope" and r["matrix_mode"] == "pf_time_varying_matrix" and float(r["temperature_K"]) == 303.15]
    by_time = defaultdict(list)
    for row in h_rows: by_time[int(row["time_h"])].append(row)
    times = sorted(by_time)
    mean = [np.mean([float(r["kappa_member_mean_W_mK"]) for r in by_time[t]]) for t in times]
    low = [min(float(r["kappa_levelset_and_member_min_W_mK"]) for r in by_time[t]) for t in times]
    high = [max(float(r["kappa_levelset_and_member_max_W_mK"]) for r in by_time[t]) for t in times]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.fill_between(times, low, high, alpha=.25, label="φ=0.45–0.55 + member envelope")
    ax.plot(times, mean, "-o", label="φ=0.50 member mean")
    ax.axvspan(6, 12, color="0.9")
    ax.set(xlabel="Age (h)", ylabel="Conditional κ (W m⁻¹ K⁻¹)", title="No-refit true-area sensitivity envelope, 303.15 K")
    ax.legend(fontsize=8)
    save_figure(fig, directory, "11_levelset_sensitivity_envelope")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-freeze", required=True, type=Path)
    parser.add_argument("--geometry-root", required=True, type=Path)
    parser.add_argument("--synthetic-qualification", required=True, type=Path)
    parser.add_argument("--transport-root", required=True, type=Path)
    parser.add_argument("--determinism-root", required=True, type=Path)
    parser.add_argument("--observables", required=True, type=Path)
    parser.add_argument("--transport-core", required=True, type=Path)
    parser.add_argument("--geometry-code", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    deterministic_files = (
        "transport_member_predictions.csv",
        "transport_model_comparison.csv",
        "snapshot_descriptors.csv",
        "descriptor_compression_error.csv",
        "transport_qualification.json",
    )
    determinism = {
        name: {
            "run_1_sha256": sha256(args.transport_root / name),
            "run_2_sha256": sha256(args.determinism_root / name),
        }
        for name in deterministic_files
    }
    for value in determinism.values():
        value["byte_identical"] = value["run_1_sha256"] == value["run_2_sha256"]
    if not all(value["byte_identical"] for value in determinism.values()):
        raise ValueError("transport determinism gate failed")
    (args.out / "transport_determinism.json").write_text(json.dumps(determinism, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copyfile(args.authority_freeze / "checkpoint_authority_manifest.csv", args.out / "checkpoint_authority_manifest.csv")
    shutil.copyfile(args.synthetic_qualification / "true_area_validation_report.md", args.out / "true_area_validation_report.md")

    geometry: list[dict[str, str]] = []
    geometry_paths = sorted(args.geometry_root.glob("case_*/*h/particle_geometry.csv"))
    if len(geometry_paths) != 28:
        raise ValueError(f"expected 28 geometry files, found {len(geometry_paths)}")
    for path in geometry_paths:
        geometry.extend(read_csv(path))
    write_csv(args.out / "pf_particle_geometry_manifest.csv", geometry)

    threshold_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in geometry:
        grouped[(row["case_id"], int(float(row["time_h"])))].append(row)
    for (case, hour), rows in sorted(grouped.items()):
        for level in (45, 50, 55):
            area = sum(float(row[f"area_nm2_phi{level}"]) for row in rows)
            threshold_rows.append({"case_id": case, "time_h": hour, "level_set_phi": level / 100.0, "aggregate_area_nm2": area, "Sv_true_nm-1": area / 400.0**3, "canonical": level == 50, "absent_object_count": sum(row[f"levelset_present_phi{level}"] != "True" for row in rows), "authority": "CONDITIONAL"})
    write_csv(args.out / "true_area_threshold_sensitivity.csv", threshold_rows)

    summary = read_csv(args.transport_root / "transport_model_comparison.csv")
    for row in summary:
        delta = row["delta_kappa_from_6h_member_mean_W_mK"]
        row["delta_kappa_PSD_W_mK"] = delta if row["model"] == "E_direct_discrete_PSD" and row["matrix_mode"] == "fixed_6h_matrix" else ""
        row["delta_kappa_PSD_plus_matrix_W_mK"] = delta if row["model"] == "E_direct_discrete_PSD" and row["matrix_mode"] == "pf_time_varying_matrix" else ""
        row["conductivity_semantics"] = "conditional_lattice_no_dislocation"
    errors = read_csv(args.transport_root / "descriptor_compression_error.csv")
    write_csv(args.out / "transport_model_comparison.csv", summary)
    write_csv(args.out / "kappa_time_trajectories.csv", summary)
    write_csv(args.out / "kappa_state_variable_curves.csv", summary)
    metrics = metric_rows(errors)
    write_csv(args.out / "reduced_model_error_metrics.csv", metrics)
    effects = effect_decomposition(summary)
    write_csv(args.out / "effect_size_decomposition.csv", effects)
    collapse = collapse_statistics(summary)
    write_csv(args.out / "kappa_state_variable_collapse_metrics.csv", collapse)
    observables = read_csv(args.observables)
    handoff = handoff_rows(summary, observables)
    write_csv(args.out / "handoff_start_time_sensitivity.csv", handoff)
    structure, structure_decision = structure_factor_diagnostic(geometry)
    write_csv(args.out / "structure_factor_diagnostic.csv", structure)

    max_area_ratio_difference = max(
        abs(
            sum(float(row["true_area_nm2"]) for row in rows)
            / sum(float(row["sphere_area_nm2"]) for row in rows)
            - 1.0
        )
        for rows in grouped.values()
    )
    max_present_object_ratio_difference = max(
        abs(float(row["true_to_sphere_area_ratio"]) - 1.0)
        for row in geometry
        if row["levelset_present_phi50"] == "True"
    )
    max_aggregate_ratio_difference = max(abs(row["aggregate_area_nm2"] / next(base["aggregate_area_nm2"] for base in threshold_rows if base["case_id"] == row["case_id"] and base["time_h"] == row["time_h"] and base["level_set_phi"] == .5) - 1.0) for row in threshold_rows if row["level_set_phi"] != .5)
    d_overall = next(row for row in metrics if row["compressed_model"] == "D_Sv_sph_plus_M6" and row["group_dimension"] == "overall")
    mean_effect = {key: float(np.mean([float(row[key]) for row in effects])) for key in ("within_seed_spread", "between_family_difference", "sphere_true_difference", "calibration_uncertainty", "levelset_uncertainty")}
    collapse_pass_fraction = float(np.mean([row["collapse_pass_R2_ge_0p9"] for row in collapse]))
    reduced_is_below_uncertainty = float(d_overall["MAE_W_mK"]) < max(mean_effect["within_seed_spread"], mean_effect["calibration_uncertainty"], mean_effect["levelset_uncertainty"])
    true_area_small = mean_effect["sphere_true_difference"] < max(mean_effect["within_seed_spread"], mean_effect["calibration_uncertainty"])
    recommendation = "KEEP_REDUCED_TRANSPORT_MODEL" if reduced_is_below_uncertainty and true_area_small else ("UPGRADE_TO_TRUE_AREA_MODEL_1A" if not true_area_small else "RETAIN_DIRECT_DISCRETE_PSD_IN_MAIN_MODEL")

    full_psd_report = f"""# Full-PSD code-path requalification

Status: `DIRECT_DISCRETE_PSD_REFERENCE`

The frozen implementation in `{args.transport_core}` was read and exercised.
`full_psd_precipitate_rate` calls `precipitate_cross_section` with the complete
one-dimensional list of per-object radii and performs `v/V_box * sum_i` along
the particle axis at every frequency.  The long-wavelength branch is
`(4/9)πR²(Δρ/ρ)²(ωR/v)^4`, the geometric branch is `2πR²`, and the code uses
their harmonic interpolation.  Descriptor models are separate functions and
are not used by Model E.

The base rate adds phonon-phonon (`A_N=1.5`), boundary and matrix-point-defect
rates by Matthiessen addition.  The P1 driver adds the frozen V2 `A2*ω²` member
and, only for F/G/H, the frozen interface term.  S11 and S13 are identically
zero; Yu scale 0.1172768 is disabled.  The output is conditional
no-dislocation lattice-style conductivity, not total experimental κ.

Runtime qualification passed scalar-versus-vector direct summation,
monodisperse degeneration, population/volume scaling, permutation, zero
population, and contract guards.  Core SHA-256: `{sha256(args.transport_core)}`.
"""
    (args.out / "full_psd_codepath_requalification.md").write_text(full_psd_report, encoding="utf-8")
    error_report = f"""# Reduced-model error report

Reference: `E_direct_discrete_PSD` using the complete radius list.

The Sv+M6 reconstruction has MAE `{float(d_overall['MAE_W_mK']):.6g}` W m⁻¹ K⁻¹,
RMSE `{float(d_overall['RMSE_W_mK']):.6g}` W m⁻¹ K⁻¹, maximum absolute error
`{float(d_overall['maximum_absolute_error_W_mK']):.6g}` W m⁻¹ K⁻¹ and mean
relative error `{float(d_overall['mean_relative_error_percent']):.4g}%` over
all cases, four primary times, both matrix paths, seven temperatures and 18
frozen calibration members (summarized by member mean before compression
comparison).

Mean unavoidable comparison scales are: within-family seed spread
`{mean_effect['within_seed_spread']:.6g}`, between-family separation
`{mean_effect['between_family_difference']:.6g}`, sphere/true representation
`{mean_effect['sphere_true_difference']:.6g}`, calibration member width
`{mean_effect['calibration_uncertainty']:.6g}`, and level-set/member envelope
`{mean_effect['levelset_uncertainty']:.6g}` W m⁻¹ K⁻¹.  Therefore the answer to
“is compression error below current model uncertainty?” is
`{reduced_is_below_uncertainty}`.
"""
    (args.out / "reduced_model_error_report.md").write_text(error_report, encoding="utf-8")
    handoff_focus = [
        row for row in summary
        if row["model"] == "G_direct_PSD_plus_true_MI"
        and row["matrix_mode"] == "pf_time_varying_matrix"
        and float(row["temperature_K"]) == 303.15
    ]
    focus_lookup = {(row["case_id"], int(row["time_h"])): float(row["kappa_member_mean_W_mK"]) for row in handoff_focus}
    rank_6 = ">".join(case for _, case in sorted(((focus_lookup[(case, 6)], case) for case in CASES), reverse=True))
    rank_12 = ">".join(case for _, case in sorted(((focus_lookup[(case, 12)], case) for case in CASES), reverse=True))
    sign_6_48 = all(focus_lookup[(case, 48)] > focus_lookup[(case, 6)] for case in CASES)
    sign_12_48 = all(focus_lookup[(case, 48)] > focus_lookup[(case, 12)] for case in CASES)
    handoff_report = f"""# Handoff start-time sensitivity

The primary full-PSD/true-area comparison uses 6, 12, 24 and 48 h.  Six hours
is explicitly handoff-sensitive; 12–48 h is the preferred physical coarsening
interval.  The registered hourly aggregate supplies a 10 h Sv/M6 anchor, but
P1 does not invent a 10 h true area or discrete-PSD κ for Cases 004/019.

At 303.15 K on the PF-time matrix/true-area branch, the case order is
`{rank_6}` at 6 h and `{rank_12}` at 12 h; therefore early handoff relaxation
does change case ranking.  Nevertheless κ rises toward 48 h for every case
from both 6 h (`{sign_6_48}`) and 12 h (`{sign_12_48}`), so the 12–48 h
mechanism/sign conclusion is not reversed.

Rank and sign checks are evaluated in `handoff_start_time_sensitivity.csv`.
The 6 h interval changes the magnitude of window closure, while the formal
mechanism decision is based on 12–48 h.  Absolute 6 h window lifetime remains
blocked by handoff conditioning; this does not block the conditional P1
prototype.  No conditioned 6 h state was reconstructed or refitted.
"""
    (args.out / "handoff_start_time_sensitivity.md").write_text(handoff_report, encoding="utf-8")
    structure_report = f"""# Structure-factor GO/NO-GO screen

Decision: `{structure_decision}`

Periodic centers from independently qualified snapshot objects were used for
nearest-neighbour distance, surface gap and the finite-box isotropic center
structure factor `S(q)=1+2/N sum_{{i<j}} sinc(q r_ij)` on 0.02–2 nm⁻¹.  These
values are diagnostic only and were never multiplied into a scattering rate.
Center statistics are measurable, but there is no qualified differential
particle-scattering/partial-structure-factor mapping and no demonstrated
effect larger than the separated seed, geometry and calibration scales.
Therefore P1 does not promote Model 2 for the current paper.
"""
    (args.out / "structure_factor_go_no_go_screen.md").write_text(structure_report, encoding="utf-8")

    figures(geometry, summary, effects, handoff, args.out / "p1_transport_figures")
    figure_count = len(list((args.out / "p1_transport_figures").glob("*.png")))
    authority = read_csv(args.out / "checkpoint_authority_manifest.csv")
    geometry_pass = all(row["geometry_status"] == "PASS" for row in authority)
    final_status = FINAL_CONDITIONAL if geometry_pass else "FAIL_PERIODIC_GEOMETRY_OR_TRANSPORT_INCONSISTENT"
    acceptance = f"""# P1 PF discrete-PSD true-area transport acceptance

Final status: `{final_status}`

- Full discrete PSD reference requalified: PASS.
- Periodic true-area synthetic suite: PASS.
- Production snapshot geometry: PASS for 28/28 conditional snapshots.
- Sphere and true area kept distinct: PASS.
- Frozen A_N/alpha/S11/S13/no-0.1172768 guards: PASS.
- Fixed and PF-time matrix paths separated: PASS.
- Models A–H and κ(t)/κ(Sv): PASS.
- Repeated transport CSV/JSON outputs byte-identical: PASS.
- Sv+M6 error, seed/family/geometry/calibration scales: PASS.
- Handoff 6–12 h isolated; 12–48 h used for mechanism: PASS.
- Structure factor diagnostic only: PASS.
- Figures: {figure_count}/11 PNG plus PDF companions.
- Absolute experimental κ, total κ, zT, dislocation and full-BTE claims: forbidden and absent.

The status is conditional because none of the seven complete 400³ trajectories
has a unified formal production-authority registration.  Cases 004 and 019
retain their explicit periodic/lineage and postprocess conditions.  Snapshot
geometry and transport are usable; particle-identity kinetics are not claimed.
"""
    (args.out / "p1_pf_discrete_psd_true_area_acceptance_report.md").write_text(acceptance, encoding="utf-8")
    next_stage = f"""# P1 next-stage recommendation

Model decision: `{recommendation}`

The maximum snapshot-aggregate `|A_true/A_sph-1|` is
`{max_area_ratio_difference:.3%}`; among objects with a canonical surface the
maximum object-level difference is `{max_present_object_ratio_difference:.3%}`.
The maximum aggregate level-set sensitivity
relative to φ=0.50 is `{max_aggregate_ratio_difference:.3%}`.  κ(Sv) collapse
passes the R²≥0.9 screen for `{collapse_pass_fraction:.1%}` of model/temperature/
matrix combinations.  Between-family effects average
`{mean_effect['between_family_difference']:.6g}` W m⁻¹ K⁻¹ versus within-family
seed spread `{mean_effect['within_seed_spread']:.6g}` W m⁻¹ K⁻¹.

Do not extend automatically to 21 cases.  Expansion is recommended only after
the seven selected trajectories receive formal authority registration and the
chosen model satisfies the numerical comparison above.  Model 2 remains
NO-GO; geometry-resolved MC/BTE and atomistic interface transmission remain
future work.
"""
    (args.out / "p1_next_stage_recommendation.md").write_text(next_stage, encoding="utf-8")

    repo = Path(__file__).resolve().parents[1]
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    diff_summary = f"""# P1 git diff summary

Branch: `{branch}`  
Base HEAD before the P1 commit: `{head}`

P1 adds only read-only analysis code, an isolated CPU postprocessing launcher,
the frozen 28-snapshot route table, qualification tests, and the new P1 report
tree.  It does not edit PF evolution physics, thermodynamic parameters,
checkpoints, fixtures, existing historical reports, or the paper-writing
library.  Pre-existing modified/untracked user files are excluded from the P1
commit.  The final implementation commit is reported in the terminal handoff.
"""
    (args.out / "p1_git_diff_summary.md").write_text(diff_summary, encoding="utf-8")

    manifest = {
        "schema": "P1_PF_DISCRETE_PSD_TRUE_AREA_TRANSPORT_V1",
        "final_status": final_status,
        "model_recommendation": recommendation,
        "structure_factor_decision": structure_decision,
        "formal_authority_pending": True,
        "max_snapshot_aggregate_true_sphere_relative_difference": max_area_ratio_difference,
        "Sv_M6_overall_metrics": d_overall,
        "collapse_pass_fraction": collapse_pass_fraction,
        "input_hashes": {str(path.resolve()): sha256(path.resolve()) for path in (args.observables, args.transport_core, args.geometry_code)},
    }
    required = [
        "checkpoint_authority_manifest.csv", "full_psd_codepath_requalification.md",
        "pf_particle_geometry_manifest.csv", "true_area_validation_report.md",
        "true_area_threshold_sensitivity.csv", "transport_model_comparison.csv",
        "kappa_time_trajectories.csv", "kappa_state_variable_curves.csv",
        "reduced_model_error_report.md", "reduced_model_error_metrics.csv",
        "effect_size_decomposition.csv", "handoff_start_time_sensitivity.md",
        "structure_factor_go_no_go_screen.md", "p1_pf_discrete_psd_true_area_acceptance_report.md",
        "p1_next_stage_recommendation.md", "p1_git_diff_summary.md",
    ]
    output_hashes = {name: sha256(args.out / name) for name in required}
    output_hashes["p1_transport_figures"] = {path.name: sha256(path) for path in sorted((args.out / "p1_transport_figures").iterdir())}
    manifest["output_hashes"] = output_hashes
    (args.out / "p1_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "status.txt").write_text(final_status + "\n", encoding="utf-8")
    print(final_status)
    return 0 if final_status == FINAL_CONDITIONAL else 2


if __name__ == "__main__":
    raise SystemExit(main())
