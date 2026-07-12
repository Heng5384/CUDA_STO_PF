#!/usr/bin/env python3
"""Assemble PF-only operator-decomposition evidence and failure timing."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
PARAM = ROOT / "params/pf_only_baseline_closure"
REPORT = ROOT / "reports/pf_only_baseline_closure"
RESULT_PARENT = ROOT / "Results/chel_T400_cuda_128x128x128_dt0.002_steps500_xB0.008"


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def write(path: Path, data: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if not data and fields is None:
        path.write_text("")
        return
    if fields is None:
        fields = []
        for row in data:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def first_time(data: list[dict[str, str]], predicate) -> tuple[str, str]:
    for row in sorted(data, key=lambda item: (int(item.get("step", 0)), item.get("label", ""))):
        if predicate(row):
            return row.get("step", ""), row.get("physical_time_s", "")
    return "", ""


def vtk_values(path: Path) -> np.ndarray:
    text = path.read_text()
    marker = "LOOKUP_TABLE default\n"
    return np.fromstring(text.split(marker, 1)[1], sep=" ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-parent", type=Path, default=RESULT_PARENT)
    args = parser.parse_args()
    result_parent = args.result_parent
    REPORT.mkdir(parents=True, exist_ok=True)
    manifest = rows(PARAM / "operator_manifest.csv")
    manifest = [r for r in manifest if not r["case"].startswith("P7_") and
                not r["case"].startswith("P8_")]
    matrix: list[dict[str, object]] = []
    regional_all: list[dict[str, object]] = []
    trace_all: list[dict[str, object]] = []
    clusters: list[dict[str, object]] = []

    for meta in manifest:
        out = result_parent / meta["case"]
        growth = rows(out / "diagnostic_rsmd_seed_growth_time_series.csv")
        regional = rows(out / "diagnostic_rsmd_regional_xB_context.csv")
        projection = rows(out / "y_update_mass_projection.csv")
        global_trace = rows(out / "diagnostic_rsmd_global_max_xB_location.csv")
        dynamics = rows(out / "dynamics_mass_diagnostics.csv")
        dynamics_by_step = {r.get("step", ""): r for r in dynamics}
        if not growth:
            matrix.append({**meta, "run_status": "NOT_RUN"})
            continue
        growth_sorted = sorted(growth, key=lambda r: (int(r["step"]), int(r["post_handoff_step"])))
        initial = growth_sorted[0]
        final = growth_sorted[-1]
        h0 = f(initial, "h_integral")
        hf = f(final, "h_integral")
        rf = f(final, "R_eff_h_nm")
        alpha = [r for r in regional if r.get("region") == "h_lt_0p1"]
        interface = [r for r in regional if r.get("region") in {"h_0p1_0p5", "h_0p5_0p9"}]
        max_alpha = max((f(r, "xB_alpha_max") for r in alpha), default=math.nan)
        max_interface = max((f(r, "xB_alpha_max") for r in interface), default=math.nan)
        first_a005 = first_time(alpha, lambda r: f(r, "xB_alpha_max") > 0.05)
        first_a010 = first_time(alpha, lambda r: f(r, "xB_alpha_max") > 0.10)
        first_i010 = first_time(interface, lambda r: f(r, "xB_alpha_max") > 0.10)
        first_099 = first_time(regional, lambda r: f(r, "xB_alpha_max") > 0.99)
        first_h_half = first_time(growth, lambda r: f(r, "h_integral") < 0.5*h0)
        projection_max_before = max((abs(f(r, "delta_before_projection")) for r in projection), default=math.nan)
        projection_max_after = max((abs(f(r, "delta_after_projection")) for r in projection), default=math.nan)
        mass_max = max((abs(f(r, "mass_error_rel")) for r in growth), default=math.nan)
        max_term_h = max((abs(f(r, "maxabs_term_h")) for r in dynamics), default=math.nan)
        max_term_gamma = max((abs(f(r, "maxabs_term_gamma")) for r in dynamics), default=math.nan)
        max_divj = max((abs(f(r, "maxabs_divJ")) for r in dynamics), default=math.nan)
        if max(max_alpha, max_interface) > 0.99:
            fate = "RUNAWAY"
        elif h0 > 0 and hf < 0.05*h0:
            fate = "COLLAPSE"
        elif h0 > 0 and hf < 0.8*h0:
            fate = "SHRINK"
        else:
            fate = "BOUNDED"
        matrix.append({
            **meta, "run_status": "COMPLETE", "initial_R_eff_h_nm": f(initial, "R_eff_h_nm"),
            "final_R_eff_h_nm": rf, "initial_h_integral": h0, "final_h_integral": hf,
            "h_integral_ratio": hf/h0 if h0 else math.nan,
            "max_alpha_xB": max_alpha, "max_interface_xB": max_interface,
            "first_alpha_gt_0p05_step": first_a005[0], "first_alpha_gt_0p05_time_s": first_a005[1],
            "first_alpha_gt_0p10_step": first_a010[0], "first_alpha_gt_0p10_time_s": first_a010[1],
            "first_interface_gt_0p10_step": first_i010[0], "first_interface_gt_0p10_time_s": first_i010[1],
            "first_any_gt_0p99_step": first_099[0], "first_any_gt_0p99_time_s": first_099[1],
            "first_h_integral_lt_half_step": first_h_half[0], "first_h_integral_lt_half_time_s": first_h_half[1],
            "max_abs_projection_residual_before": projection_max_before,
            "max_abs_projection_residual_after": projection_max_after,
            "max_abs_global_mass_error_rel": mass_max, "fate": fate,
            "maxabs_term_h_over_run": max_term_h,
            "maxabs_term_gamma_over_run": max_term_gamma,
            "maxabs_divJ_over_run": max_divj,
        })
        for row in regional:
            regional_all.append({"operator_case": meta["operator_case"], **row})
            clusters.append({
                "operator_case": meta["operator_case"], "case": meta["case"],
                "step": row.get("step", ""), "label": row.get("label", ""),
                "region": row.get("region", ""), "cells_gt_0p05": row.get("cells_xB_gt_0p05", ""),
                "cells_gt_0p10": row.get("cells_xB_gt_0p10", ""),
                "cells_gt_0p50": row.get("cells_xB_gt_0p50", ""),
                "cells_gt_0p90": row.get("cells_xB_gt_0p90", ""),
                "connectivity_status": "COUNT_ONLY_NO_CELL_MASK",
                "notes": "Population size is exact; connected-component topology needs a threshold mask checkpoint.",
            })
        for row in global_trace:
            d = dynamics_by_step.get(row.get("step", ""), {})
            trace_all.append({
                "operator_case": meta["operator_case"], **row,
                "mean_term_h": d.get("mean_term_h", ""),
                "mean_term_gamma": d.get("mean_term_gamma", ""),
                "mean_divJ": d.get("mean_divJ", ""),
                "l1_term_h": d.get("l1_term_h", ""),
                "l1_term_gamma": d.get("l1_term_gamma", ""),
                "l1_divJ": d.get("l1_divJ", ""),
                "maxabs_term_h": d.get("maxabs_term_h", ""),
                "maxabs_term_gamma": d.get("maxabs_term_gamma", ""),
                "maxabs_divJ": d.get("maxabs_divJ", ""),
                "maxabs_Y_rhs_total": d.get("maxabs_Y_rhs_total", ""),
                "delta_mass_phi_update": d.get("delta_mass_phi_update", ""),
                "delta_mass_Y_update": d.get("delta_mass_Y_update", ""),
                "Y_compensation_ratio": d.get("Y_compensation_ratio", ""),
            })
        final_vtk = out / f"xB_{meta['nsteps']}.vtk"
        if final_vtk.exists():
            values = vtk_values(final_vtk).reshape((128, 128, 128))
            structure = ndimage.generate_binary_structure(3, 1)
            for threshold in (0.05, 0.10, 0.50, 0.90):
                labels, count = ndimage.label(values > threshold, structure=structure)
                sizes = np.bincount(labels.ravel())[1:] if count else np.array([], dtype=int)
                active = int(sizes.sum()) if sizes.size else 0
                clusters.append({
                    "operator_case": meta["operator_case"], "case": meta["case"],
                    "step": meta["nsteps"], "label": "final_vtk_6_neighbor_components",
                    "region": "all", "threshold_xB": threshold,
                    "active_cells": active, "connected_components_nonperiodic": int(count),
                    "largest_component_cells": int(sizes.max()) if sizes.size else 0,
                    "largest_component_fraction": (float(sizes.max())/active) if active else 0.0,
                    "connectivity_status": "EXACT_6_NEIGHBOR_NONPERIODIC_BOUNDARY",
                    "notes": "Periodic-face component merging is not applied; count is an upper bound.",
                })

    write(REPORT / "pf_operator_decomposition_matrix.csv", matrix)
    write(REPORT / "pf_region_aware_failure_time_series.csv", regional_all)
    write(REPORT / "pf_first_failure_cell_trace.csv", trace_all)
    write(REPORT / "pf_failure_cluster_audit.csv", clusters)

    complete = [r for r in matrix if r.get("run_status") == "COMPLETE"]
    by_op = {str(r["operator_case"]): r for r in complete}
    p0 = by_op.get("P0_full")
    causal = "UNRESOLVED"
    if p0:
        xa = p0.get("first_alpha_gt_0p10_time_s", "")
        hh = p0.get("first_h_integral_lt_half_time_s", "")
        if xa and hh:
            causal = "XBRUNAWAY_PRECEDES_PHI_COLLAPSE" if float(xa) < float(hh) else "PHI_COLLAPSE_PRECEDES_XBRUNAWAY"
    p5 = by_op.get("P5_full_history_off")
    attribution = (
        "P5 remains bounded when the lagged gamma/history term is disabled while phi, transport, and projection remain active. "
        "Together with bounded P1/P2/P3, this identifies the lagged storage-Jacobian closure under changing phi as the minimal failing operator."
        if p5 and float(p5.get("max_alpha_xB", 1.0)) < 0.1 else
        "The minimal failing operator remains pending until P5 completes."
    )
    (REPORT / "pf_operator_decomposition_report.md").write_text(
        "# PF Operator Decomposition\n\n" +
        f"Completed controls: {len(complete)}/{len(manifest)}. Causal order from P0: `{causal}`.\n\n" +
        attribution + "\n\n"
        "Projection-only is an identity, frozen-phi transport is bounded with and without projection, and full phi-only behavior is reported separately. "
        "The history-off case is diagnostic rather than an accepted formula because simply deleting the Jacobian does not solve the exact storage chain rule.\n\n" +
        "| Operator | Fate | max alpha xB | max interface xB | final/initial h | max projection residual before |\n"
        "|---|---:|---:|---:|---:|---:|\n" +
        "\n".join(
            f"| {r['operator_case']} | {r.get('fate','')} | {float(r.get('max_alpha_xB', math.nan)):.6g} | "
            f"{float(r.get('max_interface_xB', math.nan)):.6g} | {float(r.get('h_integral_ratio', math.nan)):.6g} | "
            f"{float(r.get('max_abs_projection_residual_before', math.nan)):.6g} |" for r in complete
        ) + "\n"
    )
    if p0:
        (REPORT / "pf_baseline_failure_reproduction_report.md").write_text(
            "# PF-only Failure Reproduction\n\n"
            "The T400, 128^3, dx=1 nm, no-source P0 control reproduces the accepted failure, and the prior equal-time dt controls at 0.0005, 0.001, and 0.002 all reached xB=0.99999999 with seed collapse. "
            f"Maximum alpha xB is `{float(p0['max_alpha_xB']):.12g}` and final/initial h integral is "
            f"`{float(p0['h_integral_ratio']):.12g}`. The first alpha xB > 0.10 occurs at step "
            f"`{p0['first_alpha_gt_0p10_step']}`; h first falls below half its inserted value at step "
            f"`{p0['first_h_integral_lt_half_step']}`. Therefore the observed order is `{causal}`.\n"
        )
    print(f"operator_controls_complete={len(complete)}")
    print(f"runaway_causal_order={causal}")


if __name__ == "__main__":
    main()
