#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Postprocess Step39 GP interface-only scan.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--root-result-dir", required=True)
    p.add_argument("--dx-nm", type=float, default=0.1)
    p.add_argument("--grid", type=int, default=96)
    p.add_argument("--cases", nargs="+", required=True)
    return p.parse_args()


def h_of_eta(arr):
    return np.where(
        arr <= 0.0,
        0.0,
        np.where(arr >= 1.0, 1.0, arr * arr * arr * (6.0 * arr * arr - 15.0 * arr + 10.0)),
    )


def read_scalar_vtk(path: Path, expected_count: int) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    marker = "LOOKUP_TABLE default"
    idx = text.find(marker)
    if idx < 0:
        raise ValueError(f"LOOKUP_TABLE default not found in {path}")
    arr = np.fromstring(text[idx + len(marker):], sep=" ")
    if arr.size != expected_count:
        raise ValueError(f"{path} expected {expected_count} values, got {arr.size}")
    return arr


def step_to_path(vtk_dir: Path, stem: str, step: int) -> Path | None:
    if step == 0 and stem == "eta":
        for cand in (vtk_dir / "eta_0.vtk", vtk_dir / "eta_init.vtk"):
            if cand.exists():
                return cand
    cand = vtk_dir / f"{stem}_{step}.vtk"
    return cand if cand.exists() else None


def periodic_delta(a: np.ndarray, center: float, L: float) -> np.ndarray:
    d = a - center
    return d - np.round(d / L) * L


def make_r_grid(n: int, dx_nm: float) -> np.ndarray:
    xs = np.arange(n, dtype=np.float64) * dx_nm
    center = 0.5 * n * dx_nm
    L = n * dx_nm
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    r = np.sqrt(
        periodic_delta(X, center, L) ** 2
        + periodic_delta(Y, center, L) ** 2
        + periodic_delta(Z, center, L) ** 2
    )
    return r.reshape(-1)


def write_report(out_path: Path, summary_rows: list[dict]):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Step 39 GP Interface-Only Scan Report\n\n")
        f.write("## Summary\n")
        for row in summary_rows:
            f.write(f"\n### {row['case']}\n")
            f.write(f"- `gamma = {row['gamma_J_m2']:.2f} J/m^2`\n")
            f.write(f"- `W_eta = {row['W_eta_J_m3']:.6e} J/m^3`\n")
            f.write(f"- `kappa_eta = {row['kappa_eta_J_m']:.6e} J/m`\n")
            f.write(f"- `R_eff_h: {row['R_eff_h_init_nm']:.6f} -> {row['R_eff_h_final_nm']:.6f} nm`\n")
            f.write(f"- `relative R change 5000->10000 = {row['relative_reff_change_5000_to_10000']:.6e}`\n")
            f.write(f"- `max_abs(dt*divJ) = {row['max_abs_dt_divJ_overall']:.6e}`\n")
            f.write(f"- `xB_range_final = [{row['xB_min_final']:.6e}, {row['xB_max_final']:.6e}]`\n")
            f.write(f"- `drift_final = {row['total_relative_drift_final']:.6e}`\n")
            f.write(f"- `classification = {row['classification']}`\n")
        best = min(summary_rows, key=lambda r: (0 if r["classification"] == "gp_zone_like" else 1, r["relative_reff_change_5000_to_10000"]))
        f.write("\n## Conclusion\n")
        f.write(f"- Best Step39 interface-only candidate: `{best['case']}`\n")
        if any(r["classification"] == "gp_zone_like" for r in summary_rows):
            f.write("- At least one `W_eta/kappa_eta` candidate shows GP-zone-like slowing growth without elastic penalty.\n")
        elif any(r["classification"] == "over_suppressed" for r in summary_rows):
            f.write("- Interface-only scan reaches suppression before clear self-limited GP behavior.\n")
        else:
            f.write("- All interface-only cases remain precipitate-like; Step40 elastic scan is required.\n")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    root_result_dir = Path(args.root_result_dir)
    total = args.grid ** 3
    dV = args.dx_nm ** 3
    summary_rows = []

    for case in args.cases:
        case_dir = root_result_dir / case
        vtk_dir = case_dir / "vtk_snapshots"
        diag_rows = list(csv.DictReader(open(case_dir / "dynamics_mass_diagnostics.csv", newline="", encoding="utf-8")))
        mass_summary = json.load(open(case_dir / "mass_drift_summary.json", encoding="utf-8"))
        meta = json.load(open(case_dir / "scan_meta.json", encoding="utf-8"))
        diag_map = {int(float(r["step"])): r for r in diag_rows}

        snap_steps = [0, 1000, 5000, 10000]
        ts_rows = []
        prev_R = None
        prev_step = None
        rel_5k_10k = math.nan
        row5000 = None
        row10000 = None
        for step in snap_steps:
            eta_path = step_to_path(vtk_dir, "eta", step)
            xb_path = step_to_path(vtk_dir, "xB", step)
            if not eta_path or not xb_path:
                continue
            eta = read_scalar_vtk(eta_path, total)
            xb = read_scalar_vtk(xb_path, total)
            h = h_of_eta(eta)
            V_h = float(h.sum() * dV)
            R_eff = float(((3.0 * V_h) / (4.0 * math.pi)) ** (1.0 / 3.0)) if V_h > 0 else 0.0
            row = {
                "step": step,
                "eta_max": float(np.max(eta)),
                "eta_integral": float(np.sum(eta) * dV),
                "V_h_nm3": V_h,
                "R_eff_h_nm": R_eff,
                "xB_min": float(np.min(xb)),
                "xB_max": float(np.max(xb)),
                "dR_eff_h_dt": math.nan,
                "dV_h_dt": math.nan,
            }
            if prev_R is not None:
                row["dR_eff_h_dt"] = (R_eff - prev_R) / (step - prev_step)
                row["dV_h_dt"] = (V_h - ts_rows[-1]["V_h_nm3"]) / (step - prev_step)
            prev_R, prev_step = R_eff, step
            dr = diag_map.get(step)
            if dr:
                row.update({
                    "time_code": float(dr.get("time", "nan") or "nan"),
                    "xB_clip_count_high": int(float(dr.get("xB_clip_count_high", "0") or 0.0)),
                    "xB_clip_count_low": int(float(dr.get("xB_clip_count_low", "0") or 0.0)),
                    "total_relative_drift": (float(dr.get("mean_xBtot_gp_end_step", "nan")) - float(mass_summary["initial_mean_xBtot_gp"])) / max(abs(float(mass_summary["initial_mean_xBtot_gp"])), 1e-30),
                    "gp_closure_error": float(dr.get("gp_closure_error", "nan") or "nan"),
                    "max_abs_dt_divJ": max(abs(float(dr.get("dt_divJ_min", "0") or 0.0)), abs(float(dr.get("dt_divJ_max", "0") or 0.0))),
                    "gp_minus_delta_mu_r_mean": float(dr.get("gp_minus_delta_mu_r_mean", "nan") or "nan"),
                    "gp_minus_delta_mu_r_min": float(dr.get("gp_minus_delta_mu_r_min", "nan") or "nan"),
                    "gp_minus_delta_mu_r_max": float(dr.get("gp_minus_delta_mu_r_max", "nan") or "nan"),
                })
            if step == 5000:
                row5000 = row
            if step == 10000:
                row10000 = row
            ts_rows.append(row)

        if row5000 is not None and row10000 is not None and abs(row5000["R_eff_h_nm"]) > 1e-30:
            rel_5k_10k = (row10000["R_eff_h_nm"] - row5000["R_eff_h_nm"]) / row5000["R_eff_h_nm"]

        # interface-term means from final snapshots if present
        final_terms = {}
        for stem, key in [
            ("eta_rhs_chem_10000.vtk", "eta_rhs_chem_interface_mean"),
            ("eta_rhs_dw_10000.vtk", "eta_rhs_dw_interface_mean"),
            ("eta_rhs_grad_10000.vtk", "eta_rhs_grad_interface_mean"),
            ("eta_rhs_elastic_10000.vtk", "eta_rhs_elastic_interface_mean"),
            ("eta_rhs_net_explicit_10000.vtk", "eta_rhs_net_interface_mean"),
        ]:
            p = vtk_dir / stem
            if p.exists():
                field = read_scalar_vtk(p, total)
                eta = read_scalar_vtk(vtk_dir / "eta_10000.vtk", total)
                mask = (eta > 0.1) & (eta < 0.9)
                final_terms[key] = float(np.mean(field[mask])) if np.any(mask) else math.nan
            else:
                final_terms[key] = math.nan

        ts_path = out_dir / f"{case}_timeseries.csv"
        with ts_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ts_rows[0].keys()) + list(final_terms.keys()))
            w.writeheader()
            for row in ts_rows:
                merged = dict(row)
                merged.update(final_terms)
                w.writerow(merged)

        init = ts_rows[0]
        final = ts_rows[-1]
        classification = "precipitate_like"
        if final["xB_min"] < 1e-6 or int(final.get("xB_clip_count_high", 0)) > 0 or int(final.get("xB_clip_count_low", 0)) > 0:
            classification = "unstable_clipping"
        elif final["R_eff_h_nm"] < 5.0 and math.isfinite(rel_5k_10k) and rel_5k_10k < 0.10 and math.isfinite(final["dR_eff_h_dt"]) and final["dR_eff_h_dt"] < ts_rows[1]["dR_eff_h_dt"]:
            classification = "gp_zone_like"
        elif final["R_eff_h_nm"] <= init["R_eff_h_nm"] * 1.05:
            classification = "over_suppressed"

        summary_rows.append({
            "case": case,
            "gamma_J_m2": meta["gamma_J_m2"],
            "W_eta_J_m3": meta["W_eta_J_m3"],
            "kappa_eta_J_m": meta["kappa_eta_J_m"],
            "R_eff_h_init_nm": init["R_eff_h_nm"],
            "R_eff_h_final_nm": final["R_eff_h_nm"],
            "V_h_init_nm3": init["V_h_nm3"],
            "V_h_final_nm3": final["V_h_nm3"],
            "eta_max_init": init["eta_max"],
            "eta_max_final": final["eta_max"],
            "xB_min_final": final["xB_min"],
            "xB_max_final": final["xB_max"],
            "relative_reff_change_5000_to_10000": rel_5k_10k,
            "dR_eff_h_dt_final": final["dR_eff_h_dt"],
            "xB_clip_count_high_total": int(sum(r.get("xB_clip_count_high", 0) for r in ts_rows)),
            "xB_clip_count_low_total": int(sum(r.get("xB_clip_count_low", 0) for r in ts_rows)),
            "total_relative_drift_final": float(mass_summary["total_relative_drift"]),
            "gp_closure_error_final": float(diag_rows[-1].get("gp_closure_error", "nan") or "nan"),
            "max_abs_dt_divJ_overall": max(
                max(abs(float(r.get("dt_divJ_min", "0") or 0.0)), abs(float(r.get("dt_divJ_max", "0") or 0.0)))
                for r in diag_rows
            ),
            "gp_minus_delta_mu_r_mean_final": final.get("gp_minus_delta_mu_r_mean", math.nan),
            "gp_minus_delta_mu_r_min_final": final.get("gp_minus_delta_mu_r_min", math.nan),
            "gp_minus_delta_mu_r_max_final": final.get("gp_minus_delta_mu_r_max", math.nan),
            "eta_rhs_chem_interface_mean_final": final_terms["eta_rhs_chem_interface_mean"],
            "eta_rhs_dw_interface_mean_final": final_terms["eta_rhs_dw_interface_mean"],
            "eta_rhs_grad_interface_mean_final": final_terms["eta_rhs_grad_interface_mean"],
            "eta_rhs_elastic_interface_mean_final": final_terms["eta_rhs_elastic_interface_mean"],
            "eta_rhs_net_interface_mean_final": final_terms["eta_rhs_net_interface_mean"],
            "classification": classification,
        })

    summary_csv = out_dir / "step39_final_key_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    write_report(out_dir / "reports/step_reports/STEP39_GP_INTERFACE_ONLY_SCAN_REPORT.md", summary_rows)


if __name__ == "__main__":
    main()
