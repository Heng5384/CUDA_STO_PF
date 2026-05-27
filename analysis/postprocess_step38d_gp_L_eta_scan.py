#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def parse_args():
    p = argparse.ArgumentParser(description="Postprocess clean GP eta kinetic scan.")
    p.add_argument("--root-result-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--grid", type=int, default=96)
    p.add_argument("--dx-nm", type=float, default=0.1)
    p.add_argument("--far-field-radius-nm", type=float, default=3.0)
    return p.parse_args()


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


def h_of_eta(arr: np.ndarray) -> np.ndarray:
    return np.where(
        arr <= 0.0,
        0.0,
        np.where(arr >= 1.0, 1.0, arr * arr * arr * (6.0 * arr * arr - 15.0 * arr + 10.0)),
    )


def logit_clamped(x: np.ndarray) -> np.ndarray:
    xc = np.clip(x, 1.0e-12, 1.0 - 1.0e-12)
    return np.log(xc / (1.0 - xc))


def build_radius_mask(grid: int, dx_nm: float, far_field_radius_nm: float):
    coords = (np.arange(grid) + 0.5) * dx_nm
    cx = cy = cz = 0.5 * grid * dx_nm
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    return R >= far_field_radius_nm


def summarize_case(case_dir: Path, mask_far: np.ndarray, grid: int, dx_nm: float):
    total = grid ** 3
    dV = dx_nm ** 3
    diag_path = case_dir / "dynamics_mass_diagnostics.csv"
    mass_path = case_dir / "mass_drift_summary.json"
    kin_path = case_dir / "gp_eta_kinetic_reference_diagnostics.csv"
    meta_path = case_dir / "scan_meta.json"
    vtk_dir = case_dir / "vtk_snapshots"

    diag_rows = list(csv.DictReader(open(diag_path, newline="", encoding="utf-8")))
    mass_summary = json.load(open(mass_path, encoding="utf-8"))
    kin = next(csv.DictReader(open(kin_path, newline="", encoding="utf-8")))
    meta = json.load(open(meta_path, encoding="utf-8"))

    ts_rows = []
    prev_t = None
    prev_R = None
    for row in diag_rows:
        step = int(float(row["step"]))
        time_code = float(row["time"])
        eta_path = vtk_dir / f"eta_{step}.vtk"
        xb_path = vtk_dir / f"xB_{step}.vtk"
        if not eta_path.exists() or not xb_path.exists():
            continue
        eta = read_scalar_vtk(eta_path, total)
        xb = read_scalar_vtk(xb_path, total)
        h = h_of_eta(eta)
        V_h = float(h.sum() * dV)
        R_eff = float(((3.0 * V_h) / (4.0 * math.pi)) ** (1.0 / 3.0)) if V_h > 0.0 else 0.0
        eta_far = float(np.mean(eta[mask_far]))
        eta_mean = float(np.mean(eta))
        Y = logit_clamped(xb)
        drift = (
            float(row["mean_xBtot_gp_end_step"]) - float(mass_summary["initial_mean_xBtot_gp"])
        ) / max(abs(float(mass_summary["initial_mean_xBtot_gp"])), 1.0e-30)
        ts = {
            "case": meta["case"],
            "f_eta": meta["f_eta"],
            "gp_L_eta": meta["gp_L_eta"],
            "L_eta_diff_ref_code": meta["L_eta_diff_ref_code"],
            "ratio_current_to_diff_ref": float(kin["ratio_current_to_diff_ref"]),
            "step": step,
            "time_code": time_code,
            "eta_max": float(np.max(eta)),
            "eta_mean": eta_mean,
            "eta_far_field": eta_far,
            "eta_localization_gap": float(np.max(eta) - eta_far),
            "V_h_nm3": V_h,
            "R_eff_h_nm": R_eff,
            "xB_min": float(np.min(xb)),
            "xB_max": float(np.max(xb)),
            "Y_min": float(np.min(Y)),
            "Y_max": float(np.max(Y)),
            "total_xBtot_drift": drift,
            "xB_clip_count_low": int(float(row.get("xB_clip_count_low", "0") or 0.0)),
            "xB_clip_count_high": int(float(row.get("xB_clip_count_high", "0") or 0.0)),
            "max_abs_dt_divJ": max(abs(float(row.get("dt_divJ_min", "0") or 0.0)),
                                   abs(float(row.get("dt_divJ_max", "0") or 0.0))),
            "S_eta_W": float(kin["S_eta_W"]),
            "S_eta_grad": float(kin["S_eta_grad"]),
            "S_eta_total": float(kin["S_eta_total"]),
            "gp_W_eta_input_source": kin["gp_W_eta_input_source"],
            "gp_kappa_eta_input_source": kin["gp_kappa_eta_input_source"],
            "gp_W_eta_kernel_used": float(kin["gp_W_eta_kernel_used"]),
            "gp_kappa_eta_kernel_used": float(kin["gp_kappa_eta_kernel_used"]),
            "gp_W_eta_kernel_vs_ref_ratio": float(kin["gp_W_eta_kernel_vs_ref_ratio"]),
            "gp_kappa_eta_kernel_vs_ref_ratio": float(kin["gp_kappa_eta_kernel_vs_ref_ratio"]),
            "NaN_or_Inf": int(not np.isfinite(eta).all() or not np.isfinite(xb).all()),
            "dR_eff_h_dt": math.nan,
        }
        if prev_t is not None and time_code > prev_t:
            ts["dR_eff_h_dt"] = (R_eff - prev_R) / (time_code - prev_t)
        prev_t, prev_R = time_code, R_eff
        ts_rows.append(ts)

    final = ts_rows[-1]
    summary = {
        "case": meta["case"],
        "f_eta": meta["f_eta"],
        "gp_L_eta": meta["gp_L_eta"],
        "L_eta_diff_ref_code": meta["L_eta_diff_ref_code"],
        "ratio_current_to_diff_ref": float(kin["ratio_current_to_diff_ref"]),
        "gp_W_eta_input_source": kin["gp_W_eta_input_source"],
        "gp_kappa_eta_input_source": kin["gp_kappa_eta_input_source"],
        "gp_W_eta_kernel_used": float(kin["gp_W_eta_kernel_used"]),
        "gp_kappa_eta_kernel_used": float(kin["gp_kappa_eta_kernel_used"]),
        "gp_W_eta_kernel_vs_ref_ratio": float(kin["gp_W_eta_kernel_vs_ref_ratio"]),
        "gp_kappa_eta_kernel_vs_ref_ratio": float(kin["gp_kappa_eta_kernel_vs_ref_ratio"]),
        "S_eta_W": float(kin["S_eta_W"]),
        "S_eta_grad": float(kin["S_eta_grad"]),
        "S_eta_total": float(kin["S_eta_total"]),
        "eta_max_final": final["eta_max"],
        "eta_mean_final": final["eta_mean"],
        "eta_far_field_final": final["eta_far_field"],
        "eta_localization_gap_final": final["eta_localization_gap"],
        "R_eff_h_final_nm": final["R_eff_h_nm"],
        "xB_min_final": final["xB_min"],
        "xB_max_final": final["xB_max"],
        "Y_min_final": final["Y_min"],
        "Y_max_final": final["Y_max"],
        "total_xBtot_drift_final": final["total_xBtot_drift"],
        "xB_clip_count_low_total": sum(r["xB_clip_count_low"] for r in ts_rows),
        "xB_clip_count_high_total": sum(r["xB_clip_count_high"] for r in ts_rows),
        "NaN_or_Inf": max(r["NaN_or_Inf"] for r in ts_rows),
        "global_ordering_warning": int(max(r["eta_far_field"] for r in ts_rows) > 0.05),
        "max_abs_dt_divJ_overall": max(r["max_abs_dt_divJ"] for r in ts_rows),
        "classification": "",
    }

    if summary["NaN_or_Inf"] or summary["xB_clip_count_low_total"] > 0 or summary["xB_clip_count_high_total"] > 0:
        summary["classification"] = "unstable"
    elif abs(summary["total_xBtot_drift_final"]) > 1.0e-2 or summary["max_abs_dt_divJ_overall"] > 1.0:
        summary["classification"] = "unstable"
    elif summary["global_ordering_warning"]:
        summary["classification"] = "global_ordering_warning"
    else:
        summary["classification"] = "acceptable"

    return summary, ts_rows


def write_plot(out_path: Path, x, ys, ylabel: str, title: str):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, y in ys:
        ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.2, label=label)
    ax.set_xlabel("time (code)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    root = Path(args.root_result_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_dir = out_dir / "step38d_timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)
    mask_far = build_radius_mask(args.grid, args.dx_nm, args.far_field_radius_nm).reshape(-1)

    summaries = []
    all_ts = []
    for case in args.cases:
        case_dir = root / case
        summary, ts_rows = summarize_case(case_dir, mask_far, args.grid, args.dx_nm)
        summaries.append(summary)
        all_ts.extend(ts_rows)
        with (ts_dir / f"{case}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ts_rows[0].keys()))
            w.writeheader()
            w.writerows(ts_rows)

    summary_path = out_dir / "step38d_gp_L_eta_scan_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    all_ts_path = out_dir / "step38d_gp_L_eta_scan_timeseries.csv"
    with all_ts_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_ts[0].keys()))
        w.writeheader()
        w.writerows(all_ts)

    if plt is not None:
        grouped = {}
        for row in all_ts:
            grouped.setdefault(row["case"], []).append(row)
        time_keys = ["eta_max", "eta_far_field", "xB_min", "xB_max", "total_xBtot_drift"]
        for metric in time_keys:
            ys = []
            x = None
            for case, rows in grouped.items():
                rows = sorted(rows, key=lambda r: r["time_code"])
                if x is None:
                    x = [r["time_code"] for r in rows]
                ys.append((case, [r[metric] for r in rows]))
            write_plot(out_dir / f"step38d_{metric}.png", x, ys, metric, f"Step38d {metric} vs time")

    report_path = out_dir / "STEP38D_GP_L_ETA_REFERENCE_SCAN_REPORT.md"
    stable = [s for s in summaries if s["classification"] == "acceptable"]
    rec = stable[0] if stable else None
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Step38D GP L_eta Reference Scan Report\n\n")
        f.write("This scan uses a clean GP diagnostic setup with `observed_gp_diffuse`, `elastic=off`, and explicit `gp_W_eta_phys/gp_kappa_eta_phys` inputs.\n\n")
        f.write("## Cases\n")
        for s in summaries:
            f.write(
                f"- `{s['case']}`: `f_eta={s['f_eta']}`, `gp_L_eta={s['gp_L_eta']:.6e}`, "
                f"`ratio_current_to_diff_ref={s['ratio_current_to_diff_ref']:.6e}`, "
                f"`classification={s['classification']}`, `eta_far_field_final={s['eta_far_field_final']:.6e}`, "
                f"`xB_final=[{s['xB_min_final']:.6e}, {s['xB_max_final']:.6e}]`, "
                f"`drift={s['total_xBtot_drift_final']:.6e}`\n"
            )
        f.write("\n## Recommendation\n")
        if rec is not None:
            f.write(
                f"- Recommended post-STEP38C baseline: `{rec['gp_L_eta']:.6e}` "
                f"(from `f_eta={rec['f_eta']}`), because it remains localized, avoids clipping, "
                f"and keeps drift controlled.\n"
            )
        else:
            f.write("- No acceptable baseline identified in this scan.\n")


if __name__ == "__main__":
    main()
