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
    p = argparse.ArgumentParser(description="Postprocess Step38J STO-SM Eq. S2.18b eta M-ratio scan.")
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


def build_radius_mask(grid: int, dx_nm: float, far_field_radius_nm: float):
    coords = (np.arange(grid) + 0.5) * dx_nm
    cx = cy = cz = 0.5 * grid * dx_nm
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    return R >= far_field_radius_nm


def parse_ratio_from_case_name(case_name: str) -> float:
    mapping = {
        "1em4": 1.0e-4,
        "1em3": 1.0e-3,
        "1em2": 1.0e-2,
        "1em1": 1.0e-1,
        "_10": 1.0e1,
        "_1": 1.0,
    }
    for key, value in mapping.items():
        if case_name.endswith(key):
            return value
    return math.nan


def read_case_dir(case_dir: Path, case_name: str):
    meta_path = case_dir / "scan_meta.json"
    if meta_path.exists():
        meta = json.load(open(meta_path, encoding="utf-8"))
    else:
        meta = {
            "case": case_name,
            "M_ratio": parse_ratio_from_case_name(case_name),
            "gp_L_eta_mode": "sto_S218b",
        }
    kin = next(csv.DictReader(open(case_dir / "gp_eta_S218b_kinetic_diagnostics.csv", newline="", encoding="utf-8")))
    vf_path = next(case_dir.glob("vf_precip_vs_time_*.csv"))
    vf_rows = list(csv.DictReader(open(vf_path, newline="", encoding="utf-8")))
    rhs_path = case_dir / "step38j_rhs_attr_per_step.csv"
    rhs_rows = list(csv.DictReader(open(rhs_path, newline="", encoding="utf-8"))) if rhs_path.exists() else []

    by_step_vf = {int(float(r["step"])): r for r in vf_rows}

    timeseries = []
    clip_count = 0
    nan_inf_flag = 0
    for row in rhs_rows:
        step = int(float(row["step"]))
        vf_row = by_step_vf.get(step, {})
        xB_clip_low = int(float(row.get("xB_clip_low_count", "0") or 0.0))
        xB_clip_high = int(float(row.get("xB_clip_high_count", "0") or 0.0))
        clip_count += xB_clip_low + xB_clip_high
        nan_inf_flag = max(nan_inf_flag, int(float(row.get("nan_inf_flag", "0") or 0.0)))
        timeseries.append({
            "case_name": case_name,
            "M_ratio": float(meta["M_ratio"]),
            "step": step,
            "time_code": float(row["time_code"]),
            "eta_max": float(row.get("eta_max_after", "nan") or "nan"),
            "eta_mean": float(row.get("eta_mean_after", "nan") or "nan"),
            "eta_far_field": float(row.get("eta_far_field_max", "nan") or "nan"),
            "eta_max_minus_far": float(row.get("eta_max_after", "nan") or "nan") - float(row.get("eta_far_field_max", "nan") or "nan"),
            "R_avg_nm": float(vf_row.get("R_avg", "nan") or "nan"),
            "xB_min": float(row.get("xB_min_after", "nan") or "nan"),
            "xB_max": float(row.get("xB_max_after", "nan") or "nan"),
            "Y_min": float(row.get("Y_min_after", "nan") or "nan"),
            "Y_max": float(row.get("Y_max_after", "nan") or "nan"),
            "xBtot_initial": math.nan,
            "xBtot_final": math.nan,
            "xBtot_drift": math.nan,
            "xBtot_rel_drift": float(row.get("xBtot_rel_delta", "nan") or "nan"),
            "clipping_count_step": xB_clip_low + xB_clip_high,
            "nan_inf_flag": nan_inf_flag,
            "eta_bulk_rhs_absmax": float(row.get("eta_rhs_bulk_abs_mean", "nan") or "nan"),
            "phi_chem_rhs_absmax": float(row.get("phi_rhs_chem_abs_max", "nan") or "nan"),
            "eta_bulk_rhs_absmax_over_phi_chem_rhs_absmax": float(
                row.get("eta_rhs_bulk_absmax_over_phi_chem_absmax", "nan") or "nan"
            ),
            "eta_Ldt_bulk_absmax_over_phi_Ldt_chem_absmax": float(
                row.get("eta_Ldt_bulk_absmax_over_phi_Ldt_chem_absmax", "nan") or "nan"
            ),
        })

    final = timeseries[-1]
    summary = {
        "M_ratio": float(meta["M_ratio"]),
        "M_eta": float(kin["M_eta_used"]),
        "Mcrit_eta": float(kin["Mcrit_eta"]),
        "L_eta_full_code": float(kin["L_eta_full_code"]),
        "L_eta_diff_code": float(kin["L_eta_diff_code"]),
        "L_eta_full_over_L_diff": float(kin["L_eta_full_over_L_diff"]),
        "eta_regime_classification": kin["eta_regime_classification"],
        "S_eta_total": float(kin["S_eta_total"]),
        "eta_max_final": final["eta_max"],
        "eta_far_final": final["eta_far_field"],
        "R_avg_final_nm": final["R_avg_nm"],
        "xB_min_final": final["xB_min"],
        "xB_max_final": final["xB_max"],
        "xBtot_drift": final["xBtot_drift"],
        "xBtot_rel_drift": final["xBtot_rel_drift"],
        "clipping_count": clip_count,
        "nan_inf_flag": nan_inf_flag,
        "recommendation_label": "",
    }

    if summary["nan_inf_flag"] or summary["xB_min_final"] <= 0.0 or summary["xB_max_final"] >= 1.0:
        summary["recommendation_label"] = "unstable"
    elif summary["eta_far_final"] > 0.05:
        summary["recommendation_label"] = "global_ordering"
    elif summary["S_eta_total"] > 10.0:
        summary["recommendation_label"] = "aggressive"
    elif summary["S_eta_total"] < 0.1:
        summary["recommendation_label"] = "too_slow"
    elif summary["xB_max_final"] > 0.1 or abs(summary["xBtot_rel_drift"]) > 1.0e-2:
        summary["recommendation_label"] = "aggressive"
    else:
        summary["recommendation_label"] = "baseline_candidate"

    return summary, timeseries


def plot_series(path: Path, grouped: dict, xkey: str, ykey: str, ylabel: str, title: str):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, rows in grouped.items():
        rows = sorted(rows, key=lambda r: r[xkey])
        ax.plot([r[xkey] for r in rows], [r[ykey] for r in rows], marker="o", markersize=2.5, linewidth=1.2, label=label)
    ax.set_xlabel(xkey)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    root = Path(args.root_result_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ts_dir = out_dir / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_ts = []
    grouped = {}
    for case in args.cases:
        case_dir = root / case
        summary, ts = read_case_dir(case_dir, case)
        summaries.append(summary)
        all_ts.extend(ts)
        grouped[case] = ts
        with (ts_dir / f"{case}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ts[0].keys()))
            w.writeheader()
            w.writerows(ts)

    summary_path = out_dir / "gp_eta_S218b_Mratio_scan_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(sorted(summaries, key=lambda r: r["M_ratio"]))

    if all_ts:
        all_ts_path = out_dir / "gp_eta_S218b_Mratio_scan_timeseries.csv"
        with all_ts_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_ts[0].keys()))
            w.writeheader()
            w.writerows(all_ts)

    plot_series(plots_dir / "eta_max_vs_time.png", grouped, "time_code", "eta_max", "eta_max", "Step38J eta_max vs time")
    plot_series(plots_dir / "eta_far_vs_time.png", grouped, "time_code", "eta_far_field", "eta_far_field", "Step38J eta far field vs time")
    plot_series(plots_dir / "R_avg_vs_time.png", grouped, "time_code", "R_avg_nm", "R_avg (nm)", "Step38J R_avg vs time")
    plot_series(plots_dir / "xB_min_vs_time.png", grouped, "time_code", "xB_min", "xB_min", "Step38J xB_min vs time")
    plot_series(plots_dir / "xB_max_vs_time.png", grouped, "time_code", "xB_max", "xB_max", "Step38J xB_max vs time")
    plot_series(plots_dir / "xBtot_rel_drift_vs_time.png", grouped, "time_code", "xBtot_rel_drift", "xBtot_rel_drift", "Step38J xBtot drift vs time")

    if plt is not None and summaries:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ordered = sorted(summaries, key=lambda r: r["M_ratio"])
        ax.plot([r["M_ratio"] for r in ordered], [r["L_eta_full_over_L_diff"] for r in ordered], marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("M_eta / Mcrit")
        ax.set_ylabel("L_eta_full / L_eta_diff")
        ax.set_title("Step38J L_eta_full/L_eta_diff vs M_ratio")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "Leta_full_over_diff_vs_Mratio.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([r["M_ratio"] for r in ordered], [r["R_avg_final_nm"] for r in ordered], marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("M_eta / Mcrit")
        ax.set_ylabel("final R_avg (nm)")
        ax.set_title("Step38J final R_avg vs M_ratio")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "final_Ravg_vs_Mratio.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
