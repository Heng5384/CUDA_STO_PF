#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def f(row, key, default=float("nan")):
    try:
        v = row.get(key, "")
        return default if v in ("", None) else float(v)
    except Exception:
        return default


def dominant_component(rows, prefix):
    comps = {
        "bulk": max(f(r, f"{prefix}_rhs_bulk_abs_mean", 0.0) for r in rows),
        "double_well": max(f(r, f"{prefix}_rhs_double_well_abs_mean", 0.0) for r in rows),
        "elastic": max(f(r, f"{prefix}_rhs_elastic_abs_mean", 0.0) for r in rows),
    }
    return max(comps, key=comps.get)


def plot_lines(path, rows, series, title, ylabel):
    if plt is None:
        return
    x = [f(r, "step", 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, key in series:
        ax.plot(x, [f(r, key, 0.0) for r in rows], label=label, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scatter(path, rows, x_key, y_key, title, xlabel, ylabel):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.scatter([f(r, x_key, 0.0) for r in rows], [f(r, y_key, 0.0) for r in rows], s=10, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize_case(case_dir: Path):
    meta = json.loads((case_dir / "scan_meta.json").read_text(encoding="utf-8"))
    csv_path = case_dir / "phi_eta_rhs_attribution_per_step.csv"
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"no rows in {csv_path}")

    summary = {
        "case_name": meta["case"],
        "f_eta": meta["f_eta"],
        "gp_L_eta": meta["gp_L_eta"],
        "max_dphi_absmax": max(f(r, "dphi_actual_abs_max", 0.0) for r in rows),
        "max_deta_absmax": max(f(r, "deta_actual_abs_max", 0.0) for r in rows),
        "max_ratio_deta_dphi": max(f(r, "deta_absmax_over_dphi_absmax", 0.0) for r in rows),
        "dominant_eta_rhs_component_by_absmax": dominant_component(rows, "eta"),
        "dominant_eta_mobility_scaled_component_by_absmax":
            max(
                {
                    "bulk": max(f(r, "eta_Ldt_bulk_absmax_over_phi_Ldt_chem_absmax", 0.0) for r in rows),
                    "double_well": max(f(r, "eta_Ldt_dw_absmax_over_phi_Ldt_dw_absmax", 0.0) for r in rows),
                    "elastic": max(f(r, "eta_Ldt_elastic_absmax_over_phi_Ldt_elastic_absmax", 0.0) for r in rows),
                },
                key=lambda k: {
                    "bulk": max(f(r, "eta_Ldt_bulk_absmax_over_phi_Ldt_chem_absmax", 0.0) for r in rows),
                    "double_well": max(f(r, "eta_Ldt_dw_absmax_over_phi_Ldt_dw_absmax", 0.0) for r in rows),
                    "elastic": max(f(r, "eta_Ldt_elastic_absmax_over_phi_Ldt_elastic_absmax", 0.0) for r in rows),
                }[k]
            ),
        "mean_phi_grad_denom_kmax": sum(f(r, "phi_grad_denom_at_kmax", 0.0) for r in rows) / len(rows),
        "mean_eta_grad_denom_kmax": sum(f(r, "eta_grad_denom_at_kmax", 0.0) for r in rows) / len(rows),
        "damping_ratio_phi_over_eta": sum(f(r, "phi_damping_stronger_than_eta", 0.0) for r in rows) / len(rows),
        "max_corr_abs_deta_abs_dxB": max(f(r, "corr_abs_deta_abs_dxB", -1.0) for r in rows),
        "first_step_xB_clip": next((int(f(r, "step", -1)) for r in rows if f(r, "xB_clip_low_count", 0.0) > 0 or f(r, "xB_clip_high_count", 0.0) > 0), -1),
        "recommendation": "review",
    }
    if meta["f_eta"] <= 1e-5 and summary["first_step_xB_clip"] < 0 and max(f(r, "eta_far_field_max", 0.0) for r in rows) < 0.05:
        summary["recommendation"] = "baseline_candidate"
    elif meta["f_eta"] >= 1e-3:
        summary["recommendation"] = "too_aggressive"
    elif meta["f_eta"] >= 1e-4:
        summary["recommendation"] = "aggressive_compare"
    return rows, summary


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for case in args.cases:
        case_dir = Path(case)
        rows, summary = summarize_case(case_dir)
        summaries.append(summary)
        case = summary["case_name"]
        plot_lines(plot_dir / f"{case}_rhs_absmax.png", rows,
                   [("phi_total_rhs", "phi_rhs_total_explicit_abs_max"),
                    ("eta_total_rhs", "eta_rhs_total_explicit_abs_max")],
                   f"{case}: phi vs eta total RHS absmax", "absmax")
        plot_lines(plot_dir / f"{case}_Ldt_rhs_absmax.png", rows,
                   [("phi explicit est", "phi_update_explicit_est_abs_max"),
                    ("eta explicit est", "eta_update_explicit_est_abs_max")],
                   f"{case}: mobility-scaled RHS absmax", "absmax")
        plot_lines(plot_dir / f"{case}_phi_components.png", rows,
                   [("chem", "phi_rhs_chem_abs_max"),
                    ("dw", "phi_rhs_double_well_abs_max"),
                    ("elastic", "phi_rhs_elastic_abs_max")],
                   f"{case}: phi component breakdown", "absmax")
        plot_lines(plot_dir / f"{case}_eta_components.png", rows,
                   [("bulk", "eta_rhs_bulk_abs_max"),
                    ("dw", "eta_rhs_double_well_abs_max"),
                    ("elastic", "eta_rhs_elastic_abs_max")],
                   f"{case}: eta component breakdown", "absmax")
        plot_lines(plot_dir / f"{case}_actual_vs_estimated.png", rows,
                   [("dphi actual", "dphi_actual_abs_max"),
                    ("phi explicit est", "phi_update_explicit_est_abs_max"),
                    ("deta actual", "deta_actual_abs_max"),
                    ("eta explicit est", "eta_update_explicit_est_abs_max")],
                   f"{case}: actual vs estimated update", "absmax")
        plot_lines(plot_dir / f"{case}_grad_denom.png", rows,
                   [("phi denom@kmax", "phi_grad_denom_at_kmax"),
                    ("eta denom@kmax", "eta_grad_denom_at_kmax")],
                   f"{case}: gradient denominator", "denom")
        plot_scatter(plot_dir / f"{case}_corr_deta_dxb.png", rows,
                     "corr_abs_deta_abs_dxB", "deta_actual_abs_max",
                     f"{case}: |deta|-|dxB| correlation", "corr(|deta|,|dxB|)", "deta abs max")
        plot_lines(plot_dir / f"{case}_eta_stacked.png", rows,
                   [("bulk", "eta_rhs_bulk_abs_mean"),
                    ("dw", "eta_rhs_double_well_abs_mean"),
                    ("elastic", "eta_rhs_elastic_abs_mean")],
                   f"{case}: eta component contribution", "abs mean")

    summary_path = out_dir / "phi_eta_rhs_attribution_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fobj:
        w = csv.DictWriter(fobj, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    report = out_dir / "STEP38F_PHI_ETA_RHS_ATTRIBUTION_REPORT.md"
    with report.open("w", encoding="utf-8") as fobj:
        fobj.write("# Phi vs Eta RHS Attribution\n\n")
        for s in summaries:
            fobj.write(
                f"- `{s['case_name']}`: `max_dphi_absmax={s['max_dphi_absmax']:.6e}`, "
                f"`max_deta_absmax={s['max_deta_absmax']:.6e}`, "
                f"`max_ratio_deta_dphi={s['max_ratio_deta_dphi']:.6e}`, "
                f"`dominant_eta_rhs_component={s['dominant_eta_rhs_component_by_absmax']}`, "
                f"`dominant_eta_mobility_scaled_component={s['dominant_eta_mobility_scaled_component_by_absmax']}`, "
                f"`mean_phi_grad_denom_kmax={s['mean_phi_grad_denom_kmax']:.6e}`, "
                f"`mean_eta_grad_denom_kmax={s['mean_eta_grad_denom_kmax']:.6e}`, "
                f"`max_corr_abs_deta_abs_dxB={s['max_corr_abs_deta_abs_dxB']:.6e}`, "
                f"`recommendation={s['recommendation']}`\n"
            )


if __name__ == "__main__":
    main()
