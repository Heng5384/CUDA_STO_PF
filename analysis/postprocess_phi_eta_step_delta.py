#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def parse_args():
    p = argparse.ArgumentParser(description="Postprocess phi-vs-eta per-step diagnostics.")
    p.add_argument("--cases", nargs="+", required=True, help="Case directories")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def as_float(row, key, default=float("nan")):
    val = row.get(key, "")
    if val == "" or val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default


def first_step_matching(rows, predicate):
    for row in rows:
        if predicate(row):
            return int(as_float(row, "step", default=-1))
    return -1


def summarize_case(case_dir: Path):
    meta = json.loads((case_dir / "scan_meta.json").read_text(encoding="utf-8"))
    per_step_path = case_dir / "phi_eta_step_delta_per_step.csv"
    rows = list(csv.DictReader(per_step_path.open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"no rows in {per_step_path}")

    max_dphi_abs_max = max(as_float(r, "dphi_abs_max", 0.0) for r in rows)
    max_deta_abs_max = max(as_float(r, "deta_abs_max", 0.0) for r in rows)
    mean_dphi_abs_mean = sum(as_float(r, "dphi_abs_mean", 0.0) for r in rows) / len(rows)
    mean_deta_abs_mean = sum(as_float(r, "deta_abs_mean", 0.0) for r in rows) / len(rows)
    max_ratio = max(as_float(r, "deta_abs_max_over_dphi_abs_max", 0.0) for r in rows)
    max_dxB_abs_max = max(as_float(r, "dxB_abs_max", 0.0) for r in rows)
    max_xBtot_rel_delta = max(abs(as_float(r, "xBtot_rel_delta", 0.0)) for r in rows)
    max_eta_far_field = max(as_float(r, "eta_far_field_max", 0.0) for r in rows)
    first_clip = first_step_matching(
        rows,
        lambda r: (as_float(r, "xB_clip_low_count", 0.0) > 0 or
                   as_float(r, "xB_clip_high_count", 0.0) > 0 or
                   as_float(r, "nan_inf_flag", 0.0) > 0),
    )
    first_eta_far = first_step_matching(rows, lambda r: as_float(r, "eta_far_field_max", 0.0) > 0.05)

    recommendation = "needs_review"
    if first_clip >= 0:
        recommendation = "unstable"
    elif first_eta_far >= 0:
        recommendation = "global_ordering_warning"
    elif meta["f_eta"] == 1e-5:
        recommendation = "baseline_candidate_if_xB_range_remains_stable"
    elif meta["f_eta"] == 1e-4:
        recommendation = "aggressive_compare_against_baseline"

    summary = {
        "case": meta["case"],
        "f_eta": meta["f_eta"],
        "gp_L_eta": meta["gp_L_eta"],
        "L_eta_diff_ref_code": meta["L_eta_diff_ref_code"],
        "ratio_current_to_diff_ref": meta["gp_L_eta"] / meta["L_eta_diff_ref_code"],
        "max_dphi_abs_max": max_dphi_abs_max,
        "max_deta_abs_max": max_deta_abs_max,
        "mean_dphi_abs_mean": mean_dphi_abs_mean,
        "mean_deta_abs_mean": mean_deta_abs_mean,
        "max_deta_over_dphi_abs_max": max_ratio,
        "max_dxB_abs_max": max_dxB_abs_max,
        "max_xBtot_rel_delta": max_xBtot_rel_delta,
        "first_step_xB_clip": first_clip,
        "first_step_eta_far_gt_0p05": first_eta_far,
        "max_eta_far_field": max_eta_far_field,
        "recommendation": recommendation,
    }
    return rows, summary


def write_plot(path: Path, rows, y_keys, title, ylabel):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = [as_float(r, "step", 0.0) for r in rows]
    for label, key in y_keys:
        ax.plot(x, [as_float(r, key, 0.0) for r in rows], linewidth=1.2, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    if rows:
        s_phi = as_float(rows[0], "S_phi_total", float("nan"))
        s_eta = as_float(rows[0], "S_eta_total", float("nan"))
        if s_phi == s_phi:
            ax.text(0.02, 0.96, f"S_phi_total={s_phi:.3e}", transform=ax.transAxes,
                    fontsize=8, va="top")
        if s_eta == s_eta:
            ax.text(0.02, 0.88, f"S_eta_total={s_eta:.3e}", transform=ax.transAxes,
                    fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    case_rows = {}
    for case in args.cases:
        case_dir = Path(case)
        rows, summary = summarize_case(case_dir)
        case_rows[summary["case"]] = rows
        summaries.append(summary)

        write_plot(plot_dir / f"{summary['case']}_dphi_deta_abs_max.png",
                   rows,
                   [("dphi_abs_max", "dphi_abs_max"), ("deta_abs_max", "deta_abs_max")],
                   f"{summary['case']} dphi_abs_max vs deta_abs_max",
                   "abs max update")
        write_plot(plot_dir / f"{summary['case']}_dphi_deta_abs_mean.png",
                   rows,
                   [("dphi_abs_mean", "dphi_abs_mean"), ("deta_abs_mean", "deta_abs_mean")],
                   f"{summary['case']} dphi_abs_mean vs deta_abs_mean",
                   "abs mean update")
        write_plot(plot_dir / f"{summary['case']}_deta_over_dphi_ratio.png",
                   rows,
                   [("deta_abs_max_over_dphi_abs_max", "deta_abs_max_over_dphi_abs_max")],
                   f"{summary['case']} deta/dphi abs-max ratio",
                   "ratio")
        write_plot(plot_dir / f"{summary['case']}_xB_range.png",
                   rows,
                   [("xB_min", "xB_min_after"), ("xB_max", "xB_max_after")],
                   f"{summary['case']} xB range",
                   "xB")
        write_plot(plot_dir / f"{summary['case']}_xBtot_delta.png",
                   rows,
                   [("xBtot_rel_delta", "xBtot_rel_delta")],
                   f"{summary['case']} xBtot relative delta",
                   "relative delta")
        write_plot(plot_dir / f"{summary['case']}_eta_far_field.png",
                   rows,
                   [("eta_far_field_max", "eta_far_field_max")],
                   f"{summary['case']} eta far-field max",
                   "eta far-field max")

    summary_csv = out_dir / "phi_eta_step_delta_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    report_md = out_dir / "reports/step_reports/STEP38E_PHI_ETA_STEP_DELTA_REPORT.md"
    by_case = {s["case"]: s for s in summaries}
    baseline = by_case.get("gp_eta_kinetic_scan_f1em5")
    aggressive = by_case.get("gp_eta_kinetic_scan_f1em4")
    unstable = by_case.get("gp_eta_kinetic_scan_f1em3")
    with report_md.open("w", encoding="utf-8") as f:
        f.write("# Phi vs Eta Per-Step Update Diagnostic\n\n")
        for s in summaries:
            f.write(
                f"- `{s['case']}`: `max_dphi_abs_max={s['max_dphi_abs_max']:.6e}`, "
                f"`max_deta_abs_max={s['max_deta_abs_max']:.6e}`, "
                f"`max_dxB_abs_max={s['max_dxB_abs_max']:.6e}`, "
                f"`max_xBtot_rel_delta={s['max_xBtot_rel_delta']:.6e}`, "
                f"`first_step_xB_clip={s['first_step_xB_clip']}`, "
                f"`first_step_eta_far_gt_0p05={s['first_step_eta_far_gt_0p05']}`, "
                f"`recommendation={s['recommendation']}`\n"
            )
        f.write("\n## Comparison\n")
        if baseline and aggressive:
            f.write(
                f"- baseline `f_eta=1e-5`: `max_deta_abs_max={baseline['max_deta_abs_max']:.6e}`, "
                f"`max_eta_far_field={baseline['max_eta_far_field']:.6e}`, "
                f"`first_step_xB_clip={baseline['first_step_xB_clip']}`\n"
            )
            f.write(
                f"- aggressive `f_eta=1e-4`: `max_deta_abs_max={aggressive['max_deta_abs_max']:.6e}`, "
                f"`max_eta_far_field={aggressive['max_eta_far_field']:.6e}`, "
                f"`first_step_xB_clip={aggressive['first_step_xB_clip']}`\n"
            )
        if unstable:
            f.write(
                f"- unstable probe `f_eta=1e-3`: `first_step_xB_clip={unstable['first_step_xB_clip']}`, "
                f"`max_deta_abs_max={unstable['max_deta_abs_max']:.6e}`\n"
            )


if __name__ == "__main__":
    main()
