#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_last_relax_row(path: Path) -> dict:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def load_diag_stats(path: Path) -> dict:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    def finite_float(v: str) -> float | None:
        try:
            x = float(v)
        except Exception:
            return None
        return x if math.isfinite(x) else None
    ratios = []
    for row in rows:
        req = finite_float(row.get("delta_M_Y_required", ""))
        val = finite_float(row.get("Y_compensation_ratio", ""))
        if req is not None and val is not None and abs(req) > 1.0e-10:
            ratios.append(val)
    last = rows[-1]
    return {
        "nrows": len(rows),
        "avg_Y_compensation_ratio": (sum(ratios) / len(ratios)) if ratios else float("nan"),
        "final_picard_Y_compensation_ratio": finite_float(last.get("picard_Y_compensation_ratio_final", "")),
        "final_picard_rms_dYdt_change": finite_float(last.get("picard_rms_dYdt_change_final", "")),
        "final_picard_maxabs_dYdt_change": finite_float(last.get("picard_maxabs_dYdt_change_final", "")),
    }


def collect_run(run_dir: Path) -> dict:
    mass = load_json(run_dir / "mass_drift_summary.json")
    perf = load_json(run_dir / "performance_summary.json")
    relax = load_last_relax_row(run_dir / "relaxation_diagnostics.csv")
    diag = load_diag_stats(run_dir / "dynamics_mass_diagnostics.csv")
    return {
        "run_dir": str(run_dir),
        "case_name": perf.get("case_name", run_dir.name),
        "picard_enabled": perf.get("picard_enabled"),
        "picard_iters": perf.get("picard_iters"),
        "picard_omega": perf.get("picard_omega"),
        "nsteps": perf.get("nsteps"),
        "dt": perf.get("dt"),
        "total_walltime_s": perf.get("total_walltime_s"),
        "avg_walltime_per_step_s": perf.get("avg_walltime_per_step_s"),
        "median_walltime_per_step_s": perf.get("median_walltime_per_step_s"),
        "min_walltime_per_step_s": perf.get("min_walltime_per_step_s"),
        "max_walltime_per_step_s": perf.get("max_walltime_per_step_s"),
        "warmup_excluded_avg_walltime_per_step_s": perf.get("warmup_excluded_avg_walltime_per_step_s"),
        "estimated_steps_per_hour": perf.get("estimated_steps_per_hour"),
        "relative_slowdown_vs_baseline": perf.get("relative_slowdown_vs_baseline"),
        "initial_mean_xBtot": mass.get("initial_mean_xBtot"),
        "final_mean_xBtot": mass.get("final_mean_xBtot"),
        "total_absolute_drift": mass.get("total_absolute_drift"),
        "total_relative_drift": mass.get("total_relative_drift"),
        "max_abs_step_drift": mass.get("max_abs_step_drift"),
        "summary_matches_csv": mass.get("summary_matches_csv"),
        "max_summary_csv_abs_mismatch": mass.get("max_summary_csv_abs_mismatch"),
        "max_abs_mass_closure_resid": mass.get("max_abs_mass_closure_resid"),
        "total_clip_count_xB": mass.get("total_clip_count_xB"),
        "total_clip_count_Y": mass.get("total_clip_count_Y"),
        "xB_min": float(relax.get("xB_min", "nan")) if relax else float("nan"),
        "xB_max": float(relax.get("xB_max", "nan")) if relax else float("nan"),
        "mean_xBtot_relax_final": float(relax.get("mean_xBtot", "nan")) if relax else float("nan"),
        "avg_Y_compensation_ratio": diag.get("avg_Y_compensation_ratio"),
        "final_picard_Y_compensation_ratio": diag.get("final_picard_Y_compensation_ratio"),
        "final_picard_rms_dYdt_change": diag.get("final_picard_rms_dYdt_change"),
        "final_picard_maxabs_dYdt_change": diag.get("final_picard_maxabs_dYdt_change"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-slowdown", type=float, default=2.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [collect_run(Path(p)) for p in args.run_dirs]
    rows.sort(key=lambda r: (float(r["picard_iters"]), float(r["picard_omega"])))

    baseline = next((r for r in rows if int(round(float(r["picard_enabled"]))) == 0), None)
    if baseline:
        b_drift = abs(float(baseline["total_relative_drift"]))
        b_step = float(baseline["warmup_excluded_avg_walltime_per_step_s"])
        for r in rows:
            r["relative_slowdown_vs_baseline"] = (
                float(r["warmup_excluded_avg_walltime_per_step_s"]) / b_step if b_step > 0 else float("nan")
            )
            r["drift_reduction_vs_baseline"] = (
                1.0 - abs(float(r["total_relative_drift"])) / b_drift if b_drift > 0 else float("nan")
            )
    else:
        for r in rows:
            r["drift_reduction_vs_baseline"] = float("nan")

    write_csv(out_dir / "picard_speed_drift_summary.csv", rows)
    with (out_dir / "picard_speed_drift_summary.json").open("w") as f:
        json.dump(rows, f, indent=2)

    xs = [float(r["picard_iters"]) for r in rows]
    drift = [float(r["total_relative_drift"]) for r in rows]
    step_t = [float(r["warmup_excluded_avg_walltime_per_step_s"]) for r in rows]
    dr_red = [float(r["drift_reduction_vs_baseline"]) for r in rows]
    slowdown = [float(r["relative_slowdown_vs_baseline"]) for r in rows]
    ycomp = [float(r["avg_Y_compensation_ratio"]) for r in rows]

    plt.figure(figsize=(6, 4))
    plt.plot(xs, drift, marker="o")
    plt.xlabel("Picard iterations")
    plt.ylabel("Total relative drift")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "picard_drift_vs_iters.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(xs, step_t, marker="o")
    plt.xlabel("Picard iterations")
    plt.ylabel("Warmup-excluded step walltime (s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "picard_step_time_vs_iters.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(slowdown, dr_red, marker="o")
    plt.xlabel("Relative slowdown vs baseline")
    plt.ylabel("Drift reduction vs baseline")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "picard_drift_reduction_vs_slowdown.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ycomp, marker="o")
    plt.xlabel("Picard iterations")
    plt.ylabel("Average Y compensation ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "picard_Y_comp_ratio_vs_iters.png", dpi=180)
    plt.close()

    best_lowest_drift = min(rows, key=lambda r: abs(float(r["total_relative_drift"])))
    max_reduction = max(dr_red) if dr_red else float("nan")
    candidates = [
        r for r in rows
        if float(r["relative_slowdown_vs_baseline"]) <= args.max_slowdown
        and float(r["drift_reduction_vs_baseline"]) >= 0.8 * max_reduction
    ]
    recommended = min(candidates, key=lambda r: float(r["picard_iters"])) if candidates else best_lowest_drift

    lines = [
        f"best_lowest_drift: iters={best_lowest_drift['picard_iters']} rel_drift={best_lowest_drift['total_relative_drift']}",
        f"best_speed_accuracy_tradeoff: iters={recommended['picard_iters']} slowdown={recommended['relative_slowdown_vs_baseline']} drift_reduction={recommended['drift_reduction_vs_baseline']}",
        f"recommended_picard_iters: {recommended['picard_iters']}",
    ]
    (out_dir / "picard_speed_drift_report.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
