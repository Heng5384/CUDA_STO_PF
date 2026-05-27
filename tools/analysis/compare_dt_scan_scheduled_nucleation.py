#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_scheduled_nucleation_drift_segments import analyze, find_diagnostics_csv, find_vf_csv, read_table, as_float


def read_diag_arrays(run_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    path = find_diagnostics_csv(run_dir)
    rows, fields = read_table(path)
    def col(name: str) -> str:
        for c in fields:
            if c == name:
                return c
        raise KeyError(f"Missing {name} in {path}")
    cols = {
        "step": col("step"),
        "t_real_s": col("t_real_s"),
        "mean_xBtot": col("mean_xBtot"),
        "relative_drift_vs_initial": col("relative_drift_vs_initial"),
        "mean_hphi": col("mean_hphi"),
        "xB_min": col("xB_min"),
        "xB_max": col("xB_max"),
    }
    arrays = {k: np.asarray([as_float(r, v) for r in rows], dtype=float) for k, v in cols.items()}
    return arrays, {"path": str(path)}


def infer_dt_steps(run_dir: Path, arrays: dict[str, np.ndarray]) -> tuple[float, int]:
    name = str(run_dir)
    m_dt = re.search(r"dt0p([0-9]+)", name)
    if m_dt:
        dt = float("0." + m_dt.group(1))
    else:
        m_dt_dot = re.search(r"dt(0\.[0-9]+)", name)
        dt = float(m_dt_dot.group(1)) if m_dt_dot else math.nan
    if not math.isfinite(dt):
        # The diagnostics CSV has both code time and physical time.  If path parsing
        # fails, prefer code-time increments; physical time is not the dimensionless dt.
        diag_path = find_diagnostics_csv(run_dir)
        rows, fields = read_table(diag_path)
        if "time" in fields and len(rows) > 1:
            dt = (as_float(rows[1], "time") - as_float(rows[0], "time")) / (
                as_float(rows[1], "step") - as_float(rows[0], "step")
            )
        else:
            dt = math.nan
    return dt, int(arrays["step"][-1])


def load_ravg(run_dir: Path) -> float:
    vf = find_vf_csv(run_dir)
    if not vf:
        return math.nan
    rows, fields = read_table(vf)
    if "R_avg" not in fields:
        return math.nan
    return as_float(rows[-1], "R_avg")


def summarize_run(run_dir: Path) -> dict[str, Any]:
    seg = analyze(run_dir)
    arrays, meta = read_diag_arrays(run_dir)
    dt, nsteps = infer_dt_steps(run_dir, arrays)
    event_rows, _ = read_table(run_dir / "scheduled_nucleation_events.csv")
    final = -1
    return {
        "run_dir": str(run_dir),
        "dt": dt,
        "nsteps": nsteps,
        "final_physical_time_s": float(arrays["t_real_s"][final]),
        "initial_mean_xBtot": float(arrays["mean_xBtot"][0]),
        "final_mean_xBtot": float(arrays["mean_xBtot"][final]),
        "total_delta_mean_xBtot": float(arrays["mean_xBtot"][final] - arrays["mean_xBtot"][0]),
        "total_relative_drift": float(seg["total_relative_drift"]),
        "final_mean_hphi": float(arrays["mean_hphi"][final]),
        "final_vf_precip": float(arrays["mean_hphi"][final]),
        "final_R_avg": load_ravg(run_dir),
        "final_xB_min": float(arrays["xB_min"][final]),
        "final_xB_max": float(arrays["xB_max"][final]),
        "number_of_nucleation_events": len(event_rows),
        "max_relative_event_mass_error": float(seg["max_relative_event_mass_error"]),
        "accumulated_absolute_event_mass_error": float(seg["accumulated_absolute_event_mass_error"]),
        "total_dynamics_drift_estimate": float(seg["total_delta_mean_xBtot"]),
        "diagnostics_csv": meta["path"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scheduled nucleation dt scan using physical-time alignment.")
    parser.add_argument("--run-dt0p025", type=Path, required=True)
    parser.add_argument("--run-dt0p0125", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_a = args.run_dt0p025.expanduser().resolve()
    run_b = args.run_dt0p0125.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else run_b.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = [summarize_run(run_a), summarize_run(run_b)]
    write_csv(out_dir / "dt_scan_comparison_summary.csv", summaries)
    drift_a = abs(summaries[0]["total_relative_drift"])
    drift_b = abs(summaries[1]["total_relative_drift"])
    reduction_ratio = (drift_a - drift_b) / drift_a if drift_a else math.nan
    dt_sensitive = bool(math.isfinite(reduction_ratio) and reduction_ratio > 0.25)
    nearly_unchanged = bool(math.isfinite(reduction_ratio) and reduction_ratio < 0.10)
    comparison = {
        "runs": summaries,
        "drift_reduction_ratio": reduction_ratio,
        "dt_reduction_reduces_drift": dt_sensitive,
        "interpretation": (
            "drift appears time-step sensitive"
            if dt_sensitive
            else (
                "drift is not strongly reduced by halving dt; consider xBtot mass projection or equation-level correction"
                if nearly_unchanged
                else "drift changed moderately; inspect growth metrics before assigning cause"
            )
        ),
    }
    (out_dir / "dt_scan_comparison_summary.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    arrays_a, _ = read_diag_arrays(run_a)
    arrays_b, _ = read_diag_arrays(run_b)
    fig, axs = plt.subplots(4, 1, figsize=(9, 12), constrained_layout=True, sharex=True)
    for label, arr in [("dt=0.025", arrays_a), ("dt=0.0125", arrays_b)]:
        axs[0].plot(arr["t_real_s"], arr["mean_xBtot"], label=label)
        axs[1].plot(arr["t_real_s"], arr["relative_drift_vs_initial"], label=label)
        axs[2].plot(arr["t_real_s"], arr["mean_hphi"], label=label)
        axs[3].plot(arr["t_real_s"], arr["xB_min"], label=f"{label} xB_min")
        axs[3].plot(arr["t_real_s"], arr["xB_max"], linestyle="--", label=f"{label} xB_max")
    axs[0].set_ylabel("mean xBtot")
    axs[1].set_ylabel("relative drift")
    axs[2].set_ylabel("mean h(phi)")
    axs[3].set_ylabel("xB range")
    axs[3].set_xlabel("physical time (s)")
    for ax in axs:
        ax.legend(fontsize=8)
    fig.suptitle("Scheduled nucleation dt scan comparison")
    fig.savefig(out_dir / "dt_scan_comparison_plot.png", dpi=180)
    plt.close(fig)

    report = [
        "Scheduled nucleation dt scan report",
        f"dt=0.025 total_relative_drift: {summaries[0]['total_relative_drift']:.12e}",
        f"dt=0.0125 total_relative_drift: {summaries[1]['total_relative_drift']:.12e}",
        f"drift_reduction_ratio: {reduction_ratio:.6e}",
        f"dt_reduction_reduces_drift: {str(dt_sensitive).lower()}",
        f"interpretation: {comparison['interpretation']}",
        "",
        "Note: mass conservation is evaluated using mean(xBtot), not mean(xB).",
    ]
    (out_dir / "dt_scan_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_dir / 'dt_scan_comparison_summary.csv'}")
    print(f"[ok] wrote {out_dir / 'dt_scan_comparison_plot.png'}")
    print(f"[summary] drift_reduction_ratio={reduction_ratio:.6e}; {comparison['interpretation']}")


if __name__ == "__main__":
    main()
