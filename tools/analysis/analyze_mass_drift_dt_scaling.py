#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def fget(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return math.nan


def fit_case(rows: list[dict[str, str]]) -> dict[str, Any]:
    case_name = rows[0]["case_name"]
    xs = np.asarray([fget(r, "dt") for r in rows], dtype=float)
    ys = np.asarray([abs(fget(r, "absolute_drift")) for r in rows], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0.0) & (ys > 0.0)
    if mask.sum() < 2:
        return {
            "case_name": case_name,
            "fitted_order_p": math.nan,
            "C": math.nan,
            "R2": math.nan,
            "interpretation": "insufficient valid points",
        }
    lx = np.log(xs[mask])
    ly = np.log(ys[mask])
    p, logC = np.polyfit(lx, ly, 1)
    fit = p * lx + logC
    ss_res = float(np.sum((ly - fit) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan
    if not math.isfinite(r2) or r2 < 0.8:
        interp = "mixed source or event-related irregular drift"
    elif abs(p - 1.0) < 0.35:
        interp = "likely first-order time-splitting / time-integration drift"
    elif abs(p - 2.0) < 0.5:
        interp = "likely second-order time discretization error"
    elif abs(p) < 0.35:
        interp = "likely clipping, implementation, k=0, or normalization bug"
    else:
        interp = "mixed source or unresolved scaling"
    return {
        "case_name": case_name,
        "fitted_order_p": float(p),
        "C": float(math.exp(logC)),
        "R2": r2,
        "interpretation": interp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit |xBtot drift| ~ C * dt^p from benchmark summary CSV.")
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    benchmark_csv = args.benchmark_csv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else benchmark_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(benchmark_csv)
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case_name"], []).append(row)

    summary_rows = [fit_case(case_rows) for _, case_rows in sorted(by_case.items())]
    summary_csv = out_dir / "mass_drift_dt_scaling_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)
    (out_dir / "mass_drift_dt_scaling_summary.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for case_name, case_rows in sorted(by_case.items()):
        case_rows_sorted = sorted(case_rows, key=lambda r: fget(r, "dt"))
        dt = np.asarray([fget(r, "dt") for r in case_rows_sorted], dtype=float)
        drift = np.asarray([abs(fget(r, "absolute_drift")) for r in case_rows_sorted], dtype=float)
        ax.plot(dt, drift, marker="o", label=case_name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dt")
    ax.set_ylabel("|total xBtot drift|")
    ax.set_title("Mass drift dt scaling")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "mass_drift_dt_scaling.png", dpi=180)
    plt.close(fig)

    report_lines = ["Mass drift dt scaling summary"]
    for row in summary_rows:
        report_lines.append(
            f"{row['case_name']}: p={row['fitted_order_p']:.6f}, R2={row['R2']:.6f}, {row['interpretation']}"
        )
    (out_dir / "mass_drift_dt_scaling_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(f"[ok] wrote {summary_csv}")
    print(f"[ok] wrote {out_dir / 'mass_drift_dt_scaling.png'}")


if __name__ == "__main__":
    main()
