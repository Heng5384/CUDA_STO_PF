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


ALIASES = {
    "step": ("step", "Step"),
    "t_real_s": ("t_real_s", "time_real_s", "physical_time_s", "t_real"),
    "mean_xBtot": ("mean_xBtot", "mean_xB_tot", "<x_B_tot>", "xBtot_mean"),
    "relative_drift_vs_initial": ("relative_drift_vs_initial", "rel_drift", "relative_drift"),
    "mean_hphi": ("mean_hphi", "vf_precip", "mean_h", "hphi_mean"),
    "mean_phi": ("mean_phi", "phi_mean"),
    "xB_min": ("xB_min", "xb_min"),
    "xB_max": ("xB_max", "xb_max"),
    "phi_min": ("phi_min",),
    "phi_max": ("phi_max",),
    "R_avg": ("R_avg", "R_equiv", "radius_avg"),
}


def find_col(fields: list[str], canonical: str, required: bool = True) -> str | None:
    lower = {f.lower(): f for f in fields}
    for cand in ALIASES.get(canonical, (canonical,)):
        if cand in fields:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise KeyError(f"Missing required column {canonical}; available={fields}")
    return None


def read_table(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows, fields


def as_float(row: dict[str, Any], key: str | None, default: float = math.nan) -> float:
    if key is None:
        return default
    value = row.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_int(row: dict[str, Any], key: str | None, default: int = 0) -> int:
    value = as_float(row, key, float(default))
    return int(value) if math.isfinite(value) else default


def find_diagnostics_csv(run_dir: Path) -> Path:
    candidates = [
        run_dir / "relaxation_diagnostics.csv",
        run_dir / "diagnostics.csv",
        run_dir / "energy_diagnostics.csv",
    ]
    candidates.extend(sorted(run_dir.glob("*diagnostics*.csv")))
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(f"No diagnostics CSV found in {run_dir}")


def find_vf_csv(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("vf_precip_vs_time*.csv"))
    return candidates[0] if candidates else None


def nearest_row(rows: list[dict[str, str]], step_col: str, step: int) -> dict[str, str]:
    steps = np.asarray([as_float(r, step_col) for r in rows])
    idx = int(np.argmin(np.abs(steps - step)))
    return rows[idx]


def row_to_point(row: dict[str, str], cols: dict[str, str | None]) -> dict[str, float]:
    return {
        "step": as_float(row, cols["step"]),
        "mean_xBtot": as_float(row, cols["mean_xBtot"]),
        "relative_drift_vs_initial": as_float(row, cols["relative_drift_vs_initial"], 0.0),
        "mean_hphi": as_float(row, cols["mean_hphi"]),
        "xB_min": as_float(row, cols["xB_min"]),
        "xB_max": as_float(row, cols["xB_max"]),
        "phi_max": as_float(row, cols["phi_max"]),
        "t_real_s": as_float(row, cols["t_real_s"], 0.0),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    diag_path = find_diagnostics_csv(run_dir)
    event_path = run_dir / "scheduled_nucleation_events.csv"
    if not event_path.exists():
        raise FileNotFoundError(f"Missing {event_path}")

    diag_rows, diag_fields = read_table(diag_path)
    event_rows, event_fields = read_table(event_path)
    cols = {
        "step": find_col(diag_fields, "step"),
        "t_real_s": find_col(diag_fields, "t_real_s", required=False),
        "mean_xBtot": find_col(diag_fields, "mean_xBtot"),
        "relative_drift_vs_initial": find_col(diag_fields, "relative_drift_vs_initial", required=False),
        "mean_hphi": find_col(diag_fields, "mean_hphi"),
        "xB_min": find_col(diag_fields, "xB_min"),
        "xB_max": find_col(diag_fields, "xB_max"),
        "phi_max": find_col(diag_fields, "phi_max"),
    }
    event_step_col = "step" if "step" in event_fields else find_col(event_fields, "step")
    event_steps = sorted({as_int(r, event_step_col) for r in event_rows})
    first_step = as_int(diag_rows[0], cols["step"])
    last_step = as_int(diag_rows[-1], cols["step"])
    cuts = [first_step] + [s for s in event_steps if first_step <= s <= last_step] + [last_step]
    cuts = sorted(dict.fromkeys(cuts))

    segment_rows: list[dict[str, Any]] = []
    for seg_id, (start, end) in enumerate(zip(cuts[:-1], cuts[1:])):
        rs = nearest_row(diag_rows, cols["step"], start)  # type: ignore[arg-type]
        re = nearest_row(diag_rows, cols["step"], end)  # type: ignore[arg-type]
        ps = row_to_point(rs, cols)
        pe = row_to_point(re, cols)
        delta = pe["mean_xBtot"] - ps["mean_xBtot"]
        rel = delta / ps["mean_xBtot"] if ps["mean_xBtot"] else math.nan
        segment_rows.append(
            {
                "segment_id": seg_id,
                "step_start": start,
                "step_end": end,
                "mean_xBtot_start": ps["mean_xBtot"],
                "mean_xBtot_end": pe["mean_xBtot"],
                "delta_mean_xBtot": delta,
                "relative_drift_segment": rel,
                "relative_drift_vs_initial_start": ps["relative_drift_vs_initial"],
                "relative_drift_vs_initial_end": pe["relative_drift_vs_initial"],
                "mean_hphi_start": ps["mean_hphi"],
                "mean_hphi_end": pe["mean_hphi"],
                "delta_mean_hphi": pe["mean_hphi"] - ps["mean_hphi"],
                "xB_min_start": ps["xB_min"],
                "xB_min_end": pe["xB_min"],
                "xB_max_start": ps["xB_max"],
                "xB_max_end": pe["xB_max"],
                "phi_max_start": ps["phi_max"],
                "phi_max_end": pe["phi_max"],
                "physical_time_start_s": ps["t_real_s"],
                "physical_time_end_s": pe["t_real_s"],
                "physical_duration_s": pe["t_real_s"] - ps["t_real_s"],
            }
        )

    event_out: list[dict[str, Any]] = []
    accumulated_event_abs = 0.0
    max_rel_event = 0.0
    for row in event_rows:
        before = as_float(row, "M_before_event")
        embed = as_float(row, "M_after_embed_before_comp")
        comp = as_float(row, "M_after_comp")
        event_err = comp - before
        rel_err = as_float(row, "relative_event_mass_error", event_err / before if before else math.nan)
        accumulated_event_abs += abs(event_err)
        max_rel_event = max(max_rel_event, abs(rel_err))
        event_out.append(
            {
                "event_id": as_int(row, "event_id"),
                "step": as_int(row, "step"),
                "M_before_event": before,
                "M_after_embed_before_comp": embed,
                "M_after_comp": comp,
                "insertion_mass_jump_before_comp": embed - before,
                "event_mass_error_after_comp": event_err,
                "relative_event_mass_error": rel_err,
                "xB_edge": as_float(row, "xB_edge"),
                "C_local": as_float(row, "C_local"),
                "local_comp_weight_active_fraction": as_float(row, "local_comp_weight_active_fraction"),
                "xB_min_after": as_float(row, "xB_min_after"),
                "xB_max_after": as_float(row, "xB_max_after"),
            }
        )

    initial = as_float(diag_rows[0], cols["mean_xBtot"])
    final = as_float(diag_rows[-1], cols["mean_xBtot"])
    total_drift = final - initial
    insertion_conservative = max_rel_event < 1.0e-6
    dynamics_drift_dominates = abs(total_drift) > 10.0 * max(accumulated_event_abs, 1.0e-30)

    write_csv(run_dir / "drift_segments_summary.csv", segment_rows)
    write_csv(run_dir / "drift_events_summary.csv", event_out)

    steps = np.asarray([as_float(r, cols["step"]) for r in diag_rows])
    t_real = np.asarray([as_float(r, cols["t_real_s"], np.nan) for r in diag_rows])
    x = steps
    xlabel = "step"
    if np.all(np.isfinite(t_real)) and np.nanmax(t_real) > 0:
        x = t_real
        xlabel = "physical time (s)"
    y_xbt = np.asarray([as_float(r, cols["mean_xBtot"]) for r in diag_rows])
    y_drift = np.asarray([as_float(r, cols["relative_drift_vs_initial"], (v - initial) / initial) for r, v in zip(diag_rows, y_xbt)])
    y_h = np.asarray([as_float(r, cols["mean_hphi"]) for r in diag_rows])
    y_xbmin = np.asarray([as_float(r, cols["xB_min"]) for r in diag_rows])
    y_xbmax = np.asarray([as_float(r, cols["xB_max"]) for r in diag_rows])
    fig, axs = plt.subplots(4, 1, figsize=(9, 12), constrained_layout=True, sharex=True)
    axs[0].plot(x, y_xbt)
    axs[0].set_ylabel("mean xBtot")
    axs[1].plot(x, y_drift)
    axs[1].set_ylabel("relative drift")
    axs[2].plot(x, y_h)
    axs[2].set_ylabel("mean h(phi)")
    axs[3].plot(x, y_xbmin, label="xB_min")
    axs[3].plot(x, y_xbmax, label="xB_max")
    axs[3].set_ylabel("xB range")
    axs[3].set_xlabel(xlabel)
    axs[3].legend()
    for step in event_steps:
        ex = as_float(nearest_row(diag_rows, cols["step"], step), cols["t_real_s"], float(step)) if xlabel.startswith("physical") else float(step)
        for ax in axs:
            ax.axvline(ex, color="0.4", linestyle="--", linewidth=1)
    fig.suptitle("Scheduled nucleation drift segments")
    fig.savefig(run_dir / "drift_segments_plot.png", dpi=180)
    plt.close(fig)

    summary = {
        "run_dir": str(run_dir),
        "diagnostics_csv": str(diag_path),
        "events_csv": str(event_path),
        "columns_used": cols,
        "event_steps": event_steps,
        "initial_mean_xBtot": initial,
        "final_mean_xBtot": final,
        "total_delta_mean_xBtot": total_drift,
        "total_relative_drift": total_drift / initial if initial else math.nan,
        "max_relative_event_mass_error": max_rel_event,
        "accumulated_absolute_event_mass_error": accumulated_event_abs,
        "insertion_mass_conservative": insertion_conservative,
        "dynamics_drift_dominates": dynamics_drift_dominates,
        "segments": segment_rows,
        "events": event_out,
    }
    (run_dir / "drift_segments_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "Scheduled nucleation drift segment report",
        f"run_dir: {run_dir}",
        f"diagnostics_csv: {diag_path.name}",
        f"events_csv: {event_path.name}",
        f"event_steps: {event_steps}",
        f"initial_mean_xBtot: {initial:.12e}",
        f"final_mean_xBtot: {final:.12e}",
        f"total_relative_drift: {summary['total_relative_drift']:.12e}",
        f"max_relative_event_mass_error: {max_rel_event:.12e}",
        f"accumulated_absolute_event_mass_error: {accumulated_event_abs:.12e}",
        f"insertion_mass_conservative: {str(insertion_conservative).lower()}",
        f"dynamics_drift_dominates: {str(dynamics_drift_dominates).lower()}",
        "",
        "Interpretation:",
    ]
    if insertion_conservative and dynamics_drift_dominates:
        report.append("Insertion events are conservative within tolerance; observed xBtot drift is dominated by PF dynamics segments.")
    elif not insertion_conservative:
        report.append("At least one insertion event has non-negligible mass error; inspect scheduled_nucleation_events.csv.")
    else:
        report.append("Dynamics drift and insertion errors are comparable at this tolerance.")
    (run_dir / "drift_segments_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze scheduled nucleation xBtot drift by dynamics segments and insertion events.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.run_dir)
    print(f"[ok] wrote {Path(summary['run_dir']) / 'drift_segments_summary.csv'}")
    print(f"[ok] wrote {Path(summary['run_dir']) / 'drift_segments_plot.png'}")
    print(
        "[summary] total_relative_drift={:.6e}, max_event_rel_err={:.3e}, dynamics_drift_dominates={}".format(
            summary["total_relative_drift"],
            summary["max_relative_event_mass_error"],
            summary["dynamics_drift_dominates"],
        )
    )


if __name__ == "__main__":
    main()
