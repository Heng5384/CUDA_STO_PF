#!/usr/bin/env python3
"""Summarize the short numerical-attribution cases without physics claims."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: object, default: float = math.nan) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def output_dir(case_dir: Path) -> Path | None:
    log = case_dir / "stdout.log"
    if not log.exists():
        return None
    matches = re.findall(r"case_output_dir\s*:\s*(\S+)", log.read_text(errors="ignore"))
    return next((Path(value) for value in reversed(matches) if Path(value).exists()), None)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def first_bound_step(projection_rows: list[dict[str, str]]) -> int | None:
    for row in projection_rows:
        if max(number(row.get("max_xB_before_projection")), number(row.get("max_xB_after_projection"))) >= 0.99:
            return int(number(row.get("step"), -1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()

    summary: list[dict[str, object]] = []
    for meta in read_rows(args.manifest):
        case_dir = args.run_root / meta["run_id"]
        status = (case_dir / "status.txt").read_text().strip() if (case_dir / "status.txt").exists() else "NOT_RUN"
        out = output_dir(case_dir)
        growth = read_rows(out / "diagnostic_rsmd_seed_growth_time_series.csv") if out else []
        projection = read_rows(out / "y_update_mass_projection.csv") if out else []
        ledger = read_rows(out / "diagnostic_rsmd_mass_ledger.csv") if out else []
        dynamics = read_rows(out / "dynamics_mass_diagnostics.csv") if out else []
        source_norm = read_rows(out / "diagnostic_rsmd_matrix_halo_source_normalization.csv") if out else []
        dedup = {int(number(row.get("post_handoff_step"), -1)): row for row in growth}
        growth = [dedup[key] for key in sorted(dedup) if key >= 0]
        first = growth[0] if growth else {}
        last = growth[-1] if growth else {}
        r0, r1 = number(first.get("R_eff_h_nm")), number(last.get("R_eff_h_nm"))
        h0, h1 = number(first.get("h_integral")), number(last.get("h_integral"))
        max_projection = max((abs(number(row.get("delta_before_projection"), 0.0)) for row in projection), default=math.nan)
        max_xb = max((number(row.get("max_xB_before_projection"), -math.inf) for row in projection), default=math.nan)
        max_mass = max((abs(number(row.get("mass_error_rel"), 0.0)) for row in ledger), default=math.nan)
        max_gamma_term = max((abs(number(row.get("maxabs_term_gamma"), 0.0)) for row in dynamics), default=math.nan)
        max_lagged_dydt = max((abs(number(row.get("max_abs_lagged_dYdt"), 0.0)) for row in dynamics), default=math.nan)
        final_picard_residual = number(dynamics[-1].get("picard_maxabs_dYdt_change_final")) if dynamics else math.nan
        final_mean_gamma = number(dynamics[-1].get("mean_gamma_local")) if dynamics else math.nan
        history_reset_rows = sum(
            int(number(row.get("Y_history_reset_after_source"), 0)) == 1 for row in source_norm
        )
        expects_history_reset = int(number(meta.get("diagnostic_rsmd_reset_Y_history_after_source"), 0)) == 1
        summary.append({
            **meta,
            "status": status,
            "case_output_dir": str(out) if out else "NOT_AVAILABLE",
            "R_eff_h_initial_nm": r0,
            "R_eff_h_final_nm": r1,
            "relative_delta_R_eff_h": (r1 - r0) / r0 if math.isfinite(r0) and r0 else math.nan,
            "h_integral_initial": h0,
            "h_integral_final": h1,
            "relative_delta_h": (h1 - h0) / h0 if math.isfinite(h0) and h0 else math.nan,
            "first_xB_bound_step": first_bound_step(projection),
            "max_xB_before_projection": max_xb,
            "max_abs_projection_correction": max_projection,
            "max_mass_error_rel": max_mass,
            "maxabs_term_gamma": max_gamma_term,
            "max_abs_lagged_dYdt": max_lagged_dydt,
            "final_picard_maxabs_dYdt_change": final_picard_residual,
            "final_mean_gamma_local": final_mean_gamma,
            "history_reset_expected": expects_history_reset,
            "history_reset_source_rows": history_reset_rows,
            "history_reset_observed": (history_reset_rows > 0) if expects_history_reset else None,
        })
    args.report_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_root / "y_solver_attribution_summary.csv", summary)
    report = [
        "# RSMD Y-Solver Attribution Report",
        "",
        "These cases retain the same source and seed. They are numerical attribution controls, not a GP-release or beta-growth calibration.",
        "",
    ]
    for row in summary:
        report.append(
            f"- `{row['run_id']}`: status={row['status']}; first_xB_bound_step={row['first_xB_bound_step']}; "
            f"dR/R={row['relative_delta_R_eff_h']}; dh/h={row['relative_delta_h']}; "
            f"max projection correction={row['max_abs_projection_correction']}; "
            f"max |gamma term|={row['maxabs_term_gamma']}; "
            f"final Picard dY/dt residual={row['final_picard_maxabs_dYdt_change']}; "
            f"history reset observed={row['history_reset_observed']}."
        )
    report.extend(
        [
            "",
            "Interpretation boundary: the eight-iteration Picard case is a directional numerical control, not proof of fixed-point convergence. "
            "A non-small final Picard residual means that this test cannot exonerate the gamma*dY/dt closure.",
        ]
    )
    history_sync_ok = all(
        not row["history_reset_expected"] or row["history_reset_observed"]
        for row in summary
    )
    done = (len(summary) == 5 and all(row["status"] == "EXIT 0" for row in summary) and
            history_sync_ok)
    status = "PASS_Y_SOLVER_ATTRIBUTION_DATA_READY" if done else "INCOMPLETE_Y_SOLVER_ATTRIBUTION"
    report.extend(["", f"final_status={status}", ""])
    (args.report_root / "rsmd_y_solver_attribution_report.md").write_text("\n".join(report))
    (args.report_root / "final_terminal_output.txt").write_text(f"final_status={status}\n")
    print(f"final_status={status}")
    if not done:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
