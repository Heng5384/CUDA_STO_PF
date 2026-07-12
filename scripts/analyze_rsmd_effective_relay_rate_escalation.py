#!/usr/bin/env python3
"""Audit the chi=4 effective-relay scenario without claiming calibration."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: object, default: float = math.nan) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def output_dir(case_dir: Path) -> Path | None:
    log = case_dir / "stdout.log"
    if not log.exists():
        return None
    found = re.findall(r"case_output_dir\s*:\s*(\S+)", log.read_text(errors="ignore"))
    return next((Path(raw) for raw in reversed(found) if Path(raw).exists()), None)


def write(path: Path, data: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in data:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(data)


def slope(items: list[tuple[float, float]]) -> float:
    if len(items) < 2:
        return math.nan
    xb = sum(x for x, _ in items) / len(items); yb = sum(y for _, y in items) / len(items)
    den = sum((x - xb) ** 2 for x, _ in items)
    return sum((x - xb) * (y - yb) for x, y in items) / den if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--report-root", type=Path, required=True)
    args = ap.parse_args()
    result: list[dict[str, object]] = []
    for meta in rows(args.manifest):
        case = args.run_root / meta["run_id"]
        status = (case / "status.txt").read_text().strip() if (case / "status.txt").exists() else "NOT_RUN"
        log = (case / "stdout.log").read_text(errors="ignore") if (case / "stdout.log").exists() else ""
        out = output_dir(case)
        timeline = rows(out / "diagnostic_rsmd_seed_growth_time_series.csv") if out else []
        by = {int(f(row.get("post_handoff_step"), -1)): row for row in timeline}
        timeline = [by[key] for key in sorted(by) if key >= 0]
        norm = rows(out / "diagnostic_rsmd_matrix_halo_source_normalization.csv") if out else []
        ledger = rows(out / "diagnostic_rsmd_mass_ledger.csv") if out else []
        first, last = (timeline[0], timeline[-1]) if timeline else ({}, {})
        r0, r1 = f(first.get("R_eff_h_nm")), f(last.get("R_eff_h_nm"))
        h0, h1 = f(first.get("h_integral")), f(last.get("h_integral"))
        tail_start = int(f(meta.get("release_window_steps"), -1))
        tail = [(f(row.get("post_handoff_step")), f(row.get("R_eff_h_nm")))
                for row in timeline if f(row.get("post_handoff_step")) > tail_start and
                math.isfinite(f(row.get("R_eff_h_nm")))]
        norm_ok = bool(norm) and all(int(f(row.get("normalization_pass"), 0)) == 1 for row in norm)
        release = sum(f(row.get("applied_mass"), 0.0) for row in norm)
        max_mass = max((abs(f(row.get("mass_error_rel"), 0.0)) for row in ledger), default=math.nan)
        max_far = max((f(row.get("far_field_xB_mean"), -math.inf) for row in timeline), default=math.nan)
        bound = bool(re.search(r"xB_range=\[[^,]+, 1\.0000\]", log))
        stable = (status == "EXIT 0" and math.isfinite(r1) and r1 > 0.0 and
                  math.isfinite(h1) and h1 > 0.01 * h0 and not bound and
                  norm_ok and max_mass <= 1e-10 and max_far < 0.010)
        result.append({**meta, "status": status, "case_output_dir": str(out) if out else "NOT_AVAILABLE",
                       "R_eff_h_initial_nm": r0, "R_eff_h_final_nm": r1,
                       "relative_delta_R_eff_h": (r1-r0)/r0 if math.isfinite(r0) and r0 else math.nan,
                       "h_initial": h0, "h_final": h1,
                       "relative_delta_h": (h1-h0)/h0 if math.isfinite(h0) and h0 else math.nan,
                       "source_mass_released": release, "source_normalization_pass": norm_ok,
                       "max_mass_error_rel": max_mass, "max_far_field_xB": max_far,
                       "xB_upper_bound_hit": bound, "source_off_tail_R_slope": slope(tail),
                       "source_off_tail_points": len(tail), "stable": stable})
    write(args.report_root / "relay_rate_escalation_summary.csv", result)
    both_on_growth = all(any(int(f(row.get("temperature_C"), -1)) == temp and row.get("group") == "source_on" and
                             str(row.get("stable")).lower() == "true" and f(row.get("relative_delta_R_eff_h")) > 0 and
                             f(row.get("relative_delta_h")) > 0 for row in result) for temp in (380,400))
    all_finished = len(result) == 4 and all(row.get("status") == "EXIT 0" for row in result)
    status = ("PASS_EFFECTIVE_RELAY_RATE_ESCALATION_GROWTH" if all_finished and both_on_growth else
              "FAIL_EFFECTIVE_RELAY_RATE_ESCALATION_GROWTH_NOT_CONFIRMED")
    report = ["# Effective Relay-Rate Escalation Report", "", f"final_status={status}", "",
              "This is an effective required-supply scenario bracket (chi_rel=4), not calibrated GP release kinetics or a GP solvus.", ""]
    for row in result:
        report.append(f"- T{row['temperature_C']} {row['group']}: stable={row['stable']}; "
                      f"dR/R={row['relative_delta_R_eff_h']}; dh/h={row['relative_delta_h']}; "
                      f"source={row['source_mass_released']}; tail_slope={row['source_off_tail_R_slope']}.")
    args.report_root.mkdir(parents=True, exist_ok=True)
    (args.report_root / "effective_relay_rate_escalation_report.md").write_text("\n".join(report) + "\n")
    (args.report_root / "final_terminal_output.txt").write_text(f"final_status={status}\n")
    print(f"final_status={status}")
    if not status.startswith("PASS"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
