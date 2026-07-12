#!/usr/bin/env python3
"""Evaluate timestep-only RSMD stability refinement without reclassifying physics."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: object, default: float = math.nan) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def find_output(case_dir: Path) -> Path | None:
    stdout = case_dir / "stdout.log"
    if not stdout.exists():
        return None
    matches = re.findall(r"case_output_dir\s*:\s*(\S+)", stdout.read_text(errors="ignore"))
    for raw in reversed(matches):
        candidate = Path(raw)
        if candidate.exists():
            return candidate
    return None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    args.report_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for meta in read_csv(args.manifest):
        case_dir = args.run_root / meta["run_id"]
        status = (case_dir / "status.txt").read_text().strip() if (case_dir / "status.txt").exists() else "NOT_RUN"
        stdout = (case_dir / "stdout.log").read_text(errors="ignore") if (case_dir / "stdout.log").exists() else ""
        output = find_output(case_dir)
        timeline = read_csv(output / "diagnostic_rsmd_seed_growth_time_series.csv") if output else []
        dedup = {int(f(row.get("post_handoff_step"), -1)): row for row in timeline}
        timeline = [dedup[key] for key in sorted(dedup) if key >= 0]
        norm = read_csv(output / "diagnostic_rsmd_matrix_halo_source_normalization.csv") if output else []
        ledger = read_csv(output / "diagnostic_rsmd_mass_ledger.csv") if output else []
        initial, final = (timeline[0], timeline[-1]) if timeline else ({}, {})
        max_mass = max((abs(f(row.get("mass_error_rel"), 0.0)) for row in ledger), default=math.nan)
        max_far = max((f(row.get("far_field_xB_mean"), -math.inf) for row in timeline), default=math.nan)
        r0, r1 = f(initial.get("R_eff_h_nm")), f(final.get("R_eff_h_nm"))
        h0, h1 = f(initial.get("h_integral")), f(final.get("h_integral"))
        source = sum(f(row.get("applied_mass"), 0.0) for row in norm)
        # main_cuda's compact range line has four decimals; a 1.0000 maximum is
        # an unambiguous bound hit, while a printed 0.0000 minimum is not enough
        # to reject a diffuse-interface calculation by itself.
        xB_upper_bound_hit = bool(re.search(r"xB_range=\[[^,]+, 1\.0000\]", stdout))
        source_norm_ok = bool(norm) and all(int(f(row.get("normalization_pass"), 0.0)) == 1 for row in norm)
        release_step = int(f(meta.get("release_window_steps"), -1))
        tail = [row for row in timeline if int(f(row.get("post_handoff_step"), -1)) > release_step]
        tail_slope = math.nan
        if len(tail) >= 2:
            xbar = sum(f(row.get("post_handoff_step")) for row in tail) / len(tail)
            ybar = sum(f(row.get("R_eff_h_nm")) for row in tail) / len(tail)
            denominator = sum((f(row.get("post_handoff_step")) - xbar) ** 2 for row in tail)
            tail_slope = (sum((f(row.get("post_handoff_step")) - xbar) *
                              (f(row.get("R_eff_h_nm")) - ybar) for row in tail) / denominator
                          if denominator else 0.0)
        stable = (
            status == "EXIT 0" and math.isfinite(r1) and r1 > 0.0 and
            math.isfinite(h1) and h1 > 0.01 * h0 and
            math.isfinite(max_mass) and max_mass <= 1.0e-10 and
            math.isfinite(max_far) and max_far < 0.010 and
            source > 0.0 and source_norm_ok and not xB_upper_bound_hit
        )
        rows.append({
            **meta,
            "status": status,
            "case_output_dir": str(output) if output else "NOT_AVAILABLE",
            "R_eff_h_initial_nm": r0,
            "R_eff_h_final_nm": r1,
            "relative_delta_R_eff_h": (r1 - r0) / r0 if math.isfinite(r0) and r0 else math.nan,
            "h_integral_initial": h0,
            "h_integral_final": h1,
            "relative_delta_h_integral": (h1 - h0) / h0 if math.isfinite(h0) and h0 else math.nan,
            "source_mass_released": source,
            "source_normalization_pass": source_norm_ok,
            "source_off_tail_R_slope_nm_per_step": tail_slope,
            "source_off_tail_steps": len(tail),
            "max_mass_error_rel": max_mass,
            "max_far_field_xB": max_far,
            "xB_upper_bound_hit_in_stdout": xB_upper_bound_hit,
            "timestep_refinement_stable": stable,
        })
    write_csv(args.report_root / "dt_refinement_summary.csv", rows)
    passed = len(rows) == 4 and all(bool(row["timestep_refinement_stable"]) for row in rows)
    status = ("PASS_TIMESTEP_REFINED_INTERFACE_SUPPLY_STABLE" if passed else
              "FAIL_TIMESTEP_REFINEMENT_STABILITY_NOT_CONFIRMED")
    report = [
        "# Timestep Refinement Stability Report",
        "",
        f"final_status={status}",
        "",
        "This is a numerical timestep refinement only. It does not calibrate GP release, "
        "change seed profiles, alter the total ledger, or establish a GP thermodynamic law.",
        "",
    ]
    for row in rows:
        report.append(
            f"- T{row['temperature_C']}: stable={row['timestep_refinement_stable']}; "
            f"R_eff_h={row['R_eff_h_initial_nm']:.6g}->{row['R_eff_h_final_nm']:.6g} nm; "
            f"h={row['h_integral_initial']:.6g}->{row['h_integral_final']:.6g}; "
            f"source={row['source_mass_released']:.6g}; "
            f"xB_upper_bound_hit={row['xB_upper_bound_hit_in_stdout']}."
        )
    (args.report_root / "timestep_refinement_stability_report.md").write_text("\n".join(report) + "\n")
    (args.report_root / "final_terminal_output.txt").write_text(
        f"dt_refinement_cases={len(rows)}\n"
        f"final_status={status}\n"
    )
    print(f"final_status={status}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
