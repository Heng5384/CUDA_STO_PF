#!/usr/bin/env python3
"""Analyze the workstation-only timestep/source-mask remediation cases."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f(value: object, default: float = math.nan) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def finite(value: float) -> bool:
    return math.isfinite(value)


def slope(points: list[tuple[float, float]]) -> float:
    points = [(x, y) for x, y in points if finite(x) and finite(y)]
    if len(points) < 2:
        return math.nan
    xbar = mean(x for x, _ in points)
    ybar = mean(y for _, y in points)
    den = sum((x - xbar) ** 2 for x, _ in points)
    return sum((x - xbar) * (y - ybar) for x, y in points) / den if den else 0.0


def find_output(case_dir: Path) -> Path | None:
    stdout = case_dir / "stdout.log"
    if not stdout.exists():
        return None
    matches = re.findall(r"case_output_dir\s*:\s*(\S+)", stdout.read_text(errors="ignore"))
    for raw in reversed(matches):
        path = Path(raw)
        if path.exists():
            return path
    return None


def classify(rows: list[dict[str, str]]) -> tuple[str, dict[str, float]]:
    by_step: dict[int, dict[str, str]] = {}
    for row in rows:
        step = int(f(row.get("post_handoff_step"), -1))
        if step >= 0:
            by_step[step] = row
    timeline = [by_step[k] for k in sorted(by_step)]
    if not timeline:
        return "NOT_AVAILABLE", {}
    first, last = timeline[0], timeline[-1]
    r0, r1 = f(first.get("R_eff_h_nm")), f(last.get("R_eff_h_nm"))
    h0, h1 = f(first.get("h_integral")), f(last.get("h_integral"))
    p0, p1 = f(first.get("support_phi_gt_0p5")), f(last.get("support_phi_gt_0p5"))
    rel_r = (r1 - r0) / r0 if r0 else math.nan
    rel_h = (h1 - h0) / h0 if h0 else math.nan
    rel_p = (p1 - p0) / p0 if p0 else math.nan
    last_step = f(last.get("post_handoff_step"))
    late = slope([(f(row.get("post_handoff_step")), f(row.get("R_eff_h_nm")))
                  for row in timeline if f(row.get("post_handoff_step")) >= last_step - 500])
    if f(last.get("phi_max")) < 0.05 or p1 <= 0 or h1 < 0.01 * h0:
        fate = "COLLAPSE"
    elif rel_r >= 0.03 and rel_h >= 0.03 and rel_p >= 0.03 and late > 0.0:
        fate = "GROW"
    elif rel_r >= -0.01 and rel_h >= -0.02 and rel_p >= -0.02 and abs(late) <= 1.0e-4:
        fate = "ROBUST_STABLE"
    elif rel_r < 0.0 and rel_h < 0.0:
        fate = "SHRINK"
    else:
        fate = "WINDOW_AMBIGUOUS"
    return fate, {
        "R_eff_h_initial_nm": r0, "R_eff_h_final_nm": r1,
        "relative_delta_R_eff_h": rel_r, "h_integral_initial": h0,
        "h_integral_final": h1, "relative_delta_h_integral": rel_h,
        "support_phi_gt_0p5_initial": p0, "support_phi_gt_0p5_final": p1,
        "relative_delta_support_phi_gt_0p5": rel_p, "R_slope_last_500": late,
        "post_handoff_steps_observed": last_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "params/resolved_beta_growth_remediation/manifest.csv")
    parser.add_argument("--run-root", type=Path, default=ROOT / "tmp_codex_ops/resolved_beta_growth_remediation_phase1")
    parser.add_argument("--report-root", type=Path, default=ROOT / "reports/resolved_beta_growth_remediation")
    args = parser.parse_args()
    manifest = read_csv(args.manifest)
    summary: list[dict[str, object]] = []
    all_ts: list[dict[str, object]] = []
    rhs_summary: list[dict[str, object]] = []
    for meta in manifest:
        case = meta["case"]
        case_dir = args.run_root / case
        status = (case_dir / "status.txt").read_text().strip() if (case_dir / "status.txt").exists() else "NOT_RUN"
        out = find_output(case_dir)
        growth = read_csv(out / "diagnostic_rsmd_seed_growth_time_series.csv") if out else []
        fate, metrics = classify(growth)
        ledger = read_csv(out / "diagnostic_rsmd_mass_ledger.csv") if out else []
        max_mass_error = max((abs(f(row.get("mass_error_rel"), 0.0)) for row in ledger), default=math.nan)
        rhs = read_csv(out / f"{case}_per_step.csv") if out else []
        rhs_rows = [
            {
                "case": case,
                "step": row.get("step", ""),
                "phi_rhs_chem_signed_mean": row.get("phi_rhs_chem_signed_mean", ""),
                "phi_rhs_chem_abs_max": row.get("phi_rhs_chem_abs_max", ""),
                "phi_rhs_double_well_signed_mean": row.get("phi_rhs_double_well_signed_mean", ""),
                "phi_rhs_elastic_signed_mean": row.get("phi_rhs_elastic_signed_mean", ""),
                "phi_rhs_total_explicit_signed_mean": row.get("phi_rhs_total_explicit_signed_mean", ""),
                "dphi_actual_signed_mean": row.get("dphi_actual_signed_mean", ""),
                "xB_min_after": row.get("xB_min_after", ""),
                "xB_max_after": row.get("xB_max_after", ""),
                "nan_inf_flag": row.get("nan_inf_flag", ""),
            }
            for row in rhs
        ]
        all_ts.extend([{**meta, **row} for row in growth])
        rhs_summary.extend(rhs_rows)
        summary.append({
            **meta,
            "status": status,
            "case_output_dir": str(out) if out else "NOT_AVAILABLE",
            "fate": fate,
            "max_mass_error_rel": max_mass_error,
            "rhs_rows": len(rhs),
            "nan_inf_seen": any(row.get("nan_inf_flag") not in ("", "0", "0.0") for row in rhs),
            **metrics,
        })
    write_csv(args.report_root / "remediation_summary.csv", summary)
    write_csv(args.report_root / "remediation_seed_time_series.csv", all_ts)
    write_csv(args.report_root / "remediation_phi_rhs_summary.csv", rhs_summary)
    done = [row for row in summary if row["status"] == "EXIT 0"]
    counts = defaultdict(int)
    for row in done:
        counts[str(row["fate"])] += 1
    report = f"""# Resolved Beta Growth Remediation Diagnostic

This is a workstation-only timestep and alpha-side mask sensitivity test at the existing upper-guard source target. It does not modify the PF equation, GP inventory ledger, seed profile, or source provenance.

- prepared cases: {len(summary)}
- completed cases: {len(done)}
- completed fate counts: {dict(counts)}
- status criterion for growth: simultaneous >3% growth of `R_eff_h`, `h_integral`, and `phi>0.5` support with positive late radius slope.

The result distinguishes numerical timestep sensitivity from the effect of extending the source's matrix-side eligibility mask from `h<0.10` to `h<0.49`. It remains a diagnostic source scenario, not a calibrated GP release model.
"""
    args.report_root.mkdir(parents=True, exist_ok=True)
    (args.report_root / "resolved_beta_growth_remediation_report.md").write_text(report)
    print(f"prepared={len(summary)} completed={len(done)} fates={dict(counts)}")


if __name__ == "__main__":
    main()
