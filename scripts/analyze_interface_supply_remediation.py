#!/usr/bin/env python3
"""Acceptance analysis for bounded GP-to-interface relay diagnostics.

The source ceiling constrains the cells written by the diagnostic relay.  It is
not a global hard cap on later phase-field redistribution.  In particular, the
legacy ``halo_xB_max`` diagnostic is referenced to the *initial* seed radius;
once a seed grows, that fixed annulus can overlap a moving interface.  We keep
that maximum as an observational diagnostic, but assess source overdrive from
the actual source transaction records.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: object, default: float = math.nan) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def finite(value: float) -> bool:
    return math.isfinite(value)


def slope(points: list[tuple[float, float]]) -> float:
    valid = [(x, y) for x, y in points if finite(x) and finite(y)]
    if len(valid) < 2:
        return math.nan
    xbar = mean(x for x, _ in valid)
    ybar = mean(y for _, y in valid)
    den = sum((x - xbar) ** 2 for x, _ in valid)
    return sum((x - xbar) * (y - ybar) for x, y in valid) / den if den else 0.0


def find_output(case_dir: Path) -> Path | None:
    stdout = case_dir / "stdout.log"
    if not stdout.exists():
        return None
    hits = re.findall(r"case_output_dir\s*:\s*(\S+)", stdout.read_text(errors="ignore"))
    for raw in reversed(hits):
        output = Path(raw)
        if output.exists():
            return output
    return None


def classify(rows: list[dict[str, str]]) -> tuple[str, dict[str, float]]:
    by_step: dict[int, dict[str, str]] = {}
    for row in rows:
        post = int(number(row.get("post_handoff_step"), -1))
        if post >= 0:
            by_step[post] = row
    timeline = [by_step[step] for step in sorted(by_step)]
    if not timeline:
        return "NOT_AVAILABLE", {}
    first, last = timeline[0], timeline[-1]
    r0, r1 = number(first.get("R_eff_h_nm")), number(last.get("R_eff_h_nm"))
    h0, h1 = number(first.get("h_integral")), number(last.get("h_integral"))
    p0, p1 = number(first.get("support_phi_gt_0p5")), number(last.get("support_phi_gt_0p5"))
    rel_r = (r1 - r0) / r0 if finite(r0) and r0 else math.nan
    rel_h = (h1 - h0) / h0 if finite(h0) and h0 else math.nan
    rel_p = (p1 - p0) / p0 if finite(p0) and p0 else math.nan
    last_post = number(last.get("post_handoff_step"))
    late = slope([
        (number(row.get("post_handoff_step")), number(row.get("R_eff_h_nm")))
        for row in timeline
        if number(row.get("post_handoff_step")) >= last_post - 500
    ])
    if not finite(r1) or number(last.get("phi_max")) < 0.05 or p1 <= 0.0:
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
        "R_eff_h_initial_nm": r0,
        "R_eff_h_final_nm": r1,
        "relative_delta_R_eff_h": rel_r,
        "h_integral_initial": h0,
        "h_integral_final": h1,
        "relative_delta_h_integral": rel_h,
        "support_phi_gt_0p5_initial": p0,
        "support_phi_gt_0p5_final": p1,
        "relative_delta_support_phi_gt_0p5": rel_p,
        "R_slope_last_500": late,
        "post_handoff_steps_observed": last_post,
    }


def source_transaction_metrics(
    projection_rows: list[dict[str, str]], release_rows: list[dict[str, str]]
) -> dict[str, object]:
    """Separate source-write boundedness from later PF interface enrichment."""
    before: dict[str, dict[str, str]] = {}
    after: dict[str, dict[str, str]] = {}
    for row in projection_rows:
        step = row.get("step", "")
        if row.get("label") == "before_diagnostic_rsmd_source":
            before[step] = row
        elif row.get("label") == "after_diagnostic_rsmd_interface_shell_source":
            after[step] = row

    source_max_increase = -math.inf
    paired_steps = 0
    for step, pre in before.items():
        post = after.get(step)
        if not post:
            continue
        paired_steps += 1
        source_max_increase = max(
            source_max_increase,
            number(post.get("halo_xB_max")) - number(pre.get("halo_xB_max")),
        )

    local_after_max = -math.inf
    local_cap_violation = 0
    valid_release_rows = 0
    for row in release_rows:
        after_value = number(row.get("local_halo_xB_after"))
        target_value = number(row.get("xB_halo_target"))
        if not finite(after_value) or not finite(target_value):
            continue
        valid_release_rows += 1
        local_after_max = max(local_after_max, after_value)
        if after_value > target_value + 1.0e-12:
            local_cap_violation += 1

    return {
        "source_projection_pairs": paired_steps,
        "max_source_step_delta_halo_xB_max": (
            source_max_increase if paired_steps else math.nan
        ),
        "source_release_rows_checked": valid_release_rows,
        "source_local_halo_xB_after_max": (
            local_after_max if valid_release_rows else math.nan
        ),
        "source_local_cap_violation_rows": local_cap_violation,
        "source_write_ceiling_respected": (
            valid_release_rows > 0 and local_cap_violation == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument(
        "--halo-xB-upper",
        type=float,
        default=None,
        help=(
            "Optional absolute fixed-reference halo diagnostic limit. It is not "
            "used by default because this halo moves relative to a growing seed."
        ),
    )
    parser.add_argument("--far-field-upper", type=float, default=0.010)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for meta in read_csv(args.manifest):
        case = meta["case"]
        case_dir = args.run_root / case
        status_path = case_dir / "status.txt"
        status = status_path.read_text().strip() if status_path.exists() else "NOT_RUN"
        output = find_output(case_dir)
        timeline = read_csv(output / "diagnostic_rsmd_seed_growth_time_series.csv") if output else []
        ledger = read_csv(output / "diagnostic_rsmd_mass_ledger.csv") if output else []
        projection = read_csv(output / "diagnostic_rsmd_projection_effect_on_halo.csv") if output else []
        release = read_csv(output / "diagnostic_rsmd_release_event_log.csv") if output else []
        fate, metrics = classify(timeline)
        max_halo = max((number(row.get("halo_xB_max"), -math.inf) for row in timeline),
                       default=math.nan)
        max_far = max((number(row.get("far_field_xB_mean"), -math.inf) for row in timeline),
                      default=math.nan)
        max_mass = max((abs(number(row.get("mass_error_rel"), 0.0)) for row in ledger),
                       default=math.nan)
        source_metrics = source_transaction_metrics(projection, release)
        source_capped = bool(source_metrics["source_write_ceiling_respected"])
        absolute_halo_pass = (
            args.halo_xB_upper is None
            or (finite(max_halo) and max_halo <= args.halo_xB_upper)
        )
        far_preserved = finite(max_far) and max_far < args.far_field_upper
        mass_closed = finite(max_mass) and max_mass <= 1.0e-10
        if status != "EXIT 0":
            verdict = "PENDING"
        elif not mass_closed:
            verdict = "FAIL_MASS_CLOSURE"
        elif not source_capped:
            verdict = "FAIL_SOURCE_WRITE_CEILING"
        elif not absolute_halo_pass:
            verdict = "OBSERVED_FIXED_REFERENCE_HALO_EXCURSION"
        elif not far_preserved:
            verdict = "FAIL_FAR_FIELD_PRESERVATION"
        elif fate in {"GROW", "ROBUST_STABLE"}:
            verdict = "PASS_CLEAN_INTERFACE_SUPPLY_RESPONSE"
        else:
            verdict = "CLEAN_BUT_NO_GROWTH"
        rows.append({
            **meta,
            "status": status,
            "case_output_dir": str(output) if output else "NOT_AVAILABLE",
            "fate": fate,
            "max_halo_xB": max_halo,
            "max_far_field_xB": max_far,
            "max_mass_error_rel": max_mass,
            "halo_xB_upper_acceptance": args.halo_xB_upper,
            "far_field_xB_upper_acceptance": args.far_field_upper,
            "fixed_reference_halo_xB_peak_observed": max_halo,
            "absolute_fixed_reference_halo_check": absolute_halo_pass,
            "source_write_ceiling_respected": source_capped,
            "far_field_preserved": far_preserved,
            "mass_closed": mass_closed,
            "verdict": verdict,
            **source_metrics,
            **metrics,
        })
    write_csv(args.report_root / "interface_supply_remediation_summary.csv", rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["verdict"])] = counts.get(str(row["verdict"]), 0) + 1
    args.report_root.mkdir(parents=True, exist_ok=True)
    (args.report_root / "interface_supply_remediation_report.md").write_text(
        "# Interface Supply Remediation Acceptance\n\n"
        "The source ceiling is checked against source transaction records. A fixed-radius "
        "halo maximum is retained as an observational diagnostic because it can overlap a "
        "moving beta interface after seed growth; it is not treated as source overdrive by "
        "default. The report still rejects lost far-field preservation or failed mass closure. "
        "The relay mode remains a bounded required-supply scenario diagnostic, not a "
        "calibrated GP release thermodynamic law.\n\n"
        f"- verdict counts: `{counts}`\n"
        f"- optional fixed-reference halo xB limit: `{args.halo_xB_upper}`\n"
        f"- far-field xB upper acceptance: `{args.far_field_upper}`\n"
    )
    print(f"cases={len(rows)} verdicts={counts}")


if __name__ == "__main__":
    main()
