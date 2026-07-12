#!/usr/bin/env python3
"""Assess the early-time numerical transport repair without reinterpreting physics."""

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


def f(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return math.nan


def output_dir(case: Path) -> Path | None:
    stdout = case / "stdout.log"
    if not stdout.exists():
        return None
    matches = re.findall(r"case_output_dir\s*:\s*(\S+)", stdout.read_text(errors="ignore"))
    return Path(matches[-1]) if matches and Path(matches[-1]).exists() else None


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

    records: list[dict[str, object]] = []
    for meta in read_csv(args.manifest):
        case = args.run_root / meta["run_id"]
        status = (case / "status.txt").read_text().strip() if (case / "status.txt").exists() else "NOT_RUN"
        out = output_dir(case)
        timeline = read_csv(out / "diagnostic_rsmd_seed_growth_time_series.csv") if out else []
        normal = read_csv(out / "diagnostic_rsmd_matrix_halo_source_normalization.csv") if out else []
        stdout = (case / "stdout.log").read_text(errors="ignore") if (case / "stdout.log").exists() else ""
        values = re.findall(
            r"xB_range=\[([-+0-9.eE]+), ([-+0-9.eE]+)\] Meff_range=\[([-+0-9.eE]+), ([-+0-9.eE]+)\]",
            stdout,
        )
        mins = [f(row[0]) for row in values]
        maxs = [f(row[1]) for row in values]
        meff_maxs = [f(row[3]) for row in values]
        first = timeline[0] if timeline else {}
        last = timeline[-1] if timeline else {}
        r0, r1 = f(first.get("R_eff_h_nm")), f(last.get("R_eff_h_nm"))
        h0, h1 = f(first.get("h_integral")), f(last.get("h_integral"))
        cap = 1000.0 * (9.0 if meta["temperature_C"] == "400" else 7.471117584203761)
        source_ok = (not normal or all(int(f(row.get("normalization_pass"))) == 1 for row in normal))
        records.append({
            **meta,
            "status": status,
            "case_output_dir": str(out) if out else "NOT_AVAILABLE",
            "R_eff_h_initial_nm": r0,
            "R_eff_h_final_nm": r1,
            "relative_delta_R_eff_h": (r1 - r0) / r0 if math.isfinite(r0) and r0 else math.nan,
            "h_integral_initial": h0,
            "h_integral_final": h1,
            "relative_delta_h_integral": (h1 - h0) / h0 if math.isfinite(h0) and h0 else math.nan,
            "xB_min_over_log": min(mins, default=math.nan),
            "xB_max_over_log": max(maxs, default=math.nan),
            "Meff_max_over_log": max(meff_maxs, default=math.nan),
            "Meff_cap": cap,
            "xB_boundary_saturated": bool(min(mins, default=1.0) <= 1.0e-4 or max(maxs, default=0.0) >= 1.0 - 1.0e-4),
            "Meff_guard_engaged": bool(max(meff_maxs, default=0.0) >= 0.999 * cap),
            "source_normalization_pass": source_ok,
            "mass_error_rel_final": f(last.get("mass_error_rel")),
            "fate_final": last.get("fate_running", "NOT_AVAILABLE"),
        })

    all_ok = records and all(row["status"] == "EXIT 0" for row in records)
    source_ok = all(bool(row["source_normalization_pass"]) for row in records)
    mass_ok = all(abs(f(row["mass_error_rel_final"])) <= 1.0e-10 for row in records)
    numerical_blowup = any(not math.isfinite(f(row["h_integral_final"])) for row in records)
    status = ("PASS_TRANSPORT_MOBILITY_CONSISTENCY_SMOKE" if all_ok and source_ok and mass_ok and not numerical_blowup
              else "FAIL_TRANSPORT_MOBILITY_CONSISTENCY_SMOKE")
    write_csv(args.report_root / "transport_mobility_consistency_smoke_summary.csv", records)
    report = "\n".join([
        "# Transport Mobility Consistency Smoke",
        "",
        f"final_status={status}",
        f"completed_cases={sum(row['status'] == 'EXIT 0' for row in records)}/{len(records)}",
        f"source_normalization_pass={source_ok}",
        f"mass_closure_pass={mass_ok}",
        f"nonfinite_h_integral_detected={numerical_blowup}",
        "",
        "The test distinguishes a bounded transport update from a claim of physical beta growth. "
        "A seed may shrink physically after the mobility inconsistency is removed.",
        "",
    ])
    (args.report_root / "transport_mobility_consistency_smoke_report.md").write_text(report)
    print(status)


if __name__ == "__main__":
    main()
