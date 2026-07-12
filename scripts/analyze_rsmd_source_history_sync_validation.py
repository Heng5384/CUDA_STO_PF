#!/usr/bin/env python3
"""Summarize diagnostic RSMD source-history synchronization smokes."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def value(item: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(item.get(key, ""))
    except ValueError:
        return default


def unique_growth(data: list[dict[str, str]]) -> list[dict[str, str]]:
    by_step = {int(value(item, "post_handoff_step", -1)): item for item in data}
    return [by_step[step] for step in sorted(by_step) if step >= 0]


def parse_case(spec: str) -> tuple[str, Path]:
    name, separator, path = spec.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("case must be NAME=OUTPUT_DIR")
    return name, Path(path)


def write_csv(path: Path, data: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for item in data:
        for field in item:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def slope(points: list[tuple[float, float]]) -> float:
    usable = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    if len(usable) < 2:
        return math.nan
    x_mean = sum(x for x, _ in usable) / len(usable)
    y_mean = sum(y for _, y in usable) / len(usable)
    denominator = sum((x - x_mean) ** 2 for x, _ in usable)
    if denominator == 0.0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in usable) / denominator


def support_radius(cells: float, dx_nm: float = 1.0) -> float:
    if not math.isfinite(cells) or cells <= 0.0:
        return math.nan
    return (3.0 * cells * dx_nm**3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def summarize(name: str, output: Path) -> dict[str, object]:
    growth = unique_growth(rows(output / "diagnostic_rsmd_seed_growth_time_series.csv"))
    projection = rows(output / "y_update_mass_projection.csv")
    source = rows(output / "diagnostic_rsmd_matrix_halo_source_normalization.csv")
    ledger = rows(output / "diagnostic_rsmd_mass_ledger.csv")
    config_rows = rows(output / "diagnostic_rsmd_runtime_config.csv")
    config = config_rows[-1] if config_rows else {}
    first = growth[0] if growth else {}
    last = growth[-1] if growth else {}
    r0, r1 = value(first, "R_eff_h_nm"), value(last, "R_eff_h_nm")
    h0, h1 = value(first, "h_integral"), value(last, "h_integral")
    p0 = support_radius(value(first, "support_phi_gt_0p5"))
    p1 = support_radius(value(last, "support_phi_gt_0p5"))
    final_post = value(last, "post_handoff_step", -1)
    late = [
        (value(item, "post_handoff_step"), value(item, "R_eff_h_nm"))
        for item in growth
        if value(item, "post_handoff_step") >= final_post - 200
    ]
    first_bound = next(
        (int(value(item, "step", -1)) for item in projection
         if value(item, "max_xB_before_projection", -math.inf) >= 0.99),
        None,
    )
    source_mass = sum(value(item, "applied_mass", 0.0) for item in source)
    beta_response = max(h1 - h0, 0.0) if math.isfinite(h0) and math.isfinite(h1) else math.nan
    return {
        "case": name,
        "output_dir": str(output),
        "T_C": value(config, "T_C"),
        "chi_rel": value(config, "chi_rel"),
        "history_reset_enabled": int(value(config, "reset_Y_history_after_source", 0)),
        "final_step": int(value(last, "step", -1)),
        "final_post_handoff_step": int(value(last, "post_handoff_step", -1)),
        "physical_time_s": value(last, "physical_time_s"),
        "R_eff_h_initial_nm": r0,
        "R_eff_h_final_nm": r1,
        "relative_delta_R_eff_h": (r1 - r0) / r0 if r0 else math.nan,
        "h_integral_initial": h0,
        "h_integral_final": h1,
        "relative_delta_h_integral": (h1 - h0) / h0 if h0 else math.nan,
        "phi0p5_support_radius_initial_nm": p0,
        "phi0p5_support_radius_final_nm": p1,
        "relative_delta_phi0p5_support_radius": (p1 - p0) / p0 if p0 else math.nan,
        "late_200_R_slope_nm_per_step": slope(late),
        "final_phi_max": value(last, "phi_max"),
        "final_halo_xB_mean": value(last, "halo_xB_mean"),
        "final_halo_xB_max": value(last, "halo_xB_max"),
        "final_far_field_xB_mean": value(last, "far_field_xB_mean"),
        "first_xB_ge_0p99_step": first_bound if first_bound is not None else "",
        "max_xB_before_projection": max(
            (value(item, "max_xB_before_projection", -math.inf) for item in projection),
            default=math.nan,
        ),
        "max_abs_projection_correction": max(
            (abs(value(item, "delta_before_projection", 0.0)) for item in projection),
            default=math.nan,
        ),
        "source_rows": len(source),
        "history_reset_rows": sum(
            int(value(item, "Y_history_reset_after_source", 0)) == 1 for item in source
        ),
        "source_normalization_failures": sum(
            int(value(item, "normalization_pass", 0)) != 1 for item in source
        ),
        "source_mass_applied": source_mass,
        "positive_beta_h_response": beta_response,
        "eta_beta_h_response": beta_response / source_mass if source_mass > 0.0 else math.nan,
        "max_mass_error_rel": max(
            (abs(value(item, "mass_error_rel", 0.0)) for item in ledger),
            default=math.nan,
        ),
        "fate_final": last.get("fate_running", "missing"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    summary = [summarize(name, path) for name, path in args.case]
    args.report_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_root / "source_history_sync_case_summary.csv", summary)

    report = [
        "# RSMD Source-History Synchronization Validation",
        "",
        "This is a numerical consistency and scenario-bracketing test. It is not a calibrated GP release law.",
        "",
        "## Results",
        "",
    ]
    for item in summary:
        report.append(
            f"- `{item['case']}`: T={item['T_C']:.0f} C, chi={item['chi_rel']:.3g}, "
            f"post-handoff steps={item['final_post_handoff_step']}, "
            f"dR/R={item['relative_delta_R_eff_h']:.6g}, "
            f"dh/h={item['relative_delta_h_integral']:.6g}, "
            f"max xB={item['max_xB_before_projection']:.6g}, "
            f"max mass error={item['max_mass_error_rel']:.3e}."
        )
    completed = all(item["final_step"] > 0 for item in summary)
    reset_ok = all(
        item["history_reset_enabled"] == 1
        and item["source_rows"] > 0
        and item["history_reset_rows"] == item["source_rows"]
        and item["source_normalization_failures"] == 0
        for item in summary
    )
    bounded = all(item["first_xB_ge_0p99_step"] == "" for item in summary)
    mass_ok = all(item["max_mass_error_rel"] <= 1.0e-10 for item in summary)
    status = (
        "PASS_RSMD_SOURCE_HISTORY_SYNC_SMOKE"
        if completed and reset_ok and bounded and mass_ok
        else "PARTIAL_RSMD_SOURCE_HISTORY_SYNC_VALIDATION"
    )
    report.extend(
        [
            "",
            "## Boundary",
            "",
            "Passing establishes that the external source no longer reuses a stale local dY/dt history and that the tested bracket remains bounded. It does not calibrate chi_rel or the GP release kinetics.",
            "",
            f"final_status={status}",
            "",
        ]
    )
    (args.report_root / "rsmd_source_history_sync_validation_report.md").write_text(
        "\n".join(report)
    )
    (args.report_root / "final_terminal_output.txt").write_text(f"final_status={status}\n")
    print(f"final_status={status}")


if __name__ == "__main__":
    main()
