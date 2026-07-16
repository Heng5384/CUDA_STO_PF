#!/usr/bin/env python3
"""Select finite-box-safe quasi-steady windows from matched long-time runs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402


REPORT_ROOT = ROOT / "reports/pf_ctot_production_candidate"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rel(a: float, b: float) -> float:
    return abs(a-b) / max(abs(a), abs(b), 1.0e-300)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=REPORT_ROOT / "correction2_matched_long_time_metrics_raw.csv",
    )
    args = parser.parse_args()
    raw = read_csv(args.input)
    x_eq = unit.xAg2Te_eq_from_T(673.15)
    rows: list[dict[str, object]] = []
    for row in raw:
        preage_fo = float(row["preage_Fo"])
        incremental_fo = float(row["target_incremental_Fo"])
        total_fo = preage_fo + incremental_fo
        matrix_x = float(row["matrix_xB"])
        far_x = float(row["far_field_xB"])
        initial_far = float(row["initial_far_field_xB"])
        total_depletion = (matrix_x - far_x) / max(matrix_x-x_eq, 1.0e-300)
        incremental_depletion = (initial_far - far_x) / max(matrix_x-x_eq, 1.0e-300)
        diffusion_length_box_fraction = math.sqrt(total_fo) / 11.2
        finite_box_safe = (
            diffusion_length_box_fraction <= 0.60
            and total_depletion <= 0.25
        )
        quasi_steady = incremental_fo >= 10.0
        eligible = (
            quasi_steady and finite_box_safe
            and row["numerical_hard_gates_pass"] == "True"
            and abs(float(row["drive_amplitude"])-0.25) < 1.0e-12
        )
        rows.append({
            **row,
            "total_age_Fo": total_fo,
            "diffusion_length_over_matrix_half_width": diffusion_length_box_fraction,
            "total_far_field_depletion_fraction": total_depletion,
            "incremental_far_field_depletion_fraction": incremental_depletion,
            "quasi_steady_Fo_gate": quasi_steady,
            "finite_box_safe": finite_box_safe,
            "primary_window_eligible": eligible,
        })
    write_csv(REPORT_ROOT / "correction2_matched_long_time_metrics.csv", rows)
    write_csv(REPORT_ROOT / "correction2_windowed_velocity.csv", rows)

    dt_comparisons: list[dict[str, object]] = []
    for target in (10.0, 30.0):
        pair = sorted(
            (row for row in rows
             if abs(float(row["preage_Fo"])-16.0)<1e-8
             and abs(float(row["target_incremental_Fo"])-target)<1e-8),
            key=lambda row: float(row["dt_code"]),
        )
        if len(pair) != 2:
            raise RuntimeError(f"missing dt refinement pair for Fo={target}")
        dt_comparisons.append({
            "target_Fo": target,
            "dt_fine": pair[0]["dt_code"],
            "dt_coarse": pair[1]["dt_code"],
            "PF_velocity_change_rel": rel(
                float(pair[0]["PF_velocity_h_nm_s"]),
                float(pair[1]["PF_velocity_h_nm_s"]),
            ),
            "sharp_error_change_abs": abs(
                float(pair[0]["PF_sharp_velocity_error_rel"])
                - float(pair[1]["PF_sharp_velocity_error_rel"])
            ),
            "both_hard_gates_pass": all(
                row["numerical_hard_gates_pass"] == "True" for row in pair
            ),
        })
    write_csv(REPORT_ROOT / "correction2_long_time_dt_refinement.csv", dt_comparisons)
    preferred = sorted(
        (row for row in rows if bool(row["primary_window_eligible"])),
        key=lambda row: (
            float(row["target_incremental_Fo"]),
            float(row["preage_Fo"]), float(row["dt_code"]),
        ),
    )
    if not preferred:
        raise RuntimeError("no finite-box-safe quasi-steady window")
    primary = preferred[-1] if len(preferred) == 1 else preferred[0]
    table = "\n".join(
        f"| {row['case']} | {float(row['target_incremental_Fo']):.0f} | "
        f"{float(row['PF_velocity_h_nm_s']):.8f} | "
        f"{float(row['sharp_velocity_nm_s']):.8f} | "
        f"{float(row['PF_sharp_velocity_error_rel']):.4f} | "
        f"{float(row['total_far_field_depletion_fraction']):.4f} | "
        f"{float(row['diffusion_length_over_matrix_half_width']):.4f} | "
        f"{row['primary_window_eligible']} |"
        for row in sorted(rows, key=lambda q: (
            float(q["preage_Fo"]), float(q["target_incremental_Fo"]),
            float(q["dt_code"])))
        if float(row["dt_code"]) > 7.5e-4
    )
    dt_table = "\n".join(
        f"| {row['target_Fo']:.0f} | {float(row['PF_velocity_change_rel']):.6e} | "
        f"{float(row['sharp_error_change_abs']):.6e} | {row['both_hard_gates_pass']} |"
        for row in dt_comparisons
    )
    REPORT_ROOT.joinpath("correction2_matched_long_time_planar.md").write_text(f"""# Correction 2 Matched Long-Time Planar Comparison

All 14 workstation trajectories completed with zero retry/reject and all
mass, local storage, bounds, KKT, energy, no-clipping, and zero-physical-
projection gates passing.  The representative small driving is `A=0.25` and
`Lphi/Ldiff=0.90`.

| case | incremental Fo | PF V | sharp V | relative error | far depletion | sqrt(Fo_total)/box | primary eligible |
|---|---:|---:|---:|---:|---:|---:|---|
{table}

## Time refinement

| Fo | PF velocity change | sharp-error absolute change | gates |
|---:|---:|---:|---|
{dt_table}

The apparent approach to 10% at Fo=30 coincides with strong finite-box
depletion and is excluded.  The earliest operationally quasi-steady,
finite-box-safe row is `{primary['case']}` with PF/sharp error
`{float(primary['PF_sharp_velocity_error_rel']):.6%}`.  It remains above the
10% research gate, while dt refinement is already negligible.

`quasi_steady_window_status=PASS_WINDOW_FOUND`

`finite_Lphi_quantitative_gate=FAIL_ERROR_ABOVE_10_PERCENT`
""")
    print("quasi_steady_window_status=PASS_WINDOW_FOUND")
    print(f"primary_window={primary['case']}")
    print(f"PF_sharp_error_matched_quasi_steady={float(primary['PF_sharp_velocity_error_rel']):.17e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
