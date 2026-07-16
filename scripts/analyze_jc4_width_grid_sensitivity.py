#!/usr/bin/env python3
"""Analyze equal-time JC4 width/grid sensitivity against the frozen point."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/pf_ctot_production_candidate"
sys.path.insert(0, str(ROOT))
from scripts.analyze_jc4_long_time_planar import (  # noqa: E402
    analyze_case,
    write_rows,
)


def compare_to_baseline(metric: dict[str, object],
                        trajectory: list[dict[str, object]],
                        baseline: dict[str, object],
                        baseline_trajectory: list[dict[str, object]]) -> dict[str, object]:
    base_disp = float(baseline["PF_h_displacement_nm"])
    displacement = float(metric["PF_h_displacement_nm"])
    transfer_variation = abs(displacement - base_disp) / max(abs(base_disp), 1.0e-300)
    base_half = float(baseline["final_half_width_h_nm"])
    half = float(metric["final_half_width_h_nm"])
    phase_variation = abs(half - base_half) / max(abs(base_half), 1.0e-300)
    base_time = np.array([float(row["time_s"]) for row in baseline_trajectory])
    base_path = np.array([
        float(row["PF_displacement_h_nm"]) for row in baseline_trajectory
    ])
    time = np.array([float(row["time_s"]) for row in trajectory])
    path = np.array([float(row["PF_displacement_h_nm"]) for row in trajectory])
    common = np.unique(np.concatenate((base_time, time)))
    common = common[(common >= max(base_time[0], time[0]))
                    & (common <= min(base_time[-1], time[-1]))]
    base_interp = np.interp(common, base_time, base_path)
    path_interp = np.interp(common, time, path)
    trajectory_variation = float(np.max(np.abs(path_interp - base_interp))) / max(
        abs(base_disp), float(baseline["dx_nm"]), 1.0e-300
    )
    direction_same = (
        math.copysign(1.0, displacement) == math.copysign(1.0, base_disp)
        if displacement != 0.0 and base_disp != 0.0 else displacement == base_disp
    )
    acceptance = (
        bool(metric["numerical_hard_gates_pass"])
        and transfer_variation <= 0.10
        and trajectory_variation <= 0.15
        and phase_variation <= 0.05
        and direction_same
    )
    return {
        **metric,
        "baseline_case_id": baseline["case_id"],
        "cumulative_transfer_variation_rel": transfer_variation,
        "trajectory_variation_rel": trajectory_variation,
        "final_beta_fraction_variation_rel": phase_variation,
        "mechanism_direction_unchanged": direction_same,
        "width_grid_sensitivity_gate_pass": acceptance,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    matrix_root = args.matrix_root.resolve()
    raw: dict[str, tuple[dict[str, object], list[dict[str, object]]]] = {}
    manifests: dict[str, dict[str, object]] = {}
    for case in sorted((matrix_root / "cases").iterdir()):
        manifest_path = case / "runtime_manifest.json"
        if not manifest_path.is_file() or not (case / "run/run.log").is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metric, trajectory, _ = analyze_case(case)
        raw[str(manifest["case_id"])] = (metric, trajectory)
        manifests[str(manifest["case_id"])] = manifest
    if len(raw) != 8:
        raise RuntimeError(f"expected all 8 completed sensitivity cases, got {len(raw)}")

    rows: list[dict[str, object]] = []
    for case_id, (metric, trajectory) in sorted(raw.items()):
        manifest = manifests[case_id]
        baseline_id = f"T400_{manifest['direction']}_production_dx1_lambda4"
        baseline, baseline_trajectory = raw[baseline_id]
        row = compare_to_baseline(
            metric, trajectory, baseline, baseline_trajectory
        )
        row.update({
            "sensitivity_matrix": manifest["sensitivity_matrix"],
            "lambda_over_dx": manifest["lambda_over_dx"],
            "runtime_selector": manifest["runtime_selector"],
            "comparator_provenance": manifest["comparator_provenance"],
            "same_physical_time": manifest["same_physical_time"],
            "same_total_inventory": manifest["same_total_inventory"],
        })
        rows.append(row)

    report_root = args.report_root.resolve()
    write_rows(report_root / "jc4_width_grid_metrics.csv", rows)
    table = "\n".join(
        f"| {row['case_id']} | {float(row['dx_nm']):.3g} | "
        f"{float(row['lambda_nm']):.3g} | "
        f"{float(row['cumulative_transfer_variation_rel']):.3%} | "
        f"{float(row['trajectory_variation_rel']):.3%} | "
        f"{float(row['final_beta_fraction_variation_rel']):.3%} | "
        f"{row['width_grid_sensitivity_gate_pass']} |"
        for row in rows
    )
    passed = all(bool(row["width_grid_sensitivity_gate_pass"]) for row in rows)
    (report_root / "jc4_width_grid_sensitivity.md").write_text(f"""# JC4 interface-width and grid sensitivity

All rows use identical physical boxes, total `C_B_tot` inventory, temperature,
thermodynamics, `D_alpha`, and elapsed physical time. `W`, `kappa`, and the
strict one-sided `Lphi_diff` are recomputed for each physical `lambda`. Only
`dx=1 nm, lambda=4 nm` selects the frozen JC4 model; all other rows are
explicit comparators and cannot relax its runtime contract.

| case | dx (nm) | lambda (nm) | transfer variation | trajectory variation | beta-fraction variation | gate |
|---|---:|---:|---:|---:|---:|---|
{table}

Overall width/grid sensitivity status: **{'PASS' if passed else 'FAIL'}**.
""", encoding="utf-8")
    print(f"jc4_width_grid_cases_analyzed={len(rows)}")
    print(f"jc4_width_grid_sensitivity_status={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
