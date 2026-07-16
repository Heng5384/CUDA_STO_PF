#!/usr/bin/env python3
"""Analyze Lie-BE v2 one-full-step versus two-half-step replays."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_staggered_v1_equal_time import (
    compare_candidate,
    load_state,
)


SHAPE = (512, 1, 1)
DX_NM = 1.0
TIME_UNIT_S = 41.12958542455477
INCREMENT_QOI_TOL = 5.0e-3
INTERFACE_POSITION_ABS_TOL_NM = 1.0e-4


def output_dir(case_dir: Path) -> Path:
    candidates = list((case_dir / "run").glob("Results/**/ctot_split_step_metrics.csv"))
    if len(candidates) != 1:
        raise ValueError(f"{case_dir}: expected one metrics file, got {len(candidates)}")
    return candidates[0].parent


def checkpoint_prefix(case_dir: Path, step: int) -> Path:
    directory = output_dir(case_dir)
    prefix = directory / f"ctot_checkpoint_step{step:06d}"
    for suffix in ("_Ctot.raw", "_phi.raw", "_xB_alpha.raw"):
        if not Path(str(prefix) + suffix).is_file():
            raise FileNotFoundError(str(prefix) + suffix)
    return prefix


def runtime_metrics(case_dir: Path) -> dict[str, float | int | bool]:
    directory = output_dir(case_dir)
    with (directory / "ctot_split_step_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{case_dir}: empty split metrics")
    perf = json.loads((directory / "performance_summary.json").read_text(encoding="utf-8"))
    return {
        "steps": len(rows),
        "all_accepted": all(int(row["accepted"]) == 1 for row in rows),
        "energy_all_pass": all(int(row["energy_audit_pass"]) == 1 for row in rows),
        "method_transport_gate_all_pass": all(
            int(row["method_transport_gate_pass"]) == 1 for row in rows
        ),
        "method_transport_residual_max": max(
            float(row["method_transport_residual"]) for row in rows
        ),
        "final_phi_split_residual_max": max(
            float(row["final_phi_split_residual"]) for row in rows
        ),
        "phase_KKT_max": max(float(row["phase_KKT"]) for row in rows),
        "mass_error_abs_max": max(abs(float(row["mass_error"])) for row in rows),
        "sum_divJ_abs_max": max(abs(float(row["sum_divJ"])) for row in rows),
        "polish_applied_count": sum(int(row["polish_applied"]) for row in rows),
        "polish_skipped_count": sum(int(row["polish_skipped"]) for row in rows),
        "transport_solves": sum(int(row["transport_solves"]) for row in rows),
        "phase_solves": sum(int(row["phase_solves"]) for row in rows),
        "mechanics_solves": sum(int(row["mechanics_solves"]) for row in rows),
        "wall_s": float(perf["total_walltime_s"]),
    }


def observed_order(coarse_error: float, fine_error: float) -> float:
    if not (coarse_error > 0.0 and fine_error > 0.0):
        return math.nan
    return math.log(coarse_error / fine_error, 2.0)


def choose_selected(rows: list[dict[str, object]]) -> int | None:
    eligible = [row for row in rows if row["time_error_gate_pass"]]
    if not eligible:
        return None
    selected = max(eligible, key=lambda row: float(row["throughput_code_time_per_wall_s"]))
    return int(selected["divisor"])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--initial-prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.results_root / "manifest.json").read_text(encoding="utf-8"))
    initial = load_state(args.initial_prefix, SHAPE)
    rows: list[dict[str, object]] = []
    previous: dict[str, object] | None = None
    for divisor in manifest["divisors"]:
        full_id = f"T400_step655_lie_be_v2_dt_div{divisor}_full"
        half_id = f"T400_step655_lie_be_v2_dt_div{divisor}_two_half"
        full_dir = args.results_root / "cases" / full_id
        half_dir = args.results_root / "cases" / half_id
        full = load_state(checkpoint_prefix(full_dir, 1), SHAPE)
        half = load_state(checkpoint_prefix(half_dir, 2), SHAPE)
        comparison = compare_candidate(
            full_id, initial, half, full, DX_NM, INCREMENT_QOI_TOL
        )
        full_runtime = runtime_metrics(full_dir)
        half_runtime = runtime_metrics(half_dir)
        dt = float(manifest["base_dt"]) / int(divisor)
        row: dict[str, object] = {
            "divisor": divisor,
            "dt": dt,
            "half_dt": dt / 2.0,
            "equal_time_code": dt,
            "equal_time_s": dt * TIME_UNIT_S,
            "Ctot_increment_error_Linf_abs": comparison["Ctot_abs_linf"],
            "Ctot_increment_error_L2_rel": comparison[
                "Ctot_increment_error_l2_rel"
            ],
            "phi_increment_error_Linf_abs": comparison["phi_abs_linf"],
            "phi_increment_error_L2_rel": comparison[
                "phi_increment_error_l2_rel"
            ],
            "h_volume_increment_error_abs": comparison["h_delta_error_abs"],
            "h_volume_increment_error_rel": comparison["h_delta_error_rel"],
            "interface_position_error_nm": comparison["interface_shift_error_max"],
            "Ctot_observed_order": math.nan,
            "phi_observed_order": math.nan,
            "h_volume_observed_order": math.nan,
            "full_method_transport_residual_max": full_runtime[
                "method_transport_residual_max"
            ],
            "full_final_phi_split_residual_max": full_runtime[
                "final_phi_split_residual_max"
            ],
            "full_phase_KKT_max": full_runtime["phase_KKT_max"],
            "full_mass_error_abs_max": full_runtime["mass_error_abs_max"],
            "full_sum_divJ_abs_max": full_runtime["sum_divJ_abs_max"],
            "full_transport_solves": full_runtime["transport_solves"],
            "full_phase_solves": full_runtime["phase_solves"],
            "full_mechanics_solves": full_runtime["mechanics_solves"],
            "full_wall_s": full_runtime["wall_s"],
            "half_reference_wall_s": half_runtime["wall_s"],
            "throughput_code_time_per_wall_s": dt / float(full_runtime["wall_s"]),
            "throughput_physical_s_per_wall_h":
                dt * TIME_UNIT_S / float(full_runtime["wall_s"]) * 3600.0,
            "hard_gates_pass": bool(
                full_runtime["all_accepted"]
                and full_runtime["energy_all_pass"]
                and full_runtime["method_transport_gate_all_pass"]
                and half_runtime["all_accepted"]
                and half_runtime["energy_all_pass"]
                and half_runtime["method_transport_gate_all_pass"]
                and float(full_runtime["mass_error_abs_max"]) <= 1.0e-10
                and int(full_runtime["polish_applied_count"]) == 0
                and int(full_runtime["polish_skipped_count"]) == 0
            ),
            "time_error_gate_pass": False,
            "selected": False,
        }
        row["time_error_gate_pass"] = bool(
            row["hard_gates_pass"]
            and float(row["Ctot_increment_error_L2_rel"]) <= INCREMENT_QOI_TOL
            and float(row["phi_increment_error_L2_rel"]) <= INCREMENT_QOI_TOL
            and float(row["h_volume_increment_error_rel"]) <= INCREMENT_QOI_TOL
            and float(row["interface_position_error_nm"])
                <= INTERFACE_POSITION_ABS_TOL_NM
        )
        if previous is not None:
            row["Ctot_observed_order"] = observed_order(
                float(previous["Ctot_increment_error_L2_rel"]),
                float(row["Ctot_increment_error_L2_rel"]),
            )
            row["phi_observed_order"] = observed_order(
                float(previous["phi_increment_error_L2_rel"]),
                float(row["phi_increment_error_L2_rel"]),
            )
            row["h_volume_observed_order"] = observed_order(
                float(previous["h_volume_increment_error_rel"]),
                float(row["h_volume_increment_error_rel"]),
            )
        rows.append(row)
        previous = row

    selected = choose_selected(rows)
    if selected is not None:
        next(row for row in rows if int(row["divisor"]) == selected)["selected"] = True
    write_csv(args.output_dir / "step655_dt_ladder.csv", rows)
    summary = {
        "method": manifest["method"],
        "increment_qoi_tolerance": INCREMENT_QOI_TOL,
        "interface_position_abs_tolerance_nm": INTERFACE_POSITION_ABS_TOL_NM,
        "selected_divisor": selected,
        "selected_dt": None if selected is None else float(manifest["base_dt"]) / selected,
        "all_runtime_hard_gates_pass": all(row["hard_gates_pass"] for row in rows),
        "raw_state_modified": manifest["frozen_state"]["raw_state_modified"],
    }
    (args.output_dir / "step655_dt_ladder_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"selected_divisor={selected}")
    print(f"selected_dt={summary['selected_dt']}")
    print(f"all_runtime_hard_gates_pass={summary['all_runtime_hard_gates_pass']}")
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
