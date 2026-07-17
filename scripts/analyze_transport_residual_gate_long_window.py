#!/usr/bin/env python3
"""Compare long transport-gate trajectories at four registered windows."""

from __future__ import annotations

import csv
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from analyze_transport_residual_gate_v1 import (
    accuracy_metrics,
    load_state,
    optional_one,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "transport_residual_gate_v1"
RUNS = REPORT / "long_workstation_runs"
SHORT_RUNS = REPORT / "workstation_runs"
COMMON = ROOT / "reports" / "bounded_retry_bdf2_v1" / "workstation_runs" / "equal_time_common_input"
REFERENCE = ROOT / "reports" / "bounded_retry_bdf2_v1" / "workstation_runs" / "fine_dt32_common_state_16000"
T_REAL_UNIT_S = 41.12958542455477
DT_STEPS = {"dt32": 64000, "dt16": 32000, "dt8": 16000, "dt4": 8000, "dt2": 4000, "dt1": 2000}
COMMON_STEPS = {key: value // 4 for key, value in DT_STEPS.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def common_rows() -> list[dict[str, str]]:
    with (REPORT / "equal_time_metrics.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def find_case(gate_id: str, dt_id: str) -> Path:
    steps = DT_STEPS[dt_id]
    return RUNS / f"LONG_{gate_id}_{dt_id}_{steps}"


def common_case(gate_id: str, dt_id: str) -> Path:
    if dt_id == "dt32":
        return REFERENCE
    return SHORT_RUNS / f"{gate_id}_{dt_id}_diag_on_{COMMON_STEPS[dt_id]}"


def state_at_window(
    gate_id: str, dt_id: str, window: int
) -> dict[str, np.ndarray]:
    if window == 1:
        return load_state(common_case(gate_id, dt_id), COMMON_STEPS[dt_id])
    continuation_step = COMMON_STEPS[dt_id] * (window - 1)
    return load_state(find_case(gate_id, dt_id), continuation_step)


def raw_array(root: Path, name: str) -> np.ndarray:
    path = optional_one(root, name)
    if path is None:
        raise RuntimeError(f"missing {name} below {root}")
    return np.fromfile(path, np.float64)


def read_case(gate_id: str, dt_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = find_case(gate_id, dt_id)
    status = json.loads((root / "workstation_status.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "long_remote_summary.json").read_text(encoding="utf-8"))
    return status, summary


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def retry_efficiency_pass(
    summary: dict[str, Any], persistent_failed_cells: list[int],
    combined_max_consecutive_fallback: int,
) -> bool:
    bounded = summary.get("bounded_retry_summary", {})
    budget = as_int(bounded.get("nonlinear_iteration_budget"))
    p99 = as_float(bounded.get("accepted_iteration_p99"))
    return bool(
        as_int(bounded.get("macro_hard_rejects")) == 0
        and as_float(bounded.get("retry_fraction")) <= 0.01
        and as_float(bounded.get("fallback_fraction")) <= 0.01
        and as_float(bounded.get("reject_trial_wall_fraction")) <= 0.05
        and combined_max_consecutive_fallback <= 2
        and as_int(bounded.get("max_accepted_subcycle_depth")) <= 2
        and not persistent_failed_cells
        and budget > 0
        and math.isfinite(p99)
        and p99 < 0.8 * budget
    )


def failed_cell_attempt_counts(root: Path) -> Counter[int]:
    path = optional_one(root, "ctot_failed_cell_state.csv")
    if path is None:
        return Counter()
    worst: dict[tuple[int, int], tuple[float, int]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (as_int(row.get("physical_step"), -1),
                   as_int(row.get("attempt_id"), -1))
            candidate = (
                abs(as_float(row.get("residual"), 0.0)),
                as_int(row.get("idx"), -1),
            )
            if key not in worst or candidate[0] > worst[key][0]:
                worst[key] = candidate
    return Counter(item[1] for item in worst.values() if item[1] >= 0)


def event_fallback_steps(root: Path) -> list[int]:
    path = root / "run.log"
    if not path.is_file():
        return []
    result: list[int] = []
    pattern = re.compile(
        r"CTOT_IMEX_BDF2_HISTORY_COMMIT step=(\d+).*"
        r"accepted_integrator=EVENT_BE_SUBCYCLE"
    )
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = pattern.search(line)
            if match:
                result.append(int(match.group(1)))
    return result


def boundary_consecutive_fallback(
    common_steps: int, common_fallbacks: list[int],
    continuation_fallbacks: list[int],
) -> int:
    common_set = set(common_fallbacks)
    continuation_set = set(continuation_fallbacks)
    trailing = 0
    while common_steps - trailing in common_set:
        trailing += 1
    leading = 0
    while leading + 1 in continuation_set:
        leading += 1
    return trailing + leading


def continuation_hard_pass(
    summary: dict[str, Any], expected_time_code: float
) -> bool:
    trajectory = summary.get("trajectory_meta", {})
    accepted_rows = max(as_int(trajectory.get("accepted_rows")), 1)
    time_roundoff_bound = max(
        1.0e-12,
        2.0 * np.finfo(np.float64).eps * accepted_rows
        * max(abs(expected_time_code), 1.0),
    )
    return bool(
        summary.get("acceptance_hard_pass")
        and summary.get("energy_pass")
        and summary.get("rollback_pass")
        and as_float(summary.get("max_mass_error")) <= 1.0e-10
        and abs(as_float(trajectory.get("observed_time_code"))
                - expected_time_code) <= time_roundoff_bound
        and trajectory.get("schema")
        == "CTOT_TRANSPORT_GATE_TRAJECTORY_V2_TRANSACTIONAL"
    )


def main() -> int:
    initial = {
        "Ctot": np.fromfile(COMMON / "Ctot_init.raw", np.float64),
        "phi": np.fromfile(COMMON / "phi_init.raw", np.float64),
        "xB": np.fromfile(COMMON / "xB_init.raw", np.float64),
    }
    reference_root = find_case("G12", "dt32")
    if not (reference_root / "workstation_status.json").is_file():
        raise SystemExit("long strict dt32 reference is not complete")
    ref_status, ref_summary = read_case("G12", "dt32")
    if not ref_status.get("completed_long_window"):
        raise SystemExit("long strict dt32 reference failed")
    if not continuation_hard_pass(ref_summary, 4.6875):
        raise SystemExit("long strict dt32 reference failed unchanged hard gates")

    common = {
        (row["gate_id"], row["dt_id"]): row
        for row in common_rows()
        if row.get("full_common_window_pass", "").lower() == "true"
    }
    rows: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    for pair, common_row in common.items():
        gate_id, dt_id = pair
        root = find_case(gate_id, dt_id)
        status_path = root / "workstation_status.json"
        if not status_path.is_file():
            aggregate.append({"gate_id": gate_id, "dt_id": dt_id, "long_window_status": "PENDING"})
            continue
        status, summary = read_case(gate_id, dt_id)
        if not status.get("completed_long_window"):
            aggregate.append({"gate_id": gate_id, "dt_id": dt_id, "long_window_status": "RUNTIME_FAIL"})
            continue
        all_windows_pass = True
        final_metrics: dict[str, Any] = {}
        for window in range(1, 5):
            candidate_step = (
                COMMON_STEPS[dt_id] if window == 1
                else COMMON_STEPS[dt_id] * (window - 1)
            )
            reference_step = (
                COMMON_STEPS["dt32"] if window == 1
                else COMMON_STEPS["dt32"] * (window - 1)
            )
            candidate = state_at_window(gate_id, dt_id, window)
            reference = state_at_window("G12", "dt32", window)
            metrics = accuracy_metrics(candidate, initial, reference)
            window_pass = bool(
                metrics["cumulative_transfer_relative_error"] <= 0.03
                and metrics["hvolume_increment_relative_error"] <= 0.03
                and metrics["phi_half_interface_error_dx"] <= 0.5
                and metrics["matrix_profile_capacity_weighted_relative_error"] <= 0.05
                and metrics["far_field_relative_error"] <= 0.02
                and metrics["interface_direction_same"]
            )
            all_windows_pass &= window_pass
            row = {
                "gate_id": gate_id,
                "dt_id": dt_id,
                "window_index": window,
                "time_code": 6.25 * window / 4.0,
                "time_physical_s": 6.25 * window / 4.0 * T_REAL_UNIT_S,
                "candidate_step": candidate_step,
                "reference_step": reference_step,
                **metrics,
                "registered_long_accuracy_pass": window_pass,
            }
            rows.append(row)
            if window == 4:
                final_metrics = metrics

        bounded = summary.get("bounded_retry_summary", {})
        trajectory = summary.get("trajectory_meta", {})
        common_wall = as_float(common_row.get("wall_seconds"))
        continuation_wall = as_float(status.get("remote_wall_seconds"))
        wall = common_wall + continuation_wall
        throughput = 6.25 * T_REAL_UNIT_S / wall * 3600.0
        common_macro_steps = as_int(common_row.get("macro_steps"))
        continuation_macro_steps = as_int(bounded.get("macro_steps"))
        total_macro_steps = common_macro_steps + continuation_macro_steps
        combined_retry_fraction = (
            as_int(common_row.get("internal_trial_rejects"))
            + as_int(bounded.get("internal_trial_rejects"))
        ) / max(total_macro_steps, 1)
        combined_fallback_fraction = (
            as_int(common_row.get("fallback_macros"))
            + as_int(bounded.get("fallback_macros"))
        ) / max(total_macro_steps, 1)
        combined_retry_wall_fraction = (
            as_float(common_row.get("retry_wall_overhead_fraction"), 0.0)
            * common_wall
            + as_float(bounded.get("reject_trial_wall_fraction"), 0.0)
            * continuation_wall
        ) / max(wall, 1.0e-300)
        combined_accepted_iteration_p99 = max(
            as_float(common_row.get("accepted_iteration_p99")),
            as_float(bounded.get("accepted_iteration_p99")),
        )
        common_max_d = as_float(common_row.get("maximum_abs_local_cumulative_defect"))
        continuation_max_d = as_float(summary.get("maximum_abs_local_cumulative_defect"))
        common_defect = raw_array(
            common_case(gate_id, dt_id),
            "ctot_transport_gate_cumulative_defect.raw",
        )
        continuation_defect = raw_array(
            root, "ctot_transport_gate_cumulative_defect.raw"
        )
        total_defect = common_defect + continuation_defect
        long_max_d = float(np.max(np.abs(total_defect)))
        normalized_growth = (
            (continuation_max_d / 3.0) / max(common_max_d, 1.0e-300)
        )
        signed_common = as_float(common_row.get("signed_local_cumulative_defect_sum"))
        signed_long = float(np.sum(total_defect))
        common_er_l1 = as_float(common_row.get("ER_L1"))
        continuation_er_l1 = as_float(trajectory.get("cumulative_ER_L1"))
        long_er_l1 = common_er_l1 + continuation_er_l1
        long_c_increment = (
            as_float(common_row.get("cumulative_C_increment_L1"))
            + as_float(trajectory.get("cumulative_C_increment_L1"))
        )
        common_signed_fraction = abs(signed_common) / max(common_er_l1, 1.0e-300)
        long_signed_fraction = abs(signed_long) / max(long_er_l1, 1.0e-300)
        signed_fraction_limit = max(4.0 * common_signed_fraction, 1.0e-3)
        local_defect_rate_growth_pass = normalized_growth <= 2.0
        signed_residual_bias_growth_pass = (
            long_signed_fraction <= signed_fraction_limit
        )
        no_bias_growth = (
            local_defect_rate_growth_pass
            and signed_residual_bias_growth_pass
        )
        hard_pass = bool(
            continuation_hard_pass(summary, 4.6875)
            and common_row.get("hard_numerical_gates_pass", "").lower()
            == "true"
        )
        combined_failed_counts = failed_cell_attempt_counts(
            common_case(gate_id, dt_id)
        )
        combined_failed_counts.update({
            int(idx): int(count)
            for idx, count in summary.get(
                "failed_cell_attempt_counts", {}
            ).items()
        })
        long_persistent_failed_cells = sorted(
            idx for idx, count in combined_failed_counts.items()
            if count >= 3
        )
        continuation_fallbacks = [
            int(value) for value in summary.get("event_fallback_steps", [])
        ]
        boundary_fallback = boundary_consecutive_fallback(
            COMMON_STEPS[dt_id],
            event_fallback_steps(common_case(gate_id, dt_id)),
            continuation_fallbacks,
        )
        combined_max_consecutive_fallback = max(
            as_int(common_row.get("max_consecutive_fallback_macros")),
            as_int(bounded.get("max_consecutive_fallback_macros")),
            boundary_fallback,
        )
        long_retry_pass = retry_efficiency_pass(
            summary, long_persistent_failed_cells,
            combined_max_consecutive_fallback,
        ) and bool(
            combined_retry_fraction <= 0.01
            and combined_fallback_fraction <= 0.01
            and combined_retry_wall_fraction <= 0.05
            and combined_accepted_iteration_p99
            < 0.8 * as_int(bounded.get("nonlinear_iteration_budget"))
        )
        long_pass = (
            hard_pass and long_retry_pass and all_windows_pass
            and no_bias_growth
        )
        failure_reasons = []
        if not hard_pass:
            failure_reasons.append("HARD_NUMERICAL_GATE")
        if not long_retry_pass:
            failure_reasons.append("RETRY_EFFICIENCY_GATE")
        if not all_windows_pass:
            failure_reasons.append("FOUR_WINDOW_ACCURACY_GATE")
        if not local_defect_rate_growth_pass:
            failure_reasons.append("LOCAL_DEFECT_RATE_GROWTH_GATE")
        if not signed_residual_bias_growth_pass:
            failure_reasons.append("SIGNED_RESIDUAL_BIAS_GROWTH_GATE")
        aggregate.append({
            "gate_id": gate_id,
            "dt_id": dt_id,
            "requested_transport_gate": common_row["requested_transport_gate"],
            "dt_code": common_row["dt_code"],
            "long_time_physical_s": 6.25 * T_REAL_UNIT_S,
            "wall_seconds": wall,
            "physical_s_per_GPU_hour": throughput,
            "long_hard_gates_pass": hard_pass,
            "long_retry_efficiency_gate_pass": long_retry_pass,
            "all_registered_windows_accuracy_pass": all_windows_pass,
            "residual_bias_growth_pass": no_bias_growth,
            "local_defect_rate_growth_pass": local_defect_rate_growth_pass,
            "signed_residual_bias_growth_pass": signed_residual_bias_growth_pass,
            "local_defect_rate_growth_vs_common": normalized_growth,
            "common_signed_defect_fraction": common_signed_fraction,
            "long_signed_defect_fraction": long_signed_fraction,
            "signed_defect_fraction_limit": signed_fraction_limit,
            "eta_R": long_er_l1 / max(long_c_increment, 1.0e-300),
            "maximum_abs_local_cumulative_defect": long_max_d,
            "continuation_max_abs_local_cumulative_defect": continuation_max_d,
            "continuation_observed_time_code": trajectory.get("observed_time_code"),
            "continuation_time_closure_error": abs(
                as_float(trajectory.get("observed_time_code")) - 4.6875
            ),
            "continuation_time_roundoff_bound": max(
                1.0e-12,
                2.0 * np.finfo(np.float64).eps
                * max(as_int(trajectory.get("accepted_rows")), 1)
                * 4.6875,
            ),
            "continuation_trajectory_schema": trajectory.get("schema"),
            "combined_ER_L1": long_er_l1,
            "combined_C_increment_L1": long_c_increment,
            "retry_fraction": combined_retry_fraction,
            "fallback_fraction": combined_fallback_fraction,
            "retry_wall_overhead_fraction": combined_retry_wall_fraction,
            "max_consecutive_fallback_macros": combined_max_consecutive_fallback,
            "boundary_consecutive_fallback_macros": boundary_fallback,
            "max_accepted_subcycle_depth": bounded.get("max_accepted_subcycle_depth"),
            "accepted_iteration_p99": combined_accepted_iteration_p99,
            "persistent_failed_cells": ";".join(
                str(value) for value in long_persistent_failed_cells
            ),
            "final_Ctot_increment_relative_L2_error": final_metrics.get("Ctot_increment_relative_L2_error"),
            "final_phi_increment_relative_L2_error": final_metrics.get("phi_increment_relative_L2_error"),
            "final_hvolume_increment_relative_error": final_metrics.get("hvolume_increment_relative_error"),
            "final_matrix_profile_error": final_metrics.get("matrix_profile_capacity_weighted_relative_error"),
            "final_interface_error_dx": final_metrics.get("phi_half_interface_error_dx"),
            "final_transfer_error": final_metrics.get("cumulative_transfer_relative_error"),
            "final_far_field_error": final_metrics.get("far_field_relative_error"),
            "failure_reasons": ";".join(failure_reasons) if failure_reasons else "NONE",
            "long_window_status": "PASS" if long_pass else "FAIL",
        })

    write_csv(REPORT / "long_window_metrics.csv", rows)
    write_csv(REPORT / "long_window_candidate_summary.csv", aggregate)
    passed = [row for row in aggregate if row.get("long_window_status") == "PASS"]
    best = max(passed, key=lambda row: float(row["physical_s_per_GPU_hour"])) if passed else None
    lines = [
        "# Long-window transport residual-gate validation", "",
        "The strict `G12+dt/32` reference and every common-window candidate are evaluated at pre-registered 1x, 2x, 3x, and 4x windows. The accepted 1x checkpoint and BDF2 history are reused byte-for-byte; each long job continues for three further windows. Wall time and trajectory defect are combined across the first segment and continuation. The final window is 6.25 code time = 257.059909 s.", "",
        "## Strict Reference", "",
        f"The `G12+dt/32` continuation passed unchanged hard gates with max mass error `{as_float(ref_summary.get('max_mass_error')):.3e}`, transactional observed time `{as_float(ref_summary.get('trajectory_meta', {}).get('observed_time_code')):.17e}`, and `{len(ref_summary.get('event_fallback_steps', []))}` event-safe fallback macros.", "",
        "## Candidates", "",
        "| Gate | dt | Hard | Retry/efficiency | Four-window accuracy | Local defect rate | Signed bias | Throughput (physical s/GPU h) | Status | Reason |",
        "|---|---|---|---|---|---|---|---:|---|---|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['gate_id']} | {row['dt_id']} | {row.get('long_hard_gates_pass', '-')} | "
            f"{row.get('long_retry_efficiency_gate_pass', '-')} | "
            f"{row.get('all_registered_windows_accuracy_pass', '-')} | "
            f"{row.get('local_defect_rate_growth_pass', '-')} | "
            f"{row.get('signed_residual_bias_growth_pass', '-')} | "
            f"{as_float(row.get('physical_s_per_GPU_hour')):.1f} | {row['long_window_status']} | "
            f"{row.get('failure_reasons', '-')} |"
        )
    lines += ["", f"Best fully passed long-window candidate: `{best['gate_id'] + '+' + best['dt_id'] if best else 'NONE'}`.", ""]
    (REPORT / "long_window_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"long_candidates": len(aggregate), "passed": len(passed), "best": best}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
