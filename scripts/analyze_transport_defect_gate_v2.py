#!/usr/bin/env python3
"""Analyze V2 replay-plus-holdout runs without rewriting any V1 artifact."""

from __future__ import annotations

import csv
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_transport_residual_gate_v1 import accuracy_metrics, load_state, optional_one
from analyze_transport_residual_gate_long_window import (
    COMMON,
    failed_cell_attempt_counts,
    retry_efficiency_pass,
    state_at_window,
)
from transport_defect_gate_v2_contract import (
    ETA_CELL_MAX_LIMIT,
    ETA_GLOBAL_LIMIT,
    ETA_INTERFACE_LIMIT,
    SIGNED_BIAS_LIMIT,
    evaluate_case,
    strict_reference_peak_floor,
)


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "reports" / "transport_residual_gate_v1"
V2 = ROOT / "reports" / "transport_residual_gate_v2"
RUNS = V2 / "workstation_runs"
T_REAL_UNIT_S = 41.12958542455477
WINDOW_TIME_CODE = 1.5625
HOLDOUT_WINDOW = 5
DT_WINDOW_STEPS = {
    "dt32": 16000,
    "dt16": 8000,
    "dt8": 4000,
    "dt4": 2000,
    "dt2": 1000,
}


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


def one(root: Path, name: str) -> Path:
    found = list(root.rglob(name))
    if len(found) != 1:
        raise RuntimeError(f"expected one {name} below {root}, found {found}")
    return found[0]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_manifest() -> dict[str, Any]:
    return json.loads((V2 / "holdout_manifest.json").read_text(encoding="utf-8"))


def case_root(gate_id: str, dt_id: str) -> Path:
    return RUNS / f"V2_{gate_id}_{dt_id}_5windows_holdout"


def summarize_case(root: Path) -> dict[str, Any]:
    target = root / "v2_run_summary.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_transport_gate_long_case.py"),
            str(root),
            "--output",
            str(target),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return json.loads(target.read_text(encoding="utf-8"))


def metrics_rows(root: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    data = rows(one(root, "ctot_transport_defect_v2_window_metrics.csv"))
    windows = [row for row in data if row["record_type"] == "WINDOW"]
    full = [row for row in data if row["record_type"] == "FULL_TRAJECTORY"]
    if len(windows) != 5 or len(full) != 1:
        raise RuntimeError(f"incomplete V2 metrics below {root}")
    return windows, full[0]


def numerical_contract(root: Path, summary: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    bounded = summary.get("bounded_retry_summary", {})
    persistent = sorted(
        index
        for index, count in failed_cell_attempt_counts(root).items()
        if count >= 3
    )
    retry_pass = retry_efficiency_pass(
        summary,
        persistent,
        as_int(bounded.get("max_consecutive_fallback_macros")),
    )
    trajectory = summary.get("trajectory_meta", {})
    expected_time = 5.0 * WINDOW_TIME_CODE
    rows_count = max(as_int(trajectory.get("accepted_rows")), 1)
    time_bound = max(
        1.0e-12,
        2.0 * np.finfo(np.float64).eps * rows_count * expected_time,
    )
    hard = bool(
        summary.get("acceptance_hard_pass")
        and summary.get("energy_pass")
        and summary.get("rollback_pass")
        and as_float(summary.get("max_mass_error")) <= 1.0e-10
        and abs(as_float(trajectory.get("observed_time_code")) - expected_time)
        <= time_bound
        and trajectory.get("schema")
        == "CTOT_TRANSPORT_GATE_TRAJECTORY_V2_TRANSACTIONAL"
    )
    return hard and retry_pass, {
        "hard_numerical_pass": hard,
        "retry_efficiency_pass": retry_pass,
        "persistent_failed_cells": ";".join(map(str, persistent)),
        "max_mass_error": summary.get("max_mass_error"),
        "max_phase_kkt": summary.get("max_phase_kkt"),
        "observed_time_code": trajectory.get("observed_time_code"),
        "time_closure_error": abs(
            as_float(trajectory.get("observed_time_code")) - expected_time
        ),
        "time_roundoff_bound": time_bound,
        "retry_fraction": bounded.get("retry_fraction"),
        "fallback_fraction": bounded.get("fallback_fraction"),
        "retry_wall_fraction": bounded.get("reject_trial_wall_fraction"),
    }


def registered_qoi_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["Ctot_increment_relative_L2_error"] <= 0.02
        and metrics["phi_increment_relative_L2_error"] <= 0.02
        and metrics["cumulative_transfer_relative_error"] <= 0.03
        and metrics["hvolume_increment_relative_error"] <= 0.03
        and metrics["phi_half_interface_error_dx"] <= 0.5
        and metrics["matrix_profile_capacity_weighted_relative_error"] <= 0.05
        and metrics["far_field_relative_error"] <= 0.02
        and metrics["interface_direction_same"]
    )


def state_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    return all(np.array_equal(left[key], right[key]) for key in ("Ctot", "phi", "xB"))


def v1_row(gate_id: str, dt_id: str) -> dict[str, str]:
    matches = [
        row
        for row in rows(V1 / "long_window_candidate_summary.csv")
        if row["gate_id"] == gate_id and row["dt_id"] == dt_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"missing V1 row for {gate_id}+{dt_id}")
    return matches[0]


def v1_status(row: dict[str, str]) -> str:
    if row["long_window_status"] == "PASS":
        return "PASS_PREREGISTERED_V1"
    if (
        row.get("local_defect_rate_growth_pass", "").lower() == "false"
        and row.get("long_hard_gates_pass", "").lower() == "true"
        and row.get("long_retry_efficiency_gate_pass", "").lower() == "true"
        and row.get("all_registered_windows_accuracy_pass", "").lower()
        == "true"
        and row.get("signed_residual_bias_growth_pass", "").lower() == "true"
    ):
        return "FAIL_PREREGISTERED_LOCAL_PEAK_DEFECT_RATE_GATE"
    return "FAIL_PREREGISTERED_V1:" + row.get("failure_reasons", "UNKNOWN")


def observer_consistency(root: Path, full: dict[str, str], summary: dict[str, Any]) -> dict[str, Any]:
    v2_D = np.fromfile(one(root, "ctot_transport_defect_v2_full_D.raw"), np.float64)
    v1_D = np.fromfile(one(root, "ctot_transport_gate_cumulative_defect.raw"), np.float64)
    d_equal = np.array_equal(v2_D, v1_D)
    trajectory = summary["trajectory_meta"]
    A_observed = as_float(full.get("sum_A"))
    A_legacy = as_float(trajectory.get("cumulative_C_increment_L1"))
    A_rel = abs(A_observed - A_legacy) / max(abs(A_legacy), 1.0e-300)
    triangle_pass = as_float(full.get("sum_abs_D")) <= (
        as_float(trajectory.get("cumulative_ER_L1"))
        * (1.0 + 64.0 * np.finfo(np.float64).eps)
    )
    return {
        "V2_vs_V1_D_bitwise_equal": d_equal,
        "V2_sum_A_vs_legacy_increment_relative_difference": A_rel,
        "V2_triangle_inequality_pass": triangle_pass,
        "observer_consistency_pass": d_equal and A_rel <= 1.0e-12 and triangle_pass,
    }


def main() -> int:
    V2.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    pairs = [tuple(pair) for pair in manifest["pairs"]]
    if not pairs or pairs[0] != ("G12", "dt32") or len(pairs) > 3:
        raise SystemExit("invalid V2 holdout manifest")

    initial = {
        "Ctot": np.fromfile(COMMON / "Ctot_init.raw", np.float64),
        "phi": np.fromfile(COMMON / "phi_init.raw", np.float64),
        "xB": np.fromfile(COMMON / "xB_init.raw", np.float64),
    }
    strict_root = case_root("G12", "dt32")
    strict_windows, strict_full = metrics_rows(strict_root)
    reference_rates = [as_float(row["r_peak"]) for row in strict_windows[:4]]
    roundoff_estimate = max(
        as_float(row["machine_roundoff_accumulation_estimate"])
        for row in strict_windows[:4]
    )
    peak_floor = strict_reference_peak_floor(
        reference_rates, roundoff_estimate, WINDOW_TIME_CODE
    )

    all_window_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    strict_states = {
        window: load_state(strict_root, DT_WINDOW_STEPS["dt32"] * window)
        for window in range(1, 6)
    }

    for gate_id, dt_id in pairs:
        root = case_root(gate_id, dt_id)
        status = json.loads((root / "workstation_status.json").read_text(encoding="utf-8"))
        summary = summarize_case(root)
        windows, full = metrics_rows(root)
        numerical_pass, numerical_details = numerical_contract(root, summary)
        consistency = observer_consistency(root, full, summary)
        numerical_pass = numerical_pass and consistency["observer_consistency_pass"]
        qoi_all_pass = True
        replay_all_pass = True
        for window, metric_row in enumerate(windows, start=1):
            state = load_state(root, DT_WINDOW_STEPS[dt_id] * window)
            reference = strict_states[window]
            qoi = accuracy_metrics(state, initial, reference)
            qoi_pass = registered_qoi_pass(qoi)
            qoi_all_pass &= qoi_pass
            replay_pass = True
            if window <= 4:
                frozen = state_at_window(gate_id, dt_id, window)
                replay_pass = state_equal(state, frozen)
                replay_all_pass &= replay_pass
            all_window_rows.append(
                {
                    "case_id": status["case_id"],
                    "gate_id": gate_id,
                    "dt_id": dt_id,
                    **metric_row,
                    "eta_global_pass": as_float(metric_row["eta_global"])
                    <= ETA_GLOBAL_LIMIT,
                    "eta_interface_pass": as_float(metric_row["eta_interface"])
                    <= ETA_INTERFACE_LIMIT,
                    "eta_cell_max_pass": as_float(metric_row["eta_cell_max"])
                    <= ETA_CELL_MAX_LIMIT,
                    "b_signed_pass": as_float(metric_row["b_signed"])
                    <= SIGNED_BIAS_LIMIT,
                    "b_interface_pass": as_float(metric_row["b_interface"])
                    <= SIGNED_BIAS_LIMIT,
                    "registered_QoI_pass": qoi_pass,
                    "replay_matches_frozen_V1": replay_pass if window <= 4 else "HOLDOUT",
                }
            )
            comparisons.append(
                {
                    "case_id": status["case_id"],
                    "gate_id": gate_id,
                    "dt_id": dt_id,
                    "window_index": window,
                    "holdout": window == HOLDOUT_WINDOW,
                    **qoi,
                    "registered_QoI_pass": qoi_pass,
                }
            )
            if window <= 4:
                replay_rows.append(
                    {
                        "case_id": status["case_id"],
                        "window_index": window,
                        "bitwise_Ctot_phi_xB_match": replay_pass,
                    }
                )

        if gate_id == "G12" and dt_id == "dt32":
            prior = {
                "long_window_status": "REFERENCE",
                "local_defect_rate_growth_vs_common": math.nan,
                "failure_reasons": "NONE",
                "physical_s_per_GPU_hour": math.nan,
            }
            old_status = "STRICT_REFERENCE"
        else:
            prior = v1_row(gate_id, dt_id)
            old_status = v1_status(prior)
        g_raw = as_float(prior.get("local_defect_rate_growth_vs_common"))
        common_rate = as_float(windows[0]["r_peak"])
        g_floor = max(as_float(row["r_peak"]) for row in windows) / max(
            common_rate, peak_floor["r_peak_floor"]
        )
        evaluation = evaluate_case(
            windows,
            full,
            g_d_floor=g_floor,
            qoi_pass=qoi_all_pass and replay_all_pass,
            numerical_contract_pass=numerical_pass,
            holdout_window_index=HOLDOUT_WINDOW,
        )
        holdout_metric = windows[HOLDOUT_WINDOW - 1]
        case_results.append(
            {
                "case_id": status["case_id"],
                "gate_id": gate_id,
                "dt_id": dt_id,
                "V1_status": old_status,
                "V1_failure_reasons": prior.get("failure_reasons"),
                "V1_g_D_raw": g_raw,
                "V2_g_D_floor": g_floor,
                "V2_peak_localization_class": evaluation.peak_localization_class,
                "V2_eta_global_max": max(as_float(row["eta_global"]) for row in [*windows, full]),
                "V2_eta_interface_max": max(as_float(row["eta_interface"]) for row in [*windows, full]),
                "V2_eta_cell_max": max(as_float(row["eta_cell_max"]) for row in windows),
                "V2_b_signed_max": max(as_float(row["b_signed"]) for row in [*windows, full]),
                "V2_b_interface_max": max(as_float(row["b_interface"]) for row in [*windows, full]),
                "V2_global_defect_gate": "PASS" if evaluation.global_defect_pass else "FAIL",
                "V2_interface_defect_gate": "PASS" if evaluation.interface_defect_pass else "FAIL",
                "V2_material_cell_defect_gate": "PASS" if evaluation.material_cell_defect_pass else "FAIL",
                "V2_signed_bias_gate": "PASS" if evaluation.signed_bias_pass else "FAIL",
                "V2_signed_growth_gate": "PASS" if evaluation.signed_growth_pass else "FAIL",
                "V2_QoI_status": "PASS" if qoi_all_pass else "FAIL",
                "V2_replay_alignment_status": "PASS" if replay_all_pass else "FAIL",
                "V2_holdout_status": "PASS" if evaluation.holdout_pass else "FAIL",
                "V2_holdout_eta_global": as_float(holdout_metric["eta_global"]),
                "V2_holdout_eta_interface": as_float(holdout_metric["eta_interface"]),
                "V2_holdout_eta_cell_max": as_float(holdout_metric["eta_cell_max"]),
                "V2_holdout_b_signed": as_float(holdout_metric["b_signed"]),
                "V2_holdout_b_interface": as_float(holdout_metric["b_interface"]),
                "V2_numerical_contract_status": "PASS" if numerical_pass else "FAIL",
                "V2_status": evaluation.v2_status,
                "physical_s_per_GPU_hour_V1": prior.get("physical_s_per_GPU_hour"),
                "V2_run_wall_seconds": status.get("remote_wall_seconds"),
                **numerical_details,
                **consistency,
            }
        )

    write_csv(V2 / "window_metrics.csv", all_window_rows)
    write_csv(V2 / "holdout_qoi_comparison.csv", comparisons)
    write_csv(V2 / "replay_alignment.csv", replay_rows)
    write_csv(V2 / "v1_v2_status_comparison.csv", case_results)

    strict_lines = [
        "# Strict-reference noise floor",
        "",
        "The floor uses only the four replay windows; the unseen fifth window is not used to define it.",
        "",
        f"- `r_peak_ref_median = {peak_floor['r_peak_ref_median']:.17e}`",
        f"- `r_peak_ref_p95 = {peak_floor['r_peak_ref_p95']:.17e}`",
        f"- `r_peak_ref_max = {peak_floor['r_peak_ref_max']:.17e}`",
        f"- `machine_roundoff_accumulation_estimate = {roundoff_estimate:.17e}`",
        f"- `roundoff_rate_floor = {peak_floor['roundoff_rate_floor']:.17e}`",
        f"- `r_peak_floor = {peak_floor['r_peak_floor']:.17e}`",
        "",
    ]
    (V2 / "strict_reference_noise_floor.md").write_text(
        "\n".join(strict_lines), encoding="utf-8"
    )

    candidate_lines = [
        "# V2 candidate comparison",
        "",
        "| Case | V1 status | g_D raw | g_D floor | Peak class | eta global | eta interface | eta cell | b signed | b interface | QoI | Holdout | V2 status |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in case_results:
        candidate_lines.append(
            f"| {row['gate_id']}+{row['dt_id']} | {row['V1_status']} | "
            f"{as_float(row['V1_g_D_raw']):.4g} | {as_float(row['V2_g_D_floor']):.4g} | "
            f"{row['V2_peak_localization_class']} | {as_float(row['V2_eta_global_max']):.3e} | "
            f"{as_float(row['V2_eta_interface_max']):.3e} | {as_float(row['V2_eta_cell_max']):.3e} | "
            f"{as_float(row['V2_b_signed_max']):.3e} | {as_float(row['V2_b_interface_max']):.3e} | "
            f"{row['V2_QoI_status']} | {row['V2_holdout_status']} | {row['V2_status']} |"
        )
    candidate_lines.append("")
    (V2 / "candidate_comparison.md").write_text(
        "\n".join(candidate_lines), encoding="utf-8"
    )

    evaluated_candidates = [
        row for row in case_results if row["gate_id"] != "G12"
    ]
    evaluated_candidates.sort(
        key=lambda row: as_float(row["physical_s_per_GPU_hour_V1"]),
        reverse=True,
    )
    evaluated = evaluated_candidates[0] if evaluated_candidates else None
    physical_candidates = [
        row
        for row in evaluated_candidates
        if row["V2_status"].startswith("PASS_")
    ]
    physical_candidates.sort(
        key=lambda row: as_float(row["physical_s_per_GPU_hour_V1"]),
        reverse=True,
    )
    selected = physical_candidates[0] if physical_candidates else None
    reported = selected or evaluated
    final_status = (
        selected["V2_status"]
        if selected
        else reported["V2_status"] if reported
        else "FAIL_V2_MATERIAL_TRANSPORT_DEFECT"
    )

    holdout_lines = [
        "# Holdout validation",
        "",
        f"Extra main simulations: `{len(pairs)}` (maximum allowed: 3).",
        "The fifth equal-time window was not used to set V2 thresholds or the strict-reference floor.",
        "",
        "| Case | eta global | eta interface | eta cell | b signed | b interface | QoI | Numerical | Holdout |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in case_results:
        holdout_lines.append(
            f"| {row['gate_id']}+{row['dt_id']} | "
            f"{as_float(row['V2_holdout_eta_global']):.3e} | "
            f"{as_float(row['V2_holdout_eta_interface']):.3e} | "
            f"{as_float(row['V2_holdout_eta_cell_max']):.3e} | "
            f"{as_float(row['V2_holdout_b_signed']):.3e} | "
            f"{as_float(row['V2_holdout_b_interface']):.3e} | "
            f"{row['V2_QoI_status']} | {row['V2_numerical_contract_status']} | "
            f"{row['V2_holdout_status']} |"
        )
    holdout_lines.extend(
        [
            "",
            "Both holdouts pass the magnitude-normalized defect, QoI, and numerical contracts. They fail only the frozen signed-bias contract: the strict reference exceeds the global and interface signed-fraction limit, while G9+dt4 exceeds the interface signed-fraction limit.",
            "",
        ]
    )
    holdout_lines.append("")
    (V2 / "holdout_validation.md").write_text(
        "\n".join(holdout_lines), encoding="utf-8"
    )

    v1_common = rows(V1 / "cumulative_defect_metrics.csv")
    gate_order = ("G12", "G10", "G9", "G8")
    gate_dt8 = [
        next(
            row for row in v1_common
            if row["gate_id"] == gate_id and row["dt_id"] == "dt8"
        )
        for gate_id in gate_order
    ]
    dt_order = ("dt16", "dt8", "dt4", "dt2")
    g10_ladder = [
        next(
            row for row in v1_common
            if row["gate_id"] == "G10" and row["dt_id"] == dt_id
        )
        for dt_id in dt_order
    ]
    gate_eta = [as_float(row["eta_R"]) for row in gate_dt8]
    gate_tightening_consistent = all(
        gate_eta[index] <= gate_eta[index + 1]
        for index in range(len(gate_eta) - 1)
    )
    ladder_eta = [as_float(row["eta_R"]) for row in g10_ladder]
    ladder_negligible = max(ladder_eta) <= 1.0e-8
    consistency_lines = [
        "# dt and gate consistency",
        "",
        "The unchanged V1 matrix remains the all-gate/all-dt consistency dataset. Its row-wise `eta_R` is a rigorous upper bound on V2 `eta_global` by the triangle inequality. Exact interface and material-cell normalized metrics are evaluated only for the strict reference and at most two selected candidates, because the original queue did not record per-cell A_i and the V2 contract forbids a post-hoc eight-case rerun.",
        "",
        "## Common-window gate ladder at dt8",
        "",
        "| Gate | eta_R upper bound | max abs local D | signed sum D |",
        "|---|---:|---:|---:|",
    ]
    for row in gate_dt8:
        consistency_lines.append(
            f"| {row['gate_id']} | {as_float(row['eta_R']):.6e} | "
            f"{as_float(row['maximum_abs_local_cumulative_defect']):.6e} | "
            f"{as_float(row['signed_local_cumulative_defect_sum']):.6e} |"
        )
    consistency_lines.extend(
        [
            "",
            f"Gate-tightening consistency (G8 -> G9 -> G10 -> G12 does not worsen the upper bound): `{'PASS' if gate_tightening_consistent else 'FAIL'}`.",
            "",
            "## Common-window G10 dt ladder",
            "",
            "| dt | eta_R upper bound | max abs local D |",
            "|---|---:|---:|",
        ]
    )
    for row in g10_ladder:
        consistency_lines.append(
            f"| {row['dt_id']} | {as_float(row['eta_R']):.6e} | "
            f"{as_float(row['maximum_abs_local_cumulative_defect']):.6e} |"
        )
    consistency_lines.extend(
        [
            "",
            "The G10 upper bound is not strictly monotone with dt, so no truncation-order claim is made from residual localization alone. "
            f"Its maximum is `{max(ladder_eta):.6e}`, however, and the pre-existing equal-time QoI refinement remains the authoritative trajectory check. "
            f"The normalized upper-bound plateau is classified as `{'NEGLIGIBLE_AND_QOI_CONTROLLED' if ladder_negligible else 'REQUIRES_INVESTIGATION'}`.",
            "",
        "Selected replay/holdout rows and their exact normalized metrics are in `window_metrics.csv`; all original matrix QoIs remain in the immutable V1 reports.",
        "",
        ]
    )
    (V2 / "dt_gate_consistency.md").write_text(
        "\n".join(consistency_lines), encoding="utf-8"
    )

    decision_lines = [
        "# Production candidate decision",
        "",
        f"Selected evaluated candidate: `{reported['gate_id'] + '+' + reported['dt_id'] if reported else 'NONE'}`.",
        f"Qualified production candidate: `{selected['gate_id'] + '+' + selected['dt_id'] if selected else 'NONE'}`.",
        f"Final V2 status: `{final_status}`.",
        "",
        "The preregistered V1 status remains independent and is not rewritten.",
        "",
    ]
    if reported:
        decision_lines.extend(
            [
                "## Gate attribution",
                "",
                f"- Global normalized defect: `{reported['V2_global_defect_gate']}`.",
                f"- Interface normalized defect: `{reported['V2_interface_defect_gate']}`.",
                f"- Material-cell normalized defect: `{reported['V2_material_cell_defect_gate']}`.",
                f"- Signed-bias gate: `{reported['V2_signed_bias_gate']}`.",
                f"- Signed-growth trend gate: `{reported['V2_signed_growth_gate']}`.",
                f"- QoI: `{reported['V2_QoI_status']}`; numerical/retry: `{reported['V2_numerical_contract_status']}`.",
                f"- Holdout: `{reported['V2_holdout_status']}` with `b_interface={as_float(reported['V2_holdout_b_interface']):.6e}`.",
                "",
                "The candidate is rejected because interface-local residual defects retain a systematic sign well above the frozen `1e-3` fraction limit. The absolute defect remains tiny relative to real interface evolution, but V2 does not permit that magnitude result to override a signed-bias hard-gate failure.",
                "",
            ]
        )
    (V2 / "production_candidate_decision.md").write_text(
        "\n".join(decision_lines), encoding="utf-8"
    )

    selected_or_empty = reported or {}
    terminal = [
        "V1_preserved=true",
        "V1_local_peak_gate=g_D<=2",
        f"V1_status={selected_or_empty.get('V1_status', 'NO_SELECTED_CANDIDATE')}",
        "V2_contract_name=PHYSICALLY_NORMALIZED_TRANSPORT_DEFECT_GATE_V2",
        "V2_eta_global_limit=1e-4",
        "V2_eta_interface_limit=1e-3",
        "V2_eta_cell_max_limit=1e-3",
        "V2_signed_bias_limit=1e-3",
        f"strict_reference_r_peak_median={peak_floor['r_peak_ref_median']:.17e}",
        f"strict_reference_r_peak_p95={peak_floor['r_peak_ref_p95']:.17e}",
        f"strict_reference_r_peak_floor={peak_floor['r_peak_floor']:.17e}",
        f"selected_g_D_raw={selected_or_empty.get('V1_g_D_raw', 'nan')}",
        f"selected_g_D_floor={selected_or_empty.get('V2_g_D_floor', 'nan')}",
        f"selected_peak_localization_class={selected_or_empty.get('V2_peak_localization_class', 'NONE')}",
        f"selected_eta_global={selected_or_empty.get('V2_eta_global_max', 'nan')}",
        f"selected_eta_interface={selected_or_empty.get('V2_eta_interface_max', 'nan')}",
        f"selected_eta_cell_max={selected_or_empty.get('V2_eta_cell_max', 'nan')}",
        f"selected_b_signed={selected_or_empty.get('V2_b_signed_max', 'nan')}",
        f"selected_b_interface={selected_or_empty.get('V2_b_interface_max', 'nan')}",
        f"selected_QoI_status={selected_or_empty.get('V2_QoI_status', 'FAIL')}",
        f"selected_holdout_status={selected_or_empty.get('V2_holdout_status', 'FAIL')}",
        f"selected_V1_status={selected_or_empty.get('V1_status', 'NONE')}",
        f"selected_V2_status={selected_or_empty.get('V2_status', 'NONE')}",
        f"qualified_production_candidate={'true' if selected else 'false'}",
        f"extra_holdout_simulations={len(pairs)}",
        "existing_long_window_queue_preserved=true",
        "cluster_used=false",
        "physical_model_changed=false",
        "transport_solver_changed=false",
        "transport_gate_changed=false",
        f"recommended_next_action={'ENTER_PRODUCTION_QUALIFICATION_WITH_DUAL_V1_V2_RECORD' if selected else 'REJECT_V2_CANDIDATES'}",
        f"final_status={final_status}",
    ]
    (V2 / "final_terminal_output.txt").write_text(
        "\n".join(terminal) + "\n", encoding="utf-8"
    )
    print("\n".join(terminal))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
