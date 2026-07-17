#!/usr/bin/env python3
"""Analyze the fixed-step transport residual-gate qualification matrix."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "transport_residual_gate_v1"
RUNS = REPORT / "workstation_runs"
COMMON = (
    ROOT / "reports" / "bounded_retry_bdf2_v1" / "workstation_runs" /
    "equal_time_common_input"
)
REFERENCE = (
    ROOT / "reports" / "bounded_retry_bdf2_v1" / "workstation_runs" /
    "fine_dt32_common_state_16000"
)
GATES = {"G12": 1.0e-12, "G10": 1.0e-10, "G9": 1.0e-9, "G8": 1.0e-8}
DTS = {
    "dt16": (1.95312500000000011e-4, 8000),
    "dt8": (3.90625000000000022e-4, 4000),
    "dt4": (7.81250000000000043e-4, 2000),
    "dt2": (1.56250000000000009e-3, 1000),
    "dt1": (3.12500000000000017e-3, 500),
}
T_REAL_UNIT_S = 41.12958542455477


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def one(root: Path, pattern: str) -> Path:
    values = list(root.rglob(pattern))
    if len(values) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, got {values}")
    return values[0]


def optional_one(root: Path, pattern: str) -> Path | None:
    values = list(root.rglob(pattern))
    return values[0] if len(values) == 1 else None


def kv_line(text: str, prefix: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    if not lines:
        return {}
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", lines[-1]))


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return default


def integer(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(row[key])
    except (KeyError, ValueError):
        return default


def right_interface_position(phi: np.ndarray) -> float:
    center = int(np.argmax(phi))
    for offset in range(1, phi.size // 2 + 1):
        left = (center + offset - 1) % phi.size
        right = (center + offset) % phi.size
        a, b = float(phi[left]), float(phi[right])
        if a >= 0.5 and b < 0.5:
            return center + offset - 1 + (a - 0.5) / max(a - b, 1.0e-300)
    return math.nan


def load_state(root: Path, step: int) -> dict[str, np.ndarray]:
    return {
        "Ctot": np.fromfile(one(root, f"*step{step:06d}_Ctot.raw"), np.float64),
        "phi": np.fromfile(one(root, f"*step{step:06d}_phi.raw"), np.float64),
        "xB": np.fromfile(one(root, f"*step{step:06d}_xB_alpha.raw"), np.float64),
    }


def runtime_metrics(case: Path, status: dict[str, Any]) -> dict[str, Any]:
    text = (case / "run.log").read_text(encoding="utf-8", errors="replace")
    summary = kv_line(text, "CTOT_BOUNDED_RETRY_SUMMARY")
    acceptance_path = optional_one(case, "ctot_acceptance_predicate.csv")
    accepted = read_csv(acceptance_path) if acceptance_path else []
    accepted = [row for row in accepted if row.get("accepted") == "1"]
    hard_columns = [
        "transport_nonlinear_converged", "phase_constraint_converged",
        "elasticity_converged", "outer_coupling_converged",
        "local_phase_storage_closed", "global_mass_closed",
        "Ctot_admissible", "q_alpha_admissible", "xB_alpha_admissible",
        "no_nan_inf", "physical_projection_zero", "no_mass_loss_clipping",
        "energy_work_audit_passed", "restart_metadata_consistent",
    ]
    hard_pass = bool(accepted) and all(
        all(row.get(column) == "1" for column in hard_columns)
        for row in accepted
    )
    energy_path = optional_one(case, "ctot_energy_work.csv")
    energy = read_csv(energy_path) if energy_path else []
    energy = [row for row in energy if row.get("accepted") == "1"]
    energy_pass = bool(energy) and all(
        row.get("monotone_pass") == "1" and row.get("balance_pass") == "1"
        and math.isfinite(f(row, "energy_balance_rel"))
        for row in energy
    )
    retry_path = optional_one(case, "ctot_retry_attempts.csv")
    retry = read_csv(retry_path) if retry_path else []
    rejected = [row for row in retry if row.get("accepted") == "0"]
    rollback_pass = not rejected or all(row.get("rollback_bitwise") == "1" for row in rejected)
    split_path = optional_one(case, "ctot_split_step_metrics.csv")
    split = read_csv(split_path) if split_path else []
    accepted_split = [row for row in split if row.get("accepted") == "1"]
    max_mass_error = max((abs(f(row, "mass_error")) for row in accepted_split), default=math.nan)
    max_phase_kkt = max((abs(f(row, "phase_KKT")) for row in accepted_split), default=math.nan)
    transport_solves = sum(integer(row, "transport_solves") for row in split)
    phase_solves = sum(integer(row, "phase_solves") for row in split)
    nonlinear_path = optional_one(case, "ctot_nonlinear_iterations.csv")
    line_search_evaluations = 0
    if nonlinear_path:
        with nonlinear_path.open("rb") as stream:
            line_search_evaluations = max(sum(1 for _ in stream) - 1, 0)

    if not summary:
        return {
            "completed": False,
            "hard_numerical_gates_pass": False,
            "energy_work_pass": energy_pass,
            "rollback_pass": rollback_pass,
            "max_mass_error": max_mass_error,
            "max_phase_KKT": max_phase_kkt,
            "transport_solves": transport_solves,
            "phase_solves": phase_solves,
            "line_search_evaluations": line_search_evaluations,
            "first_runtime_failure": next(
                (line for line in text.splitlines() if line.startswith("[reject]")),
                "NO_SUMMARY_NO_REJECT_LINE",
            ),
        }

    get_i = lambda key: int(summary.get(key, "0"))
    get_f = lambda key: float(summary.get(key, "nan"))
    retry_fraction = get_f("retry_fraction")
    fallback_fraction = get_f("fallback_fraction")
    retry_wall = get_f("reject_trial_wall_fraction")
    max_consecutive = get_i("max_consecutive_fallback_macros")
    max_depth = get_i("max_accepted_subcycle_depth")
    p99 = get_f("accepted_iteration_p99")
    budget = get_i("nonlinear_iteration_budget")
    failed_cell_path = optional_one(case, "ctot_failed_cell_state.csv")
    persistent_cells: list[int] = []
    if failed_cell_path and rejected:
        rows = read_csv(failed_cell_path)
        worst: dict[tuple[int, int], tuple[float, int]] = {}
        for row in rows:
            key = (integer(row, "physical_step"), integer(row, "attempt_id"))
            candidate = (abs(f(row, "residual")), integer(row, "idx", -1))
            if key not in worst or candidate[0] > worst[key][0]:
                worst[key] = candidate
        counts = Counter(value[1] for value in worst.values())
        persistent_cells = sorted(idx for idx, count in counts.items() if count >= 3)
    efficiency_pass = (
        get_i("macro_hard_rejects") == 0
        and retry_fraction <= 0.01
        and fallback_fraction <= 0.01
        and retry_wall <= 0.05
        and max_consecutive <= 2
        and max_depth <= 2
        and not persistent_cells
        and p99 < 0.8 * budget
    )
    wall = float(status.get("remote_wall_seconds") or status.get("client_wall_seconds"))
    physical = float(status["equal_time_physical_s"])
    return {
        "completed": True,
        "macro_steps": get_i("macro_steps"),
        "wall_seconds": wall,
        "physical_s_per_GPU_hour": physical / wall * 3600.0,
        "macro_hard_rejects": get_i("macro_hard_rejects"),
        "internal_trial_rejects": get_i("internal_trial_rejects"),
        "retry_fraction": retry_fraction,
        "fallback_macros": get_i("fallback_macros"),
        "fallback_fraction": fallback_fraction,
        "retry_wall_overhead_fraction": retry_wall,
        "max_consecutive_fallback_macros": max_consecutive,
        "max_accepted_subcycle_depth": max_depth,
        "accepted_iteration_p99": p99,
        "nonlinear_iteration_budget": budget,
        "persistent_interface_cells": ";".join(map(str, persistent_cells)),
        "hard_numerical_gates_pass": hard_pass and energy_pass and rollback_pass,
        "energy_work_pass": energy_pass,
        "rollback_pass": rollback_pass,
        "max_mass_error": max_mass_error,
        "max_phase_KKT": max_phase_kkt,
        "transport_solves": transport_solves,
        "phase_solves": phase_solves,
        "line_search_evaluations": line_search_evaluations,
        "event_BE_solves": text.count("accepted_integrator=EVENT_BE_SUBCYCLE"),
        "history_rebuild_BE_solves": text.count("event_history_rebuild"),
        "retry_efficiency_gate_pass": efficiency_pass,
        "first_runtime_failure": "NONE",
    }


def accuracy_metrics(
    state: dict[str, np.ndarray], initial: dict[str, np.ndarray],
    reference: dict[str, np.ndarray]
) -> dict[str, Any]:
    C, phi, xB = state["Ctot"], state["phi"], state["xB"]
    C0, phi0, x0 = initial["Ctot"], initial["phi"], initial["xB"]
    Cr, phir, xr = reference["Ctot"], reference["phi"], reference["xB"]
    hc, h0, hr = h(phi), h(phi0), h(phir)
    Cdiff, phidiff = C - Cr, phi - phir
    C_increment_scale = rms(Cr - C0)
    phi_increment_scale = rms(phir - phi0)
    h_increment_scale = abs(float(np.sum(hr - h0)))
    alpha_ref = 1.0 - hr
    profile_scale = float(np.sum(alpha_ref * np.abs(xr - x0)))
    pos, pos0, posr = map(right_interface_position, (phi, phi0, phir))
    direction_same = np.sign(pos - pos0) == np.sign(posr - pos0)
    transfer = 0.5 * float(np.sum(np.abs(C - C0)))
    transfer_ref = 0.5 * float(np.sum(np.abs(Cr - C0)))
    far_mask = (h0 < 1.0e-8) & (hr < 1.0e-8)
    far = float(np.mean(xB[far_mask]))
    far_ref = float(np.mean(xr[far_mask]))
    far_error = abs(far - far_ref) / max(abs(far_ref), 1.0e-300)
    result = {
        "Ctot_field_relative_L2_error": rms(Cdiff) / max(rms(Cr), 1.0e-300),
        "Ctot_field_Linf_error": float(np.max(np.abs(Cdiff))),
        "Ctot_increment_relative_L2_error": rms(Cdiff) / max(C_increment_scale, 1.0e-300),
        "phi_field_relative_L2_error": rms(phidiff) / max(rms(phir), 1.0e-300),
        "phi_field_Linf_error": float(np.max(np.abs(phidiff))),
        "phi_increment_relative_L2_error": rms(phidiff) / max(phi_increment_scale, 1.0e-300),
        "hvolume": float(np.sum(hc)),
        "hvolume_reference": float(np.sum(hr)),
        "hvolume_increment_relative_error": abs(float(np.sum(hc - hr))) / max(h_increment_scale, 1.0e-300),
        "equimolar_hvolume_interface_error_dx": 0.5 * abs(float(np.sum(hc - hr))),
        "phi_half_interface_position_dx": pos,
        "phi_half_interface_reference_dx": posr,
        "phi_half_interface_error_dx": abs(pos - posr),
        "interface_direction_same": bool(direction_same),
        "matrix_profile_capacity_weighted_relative_error": float(np.sum(alpha_ref * np.abs(xB - xr))) / max(profile_scale, 1.0e-300),
        "cumulative_transfer": transfer,
        "cumulative_transfer_reference": transfer_ref,
        "cumulative_transfer_relative_error": abs(transfer - transfer_ref) / max(abs(transfer_ref), 1.0e-300),
        "far_field_matrix_composition": far,
        "far_field_reference": far_ref,
        "far_field_relative_error": far_error,
        "final_mass_difference_from_reference_abs": abs(float(np.sum(C) - np.sum(Cr))),
    }
    result["equal_time_accuracy_pass"] = bool(
        result["Ctot_increment_relative_L2_error"] <= 0.02
        and result["phi_increment_relative_L2_error"] <= 0.02
        and result["hvolume_increment_relative_error"] <= 0.02
        and result["matrix_profile_capacity_weighted_relative_error"] <= 0.03
        and result["phi_half_interface_error_dx"] <= 0.25
        and result["cumulative_transfer_relative_error"] <= 0.02
        and result["far_field_relative_error"] <= 0.01
        and direction_same
    )
    return result


def defect_metrics(case: Path, expected_time_code: float) -> dict[str, Any]:
    meta_path = optional_one(case, "ctot_transport_gate_trajectory_meta.json")
    csv_path = optional_one(case, "ctot_transport_gate_accepted_residuals.csv")
    defect_path = optional_one(case, "ctot_transport_gate_cumulative_defect.raw")
    phi_path = optional_one(case, "ctot_transport_gate_final_phi.raw")
    if not all((meta_path, csv_path, defect_path, phi_path)):
        return {"defect_evidence_complete": False, "accumulated_defect_gate_pass": False}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = read_csv(csv_path)
    residuals = np.array([f(row, "array_cold_Linf") for row in rows])
    defect = np.fromfile(defect_path, np.float64)
    phi = np.fromfile(phi_path, np.float64)
    hv = h(phi)
    masks = {
        "matrix": hv < 0.01,
        "interface": (hv >= 0.01) & (hv <= 0.99),
        "beta_core": hv > 0.99,
    }
    result: dict[str, Any] = {
        "defect_evidence_complete": True,
        "trajectory_schema": meta.get("schema", "UNKNOWN"),
        "trajectory_observed_time_code": float(meta["observed_time_code"]),
        "trajectory_time_closure_error": abs(
            float(meta["observed_time_code"]) - expected_time_code
        ),
        "accepted_residual_p50": float(np.percentile(residuals, 50)),
        "accepted_residual_p95": float(np.percentile(residuals, 95)),
        "accepted_residual_p99": float(np.percentile(residuals, 99)),
        "accepted_residual_max": float(np.max(residuals)),
        "ER_L1": float(meta["cumulative_ER_L1"]),
        "ER_L2": float(meta["cumulative_ER_L2"]),
        "cumulative_C_increment_L1": float(
            meta["cumulative_C_increment_L1"]
        ),
        "eta_R": float(meta["eta_R"]),
        "maximum_abs_local_cumulative_defect": float(np.max(np.abs(defect))),
        "signed_local_cumulative_defect_sum": float(np.sum(defect)),
    }
    for label, mask in masks.items():
        values = defect[mask]
        result[f"{label}_cell_count"] = int(np.sum(mask))
        result[f"{label}_max_abs_D"] = float(np.max(np.abs(values))) if values.size else math.nan
        result[f"{label}_mean_abs_D"] = float(np.mean(np.abs(values))) if values.size else math.nan
        result[f"{label}_signed_D"] = float(np.sum(values)) if values.size else math.nan
    result["trajectory_transaction_closed"] = (
        result["trajectory_time_closure_error"] <= 1.0e-12
    )
    result["accumulated_defect_gate_pass"] = bool(
        result["eta_R"] <= 0.01
        and result["trajectory_transaction_closed"]
    )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("status\nNO_ROWS\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_replay_report() -> None:
    rows: list[dict[str, Any]] = []
    dt4 = REPORT / "frozen_reject_trace" / "dt4_gate_replay.csv"
    if dt4.is_file():
        rows.extend(read_csv(dt4))
    dt8_catalog = (
        ROOT / "reports" / "bounded_retry_bdf2_v1" / "forensics" /
        "dt8_internal_reject_catalog.csv"
    )
    for event in read_csv(dt8_catalog):
        terminal = f(event, "nonlinear_last_accepted_residual")
        for gate_id, gate in GATES.items():
            converged = gate_id != "G12" and terminal <= gate
            rows.append({
                "run": "dt8",
                "reject_index": event["reject_index"],
                "macro_step": event["macro_step"],
                "original_category": event["category"],
                "gate_id": gate_id,
                "requested_gate": gate,
                "relative_gate": 1.0e-10,
                "inferred_initial_residual": f(event, "nonlinear_first_accepted_residual"),
                "effective_gate": gate,
                "replay_status": "CONVERGED_AT_RELAXED_GATE" if converged else "REJECTED_AS_FROZEN",
                "classification": (
                    "RESIDUAL_FLOOR_ONLY" if converged else
                    "LINE_SEARCH_DIRECTION_FAILURE" if event["category"].startswith("C_") else
                    "ITERATION_LIMIT_WITHOUT_GATE_CROSSING"
                ),
                "final_true_cold_residual": terminal,
                "nonlinear_iterations": event["nonlinear_iterations"],
                "line_search_trials": event["nonlinear_rows"],
                "minimum_lambda": event["nonlinear_min_lambda"],
                "iteration_limit_hit": int(event["iteration_limit_status"] == "HIT" and not converged),
                "stagnation_status": int(event["stagnation_status"] == "MIN_LAMBDA_EXHAUSTED" and not converged),
                "fallback_triggered": int(not converged),
                "accepted_endpoint_difference": "NOT_AVAILABLE_NO_ITERATE_FIELD_IN_FROZEN_TRACE",
                "wall_time": "NOT_RECORDED_PER_FROZEN_REJECT",
                "evidence_mode": "FROZEN_CONDENSED_TRACE_CONTROL_FLOW_REPLAY",
            })
    write_csv(REPORT / "frozen_reject_replay.csv", rows)
    counts = Counter(
        (row["gate_id"], row["replay_status"], row["classification"])
        for row in rows
    )
    lines = [
        "# Frozen reject replay", "",
        "The dt/4 replay uses the retained full nonlinear iteration trace. The 16-row dt/8 catalog originates from `active_manifold_bdf2_v1/workstation_runs/dt8_active_manifold_fixed_qualification_1000`, not the later 4000-step bounded-retry trajectory; its full per-iteration file was not retained. Those rows therefore use the frozen condensed trace, and unavailable endpoint/wall quantities remain explicitly `NOT_AVAILABLE` rather than reconstructed.",
        "", "| Gate | Outcome | Classification | Count |", "|---|---|---|---:|",
    ]
    for (gate_id, outcome, classification), count in sorted(counts.items()):
        lines.append(f"| {gate_id} | {outcome} | {classification} | {count} |")
    lines += [
        "", "All relaxed gates are also exercised end-to-end from the byte-identical common checkpoint in the 20-case matrix; those trajectories, rather than condensed replay, are authoritative for endpoint accuracy and wall time.", "",
    ]
    (REPORT / "frozen_reject_replay.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    initial = {
        "Ctot": np.fromfile(COMMON / "Ctot_init.raw", np.float64),
        "phi": np.fromfile(COMMON / "phi_init.raw", np.float64),
        "xB": np.fromfile(COMMON / "xB_init.raw", np.float64),
    }
    reference = load_state(REFERENCE, 16000)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for gate_id, gate in GATES.items():
        for dt_id, (dt, steps) in DTS.items():
            case_id = f"{gate_id}_{dt_id}_diag_on_{steps}"
            case = RUNS / case_id
            status_path = case / "workstation_status.json"
            base: dict[str, Any] = {
                "case_id": f"{gate_id}_{dt_id}",
                "gate_id": gate_id,
                "requested_transport_gate": gate,
                "relative_gate": 1.0e-10,
                "dt_id": dt_id,
                "dt_code": dt,
                "dt_physical_s": dt * T_REAL_UNIT_S,
                "steps": steps,
                "equal_time_code": dt * steps,
                "equal_time_physical_s": dt * steps * T_REAL_UNIT_S,
            }
            if not status_path.is_file():
                base.update({"matrix_status": "PENDING", "full_common_window_pass": False})
                rows.append(base)
                failures.append({"case_id": base["case_id"], "first_failed_gate": "RUN_PENDING", "detail": str(case)})
                continue
            status = json.loads(status_path.read_text(encoding="utf-8"))
            runtime = runtime_metrics(case, status)
            base.update(runtime)
            if not runtime.get("completed"):
                base.update({"matrix_status": "RUNTIME_FAIL", "full_common_window_pass": False})
                rows.append(base)
                failures.append({"case_id": base["case_id"], "first_failed_gate": "RUNTIME_COMPLETION", "detail": runtime.get("first_runtime_failure")})
                continue
            state = load_state(case, steps)
            base.update(accuracy_metrics(state, initial, reference))
            base.update(defect_metrics(case, dt * steps))
            base["full_common_window_pass"] = bool(
                base["hard_numerical_gates_pass"]
                and base["equal_time_accuracy_pass"]
                and base["accumulated_defect_gate_pass"]
                and base["retry_efficiency_gate_pass"]
            )
            base["matrix_status"] = "PASS" if base["full_common_window_pass"] else "FAIL"
            rows.append(base)
            if not base["full_common_window_pass"]:
                for key in (
                    "hard_numerical_gates_pass", "equal_time_accuracy_pass",
                    "accumulated_defect_gate_pass", "retry_efficiency_gate_pass",
                ):
                    if not base.get(key):
                        failures.append({"case_id": base["case_id"], "first_failed_gate": key, "detail": base.get("first_runtime_failure", "")})
                        break

    write_csv(REPORT / "equal_time_metrics.csv", rows)
    manifest_fields = (
        "case_id", "gate_id", "requested_transport_gate", "relative_gate",
        "dt_id", "dt_code", "dt_physical_s", "steps", "equal_time_code",
        "equal_time_physical_s", "matrix_status", "trajectory_schema",
        "trajectory_observed_time_code", "trajectory_time_closure_error",
        "trajectory_transaction_closed", "hard_numerical_gates_pass",
        "equal_time_accuracy_pass", "accumulated_defect_gate_pass",
        "retry_efficiency_gate_pass", "full_common_window_pass",
    )
    write_csv(
        REPORT / "matrix_manifest.csv",
        [{key: row.get(key, "") for key in manifest_fields} for row in rows],
    )
    defect_fields = [
        "case_id", "gate_id", "dt_id", "accepted_residual_p50",
        "accepted_residual_p95", "accepted_residual_p99", "accepted_residual_max",
        "ER_L1", "ER_L2", "eta_R", "maximum_abs_local_cumulative_defect",
        "trajectory_schema", "trajectory_observed_time_code",
        "trajectory_time_closure_error", "trajectory_transaction_closed",
        "signed_local_cumulative_defect_sum", "matrix_cell_count", "matrix_max_abs_D",
        "matrix_mean_abs_D", "matrix_signed_D", "interface_cell_count",
        "interface_max_abs_D", "interface_mean_abs_D", "interface_signed_D",
        "beta_core_cell_count", "beta_core_max_abs_D", "beta_core_mean_abs_D",
        "beta_core_signed_D", "accumulated_defect_gate_pass",
    ]
    defect_rows = [{key: row.get(key, "") for key in defect_fields} for row in rows]
    write_csv(REPORT / "cumulative_defect_metrics.csv", defect_rows)
    write_csv(REPORT / "first_failure.csv", failures)

    table = [
        "| Case | C inc | phi inc | h inc | profile | interface | transfer | far field | eta_R | retry wall | throughput | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if not row.get("completed"):
            table.append(f"| {row['case_id']} | - | - | - | - | - | - | - | - | - | - | {row['matrix_status']} |")
            continue
        table.append(
            f"| {row['case_id']} | {row['Ctot_increment_relative_L2_error']:.3%} | "
            f"{row['phi_increment_relative_L2_error']:.3%} | {row['hvolume_increment_relative_error']:.3%} | "
            f"{row['matrix_profile_capacity_weighted_relative_error']:.3%} | {row['phi_half_interface_error_dx']:.4f} | "
            f"{row['cumulative_transfer_relative_error']:.3%} | {row['far_field_relative_error']:.3%} | "
            f"{row['eta_R']:.3e} | {row['retry_wall_overhead_fraction']:.3%} | "
            f"{row['physical_s_per_GPU_hour']:.1f} | {row['matrix_status']} |"
        )
    (REPORT / "equal_time_comparison.md").write_text(
        "# Transport residual-gate equal-time comparison\n\n"
        "All rows start from the same raw checkpoint and end at 1.5625 code time (64.264977 s). The reference is strict dt/32. Errors are increment-normalized where registered, so the background field cannot hide evolution error.\n\n"
        + "\n".join(table) + "\n",
        encoding="utf-8",
    )
    completed = [row for row in rows if row.get("completed")]
    worst_region = "NOT_AVAILABLE"
    if completed:
        region_scores = {
            region: max(float(row.get(f"{region}_max_abs_D", 0.0)) for row in completed)
            for region in ("matrix", "interface", "beta_core")
        }
        worst_region = max(region_scores, key=region_scores.get)
    (REPORT / "cumulative_defect_audit.md").write_text(
        "# Cumulative transport defect audit\n\n"
        "The runtime observer records only accepted method-consistent cold residuals. `E_R_L1=sum dt ||R||_1`, `E_R_L2=sqrt(sum dt ||R||_2^2)`, `eta_R=E_R_L1/sum ||Delta C||_1`, and `D_i=sum dt R_i`. Regions are registered from final `h(phi)`: matrix `<0.01`, interface `[0.01,0.99]`, beta core `>0.99`.\n\n"
        f"Largest regional maximum in the currently completed matrix: `{worst_region}`. Signed regional sums and every candidate metric are retained in `cumulative_defect_metrics.csv`; no zero global mass residual is used as a substitute for local-defect evidence.\n",
        encoding="utf-8",
    )
    build_replay_report()
    print(json.dumps({
        "rows": len(rows),
        "completed": sum(bool(row.get("completed")) for row in rows),
        "passed": sum(bool(row.get("full_common_window_pass")) for row in rows),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
