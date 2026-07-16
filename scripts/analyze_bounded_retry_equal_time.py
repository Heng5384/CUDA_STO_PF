#!/usr/bin/env python3
"""Evaluate the preregistered bounded-retry common-state equal-time matrix."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "bounded_retry_bdf2_v1"
RUNS = REPORT / "workstation_runs"
SOURCE = (
    ROOT / "reports" / "T400_longtime_v1" / "coarse4_runs" /
    "growth_N512_shift0_dt8_pre_event_freeze"
)
TIME_UNIT_S = 41.1295854245547687
CASES = {
    "dt4": (7.81250000000000043e-4, 2000),
    "dt8": (3.90625000000000022e-4, 4000),
    "dt16": (1.95312500000000011e-4, 8000),
    "fine_dt32": (9.76562500000000054e-5, 16000),
}


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def l2(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def right_interface_position(phi: np.ndarray) -> float:
    center = int(np.argmax(phi))
    for offset in range(1, phi.size // 2 + 1):
        left = (center + offset - 1) % phi.size
        right = (center + offset) % phi.size
        a, b = float(phi[left]), float(phi[right])
        if a >= 0.5 and b < 0.5:
            return center + offset - 1 + (a - 0.5) / max(a - b, 1.0e-300)
    return math.nan


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def find_one(root: Path, pattern: str) -> Path:
    values = list(root.glob(pattern))
    if len(values) != 1:
        raise RuntimeError(f"expected one {pattern} in {root}, got {values}")
    return values[0]


def state(root: Path, step: int) -> dict[str, np.ndarray]:
    return {
        "Ctot": np.fromfile(find_one(root, f"*step{step:06d}_Ctot.raw"), np.float64),
        "phi": np.fromfile(find_one(root, f"*step{step:06d}_phi.raw"), np.float64),
        "xB": np.fromfile(find_one(root, f"*step{step:06d}_xB_alpha.raw"), np.float64),
    }


def kv_line(text: str, prefix: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise RuntimeError(f"expected one {prefix}, got {len(lines)}")
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", lines[0]))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def i(row: dict[str, str], key: str) -> int:
    return int(row[key])


def runtime_metrics(name: str, run: Path) -> dict[str, object]:
    text = (run / "run.log").read_text(errors="replace")
    summary = kv_line(text, "CTOT_BOUNDED_RETRY_SUMMARY")
    status = json.loads((run / "workstation_status.json").read_text())
    accepted = read_csv(run / "ctot_acceptance_predicate.csv")
    accepted_rows = [r for r in accepted if r.get("accepted") == "1"]
    accepted_gate_columns = [
        "transport_nonlinear_converged", "phase_constraint_converged",
        "elasticity_converged", "outer_coupling_converged",
        "local_phase_storage_closed", "global_mass_closed",
        "Ctot_admissible", "q_alpha_admissible", "xB_alpha_admissible",
        "no_nan_inf", "physical_projection_zero", "no_mass_loss_clipping",
        "energy_work_audit_passed", "restart_metadata_consistent",
    ]
    accepted_gates_pass = bool(accepted_rows) and all(
        all(r.get(key) == "1" for key in accepted_gate_columns)
        for r in accepted_rows
    )
    energy = [r for r in read_csv(run / "ctot_energy_work.csv") if r.get("accepted") == "1"]
    energy_all_finite = bool(energy) and all(
        all(math.isfinite(float(r[key])) for key in (
            "F_before", "F_after_phase", "F_after_transport", "F_final",
            "energy_balance_residual", "energy_balance_rel",
        )) for r in energy
    )
    retry = read_csv(run / "ctot_retry_attempts.csv")
    rejected = [r for r in retry if r.get("accepted") == "0"]
    rollback_log = re.findall(r"CTOT_BOUNDED_RETRY_ROLLBACK[^\n]*bitwise=(\d+)", text)
    rollback_bitwise = bool(rollback_log) and all(value == "1" for value in rollback_log)
    if not rejected:
        rollback_bitwise = True

    failed = read_csv(run / "ctot_failed_cell_state.csv")
    worst_by_step: dict[int, tuple[float, int]] = {}
    for row in failed:
        try:
            step = int(row["physical_step"])
            value = abs(float(row["residual"]))
            candidate = (value, int(row["idx"]))
            if step not in worst_by_step or candidate[0] > worst_by_step[step][0]:
                worst_by_step[step] = candidate
        except (KeyError, ValueError):
            continue
    dominant = {step: value[1] for step, value in worst_by_step.items()}
    repeated = Counter(dominant.values())
    persistent_cells = sorted(idx for idx, count in repeated.items() if count >= 3)
    no_persistent = not persistent_cells

    dt, steps = CASES[name]
    wall = float(status["wall_seconds"])
    physical = dt * steps * TIME_UNIT_S
    row: dict[str, object] = {
        "case": name,
        "dt_code": dt,
        "dt_physical_s": dt * TIME_UNIT_S,
        "macro_steps": i(summary, "macro_steps"),
        "equal_time_code": dt * steps,
        "equal_time_physical_s": physical,
        "wall_seconds": wall,
        "physical_s_per_GPU_hour": physical / wall * 3600.0,
        "macro_hard_rejects": i(summary, "macro_hard_rejects"),
        "internal_trial_rejects": i(summary, "internal_trial_rejects"),
        "retry_fraction": f(summary, "retry_fraction"),
        "fallback_macros": i(summary, "fallback_macros"),
        "fallback_fraction": f(summary, "fallback_fraction"),
        "retry_wall_overhead_fraction": f(summary, "reject_trial_wall_fraction"),
        "max_consecutive_fallback_macros": i(summary, "max_consecutive_fallback_macros"),
        "max_accepted_subcycle_depth": i(summary, "max_accepted_subcycle_depth"),
        "accepted_iteration_p99": f(summary, "accepted_iteration_p99"),
        "nonlinear_iteration_budget": i(summary, "nonlinear_iteration_budget"),
        "accepted_state_gates_pass": accepted_gates_pass,
        "energy_all_finite": energy_all_finite,
        "rollback_bitwise": rollback_bitwise,
        "rejected_retry_rows": len(rejected),
        "persistent_interface_cell_rejection": not no_persistent,
        "persistent_interface_cells": ";".join(map(str, persistent_cells)),
    }
    row["bounded_retry_frequency_gate_pass"] = (
        row["macro_hard_rejects"] == 0 and accepted_gates_pass and
        row["retry_fraction"] <= 0.01 and row["fallback_fraction"] <= 0.01 and
        row["retry_wall_overhead_fraction"] <= 0.05 and
        row["max_consecutive_fallback_macros"] <= 2 and
        row["max_accepted_subcycle_depth"] <= 2 and no_persistent and
        row["accepted_iteration_p99"] < 0.8 * row["nonlinear_iteration_budget"]
    )
    return row


def main() -> int:
    initial = {
        "Ctot": np.fromfile(SOURCE / "Ctot_init.raw", np.float64),
        "phi": np.fromfile(SOURCE / "phi_init.raw", np.float64),
        "xB": np.fromfile(SOURCE / "xB_init.raw", np.float64),
    }
    states: dict[str, dict[str, np.ndarray]] = {}
    runtime: dict[str, dict[str, object]] = {}
    for name, (_, steps) in CASES.items():
        run = RUNS / f"{name}_common_state_{steps}"
        states[name] = state(run, steps)
        runtime[name] = runtime_metrics(name, run)

    ref = states["fine_dt32"]
    h0, href = h(initial["phi"]), h(ref["phi"])
    ref_interface = right_interface_position(ref["phi"])
    initial_interface = right_interface_position(initial["phi"])
    rows: list[dict[str, object]] = []
    for name in CASES:
        current = states[name]
        hc = h(current["phi"])
        alpha_ref = 1.0 - href
        C_increment_scale = l2(ref["Ctot"] - initial["Ctot"])
        phi_increment_scale = l2(ref["phi"] - initial["phi"])
        h_increment_scale = abs(float(np.sum(href - h0)))
        profile_scale = float(np.sum(alpha_ref * np.abs(ref["xB"] - initial["xB"])))
        pos = right_interface_position(current["phi"])
        direction_same = (
            np.sign(pos - initial_interface) == np.sign(ref_interface - initial_interface)
        )
        row = dict(runtime[name])
        row.update({
            "Ctot_field_relative_L2_error": l2(current["Ctot"] - ref["Ctot"])
                / max(l2(ref["Ctot"]), 1.0e-300),
            "Ctot_increment_relative_L2_error": l2(current["Ctot"] - ref["Ctot"])
                / max(C_increment_scale, 1.0e-300),
            "phi_field_relative_L2_error": l2(current["phi"] - ref["phi"])
                / max(l2(ref["phi"]), 1.0e-300),
            "phi_increment_relative_L2_error": l2(current["phi"] - ref["phi"])
                / max(phi_increment_scale, 1.0e-300),
            "hvolume": float(np.sum(hc)),
            "hvolume_reference": float(np.sum(href)),
            "hvolume_increment_relative_error": abs(float(np.sum(hc - href)))
                / max(h_increment_scale, 1.0e-300),
            "interface_position_dx": pos,
            "interface_reference_dx": ref_interface,
            "interface_error_dx": abs(pos - ref_interface),
            "interface_direction_same": bool(direction_same),
            "cumulative_transfer": 0.5 * float(np.sum(np.abs(current["Ctot"] - initial["Ctot"]))),
            "cumulative_transfer_reference": 0.5 * float(np.sum(np.abs(ref["Ctot"] - initial["Ctot"]))),
            "matrix_profile_capacity_weighted_relative_error":
                float(np.sum(alpha_ref * np.abs(current["xB"] - ref["xB"])))
                / max(profile_scale, 1.0e-300),
            "final_mass_difference_from_reference_abs":
                abs(float(np.sum(current["Ctot"]) - np.sum(ref["Ctot"]))),
        })
        row["equal_time_accuracy_pass"] = bool(
            row["Ctot_increment_relative_L2_error"] <= 0.02 and
            row["phi_increment_relative_L2_error"] <= 0.02 and
            row["hvolume_increment_relative_error"] <= 0.02 and
            row["matrix_profile_capacity_weighted_relative_error"] <= 0.03 and
            row["interface_error_dx"] <= 0.25 and direction_same
        )
        row["full_bounded_retry_contract_pass"] = bool(
            row["bounded_retry_frequency_gate_pass"] and
            row["equal_time_accuracy_pass"]
        )
        rows.append(row)

    with (REPORT / "equal_time_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (REPORT / "equal_time_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )

    table = "\n".join(
        f"| {r['case']} | {r['dt_code']:.9g} | {r['macro_steps']} | "
        f"{r['Ctot_increment_relative_L2_error']:.3%} | "
        f"{r['phi_increment_relative_L2_error']:.3%} | "
        f"{r['hvolume_increment_relative_error']:.3%} | "
        f"{r['matrix_profile_capacity_weighted_relative_error']:.3%} | "
        f"{r['interface_error_dx']:.4f} | {r['retry_fraction']:.3%} | "
        f"{r['retry_wall_overhead_fraction']:.3%} | "
        f"{r['physical_s_per_GPU_hour']:.3f} | "
        f"{'PASS' if r['full_bounded_retry_contract_pass'] else 'FAIL'} |"
        for r in rows
    )
    (REPORT / "equal_time_comparison.md").write_text(
        "# Common-state equal-time comparison\n\n"
        "All four runs start from byte-identical `Ctot`, `phi`, and `xB_alpha` "
        "fields with invalid BDF2 history, and therefore share the same BE startup. "
        "They end at code time 1.5625 (64.264977 s). The fine reference is active-"
        "manifold BDF2 at dt/32.\n\n"
        "| Case | dt code | Steps | C increment error | phi increment error | "
        "h increment error | matrix profile error | interface error (dx) | "
        "reject fraction | reject wall | physical s/GPU h | Full gate |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n" +
        table + "\n\nThe errors use the preregistered increment-normalized definitions; "
        "the matrix profile is alpha-capacity weighted.\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
