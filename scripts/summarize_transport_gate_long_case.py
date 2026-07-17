#!/usr/bin/env python3
"""Condense one long transport-gate run without loading solver data into memory."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import re
from array import array
from pathlib import Path


HARD_COLUMNS = (
    "transport_nonlinear_converged",
    "phase_constraint_converged",
    "elasticity_converged",
    "outer_coupling_converged",
    "local_phase_storage_closed",
    "global_mass_closed",
    "Ctot_admissible",
    "q_alpha_admissible",
    "xB_alpha_admissible",
    "no_nan_inf",
    "physical_projection_zero",
    "no_mass_loss_clipping",
    "energy_work_audit_passed",
    "restart_metadata_consistent",
)


def find_one(root: Path, name: str) -> Path | None:
    found = list(root.rglob(name))
    return found[0] if len(found) == 1 else None


def rows(path: Path | None):
    if path is None:
        return
    with path.open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def number(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return default


def read_doubles(path: Path | None) -> array:
    values = array("d")
    if path is not None:
        with path.open("rb") as stream:
            values.fromfile(stream, path.stat().st_size // values.itemsize)
    return values


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    position = (len(values) - 1) * pct / 100.0
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return values[lo]
    weight = position - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def h(phi: float) -> float:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def parse_last_kv(path: Path, prefix: str) -> dict[str, str]:
    last = ""
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith(prefix):
                last = line
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", last))


def event_fallback_steps(path: Path) -> list[int]:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.case_dir.resolve()
    output = args.output or root / "long_remote_summary.json"

    accepted_count = 0
    acceptance_hard_pass = True
    for row in rows(find_one(root, "ctot_acceptance_predicate.csv")) or ():
        if row.get("accepted") != "1":
            continue
        accepted_count += 1
        acceptance_hard_pass &= all(row.get(key) == "1" for key in HARD_COLUMNS)

    energy_count = 0
    energy_pass = True
    for row in rows(find_one(root, "ctot_energy_work.csv")) or ():
        if row.get("accepted") != "1":
            continue
        energy_count += 1
        energy_pass &= (
            row.get("monotone_pass") == "1"
            and row.get("balance_pass") == "1"
            and math.isfinite(number(row, "energy_balance_rel"))
        )

    retry_rows = 0
    rejected_rows = 0
    rollback_pass = True
    for row in rows(find_one(root, "ctot_retry_attempts.csv")) or ():
        retry_rows += 1
        if row.get("accepted") == "0":
            rejected_rows += 1
            rollback_pass &= row.get("rollback_bitwise") == "1"

    failed_cell_worst: dict[tuple[int, int], tuple[float, int]] = {}
    for row in rows(find_one(root, "ctot_failed_cell_state.csv")) or ():
        key = (
            int(number(row, "physical_step", -1)),
            int(number(row, "attempt_id", -1)),
        )
        candidate = (abs(number(row, "residual", 0.0)),
                     int(number(row, "idx", -1)))
        if key not in failed_cell_worst or candidate[0] > failed_cell_worst[key][0]:
            failed_cell_worst[key] = candidate
    failed_cell_counts = Counter(item[1] for item in failed_cell_worst.values())
    persistent_failed_cells = sorted(
        idx for idx, count in failed_cell_counts.items()
        if idx >= 0 and count >= 3
    )

    split_rows = 0
    accepted_split_rows = 0
    max_mass_error = 0.0
    max_phase_kkt = 0.0
    transport_solves = 0
    phase_solves = 0
    for row in rows(find_one(root, "ctot_split_step_metrics.csv")) or ():
        split_rows += 1
        transport_solves += int(number(row, "transport_solves", 0.0))
        phase_solves += int(number(row, "phase_solves", 0.0))
        if row.get("accepted") != "1":
            continue
        accepted_split_rows += 1
        max_mass_error = max(max_mass_error, abs(number(row, "mass_error", 0.0)))
        max_phase_kkt = max(max_phase_kkt, abs(number(row, "phase_KKT", 0.0)))

    residuals = [
        number(row, "array_cold_Linf")
        for row in rows(find_one(root, "ctot_transport_gate_accepted_residuals.csv")) or ()
    ]
    residuals = [value for value in residuals if math.isfinite(value)]

    trajectory_meta_path = find_one(root, "ctot_transport_gate_trajectory_meta.json")
    trajectory_meta = (
        json.loads(trajectory_meta_path.read_text(encoding="utf-8"))
        if trajectory_meta_path else {}
    )
    defect = read_doubles(find_one(root, "ctot_transport_gate_cumulative_defect.raw"))
    phi = read_doubles(find_one(root, "ctot_transport_gate_final_phi.raw"))
    region: dict[str, dict[str, float | int]] = {
        key: {"count": 0, "max_abs": 0.0, "mean_abs": 0.0, "signed": 0.0}
        for key in ("matrix", "interface", "beta_core")
    }
    if len(defect) == len(phi):
        for d_value, phi_value in zip(defect, phi):
            hv = h(phi_value)
            key = "matrix" if hv < 0.01 else "beta_core" if hv > 0.99 else "interface"
            item = region[key]
            item["count"] = int(item["count"]) + 1
            item["max_abs"] = max(float(item["max_abs"]), abs(d_value))
            item["mean_abs"] = float(item["mean_abs"]) + abs(d_value)
            item["signed"] = float(item["signed"]) + d_value
        for item in region.values():
            if item["count"]:
                item["mean_abs"] = float(item["mean_abs"]) / int(item["count"])

    log_path = root / "run.log"
    bounded = parse_last_kv(log_path, "CTOT_BOUNDED_RETRY_SUMMARY")
    fallback_steps = event_fallback_steps(log_path)
    result = {
        "schema": "CTOT_TRANSPORT_GATE_LONG_REMOTE_SUMMARY_V1",
        "accepted_count": accepted_count,
        "acceptance_hard_pass": bool(accepted_count and acceptance_hard_pass),
        "energy_count": energy_count,
        "energy_pass": bool(energy_count and energy_pass),
        "retry_rows": retry_rows,
        "rejected_rows": rejected_rows,
        "rollback_pass": rollback_pass,
        "persistent_failed_cells": persistent_failed_cells,
        "failed_cell_attempt_counts": {
            str(idx): count for idx, count in sorted(failed_cell_counts.items())
            if idx >= 0
        },
        "split_rows": split_rows,
        "accepted_split_rows": accepted_split_rows,
        "max_mass_error": max_mass_error,
        "max_phase_kkt": max_phase_kkt,
        "transport_solves": transport_solves,
        "phase_solves": phase_solves,
        "accepted_residual_p50": percentile(residuals, 50.0),
        "accepted_residual_p95": percentile(residuals, 95.0),
        "accepted_residual_p99": percentile(residuals, 99.0),
        "accepted_residual_max": max(residuals, default=math.nan),
        "trajectory_meta": trajectory_meta,
        "defect_region": region,
        "maximum_abs_local_cumulative_defect": max((abs(value) for value in defect), default=math.nan),
        "signed_local_cumulative_defect_sum": sum(defect),
        "bounded_retry_summary": bounded,
        "event_fallback_steps": fallback_steps,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
