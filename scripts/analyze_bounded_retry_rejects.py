#!/usr/bin/env python3
"""Compact forensic extraction for frozen active-manifold BDF2 rejects."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ;]+)")


def kv(line: str) -> dict[str, str]:
    return dict(KV_RE.findall(line))


def as_float(value: str | None) -> float:
    try:
        return float(value) if value is not None else math.nan
    except ValueError:
        return math.nan


def as_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def classify(event: dict[str, object], max_iter: int) -> tuple[str, str, str]:
    reason = str(event.get("baseline_failure_reason", ""))
    iterations = int(event["nonlinear_iterations"])
    if reason == "coupled_outer_max_iter_not_converged":
        return (
            "J_OTHER_PROVEN_CAUSE",
            "METHOD_TRANSPORT_COLD_GATE_MARGINAL",
            "main_cuda.cu:37773-37849",
        )
    if iterations + 1 >= max_iter:
        return (
            "B_TRANSPORT_NONLINEAR_ITERATION_LIMIT",
            "TRANSPORT_NONLINEAR_ITERATION_LIMIT",
            "main_cuda.cu:34422-34429",
        )
    if event.get("feasible_iterate_restored"):
        return (
            "C_TRANSPORT_LINE_SEARCH_STAGNATION",
            "TRANSPORT_LINE_SEARCH_STAGNATION",
            "main_cuda.cu:34388-34419",
        )
    return (
        "J_OTHER_PROVEN_CAUSE",
        "UNCLASSIFIED_BASELINE_DIAGNOSTIC_GAP",
        "main_cuda.cu:38666-38734",
    )


def parse_log(path: Path, run: str, dt: float, t_real_unit: float,
              max_iter: int) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    pending_active: dict[str, str] = {}
    context: dict[str, object] = {}
    last_phase: dict[str, str] = {}
    last_energy: dict[str, str] = {}
    last_outer: dict[str, str] = {}
    coordinate_switch = False
    feasible_restore = False
    current_event: dict[str, object] | None = None
    ordinal_by_step: defaultdict[int, int] = defaultdict(int)

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "CTOT_BDF2_ACTIVE_MANIFOLD_CONTEXT" in line:
            pending_active = kv(line)
            continue
        if "CTOT_IMEX_BDF2_STEP_CONTEXT" in line:
            values = kv(line)
            context = dict(values)
            context["active"] = (
                dict(pending_active) if values.get("integrator") == "BDF2" else {}
            )
            pending_active = {}
            last_phase = {}
            last_energy = {}
            last_outer = {}
            coordinate_switch = False
            feasible_restore = False
            continue
        if "CTOT_TRANSPORT_COORDINATE_SWITCH" in line:
            coordinate_switch = True
            continue
        if "CTOT_TRANSPORT_FAILED_ITERATE_RESTORED" in line:
            feasible_restore = kv(line).get("status") == "PASS"
            continue
        if "CTOT_PHASE_INNER" in line:
            last_phase = kv(line)
            continue
        if "CTOT_IMEX_BDF2_ENERGY_WORK" in line:
            last_energy = kv(line)
            continue
        if "CTOT_ELASTIC_OUTER" in line:
            last_outer = kv(line)
            continue
        if line.startswith("[reject]"):
            values = kv(line)
            step = as_int(values.get("step"))
            ordinal_by_step[step] += 1
            active = context.get("active", {})
            current_event = {
                "run": run,
                "reject_index": len(events) + 1,
                "macro_step": step,
                "reject_ordinal_in_step": ordinal_by_step[step],
                "physical_time_code": step * dt,
                "physical_time_s": step * dt * t_real_unit,
                "baseline_log_line": line_no,
                "integrator_mode": context.get("integrator", "UNKNOWN"),
                "history_valid": as_int(context.get("history_valid")),
                "fallback_pending_before": as_int(
                    context.get("fallback_pending")
                ),
                "active_adjusted_cells": as_int(active.get("adjusted_cells")),
                "active_qalpha_lower_cells": as_int(
                    active.get("qalpha_lower_cells")
                ),
                "active_phi_lower_cells": as_int(
                    active.get("phi_lower_cells")
                ),
                "active_first_cell": as_int(active.get("first_cell"), -1),
                "active_set_transition_type": (
                    "QALPHA_LOWER_MANIFOLD"
                    if as_int(active.get("qalpha_lower_cells")) > 0
                    else "PHI_LOWER_MANIFOLD"
                    if as_int(active.get("phi_lower_cells")) > 0
                    else "FREE_OR_BE_CONTEXT"
                ),
                "nonlinear_iterations": as_int(values.get("iters")),
                "reject_residual_Linf": as_float(values.get("res_inf")),
                "mass_error": as_float(values.get("mass_error")),
                "sum_divJ": as_float(values.get("sum_divJ")),
                "nonfinite_count": as_float(values.get("nonfinite")),
                "bound_violation_count": as_float(values.get("bounds")),
                "mobility_failure_count": as_float(values.get("mobility_fail")),
                "min_mobility": as_float(values.get("M")),
                "coordinate_switch": coordinate_switch,
                "feasible_iterate_restored": feasible_restore,
                "phase_iterations": as_int(last_phase.get("iterations")),
                "phase_KKT_Linf": as_float(last_phase.get("final_KKT")),
                "phase_converged": as_int(last_phase.get("converged")),
                "energy_work_finite": bool(last_energy),
                "energy_work_pass": as_int(last_energy.get("pass")),
                "energy_balance_rel": as_float(last_energy.get("balance_rel")),
                "outer_transport_contract_pass": as_int(
                    last_outer.get("transport_contract_pass"), -1
                ),
                "outer_converged": as_int(last_outer.get("converged"), -1),
                "method_transport_residual": as_float(
                    last_outer.get("transport_solve")
                ),
                "baseline_failure_stage": "pending",
                "baseline_failure_reason": "pending",
                "fallback_subcycle_depth": 0,
                "rollback_hash_status": "NOT_LOGGED_IN_FROZEN_BASELINE",
                "same_macro_recovered": False,
            }
            events.append(current_event)
            continue
        if "CTOT_BDF2_EVENT_SUBCYCLE_RETRY" in line and current_event is not None:
            values = kv(line)
            current_event["baseline_failure_stage"] = values.get(
                "failed_stage", "UNKNOWN"
            )
            current_event["baseline_failure_reason"] = values.get(
                "failed_reason", "UNKNOWN"
            )
            current_event["fallback_subcycle_depth"] = as_int(values.get("depth"))
            category, predicate, source = classify(current_event, max_iter)
            current_event["category"] = category
            current_event["first_failing_predicate"] = predicate
            current_event["first_failing_function"] = (
                "solve_ctot_transport_operator"
                if category.startswith(("B_", "C_"))
                else "coupled_outer_acceptance_gate"
            )
            current_event["baseline_source_line_range"] = source
            continue
        if "CTOT_BDF2_EVENT_MACRO_READY" in line and current_event is not None:
            step = as_int(kv(line).get("step"))
            for event in reversed(events):
                if event["macro_step"] != step:
                    break
                event["same_macro_recovered"] = True
            current_event = None

    return events


def failed_cell_groups(path: Path, events: list[dict[str, object]], total_r: int) -> None:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(events) * total_r:
        raise RuntimeError(
            f"{path}: expected {len(events) * total_r} failed-cell rows, got {len(rows)}"
        )
    for event, start in zip(events, range(0, len(rows), total_r)):
        group = rows[start : start + total_r]
        worst = max(group, key=lambda row: abs(float(row["residual"])))
        for key in (
            "idx", "i", "j", "k", "C_old", "C_trial", "phi", "h",
            "alpha", "q_alpha", "xB", "mu", "divJ", "residual",
            "face_x_out", "face_x_in", "face_y_out", "face_y_in",
            "face_z_out", "face_z_in",
        ):
            event[f"worst_{key}"] = worst[key]


def nonlinear_segments(path: Path, events: list[dict[str, object]]) -> None:
    events_by_step: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        events_by_step[int(event["macro_step"])].append(event)
    wanted = set(events_by_step)
    segments_by_step: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    current_step: int | None = None
    current: dict[str, object] | None = None
    previous_iteration: int | None = None

    def finish() -> None:
        nonlocal current
        if current is not None:
            history = current.pop("accepted_history")
            current["accepted_residual_history"] = json.dumps(
                history[:5] + (["..."] if len(history) > 10 else []) + history[-5:]
            )
            segments_by_step[int(current["step"])].append(current)
        current = None

    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            step = int(row["step"])
            if step not in wanted:
                continue
            iteration = int(row["iteration"])
            if current_step != step or (
                previous_iteration is not None and iteration < previous_iteration
            ):
                finish()
                current_step = step
                current = {
                    "step": step,
                    "rows": 0,
                    "max_iteration": 0,
                    "accepted_trial_count": 0,
                    "min_lambda": math.inf,
                    "last_lambda": math.nan,
                    "last_trial_accepted": 0,
                    "accepted_history": [],
                }
            assert current is not None
            current["rows"] = int(current["rows"]) + 1
            current["max_iteration"] = max(int(current["max_iteration"]), iteration)
            accepted = int(row["accepted"])
            current["accepted_trial_count"] = int(current["accepted_trial_count"]) + accepted
            current["min_lambda"] = min(
                float(current["min_lambda"]), float(row["line_search_lambda"])
            )
            current["last_lambda"] = float(row["line_search_lambda"])
            current["last_trial_accepted"] = accepted
            if accepted:
                current["accepted_history"].append(float(row["res_inf"]))
            previous_iteration = iteration
    finish()

    for step, step_events in events_by_step.items():
        segments = segments_by_step[step]
        if len(segments) < len(step_events):
            raise RuntimeError(
                f"step {step}: {len(step_events)} rejects but {len(segments)} segments"
            )
        cursor = 0
        matched: list[tuple[dict[str, object], dict[str, object]]] = []
        for event in step_events:
            category = str(event["category"])
            reported_iters = int(event["nonlinear_iterations"])
            selected_index = -1
            for index in range(cursor, len(segments)):
                candidate = segments[index]
                max_iteration = int(candidate["max_iteration"])
                last_accepted = int(candidate["last_trial_accepted"])
                if category.startswith("B_"):
                    matches = max_iteration == reported_iters + 1
                elif category.startswith("C_"):
                    matches = (
                        max_iteration == reported_iters + 1 and
                        last_accepted == 0
                    )
                else:
                    matches = max_iteration in (reported_iters, reported_iters + 1)
                if matches:
                    selected_index = index
                    break
            if selected_index < 0:
                raise RuntimeError(
                    f"step {step}: cannot map reject {reported_iters}/{category} "
                    f"after segment {cursor}"
                )
            matched.append((event, segments[selected_index]))
            cursor = selected_index + 1
        for event, segment in matched:
            for key, value in segment.items():
                if key != "step":
                    event[f"nonlinear_{key}"] = value
            history_text = str(segment["accepted_residual_history"])
            parsed = json.loads(history_text)
            numeric = [value for value in parsed if isinstance(value, float)]
            event["nonlinear_first_accepted_residual"] = (
                numeric[0] if numeric else math.nan
            )
            event["nonlinear_last_accepted_residual"] = (
                numeric[-1] if numeric else math.nan
            )
            event["iteration_limit_status"] = (
                "HIT" if str(event["category"]).startswith("B_") else "NOT_HIT"
            )
            event["stagnation_status"] = (
                "MIN_LAMBDA_EXHAUSTED"
                if str(event["category"]).startswith("C_")
                else "NOT_PRIMARY_CAUSE"
            )
            event["cold_residual_history_status"] = (
                "FINAL_METHOD_COLD_RESIDUAL_ONLY"
                if event["energy_work_finite"]
                else "NOT_REACHED_BEFORE_REJECT"
            )


def write_catalog(path: Path, events: list[dict[str, object]]) -> None:
    preferred = [
        "run", "reject_index", "macro_step", "reject_ordinal_in_step",
        "physical_time_code", "physical_time_s", "category",
        "first_failing_predicate", "first_failing_function",
        "baseline_source_line_range", "baseline_log_line", "integrator_mode",
        "baseline_failure_stage", "baseline_failure_reason",
        "fallback_subcycle_depth", "history_valid", "active_set_transition_type",
        "active_adjusted_cells", "active_qalpha_lower_cells",
        "active_phi_lower_cells", "nonlinear_iterations",
        "iteration_limit_status", "stagnation_status", "coordinate_switch",
        "feasible_iterate_restored", "reject_residual_Linf", "mass_error",
        "sum_divJ", "nonfinite_count", "bound_violation_count",
        "mobility_failure_count", "nonlinear_rows", "nonlinear_max_iteration",
        "nonlinear_accepted_trial_count", "nonlinear_min_lambda",
        "nonlinear_last_lambda", "nonlinear_last_trial_accepted",
        "nonlinear_first_accepted_residual", "nonlinear_last_accepted_residual",
        "nonlinear_accepted_residual_history", "cold_residual_history_status",
        "worst_idx", "worst_i", "worst_j", "worst_k", "worst_phi",
        "worst_h", "worst_alpha", "worst_q_alpha", "worst_xB", "worst_mu",
        "worst_divJ", "worst_residual", "worst_face_x_out", "worst_face_x_in",
        "worst_face_y_out", "worst_face_y_in", "worst_face_z_out",
        "worst_face_z_in", "phase_iterations", "phase_KKT_Linf",
        "phase_converged", "energy_work_finite", "energy_work_pass",
        "energy_balance_rel", "outer_transport_contract_pass",
        "outer_converged", "method_transport_residual",
        "rollback_hash_status", "same_macro_recovered",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=preferred, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)


def write_summary(path: Path, events: list[dict[str, object]]) -> None:
    counts = Counter(str(event["category"]) for event in events)
    cells = Counter(int(event["worst_idx"]) for event in events)
    consecutive: defaultdict[str, list[list[int]]] = defaultdict(list)
    for run in sorted({str(event["run"]) for event in events}):
        steps = sorted({int(e["macro_step"]) for e in events if e["run"] == run})
        groups: list[list[int]] = []
        for step in steps:
            if groups and step == groups[-1][-1] + 1:
                groups[-1].append(step)
            else:
                groups.append([step])
        consecutive[run] = groups
    lines = [
        "# Internal reject forensics",
        "",
        "The frozen zero-reject reports and their FAIL conclusion are unchanged.",
        "This report only classifies the already-recorded internal trials.",
        "",
        "## Counts",
        "",
    ]
    for category, count in sorted(counts.items()):
        lines.append(f"- `{category}`: {count}")
    lines += ["", "## Spatial clustering", ""]
    for cell, count in cells.most_common(12):
        lines.append(f"- worst cell `{cell}`: {count} rejects")
    lines += ["", "## Temporal clusters", ""]
    for run, groups in consecutive.items():
        formatted = [
            str(group[0]) if len(group) == 1 else f"{group[0]}-{group[-1]}"
            for group in groups
        ]
        lines.append(f"- `{run}`: {', '.join(formatted)}")
    lines += [
        "",
        "## Evidence limits",
        "",
        "- The frozen baseline did not hash every rejected rollback. The new default-off contract adds this evidence; the forced rollback smoke is bitwise PASS.",
        "- Per-iteration cold residuals were not stored. The catalog reports accepted nonlinear trial history and the final method-consistent cold audit when reached.",
        "- Cell mobility was not stored per cell; the frozen reject line only recorded its global range. No value is reconstructed or fabricated.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--detail-dir", type=Path, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--t-real-unit", type=float, required=True)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--total-r", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    events = parse_log(args.log, args.run, args.dt, args.t_real_unit, args.max_iter)
    failed_cell_groups(
        args.detail_dir / "ctot_failed_cell_state.csv", events, args.total_r
    )
    nonlinear_segments(args.detail_dir / "ctot_nonlinear_iterations.csv", events)
    write_catalog(args.output / f"{args.run}_internal_reject_catalog.csv", events)
    write_summary(args.output / f"{args.run}_internal_reject_forensics.md", events)
    (args.output / f"{args.run}_events.json").write_text(
        json.dumps(events, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "run": args.run,
        "rejects": len(events),
        "categories": Counter(str(e["category"]) for e in events),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
