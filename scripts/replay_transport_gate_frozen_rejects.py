#!/usr/bin/env python3
"""Counterfactual gate replay over frozen nonlinear iteration traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


GATES = {"G12": 1.0e-12, "G10": 1.0e-10, "G9": 1.0e-9, "G8": 1.0e-8}
REL_GATE = 1.0e-10


def number(value: str | None, default: float = math.nan) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def integer(value: str | None, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def split_segments(
    nonlinear_path: Path, wanted_steps: set[int]
) -> dict[int, list[dict[str, object]]]:
    segments: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    current: dict[str, object] | None = None
    current_step: int | None = None
    previous_iteration: int | None = None

    def finish() -> None:
        nonlocal current
        if current is not None:
            segments[int(current["step"])].append(current)
        current = None

    with nonlinear_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            step = int(row["step"])
            if step not in wanted_steps:
                continue
            iteration = int(row["iteration"])
            if current_step != step or (
                previous_iteration is not None and iteration < previous_iteration
            ):
                finish()
                current_step = step
                current = {"step": step, "rows": []}
            assert current is not None
            current["rows"].append({
                "iteration": iteration,
                "lambda": float(row["line_search_lambda"]),
                "residual": float(row["res_inf"]),
                "accepted": int(row["accepted"]),
            })
            previous_iteration = iteration
    finish()
    return dict(segments)


def map_segments(
    events: list[dict[str, str]], segments: dict[int, list[dict[str, object]]]
) -> list[tuple[dict[str, str], dict[str, object]]]:
    by_step: defaultdict[int, list[dict[str, str]]] = defaultdict(list)
    for event in events:
        by_step[int(event["macro_step"])].append(event)
    mapped: list[tuple[dict[str, str], dict[str, object]]] = []
    for step, step_events in by_step.items():
        candidates = segments.get(step, [])
        cursor = 0
        for event in step_events:
            category = event["category"]
            reported = int(event["nonlinear_iterations"])
            selected: tuple[int, dict[str, object]] | None = None
            for index in range(cursor, len(candidates)):
                segment = candidates[index]
                rows = segment["rows"]
                max_iteration = max(int(row["iteration"]) for row in rows)
                last_accepted = int(rows[-1]["accepted"])
                if category.startswith("B_"):
                    matches = max_iteration == reported + 1
                elif category.startswith("C_"):
                    matches = max_iteration == reported + 1 and last_accepted == 0
                else:
                    matches = max_iteration in (reported, reported + 1)
                if matches:
                    selected = (index, segment)
                    break
            if selected is None:
                raise RuntimeError(
                    f"cannot map {event['run']} reject {event['reject_index']} "
                    f"at step {step}"
                )
            cursor = selected[0] + 1
            mapped.append((event, selected[1]))
    return mapped


def replay_row(
    event: dict[str, str], segment: dict[str, object], gate_id: str, gate: float
) -> dict[str, object]:
    rows = list(segment["rows"])
    accepted_rows = [row for row in rows if int(row["accepted"]) == 1]
    initial_residual = (
        float(accepted_rows[0]["residual"])
        if accepted_rows
        else number(event.get("nonlinear_first_accepted_residual"))
    )
    effective_gate = gate + REL_GATE * initial_residual
    crossing = next(
        (
            (index, row)
            for index, row in enumerate(rows, 1)
            if int(row["accepted"]) == 1
            and float(row["residual"]) <= effective_gate
        ),
        None,
    )
    method_residual = number(event.get("method_transport_residual"))
    if gate_id == "G12":
        # The strict run is the executable oracle. Do not reinterpret condensed
        # outer-context bookkeeping as a counterfactual strict acceptance.
        crossing = None
    elif crossing is None and math.isfinite(method_residual):
        if method_residual <= effective_gate:
            crossing = (len(rows), accepted_rows[-1] if accepted_rows else rows[-1])

    if crossing is not None:
        trials, accepted = crossing
        replay_status = "CONVERGED_AT_RELAXED_GATE"
        classification = "RESIDUAL_FLOOR_ONLY"
        iterations = int(accepted["iteration"])
        final_residual = float(accepted["residual"])
        fallback = False
        iteration_limit = False
        stagnation = False
    else:
        replay_status = "REJECTED_AS_FROZEN"
        category = event["category"]
        if category.startswith("C_"):
            classification = "LINE_SEARCH_DIRECTION_FAILURE"
        elif category.startswith("B_"):
            classification = "ITERATION_LIMIT_WITHOUT_GATE_CROSSING"
        elif "ACTIVE" in category:
            classification = "ACTIVE_SET_REPEATED_FAILURE"
        else:
            classification = "OTHER"
        iterations = int(event["nonlinear_iterations"])
        final_residual = number(event.get("reject_residual_Linf"))
        fallback = True
        iteration_limit = event["iteration_limit_status"] == "HIT"
        stagnation = event["stagnation_status"] == "MIN_LAMBDA_EXHAUSTED"
        trials = len(rows)

    used = rows[:trials]
    return {
        "run": event["run"],
        "reject_index": event["reject_index"],
        "macro_step": event["macro_step"],
        "original_category": event["category"],
        "gate_id": gate_id,
        "requested_gate": gate,
        "relative_gate": REL_GATE,
        "inferred_initial_residual": initial_residual,
        "effective_gate": effective_gate,
        "replay_status": replay_status,
        "classification": classification,
        "final_true_cold_residual": final_residual,
        "nonlinear_iterations": iterations,
        "line_search_trials": len(used),
        "minimum_lambda": min(float(row["lambda"]) for row in used),
        "iteration_limit_hit": int(iteration_limit),
        "stagnation_status": int(stagnation),
        "fallback_triggered": int(fallback),
        "accepted_endpoint_difference": "NOT_AVAILABLE_NO_ITERATE_FIELD_IN_FROZEN_TRACE",
        "wall_time": "NOT_RECORDED_PER_FROZEN_REJECT",
        "evidence_mode": "FROZEN_NONLINEAR_TRACE_CONTROL_FLOW_REPLAY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--nonlinear", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.catalog.open(newline="", encoding="utf-8") as stream:
        events = list(csv.DictReader(stream))
    segments = split_segments(
        args.nonlinear, {int(event["macro_step"]) for event in events}
    )
    mapped = map_segments(events, segments)
    rows = [
        replay_row(event, segment, gate_id, gate)
        for event, segment in mapped
        for gate_id, gate in GATES.items()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "events": len(events),
        "replay_rows": len(rows),
        "relaxed_converged": sum(
            row["replay_status"] == "CONVERGED_AT_RELAXED_GATE" for row in rows
        ),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
