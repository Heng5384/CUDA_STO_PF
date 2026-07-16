#!/usr/bin/env python3
"""Compare staggered step655 candidates over one equal physical-time window.

The fine accepted trajectory is the numerical reference.  Errors are reported
both on the absolute field scale and relative to the reference evolution
increment.  The latter prevents a nearly stationary field from looking
accurate merely because its background value is large.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


FIELDS = {
    "Ctot": "_Ctot.raw",
    "phi": "_phi.raw",
    "xB_alpha": "_xB_alpha.raw",
}


def h_phi(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def parse_shape(text: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in text.replace("x", ",").split(","))
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"invalid grid shape: {text}")
    return values


def parse_candidate(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError("candidate must have LABEL=PREFIX form")
    label, prefix = text.split("=", 1)
    if not label or not prefix:
        raise ValueError("candidate must have LABEL=PREFIX form")
    return label, Path(prefix)


def load_state(prefix: Path, shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    expected = math.prod(shape)
    state: dict[str, np.ndarray] = {}
    for name, suffix in FIELDS.items():
        path = Path(str(prefix) + suffix)
        values = np.fromfile(path, dtype=np.float64)
        if values.size != expected:
            raise ValueError(f"{path}: expected {expected} values, found {values.size}")
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite value")
        state[name] = values.reshape(shape)
    return state


def load_frozen_csv(path: Path, shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    expected = math.prod(shape)
    columns: dict[str, list[float]] = {
        "Ctot": [],
        "phi": [],
        "xB_alpha": [],
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"C_trial", "phi", "xalpha"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            columns["Ctot"].append(float(row["C_trial"]))
            columns["phi"].append(float(row["phi"]))
            columns["xB_alpha"].append(float(row["xalpha"]))
    if any(len(values) != expected for values in columns.values()):
        raise ValueError(f"{path}: expected {expected} rows")
    state = {
        name: np.asarray(values, dtype=np.float64).reshape(shape)
        for name, values in columns.items()
    }
    if any(not np.isfinite(values).all() for values in state.values()):
        raise ValueError(f"{path}: non-finite value")
    return state


def load_split_metrics(prefix: Path) -> dict[str, object]:
    path = prefix.parent / "ctot_split_step_metrics.csv"
    result: dict[str, object] = {
        "runtime_metrics_available": False,
        "runtime_step_rows": 0,
        "split_eta_min": math.nan,
        "split_eta_max": math.nan,
        "polish_applied_count": 0,
        "polish_skipped_count": 0,
        "transport_residual_max": math.nan,
        "phase_KKT_max": math.nan,
        "mass_error_abs_max": math.nan,
        "runtime_all_accepted": False,
    }
    if not path.is_file():
        return result
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return result
    eta = [float(row["split_defect_eta"]) for row in rows]
    result.update(
        {
            "runtime_metrics_available": True,
            "runtime_step_rows": len(rows),
            "split_eta_min": min(eta),
            "split_eta_max": max(eta),
            "polish_applied_count": sum(int(row["polish_applied"]) for row in rows),
            "polish_skipped_count": sum(int(row["polish_skipped"]) for row in rows),
            "transport_residual_max": max(float(row["transport_residual"]) for row in rows),
            "phase_KKT_max": max(float(row["phase_KKT"]) for row in rows),
            "mass_error_abs_max": max(abs(float(row["mass_error"])) for row in rows),
            "runtime_all_accepted": all(int(row["accepted"]) == 1 for row in rows),
        }
    )
    return result


def norm_metrics(error: np.ndarray, reference_increment: np.ndarray) -> dict[str, float]:
    flat_error = error.ravel()
    flat_increment = reference_increment.ravel()
    error_linf = float(np.max(np.abs(flat_error)))
    error_l2 = float(np.linalg.norm(flat_error))
    error_rms = float(np.sqrt(np.mean(flat_error**2)))
    increment_linf = float(np.max(np.abs(flat_increment)))
    increment_l2 = float(np.linalg.norm(flat_increment))
    increment_rms = float(np.sqrt(np.mean(flat_increment**2)))
    floor = np.finfo(np.float64).tiny
    return {
        "abs_linf": error_linf,
        "abs_l2": error_l2,
        "abs_rms": error_rms,
        "reference_increment_linf": increment_linf,
        "reference_increment_l2": increment_l2,
        "reference_increment_rms": increment_rms,
        "increment_error_linf_rel": error_linf / max(increment_linf, floor),
        "increment_error_l2_rel": error_l2 / max(increment_l2, floor),
        "increment_error_rms_rel": error_rms / max(increment_rms, floor),
    }


def direction_cosine(candidate_increment: np.ndarray, reference_increment: np.ndarray) -> float:
    candidate = candidate_increment.ravel()
    reference = reference_increment.ravel()
    denominator = float(np.linalg.norm(candidate) * np.linalg.norm(reference))
    if denominator == 0.0:
        return 1.0 if np.array_equal(candidate, reference) else math.nan
    return float(np.dot(candidate, reference) / denominator)


def line_profile(phi: np.ndarray) -> np.ndarray:
    return phi.mean(axis=(1, 2))


def threshold_crossings(profile: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    crossings: list[float] = []
    size = profile.size
    for index in range(size):
        next_index = (index + 1) % size
        left = float(profile[index] - threshold)
        right = float(profile[next_index] - threshold)
        if left == 0.0:
            crossings.append(float(index))
        elif left * right < 0.0:
            fraction = -left / (right - left)
            crossings.append((index + fraction) % size)
    return np.asarray(sorted(crossings), dtype=np.float64)


def periodic_distance(left: float, right: float, period: float) -> float:
    delta = abs(left - right)
    return min(delta, period - delta)


def crossing_metrics(
    initial_phi: np.ndarray,
    reference_phi: np.ndarray,
    candidate_phi: np.ndarray,
    dx: float,
) -> dict[str, float | int]:
    initial = threshold_crossings(line_profile(initial_phi))
    reference = threshold_crossings(line_profile(reference_phi))
    candidate = threshold_crossings(line_profile(candidate_phi))
    result: dict[str, float | int] = {
        "initial_crossing_count": int(initial.size),
        "reference_crossing_count": int(reference.size),
        "candidate_crossing_count": int(candidate.size),
        "interface_shift_error_max": math.nan,
        "reference_interface_motion_max": math.nan,
        "interface_motion_error_rel": math.nan,
    }
    if not (initial.size == reference.size == candidate.size) or reference.size == 0:
        return result
    period = float(initial_phi.shape[0])
    shift_errors = [
        periodic_distance(float(c), float(r), period) * dx
        for c, r in zip(candidate, reference)
    ]
    reference_motion = [
        periodic_distance(float(r), float(i), period) * dx
        for i, r in zip(initial, reference)
    ]
    max_shift_error = max(shift_errors)
    max_reference_motion = max(reference_motion)
    result.update(
        {
            "interface_shift_error_max": max_shift_error,
            "reference_interface_motion_max": max_reference_motion,
            "interface_motion_error_rel": (
                max_shift_error / max(max_reference_motion, np.finfo(np.float64).tiny)
            ),
        }
    )
    return result


def compare_candidate(
    label: str,
    initial: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    dx: float,
    qoi_tolerance: float,
    candidate_role: str = "optional_skip_candidate",
    runtime_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {"case": label, "candidate_role": candidate_role}
    if runtime_metrics is None:
        runtime_metrics = {
            "runtime_metrics_available": False,
            "runtime_step_rows": 0,
            "split_eta_min": math.nan,
            "split_eta_max": math.nan,
            "polish_applied_count": 0,
            "polish_skipped_count": 0,
            "transport_residual_max": math.nan,
            "phase_KKT_max": math.nan,
            "mass_error_abs_max": math.nan,
            "runtime_all_accepted": False,
        }
    row.update(runtime_metrics)
    for field in FIELDS:
        error = candidate[field] - reference[field]
        reference_increment = reference[field] - initial[field]
        candidate_increment = candidate[field] - initial[field]
        for metric, value in norm_metrics(error, reference_increment).items():
            row[f"{field}_{metric}"] = value
        row[f"{field}_increment_direction_cosine"] = direction_cosine(
            candidate_increment, reference_increment
        )

    h_initial = h_phi(initial["phi"])
    h_reference = h_phi(reference["phi"])
    h_candidate = h_phi(candidate["phi"])
    reference_h_delta = float(np.sum(h_reference - h_initial))
    candidate_h_delta = float(np.sum(h_candidate - h_initial))
    h_delta_error = candidate_h_delta - reference_h_delta
    row.update(
        {
            "Ctot_mass_initial": float(np.sum(initial["Ctot"])),
            "Ctot_mass_reference": float(np.sum(reference["Ctot"])),
            "Ctot_mass_candidate": float(np.sum(candidate["Ctot"])),
            "Ctot_mass_candidate_minus_reference": float(
                np.sum(candidate["Ctot"]) - np.sum(reference["Ctot"])
            ),
            "h_integral_initial": float(np.sum(h_initial)),
            "h_integral_reference": float(np.sum(h_reference)),
            "h_integral_candidate": float(np.sum(h_candidate)),
            "h_delta_reference": reference_h_delta,
            "h_delta_candidate": candidate_h_delta,
            "h_delta_error_abs": abs(h_delta_error),
            "h_delta_error_rel": abs(h_delta_error)
            / max(abs(reference_h_delta), np.finfo(np.float64).tiny),
            "phase_volume_direction_unchanged": (
                reference_h_delta == 0.0
                and candidate_h_delta == 0.0
            )
            or (reference_h_delta * candidate_h_delta > 0.0),
        }
    )
    row.update(crossing_metrics(
        initial["phi"], reference["phi"], candidate["phi"], dx
    ))

    primary_errors = [
        float(row["Ctot_increment_error_l2_rel"]),
        float(row["phi_increment_error_l2_rel"]),
        float(row["h_delta_error_rel"]),
    ]
    row["max_primary_increment_qoi_error_rel"] = max(primary_errors)
    row["optional_skip_qoi_eligible"] = bool(
        candidate_role == "optional_skip_candidate"
        and
        max(primary_errors) <= qoi_tolerance
        and row["phase_volume_direction_unchanged"]
        and int(row["candidate_crossing_count"]) == int(row["reference_crossing_count"])
    )
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-prefix", type=Path, required=True)
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--candidate-csv", action="append", default=[])
    parser.add_argument("--shape", default="512,1,1")
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--qoi-tolerance", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    shape = parse_shape(args.shape)
    initial = load_state(args.initial_prefix, shape)
    reference = load_state(args.reference_prefix, shape)
    rows = []
    for candidate_text in args.candidate:
        label, prefix = parse_candidate(candidate_text)
        rows.append(compare_candidate(
            label, initial, reference, load_state(prefix, shape),
            args.dx, args.qoi_tolerance,
            runtime_metrics=load_split_metrics(prefix),
        ))
    for candidate_text in args.candidate_csv:
        label, path = parse_candidate(candidate_text)
        rows.append(compare_candidate(
            label, initial, reference, load_frozen_csv(path, shape),
            args.dx, args.qoi_tolerance,
            candidate_role="rejected_coupled_comparator",
        ))
    write_csv(args.output, rows)

    eligible = [str(row["case"]) for row in rows if row["optional_skip_qoi_eligible"]]
    summary = {
        "reference_prefix": str(args.reference_prefix),
        "initial_prefix": str(args.initial_prefix),
        "qoi_tolerance": args.qoi_tolerance,
        "eligible_cases": eligible,
        "all_cases": [str(row["case"]) for row in rows],
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"analyzed_cases={len(rows)}")
    print(f"eligible_cases={','.join(eligible) if eligible else 'none'}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
