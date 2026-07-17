#!/usr/bin/env python3
"""Frozen decision contract for physically normalized transport defect V2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


CONTRACT_NAME = "PHYSICALLY_NORMALIZED_TRANSPORT_DEFECT_GATE_V2"
ETA_GLOBAL_LIMIT = 1.0e-4
ETA_INTERFACE_LIMIT = 1.0e-3
ETA_CELL_MAX_LIMIT = 1.0e-3
SIGNED_BIAS_LIMIT = 1.0e-3
SIGNED_GROWTH_ABSOLUTE_FLOOR = 1.0e-4
SIGNED_GROWTH_FACTOR = 4.0
MATERIAL_CELL_RELATIVE_CUTOFF = 1.0e-6
INTERFACE_H_LOWER = 1.0e-4
INTERFACE_H_UPPER = 1.0 - INTERFACE_H_LOWER
V1_LOCAL_PEAK_LIMIT = 2.0


@dataclass(frozen=True)
class GateEvaluation:
    global_defect_pass: bool
    interface_defect_pass: bool
    material_cell_defect_pass: bool
    signed_bias_pass: bool
    signed_growth_pass: bool
    qoi_pass: bool
    numerical_contract_pass: bool
    holdout_pass: bool
    normalized_hard_gates_pass: bool
    peak_localization_class: str
    v2_status: str


def number(row: Mapping[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan
    return value


def finite_le(value: float, limit: float) -> bool:
    return math.isfinite(value) and value <= limit


def three_consecutive_growth_failure(values: Sequence[float]) -> bool:
    """Apply the preregistered absolute-plus-relative signed-bias trend rule."""
    if len(values) < 4 or not all(math.isfinite(value) for value in values):
        return False
    has_three_consecutive_increases = any(
        values[start] < values[start + 1] < values[start + 2] < values[start + 3]
        for start in range(len(values) - 3)
    )
    if not has_three_consecutive_increases:
        return False
    if values[-1] <= SIGNED_GROWTH_ABSOLUTE_FLOOR:
        return False
    denominator = max(values[0], math.ulp(1.0))
    return values[-1] / denominator > SIGNED_GROWTH_FACTOR


def peak_localization_class(g_d_floor: float) -> str:
    if not math.isfinite(g_d_floor):
        return "PEAK_LOCALIZATION_UNAVAILABLE"
    if g_d_floor <= 2.0:
        return "PEAK_LOCALIZATION_STABLE"
    if g_d_floor <= 5.0:
        return "PEAK_LOCALIZATION_INCREASED"
    if g_d_floor <= 10.0:
        return "PEAK_LOCALIZATION_WARNING"
    return "STRONG_PEAK_LOCALIZATION_WARNING"


def strict_reference_peak_floor(
    reference_window_rates: Iterable[float],
    machine_roundoff_accumulation_estimate: float,
    window_time: float,
) -> dict[str, float]:
    rates = sorted(float(value) for value in reference_window_rates)
    if not rates or not all(math.isfinite(value) and value >= 0.0 for value in rates):
        raise ValueError("strict reference requires finite nonnegative window rates")
    if not (machine_roundoff_accumulation_estimate >= 0.0):
        raise ValueError("invalid machine roundoff accumulation estimate")
    if not (window_time > 0.0):
        raise ValueError("window_time must be positive")

    def percentile(p: float) -> float:
        position = (len(rates) - 1) * p
        lo = math.floor(position)
        hi = math.ceil(position)
        if lo == hi:
            return rates[lo]
        weight = position - lo
        return rates[lo] * (1.0 - weight) + rates[hi] * weight

    median = percentile(0.5)
    p95 = percentile(0.95)
    maximum = rates[-1]
    roundoff_rate = 100.0 * machine_roundoff_accumulation_estimate / window_time
    return {
        "r_peak_ref_median": median,
        "r_peak_ref_p95": p95,
        "r_peak_ref_max": maximum,
        "roundoff_rate_floor": roundoff_rate,
        "r_peak_floor": max(p95, roundoff_rate),
    }


def evaluate_case(
    window_rows: Sequence[Mapping[str, object]],
    full_row: Mapping[str, object],
    *,
    g_d_floor: float,
    qoi_pass: bool,
    numerical_contract_pass: bool,
    holdout_window_index: int,
) -> GateEvaluation:
    if not window_rows:
        raise ValueError("at least one complete equal-time window is required")
    all_rows = list(window_rows) + [full_row]
    global_pass = all(
        finite_le(number(row, "eta_global"), ETA_GLOBAL_LIMIT)
        for row in all_rows
    )
    interface_pass = all(
        finite_le(number(row, "eta_interface"), ETA_INTERFACE_LIMIT)
        for row in all_rows
    )
    cell_pass = all(
        finite_le(number(row, "eta_cell_max"), ETA_CELL_MAX_LIMIT)
        for row in window_rows
    )
    signed_pass = all(
        finite_le(number(row, "b_signed"), SIGNED_BIAS_LIMIT)
        and finite_le(number(row, "b_interface"), SIGNED_BIAS_LIMIT)
        for row in all_rows
    )
    signed_growth_pass = not (
        three_consecutive_growth_failure(
            [number(row, "b_signed") for row in window_rows]
        )
        or three_consecutive_growth_failure(
            [number(row, "b_interface") for row in window_rows]
        )
    )
    holdout_rows = [
        row
        for row in window_rows
        if int(number(row, "window_index")) == holdout_window_index
    ]
    holdout_pass = len(holdout_rows) == 1 and all(
        (
            finite_le(number(row, "eta_global"), ETA_GLOBAL_LIMIT)
            and finite_le(number(row, "eta_interface"), ETA_INTERFACE_LIMIT)
            and finite_le(number(row, "eta_cell_max"), ETA_CELL_MAX_LIMIT)
            and finite_le(number(row, "b_signed"), SIGNED_BIAS_LIMIT)
            and finite_le(number(row, "b_interface"), SIGNED_BIAS_LIMIT)
        )
        for row in holdout_rows
    )
    normalized_pass = all(
        (
            global_pass,
            interface_pass,
            cell_pass,
            signed_pass,
            signed_growth_pass,
            qoi_pass,
            numerical_contract_pass,
            holdout_pass,
        )
    )
    peak_class = peak_localization_class(g_d_floor)
    if not normalized_pass:
        status = "FAIL_V2_MATERIAL_TRANSPORT_DEFECT"
    elif g_d_floor > 2.0:
        status = "PASS_V2_WITH_LOCAL_PEAK_WARNING"
    else:
        status = "PASS_PHYSICALLY_NORMALIZED_DEFECT_GATE_V2"
    return GateEvaluation(
        global_defect_pass=global_pass,
        interface_defect_pass=interface_pass,
        material_cell_defect_pass=cell_pass,
        signed_bias_pass=signed_pass,
        signed_growth_pass=signed_growth_pass,
        qoi_pass=qoi_pass,
        numerical_contract_pass=numerical_contract_pass,
        holdout_pass=holdout_pass,
        normalized_hard_gates_pass=normalized_pass,
        peak_localization_class=peak_class,
        v2_status=status,
    )
