#!/usr/bin/env python3
"""Frozen physically scaled signed transport-defect contract (V3)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


CONTRACT_NAME = "PHYSICALLY_SCALED_SIGNED_TRANSPORT_DEFECT_GATE_V3"
ETA_GLOBAL_LIMIT = 1.0e-4
ETA_INTERFACE_LIMIT = 1.0e-3
ETA_CELL_MAX_LIMIT = 1.0e-3
BETA_GLOBAL_LIMIT = 1.0e-5
BETA_INTERFACE_LIMIT = 1.0e-4
BETA_INTERFACE_EXCESS_LIMIT = 1.0e-4
INTERFACE_H_EPS = 1.0e-4
SIGN_COHERENCE_MATERIAL_ETA_MIN = 1.0e-6


@dataclass(frozen=True)
class V3Evaluation:
    status: str
    magnitude_pass: bool
    signed_pass: bool
    all_records_pass: bool
    holdout_pass: bool
    qoi_pass: bool
    numerical_pass: bool


def number(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def finite_le(value: float, limit: float) -> bool:
    return math.isfinite(value) and value <= limit


def metric_record_pass(row: dict[str, Any], *, require_excess: bool) -> bool:
    checks = (
        finite_le(number(row, "eta_global"), ETA_GLOBAL_LIMIT),
        finite_le(number(row, "eta_interface"), ETA_INTERFACE_LIMIT),
        finite_le(number(row, "eta_cell_max"), ETA_CELL_MAX_LIMIT),
        finite_le(number(row, "beta_global"), BETA_GLOBAL_LIMIT),
        finite_le(number(row, "beta_interface"), BETA_INTERFACE_LIMIT),
    )
    if require_excess:
        checks += (
            finite_le(
                number(row, "beta_interface_excess"),
                BETA_INTERFACE_EXCESS_LIMIT,
            ),
        )
    return all(checks)


def evaluate_case(
    records: Iterable[dict[str, Any]],
    *,
    holdout_window_index: int,
    require_excess: bool,
    qoi_pass: bool,
    numerical_pass: bool,
) -> V3Evaluation:
    rows = list(records)
    all_records_pass = bool(rows) and all(
        metric_record_pass(row, require_excess=require_excess) for row in rows
    )
    holdout = [
        row
        for row in rows
        if row.get("record_type") == "WINDOW"
        and int(number(row, "window_index")) == holdout_window_index
    ]
    holdout_pass = len(holdout) == 1 and metric_record_pass(
        holdout[0], require_excess=require_excess
    )
    magnitude_pass = bool(rows) and all(
        finite_le(number(row, "eta_global"), ETA_GLOBAL_LIMIT)
        and finite_le(number(row, "eta_interface"), ETA_INTERFACE_LIMIT)
        and finite_le(number(row, "eta_cell_max"), ETA_CELL_MAX_LIMIT)
        for row in rows
    )
    signed_pass = bool(rows) and all(
        finite_le(number(row, "beta_global"), BETA_GLOBAL_LIMIT)
        and finite_le(number(row, "beta_interface"), BETA_INTERFACE_LIMIT)
        and (
            not require_excess
            or finite_le(
                number(row, "beta_interface_excess"),
                BETA_INTERFACE_EXCESS_LIMIT,
            )
        )
        for row in rows
    )
    passed = all_records_pass and holdout_pass and qoi_pass and numerical_pass
    return V3Evaluation(
        status=(
            "PASS_PHYSICALLY_SCALED_SIGNED_DEFECT_GATE_V3"
            if passed
            else "FAIL_V3_PHYSICALLY_SCALED_SIGNED_DEFECT"
        ),
        magnitude_pass=magnitude_pass,
        signed_pass=signed_pass,
        all_records_pass=all_records_pass,
        holdout_pass=holdout_pass,
        qoi_pass=qoi_pass,
        numerical_pass=numerical_pass,
    )


def sign_coherence_interpretation(eta_interface: float) -> str:
    if not math.isfinite(eta_interface) or eta_interface <= SIGN_COHERENCE_MATERIAL_ETA_MIN:
        return "SIGN_COHERENCE_NOT_MATERIALLY_INTERPRETABLE"
    return "SIGN_COHERENCE_MATERIALLY_INTERPRETABLE"
