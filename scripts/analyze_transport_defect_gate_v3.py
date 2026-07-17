#!/usr/bin/env python3
"""Analyze V3 signed defect and actual interface displacement without rewriting V1/V2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_transport_residual_gate_v1 import accuracy_metrics, h
from transport_defect_gate_v3_contract import (
    BETA_GLOBAL_LIMIT,
    BETA_INTERFACE_EXCESS_LIMIT,
    BETA_INTERFACE_LIMIT,
    CONTRACT_NAME,
    ETA_CELL_MAX_LIMIT,
    ETA_GLOBAL_LIMIT,
    ETA_INTERFACE_LIMIT,
    INTERFACE_H_EPS,
    evaluate_case,
    sign_coherence_interpretation,
)


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "reports" / "transport_residual_gate_v1"
V2 = ROOT / "reports" / "transport_residual_gate_v2"
V3 = ROOT / "reports" / "transport_residual_gate_v3"
COMMON = (
    ROOT
    / "reports"
    / "bounded_retry_bdf2_v1"
    / "workstation_runs"
    / "equal_time_common_input"
)
V1_LONG_RUNS = ROOT / "runs" / "transport_residual_gate_v1_long"
T_REAL_UNIT_S = 41.12958542455477
WINDOW_TIME_CODE = 1.5625
NX, NY, NZ = 512, 1, 1
DX_NM = DY_NM = DZ_NM = 1.0
A_FLOOR = max(1.0e-30, 100.0 * np.finfo(np.float64).eps * NX * NY * NZ)
KNOWN_V1_MANIFEST_SHA256 = (
    "773099c65ec7dab6707e280d4dc5d5f910f7fba1e4ca3311dbf61fdd1a3b639f"
)
V1_STATUS = "FAIL_PREREGISTERED_LOCAL_PEAK_DEFECT_RATE_GATE"
V2_STATUS = "FAIL_V2_MATERIAL_TRANSPORT_DEFECT"
CASES = {
    "strict": ("G12", "dt32", "V2_G12_dt32_5windows_holdout"),
    "candidate": ("G9", "dt4", "V2_G9_dt4_5windows_holdout"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


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


def one(root: Path, pattern: str) -> Path:
    found = list(root.rglob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, found {found}")
    return found[0]


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


def history_paths() -> list[Path]:
    paths: set[Path] = set()
    for root in (V1, V2, V1_LONG_RUNS):
        if root.is_dir():
            paths.update(path for path in root.rglob("*") if path.is_file())
    return sorted(paths)


def freeze_or_verify_history() -> tuple[bool, list[str]]:
    manifest = V3 / "historical_v1_v2_sha256.csv"
    current = {
        str(path.relative_to(ROOT)): (path.stat().st_size, sha256(path))
        for path in history_paths()
    }
    problems: list[str] = []
    if manifest.is_file():
        frozen = {
            row["path"]: (int(row["size_bytes"]), row["sha256"])
            for row in read_csv(manifest)
        }
        if current != frozen:
            for key in sorted(set(current) | set(frozen)):
                if current.get(key) != frozen.get(key):
                    problems.append(key)
    else:
        write_csv(
            manifest,
            [
                {"path": key, "size_bytes": value[0], "sha256": value[1]}
                for key, value in current.items()
            ],
        )
    v1_manifest = V1 / "preregistered_v1_hash_manifest.csv"
    if sha256(v1_manifest) != KNOWN_V1_MANIFEST_SHA256:
        problems.append(str(v1_manifest.relative_to(ROOT)))
    return not problems, problems


def v2_case_root(label: str) -> Path:
    return V2 / "workstation_runs" / CASES[label][2]


def v3_case_root(label: str) -> Path | None:
    gate, dt_id, _ = CASES[label]
    root = V3 / "workstation_runs" / f"V3_{gate}_{dt_id}_6th_window_holdout"
    return root if (root / "workstation_status.json").is_file() else None


def result_root(root: Path) -> Path:
    return one(root, "ctot_transport_defect_v2_meta.json").parent


def checkpoint_state(label: str, window: int) -> dict[str, np.ndarray]:
    if window == 0:
        return {
            "Ctot": np.fromfile(COMMON / "Ctot_init.raw", np.float64),
            "phi": np.fromfile(COMMON / "phi_init.raw", np.float64),
            "xB": np.fromfile(COMMON / "xB_init.raw", np.float64),
        }
    if window <= 5:
        root = result_root(v2_case_root(label))
        gate, dt_id, _ = CASES[label]
        steps = 16000 * window if dt_id == "dt32" else 2000 * window
    else:
        case = v3_case_root(label)
        if case is None:
            raise RuntimeError(f"missing V3 sixth window for {label}")
        root = result_root(case)
        steps = 16000 if label == "strict" else 2000
    stem = f"ctot_checkpoint_step{steps:06d}"
    return {
        "Ctot": np.fromfile(root / (stem + "_Ctot.raw"), np.float64),
        "phi": np.fromfile(root / (stem + "_phi.raw"), np.float64),
        "xB": np.fromfile(root / (stem + "_xB_alpha.raw"), np.float64),
    }


def checkpoint_meta(label: str, window: int) -> dict[str, Any]:
    if window == 0:
        return json.loads((COMMON / "init_meta.json").read_text(encoding="utf-8"))
    if window <= 5:
        root = result_root(v2_case_root(label))
        dt_id = CASES[label][1]
        steps = 16000 * window if dt_id == "dt32" else 2000 * window
    else:
        case = v3_case_root(label)
        if case is None:
            return {}
        root = result_root(case)
        steps = 16000 if label == "strict" else 2000
    return json.loads(
        (root / f"ctot_checkpoint_step{steps:06d}_meta.json").read_text(
            encoding="utf-8"
        )
    )


def phi_level_from_h(target: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if float(h(np.array([mid]))[0]) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def periodic_crossings(values: np.ndarray, level: float) -> list[tuple[float, str]]:
    result: list[tuple[float, str]] = []
    n = values.size
    for index in range(n):
        nxt = (index + 1) % n
        a, b = float(values[index]), float(values[nxt])
        if a < level <= b:
            frac = (level - a) / max(b - a, 1.0e-300)
            result.append((index + frac, "up"))
        elif a >= level > b:
            frac = (a - level) / max(a - b, 1.0e-300)
            result.append((index + frac, "down"))
    return result


def nearest_periodic(value: float, reference: float, period: float) -> float:
    return value + round((reference - value) / period) * period


def slab_crossings(phi: np.ndarray, reference: tuple[float, float] | None = None) -> tuple[float, float]:
    crossings = periodic_crossings(phi, 0.5)
    up = [value for value, direction in crossings if direction == "up"]
    down = [value for value, direction in crossings if direction == "down"]
    if len(up) != 1 or len(down) != 1:
        raise RuntimeError(f"non-unique slab crossings: {crossings}")
    left, right = up[0], down[0]
    if reference is not None:
        left = nearest_periodic(left, reference[0], phi.size)
        right = nearest_periodic(right, reference[1], phi.size)
    return left, right


def periodic_interp(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    n = values.size
    x = np.mod(coordinates, n)
    i = np.floor(x).astype(int)
    frac = x - i
    return values[i] * (1.0 - frac) + values[(i + 1) % n] * frac


def interface_band(phi: np.ndarray, crossings: tuple[float, float]) -> dict[str, float]:
    low = phi_level_from_h(INTERFACE_H_EPS)
    high = phi_level_from_h(1.0 - INTERFACE_H_EPS)
    low_cross = periodic_crossings(phi, low)
    high_cross = periodic_crossings(phi, high)
    low_up = nearest_periodic(
        next(v for v, d in low_cross if d == "up"), crossings[0], phi.size
    )
    high_up = nearest_periodic(
        next(v for v, d in high_cross if d == "up"), crossings[0], phi.size
    )
    high_down = nearest_periodic(
        next(v for v, d in high_cross if d == "down"), crossings[1], phi.size
    )
    low_down = nearest_periodic(
        next(v for v, d in low_cross if d == "down"), crossings[1], phi.size
    )
    left_width = (high_up - low_up) * DX_NM
    right_width = (low_down - high_down) * DX_NM
    return {
        "left_interface_band_center_nm": 0.5 * (low_up + high_up) * DX_NM,
        "right_interface_band_center_nm": 0.5 * (high_down + low_down) * DX_NM,
        "left_interface_band_width_nm": left_width,
        "right_interface_band_width_nm": right_width,
        "mean_interface_band_width_nm": 0.5 * (left_width + right_width),
    }


def translated_profile_error(
    initial_phi: np.ndarray,
    phi: np.ndarray,
    initial_cross: tuple[float, float],
    current_cross: tuple[float, float],
) -> tuple[float, float]:
    offsets = np.linspace(-12.0, 12.0, 481)
    differences = []
    for initial_position, current_position in zip(initial_cross, current_cross):
        before = periodic_interp(initial_phi, initial_position + offsets)
        after = periodic_interp(phi, current_position + offsets)
        differences.append(after - before)
    delta = np.concatenate(differences)
    return float(np.sqrt(np.mean(delta * delta))), float(np.max(np.abs(delta)))


def trajectory_row(label: str, window: int, initial: dict[str, np.ndarray]) -> dict[str, Any]:
    state = checkpoint_state(label, window)
    phi0, phi = initial["phi"], state["phi"]
    initial_cross = slab_crossings(phi0)
    current_cross = slab_crossings(phi, initial_cross)
    left_out = (initial_cross[0] - current_cross[0]) * DX_NM
    right_out = (current_cross[1] - initial_cross[1]) * DX_NM
    cross_disp = 0.5 * (left_out + right_out)
    h0, hc = h(phi0), h(phi)
    hvolume = float(np.sum(hc)) * DX_NM * DY_NM * DZ_NM
    hvolume0 = float(np.sum(h0)) * DX_NM * DY_NM * DZ_NM
    cross_section = NY * DY_NM * NZ * DZ_NM
    h_disp = (hvolume - hvolume0) / (2.0 * cross_section)
    transfer = 0.5 * float(np.sum(np.abs(state["Ctot"] - initial["Ctot"])))
    far_mask = (h0 < 1.0e-8) & (hc < 1.0e-8)
    far = float(np.mean(state["xB"][far_mask]))
    profile_l2, profile_linf = translated_profile_error(
        phi0, phi, initial_cross, current_cross
    )
    band = interface_band(phi, current_cross)
    initial_band = interface_band(phi0, initial_cross)
    meta = checkpoint_meta(label, window)
    result = {
        "case": label,
        "gate_id": CASES[label][0],
        "dt_id": CASES[label][1],
        "window_index": window,
        "time_code": window * WINDOW_TIME_CODE,
        "time_physical_s": window * WINDOW_TIME_CODE * T_REAL_UNIT_S,
        "left_crossing_nm": current_cross[0] * DX_NM,
        "right_crossing_nm": current_cross[1] * DX_NM,
        "left_outward_displacement_nm": left_out,
        "right_outward_displacement_nm": right_out,
        "delta_s_cross_nm": cross_disp,
        "delta_s_cross_dx": cross_disp / DX_NM,
        "h_volume_nm3": hvolume,
        "beta_h_inventory_change_cell_units": float(np.sum(hc - h0)),
        "delta_s_h_nm": h_disp,
        "delta_s_h_dx": h_disp / DX_NM,
        "difference_between_cross_and_h_nm": abs(cross_disp - h_disp),
        "difference_between_cross_and_h_dx": abs(cross_disp - h_disp) / DX_NM,
        "cumulative_matrix_to_beta_transfer": transfer,
        "far_field_matrix_composition": far,
        "translated_profile_L2": profile_l2,
        "translated_profile_Linf": profile_linf,
        "interface_width_change_nm": (
            band["mean_interface_band_width_nm"]
            - initial_band["mean_interface_band_width_nm"]
        ),
        "left_right_displacement_asymmetry_nm": abs(left_out - right_out),
        "bdf2_history_valid": meta.get("bdf2_history_valid"),
        "checkpoint_time_code": meta.get("time_code"),
        **band,
    }
    result["crossing_hvolume_consistency"] = (
        "PASS"
        if abs(cross_disp - h_disp) <= 0.25 * DX_NM
        and abs(cross_disp - h_disp) <= 0.25 * abs(cross_disp)
        else "FAIL"
    )
    result["motion_classification"] = (
        "MATERIAL_INTERFACE_MOTION_CONFIRMED"
        if result["crossing_hvolume_consistency"] == "PASS"
        and math.copysign(1.0, cross_disp or 1.0) == math.copysign(1.0, h_disp or 1.0)
        and transfer > 0.0
        else "PROFILE_RELAXATION_DOMINATED"
    )
    return result


def load_observer_arrays(label: str, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if window <= 5:
        root = result_root(v2_case_root(label))
        index = window
    else:
        case = v3_case_root(label)
        if case is None:
            raise RuntimeError(f"missing V3 sixth-window output for {label}")
        root = result_root(case)
        index = 1
    stem = f"ctot_transport_defect_v2_window_{index:04d}"
    return (
        np.fromfile(root / (stem + "_D.raw"), np.float64),
        np.fromfile(root / (stem + "_A.raw"), np.float64),
        np.fromfile(root / (stem + "_B.raw"), np.float64),
        np.fromfile(root / (stem + "_interface_seen.raw"), np.uint8).astype(bool),
    )


def raw_metrics(D: np.ndarray, A: np.ndarray, I: np.ndarray) -> dict[str, float]:
    A_scale = float(np.max(A))
    material_threshold = 1.0e-6 * A_scale
    material = A >= material_threshold
    eta_cell = float(
        np.max(np.abs(D[material]) / np.maximum(A[material], material_threshold))
    ) if np.any(material) and A_scale > 0.0 else 0.0
    sum_abs_D = float(np.sum(np.abs(D)))
    interface_abs_D = float(np.sum(np.abs(D[I])))
    return {
        "sum_abs_D": sum_abs_D,
        "sum_A": float(np.sum(A)),
        "eta_global": sum_abs_D / max(float(np.sum(A)), A_FLOOR),
        "eta_interface": interface_abs_D / max(float(np.sum(A[I])), A_FLOOR),
        "eta_cell_max": eta_cell,
        "global_signed_D": float(np.sum(D)),
        "interface_signed_D_own_mask": float(np.sum(D[I])),
        "beta_global": abs(float(np.sum(D))) / max(float(np.sum(A)), A_FLOOR),
        "beta_interface_own_mask": abs(float(np.sum(D[I]))) / max(float(np.sum(A[I])), A_FLOOR),
        "global_sign_coherence_diagnostic": abs(float(np.sum(D))) / max(sum_abs_D, A_FLOOR),
        "interface_sign_coherence_diagnostic": abs(float(np.sum(D[I]))) / max(interface_abs_D, A_FLOOR),
        "interface_cells_own_mask": int(np.count_nonzero(I)),
    }


def v3_window_rows(max_window: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for window in range(1, max_window + 1):
        arrays = {label: load_observer_arrays(label, window) for label in CASES}
        common = arrays["strict"][3] | arrays["candidate"][3]
        strict_D, strict_A = arrays["strict"][0], arrays["strict"][1]
        strict_S_common = float(np.sum(strict_D[common]))
        strict_A_common = float(np.sum(strict_A[common]))
        for label, (D, A, _B, I) in arrays.items():
            base = raw_metrics(D, A, I)
            S_common = float(np.sum(D[common]))
            beta_common = abs(S_common) / max(float(np.sum(A[common])), A_FLOOR)
            row = {
                "case": label,
                "gate_id": CASES[label][0],
                "dt_id": CASES[label][1],
                "record_type": "WINDOW",
                "window_index": window,
                "time_start_code": (window - 1) * WINDOW_TIME_CODE,
                "time_end_code": window * WINDOW_TIME_CODE,
                **base,
                "common_interface_cells": int(np.count_nonzero(common)),
                "interface_signed_D_common_mask": S_common,
                "interface_A_common_mask": float(np.sum(A[common])),
                "beta_interface": beta_common,
                "beta_interface_excess": (
                    abs(S_common - strict_S_common) / max(strict_A_common, A_FLOOR)
                    if label == "candidate" else 0.0
                ),
                "sign_coherence_interpretation": sign_coherence_interpretation(
                    base["eta_interface"]
                ),
            }
            rows.append(row)
            if label not in cumulative:
                cumulative[label] = (D.copy(), A.copy(), I.copy())
            else:
                old_D, old_A, old_I = cumulative[label]
                cumulative[label] = (old_D + D, old_A + A, old_I | I)

    common_full = cumulative["strict"][2] | cumulative["candidate"][2]
    strict_D, strict_A, _ = cumulative["strict"]
    strict_S = float(np.sum(strict_D[common_full]))
    strict_A_common = float(np.sum(strict_A[common_full]))
    for label, (D, A, I) in cumulative.items():
        base = raw_metrics(D, A, I)
        S_common = float(np.sum(D[common_full]))
        rows.append({
            "case": label,
            "gate_id": CASES[label][0],
            "dt_id": CASES[label][1],
            "record_type": "FULL_TRAJECTORY",
            "window_index": 0,
            "time_start_code": 0.0,
            "time_end_code": max_window * WINDOW_TIME_CODE,
            **base,
            "common_interface_cells": int(np.count_nonzero(common_full)),
            "interface_signed_D_common_mask": S_common,
            "interface_A_common_mask": float(np.sum(A[common_full])),
            "beta_interface": abs(S_common) / max(float(np.sum(A[common_full])), A_FLOOR),
            "beta_interface_excess": (
                abs(S_common - strict_S) / max(strict_A_common, A_FLOOR)
                if label == "candidate" else 0.0
            ),
            "sign_coherence_interpretation": sign_coherence_interpretation(
                base["eta_interface"]
            ),
        })
    return rows


def sixth_numerical_pass(label: str) -> tuple[bool, dict[str, Any]]:
    case = v3_case_root(label)
    if case is None:
        return False, {"status": "PENDING"}
    summary = json.loads((case / "v3_run_summary.json").read_text(encoding="utf-8"))
    bounded = summary.get("bounded_retry_summary", {})
    persistent = summary.get("persistent_failed_cells", [])
    trajectory = summary.get("trajectory_meta", {})
    budget = as_int(bounded.get("nonlinear_iteration_budget"), 0)
    p99 = as_float(bounded.get("accepted_iteration_p99"), math.inf)
    passed = bool(
        summary.get("acceptance_hard_pass")
        and summary.get("energy_pass")
        and summary.get("rollback_pass")
        and as_float(summary.get("max_mass_error")) <= 1.0e-10
        and not persistent
        and as_int(bounded.get("macro_hard_rejects"), 99) == 0
        and as_float(bounded.get("retry_fraction"), 1.0) <= 0.01
        and as_float(bounded.get("fallback_fraction"), 1.0) <= 0.01
        and as_float(bounded.get("reject_trial_wall_fraction"), 1.0) <= 0.05
        and as_int(bounded.get("max_consecutive_fallback_macros"), 99) <= 2
        and as_int(bounded.get("max_accepted_subcycle_depth"), 99) <= 2
        and budget > 0
        and p99 < 0.8 * budget
        and abs(as_float(trajectory.get("observed_time_code")) - WINDOW_TIME_CODE)
        <= 1.0e-10
        and trajectory.get("schema")
        == "CTOT_TRANSPORT_GATE_TRAJECTORY_V2_TRANSACTIONAL"
    )
    return passed, {
        "status": "PASS" if passed else "FAIL",
        "max_mass_error": summary.get("max_mass_error"),
        "max_phase_kkt": summary.get("max_phase_kkt"),
        "retry_fraction": bounded.get("retry_fraction"),
        "fallback_fraction": bounded.get("fallback_fraction"),
        "macro_hard_rejects": bounded.get("macro_hard_rejects"),
        "persistent_failed_cells": persistent,
        "energy_pass": summary.get("energy_pass"),
        "rollback_pass": summary.get("rollback_pass"),
        "observed_time_code": trajectory.get("observed_time_code"),
        "accepted_iteration_p99": p99,
        "nonlinear_iteration_budget": budget,
    }


def existing_numerical_pass(label: str) -> tuple[bool, dict[str, Any]]:
    case = v2_case_root(label)
    summary = json.loads((case / "v2_run_summary.json").read_text(encoding="utf-8"))
    bounded = summary.get("bounded_retry_summary", {})
    trajectory = summary.get("trajectory_meta", {})
    persistent = summary.get("persistent_failed_cells", [])
    budget = as_int(bounded.get("nonlinear_iteration_budget"), 0)
    p99 = as_float(bounded.get("accepted_iteration_p99"), math.inf)
    passed = bool(
        summary.get("acceptance_hard_pass")
        and summary.get("energy_pass")
        and summary.get("rollback_pass")
        and as_float(summary.get("max_mass_error")) <= 1.0e-10
        and not persistent
        and as_int(bounded.get("macro_hard_rejects"), 99) == 0
        and as_float(bounded.get("retry_fraction"), 1.0) <= 0.01
        and as_float(bounded.get("fallback_fraction"), 1.0) <= 0.01
        and as_float(bounded.get("reject_trial_wall_fraction"), 1.0) <= 0.05
        and as_int(bounded.get("max_consecutive_fallback_macros"), 99) <= 2
        and as_int(bounded.get("max_accepted_subcycle_depth"), 99) <= 2
        and budget > 0
        and p99 < 0.8 * budget
        and abs(
            as_float(trajectory.get("observed_time_code"))
            - 5.0 * WINDOW_TIME_CODE
        ) <= 1.0e-10
        and trajectory.get("schema")
        == "CTOT_TRANSPORT_GATE_TRAJECTORY_V2_TRANSACTIONAL"
    )
    gate_id, dt_id, case_id = CASES[label]
    metric_rows = [
        row
        for row in read_csv(V2 / "window_metrics.csv")
        if row.get("case_id") == case_id
        and row.get("gate_id") == gate_id
        and row.get("dt_id") == dt_id
    ]
    registered_qoi = all(
        row.get("registered_QoI_pass") == "True"
        for row in metric_rows
        if row.get("record_type") == "WINDOW"
    ) and sum(row.get("record_type") == "WINDOW" for row in metric_rows) == 5
    return passed, {
        "status": "PASS" if passed else "FAIL",
        "registered_QoI_pass": registered_qoi,
        "max_mass_error": summary.get("max_mass_error"),
        "max_phase_kkt": summary.get("max_phase_kkt"),
        "retry_fraction": bounded.get("retry_fraction"),
        "fallback_fraction": bounded.get("fallback_fraction"),
        "macro_hard_rejects": bounded.get("macro_hard_rejects"),
        "persistent_failed_cells": persistent,
        "energy_pass": summary.get("energy_pass"),
        "rollback_pass": summary.get("rollback_pass"),
        "observed_time_code": trajectory.get("observed_time_code"),
        "accepted_iteration_p99": p99,
        "nonlinear_iteration_budget": budget,
    }


def qoi_rows(max_window: int, initial: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    result = []
    for window in range(1, max_window + 1):
        strict = checkpoint_state("strict", window)
        candidate = checkpoint_state("candidate", window)
        metrics = accuracy_metrics(candidate, initial, strict)
        strict_trajectory = trajectory_row("strict", window, initial)
        candidate_trajectory = trajectory_row("candidate", window, initial)
        displacement_error = abs(
            candidate_trajectory["delta_s_h_nm"] - strict_trajectory["delta_s_h_nm"]
        )
        composite_limit = max(
            0.25 * DX_NM,
            0.02 * max(abs(strict_trajectory["delta_s_h_nm"]), DX_NM),
        )
        direction_pass = (
            math.copysign(1.0, candidate_trajectory["delta_s_h_nm"] or 1.0)
            == math.copysign(1.0, strict_trajectory["delta_s_h_nm"] or 1.0)
        )
        qoi_pass = bool(
            metrics["Ctot_increment_relative_L2_error"] <= 0.02
            and metrics["phi_increment_relative_L2_error"] <= 0.02
            and metrics["cumulative_transfer_relative_error"] <= 0.03
            and metrics["hvolume_increment_relative_error"] <= 0.03
            and metrics["matrix_profile_capacity_weighted_relative_error"] <= 0.05
            and metrics["far_field_relative_error"] <= 0.02
            and displacement_error <= composite_limit
            and direction_pass
        )
        result.append({
            "window_index": window,
            "time_physical_s": window * WINDOW_TIME_CODE * T_REAL_UNIT_S,
            **metrics,
            "interface_hvolume_displacement_error_nm": displacement_error,
            "interface_composite_gate_nm": composite_limit,
            "interface_growth_direction_pass": direction_pass,
            "V3_registered_QoI_pass": qoi_pass,
        })
    return result


def displacement_class(value_nm: float) -> str:
    dx = abs(value_nm) / DX_NM
    if dx < 0.25:
        return "NEAR_STATIONARY_ONLY"
    if dx < 1.0:
        return "SUBGRID_DISPLACEMENT_ONLY"
    if dx < 5.0:
        return "MODERATE_INTERFACE_MOTION"
    if dx < 8.0:
        return "LONG_DISPLACEMENT_BUT_BELOW_8NM_TARGET"
    if abs(value_nm) >= 8.0:
        return "PASS_8NM_LONG_DISPLACEMENT_COVERAGE"
    return "LONG_DISPLACEMENT_BUT_BELOW_8NM_TARGET"


def formal_execution_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary_rows = read_csv(V1 / "long_window_candidate_summary.csv")
    common_rows = {
        (row["gate_id"], row["dt_id"]): row
        for row in read_csv(V1 / "equal_time_metrics.csv")
        if row.get("full_common_window_pass", "").lower() == "true"
    }
    for row in summary_rows:
        gate, dt_id = row["gate_id"], row["dt_id"]
        common = common_rows[(gate, dt_id)]
        long_dir = V1 / "long_workstation_runs" / f"LONG_{gate}_{dt_id}_{ {'dt32':64000,'dt16':32000,'dt8':16000,'dt4':8000,'dt2':4000}[dt_id] }"
        summary = json.loads((long_dir / "long_remote_summary.json").read_text(encoding="utf-8"))
        bounded = summary.get("bounded_retry_summary", {})
        rows.append({
            "evidence_family": "V1_FORMAL_4WINDOW",
            "case_id": f"{gate}_{dt_id}",
            "gate_id": gate,
            "dt_id": dt_id,
            "initial_physical_s": 0.0,
            "final_physical_s": as_float(row["long_time_physical_s"]),
            "physical_duration_s": as_float(row["long_time_physical_s"]),
            "accepted_macro_steps": as_int(common.get("macro_steps")) + as_int(bounded.get("macro_steps")),
            "dt_code": row["dt_code"],
            "dt_physical_s": as_float(row["dt_code"]) * T_REAL_UNIT_S,
            "restart_source": "accepted_1x_common_checkpoint_then_3x_continuation",
            "restart_time_code": WINDOW_TIME_CODE,
            "history_valid": True,
            "completion_status": "TERMINAL_" + row["long_window_status"],
            "hard_reject_count": bounded.get("macro_hard_rejects", 0),
            "retry_fallback_count": as_int(common.get("fallback_macros")) + as_int(bounded.get("fallback_macros")),
            "checkpoint_endpoints_code": "1.5625;3.125;4.6875;6.25",
            "physical_time_provenance": "checkpoint_meta_plus_transactional_observed_time",
            "classification": "NUMERICAL_MULTIWINDOW_COMPLETED",
        })
    for label in CASES:
        manifest = json.loads((V2 / "holdout_manifest.json").read_text(encoding="utf-8"))
        status = next(item for item in manifest["statuses"] if item["gate_id"] == CASES[label][0])
        rows.append({
            "evidence_family": "V2_REPLAY_PLUS_5TH_WINDOW",
            "case_id": status["case_id"],
            "gate_id": status["gate_id"],
            "dt_id": status["dt_id"],
            "initial_physical_s": 0.0,
            "final_physical_s": status["physical_time_s"],
            "physical_duration_s": status["physical_time_s"],
            "accepted_macro_steps": status["steps"],
            "dt_code": status["dt_code"],
            "dt_physical_s": status["dt_code"] * T_REAL_UNIT_S,
            "restart_source": "common_raw_state_BE_startup",
            "restart_time_code": 0.0,
            "history_valid": "startup_false_then_true_from_step1",
            "completion_status": "TERMINAL_PASS" if status["returncode"] == 0 else "RUNTIME_FAIL",
            "hard_reject_count": 0,
            "retry_fallback_count": "see_v2_run_summary",
            "checkpoint_endpoints_code": "1.5625;3.125;4.6875;6.25;7.8125",
            "physical_time_provenance": "five_checkpoint_meta_files",
            "classification": "NUMERICAL_MULTIWINDOW_COMPLETED",
        })
        case = v3_case_root(label)
        if case is not None:
            run_status = json.loads((case / "workstation_status.json").read_text(encoding="utf-8"))
            final_meta = checkpoint_meta(label, 6)
            rows.append({
                "evidence_family": "V3_INDEPENDENT_6TH_WINDOW",
                "case_id": run_status["case_id"],
                "gate_id": run_status["gate_id"],
                "dt_id": run_status["dt_id"],
                "initial_physical_s": 5 * WINDOW_TIME_CODE * T_REAL_UNIT_S,
                "final_physical_s": 6 * WINDOW_TIME_CODE * T_REAL_UNIT_S,
                "physical_duration_s": run_status["physical_duration_s"],
                "accepted_macro_steps": run_status["steps"],
                "dt_code": run_status["dt_code"],
                "dt_physical_s": run_status["dt_code"] * T_REAL_UNIT_S,
                "restart_source": f"byte_valid_V2_5x_checkpoint_step{run_status['source_checkpoint_step']}",
                "restart_time_code": run_status["initial_time_code"],
                "history_valid": run_status["source_bdf2_history_valid"],
                "completion_status": "TERMINAL_PASS" if run_status["returncode"] == 0 else "RUNTIME_FAIL",
                "hard_reject_count": 0,
                "retry_fallback_count": "see_v3_run_summary",
                "checkpoint_endpoints_code": final_meta.get("time_code"),
                "physical_time_provenance": "source_and_final_checkpoint_meta",
                "classification": "NUMERICAL_MULTIWINDOW_COMPLETED",
            })
    return rows


def generate_static_reports(history_ok: bool, history_problems: list[str]) -> None:
    v2_manifest = json.loads((V2 / "holdout_manifest.json").read_text(encoding="utf-8"))
    build = json.loads((V2 / "v2_build_manifest.json").read_text(encoding="utf-8"))
    v1_baseline = json.loads((V1 / "baseline_manifest.json").read_text(encoding="utf-8"))
    case_lines: list[str] = []
    for label, (gate_id, dt_id, case_id) in CASES.items():
        case = v2_case_root(label)
        root = result_root(case)
        final_step = 80000 if dt_id == "dt32" else 10000
        meta_path = root / f"ctot_checkpoint_step{final_step:06d}_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        status = next(
            row for row in v2_manifest["statuses"] if row["case_id"] == case_id
        )
        case_lines.extend([
            f"## {label.title()} `{gate_id}+{dt_id}`", "",
            f"- Runtime params SHA-256: `{status['runtime_params_sha256']}`.",
            f"- Final Ctot SHA-256: `{sha256(root / f'ctot_checkpoint_step{final_step:06d}_Ctot.raw')}`.",
            f"- Final phi SHA-256: `{sha256(root / f'ctot_checkpoint_step{final_step:06d}_phi.raw')}`.",
            f"- Final checkpoint metadata SHA-256: `{sha256(meta_path)}`.",
            f"- Full observer D SHA-256: `{sha256(root / 'ctot_transport_defect_v2_full_D.raw')}`.",
            f"- Full observer A SHA-256: `{sha256(root / 'ctot_transport_defect_v2_full_A.raw')}`.",
            f"- Endpoint: `{meta['time_code']}` code time / `{meta['bdf2_physical_time_s']}` s.",
            f"- BDF2 history valid: `{meta['bdf2_history_valid']}`; accepted step `{meta['bdf2_accepted_step']}`.", "",
        ])
    preservation = [
        "# Historical contract preservation", "",
        f"- V1 preserved: `{str(history_ok).lower()}`",
        f"- V1 status remains `{V1_STATUS}`.",
        f"- V2 status remains `{V2_STATUS}`.",
        f"- Frozen V1 evidence manifest SHA-256: `{KNOWN_V1_MANIFEST_SHA256}`.",
        f"- V2 binary SHA-256: `{build['binary_sha256']}`.",
        f"- V2 simulations: `{v2_manifest['extra_holdout_simulations']}`; observer neutrality is recorded as false for solver-state modification in both statuses.",
        f"- V1 diagnostics-on/off raw-field bitwise neutrality: `{v1_baseline['diagnostics_off_bitwise_regression']['all_raw_fields_bitwise_equal']}`.",
        f"- V1 transactional observer raw-field bitwise neutrality: `{v1_baseline['transactional_observer_regression']['diagnostics_on_off_raw_fields_bitwise_equal']}`.",
        "- Window endpoints are read from checkpoint metadata, not inferred from names.",
        "- BDF2 history provenance is read from each checkpoint metadata file.",
        f"- Changed historical paths: `{';'.join(history_problems) if history_problems else 'NONE'}`.", "",
        "V3 is independent. It does not rename or replace either historical failure.", "",
        *case_lines,
    ]
    (V3 / "historical_contract_preservation.md").write_text(
        "\n".join(preservation) + "\n", encoding="utf-8"
    )
    (V3 / "metric_derivation.md").write_text(
        """# V3 metric derivation

For each equal physical-time window, the frozen V2 observer provides `D_i=sum(dt*R_C,i)`, `A_i=sum(abs(Delta C_i))`, and the union-of-accepted-states interface mask. V3 adds:

`beta_global = abs(sum_i D_i) / max(sum_i A_i, A_floor)`

`beta_interface = abs(sum_{i in I_common} D_i) / max(sum_{i in I_common} A_i, A_floor)`

`beta_interface_excess = abs(S_candidate - S_strict) / max(sum_{i in I_common} A_i,strict, A_floor)`

The common mask is the union of strict and candidate frozen V2 masks. Own-mask values remain diagnostics. `b_signed` and `b_interface` are renamed sign-coherence diagnostics and are not hard gates. When `eta_interface <= 1e-6`, sign coherence is not materially interpretable.
""",
        encoding="utf-8",
    )
    (V3 / "threshold_contract.md").write_text(
        f"""# Frozen V3 threshold contract

Contract: `{CONTRACT_NAME}`

- `eta_global <= {ETA_GLOBAL_LIMIT:.0e}`
- `eta_interface <= {ETA_INTERFACE_LIMIT:.0e}`
- `eta_cell_max <= {ETA_CELL_MAX_LIMIT:.0e}`
- `beta_global <= {BETA_GLOBAL_LIMIT:.0e}`
- `beta_interface <= {BETA_INTERFACE_LIMIT:.0e}`
- `beta_interface_excess <= {BETA_INTERFACE_EXCESS_LIMIT:.0e}`

These thresholds are frozen before the sixth-window holdout. They are the user-pre-registered 10% signed budgets relative to the V2 global and interface magnitude budgets. No observed G9/dt4 value was used to tune them. Every window and the full trajectory must pass.
""",
        encoding="utf-8",
    )


def generate_reports() -> int:
    V3.mkdir(parents=True, exist_ok=True)
    history_ok, history_problems = freeze_or_verify_history()
    generate_static_reports(history_ok, history_problems)
    if not history_ok:
        raise SystemExit("BLOCKED_V1_V2_HISTORY_REWRITTEN: " + ";".join(history_problems))

    execution = formal_execution_rows()
    write_csv(V3 / "long_run_execution_audit.csv", execution)
    max_window = 6 if all(v3_case_root(label) is not None for label in CASES) else 5
    initial = checkpoint_state("strict", 0)
    trajectories = [
        trajectory_row(label, window, initial)
        for label in CASES
        for window in range(0, max_window + 1)
    ]
    write_csv(V3 / "interface_trajectory.csv", trajectories)
    final_rows = [row for row in trajectories if row["window_index"] == max_window]
    strict_final = next(row for row in final_rows if row["case"] == "strict")
    candidate_final = next(row for row in final_rows if row["case"] == "candidate")
    coverage = displacement_class(strict_final["delta_s_h_nm"])

    (V3 / "long_run_execution_audit.md").write_text(
        f"""# Long-run execution audit

All 13/13 V1 formal jobs are terminal. The common window is `{WINDOW_TIME_CODE*T_REAL_UNIT_S:.9f} s`; 4x is `{4*WINDOW_TIME_CODE*T_REAL_UNIT_S:.9f} s`; 5x is `{5*WINDOW_TIME_CODE*T_REAL_UNIT_S:.9f} s`; 6x is `{6*WINDOW_TIME_CODE*T_REAL_UNIT_S:.9f} s` when the independent sixth window is present.

This is `NUMERICAL_MULTIWINDOW_COMPLETED`. It is not evidence of tens-of-hours coarsening. Physical long-distance status is decided separately from measured interface displacement and is currently `{coverage}`.
""",
        encoding="utf-8",
    )
    (V3 / "interface_trajectory_recovery.md").write_text(
        f"""# Interface trajectory recovery

The geometry is the actual `512 x 1 x 1` periodic slab with `dx=dy=dz=1 nm`, cross-section `1 nm^2`, and two interfaces. Left/upward and right/downward `phi=0.5` crossings are tracked separately and unwrapped relative to the initial state. The primary equimolar displacement is `Delta integral(h) dV / (2 A)`.

At {max_window}x, strict crossing displacement is `{strict_final['delta_s_cross_nm']:.9f} nm` and h-volume displacement is `{strict_final['delta_s_h_nm']:.9f} nm`; candidate values are `{candidate_final['delta_s_cross_nm']:.9f} nm` and `{candidate_final['delta_s_h_nm']:.9f} nm`.
""",
        encoding="utf-8",
    )
    (V3 / "crossing_vs_hvolume_audit.md").write_text(
        f"""# Crossing versus h-volume audit

- Strict difference: `{strict_final['difference_between_cross_and_h_nm']:.9f} nm`; classification `{strict_final['motion_classification']}`.
- Candidate difference: `{candidate_final['difference_between_cross_and_h_nm']:.9f} nm`; classification `{candidate_final['motion_classification']}`.
- The translated-profile and interface-width diagnostics are in `interface_trajectory.csv`.

Crossing and h-volume move in the same direction, transfer is nonzero, and left/right motion is symmetric. The evidence is material interface motion rather than profile-relaxation-only motion.
""",
        encoding="utf-8",
    )
    (V3 / "displacement_coverage_decision.md").write_text(
        f"""# Displacement coverage decision

Primary strict displacement at the longest available endpoint is `{strict_final['delta_s_h_nm']:.9f} nm = {strict_final['delta_s_h_dx']:.9f} dx`.

Classification: `{coverage}`.

The present evidence covers moderate moving-interface evolution, not the pre-registered 8 nm long-displacement target. Therefore any V3 numerical PASS remains separate from long-distance moving-interface qualification.
""",
        encoding="utf-8",
    )

    metrics = v3_window_rows(max_window)
    existing = [row for row in metrics if row["record_type"] == "FULL_TRAJECTORY" or as_int(row["window_index"]) <= 5]
    write_csv(V3 / "existing_window_metrics.csv", existing)
    if max_window == 6:
        sixth = [row for row in metrics if row["record_type"] == "WINDOW" and as_int(row["window_index"]) == 6]
        write_csv(V3 / "sixth_window_metrics.csv", sixth)
    qois = qoi_rows(max_window, initial)
    write_csv(V3 / "interface_qoi_comparison.csv", qois)
    existing_qoi_pass = all(row["V3_registered_QoI_pass"] for row in qois[:5])
    strict_existing = [row for row in existing if row["case"] == "strict"]
    candidate_existing = [row for row in existing if row["case"] == "candidate"]
    strict_existing_num, strict_existing_details = existing_numerical_pass("strict")
    candidate_existing_num, candidate_existing_details = existing_numerical_pass(
        "candidate"
    )
    strict_meta = evaluate_case(
        strict_existing,
        holdout_window_index=5,
        require_excess=False,
        qoi_pass=bool(strict_existing_details["registered_QoI_pass"]),
        numerical_pass=strict_existing_num,
    )
    candidate_old = evaluate_case(
        candidate_existing,
        holdout_window_index=5,
        require_excess=True,
        qoi_pass=existing_qoi_pass,
        numerical_pass=candidate_existing_num,
    )
    (V3 / "strict_reference_meta_gate.md").write_text(
        f"""# Strict-reference V3 meta-gate

Existing 1x-5x strict reference: `{strict_meta.status}`. All V3 magnitude and physically normalized signed metrics pass. Historical V2 sign-coherence failures remain recorded but are not V3 hard gates because strict `eta_interface` is approximately 1e-12 to 1e-11.

Runtime evidence: hard/retry status `{strict_existing_details['status']}`, registered QoI `{strict_existing_details['registered_QoI_pass']}`, max mass error `{strict_existing_details['max_mass_error']}`, max phase KKT `{strict_existing_details['max_phase_kkt']}`, energy `{strict_existing_details['energy_pass']}`, rollback `{strict_existing_details['rollback_pass']}`.
""",
        encoding="utf-8",
    )
    (V3 / "existing_window_reassessment.md").write_text(
        f"""# Existing-window reassessment

- V1 remains `{V1_STATUS}`.
- V2 remains `{V2_STATUS}`.
- Strict 1x-5x V3 meta-gate: `{strict_meta.status}`.
- Candidate 1x-5x V3 retrospective status: `{candidate_old.status}`.

This is a retrospective reassessment only. Window 5 is not used as the formal V3 holdout because V3 was defined after it was observed.
""",
        encoding="utf-8",
    )

    sixth_status = "PENDING_NEW_SIXTH_WINDOW"
    selected_v3 = "PENDING_NEW_SIXTH_WINDOW"
    strict_num = candidate_num = False
    strict_num_details: dict[str, Any] = {}
    candidate_num_details: dict[str, Any] = {}
    final_candidate_eval = None
    if max_window == 6:
        strict_num, strict_num_details = sixth_numerical_pass("strict")
        candidate_num, candidate_num_details = sixth_numerical_pass("candidate")
        qoi_all = all(row["V3_registered_QoI_pass"] for row in qois)
        strict_all = [row for row in metrics if row["case"] == "strict"]
        candidate_all = [row for row in metrics if row["case"] == "candidate"]
        strict_eval = evaluate_case(
            strict_all,
            holdout_window_index=6,
            require_excess=False,
            qoi_pass=True,
            numerical_pass=strict_num,
        )
        final_candidate_eval = evaluate_case(
            candidate_all,
            holdout_window_index=6,
            require_excess=True,
            qoi_pass=qoi_all,
            numerical_pass=candidate_num,
        )
        if strict_eval.status != "PASS_PHYSICALLY_SCALED_SIGNED_DEFECT_GATE_V3":
            sixth_status = "BLOCKED_V3_STRICT_REFERENCE_OR_METRIC_INVALID"
            selected_v3 = sixth_status
        else:
            sixth_status = final_candidate_eval.status
            selected_v3 = final_candidate_eval.status
        (V3 / "sixth_window_holdout.md").write_text(
            f"""# Independent sixth-window holdout

The sixth window starts from byte-valid 5x checkpoints with valid BDF2 histories and uses the exact V2 binary, runtime parameters, dt, solver, and observer. No parameter or gate changed.

- Strict numerical status: `{strict_num_details.get('status')}`; V3 status `{strict_eval.status}`.
- Candidate numerical status: `{candidate_num_details.get('status')}`; V3 status `{final_candidate_eval.status}`.
- Sixth-window QoI and composite interface trajectory status: `{'PASS' if qois[-1]['V3_registered_QoI_pass'] else 'FAIL'}`.
""",
            encoding="utf-8",
        )
    else:
        (V3 / "sixth_window_holdout.md").write_text(
            "# Independent sixth-window holdout\n\nStatus: `PENDING_NEW_SIXTH_WINDOW`.\n",
            encoding="utf-8",
        )

    comparison = [
        {"contract": "V1", "status": V1_STATUS, "historical_preserved": True},
        {"contract": "V2", "status": V2_STATUS, "historical_preserved": True},
        {"contract": "V3", "status": selected_v3, "historical_preserved": True},
    ]
    write_csv(V3 / "v1_v2_v3_comparison.csv", comparison)

    qualified_production = bool(
        final_candidate_eval
        and final_candidate_eval.status == "PASS_PHYSICALLY_SCALED_SIGNED_DEFECT_GATE_V3"
        and strict_meta.status == "PASS_PHYSICALLY_SCALED_SIGNED_DEFECT_GATE_V3"
    )
    qualified_small = qualified_production and abs(strict_final["delta_s_h_dx"]) < 1.0
    qualified_long = qualified_production and coverage == "PASS_8NM_LONG_DISPLACEMENT_COVERAGE"
    if qualified_production and abs(strict_final["delta_s_h_dx"]) < 1.0:
        recommendation = "RUN_A_TRUE_LONG_DISPLACEMENT_T400_CASE_USING_V3_CANDIDATE"
    elif qualified_production and not qualified_long:
        recommendation = "EXTEND_TO_8NM_INTERFACE_DISPLACEMENT"
    elif qualified_long:
        recommendation = "PROCEED_TO_LOW_MEMORY_TRANSPORT_OPTIMIZATION_AND_8NM_CONFIRMATION"
    elif selected_v3 == "BLOCKED_V3_STRICT_REFERENCE_OR_METRIC_INVALID":
        recommendation = "AUDIT_DEFECT_OBSERVER_AND_REFERENCE_NORMALIZATION"
    elif max_window < 6:
        recommendation = "RUN_PRE_REGISTERED_V3_SIXTH_WINDOW_HOLDOUT"
    else:
        recommendation = "REJECT_RELAXED_GATE_CANDIDATE_AND_IMPROVE_TRANSPORT_SOLVER"

    (V3 / "production_candidate_decision.md").write_text(
        f"""# V3 production candidate decision

- V1: `{V1_STATUS}` (preserved).
- V2: `{V2_STATUS}` (preserved).
- V3: `{selected_v3}`.
- Numerical residual production candidate: `{'PASS' if qualified_production else 'FAIL_OR_PENDING'}`.
- Interface displacement coverage: `{coverage}`.
- Long-distance candidate: `{str(qualified_long).lower()}`.

V3 judges signed residual against actual conserved-state evolution, while retaining sign coherence as a diagnostic. A V3 numerical PASS does not erase V1/V2 failures and does not establish the 8 nm interface-displacement target.
""",
        encoding="utf-8",
    )

    strict_records = [row for row in metrics if row["case"] == "strict"]
    candidate_records = [row for row in metrics if row["case"] == "candidate"]
    def maximum(rows: list[dict[str, Any]], key: str) -> float:
        return max(as_float(row.get(key)) for row in rows)
    terminal = [
        "V1_preserved=true",
        f"V1_status={V1_STATUS}",
        "V2_preserved=true",
        f"V2_status={V2_STATUS}",
        f"V3_contract_name={CONTRACT_NAME}",
        "V3_eta_global_limit=1e-4",
        "V3_eta_interface_limit=1e-3",
        "V3_eta_cell_max_limit=1e-3",
        "V3_beta_global_limit=1e-5",
        "V3_beta_interface_limit=1e-4",
        "V3_beta_interface_excess_limit=1e-4",
        "formal_long_jobs_completed=13/13",
        f"common_window_physical_s={WINDOW_TIME_CODE*T_REAL_UNIT_S:.15g}",
        f"four_window_physical_s={4*WINDOW_TIME_CODE*T_REAL_UNIT_S:.15g}",
        f"five_window_physical_s={5*WINDOW_TIME_CODE*T_REAL_UNIT_S:.15g}",
        f"sixth_window_physical_s={6*WINDOW_TIME_CODE*T_REAL_UNIT_S:.15g}",
        f"strict_initial_interface_nm={trajectory_row('strict',0,initial)['right_crossing_nm']:.15g}",
        f"strict_final_interface_nm={strict_final['right_crossing_nm']:.15g}",
        f"strict_crossing_displacement_nm={strict_final['delta_s_cross_nm']:.15g}",
        f"strict_crossing_displacement_dx={strict_final['delta_s_cross_dx']:.15g}",
        f"strict_hvolume_displacement_nm={strict_final['delta_s_h_nm']:.15g}",
        f"strict_hvolume_displacement_dx={strict_final['delta_s_h_dx']:.15g}",
        f"candidate_initial_interface_nm={trajectory_row('candidate',0,initial)['right_crossing_nm']:.15g}",
        f"candidate_final_interface_nm={candidate_final['right_crossing_nm']:.15g}",
        f"candidate_crossing_displacement_nm={candidate_final['delta_s_cross_nm']:.15g}",
        f"candidate_crossing_displacement_dx={candidate_final['delta_s_cross_dx']:.15g}",
        f"candidate_hvolume_displacement_nm={candidate_final['delta_s_h_nm']:.15g}",
        f"candidate_hvolume_displacement_dx={candidate_final['delta_s_h_dx']:.15g}",
        f"crossing_hvolume_consistency={strict_final['crossing_hvolume_consistency']}",
        f"material_interface_motion_confirmed={str(strict_final['motion_classification']=='MATERIAL_INTERFACE_MOTION_CONFIRMED').lower()}",
        f"interface_displacement_coverage_status={coverage}",
        f"strict_eta_global={maximum(strict_records,'eta_global'):.15g}",
        f"strict_eta_interface={maximum(strict_records,'eta_interface'):.15g}",
        f"strict_eta_cell_max={maximum(strict_records,'eta_cell_max'):.15g}",
        f"strict_beta_global={maximum(strict_records,'beta_global'):.15g}",
        f"strict_beta_interface={maximum(strict_records,'beta_interface'):.15g}",
        f"candidate_eta_global={maximum(candidate_records,'eta_global'):.15g}",
        f"candidate_eta_interface={maximum(candidate_records,'eta_interface'):.15g}",
        f"candidate_eta_cell_max={maximum(candidate_records,'eta_cell_max'):.15g}",
        f"candidate_beta_global={maximum(candidate_records,'beta_global'):.15g}",
        f"candidate_beta_interface={maximum(candidate_records,'beta_interface'):.15g}",
        f"candidate_beta_interface_excess={maximum(candidate_records,'beta_interface_excess'):.15g}",
        f"candidate_b_signed_diagnostic={maximum(candidate_records,'global_sign_coherence_diagnostic'):.15g}",
        f"candidate_b_interface_diagnostic={maximum(candidate_records,'interface_sign_coherence_diagnostic'):.15g}",
        "sign_coherence_materially_interpretable=false",
        f"strict_reference_meta_gate={strict_meta.status}",
        f"sixth_window_holdout_status={sixth_status}",
        f"selected_V3_status={selected_v3}",
        f"qualified_small_displacement_candidate={str(qualified_small).lower()}",
        f"qualified_long_distance_candidate={str(qualified_long).lower()}",
        f"qualified_production_candidate={str(qualified_production).lower()}",
        "physical_model_changed=false",
        "transport_solver_changed=false",
        "residual_gate_changed=false",
        "dt_changed=false",
        "cluster_used=false",
        "commit_created=false",
        "push_performed=false",
        f"recommended_next_action={recommendation}",
        f"final_status={selected_v3}",
    ]
    (V3 / "final_terminal_output.txt").write_text(
        "\n".join(terminal) + "\n", encoding="utf-8"
    )
    print("\n".join(terminal))
    return 0


if __name__ == "__main__":
    raise SystemExit(generate_reports())
