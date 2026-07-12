#!/usr/bin/env python3
"""Minimal observation-vs-CNT validation loop.

This module is read-only with respect to the physics model. It extracts CUDA
nucleation observations, reads offline CNT predictions, and compares rates and
barriers. It does not update S(x), fit feedback parameters, modify PF equations,
or introduce a new nucleation model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = REPO_ROOT / "reports"
VALIDATION_ROOT = REPORTS_ROOT / "validation"
DATA_ROOT = REPORTS_ROOT / "data"
CNT_ROOT = REPORTS_ROOT / "cnt"
K_B_J_PER_K = 1.380649e-23


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prefer_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_path_metadata(path_text: str) -> dict[str, Any]:
    text = path_text or ""
    out: dict[str, Any] = {}
    mt = re.search(r"T([0-9]+(?:p[0-9]+)?)", text)
    mx = re.search(r"xB([0-9]+p[0-9]+|[0-9.]+)", text)
    mdt = re.search(r"dt([0-9]+p[0-9]+|[0-9.]+)", text)
    mgrid = re.search(r"cuda_([0-9]+)x([0-9]+)x([0-9]+)", text)
    if mt:
        out["T_C"] = float(mt.group(1).replace("p", "."))
    if mx:
        out["xB"] = float(mx.group(1).replace("p", "."))
    if mdt:
        out["dt_code"] = float(mdt.group(1).replace("p", "."))
    if mgrid:
        out["grid_nx"], out["grid_ny"], out["grid_nz"] = [int(v) for v in mgrid.groups()]
    for mode in ("exx_eyy", "exx", "eyy", "ezz"):
        mstrain = re.search(rf"{mode}_(sm?[0-9]+(?:p[0-9]+)?|s[0-9]+(?:p[0-9]+)?|0p[0-9]+)", text)
        if mstrain:
            raw = mstrain.group(1)
            sign = -1.0 if raw.startswith("sm") else 1.0
            raw = raw.removeprefix("sm").removeprefix("s")
            out["strain_mode"] = mode
            out["strain"] = sign * float(raw.replace("p", "."))
            break
    return out


def norm_value(value: Any, digits: int = 8, default: str = "unknown") -> str:
    val = safe_float(value)
    return f"{val:.{digits}g}" if val is not None else default


def condition_key(T: Any, xB: Any, strain: Any) -> tuple[str, str, str]:
    return (norm_value(T, 6), norm_value(xB, 8), norm_value(strain, 8))


def shape_class(shape: str) -> str:
    s = (shape or "").strip().lower()
    if "facet" in s:
        return "faceted"
    if "anis" in s or "ellip" in s:
        return "anisotropic"
    if "sphere" in s or "spherical" in s:
        return "spherical"
    return "unknown"


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def infer_event_time(row: dict[str, str], meta: dict[str, Any]) -> tuple[float | None, str]:
    time_s = safe_float(row.get("time_physical_s"))
    if time_s is not None:
        return time_s, "seconds"
    step = safe_float(row.get("step"))
    dt_code = safe_float(row.get("dt_code"), safe_float(meta.get("dt_code")))
    if step is not None and dt_code is not None:
        return step * dt_code, "code_time"
    return None, "not_available"


def extract_observations(event_log: Path, output: Path, spatial_density_output: Path, histogram_output: Path) -> list[dict[str, Any]]:
    raw_events = read_csv(event_log)
    accepted = [row for row in raw_events if row.get("event_status", "accepted") in {"", "accepted"}]
    normalized: list[dict[str, Any]] = []

    for idx, row in enumerate(accepted, start=1):
        meta = parse_path_metadata(row.get("_source_file", ""))
        T_C = safe_float(row.get("T_C"), safe_float(meta.get("T_C")))
        xB = safe_float(row.get("local_xB"), safe_float(meta.get("xB")))
        strain = safe_float(row.get("strain"), safe_float(meta.get("strain"), 0.0))
        strain_mode = row.get("strain_mode") or meta.get("strain_mode", "unknown")
        event_time, time_basis = infer_event_time(row, meta)
        gp_flag = 1 if safe_int(row.get("gp_presence_flag")) or safe_int(row.get("gp_assisted_flag")) else 0
        phi_gp = safe_float(row.get("local_phi_GP"), float(gp_flag))
        domain_nm3 = safe_float(row.get("domain_volume_nm3"))
        normalized.append(
            {
                "observation_id": f"obs_{idx:06d}",
                "event_id": row.get("event_id", idx),
                "event_time": event_time if event_time is not None else "",
                "event_time_basis": time_basis,
                "event_position_x": row.get("center_x_nm", ""),
                "event_position_y": row.get("center_y_nm", ""),
                "event_position_z": row.get("center_z_nm", ""),
                "T_C": T_C if T_C is not None else "",
                "xB": xB if xB is not None else "",
                "strain": strain if strain is not None else "",
                "strain_mode": strain_mode,
                "local_composition_xB": xB if xB is not None else "",
                "local_GP_field_phi_GP": phi_gp if phi_gp is not None else "",
                "local_GP_field_basis": "local_phi_GP" if row.get("local_phi_GP") else "gp_presence_flag_proxy",
                "local_strain": strain if strain is not None else "",
                "identified_nucleus_type": "beta",
                "nucleus_shape": shape_class(row.get("nucleus_shape_type", "")),
                "nucleus_size_rc_nm": row.get("rc_nm", ""),
                "gp_assisted_flag": gp_flag,
                "domain_volume_nm3": domain_nm3 if domain_nm3 is not None else "",
                "source_file": row.get("_source_file", ""),
                "extraction_status": "ok" if event_time is not None and xB is not None and T_C is not None else "partial_missing_fields",
            }
        )

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        groups[condition_key(row["T_C"], row["xB"], row["strain"])].append(row)

    rate_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, rows in groups.items():
        event_times = [safe_float(r["event_time"]) for r in rows if safe_float(r["event_time"]) is not None]
        volumes = [safe_float(r["domain_volume_nm3"]) for r in rows if safe_float(r["domain_volume_nm3"]) is not None and safe_float(r["domain_volume_nm3"]) > 0.0]
        if event_times and volumes and rows[0]["event_time_basis"] == "seconds":
            duration = max(event_times) - min(0.0, min(event_times))
            volume_m3 = max(volumes) * 1.0e-27
            J_obs = len(rows) / max(duration * volume_m3, 1.0e-300)
            units = "m^-3_s^-1"
            status = "physical_rate"
        elif event_times:
            duration = max(event_times) - min(0.0, min(event_times))
            J_obs = len(rows) / max(duration, 1.0e-300)
            volume_m3 = ""
            units = "events_per_code_time"
            status = "proxy_rate_missing_physical_volume_or_time"
        else:
            duration = ""
            J_obs = ""
            volume_m3 = ""
            units = "not_available"
            status = "missing_time"
        rate_by_key[key] = {
            "nucleation_rate_J_obs": J_obs,
            "J_obs_units": units,
            "observation_window": duration,
            "observation_volume_m3": volume_m3,
            "rate_status": status,
            "event_count_for_condition": len(rows),
        }

    observation_rows = []
    for row in normalized:
        payload = {**row, **rate_by_key[condition_key(row["T_C"], row["xB"], row["strain"])]}
        observation_rows.append(payload)

    fields = [
        "observation_id",
        "event_id",
        "event_time",
        "event_time_basis",
        "event_position_x",
        "event_position_y",
        "event_position_z",
        "T_C",
        "xB",
        "strain",
        "strain_mode",
        "local_composition_xB",
        "local_GP_field_phi_GP",
        "local_GP_field_basis",
        "local_strain",
        "identified_nucleus_type",
        "nucleus_shape",
        "nucleus_size_rc_nm",
        "gp_assisted_flag",
        "nucleation_rate_J_obs",
        "J_obs_units",
        "observation_window",
        "observation_volume_m3",
        "rate_status",
        "event_count_for_condition",
        "source_file",
        "extraction_status",
    ]
    write_csv(output, observation_rows, fields)

    density_counter: Counter[tuple[str, str, str, str, str, str]] = Counter()
    for row in observation_rows:
        key = (
            norm_value(row["T_C"], 6),
            norm_value(row["xB"], 8),
            norm_value(row["strain"], 8),
            norm_value(row["event_position_x"], 8),
            norm_value(row["event_position_y"], 8),
            norm_value(row["event_position_z"], 8),
        )
        density_counter[key] += 1
    density_rows = [
        {
            "T_C": k[0],
            "xB": k[1],
            "strain": k[2],
            "x": k[3],
            "y": k[4],
            "z": k[5],
            "event_count": count,
            "density_units": "events_per_position_bin",
        }
        for k, count in sorted(density_counter.items())
    ]
    write_csv(spatial_density_output, density_rows, ["T_C", "xB", "strain", "x", "y", "z", "event_count", "density_units"])

    histogram_counter: Counter[tuple[str, str, str, str]] = Counter()
    for row in observation_rows:
        t = safe_float(row["event_time"])
        bin_id = "unknown" if t is None else f"{math.floor(t):.0f}"
        histogram_counter[(norm_value(row["T_C"], 6), norm_value(row["xB"], 8), norm_value(row["strain"], 8), bin_id)] += 1
    histogram_rows = [
        {"T_C": k[0], "xB": k[1], "strain": k[2], "time_bin": k[3], "event_count": count}
        for k, count in sorted(histogram_counter.items())
    ]
    write_csv(histogram_output, histogram_rows, ["T_C", "xB", "strain", "time_bin", "event_count"])

    return observation_rows


def barrier_kBT_from_prediction(row: dict[str, str]) -> float | None:
    for key in ("DeltaG_star_kBT", "energy_barrier_kBT", "predicted_barrier"):
        val = safe_float(row.get(key))
        if val is not None:
            units = (row.get("barrier_units") or row.get("energy_units") or "kBT").lower()
            if units in {"j", "joule", "joules"}:
                T_C = safe_float(row.get("T_C"), safe_float(row.get("T")))
                if T_C is None:
                    return None
                return val / (K_B_J_PER_K * (T_C + 273.15))
            return val
    return None


def load_prediction_sources(track1_barrier: Path, prediction_table: Path, catalog_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for row in read_csv(track1_barrier):
        barrier = barrier_kBT_from_prediction(row)
        if barrier is None:
            continue
        T = safe_float(row.get("T_C"), safe_float(row.get("T")))
        xB = safe_float(row.get("xB"))
        strain = safe_float(row.get("strain"), safe_float(row.get("strain_value"), 0.0))
        shape = row.get("shape_type", "")
        key = (*condition_key(T, xB, strain), shape)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "condition_id": f"T{key[0]}_xB{key[1]}_strain{key[2]}_{shape or 'unknown'}",
                "T_C": T,
                "xB": xB,
                "strain": strain,
                "strain_mode": row.get("strain_mode", "unknown"),
                "DeltaG_star_CNT_kBT": barrier,
                "r_star_CNT_nm": safe_float(row.get("rc_nm"), safe_float(row.get("r_star_nm"))),
                "shape_CNT": shape or "unknown",
                "S_CNT": row.get("S_CNT", ""),
                "prediction_source": str(track1_barrier),
            }
        )

    for row in read_csv(prediction_table):
        barrier = barrier_kBT_from_prediction(row)
        if barrier is None:
            continue
        T = safe_float(row.get("T"), safe_float(row.get("T_C")))
        xB = safe_float(row.get("xB"))
        strain = safe_float(row.get("strain"), 0.0)
        shape = row.get("predicted_shape", row.get("shape_type", ""))
        key = (*condition_key(T, xB, strain), shape)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "condition_id": f"T{key[0]}_xB{key[1]}_strain{key[2]}_{shape or 'unknown'}",
                "T_C": T,
                "xB": xB,
                "strain": strain,
                "strain_mode": row.get("strain_mode", "unknown"),
                "DeltaG_star_CNT_kBT": barrier,
                "r_star_CNT_nm": safe_float(row.get("predicted_rc")),
                "shape_CNT": shape or "unknown",
                "S_CNT": row.get("S_CNT", ""),
                "prediction_source": row.get("prediction_source", str(prediction_table)),
            }
        )

    if catalog_path.exists():
        try:
            payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        for item in payload.get("entries", []):
            if not isinstance(item, dict):
                continue
            barrier = safe_float(item.get("energy_barrier_kBT"))
            if barrier is None or barrier >= 1.0e299:
                continue
            T = safe_float(item.get("T_C"))
            xB = safe_float(item.get("xB"))
            strain = safe_float(item.get("strain_value"), 0.0)
            shape = item.get("shape_type", "")
            key = (*condition_key(T, xB, strain), shape)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "condition_id": f"T{key[0]}_xB{key[1]}_strain{key[2]}_{shape or 'unknown'}",
                    "T_C": T,
                    "xB": xB,
                    "strain": strain,
                    "strain_mode": item.get("strain_mode", "unknown"),
                    "DeltaG_star_CNT_kBT": barrier,
                    "r_star_CNT_nm": safe_float(item.get("rc_nm")),
                    "shape_CNT": shape or "unknown",
                    "S_CNT": item.get("S_CNT", ""),
                    "prediction_source": str(catalog_path),
                }
            )
    return rows


def build_cnt_prediction_table(predictions: list[dict[str, Any]], output: Path, J0: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in predictions:
        barrier = safe_float(row.get("DeltaG_star_CNT_kBT"))
        T_C = safe_float(row.get("T_C"))
        J_CNT = J0 * math.exp(-barrier) if barrier is not None and barrier < 745.0 else 0.0
        rows.append(
            {
                **row,
                "J0": J0,
                "J_CNT": J_CNT,
                "J_CNT_units": "same_as_J0",
                "prediction_status": "ok" if T_C is not None and barrier is not None else "partial",
            }
        )
    fields = [
        "condition_id",
        "T_C",
        "xB",
        "strain",
        "strain_mode",
        "DeltaG_star_CNT_kBT",
        "r_star_CNT_nm",
        "shape_CNT",
        "S_CNT",
        "J0",
        "J_CNT",
        "J_CNT_units",
        "prediction_source",
        "prediction_status",
    ]
    write_csv(output, rows, fields)
    return rows


def nearest_prediction(obs: dict[str, Any], predictions: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    T = safe_float(obs.get("T_C"))
    xB = safe_float(obs.get("xB"))
    strain = safe_float(obs.get("strain"), 0.0)
    if T is None or xB is None:
        return None, math.inf
    best = None
    best_score = math.inf
    for pred in predictions:
        Tp = safe_float(pred.get("T_C"))
        xp = safe_float(pred.get("xB"))
        sp = safe_float(pred.get("strain"), 0.0)
        if Tp is None or xp is None:
            continue
        score = abs(T - Tp) / 100.0 + abs(xB - xp) / 0.01 + abs((strain or 0.0) - (sp or 0.0)) / 0.01
        if score < best_score:
            best = pred
            best_score = score
    return best, best_score


def build_comparison(observations: list[dict[str, Any]], predictions: list[dict[str, Any]], output: Path, J0: float) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        groups[condition_key(obs.get("T_C"), obs.get("xB"), obs.get("strain"))].append(obs)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        representative = items[0]
        pred, mismatch = nearest_prediction(representative, predictions)
        if pred is None:
            rows.append(
                {
                    "condition_key": "|".join(key),
                    "T_C_obs": representative.get("T_C", ""),
                    "xB_obs": representative.get("xB", ""),
                    "strain_obs": representative.get("strain", ""),
                    "comparison_status": "missing_prediction",
                }
            )
            continue

        J_obs = safe_float(representative.get("nucleation_rate_J_obs"))
        J_CNT = safe_float(pred.get("J_CNT"))
        barrier_CNT = safe_float(pred.get("DeltaG_star_CNT_kBT"))
        barrier_obs = None
        if J_obs is not None and J_obs > 0.0 and J0 > 0.0:
            barrier_obs = -math.log(max(J_obs / J0, 1.0e-300))
        if J_obs is not None and J_CNT is not None and J_CNT > 0.0:
            error_J = abs(J_obs - J_CNT) / J_CNT
        else:
            error_J = None
        barrier_error = abs(barrier_obs - barrier_CNT) if barrier_obs is not None and barrier_CNT is not None else None
        rate_is_physical = representative.get("J_obs_units") == "m^-3_s^-1"
        condition_match_is_close = mismatch <= 0.25
        if error_J is None or barrier_error is None:
            comparison_status = "partial_missing_physical_rate_or_prediction"
        elif not rate_is_physical:
            comparison_status = "partial_proxy_rate_units"
        elif not condition_match_is_close:
            comparison_status = "partial_condition_mismatch"
        else:
            comparison_status = "ok"
        rows.append(
            {
                "condition_key": "|".join(key),
                "T_C_obs": representative.get("T_C", ""),
                "xB_obs": representative.get("xB", ""),
                "strain_obs": representative.get("strain", ""),
                "event_count": len(items),
                "GP_event_fraction": mean([safe_float(i.get("gp_assisted_flag"), 0.0) for i in items]),
                "J_obs": J_obs if J_obs is not None else "",
                "J_obs_units": representative.get("J_obs_units", ""),
                "DeltaG_effective_from_CUDA_kBT": barrier_obs if barrier_obs is not None else "",
                "matched_T_C_CNT": pred.get("T_C", ""),
                "matched_xB_CNT": pred.get("xB", ""),
                "matched_strain_CNT": pred.get("strain", ""),
                "DeltaG_star_CNT_kBT": barrier_CNT if barrier_CNT is not None else "",
                "r_star_CNT_nm": pred.get("r_star_CNT_nm", ""),
                "shape_CNT": pred.get("shape_CNT", ""),
                "J_CNT": J_CNT if J_CNT is not None else "",
                "J0": J0,
                "error_J": error_J if error_J is not None else "",
                "barrier_error_kBT": barrier_error if barrier_error is not None else "",
                "prediction_mismatch_score": mismatch,
                "comparison_status": comparison_status,
                "prediction_source": pred.get("prediction_source", ""),
            }
        )

    fields = [
        "condition_key",
        "T_C_obs",
        "xB_obs",
        "strain_obs",
        "event_count",
        "GP_event_fraction",
        "J_obs",
        "J_obs_units",
        "DeltaG_effective_from_CUDA_kBT",
        "matched_T_C_CNT",
        "matched_xB_CNT",
        "matched_strain_CNT",
        "DeltaG_star_CNT_kBT",
        "r_star_CNT_nm",
        "shape_CNT",
        "J_CNT",
        "J0",
        "error_J",
        "barrier_error_kBT",
        "prediction_mismatch_score",
        "comparison_status",
        "prediction_source",
    ]
    write_csv(output, rows, fields)
    return rows


def ranking_preserved(rows: list[dict[str, Any]], obs_key: str, pred_key: str) -> str:
    valid = [(safe_float(r.get(obs_key)), safe_float(r.get(pred_key))) for r in rows]
    valid = [(a, b) for a, b in valid if a is not None and b is not None]
    if len(valid) < 2:
        return "not_enough_data"
    obs_order = sorted(range(len(valid)), key=lambda i: valid[i][0])
    pred_order = sorted(range(len(valid)), key=lambda i: valid[i][1])
    return "yes" if obs_order == pred_order else "no"


@dataclass(frozen=True)
class ValidationSummary:
    status: str
    accuracy_score: float
    gp_effect_detected: str
    consistency_level: str
    largest_deviation: str


def summarize(comparison_rows: list[dict[str, Any]], observations: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> ValidationSummary:
    quantitative_rows = [r for r in comparison_rows if r.get("comparison_status") == "ok"]
    error_vals = [safe_float(r.get("error_J")) for r in quantitative_rows]
    error_vals = [v for v in error_vals if v is not None and math.isfinite(v)]
    barrier_errors = [safe_float(r.get("barrier_error_kBT")) for r in quantitative_rows]
    barrier_errors = [v for v in barrier_errors if v is not None and math.isfinite(v)]

    if error_vals:
        median_error = sorted(error_vals)[len(error_vals) // 2]
        score = 1.0 / (1.0 + median_error)
    elif barrier_errors:
        median_barrier = sorted(barrier_errors)[len(barrier_errors) // 2]
        score = 1.0 / (1.0 + median_barrier)
    else:
        score = 0.0

    gp_events = sum(1 for row in observations if safe_int(row.get("gp_assisted_flag")) > 0)
    gp_effect = "YES" if gp_events > 0 else "NO"

    complete = sum(1 for row in comparison_rows if row.get("comparison_status") == "ok")
    if complete and score >= 0.5:
        status = "PASS"
        level = "HIGH"
    elif comparison_rows and (complete > 0 or gp_events > 0 or predictions):
        status = "PARTIAL"
        level = "MEDIUM" if score >= 0.1 else "LOW"
    else:
        status = "FAIL"
        level = "LOW"

    deviations = sorted(
        comparison_rows,
        key=lambda r: safe_float(r.get("error_J"), safe_float(r.get("barrier_error_kBT"), -1.0)) or -1.0,
        reverse=True,
    )
    largest = deviations[0].get("condition_key", "not_available") if deviations else "not_available"
    return ValidationSummary(status, score, gp_effect, level, largest)


def write_summary_md(path: Path, comparison_rows: list[dict[str, Any]], summary: ValidationSummary) -> None:
    ranking_rate = ranking_preserved(comparison_rows, "J_obs", "J_CNT")
    ranking_barrier = ranking_preserved(comparison_rows, "DeltaG_effective_from_CUDA_kBT", "DeltaG_star_CNT_kBT")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Minimal CNT-vs-CUDA Comparison Summary",
        "",
        "## Scope",
        "",
        "This is an observation-vs-prediction validation loop only. It performs no S-field update, no feedback fitting, no PF-equation modification, and no new nucleation-physics modeling.",
        "",
        "## Metrics",
        "",
        f"- validation_status: {summary.status}",
        f"- CNT_predictive_accuracy_score: {summary.accuracy_score:.6g}",
        f"- GP_effect_detected: {summary.gp_effect_detected}",
        f"- system_consistency_level: {summary.consistency_level}",
        f"- largest_deviation_regime: {summary.largest_deviation}",
        f"- rate_ranking_preserved: {ranking_rate}",
        f"- barrier_ranking_preserved: {ranking_barrier}",
        "",
        "## Data Caveat",
        "",
        "Rows with `comparison_status` beginning with `partial_` are not quantitative rate validation points. They remain useful for checking whether CUDA emitted beta nucleation observations and whether a CNT prediction exists nearby.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_report(path: Path, comparison_rows: list[dict[str, Any]], observations: list[dict[str, Any]], predictions: list[dict[str, Any]], summary: ValidationSummary) -> None:
    gp_events = [row for row in observations if safe_int(row.get("gp_assisted_flag")) > 0]
    complete = [row for row in comparison_rows if row.get("comparison_status") == "ok"]
    partial = [row for row in comparison_rows if row.get("comparison_status") != "ok"]
    rate_ranking = ranking_preserved(comparison_rows, "J_obs", "J_CNT")
    barrier_ranking = ranking_preserved(comparison_rows, "DeltaG_effective_from_CUDA_kBT", "DeltaG_star_CNT_kBT")
    largest_rows = sorted(
        comparison_rows,
        key=lambda r: safe_float(r.get("error_J"), safe_float(r.get("barrier_error_kBT"), -1.0)) or -1.0,
        reverse=True,
    )[:5]
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Minimal Physics Closure Validation Report",
        "",
        "## Contract",
        "",
        "- No model fitting was performed.",
        "- No S(x) update or learning was performed.",
        "- No PF equation was modified.",
        "- No new nucleation physics model was introduced.",
        "- The module only compares CUDA observations against CNT predictions.",
        "",
        "## Generated Tables",
        "",
        "- `reports/data/cuda_nucleation_observation.csv`",
        "- `reports/data/cuda_nucleation_spatial_density.csv`",
        "- `reports/data/cuda_nucleation_time_histogram.csv`",
        "- `reports/cnt/cnt_prediction_table.csv`",
        "- `reports/validation/comparison_report.csv`",
        "- `reports/validation/comparison_summary.md`",
        "",
        "## 1. Does CNT predict the correct nucleation rate trend?",
        "",
    ]
    if complete:
        lines.append("Quantitative rate comparison points exist, so trend comparison can be evaluated from `comparison_report.csv`.")
    else:
        lines.append("Only partial trend validation is possible because the available CUDA event logs do not provide enough physical time/volume information for a fully dimensional `J_obs` at the matched CNT conditions.")
    lines.extend(
        [
            f"",
            f"- rate_ranking_preserved: {rate_ranking}",
            f"- complete_comparison_points: {len(complete)}",
            f"- partial_comparison_points: {len(partial)}",
            "",
            "## 2. Is GP-assisted nucleation observed in CUDA?",
            "",
            f"GP-assisted beta nucleation events detected: {len(gp_events)}.",
            "",
            "## 3. Does barrier ordering match T, xB, strain dependence?",
            "",
            f"- barrier_ranking_preserved: {barrier_ranking}",
            "- Ordering should be considered inconclusive if the value is `not_enough_data`.",
            "",
            "## 4. Largest Deviation Regimes",
            "",
        ]
    )
    if largest_rows:
        for row in largest_rows:
            lines.append(
                f"- {row.get('condition_key')}: status={row.get('comparison_status')}, "
                f"error_J={row.get('error_J', '')}, barrier_error_kBT={row.get('barrier_error_kBT', '')}"
            )
    else:
        lines.append("- No matched comparison rows were available.")

    lines.extend(
        [
            "",
            "## Final Status",
            "",
            f"- validation_status = {summary.status}",
            f"- CNT_predictive_accuracy_score = {summary.accuracy_score:.6g}",
            f"- GP_effect_detected = {summary.gp_effect_detected}",
            f"- system_consistency_level = {summary.consistency_level}",
            "",
            "## Interpretation",
            "",
            "This module establishes the minimal physical closure audit path: CUDA observation -> CNT prediction -> direct comparison. Current status is limited by the completeness of event logs, especially physical time, domain volume, local strain, and local phi_GP values.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> ValidationSummary:
    observations = extract_observations(
        event_log=args.event_log,
        output=args.cuda_observation,
        spatial_density_output=args.spatial_density,
        histogram_output=args.time_histogram,
    )
    raw_predictions = load_prediction_sources(args.track1_barrier, args.prediction_table, args.catalog)
    predictions = build_cnt_prediction_table(raw_predictions, args.cnt_prediction, args.J0)
    comparison = build_comparison(observations, predictions, args.comparison_report, args.J0)
    summary = summarize(comparison, observations, predictions)
    write_summary_md(args.comparison_summary, comparison, summary)
    write_validation_report(args.validation_report, comparison, observations, predictions, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-log",
        type=Path,
        default=prefer_existing_path(REPORTS_ROOT / "nucleation/nucleation_event_log.csv", REPO_ROOT / "nucleation_event_log.csv"),
    )
    parser.add_argument("--track1-barrier", type=Path, default=REPO_ROOT / "Results/dual_track_calibration/barrier_surface_fit.csv")
    parser.add_argument(
        "--prediction-table",
        type=Path,
        default=prefer_existing_path(REPORTS_ROOT / "cnt/nucleation_prediction_table.csv", REPO_ROOT / "nucleation_prediction_table.csv"),
    )
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "nucleus_catalog.json")
    parser.add_argument("--J0", type=float, default=1.0)
    parser.add_argument("--cuda-observation", type=Path, default=DATA_ROOT / "cuda_nucleation_observation.csv")
    parser.add_argument("--spatial-density", type=Path, default=DATA_ROOT / "cuda_nucleation_spatial_density.csv")
    parser.add_argument("--time-histogram", type=Path, default=DATA_ROOT / "cuda_nucleation_time_histogram.csv")
    parser.add_argument("--cnt-prediction", type=Path, default=CNT_ROOT / "cnt_prediction_table.csv")
    parser.add_argument("--comparison-report", type=Path, default=VALIDATION_ROOT / "comparison_report.csv")
    parser.add_argument("--comparison-summary", type=Path, default=VALIDATION_ROOT / "comparison_summary.md")
    parser.add_argument("--validation-report", type=Path, default=VALIDATION_ROOT / "validation_minimal_loop_report.md")
    args = parser.parse_args()

    summary = run(args)
    print(f"validation_status = {summary.status}")
    print(f"CNT_predictive_accuracy_score = {summary.accuracy_score:.6g}")
    print(f"GP_effect_detected = {summary.gp_effect_detected}")
    print(f"system_consistency_level = {summary.consistency_level}")


if __name__ == "__main__":
    main()
