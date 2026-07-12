#!/usr/bin/env python3
"""Physical-unit normalization for CNT-vs-CUDA nucleation validation.

This layer reconstructs comparable physical quantities only. It does not fit
S(x), modify PF equations, or introduce a new nucleation model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
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


def safe_int(value: Any, default: int | None = None) -> int | None:
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


def parse_key_value_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
        elif ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def parse_path_metadata(path_text: str) -> dict[str, Any]:
    text = path_text or ""
    out: dict[str, Any] = {}
    mt = re.search(r"T([0-9]+(?:p[0-9]+)?)", text)
    mx = re.search(r"xB([0-9]+p[0-9]+|[0-9.]+)", text)
    mgrid = re.search(r"cuda_([0-9]+)x([0-9]+)x([0-9]+)", text)
    mdt = re.search(r"dt([0-9]+p[0-9]+|[0-9.]+)", text)
    msteps = re.search(r"steps([0-9]+)", text)
    if mt:
        out["T_C"] = float(mt.group(1).replace("p", "."))
    if mx:
        out["xB"] = float(mx.group(1).replace("p", "."))
    if mgrid:
        out["Nx"], out["Ny"], out["Nz"] = [int(v) for v in mgrid.groups()]
    if mdt:
        out["dt_code"] = float(mdt.group(1).replace("p", "."))
    if msteps:
        out["steps"] = int(msteps.group(1))
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


def normalize_key(T: Any, xB: Any, strain: Any) -> tuple[str, str, str]:
    def fmt(value: Any, digits: int) -> str:
        val = safe_float(value)
        return f"{val:.{digits}g}" if val is not None else "unknown"

    return (fmt(T, 6), fmt(xB, 8), fmt(strain, 8))


def candidate_param_paths(source_file: str) -> list[Path]:
    path = Path(source_file)
    if not path.is_absolute():
        path = REPO_ROOT / path
    candidates = []
    if path.parent:
        candidates.append(path.parent / "pf_input.params")
        parent = path.parent
        while parent != parent.parent and parent != REPO_ROOT.parent:
            candidates.append(parent / "pf_input.params")
            parent = parent.parent
    return candidates


@dataclass
class RunUnitConfig:
    source_file: str
    T_C: float | None
    xB: float | None
    strain: float
    strain_mode: str
    Nx: int | None
    Ny: int | None
    Nz: int | None
    dx_nm: float | None
    dy_nm: float | None
    dz_nm: float | None
    voxel_volume_m3: float | None
    system_volume_m3: float | None
    dt_code: float | None
    t_real_unit_s: float | None
    dt_mapping_factor: float | None
    total_steps: int | None
    total_time_s: float | None
    time_step_definition: str
    reconstruction_status: str
    source_params: str


def reconstruct_run_config(source_file: str, rows: list[dict[str, str]]) -> RunUnitConfig:
    meta = parse_path_metadata(source_file)
    params: dict[str, str] = {}
    params_path = ""
    for candidate in candidate_param_paths(source_file):
        params = parse_key_value_file(candidate)
        if params:
            params_path = str(candidate)
            break

    sample = rows[0] if rows else {}
    T_C = safe_float(sample.get("T_C"), safe_float(meta.get("T_C")))
    xB = safe_float(sample.get("local_xB"), safe_float(meta.get("xB")))
    strain = safe_float(sample.get("strain"), safe_float(meta.get("strain"), 0.0)) or 0.0
    strain_mode = sample.get("strain_mode") or meta.get("strain_mode", "unknown")
    Nx = safe_int(params.get("Nx"), safe_int(meta.get("Nx")))
    Ny = safe_int(params.get("Ny"), safe_int(meta.get("Ny")))
    Nz = safe_int(params.get("Nz"), safe_int(meta.get("Nz")))
    dx_nm = safe_float(params.get("dx"))
    dy_nm = safe_float(params.get("dy"))
    dz_nm = safe_float(params.get("dz"))
    dt_code = safe_float(sample.get("dt_code"), safe_float(params.get("dt"), safe_float(meta.get("dt_code"))))
    t_real_unit_s = safe_float(sample.get("t_real_unit_s"), safe_float(params.get("t_real_unit")))
    steps = safe_int(meta.get("steps"))

    voxel_volume_m3 = None
    system_volume_m3 = None
    if dx_nm is not None and dy_nm is not None and dz_nm is not None:
        voxel_volume_m3 = dx_nm * dy_nm * dz_nm * 1.0e-27
    if voxel_volume_m3 is not None and Nx and Ny and Nz:
        system_volume_m3 = voxel_volume_m3 * Nx * Ny * Nz

    dt_mapping_factor = t_real_unit_s
    total_time_s = None
    if steps is not None and dt_code is not None and t_real_unit_s is not None:
        total_time_s = steps * dt_code * t_real_unit_s
    else:
        event_times = []
        for row in rows:
            t_phys = safe_float(row.get("time_physical_s"))
            if t_phys is not None:
                event_times.append(t_phys)
            else:
                step = safe_float(row.get("step"))
                if step is not None and dt_code is not None and t_real_unit_s is not None:
                    event_times.append(step * dt_code * t_real_unit_s)
        if event_times:
            total_time_s = max(event_times)

    required = [Nx, Ny, Nz, dx_nm, dy_nm, dz_nm, system_volume_m3, dt_code, t_real_unit_s, total_time_s]
    status = "strict_physical" if all(v is not None and safe_float(v, 1.0) is not None for v in required) else "incomplete_units"
    if not params:
        status = "inferred_from_path_missing_pf_input"

    return RunUnitConfig(
        source_file=source_file,
        T_C=T_C,
        xB=xB,
        strain=strain,
        strain_mode=strain_mode,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        dx_nm=dx_nm,
        dy_nm=dy_nm,
        dz_nm=dz_nm,
        voxel_volume_m3=voxel_volume_m3,
        system_volume_m3=system_volume_m3,
        dt_code=dt_code,
        t_real_unit_s=t_real_unit_s,
        dt_mapping_factor=dt_mapping_factor,
        total_steps=steps,
        total_time_s=total_time_s,
        time_step_definition="t_phys_seconds = step * dt_code * t_real_unit_s",
        reconstruction_status=status,
        source_params=params_path,
    )


def load_events(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    return [r for r in rows if r.get("event_status", "accepted") in {"", "accepted"}]


def write_physical_units_config(events: list[dict[str, str]], output: Path, J0: float) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in events:
        by_source[row.get("_source_file") or row.get("source_file") or "unknown"].append(row)
    run_configs = [reconstruct_run_config(source, rows) for source, rows in sorted(by_source.items())]
    strict_count = sum(1 for cfg in run_configs if cfg.reconstruction_status == "strict_physical")
    payload = {
        "schema_version": 1,
        "purpose": "CNT-vs-CUDA physical unit normalization",
        "dt_mapping_factor": "per-run t_real_unit_s; see run_configs",
        "voxel_volume": "per-run voxel_volume_m3; see run_configs",
        "system_volume": "per-run system_volume_m3; see run_configs",
        "time_step_definition": "t_phys = step * dt_code * t_real_unit_s; if time_physical_s exists it is used directly",
        "J0_m3_s": J0,
        "J0_definition": "CNT prefactor used consistently for normalized_cnt_prediction and observed_barrier_reconstruction",
        "J0_source": "command_line_or_default; not fitted from CUDA observations",
        "run_count": len(run_configs),
        "strict_physical_run_count": strict_count,
        "unit_consistency": "PASS" if strict_count == len(run_configs) and run_configs else "FAIL",
        "run_configs": [asdict(cfg) for cfg in run_configs],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def normalized_rates(events: list[dict[str, str]], config_payload: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    cfg_by_source = {cfg["source_file"]: cfg for cfg in config_payload["run_configs"]}
    events_by_condition: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    sources_by_condition: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for row in events:
        source = row.get("_source_file") or row.get("source_file") or "unknown"
        cfg = cfg_by_source.get(source, {})
        T = safe_float(row.get("T_C"), cfg.get("T_C"))
        xB = safe_float(row.get("local_xB"), cfg.get("xB"))
        strain = safe_float(row.get("strain"), cfg.get("strain", 0.0))
        key = normalize_key(T, xB, strain)
        events_by_condition[key].append(row)
        sources_by_condition[key].add(source)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(events_by_condition.items()):
        exposure = 0.0
        volume_values = []
        time_values = []
        statuses = []
        grid_keys = set()
        for source in sources_by_condition[key]:
            cfg = cfg_by_source.get(source, {})
            V = safe_float(cfg.get("system_volume_m3"))
            t = safe_float(cfg.get("total_time_s"))
            if V is not None and t is not None:
                exposure += V * t
                volume_values.append(V)
                time_values.append(t)
            statuses.append(str(cfg.get("reconstruction_status", "missing_config")))
            grid_keys.add(f"{cfg.get('Nx')}x{cfg.get('Ny')}x{cfg.get('Nz')}_dx{cfg.get('dx_nm')}")
        J_obs = len(items) / exposure if exposure > 0.0 else None
        rows.append(
            {
                "T_C": key[0],
                "xB": key[1],
                "strain": key[2],
                "N_nucleation_events": len(items),
                "n_source_runs": len(sources_by_condition[key]),
                "system_volume_m3_sum_exposure_basis": sum(volume_values) if volume_values else "",
                "time_window_s_sum_exposure_basis": sum(time_values) if time_values else "",
                "exposure_m3_s": exposure if exposure > 0.0 else "",
                "J_obs": J_obs if J_obs is not None else "",
                "J_obs_units": "m^-3_s^-1" if J_obs is not None else "not_available",
                "grid_resolution_keys": ";".join(sorted(grid_keys)),
                "normalization_status": "strict_physical" if all(s == "strict_physical" for s in statuses) and J_obs is not None else "incomplete_or_inferred_units",
                "snapshot_bias_control": "uses total run duration from steps*dt*t_real_unit when available; otherwise last-event time",
            }
        )
    fields = [
        "T_C",
        "xB",
        "strain",
        "N_nucleation_events",
        "n_source_runs",
        "system_volume_m3_sum_exposure_basis",
        "time_window_s_sum_exposure_basis",
        "exposure_m3_s",
        "J_obs",
        "J_obs_units",
        "grid_resolution_keys",
        "normalization_status",
        "snapshot_bias_control",
    ]
    write_csv(output, rows, fields)
    return rows


def load_prediction_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    if rows:
        return rows
    return read_csv(prefer_existing_path(CNT_ROOT / "cnt_prediction_table.csv", REPO_ROOT / "cnt_prediction_table.csv"))


def normalized_cnt_predictions(prediction_path: Path, output: Path, J0: float) -> list[dict[str, Any]]:
    rows = []
    for row in load_prediction_rows(prediction_path):
        T_C = safe_float(row.get("T_C"), safe_float(row.get("T")))
        xB = safe_float(row.get("xB"))
        strain = safe_float(row.get("strain"), safe_float(row.get("strain_value"), 0.0))
        barrier_kBT = safe_float(row.get("DeltaG_star_CNT_kBT"), safe_float(row.get("predicted_barrier"), safe_float(row.get("energy_barrier_kBT"))))
        if T_C is None or xB is None or barrier_kBT is None:
            continue
        J_CNT = J0 * math.exp(-barrier_kBT) if barrier_kBT < 745.0 else 0.0
        rows.append(
            {
                "condition_id": row.get("condition_id") or row.get("case_id") or f"T{T_C:g}_xB{xB:g}_strain{strain:g}",
                "T_C": T_C,
                "T_K": T_C + 273.15,
                "xB": xB,
                "strain": strain,
                "strain_mode": row.get("strain_mode", ""),
                "DeltaG_star_CNT_kBT": barrier_kBT,
                "DeltaG_star_CNT_J": barrier_kBT * K_B_J_PER_K * (T_C + 273.15),
                "r_star_CNT_nm": row.get("r_star_CNT_nm") or row.get("predicted_rc") or row.get("rc_nm") or "",
                "J0": J0,
                "J0_units": "m^-3_s^-1",
                "J_CNT": J_CNT,
                "J_CNT_units": "m^-3_s^-1",
                "temperature_scaling": "DeltaG_star_CNT_J = DeltaG_star_CNT_kBT * kB * T_K; J_CNT = J0 * exp(-DeltaG_star_CNT_kBT)",
                "prediction_source": row.get("prediction_source", str(prediction_path)),
            }
        )
    fields = [
        "condition_id",
        "T_C",
        "T_K",
        "xB",
        "strain",
        "strain_mode",
        "DeltaG_star_CNT_kBT",
        "DeltaG_star_CNT_J",
        "r_star_CNT_nm",
        "J0",
        "J0_units",
        "J_CNT",
        "J_CNT_units",
        "temperature_scaling",
        "prediction_source",
    ]
    write_csv(output, rows, fields)
    return rows


def reconstruct_observed_barriers(rate_rows: list[dict[str, Any]], output: Path, J0: float) -> list[dict[str, Any]]:
    rows = []
    for row in rate_rows:
        T_C = safe_float(row.get("T_C"))
        J_obs = safe_float(row.get("J_obs"))
        if T_C is None or J_obs is None or J_obs <= 0.0 or J0 <= 0.0:
            barrier_kBT = None
            barrier_J = None
            status = "missing_rate_or_temperature"
        else:
            barrier_kBT = -math.log(max(J_obs / J0, 1.0e-300))
            barrier_J = barrier_kBT * K_B_J_PER_K * (T_C + 273.15)
            status = "ok" if barrier_kBT >= 0.0 else "negative_barrier_rate_exceeds_prefactor"
        rows.append(
            {
                "T_C": row.get("T_C", ""),
                "T_K": (T_C + 273.15) if T_C is not None else "",
                "xB": row.get("xB", ""),
                "strain": row.get("strain", ""),
                "J_obs": row.get("J_obs", ""),
                "J0": J0,
                "DeltaG_obs_kBT": barrier_kBT if barrier_kBT is not None else "",
                "DeltaG_obs_J": barrier_J if barrier_J is not None else "",
                "barrier_reconstruction_status": status,
                "reference_state": "same J0 as normalized_cnt_prediction; no fitted scaling",
            }
        )
    fields = [
        "T_C",
        "T_K",
        "xB",
        "strain",
        "J_obs",
        "J0",
        "DeltaG_obs_kBT",
        "DeltaG_obs_J",
        "barrier_reconstruction_status",
        "reference_state",
    ]
    write_csv(output, rows, fields)
    return rows


def nearest_prediction(rate_row: dict[str, Any], predictions: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    T = safe_float(rate_row.get("T_C"))
    xB = safe_float(rate_row.get("xB"))
    strain = safe_float(rate_row.get("strain"), 0.0)
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


def quantitative_validation(
    rate_rows: list[dict[str, Any]],
    barrier_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    output: Path,
) -> list[dict[str, Any]]:
    barrier_by_key = {normalize_key(r.get("T_C"), r.get("xB"), r.get("strain")): r for r in barrier_rows}
    rows = []
    for rate in rate_rows:
        pred, mismatch = nearest_prediction(rate, prediction_rows)
        key = normalize_key(rate.get("T_C"), rate.get("xB"), rate.get("strain"))
        obs_barrier = barrier_by_key.get(key, {})
        if pred is None:
            rows.append({"condition_key": "|".join(key), "comparison_status": "missing_prediction"})
            continue
        J_obs = safe_float(rate.get("J_obs"))
        J_CNT = safe_float(pred.get("J_CNT"))
        DG_obs = safe_float(obs_barrier.get("DeltaG_obs_kBT"))
        DG_CNT = safe_float(pred.get("DeltaG_star_CNT_kBT"))
        rate_error = abs(J_CNT - J_obs) / J_CNT if J_CNT is not None and J_CNT > 0.0 and J_obs is not None else None
        barrier_error = abs(DG_CNT - DG_obs) if DG_CNT is not None and DG_obs is not None else None
        status = "ok"
        if rate.get("normalization_status") != "strict_physical":
            status = "unit_incomplete"
        if mismatch > 0.25:
            status = "condition_mismatch"
        if rate_error is None or barrier_error is None:
            status = "missing_comparable_quantity"
        rows.append(
            {
                "condition_key": "|".join(key),
                "T_C_obs": rate.get("T_C", ""),
                "xB_obs": rate.get("xB", ""),
                "strain_obs": rate.get("strain", ""),
                "J_obs": J_obs if J_obs is not None else "",
                "J_CNT": J_CNT if J_CNT is not None else "",
                "rate_error": rate_error if rate_error is not None else "",
                "DeltaG_obs_kBT": DG_obs if DG_obs is not None else "",
                "DeltaG_CNT_kBT": DG_CNT if DG_CNT is not None else "",
                "barrier_error_kBT": barrier_error if barrier_error is not None else "",
                "matched_T_C_CNT": pred.get("T_C", ""),
                "matched_xB_CNT": pred.get("xB", ""),
                "matched_strain_CNT": pred.get("strain", ""),
                "prediction_mismatch_score": mismatch,
                "normalization_status": rate.get("normalization_status", ""),
                "comparison_status": status,
            }
        )
    fields = [
        "condition_key",
        "T_C_obs",
        "xB_obs",
        "strain_obs",
        "J_obs",
        "J_CNT",
        "rate_error",
        "DeltaG_obs_kBT",
        "DeltaG_CNT_kBT",
        "barrier_error_kBT",
        "matched_T_C_CNT",
        "matched_xB_CNT",
        "matched_strain_CNT",
        "prediction_mismatch_score",
        "normalization_status",
        "comparison_status",
    ]
    write_csv(output, rows, fields)
    return rows


def ranking_status(rows: list[dict[str, Any]], obs_key: str, pred_key: str) -> str:
    pairs = [(safe_float(r.get(obs_key)), safe_float(r.get(pred_key))) for r in rows if r.get("comparison_status") == "ok"]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 2:
        return "not_enough_data"
    obs_order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    pred_order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    return "preserved" if obs_order == pred_order else "not_preserved"


def score_from_errors(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v) and v >= 0.0]
    if not vals:
        return 0.0
    vals.sort()
    median = vals[len(vals) // 2]
    return 1.0 / (1.0 + median)


def write_summary(path: Path, config: dict[str, Any], validation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in validation_rows if r.get("comparison_status") == "ok"]
    rate_score = score_from_errors([safe_float(r.get("rate_error"), math.nan) for r in ok_rows])
    barrier_score = score_from_errors([safe_float(r.get("barrier_error_kBT"), math.nan) for r in ok_rows])
    unit_consistency = config.get("unit_consistency", "FAIL")
    rate_rank = ranking_status(validation_rows, "J_obs", "J_CNT")
    barrier_rank = ranking_status(validation_rows, "DeltaG_obs_kBT", "DeltaG_CNT_kBT")
    if unit_consistency == "PASS" and ok_rows and rate_score > 0.5 and barrier_score > 0.5:
        validity = "YES"
        stage = "quantitative"
        classification = "A. QUANTITATIVELY VALIDATED SYSTEM"
    elif unit_consistency == "PASS" and ok_rows:
        validity = "NO"
        stage = "semi-quantitative"
        classification = "B. SEMI-QUANTITATIVE SYSTEM"
    else:
        validity = "NO"
        stage = "qualitative"
        classification = "C. INCONSISTENT UNIT SYSTEM"
    lines = [
        "# Quantitative Validation Summary",
        "",
        "## Unit Reconstruction",
        "",
        f"- unit_consistency: {unit_consistency}",
        f"- run_count: {config.get('run_count', 0)}",
        f"- strict_physical_run_count: {config.get('strict_physical_run_count', 0)}",
        f"- J0_m3_s: {config.get('J0_m3_s')}",
        "",
        "## Consistency Scores",
        "",
        f"- rate_consistency_score: {rate_score:.6g}",
        f"- barrier_consistency_score: {barrier_score:.6g}",
        f"- CNT_predictive_validity: {validity}",
        f"- system_stage: {stage}",
        f"- final_classification: {classification}",
        "",
        "## Physical Checks",
        "",
        f"- J_obs independent of grid resolution: {'not_enough_data' if len(set(r.get('normalization_status') for r in validation_rows)) <= 1 else 'requires_multi_grid_audit'}",
        "- J_obs scales correctly with system size: not_enough_data",
        "- DeltaG_obs stable across sampling windows: not_enough_data",
        f"- CNT ranking preserved after normalization, rate: {rate_rank}",
        f"- CNT ranking preserved after normalization, barrier: {barrier_rank}",
        "",
        "## Caveat",
        "",
        "The layer reconstructs units from available logs and `pf_input.params`. If a run lacks physical time, volume, or matched CNT conditions, it is not counted as a quantitative validation point.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "unit_consistency": unit_consistency,
        "rate_consistency_score": rate_score,
        "barrier_consistency_score": barrier_score,
        "CNT_predictive_validity": validity,
        "system_stage": stage,
        "classification": classification,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    events = load_events(args.event_log)
    config = write_physical_units_config(events, args.units_config, args.J0)
    rate_rows = normalized_rates(events, config, args.normalized_rate)
    prediction_rows = normalized_cnt_predictions(args.prediction_table, args.normalized_cnt, args.J0)
    barrier_rows = reconstruct_observed_barriers(rate_rows, args.observed_barrier, args.J0)
    validation_rows = quantitative_validation(rate_rows, barrier_rows, prediction_rows, args.validation_table)
    return write_summary(args.validation_summary, config, validation_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-log",
        type=Path,
        default=prefer_existing_path(REPORTS_ROOT / "nucleation/nucleation_event_log.csv", REPO_ROOT / "nucleation_event_log.csv"),
    )
    parser.add_argument(
        "--prediction-table",
        type=Path,
        default=prefer_existing_path(CNT_ROOT / "cnt_prediction_table.csv", REPO_ROOT / "cnt_prediction_table.csv"),
    )
    parser.add_argument("--J0", type=float, default=1.0, help="CNT prefactor in m^-3 s^-1; reused for observation barrier reconstruction.")
    parser.add_argument("--units-config", type=Path, default=REPO_ROOT / "physical_units_config.json")
    parser.add_argument("--normalized-rate", type=Path, default=DATA_ROOT / "normalized_nucleation_rate.csv")
    parser.add_argument("--normalized-cnt", type=Path, default=CNT_ROOT / "normalized_cnt_prediction.csv")
    parser.add_argument("--observed-barrier", type=Path, default=VALIDATION_ROOT / "observed_barrier_reconstruction.csv")
    parser.add_argument("--validation-table", type=Path, default=VALIDATION_ROOT / "quantitative_validation_table.csv")
    parser.add_argument("--validation-summary", type=Path, default=VALIDATION_ROOT / "quantitative_validation_summary.md")
    args = parser.parse_args()

    summary = run(args)
    print(f"unit_consistency = {summary['unit_consistency']}")
    print(f"rate_consistency_score = {summary['rate_consistency_score']:.6g}")
    print(f"barrier_consistency_score = {summary['barrier_consistency_score']:.6g}")
    print(f"CNT_predictive_validity = {summary['CNT_predictive_validity']}")
    print(f"system_stage = {summary['system_stage']}")


if __name__ == "__main__":
    main()
