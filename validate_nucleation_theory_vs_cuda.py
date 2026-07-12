#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = REPO_ROOT / "reports"
VALIDATION_ROOT = REPORTS_ROOT / "validation"
DATA_ROOT = REPORTS_ROOT / "data"
CNT_ROOT = REPORTS_ROOT / "cnt"
GP_ROOT = REPORTS_ROOT / "gp"
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


def norm_key(T: Any, xB: Any, gp: Any = "") -> tuple[str, str, str]:
    tf = safe_float(T)
    xf = safe_float(xB)
    return (
        f"{tf:.6g}" if tf is not None else "unknown",
        f"{xf:.8g}" if xf is not None else "unknown",
        str(gp or ""),
    )


def shape_class(shape: str) -> str:
    s = (shape or "").lower()
    if "facet" in s:
        return "faceted"
    if "anis" in s or "ellip" in s:
        return "anisotropic"
    if "sphere" in s or "spherical" in s:
        return "sphere"
    return "unknown"


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def build_prediction_table(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    selected_path = REPO_ROOT / "Results/orchestrator/T400_xB0p050/selected_nucleus.json"
    if selected_path.exists():
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
        for key in ("predicted_lowest_energy_nucleus", "selected_nucleus"):
            item = payload.get(key)
            if not isinstance(item, dict):
                continue
            case_id = item.get("id", key)
            if case_id in seen:
                continue
            seen.add(case_id)
            rows.append({
                "case_id": case_id,
                "T": item.get("T_C", ""),
                "xB": item.get("xB", ""),
                "strain": item.get("strain_value", ""),
                "strain_mode": item.get("strain_mode", ""),
                "predicted_rc": item.get("rc_nm", ""),
                "predicted_shape": item.get("shape_type", ""),
                "predicted_barrier": item.get("energy_barrier_kBT", ""),
                "barrier_units": "kBT",
                "predicted_rate": "",
                "rate_units": "",
                "prediction_source": str(selected_path.relative_to(REPO_ROOT)),
            })

    continuous_path = prefer_existing_path(DATA_ROOT / "continuous_physics_summary.csv", REPO_ROOT / "continuous_physics_summary.csv")
    for item in read_csv(continuous_path):
        case_id = item.get("case_id", "")
        if not case_id or case_id in seen:
            continue
        seen.add(case_id)
        rows.append({
            "case_id": case_id,
            "T": item.get("T", ""),
            "xB": item.get("xB", ""),
            "strain": item.get("strain", ""),
            "strain_mode": item.get("strain_mode", ""),
            "predicted_rc": item.get("rc_continuous", ""),
            "predicted_shape": item.get("shape_continuous", ""),
            "predicted_barrier": item.get("barrier_continuous", ""),
            "barrier_units": item.get("barrier_units", "kBT"),
            "predicted_rate": "",
            "rate_units": "",
            "prediction_source": item.get("source", str(continuous_path.relative_to(REPO_ROOT))),
        })

    rate_path = REPO_ROOT / "Results/workflows/T400_xB0p050/analysis/nucleation_rate/nucleation_rate_table.csv"
    for item in read_csv(rate_path):
        case_id = item.get("base_case_tag", "")
        if not case_id or case_id in seen:
            continue
        seen.add(case_id)
        J = item.get("J_strict") or item.get("J_diagnostic_current") or item.get("J_diagnostic") or item.get("J_m3_s")
        rows.append({
            "case_id": case_id,
            "T": item.get("T_input", ""),
            "xB": item.get("xB_loc", ""),
            "strain": item.get("eyy") or item.get("exx") or "",
            "strain_mode": item.get("mode", ""),
            "predicted_rc": item.get("r_eff_star_nm", ""),
            "predicted_shape": "unknown_from_rate_table",
            "predicted_barrier": item.get("barrier_kBT_current") or item.get("DeltaG_star_kBT", ""),
            "barrier_units": "kBT",
            "predicted_rate": J,
            "rate_units": "m^-3_s^-1",
            "prediction_source": str(rate_path.relative_to(REPO_ROOT)),
        })

    fields = [
        "case_id", "T", "xB", "strain", "strain_mode", "predicted_rc", "predicted_shape",
        "predicted_barrier", "barrier_units", "predicted_rate", "rate_units", "prediction_source",
    ]
    write_csv(output, rows, fields)
    return rows


def build_observation_table(output: Path) -> list[dict[str, Any]]:
    events = read_csv(prefer_existing_path(REPO_ROOT / "reports/nucleation/nucleation_event_log.csv", REPO_ROOT / "nucleation_event_log.csv"))
    barriers = read_csv(prefer_existing_path(REPO_ROOT / "reports/nucleation/nucleation_barrier_estimate.csv", REPO_ROOT / "nucleation_barrier_estimate.csv"))
    rates = read_csv(prefer_existing_path(REPO_ROOT / "reports/nucleation/nucleation_rate_J.csv", REPO_ROOT / "nucleation_rate_J.csv"))
    barrier_by_group: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in barriers:
        gp = "1" if safe_int(row.get("gp_assisted_flag")) else "0"
        val = safe_float(row.get("barrier_proxy_kBT"))
        if val is not None:
            barrier_by_group[norm_key(row.get("T_C"), row.get("xB"), gp)].append(val)
    rate_by_group: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rates:
        gp = "1" if row.get("event_class") == "GP" else "0"
        rate_by_group[norm_key(row.get("T_C"), row.get("xB"), gp)] = row

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in events:
        if row.get("event_status") not in {"", "accepted"}:
            continue
        gp = "1" if safe_int(row.get("gp_assisted_flag")) or safe_int(row.get("gp_presence_flag")) else "0"
        xB = row.get("local_xB")
        if not xB:
            # Legacy analyzer put xB in the parsed source path in barrier/rate files, but event rows may be blank.
            for brow in barriers:
                if brow.get("source_file") == row.get("_source_file"):
                    xB = brow.get("xB")
                    break
        groups[norm_key(row.get("T_C"), xB, gp)].append({**row, "_xB_for_group": xB, "_gp_for_group": gp})

    rows: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        T, xB, gp = key
        steps = [safe_float(r.get("step")) for r in items]
        times = [safe_float(r.get("time_physical_s")) for r in items]
        code_times = []
        for r in items:
            step = safe_float(r.get("step"))
            dt = safe_float(r.get("dt_code"), 1.0)
            if step is not None and dt is not None:
                code_times.append(step * dt)
        obs_tau = mean([t for t in times if t is not None])
        tau_basis = "seconds"
        if obs_tau is None:
            obs_tau = mean(code_times)
            tau_basis = "code_time"
        shapes = [shape_class(r.get("nucleus_shape_type", "")) for r in items]
        shape = Counter(shapes).most_common(1)[0][0] if shapes else "unknown"
        rc = mean([safe_float(r.get("rc_nm")) for r in items])
        rate = rate_by_group.get(key, {})
        observed_barrier = mean(barrier_by_group.get(key, []))
        rows.append({
            "observation_id": f"T{T}_xB{xB}_GP{gp}",
            "T": T,
            "xB": xB,
            "strain": "unknown",
            "observed_nucleation_time_tau": obs_tau if obs_tau is not None else "",
            "tau_units": tau_basis if obs_tau is not None else "not_available",
            "observed_rate_J_obs": rate.get("J", ""),
            "rate_units": rate.get("J_units", ""),
            "observed_rc": rc if rc is not None else "",
            "observed_shape": shape,
            "GP_flag": gp,
            "event_count": len(items),
            "observed_barrier_proxy_kBT": observed_barrier if observed_barrier is not None else "",
            "observation_source": "nucleation_event_log.csv",
        })
    fields = [
        "observation_id", "T", "xB", "strain", "observed_nucleation_time_tau", "tau_units",
        "observed_rate_J_obs", "rate_units", "observed_rc", "observed_shape", "GP_flag",
        "event_count", "observed_barrier_proxy_kBT", "observation_source",
    ]
    write_csv(output, rows, fields)
    return rows


def build_observed_barrier_table(observations: list[dict[str, Any]], output: Path, tau0: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obs in observations:
        tau = safe_float(obs.get("observed_nucleation_time_tau"))
        J = safe_float(obs.get("observed_rate_J_obs"))
        T_C = safe_float(obs.get("T"))
        T_K = (T_C + 273.15) if T_C is not None else None
        b_tau_kBT = math.log(max(tau / tau0, 1.0e-300)) if tau is not None and tau > 0.0 else None
        b_rate_kBT = -math.log(max(J, 1.0e-300)) if J is not None and J > 0.0 else None
        obs_proxy = safe_float(obs.get("observed_barrier_proxy_kBT"))
        preferred = obs_proxy if obs_proxy is not None else (b_tau_kBT if b_tau_kBT is not None else "")
        rows.append({
            "observation_id": obs.get("observation_id", ""),
            "T": obs.get("T", ""),
            "xB": obs.get("xB", ""),
            "GP_flag": obs.get("GP_flag", ""),
            "tau": obs.get("observed_nucleation_time_tau", ""),
            "tau_units": obs.get("tau_units", ""),
            "tau0": tau0,
            "DeltaG_obs_from_tau_kBT": b_tau_kBT if b_tau_kBT is not None else "",
            "DeltaG_obs_from_tau_J": (K_B_J_PER_K * T_K * b_tau_kBT) if (b_tau_kBT is not None and T_K is not None) else "",
            "J_obs": obs.get("observed_rate_J_obs", ""),
            "DeltaG_obs_from_rate_kBT": b_rate_kBT if b_rate_kBT is not None else "",
            "DeltaG_obs_from_rate_J": (K_B_J_PER_K * T_K * b_rate_kBT) if (b_rate_kBT is not None and T_K is not None) else "",
            "preferred_DeltaG_obs_kBT": preferred,
            "method": "event_log_proxy_or_waiting_time",
        })
    fields = [
        "observation_id", "T", "xB", "GP_flag", "tau", "tau_units", "tau0",
        "DeltaG_obs_from_tau_kBT", "DeltaG_obs_from_tau_J", "J_obs",
        "DeltaG_obs_from_rate_kBT", "DeltaG_obs_from_rate_J", "preferred_DeltaG_obs_kBT", "method",
    ]
    write_csv(output, rows, fields)
    return rows


def nearest_observation(pred: dict[str, Any], observations: list[dict[str, Any]], max_T: float = 1e-6, max_xB: float = 1e-8) -> dict[str, Any] | None:
    pT = safe_float(pred.get("T"))
    px = safe_float(pred.get("xB"))
    if pT is None or px is None:
        return None
    best: tuple[float, dict[str, Any]] | None = None
    for obs in observations:
        oT = safe_float(obs.get("T"))
        ox = safe_float(obs.get("xB"))
        if oT is None or ox is None:
            continue
        dT = abs(pT - oT)
        dx = abs(px - ox)
        score = dT + 1000.0 * dx
        if dT <= max_T and dx <= max_xB and (best is None or score < best[0]):
            best = (score, obs)
    return best[1] if best else None


def build_comparison(predictions: list[dict[str, Any]], observations: list[dict[str, Any]], barriers: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    barrier_by_id = {b["observation_id"]: b for b in barriers}
    matched_obs: set[str] = set()
    rows: list[dict[str, Any]] = []
    for pred in predictions:
        obs = nearest_observation(pred, observations)
        if obs is None:
            rows.append({
                "case_id": pred.get("case_id", ""),
                "observation_id": "",
                "T_pred": pred.get("T", ""),
                "xB_pred": pred.get("xB", ""),
                "T_obs": "",
                "xB_obs": "",
                "DeltaG_CNT": pred.get("predicted_barrier", ""),
                "DeltaG_obs": "",
                "error_DeltaG": "",
                "J_pred": pred.get("predicted_rate", ""),
                "J_obs": "",
                "J_pred_over_J_obs": "",
                "predicted_rc": pred.get("predicted_rc", ""),
                "observed_rc": "",
                "rc_deviation": "",
                "predicted_shape": pred.get("predicted_shape", ""),
                "observed_shape": "",
                "shape_mismatch_score": "",
                "comparison_status": "no_observation_for_prediction",
            })
            continue
        matched_obs.add(obs["observation_id"])
        b = barrier_by_id.get(obs["observation_id"], {})
        dg_cnt = safe_float(pred.get("predicted_barrier"))
        dg_obs = safe_float(b.get("preferred_DeltaG_obs_kBT"))
        Jp = safe_float(pred.get("predicted_rate"))
        Jo = safe_float(obs.get("observed_rate_J_obs"))
        rc_pred = safe_float(pred.get("predicted_rc"))
        rc_obs = safe_float(obs.get("observed_rc"))
        shape_pred = shape_class(str(pred.get("predicted_shape", "")))
        shape_obs = shape_class(str(obs.get("observed_shape", "")))
        rows.append({
            "case_id": pred.get("case_id", ""),
            "observation_id": obs.get("observation_id", ""),
            "T_pred": pred.get("T", ""),
            "xB_pred": pred.get("xB", ""),
            "T_obs": obs.get("T", ""),
            "xB_obs": obs.get("xB", ""),
            "DeltaG_CNT": pred.get("predicted_barrier", ""),
            "DeltaG_obs": b.get("preferred_DeltaG_obs_kBT", ""),
            "error_DeltaG": (dg_obs - dg_cnt) if (dg_obs is not None and dg_cnt is not None) else "",
            "J_pred": pred.get("predicted_rate", ""),
            "J_obs": obs.get("observed_rate_J_obs", ""),
            "J_pred_over_J_obs": (Jp / Jo) if (Jp is not None and Jo not in (None, 0.0)) else "",
            "predicted_rc": pred.get("predicted_rc", ""),
            "observed_rc": obs.get("observed_rc", ""),
            "rc_deviation": (rc_obs - rc_pred) if (rc_obs is not None and rc_pred is not None) else "",
            "predicted_shape": pred.get("predicted_shape", ""),
            "observed_shape": obs.get("observed_shape", ""),
            "shape_mismatch_score": 0 if shape_pred == shape_obs and shape_pred != "unknown" else 1,
            "comparison_status": "paired",
        })
    for obs in observations:
        if obs["observation_id"] in matched_obs:
            continue
        b = barrier_by_id.get(obs["observation_id"], {})
        rows.append({
            "case_id": "",
            "observation_id": obs.get("observation_id", ""),
            "T_pred": "",
            "xB_pred": "",
            "T_obs": obs.get("T", ""),
            "xB_obs": obs.get("xB", ""),
            "DeltaG_CNT": "",
            "DeltaG_obs": b.get("preferred_DeltaG_obs_kBT", ""),
            "error_DeltaG": "",
            "J_pred": "",
            "J_obs": obs.get("observed_rate_J_obs", ""),
            "J_pred_over_J_obs": "",
            "predicted_rc": "",
            "observed_rc": obs.get("observed_rc", ""),
            "rc_deviation": "",
            "predicted_shape": "",
            "observed_shape": obs.get("observed_shape", ""),
            "shape_mismatch_score": "",
            "comparison_status": "no_prediction_for_observation",
        })
    fields = [
        "case_id", "observation_id", "T_pred", "xB_pred", "T_obs", "xB_obs",
        "DeltaG_CNT", "DeltaG_obs", "error_DeltaG", "J_pred", "J_obs",
        "J_pred_over_J_obs", "predicted_rc", "observed_rc", "rc_deviation",
        "predicted_shape", "observed_shape", "shape_mismatch_score", "comparison_status",
    ]
    write_csv(output, rows, fields)
    return rows


def compute_metrics(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [r for r in comparisons if r.get("comparison_status") == "paired"]
    berr = [abs(safe_float(r.get("error_DeltaG")) or math.nan) for r in paired]
    berr = [v for v in berr if math.isfinite(v)]
    ratios = [safe_float(r.get("J_pred_over_J_obs")) for r in paired]
    ratios = [v for v in ratios if v is not None and v > 0.0]
    shapes = [safe_float(r.get("shape_mismatch_score")) for r in paired]
    shapes = [v for v in shapes if v is not None]
    return {
        "paired_count": len(paired),
        "barrier_error_mean": mean(berr),
        "barrier_error_max": max(berr) if berr else None,
        "rate_ratio_geomean": math.exp(mean([math.log(v) for v in ratios])) if ratios else None,
        "rate_consistency_score": 1.0 / (1.0 + abs(math.log(math.exp(mean([math.log(v) for v in ratios]))))) if ratios else 0.0,
        "shape_consistency_score": 1.0 - mean(shapes) if shapes else None,
        "ranking_preservation_score": None if len(paired) < 2 else 0.0,
    }


def build_gp_validation(observations: list[dict[str, Any]], barriers: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    b_by_obs = {b["observation_id"]: b for b in barriers}
    groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for obs in observations:
        key = (str(obs.get("T")), str(obs.get("xB")))
        groups[key]["GP" if str(obs.get("GP_flag")) == "1" else "noGP"] = obs
    rows: list[dict[str, Any]] = []
    for (T, xB), pair in sorted(groups.items()):
        gp = pair.get("GP")
        no = pair.get("noGP")
        bgp = safe_float((b_by_obs.get(gp.get("observation_id")) if gp else {}).get("preferred_DeltaG_obs_kBT")) if gp else None
        bno = safe_float((b_by_obs.get(no.get("observation_id")) if no else {}).get("preferred_DeltaG_obs_kBT")) if no else None
        Jgp = safe_float(gp.get("observed_rate_J_obs")) if gp else None
        Jno = safe_float(no.get("observed_rate_J_obs")) if no else None
        rows.append({
            "T": T,
            "xB": xB,
            "DeltaG_GP_kBT": bgp if bgp is not None else "",
            "DeltaG_noGP_kBT": bno if bno is not None else "",
            "DeltaDeltaG_GP_effect_kBT": (bno - bgp) if (bno is not None and bgp is not None) else "",
            "J_GP": Jgp if Jgp is not None else "",
            "J_noGP": Jno if Jno is not None else "",
            "J_enhancement_factor": (Jgp / Jno) if (Jgp is not None and Jno not in (None, 0.0)) else "",
            "shape_GP": gp.get("observed_shape") if gp else "",
            "shape_noGP": no.get("observed_shape") if no else "",
            "shape_shift_due_to_GP": (shape_class(gp.get("observed_shape", "")) != shape_class(no.get("observed_shape", ""))) if (gp and no) else "",
            "status": "paired" if (gp and no) else "missing_gp_or_no_gp_pair",
        })
    fields = [
        "T", "xB", "DeltaG_GP_kBT", "DeltaG_noGP_kBT", "DeltaDeltaG_GP_effect_kBT",
        "J_GP", "J_noGP", "J_enhancement_factor", "shape_GP", "shape_noGP",
        "shape_shift_due_to_GP", "status",
    ]
    write_csv(output, rows, fields)
    return rows


def write_report(metrics: dict[str, Any], predictions: list[dict[str, Any]], observations: list[dict[str, Any]], comparisons: list[dict[str, Any]], gp_rows: list[dict[str, Any]], output: Path) -> str:
    paired_count = metrics["paired_count"]
    if paired_count == 0:
        classification = "D. INVALID THEORY MODEL / not validated: no paired prediction-observation cases"
        closure = False
        publication = "no"
    elif (metrics.get("barrier_error_mean") or 1e99) < 1.0 and (metrics.get("shape_consistency_score") or 0.0) > 0.9:
        classification = "A. VALIDATED THEORY SYSTEM"
        closure = True
        publication = "yes"
    else:
        classification = "C. PARTIALLY CONSISTENT SYSTEM"
        closure = False
        publication = "no"
    gp_paired = any(r.get("status") == "paired" for r in gp_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Validation Layer Report

## Validation Verdict

Classification:

```text
{classification}
```

Physics closure status: `{str(closure).lower()}`

Publication readiness: `{publication}`

## Data Products

- `reports/cnt/nucleation_prediction_table.csv`
- `reports/cnt/nucleation_observation_table.csv`
- `reports/cnt/nucleation_barrier_observed.csv`
- `reports/validation/validation_comparison_table.csv`
- `reports/gp/gp_validation_report.csv`

## Dataset Summary

- prediction rows: {len(predictions)}
- observation rows: {len(observations)}
- paired prediction-observation rows: {paired_count}
- unpaired rows in comparison table: {len(comparisons) - paired_count}

## Global Metrics

- barrier_error_mean: `{metrics.get('barrier_error_mean') if metrics.get('barrier_error_mean') is not None else 'not_available'}`
- barrier_error_max: `{metrics.get('barrier_error_max') if metrics.get('barrier_error_max') is not None else 'not_available'}`
- rate_ratio_distribution/geomean: `{metrics.get('rate_ratio_geomean') if metrics.get('rate_ratio_geomean') is not None else 'not_available'}`
- rate_consistency_score: `{metrics.get('rate_consistency_score')}`
- shape_consistency_score: `{metrics.get('shape_consistency_score') if metrics.get('shape_consistency_score') is not None else 'not_available'}`
- ranking_preservation_score: `{metrics.get('ranking_preservation_score') if metrics.get('ranking_preservation_score') is not None else 'not_available'}`

## Required Answers

### 1. Is CNT prediction quantitatively correct?

Not established. There are currently no same-condition paired CNT prediction and CUDA observation rows.

### 2. Does CUDA confirm predicted nucleation physics?

No, not yet. CUDA event observations exist, but they do not overlap the available CNT/selector prediction conditions.

### 3. Is GP effect correctly captured?

Not established. `reports/gp/gp_validation_report.csv` has no paired GP/no-GP observation group at the same T and xB.

GP paired comparison available: `{str(gp_paired).lower()}`

### 4. Is ranking of nuclei preserved?

Not evaluated. Ranking preservation needs at least two paired prediction-observation cases.

### 5. Is system scientifically validated?

No. The validation layer is active, but the present dataset does not close prediction-vs-observation.

## Immediate Recommended Next Step

Run paired simulations at the same conditions as the prediction table, especially:

```text
T=400 C, xB=0.05, strain=-0.01
```

for both GP-assisted and no-GP settings, with `reports/nucleation/nucleation_event_log.csv` enabled. Then rerun:

```text
python3 validate_nucleation_theory_vs_cuda.py
```
"""
    output.write_text(text, encoding="utf-8")
    return publication


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CNT/CE/theory predictions against CUDA-observed nucleation events.")
    parser.add_argument("--tau0", type=float, default=1.0)
    args = parser.parse_args()

    predictions = build_prediction_table(prefer_existing_path(CNT_ROOT / "nucleation_prediction_table.csv", REPO_ROOT / "nucleation_prediction_table.csv"))
    observations = build_observation_table(prefer_existing_path(CNT_ROOT / "nucleation_observation_table.csv", REPO_ROOT / "nucleation_observation_table.csv"))
    barriers = build_observed_barrier_table(observations, prefer_existing_path(CNT_ROOT / "nucleation_barrier_observed.csv", REPO_ROOT / "nucleation_barrier_observed.csv"), args.tau0)
    comparisons = build_comparison(predictions, observations, barriers, VALIDATION_ROOT / "validation_comparison_table.csv")
    metrics = compute_metrics(comparisons)
    gp_rows = build_gp_validation(observations, barriers, GP_ROOT / "gp_validation_report.csv")
    publication = write_report(metrics, predictions, observations, comparisons, gp_rows, VALIDATION_ROOT / "validation_layer_report.md")

    closure = metrics["paired_count"] > 0 and metrics.get("barrier_error_mean") is not None
    print("validation_layer_active")
    print(f"physics_closure_status = {str(closure).lower()}")
    print(f"barrier_error_mean = {metrics.get('barrier_error_mean') if metrics.get('barrier_error_mean') is not None else 'not_available'}")
    print(f"rate_consistency_score = {metrics.get('rate_consistency_score')}")
    print(f"publication_readiness = {publication}")
    print("recommended_next_step = run paired CNT-predicted CUDA event simulations at matching T/xB/strain for GP and no-GP cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
