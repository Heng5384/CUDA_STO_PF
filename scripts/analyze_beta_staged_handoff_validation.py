#!/usr/bin/env python3
"""Analyze staged GP-to-beta accumulation and resolved handoff logs.

This parser is intentionally log-level only. It does not alter simulation
inputs or infer new physics; it converts runtime diagnostics into the CSV and
report artifacts required by the staged handoff validation workflow.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ \n]+)")


def parse_kv(line: str) -> dict[str, str]:
    return dict(KV_RE.findall(line))


def as_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        return float(str(value).strip().strip(","))
    except Exception:
        return default


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(float(str(value).strip().strip(",")))
    except Exception:
        return default


def finite_or_blank(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.12e}"


def nearest_leq(history: dict[int, dict[str, float]], step: int) -> dict[str, float]:
    if not history:
        return {}
    candidates = [s for s in history if s <= step]
    if not candidates:
        return {}
    return history[max(candidates)]


def detect_runtime_nan_inf(text: str) -> bool:
    bad_patterns = [
        r"\bNaN_Inf\s*=\s*(true|1)\b",
        r"\bnan_count\s*=\s*[1-9]\d*\b",
        r"\binf_count\s*=\s*[1-9]\d*\b",
        r"\bCUDA.*\bNaN\b",
        r"\bruntime.*\bNaN\b",
        r"\boverflow\b",
        r"\bblow[- ]?up\b",
    ]
    return any(re.search(pat, text, re.IGNORECASE) for pat in bad_patterns)


def parse_log(stdout_path: Path, case: str, T_C: float, S: float) -> dict[str, Any]:
    text = stdout_path.read_text(errors="ignore")
    lines = text.splitlines()
    rates: dict[int, dict[str, float]] = {}
    states: dict[int, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    resolved_rows: list[dict[str, Any]] = []

    gp_births = 0
    embryos_created = 0
    dt_s = math.nan
    for line in lines:
        if "[GP-LITERATURE-RATE]" in line:
            d = parse_kv(line)
            step = as_int(d.get("step"))
            rates[step] = {
                "xB_source_for_JGP": as_float(d.get("xB_source_for_JGP", d.get("xB_source"))),
                "xAg_used_for_JGP": as_float(d.get("xAg_used_for_JGP", d.get("xAg_used"))),
                "J_GP_m3_s": as_float(d.get("J_GP_m3_s")),
            }
            if math.isfinite(as_float(d.get("dt_s"))):
                dt_s = as_float(d.get("dt_s"))
        elif "[GP-LITERATURE-STATE]" in line:
            d = parse_kv(line)
            step = as_int(d.get("step"))
            states[step] = {
                "global_mass_error_rel": abs(as_float(d.get("mass_error_rel"), 0.0)),
                "xB_source_for_JGP": as_float(d.get("xB_source_for_JGP", d.get("xB_source"))),
            }
            gp_births = max(gp_births, as_int(d.get("births_accepted_total"), gp_births))
        elif "BETA_STAGED_EMBRYO_BEGIN" in line:
            d = parse_kv(line)
            step = as_int(d.get("step"))
            embryos_created += 1
            rate = nearest_leq(rates, step)
            state = nearest_leq(states, step)
            current = as_float(d.get("current_embryo_inventory"))
            target = as_float(d.get("target_seed_inventory"))
            row = {
                "case": case,
                "step": step,
                "time_s": step * dt_s if math.isfinite(dt_s) else math.nan,
                "T_C": T_C,
                "S": S,
                "embryo_id": d.get("embryo_id", ""),
                "embryo_status": d.get("status", ""),
                "target_seed_inventory": target,
                "current_inventory": current,
                "remaining_inventory_needed": as_float(d.get("remaining_inventory_needed")),
                "delta_inventory_added_step": current,
                "source_GP_initial_consumed_step": as_float(d.get("source_GP_initial_consumed")),
                "source_GP_new_consumed_step": as_float(d.get("source_GP_new_consumed")),
                "source_matrix_consumed_step": as_float(d.get("source_matrix_consumed")),
                "transaction_mass_error_rel": abs(as_float(d.get("transaction_mass_error_rel"), 0.0)),
                "global_mass_error_rel": state.get("global_mass_error_rel", abs(as_float(d.get("global_mass_error_rel"), 0.0))),
                "resolved_seed_inserted": 0,
                "xB_source_for_JGP": rate.get("xB_source_for_JGP", state.get("xB_source_for_JGP", math.nan)),
                "xAg_used_for_JGP": rate.get("xAg_used_for_JGP", math.nan),
                "J_GP_m3_s": rate.get("J_GP_m3_s", math.nan),
                "GP_births_cumulative": gp_births,
                "natural_staged_embryos_created_cumulative": embryos_created,
                "NaN_Inf": False,
            }
            rows.append(row)
        elif "BETA_STAGED_ACCUMULATION_BEGIN" in line:
            d = parse_kv(line)
            step = as_int(d.get("step"))
            rate = nearest_leq(rates, step)
            state = nearest_leq(states, step)
            inserted = as_int(d.get("inserted_resolved_seed"))
            row = {
                "case": case,
                "step": step,
                "time_s": step * dt_s if math.isfinite(dt_s) else math.nan,
                "T_C": T_C,
                "S": S,
                "embryo_id": d.get("embryo_id", ""),
                "embryo_status": d.get("status_after", ""),
                "target_seed_inventory": as_float(d.get("target_seed_inventory")),
                "current_inventory": as_float(d.get("current_inventory_after")),
                "remaining_inventory_needed": as_float(d.get("remaining_after")),
                "delta_inventory_added_step": as_float(d.get("delta_inventory_added")),
                "source_GP_initial_consumed_step": as_float(d.get("source_GP_initial_consumed_step")),
                "source_GP_new_consumed_step": as_float(d.get("source_GP_new_consumed_step")),
                "source_matrix_consumed_step": as_float(d.get("source_matrix_consumed_step")),
                "transaction_mass_error_rel": abs(as_float(d.get("transaction_mass_error_rel"), 0.0)),
                "global_mass_error_rel": state.get("global_mass_error_rel", abs(as_float(d.get("global_mass_error_rel"), 0.0))),
                "resolved_seed_inserted": inserted,
                "xB_source_for_JGP": rate.get("xB_source_for_JGP", state.get("xB_source_for_JGP", math.nan)),
                "xAg_used_for_JGP": rate.get("xAg_used_for_JGP", math.nan),
                "J_GP_m3_s": rate.get("J_GP_m3_s", math.nan),
                "GP_births_cumulative": gp_births,
                "natural_staged_embryos_created_cumulative": embryos_created,
                "NaN_Inf": False,
            }
            rows.append(row)
        elif "RESOLVED_BETA_HANDOFF_BEGIN" in line:
            d = parse_kv(line)
            step = as_int(d.get("step"))
            state = nearest_leq(states, step)
            row = {
                "case": case,
                "step": step,
                "embryo_id": d.get("embryo_id", ""),
                "target_seed_inventory": as_float(d.get("target_seed_inventory")),
                "staged_inventory_before": as_float(d.get("staged_inventory_before")),
                "staged_inventory_transferred": as_float(d.get("staged_inventory_transferred")),
                "beta_inventory_before": as_float(d.get("beta_inventory_before")),
                "beta_inventory_after": as_float(d.get("beta_inventory_after")),
                "resolved_seed_radius_nm": as_float(d.get("resolved_seed_radius_nm")),
                "resolved_seed_volume": as_float(d.get("resolved_seed_volume")),
                "matrix_inventory_before": as_float(d.get("matrix_inventory_before")),
                "GP_initial_inventory_before": as_float(d.get("GP_initial_inventory_before")),
                "GP_new_inventory_before": as_float(d.get("GP_new_inventory_before")),
                "M_total_before": as_float(d.get("M_total_before")),
                "M_total_after": as_float(d.get("M_total_after")),
                "handoff_transaction_mass_error_abs": as_float(d.get("handoff_transaction_mass_error_abs")),
                "handoff_transaction_mass_error_rel": abs(as_float(d.get("handoff_transaction_mass_error_rel"), 0.0)),
                "global_mass_error_rel_after": abs(as_float(d.get("global_mass_error_rel_after"), state.get("global_mass_error_rel", 0.0))),
            }
            resolved_rows.append(row)

    nan_inf = detect_runtime_nan_inf(text)
    for row in rows:
        row["NaN_Inf"] = nan_inf

    return {
        "rows": rows,
        "resolved_rows": resolved_rows,
        "text": text,
        "dt_s": dt_s,
        "last_step": max([as_int(r["step"]) for r in rows] or [0]),
        "embryos_created": sum(1 for r in rows if as_float(r["delta_inventory_added_step"]) == as_float(r["current_inventory"]) and as_float(r["current_inventory"]) > 0.0),
        "nan_inf": nan_inf,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def monotonic_checks(rows: list[dict[str, Any]]) -> tuple[bool, bool, list[str]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by[str(row.get("embryo_id", ""))].append(row)
    current_ok = True
    remaining_ok = True
    failures: list[str] = []
    for eid, erows in by.items():
        erows.sort(key=lambda r: as_int(r.get("step")))
        currents = [as_float(r.get("current_inventory")) for r in erows if math.isfinite(as_float(r.get("current_inventory")))]
        remain = [as_float(r.get("remaining_inventory_needed")) for r in erows if math.isfinite(as_float(r.get("remaining_inventory_needed")))]
        for a, b in zip(currents, currents[1:]):
            if b + 1.0e-9 < a:
                current_ok = False
                failures.append(f"embryo {eid} current decreased {a:.12e}->{b:.12e}")
                break
        for a, b in zip(remain, remain[1:]):
            if b > a + 1.0e-9:
                remaining_ok = False
                failures.append(f"embryo {eid} remaining increased {a:.12e}->{b:.12e}")
                break
    return current_ok, remaining_ok, failures


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    ratios = []
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by[str(row.get("embryo_id", ""))].append(row)
    best_ratio = 0.0
    best_current = 0.0
    best_target = 0.0
    best_remaining = math.inf
    best_step = 0
    best_time = math.nan
    best_rate_per_step = math.nan
    best_rate_per_s = math.nan
    est_steps_to_target = math.inf
    est_time_s_to_target = math.inf
    for erows in by.values():
        erows.sort(key=lambda r: as_int(r.get("step")))
        if not erows:
            continue
        last = erows[-1]
        target = as_float(last.get("target_seed_inventory"))
        current = as_float(last.get("current_inventory"))
        if target > 0.0 and math.isfinite(current):
            ratio = current / target
            ratios.append(ratio)
            if ratio > best_ratio:
                best_ratio = ratio
                best_current = current
                best_target = target
                best_remaining = as_float(last.get("remaining_inventory_needed"))
                best_step = as_int(last.get("step"))
                best_time = as_float(last.get("time_s"))
                window = erows[-25:] if len(erows) >= 25 else erows
                if len(window) >= 2:
                    first = window[0]
                    ds = as_int(last.get("step")) - as_int(first.get("step"))
                    dt = as_float(last.get("time_s")) - as_float(first.get("time_s"))
                    dc = current - as_float(first.get("current_inventory"))
                    if ds > 0 and dc > 0.0:
                        best_rate_per_step = dc / ds
                        est_steps_to_target = max(best_remaining, 0.0) / best_rate_per_step
                    if dt > 0.0 and dc > 0.0:
                        best_rate_per_s = dc / dt
                        est_time_s_to_target = max(best_remaining, 0.0) / best_rate_per_s
    def max_finite(values: list[float], default: float = 0.0) -> float:
        finite = [v for v in values if math.isfinite(v)]
        return max(finite) if finite else default

    return {
        "rows": float(len(rows)),
        "embryos": float(len(by)),
        "max_ratio": max(ratios) if ratios else 0.0,
        "best_current_inventory": best_current,
        "best_target_inventory": best_target,
        "min_remaining_inventory_needed": best_remaining if math.isfinite(best_remaining) else math.nan,
        "best_step": float(best_step),
        "best_time_s": best_time,
        "estimated_accumulation_rate_per_step": best_rate_per_step,
        "estimated_accumulation_rate_per_s": best_rate_per_s,
        "estimated_steps_to_target": est_steps_to_target,
        "estimated_time_s_to_target": est_time_s_to_target,
        "max_transaction_error_rel": max_finite([as_float(r.get("transaction_mass_error_rel"), 0.0) for r in rows]),
        "max_global_mass_error_rel": max_finite([as_float(r.get("global_mass_error_rel"), 0.0) for r in rows]),
        "xB_source_max": max_finite([as_float(r.get("xB_source_for_JGP"), 0.0) for r in rows]),
        "resolved_inserted": max([as_int(r.get("resolved_seed_inserted")) for r in rows] or [0]),
    }


def write_report(path: Path, title: str, status: str, summary: dict[str, float], current_ok: bool, remaining_ok: bool, failures: list[str], extra: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "## Verdict",
        "",
        f"`{status}`",
        "",
        "## Metrics",
        "",
        f"- Rows parsed: `{int(summary['rows'])}`",
        f"- Embryos tracked: `{int(summary['embryos'])}`",
        f"- Max current/target ratio: `{summary['max_ratio']:.12e}`",
        f"- Leading current inventory: `{summary['best_current_inventory']:.12e}`",
        f"- Leading target inventory: `{summary['best_target_inventory']:.12e}`",
        f"- Min remaining inventory needed: `{summary['min_remaining_inventory_needed']:.12e}`",
        f"- Estimated accumulation rate per step: `{summary['estimated_accumulation_rate_per_step']:.12e}`",
        f"- Estimated accumulation rate per second: `{summary['estimated_accumulation_rate_per_s']:.12e}`",
        f"- Estimated steps to target: `{summary['estimated_steps_to_target']:.12e}`",
        f"- Estimated seconds to target: `{summary['estimated_time_s_to_target']:.12e}`",
        f"- Max transaction error rel: `{summary['max_transaction_error_rel']:.12e}`",
        f"- Max global mass error rel: `{summary['max_global_mass_error_rel']:.12e}`",
        f"- xB source max: `{summary['xB_source_max']:.12e}`",
        f"- Resolved seed inserted: `{int(summary['resolved_inserted'])}`",
        f"- Current inventory monotonic: `{str(current_ok).lower()}`",
        f"- Remaining inventory monotonic: `{str(remaining_ok).lower()}`",
        "",
    ]
    if failures:
        lines.extend(["## Monotonicity Failures", ""])
        lines.extend(f"- {failure}" for failure in failures[:20])
        lines.append("")
    if extra:
        lines.extend(["## Notes", "", extra.strip(), ""])
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", required=True, type=Path)
    ap.add_argument("--case", required=True)
    ap.add_argument("--T-C", required=True, type=float)
    ap.add_argument("--S", required=True, type=float)
    ap.add_argument("--timeseries-out", required=True, type=Path)
    ap.add_argument("--report-out", required=True, type=Path)
    ap.add_argument("--resolved-out", type=Path)
    ap.add_argument("--title", default="Staged Handoff Validation")
    args = ap.parse_args()

    parsed = parse_log(args.stdout, args.case, args.T_C, args.S)
    rows = parsed["rows"]
    fields = [
        "case", "step", "time_s", "T_C", "S", "embryo_id", "embryo_status",
        "target_seed_inventory", "current_inventory", "remaining_inventory_needed",
        "delta_inventory_added_step", "source_GP_initial_consumed_step",
        "source_GP_new_consumed_step", "source_matrix_consumed_step",
        "transaction_mass_error_rel", "global_mass_error_rel",
        "resolved_seed_inserted", "xB_source_for_JGP", "xAg_used_for_JGP",
        "J_GP_m3_s", "GP_births_cumulative",
        "natural_staged_embryos_created_cumulative", "NaN_Inf",
    ]
    write_csv(args.timeseries_out, rows, fields)
    if args.resolved_out:
        resolved_fields = [
            "case", "step", "embryo_id", "target_seed_inventory",
            "staged_inventory_before", "staged_inventory_transferred",
            "beta_inventory_before", "beta_inventory_after",
            "resolved_seed_radius_nm", "resolved_seed_volume",
            "matrix_inventory_before", "GP_initial_inventory_before",
            "GP_new_inventory_before", "M_total_before", "M_total_after",
            "handoff_transaction_mass_error_abs",
            "handoff_transaction_mass_error_rel",
            "global_mass_error_rel_after",
        ]
        write_csv(args.resolved_out, parsed["resolved_rows"], resolved_fields)
    current_ok, remaining_ok, failures = monotonic_checks(rows)
    summary = summarize_rows(rows)
    status = "PASS" if (
        summary["embryos"] >= 1.0
        and current_ok
        and remaining_ok
        and summary["max_transaction_error_rel"] <= 1.0e-10
        and summary["max_global_mass_error_rel"] <= 1.0e-7
        and summary["xB_source_max"] <= 0.02
        and not parsed["nan_inf"]
    ) else "FAIL"
    extra = (
        f"Last parsed step: `{parsed['last_step']}`\n\n"
        f"Runtime NaN/Inf detected: `{str(parsed['nan_inf']).lower()}`\n\n"
        "If no embryo reached target, this report validates stable accumulation "
        "only; resolved handoff must be validated by a natural, forced, or "
        "debug-accelerated run."
    )
    write_report(args.report_out, args.title, status, summary, current_ok, remaining_ok, failures, extra)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
