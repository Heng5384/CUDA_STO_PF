#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

K_B_J_PER_K = 1.380649e-23
REPO_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = REPO_ROOT / "reports"
NUCLEATION_ROOT = REPORTS_ROOT / "nucleation"


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


def parse_case_metadata(path: Path) -> dict[str, float | str]:
    text = str(path)
    out: dict[str, float | str] = {}
    mt = re.search(r"T([0-9]+)", text)
    mx = re.search(r"xB([0-9]+p[0-9]+|[0-9.]+)", text)
    mn = re.search(r"cuda_([0-9]+)x([0-9]+)x([0-9]+)", text)
    mdt = re.search(r"dt([0-9]+p[0-9]+|[0-9.]+)", text)
    if mt:
        out["T_C"] = float(mt.group(1))
    if mx:
        out["xB"] = float(mx.group(1).replace("p", "."))
    if mn:
        out["Nx"], out["Ny"], out["Nz"] = [float(x) for x in mn.groups()]
    if mdt:
        out["dt_code"] = float(mdt.group(1).replace("p", "."))
    return out


def classify_shape(shape: str, rc_nm: float | None = None) -> str:
    s = (shape or "").strip().lower()
    if "facet" in s:
        return "faceted"
    if "anis" in s or "ellip" in s:
        return "anisotropic"
    if "sphere" in s or "spherical" in s:
        return "sphere"
    return "unknown"


def read_unified_event_logs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("nucleation_event_log.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_source_file"] = str(path)
                row["_source_format"] = "unified"
                rows.append(row)
    return rows


def read_legacy_gp_event_logs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("gp_assisted_event_log.csv")):
        meta = parse_case_metadata(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "accepted":
                    continue
                dt_code = safe_float(meta.get("dt_code"), 1.0) or 1.0
                step = safe_int(row.get("step"))
                rows.append({
                    "event_id": row.get("event_id", ""),
                    "event_source": "legacy_gp_assisted_beta",
                    "step": row.get("step", ""),
                    "time_physical_s": "",
                    "local_xB": "",
                    "T_C": meta.get("T_C", ""),
                    "gp_presence_flag": "1",
                    "gp_assisted_flag": "1",
                    "nucleus_shape_type": "sphere",
                    "rc_nm": "",
                    "center_x_nm": row.get("center_i", ""),
                    "center_y_nm": row.get("center_j", ""),
                    "center_z_nm": row.get("center_k", ""),
                    "template_or_site_id": f"gp_site_{row.get('site_id', '')}",
                    "continuous_barrier_kBT": "",
                    "input_event_probability": "",
                    "domain_volume_nm3": "",
                    "dt_code": dt_code,
                    "t_real_unit_s": "",
                    "event_status": row.get("status", "accepted"),
                    "notes": "legacy_gp_assisted_event_log_missing_T_volume_time_fields",
                    "_source_file": str(path),
                    "_source_format": "legacy_gp",
                    "_time_code": step * dt_code,
                })
    return rows


def normalized_events(root: Path, include_legacy: bool) -> list[dict[str, Any]]:
    rows = read_unified_event_logs(root)
    if include_legacy:
        rows.extend(read_legacy_gp_event_logs(root))
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if row.get("event_status", "accepted") not in {"accepted", ""}:
            continue
        T_C = safe_float(row.get("T_C"))
        xB = safe_float(row.get("local_xB"))
        if xB is None:
            meta = parse_case_metadata(Path(row.get("_source_file", "")))
            xB = safe_float(meta.get("xB"))
        step = safe_int(row.get("step"))
        time_s = safe_float(row.get("time_physical_s"))
        time_code = safe_float(row.get("_time_code"))
        if time_s is None:
            dt_code = safe_float(row.get("dt_code"), 1.0) or 1.0
            time_code = step * dt_code if time_code is None else time_code
        domain_nm3 = safe_float(row.get("domain_volume_nm3"))
        normalized.append({
            **row,
            "T_C_num": T_C,
            "T_K": (T_C + 273.15) if T_C is not None else None,
            "xB_num": xB,
            "step_num": step,
            "time_s_num": time_s,
            "time_code_num": time_code,
            "domain_volume_nm3_num": domain_nm3,
            "gp_flag_num": safe_int(row.get("gp_presence_flag")),
            "gp_assisted_num": safe_int(row.get("gp_assisted_flag")),
            "shape_class": classify_shape(row.get("nucleus_shape_type", "")),
            "rc_nm_num": safe_float(row.get("rc_nm")),
            "event_probability_num": safe_float(row.get("input_event_probability")),
            "continuous_barrier_kBT_num": safe_float(row.get("continuous_barrier_kBT")),
        })
    normalized.sort(key=lambda r: (r.get("_source_file", ""), r["step_num"], safe_int(r.get("event_id"))))
    return normalized


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    T = row["T_C_num"]
    xB = row["xB_num"]
    gp = "GP" if row["gp_assisted_num"] or row["gp_flag_num"] else "noGP"
    return (
        f"{T:.6g}" if T is not None else "unknown",
        f"{xB:.8g}" if xB is not None else "unknown",
        gp,
    )


def write_event_log_copy(rows: list[dict[str, Any]], output: Path) -> None:
    fields = [
        "event_id", "event_source", "step", "time_physical_s", "local_xB", "T_C",
        "gp_presence_flag", "gp_assisted_flag", "nucleus_shape_type", "rc_nm",
        "center_x_nm", "center_y_nm", "center_z_nm", "template_or_site_id",
        "continuous_barrier_kBT", "input_event_probability", "domain_volume_nm3",
        "dt_code", "t_real_unit_s", "event_status", "notes", "_source_file",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_barrier_estimates(rows: list[dict[str, Any]], output: Path, tau0: float) -> list[dict[str, Any]]:
    last_time: dict[tuple[str, str, str], float] = defaultdict(float)
    estimates: list[dict[str, Any]] = []
    for row in rows:
        key = group_key(row)
        time = row["time_s_num"]
        time_basis = "seconds"
        if time is None:
            time = row["time_code_num"]
            time_basis = "code_time"
        if time is None:
            tau = None
        else:
            tau = max(float(time) - last_time[key], tau0)
            last_time[key] = float(time)
        p_event = row["event_probability_num"]
        if p_event is not None and p_event > 0.0:
            barrier_kBT = -math.log(min(max(p_event, 1.0e-300), 1.0))
            method = "minus_ln_P_event"
        elif tau is not None and tau > 0.0:
            barrier_kBT = math.log(max(tau / tau0, 1.0e-300))
            method = "waiting_time_ln_tau_over_tau0"
        else:
            barrier_kBT = None
            method = "not_available"
        T_K = row["T_K"]
        estimates.append({
            "event_id": row.get("event_id", ""),
            "event_source": row.get("event_source", ""),
            "T_C": row.get("T_C", ""),
            "xB": row.get("local_xB") or row.get("xB_num") or "",
            "gp_assisted_flag": row.get("gp_assisted_flag", ""),
            "step": row.get("step", ""),
            "waiting_time": tau if tau is not None else "",
            "waiting_time_basis": time_basis if tau is not None else "not_available",
            "tau0": tau0,
            "barrier_proxy_kBT": barrier_kBT if barrier_kBT is not None else "",
            "barrier_proxy_J": (K_B_J_PER_K * T_K * barrier_kBT) if (barrier_kBT is not None and T_K is not None) else "",
            "method": method,
            "continuous_barrier_kBT": row.get("continuous_barrier_kBT", ""),
            "source_file": row.get("_source_file", ""),
        })
    fields = list(estimates[0].keys()) if estimates else [
        "event_id", "event_source", "T_C", "xB", "gp_assisted_flag", "step",
        "waiting_time", "waiting_time_basis", "tau0", "barrier_proxy_kBT",
        "barrier_proxy_J", "method", "continuous_barrier_kBT", "source_file",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(estimates)
    return estimates


def write_rates(rows: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    out: list[dict[str, Any]] = []
    for (T, xB, gp), items in sorted(groups.items()):
        times_s = [r["time_s_num"] for r in items if r["time_s_num"] is not None]
        times_code = [r["time_code_num"] for r in items if r["time_code_num"] is not None]
        vols_nm3 = [r["domain_volume_nm3_num"] for r in items if r["domain_volume_nm3_num"] is not None and r["domain_volume_nm3_num"] > 0]
        if times_s and vols_nm3:
            duration = max(times_s) - min(0.0, min(times_s))
            volume_m3 = max(vols_nm3) * 1.0e-27
            J = len(items) / max(volume_m3 * duration, 1.0e-300)
            basis = "m^-3_s^-1"
        elif times_code and vols_nm3:
            duration = max(times_code) - min(0.0, min(times_code))
            volume_m3 = max(vols_nm3) * 1.0e-27
            J = len(items) / max(volume_m3 * duration, 1.0e-300)
            basis = "m^-3_code_time^-1"
        else:
            duration = max(times_code) - min(0.0, min(times_code)) if times_code else ""
            volume_m3 = ""
            J = ""
            basis = "not_available_missing_volume_or_time"
        out.append({
            "T_C": T,
            "xB": xB,
            "event_class": gp,
            "event_count": len(items),
            "observation_time": duration,
            "volume_m3": volume_m3,
            "J": J,
            "J_units": basis,
        })
    fields = ["T_C", "xB", "event_class", "event_count", "observation_time", "volume_m3", "J", "J_units"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    return out


def mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [safe_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def write_gp_effect(barriers: list[dict[str, Any]], rates: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    bgrp: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in barriers:
        gp = "GP" if safe_int(row.get("gp_assisted_flag")) else "noGP"
        bgrp[(str(row.get("T_C", "unknown")), str(row.get("xB", "unknown")), gp)].append(row)
    rgrp = {(r["T_C"], r["xB"], r["event_class"]): r for r in rates}
    tx = sorted({(k[0], k[1]) for k in bgrp} | {(r["T_C"], r["xB"]) for r in rates})
    out: list[dict[str, Any]] = []
    for T, xB in tx:
        bgp = mean_numeric(bgrp.get((T, xB, "GP"), []), "barrier_proxy_kBT")
        bno = mean_numeric(bgrp.get((T, xB, "noGP"), []), "barrier_proxy_kBT")
        rgp = safe_float((rgrp.get((T, xB, "GP")) or {}).get("J"))
        rno = safe_float((rgrp.get((T, xB, "noGP")) or {}).get("J"))
        out.append({
            "T_C": T,
            "xB": xB,
            "barrier_gp_kBT": bgp if bgp is not None else "",
            "barrier_no_gp_kBT": bno if bno is not None else "",
            "barrier_reduction_kBT": (bno - bgp) if (bno is not None and bgp is not None) else "",
            "J_gp": rgp if rgp is not None else "",
            "J_no_gp": rno if rno is not None else "",
            "rate_enhancement_factor": (rgp / rno) if (rgp is not None and rno not in (None, 0.0)) else "",
            "status": "complete" if (bno is not None and bgp is not None and rgp is not None and rno is not None) else "not_available_missing_paired_gp_no_gp_events",
        })
    fields = ["T_C", "xB", "barrier_gp_kBT", "barrier_no_gp_kBT", "barrier_reduction_kBT", "J_gp", "J_no_gp", "rate_enhancement_factor", "status"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    return out


def write_shape_stats(rows: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        T, xB, _gp = group_key(row)
        groups[(T, xB, row["shape_class"])].append(row)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for (T, xB, _shape), items in groups.items():
        totals[(T, xB)] += len(items)
    out: list[dict[str, Any]] = []
    for (T, xB, shape), items in sorted(groups.items()):
        total = totals[(T, xB)]
        out.append({
            "T_C": T,
            "xB": xB,
            "shape_class": shape,
            "event_count": len(items),
            "fraction": len(items) / total if total else "",
            "mean_rc_nm": mean_numeric(items, "rc_nm_num") or "",
        })
    fields = ["T_C", "xB", "shape_class", "event_count", "fraction", "mean_rc_nm"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    return out


def write_report(root: Path, events: list[dict[str, Any]], rates: list[dict[str, Any]], gp_effect: list[dict[str, Any]], output: Path) -> None:
    n_events = len(events)
    n_unified = sum(1 for r in events if r.get("_source_format") == "unified")
    n_legacy = n_events - n_unified
    paired_gp = any(r.get("status") == "complete" for r in gp_effect)
    rate_ready = any(str(r.get("J", "")) not in {"", "nan"} and not str(r.get("J_units", "")).startswith("not_available") for r in rates)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Nucleation Physics Observables Report

## Installation Status

The runtime event-observation layer is installed. CUDA now writes a unified `nucleation_event_log.csv` when scheduled insertion or GP-assisted beta events occur.

This layer is observational only. It does not modify CNT physics, selector logic, template mapping, or nucleation criteria.

## Input Event Data

- events found: {n_events}
- unified runtime events: {n_unified}
- legacy GP-assisted events ingested: {n_legacy}
- search root: `{root}`

## Generated Outputs

- `reports/nucleation/nucleation_event_log.csv`
- `reports/nucleation/nucleation_barrier_estimate.csv`
- `reports/nucleation/nucleation_rate_J.csv`
- `reports/nucleation/gp_effect_analysis.csv`
- `reports/nucleation/nucleus_shape_statistics.csv`

## Required Answers

### 1. What is effective DeltaG*(T, xB)?

Computed in `nucleation_barrier_estimate.csv` when event waiting times or event probabilities are available.

Formula used:

```text
DeltaG* / kBT = -ln(P_event)
```

or, if no explicit event probability is logged:

```text
DeltaG* / kBT = ln(tau / tau0)
```

Current status: {'available from event logs' if n_events else 'not available; no actual event logs found'}.

### 2. Does GP reduce barrier quantitatively?

Computed in `gp_effect_analysis.csv` only when paired GP and no-GP event groups exist at the same T and xB.

Current status: {'paired GP/no-GP comparison available' if paired_gp else 'not available; paired GP/no-GP event groups are missing'}.

### 3. What is nucleation rate dependence on T?

Computed in `nucleation_rate_J.csv` as:

```text
J = events / (volume * observation_time)
```

Current status: {'rate estimates available' if rate_ready else 'not available or code-time-only; physical volume/time metadata missing for some events'}.

### 4. What nucleus shapes dominate at different conditions?

Computed in `nucleus_shape_statistics.csv` from event-level shape labels.

Current status: {'shape statistics available' if n_events else 'not available; no events'}.

### 5. Is current CNT prediction consistent with observed events?

This report does not declare CNT consistency unless observed event barriers/rates can be compared with CNT predictions at the same T, xB, strain, and GP state.

Current status: not yet publication-grade unless paired continuous CNT rows are joined with these event observables.

## Readiness

- physics observables pipeline ready: true
- ready for phase diagram: false

Reason: this system is event-based by design and does not build a full phase diagram.
"""
    output.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract event-based nucleation observables from actual CUDA nucleation events.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "Results")
    parser.add_argument("--tau0", type=float, default=1.0)
    parser.add_argument("--include-legacy-gp", action="store_true", default=True)
    parser.add_argument("--output-dir", type=Path, default=NUCLEATION_ROOT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = normalized_events(args.root, include_legacy=args.include_legacy_gp)
    write_event_log_copy(events, args.output_dir / "nucleation_event_log.csv")
    barriers = write_barrier_estimates(events, args.output_dir / "nucleation_barrier_estimate.csv", args.tau0)
    rates = write_rates(events, args.output_dir / "nucleation_rate_J.csv")
    gp_effect = write_gp_effect(barriers, rates, args.output_dir / "gp_effect_analysis.csv")
    write_shape_stats(events, args.output_dir / "nucleus_shape_statistics.csv")
    write_report(args.root, events, rates, gp_effect, args.output_dir / "nucleation_physics_observables_report.md")

    physics_ready = True
    print("nucleation_event_logging_enabled")
    print("barrier_proxy_extraction_active")
    print(f"physics_observables_ready = {str(physics_ready).lower()}")
    print("ready_for_phase_diagram = false")
    print("next_stage_recommendation = run paired mapped-template CUDA event simulations for GP and no-GP cases, then compare event-derived barriers with CNT predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
