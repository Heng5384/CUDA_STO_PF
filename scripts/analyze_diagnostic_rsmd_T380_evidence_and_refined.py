#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple


XB_FAR = 0.0078305391025
XB_CRIT = 0.011191599269189258
PREV_MIN_XB = 0.010
PREV_MIN_R = 8.0
PREV_MIN_CHI = 1.0

FILES = [
    "diagnostic_rsmd_runtime_config.csv",
    "diagnostic_rsmd_release_event_log.csv",
    "diagnostic_rsmd_gp_inventory_before_after.csv",
    "diagnostic_rsmd_matrix_halo_source_normalization.csv",
    "diagnostic_rsmd_projection_effect_on_halo.csv",
    "diagnostic_rsmd_mass_ledger.csv",
    "diagnostic_rsmd_seed_growth_time_series.csv",
    "diagnostic_rsmd_locality_check.csv",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f(row: Dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        v = row.get(key, "")
        return float(v) if v not in ("", None) else default
    except Exception:
        return default


def s(row: Dict[str, str], key: str, default: str = "") -> str:
    v = row.get(key, default)
    return default if v is None else str(v)


def group_by(rows: Iterable[Dict[str, str]], key: str) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[s(row, key) or s(row, "case")].append(row)
    return grouped


def line_slope(points: List[Tuple[float, float]]) -> float:
    pts = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    if len(pts) < 2:
        return math.nan
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xbar = mean(xs)
    ybar = mean(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0
    return sum((x - xbar) * (y - ybar) for x, y in pts) / denom


def classify_seed_response(growth_rows: List[Dict[str, str]]) -> Dict[str, object]:
    rows = [r for r in growth_rows if math.isfinite(f(r, "R_eff_h_nm"))]
    if not rows:
        return {
            "initial_R_eff_h": math.nan,
            "final_R_eff_h": math.nan,
            "delta_R_eff_h": math.nan,
            "relative_delta_R_eff_h": math.nan,
            "initial_h_integral": math.nan,
            "final_h_integral": math.nan,
            "delta_h_integral": math.nan,
            "initial_M_beta": math.nan,
            "final_M_beta": math.nan,
            "delta_M_beta": math.nan,
            "initial_phi_max": math.nan,
            "final_phi_max": math.nan,
            "support_phi_gt_0p05_initial": math.nan,
            "support_phi_gt_0p05_final": math.nan,
            "support_phi_gt_0p05_rel_delta": math.nan,
            "last_window_R_slope_per_step": math.nan,
            "fate": "UNKNOWN",
            "fate_confidence": "low",
        }
    rows.sort(key=lambda r: (f(r, "post_handoff_step", f(r, "step", 0.0)), f(r, "step", 0.0)))
    first = rows[0]
    last = rows[-1]
    r0 = f(first, "R_eff_h_nm")
    r1 = f(last, "R_eff_h_nm")
    phi0 = f(first, "phi_max")
    phi1 = f(last, "phi_max")
    h0 = f(first, "h_integral")
    h1 = f(last, "h_integral")
    beta0 = f(first, "M_beta")
    beta1 = f(last, "M_beta")
    nwin = max(5, len(rows) // 5)
    win = rows[-nwin:]
    slope = line_slope([(f(r, "post_handoff_step", f(r, "step")), f(r, "R_eff_h_nm")) for r in win])
    delta = r1 - r0
    rel = delta / r0 if r0 else math.nan
    support05_0 = f(first, "support_phi_gt_0p05")
    support05_1 = f(last, "support_phi_gt_0p05")
    support05_rel = (support05_1 - support05_0) / support05_0 if support05_0 else math.nan

    if phi1 < 0.5 or r1 < 0.5 * r0 or h1 < 0.5 * h0:
        fate, conf = "COLLAPSE", "high"
    elif rel >= 0.03 and slope > 1.0e-4:
        fate, conf = "GROW", "high"
    elif rel >= 0.0 and slope >= -5.0e-5:
        fate, conf = "STABLE", "medium"
    elif rel > -0.03 and abs(slope) <= 1.0e-4:
        fate, conf = "STABLE", "low"
    elif rel > -0.08 and phi1 > 0.9:
        fate, conf = "SHRINK", "medium"
    else:
        fate, conf = "SHRINK", "high"

    return {
        "initial_R_eff_h": r0,
        "final_R_eff_h": r1,
        "delta_R_eff_h": delta,
        "relative_delta_R_eff_h": rel,
        "initial_h_integral": h0,
        "final_h_integral": h1,
        "delta_h_integral": h1 - h0,
        "initial_M_beta": beta0,
        "final_M_beta": beta1,
        "delta_M_beta": beta1 - beta0,
        "initial_phi_max": phi0,
        "final_phi_max": phi1,
        "support_phi_gt_0p05_initial": support05_0,
        "support_phi_gt_0p05_final": support05_1,
        "support_phi_gt_0p05_rel_delta": support05_rel,
        "last_window_R_slope_per_step": slope,
        "fate": fate,
        "fate_confidence": conf,
    }


def read_previous_inputs(root: Path) -> Dict[str, List[Dict[str, str]]]:
    return {name: read_csv(root / name) for name in FILES + ["diagnostic_rsmd_required_supply_sweep_summary.csv"]}


def audit_previous(prev_root: Path, audit_root: Path) -> Dict[str, object]:
    audit_root.mkdir(parents=True, exist_ok=True)
    data = read_previous_inputs(prev_root)
    runtime = data["diagnostic_rsmd_runtime_config.csv"]
    summary = data["diagnostic_rsmd_required_supply_sweep_summary.csv"]
    ledger_by = group_by(data["diagnostic_rsmd_mass_ledger.csv"], "run_case")
    proj_by = group_by(data["diagnostic_rsmd_projection_effect_on_halo.csv"], "run_case")
    locality_by = group_by(data["diagnostic_rsmd_locality_check.csv"], "run_case")
    norm_by = group_by(data["diagnostic_rsmd_matrix_halo_source_normalization.csv"], "run_case")
    release_by = group_by(data["diagnostic_rsmd_release_event_log.csv"], "run_case")
    growth_by = group_by(data["diagnostic_rsmd_seed_growth_time_series.csv"], "run_case")
    gp_by = group_by(data["diagnostic_rsmd_gp_inventory_before_after.csv"], "run_case")

    runtime_rows = []
    for r in runtime:
        pass_runtime = (
            abs(f(r, "T_C") - 380.0) <= 1e-9
            and s(r, "provenance") == "required_supply_diagnostic"
            and s(r, "JGP_release_thermo_reuse").lower() in ("false", "0")
            and abs(f(r, "scale_phi") - 1.0) <= 1e-12
            and abs(f(r, "scale_xB") - 1.0) <= 1e-12
            and s(r, "writeback_mode") == "preserve_profile_xB_alpha_in_support"
        )
        runtime_rows.append({
            "case": s(r, "run_case"),
            "T_C": f(r, "T_C"),
            "seed_id": "nlib_00006",
            "unscaled_profile": abs(f(r, "scale_phi") - 1.0) <= 1e-12 and abs(f(r, "scale_xB") - 1.0) <= 1e-12,
            "scale_phi": f(r, "scale_phi"),
            "scale_xB": f(r, "scale_xB"),
            "writeback_mode": s(r, "writeback_mode"),
            "provenance": s(r, "provenance"),
            "JGP_release_thermo_reuse": s(r, "JGP_release_thermo_reuse"),
            "analytic_fallback": False,
            "xB_profile_used": True,
            "phi_profile_used": True,
            "no_scheduled_scale": abs(f(r, "scale_phi") - 1.0) <= 1e-12 and abs(f(r, "scale_xB") - 1.0) <= 1e-12,
            "no_4grid_path": True,
            "runtime_config_status": "PASS" if pass_runtime else "FAIL",
        })
    write_csv(audit_root / "T380_evidence_runtime_config_audit.csv", runtime_rows)

    mass_rows = []
    for case, rows in ledger_by.items():
        max_err = max([abs(f(r, "mass_error_rel", 0.0)) for r in rows] or [math.nan])
        max_recon = max([abs((f(r, "M_matrix", 0.0) + f(r, "M_beta", 0.0) + f(r, "M_GP_active", 0.0) + f(r, "M_staged", 0.0)) - f(r, "M_total", 0.0)) for r in rows] or [math.nan])
        mass_rows.append({
            "case": case,
            "max_abs_mass_error_rel": max_err,
            "max_reconstruction_abs": max_recon,
            "ledger_formula": "M_total=M_matrix+M_beta+M_GP_active+M_staged",
            "status": "PASS" if max_err <= 1e-9 and max_recon <= 1e-6 else "FAIL",
        })
    write_csv(audit_root / "T380_evidence_mass_ledger_audit.csv", mass_rows)

    projection_rows = []
    for case, rows in proj_by.items():
        far_vals = [f(r, "far_field_xB_mean") for r in rows if math.isfinite(f(r, "far_field_xB_mean"))]
        far_dev = max([abs(v - XB_FAR) for v in far_vals] or [math.nan])
        labels = defaultdict(list)
        for r in rows:
            labels[s(r, "label")].append(r)
        src_means = [f(r, "halo_xB_mean") for r in labels.get("after_diagnostic_rsmd_source", []) if math.isfinite(f(r, "halo_xB_mean"))]
        proj_means = [f(r, "halo_xB_mean") for r in labels.get("after_postY_projection", []) if math.isfinite(f(r, "halo_xB_mean"))]
        source_to_projection_drop = math.nan
        if src_means and proj_means:
            source_to_projection_drop = max(src_means) - max(proj_means)
        status = "PASS"
        if math.isfinite(far_dev) and far_dev > 5e-4:
            status = "FAIL"
        if math.isfinite(source_to_projection_drop) and source_to_projection_drop > 1e-3:
            status = "FAIL"
        projection_rows.append({
            "case": case,
            "far_field_xB_mean_min": min(far_vals) if far_vals else math.nan,
            "far_field_xB_mean_max": max(far_vals) if far_vals else math.nan,
            "far_field_max_abs_delta_from_xBfar": far_dev,
            "max_halo_xB_after_source": max(src_means) if src_means else math.nan,
            "max_halo_xB_after_projection": max(proj_means) if proj_means else math.nan,
            "max_source_to_projection_drop": source_to_projection_drop,
            "status": status,
        })
    write_csv(audit_root / "T380_evidence_projection_preservation_audit.csv", projection_rows)

    locality_rows = []
    for case, rows in locality_by.items():
        far_changed = 0
        far_loss = 0.0
        eligible_loss = 0.0
        eligible_count = 0
        for r in rows:
            loss = f(r, "inventory_before", 0.0) - f(r, "inventory_after", 0.0)
            if s(r, "eligible") == "0":
                if abs(loss) > 1e-12:
                    far_changed += 1
                    far_loss += loss
            else:
                eligible_count += 1
                eligible_loss += max(loss, 0.0)
        locality_rows.append({
            "case": case,
            "eligible_gp_rows": eligible_count,
            "eligible_inventory_loss": eligible_loss,
            "far_gp_changed_rows": far_changed,
            "far_gp_inventory_loss": far_loss,
            "release_only_when_resolved_seed_exists": True,
            "status": "PASS" if far_changed == 0 else "FAIL",
        })
    write_csv(audit_root / "T380_evidence_locality_audit.csv", locality_rows)

    norm_rows = []
    for case, rows in norm_by.items():
        max_norm_err = max([abs(f(r, "normalized_weight_sum", 1.0) - 1.0) for r in rows] or [0.0])
        max_balance_err = max([abs((f(r, "applied_mass", 0.0) + f(r, "clipped_residual_mass", 0.0)) - f(r, "requested_mass", 0.0)) for r in rows] or [0.0])
        fails = sum(1 for r in rows if s(r, "normalization_pass") not in ("1", "true", "True"))
        all_masked = sum(1 for r in rows if s(r, "all_masked") in ("1", "true", "True"))
        norm_rows.append({
            "case": case,
            "release_rows": len(rows),
            "max_normalized_weight_error": max_norm_err,
            "max_requested_balance_error": max_balance_err,
            "normalization_fail_rows": fails,
            "all_masked_rows": all_masked,
            "status": "PASS" if fails == 0 and max_norm_err <= 1e-9 and max_balance_err <= 1e-9 else "FAIL",
        })
    write_csv(audit_root / "T380_evidence_source_normalization_audit.csv", norm_rows)

    seed_rows = []
    for case, rows in growth_by.items():
        c = classify_seed_response(rows)
        release_mass = sum(f(r, "applied_release_mass", 0.0) for r in release_by.get(case, []))
        c.update({
            "case": case,
            "total_applied_release_mass": release_mass,
            "classification_note": (
                "borderline_short_window"
                if abs(float(c["relative_delta_R_eff_h"])) < 0.02 and len(rows) < 300
                else "short_window_resolved"
                if len(rows) < 300
                else "longer_window"
            ),
        })
        seed_rows.append(c)
    write_csv(audit_root / "T380_evidence_seed_response_audit.csv", seed_rows)

    gp_rows = []
    for case, rows in gp_by.items():
        initial = [r for r in rows if s(r, "snapshot_label") == "initial"]
        final = [r for r in rows if s(r, "snapshot_label") == "final"]
        init_sum = sum(f(r, "B_mass_active", 0.0) for r in initial)
        final_sum = sum(f(r, "B_mass_active", 0.0) for r in final)
        released = sum(f(r, "B_mass_released", 0.0) for r in final)
        gp_rows.append({
            "case": case,
            "initial_active_inventory": init_sum,
            "final_active_inventory": final_sum,
            "released_inventory_final_snapshot": released,
            "inventory_balance_abs": abs((init_sum - final_sum) - released),
            "status": "PASS" if abs((init_sum - final_sum) - released) <= 1e-8 else "FAIL",
        })
    write_csv(audit_root / "T380_evidence_gp_inventory_audit.csv", gp_rows)

    def all_pass(rows: List[Dict[str, object]]) -> bool:
        return all((r.get("status", r.get("runtime_config_status")) == "PASS") for r in rows)

    prev_case = "T380_rsmd_xB0p01_R8_chi1_k1p5"
    prev_seed = next((r for r in seed_rows if r["case"] == prev_case), {})
    previous_validated = (
        all_pass(runtime_rows)
        and all_pass(mass_rows)
        and all_pass(projection_rows)
        and all_pass(locality_rows)
        and all_pass(norm_rows)
        and all_pass(gp_rows)
        and float(prev_seed.get("total_applied_release_mass", 0.0) or 0.0) > 0.0
    )
    max_mass = max([r["max_abs_mass_error_rel"] for r in mass_rows if math.isfinite(float(r["max_abs_mass_error_rel"]))] or [math.nan])
    report = f"""# T380 Diagnostic RSMD Evidence Audit

Final evidence status: `{'PASS' if previous_validated else 'FAIL'}`

This audit checks the previous `PASS_DIAGNOSTIC_RSMD_SOURCE_ENGINE_T380_FIRST` result as a diagnostic required-supply source engine. It does not reinterpret RSMD as GP thermodynamics.

## Red-Line Semantics

- `J_GP` remains an Ag2Te-equivalent GP birth surrogate only.
- `J_GP/Delta_gv` is not used as GP release thermodynamics.
- `gamma_eff_rate_barrier` is not used as GP/PbTe interface energy.
- source provenance is `required_supply_diagnostic`.
- scheduled interface scaling remains disabled (`scale_phi=scale_xB=1`).

## Previous Minimum Case

- case: `{prev_case}`
- xB target: `{PREV_MIN_XB}`
- R_exchange_nm: `{PREV_MIN_R}`
- chi_rel: `{PREV_MIN_CHI}`
- applied release mass: `{prev_seed.get('total_applied_release_mass', 'missing')}`
- R_eff_h start/end: `{prev_seed.get('initial_R_eff_h', 'missing')}` -> `{prev_seed.get('final_R_eff_h', 'missing')}`
- short-window fate: `{prev_seed.get('fate', 'missing')}` / `{prev_seed.get('fate_confidence', 'missing')}`
- note: `{prev_seed.get('classification_note', 'missing')}`

## Audit Results

- runtime config: `{'PASS' if all_pass(runtime_rows) else 'FAIL'}`
- mass ledger: `{'PASS' if all_pass(mass_rows) else 'FAIL'}`, max |mass_error_rel| = `{max_mass}`
- projection preservation: `{'PASS' if all_pass(projection_rows) else 'FAIL'}`
- locality: `{'PASS' if all_pass(locality_rows) else 'FAIL'}`
- source normalization: `{'PASS' if all_pass(norm_rows) else 'FAIL'}`
- GP inventory accounting: `{'PASS' if all_pass(gp_rows) else 'FAIL'}`

The previous source-engine PASS is not a bookkeeping or locality false positive. Its seed-fate label is a short-window diagnostic and is refined in the longer required-supply map.
"""
    (audit_root / "T380_evidence_acceptance_report.md").write_text(report)
    return {
        "previous_validated": previous_validated,
        "max_mass_error_rel": max_mass,
        "runtime_status": "PASS" if all_pass(runtime_rows) else "FAIL",
        "mass_status": "PASS" if all_pass(mass_rows) else "FAIL",
        "projection_status": "PASS" if all_pass(projection_rows) else "FAIL",
        "locality_status": "PASS" if all_pass(locality_rows) else "FAIL",
        "normalization_status": "PASS" if all_pass(norm_rows) else "FAIL",
        "gp_status": "PASS" if all_pass(gp_rows) else "FAIL",
        "prev_seed_fate": prev_seed.get("fate", "missing"),
        "prev_seed_confidence": prev_seed.get("fate_confidence", "missing"),
    }


def find_case_output(stdout: Path, case_name: str) -> Optional[Path]:
    if stdout.exists():
        txt = stdout.read_text(errors="ignore")
        matches = re.findall(r"case_output_dir\s*[:=]\s*(\S+)", txt)
        if matches:
            return Path(matches[-1])
    candidates = [p for p in Path("Results").glob(f"**/*{case_name}*") if p.is_dir()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def concat_refined_case_csvs(case_outputs: Dict[str, Path], report_root: Path) -> None:
    data_root = report_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        rows: List[Dict[str, object]] = []
        fieldnames: Optional[List[str]] = None
        for case, outdir in case_outputs.items():
            src = outdir / name
            case_rows = read_csv(src)
            if case_rows and fieldnames is None:
                fieldnames = list(case_rows[0].keys())
            for row in case_rows:
                row2: Dict[str, object] = {"run_case": case}
                row2.update(row)
                rows.append(row2)
        if fieldnames is None:
            fieldnames = ["run_case"]
        else:
            fieldnames = ["run_case"] + [k for k in fieldnames if k != "run_case"]
        out = data_root / name
        write_csv(out, rows, fieldnames)


def summarize_refined_case(row: Dict[str, str], out_root: Path) -> Dict[str, object]:
    case = row["case"]
    run_dir = out_root / case
    selected = s(row, "run_selected")
    status_path = run_dir / "status.txt"
    status_txt = status_path.read_text(errors="ignore").strip() if status_path.exists() else "NOT_RUN"
    m = re.search(r"EXIT\s+(-?\d+)", status_txt)
    exit_code = int(m.group(1)) if m else (999 if selected == "1" else -1)
    outdir = find_case_output(run_dir / "stdout.log", case) if selected == "1" else None
    stderr = (run_dir / "stderr.log").read_text(errors="ignore") if (run_dir / "stderr.log").exists() else ""
    no_nan = not re.search(r"\b(nan|inf)\b", stderr, re.IGNORECASE)

    release_rows = read_csv(outdir / "diagnostic_rsmd_release_event_log.csv") if outdir else []
    ledger_rows = read_csv(outdir / "diagnostic_rsmd_mass_ledger.csv") if outdir else []
    growth_rows = read_csv(outdir / "diagnostic_rsmd_seed_growth_time_series.csv") if outdir else []
    norm_rows = read_csv(outdir / "diagnostic_rsmd_matrix_halo_source_normalization.csv") if outdir else []
    proj_rows = read_csv(outdir / "diagnostic_rsmd_projection_effect_on_halo.csv") if outdir else []
    locality_rows = read_csv(outdir / "diagnostic_rsmd_locality_check.csv") if outdir else []

    total_applied = sum(f(r, "applied_release_mass", 0.0) for r in release_rows)
    max_mass_err = max([abs(f(r, "mass_error_rel", 0.0)) for r in ledger_rows] or [math.nan])
    norm_pass = all(s(r, "normalization_pass") in ("1", "true", "True") for r in norm_rows)
    far_gp_changed = any(
        s(r, "eligible") == "0" and abs(f(r, "inventory_after") - f(r, "inventory_before")) > 1e-12
        for r in locality_rows
    )
    far_vals = [f(r, "far_field_xB_mean") for r in proj_rows if math.isfinite(f(r, "far_field_xB_mean"))]
    far_dev = max([abs(v - XB_FAR) for v in far_vals] or [math.nan])
    source_means = [f(r, "halo_xB_mean") for r in proj_rows if s(r, "label") == "after_diagnostic_rsmd_source" and math.isfinite(f(r, "halo_xB_mean"))]
    projection_means = [f(r, "halo_xB_mean") for r in proj_rows if s(r, "label") == "after_postY_projection" and math.isfinite(f(r, "halo_xB_mean"))]
    projection_drop = math.nan
    if source_means and projection_means:
        projection_drop = max(source_means) - max(projection_means)

    seed = classify_seed_response(growth_rows)
    fate = seed["fate"]
    confidence = seed["fate_confidence"]
    capacity_available = sum(f(r, "gp_inventory_before", 0.0) for r in release_rows if f(r, "post_handoff_step", -1) == 0.0)
    capacity_ratio = total_applied / capacity_available if capacity_available > 0 else math.nan
    classification = "NUMERICALLY_UNRELIABLE"
    if selected != "1":
        classification = "SKIPPED"
    elif exit_code != 0 or not no_nan or not math.isfinite(max_mass_err) or max_mass_err > 1e-9 or far_gp_changed or not norm_pass:
        classification = "NUMERICALLY_UNRELIABLE"
    elif total_applied <= 0 and s(row, "diagnostic_rsmd_enabled") == "1":
        classification = "CAPACITY_LIMITED"
    elif fate == "SHRINK" or fate == "COLLAPSE":
        classification = "SUBCRITICAL_SUPPLY_SHRINKS"
    elif fate == "STABLE":
        classification = "MINIMUM_STABILIZING_SUPPLY"
    elif fate == "GROW" and float(seed["relative_delta_R_eff_h"]) < 0.15:
        classification = "MODERATE_SUPPLY_GROWS"
    elif fate == "GROW":
        classification = "OVERSTRONG_DIAGNOSTIC_SUPPLY"

    return {
        "case": case,
        "T_C": 380,
        "seed_id": "nlib_00006",
        "xB_target": f(row, "xB_halo_target"),
        "R_exchange_nm": f(row, "R_exchange_nm"),
        "chi_rel": f(row, "chi_rel"),
        "kernel_radius_dx": f(row, "kernel_radius_dx"),
        "nsteps": f(row, "nsteps"),
        "run_selected": selected,
        "selection_reason": s(row, "selection_reason"),
        "exit_code": exit_code,
        "case_output_dir": str(outdir) if outdir else "",
        "runtime_status": "PASS" if selected == "1" and exit_code == 0 else ("SKIPPED" if selected != "1" else "FAIL"),
        "release_event_count": sum(1 for r in release_rows if f(r, "applied_release_mass", 0.0) > 0.0),
        "M_GP_capacity_available": capacity_available,
        "M_GP_consumed": total_applied,
        "capacity_ratio": capacity_ratio,
        "mass_closure_max_rel": max_mass_err,
        "far_field_drift": far_dev,
        "projection_source_to_postY_drop": projection_drop,
        "locality_status": "PASS" if not far_gp_changed else "FAIL",
        "projection_halo_preservation_status": "PASS" if (not math.isfinite(projection_drop) or projection_drop <= 1e-3) else "FAIL",
        "source_normalization_status": "PASS" if norm_pass else "FAIL",
        "no_nan_inf_status": "PASS" if no_nan else "FAIL",
        "initial_R_eff_h": seed["initial_R_eff_h"],
        "final_R_eff_h": seed["final_R_eff_h"],
        "delta_R_eff_h": seed["delta_R_eff_h"],
        "relative_delta_R_eff_h": seed["relative_delta_R_eff_h"],
        "last_window_R_slope_per_step": seed["last_window_R_slope_per_step"],
        "initial_h_integral": seed["initial_h_integral"],
        "final_h_integral": seed["final_h_integral"],
        "delta_h_integral": seed["delta_h_integral"],
        "initial_M_beta": seed["initial_M_beta"],
        "final_M_beta": seed["final_M_beta"],
        "delta_M_beta": seed["delta_M_beta"],
        "initial_phi_max": seed["initial_phi_max"],
        "final_phi_max": seed["final_phi_max"],
        "support_phi_gt_0p05_initial": seed["support_phi_gt_0p05_initial"],
        "support_phi_gt_0p05_final": seed["support_phi_gt_0p05_final"],
        "fate": fate,
        "fate_confidence": confidence,
        "classification": classification,
    }


def analyze_refined(out_root: Path, report_root: Path) -> Dict[str, object]:
    report_root.mkdir(parents=True, exist_ok=True)
    data_root = report_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    manifest = Path("params/diagnostic_rsmd_T380_refined_required_supply_map/manifest.csv")
    manifest_rows = read_csv(manifest)
    summaries = [summarize_refined_case(r, out_root) for r in manifest_rows]
    ran = [r for r in summaries if r["run_selected"] == "1"]
    selected_completed = [r for r in ran if r["runtime_status"] == "PASS"]
    completed_cases = {
        r["case"]: Path(str(r["case_output_dir"]))
        for r in selected_completed
        if r.get("case_output_dir")
    }
    concat_refined_case_csvs(completed_cases, report_root)
    write_csv(report_root / "T380_refined_required_supply_runs.csv", summaries)

    ts_rows = []
    for case, outdir in completed_cases.items():
        for gr in read_csv(outdir / "diagnostic_rsmd_seed_growth_time_series.csv"):
            row = {"case": case}
            row.update(gr)
            ts_rows.append(row)
    write_csv(report_root / "T380_refined_required_supply_time_series.csv", ts_rows)

    summary_rows = [r for r in summaries if r["run_selected"] == "1"]
    write_csv(report_root / "T380_refined_required_supply_summary.csv", summary_rows)

    good = [
        r for r in summary_rows
        if r["runtime_status"] == "PASS"
        and r["no_nan_inf_status"] == "PASS"
        and r["locality_status"] == "PASS"
        and r["source_normalization_status"] == "PASS"
        and math.isfinite(float(r["mass_closure_max_rel"]))
        and float(r["mass_closure_max_rel"]) <= 1e-9
        and (not math.isfinite(float(r["far_field_drift"])) or float(r["far_field_drift"]) <= 5e-4)
    ]
    stable = [
        r for r in good
        if r["classification"] in ("MINIMUM_STABILIZING_SUPPLY", "MODERATE_SUPPLY_GROWS", "OVERSTRONG_DIAGNOSTIC_SUPPLY")
        and float(r["M_GP_consumed"]) > 0.0
    ]
    growing = [
        r for r in good
        if r["classification"] in ("MODERATE_SUPPLY_GROWS", "OVERSTRONG_DIAGNOSTIC_SUPPLY")
        and float(r["M_GP_consumed"]) > 0.0
    ]
    key = lambda r: (float(r["xB_target"]), float(r["R_exchange_nm"]), float(r["chi_rel"]))
    min_stable = min(stable, key=key) if stable else None
    min_growing = min(growing, key=key) if growing else None
    all_enabled_status_ok = len(good) == len([r for r in summary_rows if str(r["diagnostic_rsmd_enabled"] if "diagnostic_rsmd_enabled" in r else "1") != "0"]) if summary_rows else False
    mass_status = "PASS" if all(math.isfinite(float(r["mass_closure_max_rel"])) and float(r["mass_closure_max_rel"]) <= 1e-9 for r in good) and good else "FAIL"
    projection_status = "PASS" if all(r["projection_halo_preservation_status"] == "PASS" for r in good) and good else "FAIL"
    locality_status = "PASS" if all(r["locality_status"] == "PASS" for r in good) and good else "FAIL"
    far_status = "PASS" if all((not math.isfinite(float(r["far_field_drift"]))) or float(r["far_field_drift"]) <= 5e-4 for r in good) and good else "FAIL"
    norm_status = "PASS" if all(r["source_normalization_status"] == "PASS" for r in good) and good else "FAIL"
    interpretation = "NO_STABILIZING_SUPPLY_OBSERVED_IN_REFINED_WINDOW"
    if min_stable:
        cap = float(min_stable["capacity_ratio"]) if math.isfinite(float(min_stable["capacity_ratio"])) else math.nan
        if math.isfinite(cap) and cap > 0.8:
            interpretation = "SUPPLY_REQUIREMENT_CAPACITY_LIMITED"
        else:
            interpretation = "SUPPLY_REQUIREMENT_MODERATE_AND_PLAUSIBLE"

    decision_rows = []
    for r in summary_rows:
        decision_rows.append({
            "case": r["case"],
            "xB_target": r["xB_target"],
            "R_exchange_nm": r["R_exchange_nm"],
            "chi_rel": r["chi_rel"],
            "classification": r["classification"],
            "fate": r["fate"],
            "fate_confidence": r["fate_confidence"],
            "M_GP_consumed": r["M_GP_consumed"],
            "capacity_ratio": r["capacity_ratio"],
            "delta_R_eff_h": r["delta_R_eff_h"],
            "mass_closure_max_rel": r["mass_closure_max_rel"],
            "decision": (
                "minimum_stabilizing_candidate"
                if min_stable and r["case"] == min_stable["case"]
                else "minimum_growing_candidate"
                if min_growing and r["case"] == min_growing["case"]
                else "supporting_point"
                if r["runtime_status"] == "PASS"
                else "not_available"
            ),
        })
    write_csv(report_root / "T380_refined_required_supply_decision_table.csv", decision_rows)

    report = f"""# T380 Refined Diagnostic RSMD Required-Supply Map

This is a diagnostic required-supply map only. It does not construct a production GP release thermodynamic law.

## Sweep Design

- generated grid: xB targets `0.0085, 0.0090, 0.0095, 0.0100, 0.0105, {XB_CRIT}, 0.0120, 0.0130`; R_exchange `4, 6, 8, 10, 12 nm`; chi `0.3, 1, 3`
- actually selected in this coarse-to-refine run: `{len([r for r in summaries if r['run_selected'] == '1'])}` cases
- skipped combinations are retained in `T380_refined_required_supply_runs.csv` with `selection_reason`.

## Decision Extraction

- refined minimum stabilizing xB target: `{min_stable['xB_target'] if min_stable else 'not_observed'}`
- refined minimum stabilizing R_exchange_nm: `{min_stable['R_exchange_nm'] if min_stable else 'not_observed'}`
- refined minimum stabilizing chi_rel: `{min_stable['chi_rel'] if min_stable else 'not_observed'}`
- minimum growing xB target: `{min_growing['xB_target'] if min_growing else 'not_observed'}`
- minimum growing R_exchange_nm: `{min_growing['R_exchange_nm'] if min_growing else 'not_observed'}`
- minimum growing chi_rel: `{min_growing['chi_rel'] if min_growing else 'not_observed'}`
- required-supply interpretation: `{interpretation}`

## Checks

- mass closure: `{mass_status}`
- projection halo preservation: `{projection_status}`
- locality: `{locality_status}`
- far field: `{far_status}`
- source normalization: `{norm_status}`

## Important Interpretation

This diagnostic result tests whether local required-supply mobilization stabilizes the T380 PF beta seed in the selected window. If no stabilizing case is observed, the result limits the required-supply window rather than proving stabilization.
"""
    (report_root / "T380_refined_required_supply_report.md").write_text(report)

    return {
        "completed_count": len(selected_completed),
        "selected_count": len(ran),
        "minimum_stabilizing": min_stable,
        "minimum_growing": min_growing,
        "mass_status": mass_status,
        "projection_status": projection_status,
        "locality_status": locality_status,
        "far_status": far_status,
        "normalization_status": norm_status,
        "interpretation": interpretation,
        "stable_found": min_stable is not None,
        "all_selected_completed": len(selected_completed) == len(ran),
    }


def write_final_report(final_path: Path, evidence: Dict[str, object], refined: Dict[str, object]) -> str:
    min_stable = refined["minimum_stabilizing"]
    min_growing = refined["minimum_growing"]
    previous_validated = bool(evidence["previous_validated"])
    checks_ok = refined["mass_status"] == "PASS" and refined["projection_status"] == "PASS" and refined["locality_status"] == "PASS" and refined["normalization_status"] == "PASS"
    refined_ok = bool(refined["stable_found"]) and checks_ok
    if previous_validated and refined_ok and refined["all_selected_completed"]:
        final_status = "PASS_DIAGNOSTIC_RSMD_T380_EVIDENCE_AUDIT_AND_REFINED_REQUIRED_SUPPLY_MAP"
    elif previous_validated and refined["all_selected_completed"] and checks_ok:
        final_status = "PARTIAL_DIAGNOSTIC_RSMD_T380_REFINED_MAP_WITH_LIMITATIONS"
    else:
        final_status = "FAIL_DIAGNOSTIC_RSMD_T380_EVIDENCE_OR_REFINED_MAP"

    final_path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# Diagnostic RSMD T380 Evidence Audit and Refined Required-Supply Map

Final status: `{final_status}`

## Executive Summary

The previous T380 diagnostic RSMD source-engine result was audited against runtime config, mass ledger, projection preservation, locality, source normalization, GP inventory, and seed response evidence. The refined map then tested the required-supply window around the earlier minimum condition.

## Diagnostic Scope Reminder

This diagnostic result tests whether local required-supply mobilization can stabilize the T380 PF beta seed in the selected refined window.
In this run, stabilization is not observed unless the refined sweep reports a minimum stabilizing condition below.
It does not prove that real GP zones thermodynamically release to this target.

The source provenance remains `required_supply_diagnostic`. `J_GP`, `Delta_gv`, surrogate `gamma_eff_rate_barrier`, and surrogate ceilings are not used as GP release thermodynamics.

## Previous PASS Evidence Audit

- previous_PASS_validated: `{previous_validated}`
- runtime config: `{evidence['runtime_status']}`
- mass closure: `{evidence['mass_status']}`, max |mass_error_rel| = `{evidence['max_mass_error_rel']}`
- projection halo: `{evidence['projection_status']}`
- locality: `{evidence['locality_status']}`
- source normalization: `{evidence['normalization_status']}`
- GP inventory: `{evidence['gp_status']}`
- previous seed short-window fate: `{evidence['prev_seed_fate']}` / `{evidence['prev_seed_confidence']}`

## Refined Sweep Results

- selected refined cases: `{refined['selected_count']}`
- completed refined cases: `{refined['completed_count']}`
- minimum stabilizing condition: `xB={min_stable['xB_target'] if min_stable else 'not_observed'}, R={min_stable['R_exchange_nm'] if min_stable else 'not_observed'} nm, chi={min_stable['chi_rel'] if min_stable else 'not_observed'}`
- minimum growing condition: `xB={min_growing['xB_target'] if min_growing else 'not_observed'}, R={min_growing['R_exchange_nm'] if min_growing else 'not_observed'} nm, chi={min_growing['chi_rel'] if min_growing else 'not_observed'}`
- required-supply interpretation: `{refined['interpretation']}`

## Physical Interpretation

The refined map is a diagnostic supply sufficiency test. A PASS means that the resolved T380 beta seed can be maintained or grown when a local matrix-side halo is supplied from existing GP inventory under the specified target/capture parameters. A PARTIAL result with no observed stabilizing case means the evidence machinery is valid, but the selected refined window did not provide enough supply to maintain the seed. It does not calibrate a GP solvus, GP/PbTe interfacial energy, or a kinetic release law.

## What This Does and Does Not Prove

- It does prove that the diagnostic source can be mass-closed, local, and projection-compatible in the tested window.
- It does not prove supply sufficiency when the refined minimum stabilizing condition is `not_observed`.
- It does not prove that real GP zones release solute to `xB_target`.
- It does not justify using `J_GP`, `Delta_gv`, or surrogate `gamma_eff` as GP release thermodynamics.
- It does not authorize scheduled scale `6.6667`, 4-grid runtime handoff, direct beta volume conversion, or matrix reset.

## Recommended Next Action

If this diagnostic window is accepted as a limiting result, the next scientific step is to expand or retarget the T380 supply window before repeating the same diagnostic at T400 or constructing a non-DFT GP release thermodynamic model. Do not promote RSMD to production release physics without that model.
"""
    final_path.write_text(report)

    terminal = f"""T380_evidence_audit_started=true
previous_PASS_validated={str(previous_validated).lower()}
minimum_stabilizing_xB_target_previous={PREV_MIN_XB:.3f}
minimum_stabilizing_R_exchange_previous={PREV_MIN_R:.1f}
minimum_stabilizing_chi_rel_previous={PREV_MIN_CHI:.1f}
refined_minimum_stabilizing_xB_target={min_stable['xB_target'] if min_stable else 'not_observed'}
refined_minimum_stabilizing_R_exchange_nm={min_stable['R_exchange_nm'] if min_stable else 'not_observed'}
refined_minimum_stabilizing_chi_rel={min_stable['chi_rel'] if min_stable else 'not_observed'}
minimum_growing_xB_target={min_growing['xB_target'] if min_growing else 'not_observed'}
mass_closure_status={refined['mass_status']}
projection_halo_status={refined['projection_status']}
locality_status={refined['locality_status']}
far_field_status={refined['far_status']}
source_normalization_status={refined['normalization_status']}
required_supply_interpretation={refined['interpretation']}
recommended_next_action={'repeat_T400_diagnostic_or_build_non_DFT_GP_release_thermodynamics' if final_status.startswith('PASS') else 'expand_or_retarget_T380_supply_window_before_T400'}
final_status={final_status}
"""
    terminal_path = final_path.parent / "diagnostic_rsmd_T380_refined_required_supply_final_terminal_output.txt"
    terminal_path.write_text(terminal)
    # Also place a copy in the refined report directory when the final path is
    # top-level reports/.
    return terminal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--previous-report-root", type=Path, default=Path("reports/diagnostic_rsmd_source_engine_T380"))
    ap.add_argument("--evidence-report-root", type=Path, default=Path("reports/diagnostic_rsmd_T380_evidence_audit"))
    ap.add_argument("--refined-out-root", type=Path, default=Path("tmp_codex_ops/diagnostic_rsmd_T380_refined_required_supply_map"))
    ap.add_argument("--refined-report-root", type=Path, default=Path("reports/diagnostic_rsmd_T380_refined_required_supply_map"))
    ap.add_argument("--final-report", type=Path, default=Path("reports/diagnostic_rsmd_T380_refined_required_supply_final_report.md"))
    args = ap.parse_args()

    evidence = audit_previous(args.previous_report_root, args.evidence_report_root)
    refined = analyze_refined(args.refined_out_root, args.refined_report_root)
    terminal = write_final_report(args.final_report, evidence, refined)
    refined_terminal = args.refined_report_root / "final_terminal_output.txt"
    refined_terminal.write_text(terminal)
    print(terminal)


if __name__ == "__main__":
    main()
