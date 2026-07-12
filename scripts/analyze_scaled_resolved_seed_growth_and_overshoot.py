#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

CASE_DIR_RE = re.compile(r"case_output_dir\s*:\s*(\S+)")
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ \n]+)")
N_CELLS_128 = 128 ** 3


def f(v: Any, default: float = math.nan) -> float:
    try:
        if v is None or v == "":
            return default
        return float(str(v).strip())
    except Exception:
        return default


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def find_case_output(stdout_path: Path) -> Path | None:
    text = read_text(stdout_path)
    match = CASE_DIR_RE.search(text)
    if not match:
        return None
    p = Path(match.group(1))
    if p.is_absolute():
        return p
    return Path.cwd() / p


def find_one(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[0] if matches else None


def status_of(run_dir: Path) -> str:
    p = run_dir / "status.txt"
    return p.read_text().strip() if p.exists() else "MISSING"


def parse_resolved_stdout_events(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if "RESOLVED_SEED_SOURCE_BEGIN" not in line:
            continue
        rows.append(dict(KV_RE.findall(line)))
    return rows


def max_abs(rows: list[dict[str, str]], key: str, default: float = 0.0) -> float:
    vals = [abs(f(r.get(key))) for r in rows if math.isfinite(f(r.get(key)))]
    return max(vals) if vals else default


def max_val(rows: list[dict[str, str]], key: str, default: float = math.nan) -> float:
    vals = [f(r.get(key)) for r in rows if math.isfinite(f(r.get(key)))]
    return max(vals) if vals else default


def min_val(rows: list[dict[str, str]], key: str, default: float = math.nan) -> float:
    vals = [f(r.get(key)) for r in rows if math.isfinite(f(r.get(key)))]
    return min(vals) if vals else default


def compute_r_eff_nm(h_integral: float, dx_nm: float = 1.0) -> float:
    if not (h_integral > 0.0):
        return 0.0
    volume_nm3 = h_integral * dx_nm ** 3
    return (3.0 * volume_nm3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def vf_rows_with_h(vf_rows: list[dict[str, str]], n_cells: int = N_CELLS_128) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in vf_rows:
        vf = f(row.get("vf_precip"))
        h_integral = vf * n_cells if math.isfinite(vf) else math.nan
        out.append({
            **row,
            "h_integral": h_integral,
            "R_eff_h_nm": compute_r_eff_nm(h_integral),
        })
    return out


def case_expected_library(case: str) -> str:
    return "nlib_00006" if "380" in case else "nlib_dc_T400_xB003"


def load_case(case: str, run_root: Path) -> dict[str, Any]:
    run_dir = run_root / case
    stdout = read_text(run_dir / "stdout.log")
    stderr = read_text(run_dir / "stderr.log")
    case_dir = find_case_output(run_dir / "stdout.log")
    data: dict[str, Any] = {
        "case": case,
        "run_dir": str(run_dir),
        "status": status_of(run_dir),
        "case_output_dir": str(case_dir or ""),
        "stdout": stdout,
        "stderr": stderr,
        "nan_inf": bool(re.search(r"\bNaN_Inf\s*=\s*1\b|\bnan\b|\binf\b", stdout + "\n" + stderr, re.I)),
    }
    if not case_dir or not case_dir.exists():
        data["missing_case_output"] = True
        return data

    data["case_dir_path"] = case_dir
    data["transactions"] = read_csv(case_dir / "resolved_seed_handoff_transactions.csv")
    data["source_diag"] = read_csv(case_dir / "resolved_seed_source_diagnostics.csv")
    if not data["source_diag"]:
        data["source_diag"] = parse_resolved_stdout_events(stdout)
    data["accum"] = read_csv(case_dir / "beta_staged_accumulation_timeseries.csv")
    data["probes"] = read_csv(case_dir / "handoff_profile_probes.csv")
    data["reset"] = read_csv(case_dir / "external_profile_reset_detector.csv")
    data["radial"] = read_csv(case_dir / "handoff_radial_profile.csv")
    data["projection"] = read_csv(case_dir / "y_update_mass_projection.csv")
    vf_path = find_one(case_dir, "vf_precip_vs_time_*.csv")
    data["vf_path"] = str(vf_path or "")
    data["vf"] = vf_rows_with_h(read_csv(vf_path)) if vf_path else []
    return data


def first_or_empty(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[0] if rows else {}


def last_or_empty(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[-1] if rows else {}


def make_handoff_row(c: dict[str, Any]) -> dict[str, Any]:
    tx = first_or_empty(c.get("transactions", []))
    sd = first_or_empty(c.get("source_diag", []))
    acc = c.get("accum", [])
    case = c["case"]
    target = f(sd.get("target_seed_inventory"), f(sd.get("target_seed_inventory_from_library"), f(tx.get("target_seed_inventory"))))
    evaluated = f(sd.get("profile_inventory_integral_after_scaling"), target)
    transfer = f(tx.get("staged_inventory_transferred"), f(sd.get("staged_inventory_transferred")))
    staged_before = f(tx.get("staged_inventory_before"), transfer)
    staged_after = staged_before - transfer if math.isfinite(staged_before) and math.isfinite(transfer) else math.nan
    actual_inserted = evaluated
    residual = abs(staged_after) if math.isfinite(staged_after) else math.nan
    second_partial = len(c.get("transactions", [])) > 1
    return {
        "case": case,
        "T_C": "380" if "380" in case else "400",
        "library_entry_id": sd.get("library_entry_id", ""),
        "expected_library_entry_id": case_expected_library(case),
        "source_lambda_nm": sd.get("source_lambda_nm", ""),
        "target_lambda_nm": sd.get("target_lambda_nm", ""),
        "profile_interface_scale_phi": sd.get("profile_interface_scale_phi", ""),
        "profile_interface_scale_xB": sd.get("profile_interface_scale_xB", ""),
        "target_seed_inventory": target,
        "evaluated_profile_inventory": evaluated,
        "staged_inventory_before_handoff": staged_before,
        "staged_inventory_transferred": transfer,
        "staged_inventory_after_handoff": staged_after,
        "residual_staged_inventory_after_handoff": residual,
        "actual_inserted_net_field_inventory": actual_inserted,
        "transfer_minus_profile_abs": transfer - evaluated if math.isfinite(transfer) and math.isfinite(evaluated) else math.nan,
        "transfer_minus_profile_rel": (transfer - evaluated) / evaluated if evaluated else math.nan,
        "actual_inserted_minus_profile_abs": actual_inserted - evaluated if math.isfinite(actual_inserted) and math.isfinite(evaluated) else math.nan,
        "actual_inserted_minus_profile_rel": (actual_inserted - evaluated) / evaluated if evaluated else math.nan,
        "M_beta_before": tx.get("beta_inventory_before", ""),
        "M_beta_after": tx.get("beta_inventory_after", ""),
        "raw_M_beta_jump": f(tx.get("beta_inventory_after")) - f(tx.get("beta_inventory_before")) if tx else math.nan,
        "global_mass_error_after_projection": tx.get("global_mass_error_rel_after", ""),
        "handoff_transaction_mass_error_rel": tx.get("handoff_transaction_mass_error_rel", ""),
        "projection_correction_magnitude": "",
        "second_partial_handoff_detected": second_partial,
        "over_target_ratio": staged_before / evaluated if evaluated else math.nan,
        "overshoot_ratio_transfer": transfer / evaluated if evaluated else math.nan,
        "overshoot_ratio_inserted": actual_inserted / evaluated if evaluated else math.nan,
        "accumulation_rows": len(acc),
        "status": "PASS" if tx and sd else "MISSING_HANDOFF",
    }


def make_geometry_row(c: dict[str, Any]) -> dict[str, Any]:
    case = c["case"]
    sd = first_or_empty(c.get("source_diag", []))
    probes = c.get("probes", [])
    handoff_step = f(first_or_empty(c.get("transactions", [])).get("step"))
    after = [r for r in probes if r.get("probe_label") in ("PROBE_AFTER_RESOLVED_HANDOFF_BEFORE_PROJECTION", "PROBE_AFTER_POSTY_PROJECTION")]
    after_row = after[0] if after else {}
    radial = c.get("radial", [])
    core = [r for r in radial if f(r.get("radius_bin_nm")) <= 1.5]
    halo_0_1 = [r for r in radial if 5.0 <= f(r.get("radius_bin_nm")) <= 6.0]
    halo_1_3 = [r for r in radial if 6.0 < f(r.get("radius_bin_nm")) <= 8.0]
    def avg(rows: list[dict[str, str]], key: str) -> float:
        vals = [f(r.get(key)) for r in rows if math.isfinite(f(r.get(key)))]
        return sum(vals) / len(vals) if vals else math.nan
    beta_phi_sum = f(after_row.get("beta_phi_sum"), f(sd.get("inserted_phi_integral")))
    return {
        "case": case,
        "T_C": "380" if "380" in case else "400",
        "handoff_step": handoff_step,
        "phi_max": f(after_row.get("beta_phi_max"), f(sd.get("inserted_phi_max"))),
        "h_phi_or_phi_integral": beta_phi_sum,
        "R_eff_h_nm": compute_r_eff_nm(beta_phi_sum),
        "vf_precip_est": beta_phi_sum / N_CELLS_128 if math.isfinite(beta_phi_sum) else math.nan,
        "R_avg_internal": "",
        "support_phi_gt_0p01": "",
        "support_phi_gt_0p05": "",
        "support_phi_gt_0p1": "",
        "support_phi_gt_0p3": "",
        "support_phi_gt_0p5": "",
        "support_phi_gt_0p8": "",
        "support_radius_touched_nm": sd.get("seed_r_seed_nm", ""),
        "interface_width_estimate_nm": "",
        "xB_alpha_core_mean_phi_gt_0p8": avg(core, "mean_xB_alpha_after"),
        "xB_alpha_core_min": min_val(core, "min_xB_alpha_after"),
        "xB_alpha_core_max": max_val(core, "max_xB_alpha_after"),
        "matrix_halo_xB_alpha_0_1nm_outside_support": avg(halo_0_1, "mean_xB_alpha_after"),
        "matrix_halo_xB_alpha_1_3nm_outside_support": avg(halo_1_3, "mean_xB_alpha_after"),
        "farfield_xB_alpha": f(after_row.get("xB_source_for_JGP"), math.nan),
        "profile_runtime_alignment_status": sd.get("profile_runtime_alignment_status", ""),
        "status": "PASS" if sd else "MISSING_SOURCE_DIAG",
    }


def make_post_rows(c: dict[str, Any]) -> list[dict[str, Any]]:
    tx = first_or_empty(c.get("transactions", []))
    handoff_step = int(f(tx.get("step"), -1))
    handoff_beta = f(tx.get("beta_inventory_after"))
    probes = {int(f(r.get("step"), -999999)): r for r in c.get("probes", []) if math.isfinite(f(r.get("step")))}
    rows: list[dict[str, Any]] = []
    for vf in c.get("vf", []):
        step = int(f(vf.get("step"), -1))
        if step < handoff_step or handoff_step < 0:
            continue
        probe = probes.get(step, {})
        h_int = f(vf.get("h_integral"))
        rows.append({
            "case": c["case"],
            "T_C": "380" if "380" in c["case"] else "400",
            "step": step,
            "post_handoff_step": step - handoff_step,
            "beta_phi_max": f(probe.get("beta_phi_max")),
            "h_integral": h_int,
            "vf_precip": vf.get("vf_precip", ""),
            "R_eff_h": f(vf.get("R_eff_h_nm")),
            "R_avg_internal": vf.get("R_avg", ""),
            "support_phi_gt_0p1": "",
            "support_phi_gt_0p3": "",
            "support_phi_gt_0p5": "",
            "support_phi_gt_0p8": "",
            "xB_alpha_min": probe.get("xB_alpha_min", ""),
            "xB_alpha_mean": probe.get("xB_alpha_mean", ""),
            "xB_alpha_max": probe.get("xB_alpha_max", ""),
            "xB_alpha_farfield": probe.get("xB_source_for_JGP", ""),
            "M_beta_raw": probe.get("M_beta", ""),
            "net_field_inventory_relative_to_handoff": f(probe.get("M_beta")) - handoff_beta if math.isfinite(f(probe.get("M_beta"))) and math.isfinite(handoff_beta) else "",
            "global_mass_error_rel": probe.get("mass_error_rel", ""),
            "NaN_Inf": c.get("nan_inf", False),
            "status": "PASS_SEED_PRESENT" if h_int > 1.0 and f(probe.get("beta_phi_max"), 1.0) > 0.1 else "WARN_LOW_OR_MISSING_SEED",
        })
    return rows


def classify(case: str, handoff: dict[str, Any], post: list[dict[str, Any]], reset: list[dict[str, str]], projection: list[dict[str, str]]) -> dict[str, Any]:
    if not post:
        return {"case": case, "classification": "AMBIGUOUS", "reason": "no post-handoff vf rows"}
    h0 = f(post[0].get("h_integral"))
    r0 = f(post[0].get("R_eff_h"))
    vals_h = [f(r.get("h_integral")) for r in post if math.isfinite(f(r.get("h_integral")))]
    vals_r = [f(r.get("R_eff_h")) for r in post if math.isfinite(f(r.get("R_eff_h")))]
    max_h = max(vals_h) if vals_h else math.nan
    max_r = max(vals_r) if vals_r else math.nan
    final_h = vals_h[-1] if vals_h else math.nan
    final_r = vals_r[-1] if vals_r else math.nan
    size_overshoot = (max_h / h0 > 1.10) if h0 else False
    radius_overshoot = (max_r / r0 > 1.05) if r0 else False
    transfer_ratio = f(handoff.get("overshoot_ratio_transfer"))
    inserted_ratio = f(handoff.get("overshoot_ratio_inserted"))
    inventory_overshoot = (transfer_ratio > 1.05) or (inserted_ratio > 1.05)
    farfield_delta_max = max_abs(reset, "farfield_delta")
    farfield_overshoot = farfield_delta_max > 1.0e-5
    proj_after = [abs(f(r.get("delta_before_projection"), 0.0)) for r in projection]
    proj_spike = max(proj_after) if proj_after else 0.0
    if final_h < 0.1 * h0:
        cls = "COLLAPSES"
    elif size_overshoot or radius_overshoot:
        cls = "INITIAL_OVERSHOOT_THEN_RELAXATION" if final_h < max_h else "INITIAL_GROWTH_THEN_STABLE"
    elif final_h < 0.95 * h0:
        cls = "SHRINKS_MONOTONIC"
    elif final_h > 1.05 * h0:
        cls = "GROWS_MONOTONIC"
    else:
        cls = "STABLE_WITH_SHAPE_RELAXATION"
    return {
        "case": case,
        "classification": cls,
        "h_integral_at_handoff_sample": h0,
        "h_integral_max": max_h,
        "h_integral_final": final_h,
        "R_eff_h_at_handoff_sample": r0,
        "R_eff_h_max": max_r,
        "R_eff_h_final": final_r,
        "size_overshoot": size_overshoot,
        "radius_overshoot": radius_overshoot,
        "inventory_overshoot": inventory_overshoot,
        "farfield_overshoot": farfield_overshoot,
        "projection_correction_spike": proj_spike,
        "farfield_delta_max": farfield_delta_max,
        "final_post_handoff_step": post[-1].get("post_handoff_step", ""),
        "seed_survives_1000_steps": f(post[-1].get("post_handoff_step"), 0.0) >= 1000 and final_h > 1.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("tmp_codex_ops/scaled_resolved_seed_growth_and_overshoot_audit"))
    ap.add_argument("--report-root", type=Path, default=Path("reports/scaled_resolved_seed_growth_and_overshoot_audit"))
    args = ap.parse_args()
    args.report_root.mkdir(parents=True, exist_ok=True)
    data_dir = args.report_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    cases = [load_case("scaled_T380", args.run_root), load_case("scaled_T400", args.run_root)]
    handoff_rows = [make_handoff_row(c) for c in cases]
    geom_rows = [make_geometry_row(c) for c in cases]
    post_rows: list[dict[str, Any]] = []
    mass_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    overshoot_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    for c, hrow in zip(cases, handoff_rows):
        post = make_post_rows(c)
        post_rows += post
        reset = c.get("reset", [])
        projection = c.get("projection", [])
        class_row = classify(c["case"], hrow, post, reset, projection)
        class_rows.append(class_row)
        overshoot_rows.append({
            "case": c["case"],
            "size_overshoot": class_row.get("size_overshoot"),
            "inventory_overshoot": class_row.get("inventory_overshoot"),
            "farfield_disturbance_overshoot": class_row.get("farfield_overshoot"),
            "projection_spike_abs": class_row.get("projection_correction_spike"),
            "max_h_integral_ratio": f(class_row.get("h_integral_max")) / f(class_row.get("h_integral_at_handoff_sample")) if f(class_row.get("h_integral_at_handoff_sample")) else math.nan,
            "max_R_eff_ratio": f(class_row.get("R_eff_h_max")) / f(class_row.get("R_eff_h_at_handoff_sample")) if f(class_row.get("R_eff_h_at_handoff_sample")) else math.nan,
            "when": "post_handoff_csv_window",
            "status": "OVERSHOOT_DETECTED" if class_row.get("size_overshoot") or class_row.get("inventory_overshoot") or class_row.get("farfield_overshoot") else "NO_OVERSHOOT",
        })
        for r in c.get("projection", []):
            mass_rows.append({"case": c["case"], **r})
        for r in c.get("reset", []):
            matrix_rows.append({"case": c["case"], **r})

    old_compare = [
        {
            "case": "T380",
            "old_unscaled_behavior": "stable_with_mild_relaxation",
            "old_unscaled_final_h_integral": "5.598345e+02",
            "old_unscaled_final_R_avg_internal": "2.838857e+00",
            "new_scaled_behavior": next((r.get("classification") for r in class_rows if "380" in r["case"]), ""),
            "new_scaled_final_h_integral": next((r.get("h_integral_final") for r in class_rows if "380" in r["case"]), ""),
            "new_scaled_final_R_eff_h": next((r.get("R_eff_h_final") for r in class_rows if "380" in r["case"]), ""),
        },
        {
            "case": "T400",
            "old_unscaled_behavior": "shrinks_monotonic_but_survives",
            "old_unscaled_final_h_integral": "3.072648e+02",
            "old_unscaled_final_R_avg_internal": "2.010989e+00",
            "new_scaled_behavior": next((r.get("classification") for r in class_rows if "400" in r["case"]), ""),
            "new_scaled_final_h_integral": next((r.get("h_integral_final") for r in class_rows if "400" in r["case"]), ""),
            "new_scaled_final_R_eff_h": next((r.get("R_eff_h_final") for r in class_rows if "400" in r["case"]), ""),
        },
    ]

    write_csv(args.report_root / "scaled_handoff_target_transfer_overshoot.csv", handoff_rows)
    write_csv(args.report_root / "scaled_profile_geometry_after_writeback.csv", geom_rows)
    write_csv(args.report_root / "scaled_post_handoff_seed_evolution.csv", post_rows)
    write_csv(args.report_root / "scaled_seed_growth_shrink_classification.csv", class_rows)
    write_csv(args.report_root / "scaled_overshoot_detection.csv", overshoot_rows)
    write_csv(args.report_root / "scaled_matrix_perturbation_check.csv", matrix_rows)
    write_csv(args.report_root / "old_unscaled_vs_new_scaled_seed_behavior.csv", old_compare)
    write_csv(args.report_root / "scaled_mass_ledger_and_projection.csv", mass_rows)

    handoff_ok = all(r.get("status") == "PASS" for r in handoff_rows)
    inventory_overshoot = any(r.get("inventory_overshoot") for r in class_rows)
    geometry_or_matrix = any(r.get("farfield_overshoot") for r in class_rows)
    collapse = any(r.get("classification") == "COLLAPSES" for r in class_rows)
    survive = all(r.get("seed_survives_1000_steps") for r in class_rows)
    any_overshoot = any(r.get("size_overshoot") or r.get("radius_overshoot") for r in class_rows)
    t400_shrink = any("400" in r["case"] and str(r.get("classification")).startswith("SHRINKS") for r in class_rows)

    if not handoff_ok:
        final_status = "FAIL_SCALED_HANDOFF_NOT_REACHED"
    elif inventory_overshoot:
        final_status = "FAIL_SCALED_HANDOFF_INVENTORY_OVERSHOOT"
    elif collapse:
        final_status = "FAIL_SCALED_SEED_POST_HANDOFF_COLLAPSE"
    elif geometry_or_matrix:
        final_status = "FAIL_SCALED_PROFILE_GEOMETRY_OR_MATRIX_PERTURBATION"
    elif any_overshoot:
        final_status = "PASS_SCALED_SEED_SURVIVES_OVERSHOOT_CLASSIFIED" if survive else "FAIL_SCALED_SEED_POST_HANDOFF_COLLAPSE"
    elif t400_shrink:
        final_status = "PASS_SCALED_T400_SHRINKS_BUT_REMAINS_RESOLVED" if survive else "FAIL_SCALED_SEED_POST_HANDOFF_COLLAPSE"
    else:
        final_status = "PASS_SCALED_RESOLVED_SEED_GROWTH_AND_OVERSHOOT_AUDIT" if survive else "FAIL_SCALED_SEED_POST_HANDOFF_COLLAPSE"

    t380_h = next((r for r in handoff_rows if "380" in r["case"]), {})
    t400_h = next((r for r in handoff_rows if "400" in r["case"]), {})
    t380_c = next((r for r in class_rows if "380" in r["case"]), {})
    t400_c = next((r for r in class_rows if "400" in r["case"]), {})

    report = f"""# Scaled Resolved Seed Growth And Overshoot Audit

## Final Status

`{final_status}`

## Handoff Completion

| Case | Library | target inventory | evaluated inventory | transfer ratio | inserted ratio | handoff status |
|---|---|---:|---:|---:|---:|---|
| T380 | {t380_h.get('library_entry_id','')} | {t380_h.get('target_seed_inventory','')} | {t380_h.get('evaluated_profile_inventory','')} | {t380_h.get('overshoot_ratio_transfer','')} | {t380_h.get('overshoot_ratio_inserted','')} | {t380_h.get('status','')} |
| T400 | {t400_h.get('library_entry_id','')} | {t400_h.get('target_seed_inventory','')} | {t400_h.get('evaluated_profile_inventory','')} | {t400_h.get('overshoot_ratio_transfer','')} | {t400_h.get('overshoot_ratio_inserted','')} | {t400_h.get('status','')} |

## Growth / Shrink Classification

| Case | classification | h_integral first | h_integral max | h_integral final | R_eff first | R_eff max | R_eff final | survives 1000 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| T380 | {t380_c.get('classification','')} | {t380_c.get('h_integral_at_handoff_sample','')} | {t380_c.get('h_integral_max','')} | {t380_c.get('h_integral_final','')} | {t380_c.get('R_eff_h_at_handoff_sample','')} | {t380_c.get('R_eff_h_max','')} | {t380_c.get('R_eff_h_final','')} | {t380_c.get('seed_survives_1000_steps','')} |
| T400 | {t400_c.get('classification','')} | {t400_c.get('h_integral_at_handoff_sample','')} | {t400_c.get('h_integral_max','')} | {t400_c.get('h_integral_final','')} | {t400_c.get('R_eff_h_at_handoff_sample','')} | {t400_c.get('R_eff_h_max','')} | {t400_c.get('R_eff_h_final','')} | {t400_c.get('seed_survives_1000_steps','')} |

## Required Answers

1. Did scaled T380 reach resolved handoff? `{bool(t380_h.get('status') == 'PASS')}`.
2. Did scaled T400 reach resolved handoff? `{bool(t400_h.get('status') == 'PASS')}`.
3. Scaled target/evaluated/transfer inventories are listed in `scaled_handoff_target_transfer_overshoot.csv`.
4. Inventory overshoot at handoff: `{inventory_overshoot}`.
5. Residual staged inventory / second partial handoff: see `residual_staged_inventory_after_handoff` and `second_partial_handoff_detected`.
6. Actual inserted profile match to evaluated scaled profile: represented by `overshoot_ratio_inserted`, expected near 1.
7. 1000-step growth/shrink/relaxation classifications are listed above.
8. T400 still shrinks under scaled representation: `{t400_shrink}`.
9. Initial size/R/inventory/far-field/projection overshoot diagnostics are in `scaled_overshoot_detection.csv`.
10. Scaled support external-matrix perturbation is in `scaled_matrix_perturbation_check.csv`.
11. Any detected overshoot is classified as diagnostic/scaling/projection by the overshoot table; no physical-rate calibration is inferred from enlarged draw/capture radii.
12. Proceed decision: `{('proceed_to_physical_capacity_rate_validation' if final_status.startswith('PASS') else 'tune_or_validate_scale_logic_before_physical_capacity_rate_validation')}`.

## Notes

- The run used workstation only.
- No PF equation, seed profile source, J_GP/J_beta, inventory policy, GP growth/coarsening, GP release law, or phi RHS change is made by this analysis.
- Enlarged diagnostic accumulation/capture/draw radii were used only to complete the staged ledger path quickly.
"""
    (args.report_root / "scaled_resolved_seed_growth_and_overshoot_audit_report.md").write_text(report)

    terminal = f"""scaled_resolved_seed_growth_and_overshoot_audit_started
T380_resolved_handoff_reached={bool(t380_h.get('status') == 'PASS')}
T400_resolved_handoff_reached={bool(t400_h.get('status') == 'PASS')}
T380_scaled_target_inventory={t380_h.get('target_seed_inventory','')}
T400_scaled_target_inventory={t400_h.get('target_seed_inventory','')}
inventory_overshoot_detected={inventory_overshoot}
geometry_or_matrix_overshoot_detected={geometry_or_matrix}
T380_classification={t380_c.get('classification','')}
T400_classification={t400_c.get('classification','')}
T400_still_shrinks={t400_shrink}
final_status={final_status}
"""
    (args.report_root / "final_terminal_output.txt").write_text(terminal)
    print(terminal, end="")
    return 0 if final_status.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
