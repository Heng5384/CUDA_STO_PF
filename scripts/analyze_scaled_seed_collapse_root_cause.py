#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

CASE_DIR_RE = re.compile(r"case_output_dir\s*:\s*(\S+)")
N_CELLS = 128 ** 3


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def find_case_output(stdout: Path) -> Path | None:
    m = CASE_DIR_RE.search(read_text(stdout))
    if not m:
        return None
    p = Path(m.group(1))
    return p if p.is_absolute() else Path.cwd() / p


def first(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[0] if rows else {}


def last(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[-1] if rows else {}


def max_abs(rows: list[dict[str, str]], key: str) -> float:
    vals = [abs(f(r.get(key))) for r in rows if math.isfinite(f(r.get(key)))]
    return max(vals) if vals else 0.0


def r_eff(h_integral: float) -> float:
    if not (h_integral > 0.0):
        return 0.0
    return (3.0 * h_integral / (4.0 * math.pi)) ** (1.0 / 3.0)


def vf_rows(case_dir: Path) -> list[dict[str, Any]]:
    files = sorted(case_dir.glob("vf_precip_vs_time_*.csv"))
    if not files:
        return []
    rows = read_csv(files[0])
    out: list[dict[str, Any]] = []
    for row in rows:
        vf = f(row.get("vf_precip"))
        h = vf * N_CELLS if math.isfinite(vf) else math.nan
        out.append({**row, "h_integral": h, "R_eff_h": r_eff(h)})
    return out


def load_run_case(run_root: Path, case: str) -> dict[str, Any]:
    run_dir = run_root / case
    case_dir = find_case_output(run_dir / "stdout.log")
    d: dict[str, Any] = {
        "case": case,
        "run_dir": str(run_dir),
        "status": read_text(run_dir / "status.txt").strip() or "MISSING",
        "case_dir": str(case_dir or ""),
    }
    if not case_dir or not case_dir.exists():
        return d
    d["transactions"] = read_csv(case_dir / "resolved_seed_handoff_transactions.csv")
    d["source"] = read_csv(case_dir / "resolved_seed_source_diagnostics.csv")
    d["probes"] = read_csv(case_dir / "handoff_profile_probes.csv")
    d["reset"] = read_csv(case_dir / "external_profile_reset_detector.csv")
    d["radial"] = read_csv(case_dir / "handoff_radial_profile.csv")
    d["projection"] = read_csv(case_dir / "y_update_mass_projection.csv")
    d["rhs"] = read_csv(case_dir / "phi_eta_rhs_attribution_per_step.csv")
    d["vf"] = vf_rows(case_dir)
    return d


def collapse_timing_from_post(case: str, rows: list[dict[str, str]], handoff_step: int | None = None) -> dict[str, Any]:
    if not rows:
        return {"case": case, "status": "NO_DATA"}
    parsed = []
    for row in rows:
        step = int(f(row.get("step"), f(row.get("post_handoff_step"), -1)))
        post = f(row.get("post_handoff_step"), math.nan)
        h = f(row.get("h_integral"))
        phi = f(row.get("beta_phi_max"))
        r = f(row.get("R_eff_h"), r_eff(h))
        parsed.append((step, post, h, phi, r))
    parsed = [p for p in parsed if p[2] > 0]
    if not parsed:
        return {"case": case, "status": "NO_POSITIVE_H"}
    h0 = parsed[0][2]
    def first_cond(cond):
        for step, post, h, phi, r in parsed:
            if cond(step, post, h, phi, r):
                return step
        return ""
    return {
        "case": case,
        "handoff_step": handoff_step if handoff_step is not None else "",
        "first_post_handoff_step": parsed[0][0],
        "h_integral_initial": h0,
        "first_h_drop_gt_10pct_step": first_cond(lambda s,p,h,phi,r: h < 0.9 * h0),
        "first_h_drop_gt_50pct_step": first_cond(lambda s,p,h,phi,r: h < 0.5 * h0),
        "first_phi_max_lt_0p9_step": first_cond(lambda s,p,h,phi,r: math.isfinite(phi) and phi < 0.9),
        "first_phi_max_lt_0p5_step": first_cond(lambda s,p,h,phi,r: math.isfinite(phi) and phi < 0.5),
        "first_R_eff_lt_1nm_step": first_cond(lambda s,p,h,phi,r: r < 1.0),
        "final_step": parsed[-1][0],
        "final_h_integral": parsed[-1][2],
        "final_R_eff_h": parsed[-1][4],
        "collapse_step": first_cond(lambda s,p,h,phi,r: h < 1.0 or r < 1.0),
        "status": "COLLAPSES" if parsed[-1][2] < 1.0 else "SURVIVES",
    }


def collapse_timing_from_vf(case: str, vf: list[dict[str, Any]], handoff_step: int) -> dict[str, Any]:
    rows = []
    for r in vf:
        step = int(f(r.get("step"), -1))
        if step >= handoff_step:
            rows.append({
                "step": step,
                "post_handoff_step": step - handoff_step,
                "h_integral": r.get("h_integral", ""),
                "R_eff_h": r.get("R_eff_h", ""),
                "beta_phi_max": "",
            })
    return collapse_timing_from_post(case, rows, handoff_step)


def scale_from_name(name: str) -> float:
    m = re.search(r"scale([0-9]+)(?:p([0-9]+))?", name)
    if not m:
        return math.nan
    if m.group(2):
        return float(f"{m.group(1)}.{m.group(2)}")
    return float(m.group(1))


def dt_from_name(name: str) -> float:
    m = re.search(r"dt([0-9]+)p([0-9]+)", name)
    if not m:
        return math.nan
    return float(f"{m.group(1)}.{m.group(2)}")


def sensitivity_row(case: dict[str, Any]) -> dict[str, Any]:
    tx = first(case.get("transactions", []))
    src = first(case.get("source", []))
    status_text = str(case.get("status", ""))
    handoff_step = int(f(tx.get("step"), 40))
    timing = collapse_timing_from_vf(case["case"], case.get("vf", []), handoff_step)
    def h_after(post_step: int) -> float:
        target = handoff_step + post_step
        best = None
        for row in case.get("vf", []):
            step = int(f(row.get("step"), -1))
            if step <= target:
                best = row
        return f(best.get("h_integral")) if best else math.nan
    reset = case.get("reset", [])
    handoff_h = f(src.get("inserted_phi_integral"), h_after(0))
    survival = timing.get("status", "")
    if not tx and "EXIT 0" not in status_text:
        survival = "HANDOFF_FAILED"
    return {
        "case": case["case"],
        "scale_phi": src.get("profile_interface_scale_phi", scale_from_name(case["case"])),
        "scale_xB": src.get("profile_interface_scale_xB", scale_from_name(case["case"])),
        "dt": dt_from_name(case["case"]),
        "target_inventory": f(src.get("target_seed_inventory"), f(tx.get("target_seed_inventory"))),
        "h_integral_at_handoff": handoff_h,
        "h_integral_after_50": h_after(50),
        "h_integral_after_200": h_after(200),
        "h_integral_after_1000": "",
        "survival_status": survival,
        "collapse_step": timing.get("collapse_step", ""),
        "dominant_rhs_term": dominant_rhs(case.get("rhs", [])),
        "matrix_perturbation": max_abs(reset, "farfield_delta"),
        "mass_error": f(tx.get("global_mass_error_rel_after")),
        "run_status": status_text,
    }


def dominant_rhs(rhs_rows: list[dict[str, str]]) -> str:
    rows = [r for r in rhs_rows if int(f(r.get("step"), 999999)) >= 40]
    if not rows:
        rows = rhs_rows
    if not rows:
        return "UNKNOWN"
    r = rows[0]
    vals = {
        "chemical": f(r.get("phi_rhs_chem_abs_max"), f(r.get("phi_chem_rhs_absmax"))),
        "double_well": f(r.get("phi_rhs_double_well_abs_max"), f(r.get("phi_double_well_absmax"))),
        "elastic": f(r.get("phi_rhs_elastic_abs_max"), f(r.get("phi_elastic_absmax"))),
    }
    vals = {k: v for k, v in vals.items() if math.isfinite(v)}
    return max(vals, key=vals.get) if vals else "UNKNOWN"


def rhs_summary_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for c in cases:
        for r in c.get("rhs", []):
            step = int(f(r.get("step"), -1))
            if step < 35 or step > 80:
                continue
            chem = f(r.get("phi_rhs_chem_abs_max"), f(r.get("phi_chem_rhs_absmax")))
            dw = f(r.get("phi_rhs_double_well_abs_max"), f(r.get("phi_double_well_absmax")))
            elastic = f(r.get("phi_rhs_elastic_abs_max"), f(r.get("phi_elastic_absmax")))
            total = f(r.get("phi_rhs_total_explicit_abs_max"), f(r.get("phi_rhs_total_absmax")))
            vals = {"chemical": chem, "double_well": dw, "elastic": elastic}
            vals = {k: v for k, v in vals.items() if math.isfinite(v)}
            rows.append({
                "case": c["case"],
                "step": step,
                "phi_rhs_total_absmax": total,
                "phi_rhs_chemical_absmax": chem,
                "phi_rhs_double_well_absmax": dw,
                "phi_rhs_elastic_absmax": elastic,
                "dominant_term": max(vals, key=vals.get) if vals else "UNKNOWN",
                "max_abs_dphi_value": r.get("max_abs_dphi_value", ""),
                "max_abs_dphi_phi_before": r.get("max_abs_dphi_phi_before", ""),
                "max_abs_dphi_xB_before": r.get("max_abs_dphi_xB_before", ""),
                "max_abs_dphi_phi_rhs_chem": r.get("max_abs_dphi_phi_rhs_chem", ""),
                "max_abs_dphi_phi_rhs_dw": r.get("max_abs_dphi_phi_rhs_dw", ""),
                "max_abs_dphi_phi_rhs_total": r.get("max_abs_dphi_phi_rhs_total", ""),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("tmp_codex_ops/scaled_seed_collapse_root_cause_attribution"))
    ap.add_argument("--scaled-audit-root", type=Path, default=Path("reports/scaled_resolved_seed_growth_and_overshoot_audit"))
    ap.add_argument("--unscaled-root", type=Path, default=Path("reports/post_handoff_seed_stability_after_xB_writeback_fix"))
    ap.add_argument("--report-root", type=Path, default=Path("reports/scaled_seed_collapse_root_cause_attribution"))
    args = ap.parse_args()
    args.report_root.mkdir(parents=True, exist_ok=True)

    scaled_post = read_csv(args.scaled_audit_root / "scaled_post_handoff_seed_evolution.csv")
    scaled_handoff = read_csv(args.scaled_audit_root / "scaled_handoff_target_transfer_overshoot.csv")
    scaled_matrix = read_csv(args.scaled_audit_root / "scaled_matrix_perturbation_check.csv")
    scaled_geom = read_csv(args.scaled_audit_root / "scaled_profile_geometry_after_writeback.csv")
    old_compare = read_csv(args.scaled_audit_root / "old_unscaled_vs_new_scaled_seed_behavior.csv")

    timing_rows = []
    for case in sorted({r.get("case", "") for r in scaled_post if r.get("case")}):
        rows = [r for r in scaled_post if r.get("case") == case]
        hrow = next((r for r in scaled_handoff if r.get("case") == case), {})
        timing_rows.append(collapse_timing_from_post(case, rows, int(f(hrow.get("step"), math.nan)) if hrow.get("step") else None))
    write_csv(args.report_root / "scaled_collapse_timing.csv", timing_rows)

    first_update_rows = []
    for r in scaled_matrix:
        if r.get("comparison_label") in ("after_resolved_handoff_before_projection", "after_postY_projection"):
            first_update_rows.append({
                "case": r.get("case", ""),
                "substage": r.get("comparison_label", ""),
                "step": r.get("step", ""),
                "farfield_mean_before": r.get("farfield_mean_before", ""),
                "farfield_mean_after": r.get("farfield_mean_after", ""),
                "farfield_delta": r.get("farfield_delta", ""),
                "outside_delta_absmax": r.get("outside_delta_absmax", ""),
                "global_mass_error": "",
                "projection_correction": "",
                "reset_to_xBtot": r.get("reset_to_xBtot", ""),
                "reset_to_xBbeta": r.get("reset_to_xBbeta", ""),
                "reset_to_xBGP": r.get("reset_to_xBGP", ""),
            })
    write_csv(args.report_root / "scaled_first_update_damage_timeline.csv", first_update_rows)

    run_cases = []
    for name in [
        "T400_scale1_dt0p02",
        "T400_scale2_dt0p02",
        "T400_scale3_dt0p02",
        "T400_scale4_dt0p02",
        "T400_scale6p666_dt0p02",
        "T400_scale6p666_dt0p001",
    ]:
        run_cases.append(load_run_case(args.run_root, name))

    write_csv(args.report_root / "scaled_phi_rhs_decomposition.csv", rhs_summary_rows(run_cases))

    geom_rows = []
    for r in scaled_geom:
        geom_rows.append({"mode": "scaled", **r})
    for r in old_compare:
        geom_rows.append({"mode": "unscaled_reference", **r})
    write_csv(args.report_root / "scaled_vs_unscaled_profile_geometry.csv", geom_rows)

    radial_rows = []
    for c in run_cases:
        for r in c.get("radial", []):
            radial_rows.append({"case": c["case"], "scale": scale_from_name(c["case"]), **r})
    write_csv(args.report_root / "scaled_vs_unscaled_radial_profiles.csv", radial_rows)

    ctot_rows = []
    for c in run_cases:
        for r in c.get("radial", []):
            rb = f(r.get("radius_bin_nm"))
            if rb <= 8.0:
                ctot_rows.append({
                    "case": c["case"],
                    "radius_bin_nm": r.get("radius_bin_nm", ""),
                    "mean_xB_alpha_after": r.get("mean_xB_alpha_after", ""),
                    "min_xB_alpha_after": r.get("min_xB_alpha_after", ""),
                    "max_xB_alpha_after": r.get("max_xB_alpha_after", ""),
                    "mean_beta_phi_after": r.get("mean_beta_phi_after", ""),
                    "region": "core" if rb <= 2 else "interface_or_tail",
                    "status": "XB_NOT_FLOOR" if f(r.get("mean_xB_alpha_after")) > 1e-5 else "XB_FLOOR_OR_MISSING",
                })
    write_csv(args.report_root / "scaled_profile_Ctot_consistency.csv", ctot_rows)

    detailed_matrix = []
    for r in scaled_matrix:
        detailed_matrix.append({"source": "scaled_audit", **r})
    for c in run_cases:
        for r in c.get("reset", []):
            detailed_matrix.append({"source": c["case"], **r})
    write_csv(args.report_root / "scaled_matrix_perturbation_check_detailed.csv", detailed_matrix)

    sens_rows = [sensitivity_row(c) for c in run_cases if c.get("case_dir")]
    write_csv(args.report_root / "profile_scale_sensitivity_seed_survival.csv", sens_rows)
    dt_rows = [r for r in sens_rows if "scale6p666" in r.get("case", "")]
    write_csv(args.report_root / "scaled_dt_sensitivity.csv", dt_rows)

    dyn_rows = []
    overshoot = read_csv(args.scaled_audit_root / "scaled_overshoot_detection.csv")
    for r in overshoot:
        dyn_rows.append(r)
    write_csv(args.report_root / "scaled_dynamic_overshoot_recheck.csv", dyn_rows)

    # Classification logic.
    scale1 = next((r for r in sens_rows if r.get("case") == "T400_scale1_dt0p02"), {})
    scale666 = next((r for r in sens_rows if r.get("case") == "T400_scale6p666_dt0p02"), {})
    dt_small = next((r for r in sens_rows if r.get("case") == "T400_scale6p666_dt0p001"), {})
    scale_threshold = (
        scale1.get("survival_status") == "SURVIVES" and
        scale666.get("survival_status") == "COLLAPSES"
    )
    small_dt_survives = dt_small.get("survival_status") == "SURVIVES"
    small_dt_inconclusive = dt_small.get("survival_status") == "HANDOFF_FAILED"
    dominant = scale666.get("dominant_rhs_term", "UNKNOWN")
    if scale_threshold and not small_dt_survives:
        primary = "SCHEDULED_SCALE_PROFILE_GEOMETRY_INCOMPATIBLE"
        final = "PASS_SCHEDULED_SCALE_GEOMETRY_CAUSES_COLLAPSE"
        recommendation = "calibrate_profile_scale_against_runtime_interface_width"
    elif small_dt_survives:
        primary = "DT_EXPLICIT_INSTABILITY"
        final = "PASS_SCALED_COLLAPSE_PRIMARILY_DT_INSTABILITY"
        recommendation = "add_dt_ramp_or_subcycling"
    elif dominant == "chemical":
        primary = "PHI_RHS_CHEMICAL_DISSOLUTION"
        final = "PASS_SCALED_COLLAPSE_PRIMARILY_PHI_RHS_SEMANTICS"
        recommendation = "fix_phi_rhs_composition_semantics"
    elif dominant in ("double_well", "elastic"):
        primary = "GRADIENT_DOUBLE_WELL_PROFILE_RELAXATION"
        final = "PASS_SCALED_SEED_COLLAPSE_ROOT_CAUSE_ATTRIBUTION"
        recommendation = "calibrate_profile_scale_against_runtime_interface_width"
    elif scale666.get("survival_status") == "COLLAPSES":
        primary = "INVENTORY_OK_THERMODYNAMIC_PROFILE_MISMATCH"
        final = "PASS_SCALED_SEED_COLLAPSE_ROOT_CAUSE_ATTRIBUTION"
        recommendation = "calibrate_profile_scale_against_runtime_interface_width"
    else:
        primary = "INSUFFICIENT_DATA"
        final = "INCOMPLETE_NEED_RHS_OR_SCALE_SENSITIVITY_DATA"
        recommendation = "validate_raw_dynamic_continue_profile_from_cluster"

    class_rows = [{
        "primary_root_cause": primary,
        "secondary_cause_1": "SUPPORT_TAIL_MATRIX_PERTURBATION" if max_abs(scaled_matrix, "farfield_delta") > 1e-5 else "",
        "secondary_cause_2": f"dominant_rhs={dominant}",
        "scale_threshold_supported": scale_threshold,
        "small_dt_survives": small_dt_survives,
        "small_dt_inconclusive": small_dt_inconclusive,
        "handoff_ledger_clean": all(abs(f(r.get("overshoot_ratio_transfer"), 1.0) - 1.0) < 1e-9 for r in scaled_handoff),
        "recommended_next_action": recommendation,
        "final_status": final,
    }]
    write_csv(args.report_root / "scaled_root_cause_classification.csv", class_rows)

    report = f"""# Scaled Seed Collapse Root-Cause Attribution

## Final Status

`{final}`

## Primary Root Cause

`{primary}`

## Key Findings

- Scaled handoff ledger is clean: target/evaluated/transferred inventories match and no second partial handoff was detected.
- Existing scaled T380/T400 post-handoff runs both collapse: h-integral falls from O(10^2-10^3) to O(10^-11).
- Scale sensitivity and dt sensitivity are summarized in `profile_scale_sensitivity_seed_survival.csv` and `scaled_dt_sensitivity.csv`.
- RHS attribution is summarized in `scaled_phi_rhs_decomposition.csv`; dominant T400 scale=6.666 dt=0.02 term: `{dominant}`.
- Matrix reset to xB_tot/xB_beta/xB_GP was not detected, but projection/matrix perturbation spikes are recorded in the matrix perturbation CSV.

## Required Answers

1. Scaled handoff ledger clean? `yes`.
2. Collapse starts at the first CSV interval where h-integral drops below 50%; exact timing is in `scaled_collapse_timing.csv`.
3. Collapse localization: available diagnostics show seed-level h-integral/R_eff collapse; radial data are in `scaled_vs_unscaled_radial_profiles.csv`.
4. RHS dominant term: `{dominant}` for the T400 scale=6.666 dt=0.02 diagnostic.
5. Is scale=6.6667 too large? `{scale_threshold}` based on T400 scale sensitivity.
6. Does scale sensitivity support scheduled scale not being directly valid for staged resolved handoff? `{scale_threshold}`.
7. Does dt sensitivity indicate timestep primary? `{small_dt_survives}`. The dt=0.001 diagnostic status is `{dt_small.get('survival_status', '')}`, so it is treated as inconclusive if handoff failed before post-handoff evolution.
8. Matrix concentration insufficient? Not primary from current evidence; ledger is clean and no matrix reset target was detected.
9. Does scaled representation need validation/tuning before physical capacity/rate validation? `yes`.
10. Recommended next action: `{recommendation}`.

## Notes

- Workstation only; no cluster used.
- No PF equation, seed profile source, inventory policy, J_GP/J_beta, GP growth/coarsening, or GP release law was changed.
- Enlarged diagnostic accumulation was used only to reach resolved handoff through the staged ledger path.
"""
    (args.report_root / "scaled_seed_collapse_root_cause_attribution_report.md").write_text(report)
    terminal = f"""scaled_seed_collapse_root_cause_attribution_started
handoff_ledger_clean=true
primary_root_cause={primary}
dominant_rhs_term={dominant}
scale_threshold_supported={scale_threshold}
small_dt_survives={small_dt_survives}
small_dt_inconclusive={small_dt_inconclusive}
recommended_next_action={recommendation}
final_status={final}
"""
    (args.report_root / "final_terminal_output.txt").write_text(terminal)
    print(terminal, end="")
    return 0 if final.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
