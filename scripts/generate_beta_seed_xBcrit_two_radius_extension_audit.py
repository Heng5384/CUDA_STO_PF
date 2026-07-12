#!/usr/bin/env python3
"""Generate the beta-seed xBcrit titration audit with two-radius extension.

This generator is deliberately diagnostic/report-only. It does not run CUDA,
modify PF equations, modify seed profiles, or reinterpret the J_GP literature
driving force as GP-zone thermodynamics.
"""

from __future__ import annotations

import csv
import json
import math
import argparse
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/beta_seed_xBcrit_titration_with_two_radius_extension"
XBFAR = 0.0078305391025
XBSWEEP = [XBFAR, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.024, 0.030]
TEMPS = [380, 400]
LIB = {380: "nlib_00006", 400: "nlib_dc_T400_xB003"}
DT_CODE = 0.02
T_REAL_UNIT = {380: 49.54630476715921, 400: 41.12958542455477}
CURRENT_RADIUS_REPORT = ROOT / "reports/resolved_handoff_inserted_seed_effective_radius_audit/inserted_seed_effective_radius_summary.csv"
SURROGATE_CEILING_TABLE = ROOT / "reports/rsmd_internal_thermo_extraction_audit/gp_curvature_release_ceiling_table.csv"
CACHE_MANIFEST = ROOT / "Results/runtime_profile_cache/dx1p0/cache_manifest.csv"
LIBRARY_STAGING = ROOT / "data/nucleus_library/nucleus_library.with_Zr.staging.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def fmt_xb(xb: float) -> str:
    return f"{xb:.12f}".rstrip("0").rstrip(".").replace(".", "p")


def selected_seed_rows() -> dict[int, dict[str, str]]:
    rows = read_csv(CURRENT_RADIUS_REPORT)
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        if row.get("profile_mode") != "baseline":
            continue
        T = int(round(fnum(row.get("T_C"))))
        if T in TEMPS and row.get("library_entry_id") == LIB[T]:
            out[T] = row
    return out


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def runtime_main_outputs_available() -> bool:
    path = OUT / "beta_seed_xBcrit_titration_runs.csv"
    rows = read_csv(path)
    if not rows:
        return False
    for row in rows:
        if truthy(row.get("run_completed")):
            return True
        if row.get("final_fate") and row.get("final_fate") != "NOT_RUN":
            return True
        if "workstation runtime unavailable" not in row.get("comments", ""):
            return True
    return False


def current_seed_metric(T: int, field: str, default: Any = "") -> Any:
    return selected_seed_rows().get(T, {}).get(field, default)


def read_metadata(entry_id: str) -> dict[str, Any]:
    path = ROOT / f"Results/runtime_profile_cache/dx1p0/{entry_id}/seed_profile_metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def planned_run_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    selected = selected_seed_rows()
    for T in TEMPS:
        seed = selected.get(T, {})
        for xb in XBSWEEP:
            run_id = f"T{T}_xB{fmt_xb(xb)}_stable_unscaled_noAQ"
            common = {
                "run_id": run_id,
                "T_C": T,
                "xB_uniform": f"{xb:.12e}",
                "library_entry": LIB[T],
            }
            config_rows.append({
                **common,
                "dx_nm": 1.0,
                "lambda_sm_m": "6.000000000000e-10",
                "runtime_interface_width_nm": 0.6,
                "scale_phi": 1.0,
                "scale_xB": 1.0,
                "scheduled_scale_disabled": True,
                "catalog_T_tol_C": 1.0,
                "catalog_xB_tol": 1.0,
                "catalog_selection_policy": "fixed_selected_temperature_seed_for_uniform_matrix_titration",
                "selected_library_entry_id": LIB[T],
                "analytic_fallback_used": False,
                "xB_profile_used": True,
                "phi_profile_used": True,
                "writeback_mode": "preserve_profile_xB_alpha_in_support",
                "AQ_GP_population_disabled": True,
                "GP_inventory_disabled_or_epsilon_only": True,
                "GP_release_enabled": False,
                "RSMD_enabled": False,
                "J_GP_birth_enabled": False,
                "surrogate_JGP_used_for_GP_thermo": False,
                "external_matrix_reset_detected": "not_evaluated",
                "extra_matrix_draw_detected": "not_evaluated",
                "double_counting_detected": "not_evaluated",
                "runtime_config_status": "PLANNED_NOT_EXECUTED_WORKSTATION_UNAVAILABLE",
            })
            run_rows.append({
                **common,
                "handoff_step": "",
                "post_handoff_steps": "",
                "physical_time_s": "",
                "run_completed": False,
                "no_nan_inf": "",
                "mass_closure_max_rel": "",
                "initial_R_eff_h_nm": seed.get("R_eff_h_after_projection_nm", ""),
                "final_R_eff_h_nm": "",
                "delta_R_eff_h_nm": "",
                "initial_h_integral": seed.get("h_integral_after_projection", ""),
                "final_h_integral": "",
                "delta_h_integral_frac": "",
                "final_M_beta": "",
                "final_phi_max": "",
                "final_fate": "NOT_RUN",
                "comments": "workstation runtime unavailable; valid uniform no-GP stable-baseline titration was not executed",
            })
            class_rows.append({
                **common,
                "initial_R_eff_h_nm": seed.get("R_eff_h_after_projection_nm", ""),
                "final_R_eff_h_nm": "",
                "delta_R_eff_h_nm": "",
                "delta_R_threshold_nm": max(0.10, 0.02 * fnum(seed.get("R_eff_h_after_projection_nm"), 0.0)),
                "slope_last_window_nm_per_step": "",
                "initial_h_integral": seed.get("h_integral_after_projection", ""),
                "final_h_integral": "",
                "delta_h_integral_frac": "",
                "final_M_beta": "",
                "final_phi_max": "",
                "fate": "NOT_RUN",
                "fate_confidence": "none",
                "reason": "no valid main-line runtime data",
                "physical_or_diagnostic_label": "DIAGNOSTIC_UNIFORM_MATRIX_NO_AQ_GP_PLANNED",
            })
    return config_rows, run_rows, class_rows


def reference_time_series_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aq = read_csv(ROOT / "reports/post_handoff_seed_stability_after_xB_writeback_fix/post_handoff_seed_evolution_after_writeback_fix.csv")
    for row in aq:
        T = int(round(fnum(row.get("T_C"))))
        if T not in TEMPS:
            continue
        rows.append({
            "run_id": f"REFERENCE_AQ_T{T}_NOT_VALID_TITRATION",
            "T_C": T,
            "xB_uniform": f"{XBFAR:.12e}",
            "step": row.get("step", ""),
            "post_handoff_step": row.get("post_handoff_step", ""),
            "physical_time_s": fnum(row.get("post_handoff_step"), 0.0) * DT_CODE * T_REAL_UNIT[T],
            "R_eff_h_nm": row.get("R_eff_nm", ""),
            "h_integral": row.get("h_phi_integral", ""),
            "M_beta": row.get("M_beta_raw", ""),
            "phi_max": row.get("beta_phi_max", ""),
            "far_field_xB_mean": row.get("farfield_xB_alpha", ""),
            "mass_error_rel": row.get("global_mass_error_rel", ""),
            "classification_running": "REFERENCE_AQ_GP_ENABLED_NOT_VALID_TITRATION",
        })
    xB03 = read_csv(ROOT / "reports/runtime_4grid_xB03_farfield_matrix_growth_test/xB03_post_handoff_seed_evolution.csv")
    for row in xB03:
        case = row.get("case", "")
        if "unscaled" not in case or "xB03" not in case:
            continue
        T = 380 if "T380" in case else 400 if "T400" in case else 0
        if T not in TEMPS:
            continue
        rows.append({
            "run_id": f"REFERENCE_4GRID_{case}_EXCLUDED",
            "T_C": T,
            "xB_uniform": "3.000000000000e-02",
            "step": row.get("step", ""),
            "post_handoff_step": row.get("post_handoff_step", ""),
            "physical_time_s": fnum(row.get("post_handoff_step"), 0.0) * DT_CODE * T_REAL_UNIT[T],
            "R_eff_h_nm": row.get("R_eff_h_nm", ""),
            "h_integral": row.get("h_integral", ""),
            "M_beta": row.get("M_beta_raw", ""),
            "phi_max": row.get("beta_phi_max", ""),
            "far_field_xB_mean": row.get("farfield_xB_alpha_proxy", ""),
            "mass_error_rel": row.get("mass_error_rel", ""),
            "classification_running": "REFERENCE_4GRID_INTERFACE_EXCLUDED_FROM_MAIN_TITRATION",
        })
    return rows


def interpolation_rows() -> list[dict[str, Any]]:
    rows = []
    for T in TEMPS:
        seed = selected_seed_rows().get(T, {})
        rows.append({
            "T_C": T,
            "library_entry": LIB[T],
            "R_eff_h_initial_nm": seed.get("R_eff_h_after_projection_nm", ""),
            "xBcrit_lower_bound": "",
            "xBcrit_upper_bound": "",
            "xBcrit_interpolated": "",
            "bracket_width": "",
            "interpolation_metric": "not_available",
            "monotonicity_status": "NOT_EVALUATED_VALID_MAIN_SWEEP_MISSING",
            "xBcrit_status": "NOT_BRACKETED_NO_VALID_STABLE_BASELINE_SWEEP",
            "confidence": "none",
            "recommended_next_action": "run stable 0.6 nm unscaled preserve-mode uniform-matrix sweep on workstation",
        })
    return rows


def cache_rows_by_entry() -> dict[str, dict[str, str]]:
    return {row.get("library_entry_id", ""): row for row in read_csv(CACHE_MANIFEST)}


def library_rows() -> list[dict[str, str]]:
    return read_csv(LIBRARY_STAGING)


def two_radius_design_rows() -> list[dict[str, Any]]:
    selected = selected_seed_rows()
    cache = cache_rows_by_entry()
    rows: list[dict[str, Any]] = []
    for T in TEMPS:
        seed = selected.get(T, {})
        rows.append({
            "T_C": T,
            "radius_point_role": "current_selected_profile",
            "library_entry_id": LIB[T],
            "candidate_T_C": T,
            "candidate_xB": "0.03",
            "R_eff_h_nm": seed.get("R_eff_h_after_projection_nm", ""),
            "r_seed_metadata_nm": seed.get("r_seed_metadata_nm", ""),
            "profile_file": seed.get("profile_file", ""),
            "profile_status": "BUILT",
            "valid_for_two_radius_fit": True,
            "validity_class": "VALID_CURRENT_RADIUS_POINT",
            "invalid_reason": "",
            "required_action": "none for first radius point",
        })
    for row in library_rows():
        T = int(round(fnum(row.get("T_C"), -1)))
        if T != 380 or row.get("library_entry_id") == "nlib_00006":
            continue
        rows.append({
            "T_C": 380,
            "radius_point_role": "same_temperature_missing_bridge_candidate",
            "library_entry_id": row.get("library_entry_id"),
            "candidate_T_C": row.get("T_C"),
            "candidate_xB": row.get("xB"),
            "R_eff_h_nm": "",
            "r_seed_metadata_nm": row.get("r_seed_nm"),
            "profile_file": "",
            "profile_status": cache.get(row.get("library_entry_id", ""), {}).get("status", "MISSING"),
            "valid_for_two_radius_fit": False,
            "validity_class": "PENDING_PROFILE_AVAILABILITY",
            "invalid_reason": row.get("missing_reason") or cache.get(row.get("library_entry_id", ""), {}).get("reason", "DYNAMIC_CONTINUE_BRIDGE_MISSING"),
            "required_action": "complete dynamic-continue bridge and build dx=1 runtime cache before using as second radius",
        })
    rows.append({
        "T_C": 400,
        "radius_point_role": "same_temperature_second_radius_candidate",
        "library_entry_id": "NONE_FOUND",
        "candidate_T_C": 400,
        "candidate_xB": "",
        "R_eff_h_nm": "",
        "r_seed_metadata_nm": "",
        "profile_file": "",
        "profile_status": "MISSING",
        "valid_for_two_radius_fit": False,
        "validity_class": "PENDING_PROFILE_AVAILABILITY",
        "invalid_reason": "no second production-valid T400 dynamic-continued runtime profile found in dx=1 cache or staging table",
        "required_action": "generate a second T400 dynamic-continued seed profile at another radius/composition before capillary inference",
    })
    meta450 = read_metadata("nlib_dc_T450_xB003")
    rows.append({
        "T_C": 400,
        "radius_point_role": "temperature_mismatch_candidate",
        "library_entry_id": "nlib_dc_T450_xB003",
        "candidate_T_C": meta450.get("T_C", 450),
        "candidate_xB": meta450.get("xB", 0.03),
        "R_eff_h_nm": "",
        "r_seed_metadata_nm": meta450.get("r_seed_nm", ""),
        "profile_file": "Results/runtime_profile_cache/dx1p0/nlib_dc_T450_xB003/faceted_family_profiles.csv",
        "profile_status": "BUILT_BUT_TEMPERATURE_MISMATCH",
        "valid_for_two_radius_fit": False,
        "validity_class": "INVALID_TEMPERATURE_MISMATCH",
        "invalid_reason": "T450 profile cannot serve as a T400 same-temperature second-radius point",
        "required_action": "do not use for two-radius PF capillary inference at T400",
    })
    for T in TEMPS:
        seed = selected.get(T, {})
        scaled_r = ""
        if T == 400:
            scaled = next((r for r in read_csv(CURRENT_RADIUS_REPORT) if r.get("case") == "T400_scheduled_scale6p666"), {})
            scaled_r = scaled.get("R_eff_h_after_projection_nm", "")
        rows.append({
            "T_C": T,
            "radius_point_role": "scheduled_scaled_diagnostic_invalid",
            "library_entry_id": LIB[T],
            "candidate_T_C": T,
            "candidate_xB": "0.03",
            "R_eff_h_nm": scaled_r,
            "r_seed_metadata_nm": seed.get("r_seed_metadata_nm", ""),
            "profile_file": seed.get("profile_file", ""),
            "profile_status": "DIAGNOSTIC_ONLY",
            "valid_for_two_radius_fit": False,
            "validity_class": "INVALID_SCHEDULED_SCALE_COLLAPSE_PATH",
            "invalid_reason": "scheduled scale=6.6667 is not a production radius point and previously produced collapse",
            "required_action": "exclude from production PF capillary inference",
        })
    return rows


def capillary_inference_rows(interp_rows_arg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interp = {int(fnum(row.get("T_C"))): row for row in interp_rows_arg}
    selected = selected_seed_rows()
    rows = []
    for T in TEMPS:
        seed = selected.get(T, {})
        xcrit = interp.get(T, {}).get("xBcrit_interpolated", "")
        rows.append({
            "T_C": T,
            "available_valid_radius_count": 1 if seed else 0,
            "radius_1_library_entry": LIB[T],
            "radius_1_R_eff_h_nm": seed.get("R_eff_h_after_projection_nm", ""),
            "radius_1_xBcrit": xcrit,
            "radius_2_library_entry": "",
            "radius_2_R_eff_h_nm": "",
            "radius_2_xBcrit": "",
            "fit_status": "INSUFFICIENT_RADII",
            "inferred_planar_xBcrit": "",
            "inferred_capillary_coefficient": "",
            "usable_for_GP_release_planning": False,
            "comments": "two same-temperature production-valid dynamic-continued radius points are required; current cache has only one for this T",
        })
    return rows


def surrogate_warning_rows(interp_rows_arg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interp = {int(fnum(row.get("T_C"))): row for row in interp_rows_arg}
    rows: list[dict[str, Any]] = []
    for row in read_csv(SURROGATE_CEILING_TABLE):
        T = int(round(fnum(row.get("T_C"), -1)))
        if T not in TEMPS:
            continue
        Rgp = fnum(row.get("R_i_nm"))
        if not any(abs(Rgp - r) < 1e-12 for r in [1.0, 1.269245574815, 2.0, 4.0, 5.0]):
            continue
        xcrit = interp[T]["xBcrit_interpolated"]
        rows.append({
            "T_C": T,
            "seed_library_entry": LIB[T],
            "seed_R_eff_h_nm": selected_seed_rows().get(T, {}).get("R_eff_h_after_projection_nm", ""),
            "xBcrit": xcrit,
            "xBcrit_status": interp[T]["xBcrit_status"],
            "GP_R_nm": row.get("R_i_nm"),
            "xB_surrogate_drive_ceiling": row.get("xB_release_ceiling"),
            "source_file": str(SURROGATE_CEILING_TABLE.relative_to(ROOT)),
            "diagnostic_relation": "X_BCRIT_NOT_AVAILABLE_DIAGNOSTIC_ONLY",
            "warning_status": "SURROGATE_DRIVE_ONLY_NOT_GP_THERMO",
            "comments": "This table came from runtime Ag2Te-equivalent delta_gv surrogate extraction. It is not a GP solvus, not GP interface energy, and not a GP release ceiling.",
        })
    if not rows:
        for T in TEMPS:
            rows.append({
                "T_C": T,
                "seed_library_entry": LIB[T],
                "seed_R_eff_h_nm": selected_seed_rows().get(T, {}).get("R_eff_h_after_projection_nm", ""),
                "xBcrit": "",
                "xBcrit_status": "NOT_BRACKETED_NO_VALID_STABLE_BASELINE_SWEEP",
                "GP_R_nm": "",
                "xB_surrogate_drive_ceiling": "",
                "source_file": str(SURROGATE_CEILING_TABLE.relative_to(ROOT)),
                "diagnostic_relation": "SURROGATE_TABLE_NOT_AVAILABLE",
                "warning_status": "SURROGATE_DRIVE_WARNING_GENERATED",
                "comments": "surrogate-drive table missing",
            })
    return rows


def required_supply_rows(interp_rows_arg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interp = {int(fnum(row.get("T_C"))): row for row in interp_rows_arg}
    rows = []
    for T in TEMPS:
        seed = selected_seed_rows().get(T, {})
        inv = fnum(seed.get("evaluated_profile_inventory"))
        xcrit = interp.get(T, {}).get("xBcrit_interpolated", "")
        xcrit_status = interp.get(T, {}).get("xBcrit_status", "NOT_BRACKETED_NO_VALID_STABLE_BASELINE_SWEEP")
        ready = xcrit not in {"", "nan", "NA"} and math.isfinite(fnum(xcrit))
        rows.append({
            "T_C": T,
            "seed_library_entry": LIB[T],
            "seed_R_eff_h_nm": seed.get("R_eff_h_after_projection_nm", ""),
            "evaluated_profile_inventory_xB_cell_units": seed.get("evaluated_profile_inventory", ""),
            "xBcrit_status": xcrit_status,
            "xBcrit": xcrit,
            "required_matrix_halo_xB": "",
            "required_halo_radius_nm": "",
            "required_supply_status": "READY_FOR_SUPPLY_MAPPING_INPUT" if ready else "NOT_READY_XBCRIT_MISSING",
            "candidate_uniform_xB_grid": ",".join(f"{x:.12g}" for x in XBSWEEP),
            "next_inputs_needed": "valid workstation CUDA sweep with AQ GP disabled; second same-temperature dynamic-continued radius profile for capillary inference",
            "comments": (
                f"profile inventory is known ({inv:.6g}); xBcrit is available for supply-map fitting"
                if ready else
                f"profile inventory is known ({inv:.6g}) but required halo supply cannot be mapped until xBcrit is bracketed"
            ),
        })
    return rows


def build_report(final_status: str, interp_rows_arg: list[dict[str, Any]], runtime_status: str) -> str:
    selected = selected_seed_rows()
    interp = interp_rows_arg
    design = two_radius_design_rows()
    valid_two_radius = [r for r in design if r.get("valid_for_two_radius_fit") is True]
    run_rows = read_csv(OUT / "beta_seed_xBcrit_titration_runs.csv")
    caveat_rows = [
        r for r in run_rows
        if r.get("final_fate") == "COLLAPSE" or fnum(r.get("mass_closure_max_rel"), 0.0) > 1.0
    ]
    lines = [
        "# Beta Seed xBcrit Titration With Two-Radius Extension Audit",
        "",
        f"final_status = `{final_status}`",
        "",
        "## Scope",
        "",
        "This audit is diagnostic only. It does not modify PF equations, seed profiles, J_GP parameters, inventory policy, GP release, or RSMD source terms.",
        "",
        "Critical semantic correction: the current `J_GP` literature path uses an Ag2Te-equivalent surrogate chemical driving force for GP birth flux only. It is not a GP-zone thermodynamic model, not a GP solvus, not a GP interface-energy measurement, and must not be used to infer a true GP release ceiling.",
        "",
        "## Runtime Execution Status",
        "",
        runtime_status,
        "",
        "## Selected Current Seed Profiles",
        "",
        "| T_C | library entry | active R_eff_h after write/projection (nm) | profile | scale active | writeback |",
        "|---:|---|---:|---|---|---|",
    ]
    for T in TEMPS:
        seed = selected.get(T, {})
        lines.append(
            f"| {T} | {LIB[T]} | {seed.get('R_eff_h_after_projection_nm', '')} | "
            f"{seed.get('profile_file', '')} | {seed.get('scale_active', '')} | {seed.get('writeback_mode', '')} |"
        )
    lines.extend([
        "",
        "## xBcrit Interpolation",
        "",
        "| T_C | status | xBcrit | lower | upper | confidence |",
        "|---:|---|---:|---:|---:|---|",
    ])
    for row in interp:
        lines.append(
            f"| {row['T_C']} | {row['xBcrit_status']} | {row['xBcrit_interpolated']} | "
            f"{row['xBcrit_lower_bound']} | {row['xBcrit_upper_bound']} | {row['confidence']} |"
        )
    lines.extend([
        "",
        "Existing AQ-GP and 4-grid xB=0.03 data are intentionally labeled as references and excluded from the main interpolation.",
        "",
        "## Numerical Caveats",
        "",
    ])
    if caveat_rows:
        lines.extend([
            "The following runtime points require numerical caution because they collapse or show very large global closure residuals. They are retained in the raw tables, but should not be used as clean high-xB physical growth evidence without a smaller-dt retest.",
            "",
            "| run_id | T_C | xB | fate | max mass rel | final R_eff_h |",
            "|---|---:|---:|---|---:|---:|",
        ])
        for row in caveat_rows:
            lines.append(
                f"| {row.get('run_id')} | {row.get('T_C')} | {row.get('xB_uniform')} | "
                f"{row.get('final_fate')} | {row.get('mass_closure_max_rel')} | {row.get('final_R_eff_h')} |"
            )
    else:
        lines.append("No collapse or large global residual rows were detected in the parsed runtime table.")
    lines.extend([
        "",
        "## Two-Radius Extension",
        "",
        f"Valid same-temperature radius points currently available: `{len(valid_two_radius)}` total, but only one per target temperature.",
        "",
        "- T380 has current `nlib_00006`; neighboring T380 entries exist in the staging table but are missing dynamic-continue bridges/runtime profiles.",
        "- T400 has current `nlib_dc_T400_xB003`; no second production-valid T400 runtime profile was found.",
        "- T450 `nlib_dc_T450_xB003` is built but cannot be used as a T400 same-temperature radius point.",
        "- Scheduled scale=6.6667 is diagnostic-only and invalid for production two-radius fitting because that path has already shown collapse/scale pathology.",
        "",
        "Therefore `beta_seed_two_radius_pf_capillary_inference.csv` reports `INSUFFICIENT_RADII` for both temperatures.",
        "",
        "## Surrogate Ceiling Warning",
        "",
        "`beta_seed_xBcrit_vs_surrogate_ceiling_warning.csv` intentionally renames the previous comparison as a surrogate-drive warning table. It must not be read as true GP release thermodynamics. Even with bracketed seed-side xBcrit, the surrogate ceiling rows remain diagnostic-only.",
        "",
        "## Required Supply Map",
        "",
        "`required_supply_map_next_inputs.csv` records seed inventory and the bracketed seed-side xBcrit values. It is ready as an input table for a separate supply-map fit, but not a physical GP release law.",
        "",
        "## Answers",
        "",
        f"1. T380 xBcrit status: `{next((r.get('xBcrit_status') for r in interp if int(fnum(r.get('T_C'))) == 380), 'MISSING')}`.",
        f"2. T400 xBcrit status: `{next((r.get('xBcrit_status') for r in interp if int(fnum(r.get('T_C'))) == 400), 'MISSING')}`.",
        "3. The current active profiles are dynamic-continued library profiles with unscaled `scale_phi=1`, `scale_xB=1`, preserve-mode xB writeback, no analytic fallback.",
        "4. The two-radius extension is pending profile availability: current cache does not contain two same-temperature production-valid radius points.",
        "5. The Ag2Te-equivalent J_GP surrogate driving force is not used here to define GP release thermodynamics.",
        "",
        "## Recommended Next Action",
        "",
        "If xBcrit is not bracketed, run/refine the stable-baseline uniform-matrix sweep. In parallel, generate one additional same-temperature dynamic-continued seed profile for T380 and T400 before attempting PF capillary inference.",
    ])
    return "\n".join(lines) + "\n"


def final_status_from_interp(interp_rows_arg: list[dict[str, Any]]) -> str:
    statuses = {
        int(fnum(row.get("T_C"))): row.get("xBcrit_status", "")
        for row in interp_rows_arg
    }
    both_bracketed = all(statuses.get(T) == "BRACKETED" for T in TEMPS)
    any_bracketed = any(statuses.get(T) == "BRACKETED" for T in TEMPS)
    if both_bracketed:
        return "PASS_BETA_SEED_XB_CRIT_TITRATION_AUDIT_TWO_RADIUS_PENDING"
    if any_bracketed:
        return "PARTIAL_BETA_SEED_XB_CRIT_TITRATION_AUDIT"
    return "FAIL_BETA_SEED_XB_CRIT_TITRATION_AUDIT"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-static-main", action="store_true",
                        help="overwrite main titration CSVs with planned-not-run rows")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    use_existing_runtime = runtime_main_outputs_available() and not args.force_static_main
    if use_existing_runtime:
        interp_rows = read_csv(OUT / "beta_seed_xBcrit_interpolation.csv") or interpolation_rows()
        runtime_status = "Runtime titration CSVs were detected and preserved. This generator appended the two-radius extension, surrogate-drive warning, required-supply map, and final acceptance report without overwriting the parsed CUDA results."
    else:
        config_rows, run_rows, class_rows = planned_run_rows()
        time_rows = reference_time_series_rows()
        interp_rows = interpolation_rows()
        runtime_status = "The required uniform-matrix no-AQ-GP stable-baseline titration was not executed in this pass. The generated CSVs therefore contain a complete planned run grid plus labeled reference-only rows where available."
        write_csv(OUT / "beta_seed_xBcrit_runtime_config_audit.csv", config_rows, [
            "run_id", "T_C", "xB_uniform", "dx_nm", "lambda_sm_m", "runtime_interface_width_nm",
            "scale_phi", "scale_xB", "scheduled_scale_disabled", "selected_library_entry_id",
            "catalog_T_tol_C", "catalog_xB_tol", "catalog_selection_policy",
            "analytic_fallback_used", "xB_profile_used", "phi_profile_used", "writeback_mode",
            "AQ_GP_population_disabled", "GP_inventory_disabled_or_epsilon_only", "GP_release_enabled",
            "RSMD_enabled", "J_GP_birth_enabled", "surrogate_JGP_used_for_GP_thermo",
            "external_matrix_reset_detected", "extra_matrix_draw_detected", "double_counting_detected",
            "runtime_config_status",
        ])
        write_csv(OUT / "beta_seed_xBcrit_titration_runs.csv", run_rows, [
            "run_id", "T_C", "xB_uniform", "library_entry", "handoff_step", "post_handoff_steps",
            "physical_time_s", "run_completed", "no_nan_inf", "mass_closure_max_rel",
            "initial_R_eff_h_nm", "final_R_eff_h_nm", "delta_R_eff_h_nm",
            "initial_h_integral", "final_h_integral", "delta_h_integral_frac",
            "final_M_beta", "final_phi_max", "final_fate", "comments",
        ])
        write_csv(OUT / "beta_seed_xBcrit_time_series.csv", time_rows, [
            "run_id", "T_C", "xB_uniform", "step", "post_handoff_step", "physical_time_s",
            "R_eff_h_nm", "h_integral", "M_beta", "phi_max", "far_field_xB_mean",
            "mass_error_rel", "classification_running",
        ])
        write_csv(OUT / "beta_seed_xBcrit_classification.csv", class_rows, [
            "run_id", "T_C", "xB_uniform", "initial_R_eff_h_nm", "final_R_eff_h_nm",
            "delta_R_eff_h_nm", "delta_R_threshold_nm", "slope_last_window_nm_per_step",
            "initial_h_integral", "final_h_integral", "delta_h_integral_frac",
            "final_M_beta", "final_phi_max", "fate", "fate_confidence", "reason",
            "physical_or_diagnostic_label",
        ])
        write_csv(OUT / "beta_seed_xBcrit_interpolation.csv", interp_rows, [
            "T_C", "library_entry", "R_eff_h_initial_nm", "xBcrit_lower_bound",
            "xBcrit_upper_bound", "xBcrit_interpolated", "bracket_width",
            "interpolation_metric", "monotonicity_status", "xBcrit_status",
            "confidence", "recommended_next_action",
        ])
    design_rows = two_radius_design_rows()
    cap_rows = capillary_inference_rows(interp_rows)
    warning_rows = surrogate_warning_rows(interp_rows)
    supply_rows = required_supply_rows(interp_rows)
    write_csv(OUT / "beta_seed_two_radius_extension_design.csv", design_rows, [
        "T_C", "radius_point_role", "library_entry_id", "candidate_T_C", "candidate_xB",
        "R_eff_h_nm", "r_seed_metadata_nm", "profile_file", "profile_status",
        "valid_for_two_radius_fit", "validity_class", "invalid_reason", "required_action",
    ])
    write_csv(OUT / "beta_seed_two_radius_pf_capillary_inference.csv", cap_rows, [
        "T_C", "available_valid_radius_count", "radius_1_library_entry", "radius_1_R_eff_h_nm",
        "radius_1_xBcrit", "radius_2_library_entry", "radius_2_R_eff_h_nm",
        "radius_2_xBcrit", "fit_status", "inferred_planar_xBcrit",
        "inferred_capillary_coefficient", "usable_for_GP_release_planning", "comments",
    ])
    write_csv(OUT / "beta_seed_xBcrit_vs_surrogate_ceiling_warning.csv", warning_rows, [
        "T_C", "seed_library_entry", "seed_R_eff_h_nm", "xBcrit", "xBcrit_status",
        "GP_R_nm", "xB_surrogate_drive_ceiling", "source_file", "diagnostic_relation",
        "warning_status", "comments",
    ])
    write_csv(OUT / "required_supply_map_next_inputs.csv", supply_rows, [
        "T_C", "seed_library_entry", "seed_R_eff_h_nm",
        "evaluated_profile_inventory_xB_cell_units", "xBcrit_status", "xBcrit",
        "required_matrix_halo_xB", "required_halo_radius_nm", "required_supply_status",
        "candidate_uniform_xB_grid", "next_inputs_needed", "comments",
    ])

    final_status = final_status_from_interp(interp_rows)
    (OUT / "beta_seed_xBcrit_titration_acceptance_report.md").write_text(
        build_report(final_status, interp_rows, runtime_status)
    )
    by_T = {int(fnum(row.get("T_C"))): row for row in interp_rows}
    terminal = [
        "beta_seed_xBcrit_two_radius_extension_audit_started",
        f"T380_xBcrit_status={by_T.get(380, {}).get('xBcrit_status', 'MISSING')}",
        f"T380_xBcrit={by_T.get(380, {}).get('xBcrit_interpolated') or 'NA'}",
        f"T400_xBcrit_status={by_T.get(400, {}).get('xBcrit_status', 'MISSING')}",
        f"T400_xBcrit={by_T.get(400, {}).get('xBcrit_interpolated') or 'NA'}",
        "two_radius_extension_status=TWO_RADIUS_EXTENSION_PENDING_PROFILE_AVAILABILITY",
        "PF_capillary_inference_status=INSUFFICIENT_RADII",
        "surrogate_ceiling_warning_status=GENERATED_DIAGNOSTIC_ONLY_NOT_GP_THERMO",
        f"required_supply_map_ready={'true' if any(r.get('required_supply_status') == 'READY_FOR_SUPPLY_MAPPING_INPUT' for r in supply_rows) else 'false'}",
        f"workstation_runtime_status={'RUNTIME_RESULTS_PRESERVED' if use_existing_runtime else 'NO_RUNTIME_RESULTS_STATIC_AUDIT'}",
        f"final_status={final_status}",
    ]
    (OUT / "final_terminal_output.txt").write_text("\n".join(terminal) + "\n")
    print("\n".join(terminal))


if __name__ == "__main__":
    main()
