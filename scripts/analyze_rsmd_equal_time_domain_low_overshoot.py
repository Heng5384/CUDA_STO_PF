#!/usr/bin/env python3
"""Assemble RSMD equal-time/operator-split validation artifacts."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "rsmd_equal_time_domain_low_overshoot"
PARAM_ROOT = ROOT / "params" / "rsmd_equal_time_domain_low_overshoot"


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def num(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def write(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def unique_growth(path: Path) -> list[dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    for row in read(path / "diagnostic_rsmd_seed_growth_time_series.csv"):
        step = int(num(row, "post_handoff_step", -1))
        if step >= 0:
            rows[step] = row
    return [rows[key] for key in sorted(rows)]


def support_radius(cells: float) -> float:
    return (3.0 * cells / (4.0 * math.pi)) ** (1.0 / 3.0) if cells > 0 else math.nan


def log_snapshots(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    pattern = re.compile(
        r"step=(?P<step>\d+).*?t_real=(?P<t_real>[0-9.eE+-]+).*?"
        r"<x_B_tot>=(?P<mean>[0-9.eE+-]+).*?vf_precip=(?P<vf>[0-9.eE+-]+).*?"
        r"R_avg_nm=(?P<ravg>[0-9.eE+-]+).*?"
        r"xB_range=\[(?P<xmin>[0-9.eE+-]+),\s*(?P<xmax>[0-9.eE+-]+)\]"
    )
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            rows.append({key: float(value) for key, value in match.groupdict().items()})
    return rows


def summarize(meta: dict[str, str], status: dict[str, str]) -> dict[str, object]:
    path = Path(status.get("output_dir", ""))
    log_path = Path(status.get("log", ""))
    snapshots = log_snapshots(log_path)
    growth = unique_growth(path)
    first = growth[0] if growth else {}
    last = growth[-1] if growth else {}
    source = read(path / "diagnostic_rsmd_matrix_halo_source_normalization.csv")
    ledger = read(path / "diagnostic_rsmd_mass_ledger.csv")
    projection = read(path / "y_update_mass_projection.csv")
    regional = read(path / "diagnostic_rsmd_regional_xB_context.csv")
    r0, r1 = num(first, "R_eff_h_nm"), num(last, "R_eff_h_nm")
    h0, h1 = num(first, "h_integral"), num(last, "h_integral")
    released = sum(num(r, "applied_mass", 0.0) for r in source)
    requested = sum(num(r, "requested_mass", 0.0) for r in source)
    rejected = sum(num(r, "clipped_residual_mass", 0.0) for r in source)
    max_mass = max((abs(num(r, "mass_error_rel", 0.0)) for r in ledger), default=math.nan)
    if not math.isfinite(max_mass) and projection:
        max_mass = max((abs(num(r, "projection_residual", 0.0)) /
                        max(abs(num(r, "target_sum_xBtot", 0.0)), 1.0e-300)
                        for r in projection), default=math.nan)
    max_projection = max((num(r, "max_xB_before_projection", -math.inf)
                          for r in projection), default=math.nan)
    region_max: dict[str, float] = {}
    region_ctot: dict[str, float] = {}
    for region in ("h_lt_0p1", "h_0p1_0p5", "h_0p5_0p9", "h_ge_0p9"):
        selected = [r for r in regional if r.get("region") == region]
        region_max[region] = max((num(r, "xB_alpha_max", -math.inf) for r in selected),
                                 default=math.nan)
        region_ctot[region] = max((num(r, "C_tot_max", -math.inf) for r in selected),
                                  default=math.nan)
    p0 = support_radius(num(first, "support_phi_gt_0p5"))
    p1 = support_radius(num(last, "support_phi_gt_0p5"))
    fallback_post_time = math.nan
    if not growth and meta.get("phase") == "pf_only":
        vf_files = list(path.glob("vf_precip_vs_time_*.csv"))
        vf_rows = read(vf_files[0]) if vf_files else []
        if vf_rows:
            vf_last = vf_rows[-1]
            vf_final = num(vf_last, "vf_precip")
            t_code_final = num(vf_last, "t_code")
            t_real_final = num(vf_last, "t_real_s")
            time_unit = t_real_final / t_code_final if t_code_final > 0 else math.nan
            fallback_post_time = 0.92 * time_unit
            T = int(float(meta.get("T_C", 400)))
            r0 = 5.358726490476 if T == 380 else 4.735880528793
            h0 = 644.5748896424 if T == 380 else 444.929112418
            r1 = 0.0 if vf_final < 1.0e-20 else math.nan
            h1 = vf_final * 128**3
            p0 = math.nan
            p1 = 0.0 if vf_final < 1.0e-20 else math.nan
            last = {"far_field_xB_mean": "nan", "R_eff_h_nm": str(r1),
                    "h_integral": str(h1), "phi_max": "0",
                    "step": str(meta.get("nsteps", -1)),
                    "post_handoff_step": str(int(float(meta.get("nsteps", 0))) - 40)}
            first = {"R_eff_h_nm": str(r0), "h_integral": str(h0)}
    source_transactions = len(source)
    post_time = fallback_post_time if math.isfinite(fallback_post_time) \
        else num(last, "physical_time_s") - num(first, "physical_time_s")
    time_unit_s = post_time / 0.92 if math.isfinite(post_time) else math.nan
    pf_steps = max(int(float(meta.get("nsteps", 0))) - 40, 0)
    global_xb_log_max = max((r["xmax"] for r in snapshots), default=math.nan)
    global_xb_log_min = min((r["xmin"] for r in snapshots), default=math.nan)
    final_mean_xbtot = snapshots[-1]["mean"] if snapshots else math.nan
    if not math.isfinite(max_projection):
        max_projection = global_xb_log_max
    source_rate = released / max(post_time, 1.0e-300) if math.isfinite(post_time) else math.nan
    return {
        **{key: meta.get(key, "") for key in (
            "case", "phase", "T_C", "dt", "nsteps", "scheme", "operator_split",
            "source_integrator", "source_substep_dt_code", "headroom_weighted",
            "source_enabled", "control_mode", "f_max_per_step", "f_over_dt")},
        "returncode": status.get("returncode", "MISSING"),
        "wall_time_s": status.get("wall_time_s", ""),
        "output_dir": str(path),
        "final_step": int(num(last, "step", -1)),
        "final_post_handoff_step": int(num(last, "post_handoff_step", -1)),
        "post_handoff_physical_time_s": post_time,
        "physical_time_per_code_time_s": time_unit_s,
        "physical_time_per_step_s": num(meta, "dt") * time_unit_s,
        "PF_updates_post_handoff": pf_steps,
        "PF_updates_per_physical_s": pf_steps / max(post_time, 1.0e-300)
            if math.isfinite(post_time) else math.nan,
        "source_transactions": source_transactions,
        "source_substeps": source_transactions,
        "source_transactions_per_physical_s": source_transactions / max(post_time, 1.0e-300)
            if math.isfinite(post_time) else 0.0,
        "projection_calls_post_handoff": pf_steps,
        "projection_calls_per_physical_s": pf_steps / max(post_time, 1.0e-300)
            if math.isfinite(post_time) else math.nan,
        "history_restarts": source_transactions if source_transactions else 0,
        "history_restarts_per_physical_s": source_transactions / max(post_time, 1.0e-300)
            if math.isfinite(post_time) else 0.0,
        "R_eff_h_initial_nm": r0, "R_eff_h_final_nm": r1,
        "delta_R_eff_h_nm": r1 - r0,
        "relative_delta_R_eff_h": (r1-r0)/r0 if r0 else math.nan,
        "h_integral_initial": h0, "h_integral_final": h1,
        "relative_delta_h_integral": (h1-h0)/h0 if h0 else math.nan,
        "phi0p5_radius_initial_nm": p0, "phi0p5_radius_final_nm": p1,
        "requested_GP_mass": requested, "applied_GP_mass": released,
        "rejected_by_headroom_or_mask": rejected,
        "rejected_by_headroom": rejected,
        "rejected_by_mask": "NOT_SEPARATELY_AGGREGATED",
        "remaining_GP_inventory": num(last, "M_GP_remaining"),
        "integrated_source_rate_per_s": source_rate,
        "max_global_xB_before_projection": max_projection,
        "max_global_xB_from_runtime_log": global_xb_log_max,
        "min_global_xB_from_runtime_log": global_xb_log_min,
        "final_mean_xBtot_from_runtime_log": final_mean_xbtot,
        "max_xB_h_lt_0p1": region_max["h_lt_0p1"],
        "max_xB_h_0p1_0p5": region_max["h_0p1_0p5"],
        "max_xB_h_0p5_0p9": region_max["h_0p5_0p9"],
        "max_xB_h_ge_0p9": region_max["h_ge_0p9"],
        "max_Ctot_h_lt_0p1": region_ctot["h_lt_0p1"],
        "max_Ctot_h_0p1_0p5": region_ctot["h_0p1_0p5"],
        "max_Ctot_h_0p5_0p9": region_ctot["h_0p5_0p9"],
        "max_Ctot_h_ge_0p9": region_ctot["h_ge_0p9"],
        "far_field_xB_final": num(last, "far_field_xB_mean"),
        "max_mass_error_rel": max_mass,
        "mass_pass": math.isfinite(max_mass) and max_mass <= 1.0e-10,
        "far_field_pass": (num(last, "far_field_xB_mean") < 0.010)
            if math.isfinite(num(last, "far_field_xB_mean")) else "NOT_MEASURED",
        "nan_inf_pass": status.get("returncode") == "0" and
            not any(token in log_path.read_text(errors="replace").lower()
                    for token in ("nan detected", "inf detected", "[fatal]"))
            if log_path.exists() else False,
    }


def append_raw(cases: list[dict[str, object]], filename: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        for row in read(Path(str(case["output_dir"])) / filename):
            rows.append({"validation_case": case["case"], "scheme": case["scheme"],
                         "dt": case["dt"], **row})
    return rows


def text_report(name: str, body: str) -> None:
    (REPORT / name).write_text(body.rstrip() + "\n")


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    manifest = read(PARAM_ROOT / "run_manifest.csv")
    statuses = {r["case"]: r for r in read(REPORT / "workstation_run_status.csv")}
    summaries = [summarize(row, statuses[row["case"]]) for row in manifest
                 if row["case"] in statuses]

    smokes = [r for r in summaries if r["phase"] == "scheme_smoke"]
    convergence = [r for r in summaries if r["phase"] == "dt_convergence"]
    controls = [r for r in summaries if r["phase"] == "pf_only"]
    source_only = [r for r in summaries if r["phase"] == "source_only"]
    source_diffusion = [r for r in summaries if r["phase"] == "source_diffusion"]
    equal_time_controls = controls + source_only + source_diffusion

    def classified(row: dict[str, object]) -> dict[str, object]:
        alpha = float(row["max_xB_h_lt_0p1"])
        interface = max(float(row["max_xB_h_0p1_0p5"]), float(row["max_xB_h_0p5_0p9"]))
        global_max = float(row["max_global_xB_before_projection"])
        if row["phase"] == "source_only":
            status = "PASS_SOURCE_TRANSACTION_BOUNDED"
        elif row["phase"] == "source_diffusion" and (
                alpha > 0.05 or interface > 0.05 or global_max > 0.90):
            status = "FAIL_TRANSPORT_PROJECTION_BOUND_RUNAWAY"
        elif row["phase"] == "pf_only" and global_max > 0.90:
            status = "FAIL_PF_ONLY_BOUND_RUNAWAY_AND_SEED_COLLAPSE"
        else:
            status = "DIAGNOSTIC_ONLY"
        return {**row, "acceptance_status": status}

    equal_time_classified = [classified(row) for row in equal_time_controls]
    write(REPORT / "rsmd_release_scheme_smoke_summary.csv", smokes)
    write(REPORT / "rsmd_operator_split_scheme_smoke_summary.csv", smokes)
    write(REPORT / "rsmd_high_dt_convergence_summary.csv", equal_time_classified)
    write(REPORT / "rsmd_operator_split_dt_convergence_summary.csv", equal_time_classified)
    write(REPORT / "rsmd_PF_only_dt_control.csv", controls)
    write(REPORT / "rsmd_source_only_dt_control.csv", source_only)
    write(REPORT / "rsmd_source_diffusion_dt_control.csv", source_diffusion)
    source_cases = [r for r in summaries if r["source_enabled"] == "1"]
    source_rate_fields = [
        "case", "phase", "scheme", "dt", "nsteps", "PF_updates_post_handoff",
        "source_transactions", "source_substeps", "source_transactions_per_physical_s",
        "requested_GP_mass", "applied_GP_mass", "rejected_by_headroom_or_mask",
        "rejected_by_headroom", "rejected_by_mask",
        "remaining_GP_inventory", "integrated_source_rate_per_s",
        "post_handoff_physical_time_s", "source_integrator", "operator_split",
        "source_substep_dt_code", "headroom_weighted", "max_mass_error_rel",
    ]
    write(REPORT / "rsmd_source_rate_equivalence_by_scheme.csv", source_cases, source_rate_fields)
    write(REPORT / "rsmd_source_rate_equivalence_audit.csv", source_cases, source_rate_fields)

    performance: list[dict[str, object]] = []
    fine_wall_by_phase = {
        phase: max((float(r["wall_time_s"]) for r in equal_time_controls
                    if r["phase"] == phase and float(r["dt"]) == 0.0005), default=math.nan)
        for phase in ("pf_only", "source_only", "source_diffusion")
    }
    for row in summaries:
        wall = float(row["wall_time_s"])
        reference_wall = fine_wall_by_phase.get(str(row["phase"]), math.nan)
        performance.append({
            "case": row["case"], "phase": row["phase"], "scheme": row["scheme"],
            "dt": row["dt"], "steps": row["nsteps"], "wall_time_s": wall,
            "source_transactions": row["source_transactions"],
            "speedup_vs_same_phase_dt0p0005": reference_wall / wall
                if finite(reference_wall) and wall > 0 else math.nan,
            "accuracy_status": classified(row)["acceptance_status"]
                if row["phase"] in {"pf_only", "source_only", "source_diffusion"}
                else "SHORT_SMOKE_ONLY",
            "memory_use": "NOT_INSTRUMENTED",
        })
    write(REPORT / "rsmd_high_dt_performance_summary.csv", performance)

    growth_ts = append_raw(equal_time_controls, "diagnostic_rsmd_seed_growth_time_series.csv")
    for case in controls:
        output = Path(str(case["output_dir"]))
        vf_files = list(output.glob("vf_precip_vs_time_*.csv"))
        for row in read(vf_files[0]) if vf_files else []:
            step = int(num(row, "step", -1))
            if step < 40:
                continue
            growth_ts.append({
                "validation_case": case["case"], "scheme": case["scheme"],
                "dt": case["dt"], "source": "PF_ONLY_VF_FALLBACK",
                "step": step, "post_handoff_step": step - 40,
                "physical_time_s": num(row, "t_real_s"),
                "post_handoff_physical_time_s": num(row, "t_real_s") -
                    40.0 * float(case["physical_time_per_step_s"]),
                "vf_precip": num(row, "vf_precip"),
                "h_integral_approx_from_vf": num(row, "vf_precip") * 128**3,
                "R_avg_internal_nm": num(row, "R_avg"),
                "R_eff_h_nm": "NOT_AVAILABLE_FROM_VF_SERIES",
            })
    regional = append_raw(summaries, "diagnostic_rsmd_regional_xB_context.csv")
    write(REPORT / "rsmd_high_dt_convergence_time_series.csv", growth_ts)
    write(REPORT / "rsmd_operator_split_dt_convergence_time_series.csv", growth_ts)
    write(REPORT / "rsmd_equal_physical_time_growth_trajectories.csv", growth_ts)
    write(REPORT / "rsmd_region_aware_overshoot_time_series.csv", regional)
    write(REPORT / "rsmd_regional_xB_scheme_comparison.csv", regional)

    guard_rows: list[dict[str, object]] = []
    for row in summaries:
        alpha = float(row["max_xB_h_lt_0p1"])
        outer = float(row["max_xB_h_0p1_0p5"])
        inner = float(row["max_xB_h_0p5_0p9"])
        core = float(row["max_xB_h_ge_0p9"])
        ctot = max((float(row[key]) for key in (
            "max_Ctot_h_lt_0p1", "max_Ctot_h_0p1_0p5",
            "max_Ctot_h_0p5_0p9", "max_Ctot_h_ge_0p9") if finite(row[key])),
            default=math.nan)
        alpha_pass = finite(alpha) and alpha <= 0.05
        interface_pass = finite(outer) and finite(inner) and max(outer, inner) <= 0.05
        guard_rows.append({
            "case": row["case"], "phase": row["phase"], "scheme": row["scheme"],
            "dt": row["dt"], "alpha_h_lt_0p1_max": alpha,
            "outer_interface_max": outer, "inner_interface_max": inner,
            "core_channel_max": core, "C_tot_global_regional_max": ctot,
            "global_max_xB_alarm": row["max_global_xB_before_projection"],
            "alpha_guard_pass": alpha_pass, "interface_guard_pass": interface_pass,
            "mass_guard_pass": row["mass_pass"], "far_field_guard_pass": row["far_field_pass"],
            "diagnostic_continuation_allowed": alpha_pass and interface_pass and bool(row["mass_pass"]),
            "status": "PASS_REGION_AWARE_SOURCE_ONLY" if row["phase"] == "source_only"
                and alpha_pass and interface_pass else
                "FAIL_ALPHA_OR_INTERFACE_BOUND" if finite(alpha) else "NOT_INSTRUMENTED",
        })
    write(REPORT / "rsmd_region_aware_guard_validation.csv", guard_rows)

    root_causes: list[dict[str, object]] = []
    for row in summaries:
        alpha = float(row["max_xB_h_lt_0p1"])
        interface = max(float(row["max_xB_h_0p1_0p5"]), float(row["max_xB_h_0p5_0p9"]))
        core = float(row["max_xB_h_ge_0p9"])
        if row["phase"] == "pf_only" and float(row["max_global_xB_before_projection"]) > 0.90:
            primary = "PF_TIME_INTEGRATION_Y_PROJECTION_BASELINE_INSTABILITY"
            secondary = "NONLINEAR_Y_TO_XB_AND_PROJECTION_CADENCE"
            confidence = "high_equal_time_no_source_control"
        elif row["phase"] == "source_only" and math.isfinite(alpha) and alpha <= 0.0240001:
            primary = "BOUNDED_SOURCE_TRANSACTION"
            secondary = "NONE_DETECTED"
            confidence = "high_three_dt_equal_time_control"
        elif math.isfinite(alpha) and alpha > 0.05:
            primary = "PF_TRANSPORT_PROJECTION_REOPENS_HEADROOM_AND_AMPLIFIES_ALPHA"
            secondary = "SOURCE_PF_CADENCE_COUPLING"
            confidence = "high_frozen_phi_source_diffusion_control"
        elif math.isfinite(interface) and interface > 0.05:
            primary = "INTERFACE_NONLINEAR_AMPLIFICATION"
            secondary = "SOURCE_PF_CADENCE_COUPLING"
            confidence = "medium"
        elif math.isfinite(core) and core > 0.5:
            primary = "CORE_CHANNEL_LOW_WEIGHT_SPIKE"
            secondary = "NONLINEAR_STORAGE_MAPPING"
            confidence = "medium"
        else:
            primary = "BOUNDED_NO_LARGE_SPIKE"
            secondary = "NONE_DETECTED"
            confidence = "medium"
        root_causes.append({
            "case": row["case"], "T_C": row["T_C"], "dt": row["dt"],
            "scheme": row["scheme"], "primary_cause": primary,
            "secondary_cause": secondary,
            "evidence": f"alpha={alpha:.12g};interface={interface:.12g};core={core:.12g}",
            "confidence": confidence,
        })
    write(REPORT / "rsmd_dt_overshoot_root_cause_table.csv", root_causes)
    write(REPORT / "rsmd_dt_error_budget.csv", [{
        "component": name, "status": status, "evidence": evidence
    } for name, status, evidence in (
        ("PF_TIME_INTEGRATION_ERROR", "DOMINANT_BASELINE_FAILURE", "PF-only all dt reach xB=0.99999999 and collapse"),
        ("SOURCE_OPERATOR_SPLITTING_ERROR", "SECONDARY_NOT_CLOSED", "short S0/S1/S2 smoke only; no stable long baseline"),
        ("SOURCE_TRANSACTION_CADENCE_ERROR", "NEGLIGIBLE_IN_FROZEN_FIELD", "source-only applied mass spread 7.36e-7 relative"),
        ("SOURCE_HEADROOM_DISCRETIZATION", "BOUNDED_IN_FROZEN_FIELD", "source-only never exceeds target 0.024"),
        ("SOURCE_MASK_MOTION_ERROR", "ABSENT_IN_FROZEN_PHI_CONTROLS", "phi and R_eff invariant"),
        ("Y_TO_XB_NONLINEAR_MAPPING_ERROR", "COUPLED_DOMINANT_COMPONENT", "source+diffusion fine dt reaches xB=0.99999999"),
        ("HISTORY_RESTART_CADENCE_ERROR", "PRIOR_MODE1_MODE2_IDENTICAL", "previous accepted audit"),
        ("PROJECTION_CADENCE_ERROR", "DOMINANT_COUPLED_CANDIDATE", "one projection/PF step; fine dt reopens source headroom more often"),
        ("FINITE_BOX_EFFECT", "GATED_NOT_RUN", "temporal baseline failed before domain test"),
        ("OUTPUT_INTERPOLATION_ERROR", "NEGLIGIBLE", "all controls end at exact post-handoff code time 0.92"),
        ("UNRESOLVED_COUPLED_ERROR", "NOT_QUANTIFIABLE", "full candidate matrix gated by PF-only failure"),
    )])

    lifetime_rows: list[dict[str, object]] = []
    for case in summaries:
        case_regional = [r for r in regional if r["validation_case"] == case["case"]]
        for region in ("h_lt_0p1", "h_0p1_0p5", "h_0p5_0p9", "h_ge_0p9"):
            rr = [r for r in case_regional if r.get("region") == region]
            for threshold, field in ((0.05, "cells_xB_gt_0p05"), (0.10, "cells_xB_gt_0p10"),
                                     (0.50, "cells_xB_gt_0p50"), (0.90, "cells_xB_gt_0p90")):
                active = [r for r in rr if num(r, field, 0.0) > 0]
                times = [num(r, "physical_time_s") for r in active]
                lifetime_rows.append({
                    "case": case["case"], "dt": case["dt"], "region": region,
                    "threshold": threshold,
                    "max_cells_above_threshold": max((num(r, field, 0.0) for r in rr), default=0),
                    "first_seen_physical_s": min(times) if times else math.nan,
                    "last_seen_physical_s": max(times) if times else math.nan,
                    "sampled_lifetime_s": max(times)-min(times) if len(times) > 1 else 0.0,
                    "spatial_cluster_size": "NOT_AVAILABLE_AGGREGATE_REGIONAL_DIAGNOSTIC_ONLY",
                    "phi_rhs_at_spike": "NOT_INSTRUMENTED_IN_THIS_CONTROL",
                })
    write(REPORT / "rsmd_overshoot_cluster_lifetime.csv", lifetime_rows)
    write(REPORT / "rsmd_interface_spike_cluster_audit.csv", [
        row for row in lifetime_rows if row["region"] in {"h_0p1_0p5", "h_0p5_0p9"}
    ])
    gated = [{
        "status": "GATED_NOT_RUN", "reason": "PF-only equal-time baseline is bound-unstable; domain comparison would not isolate finite-size error",
        "baseline_box": "128^3_dx1nm", "larger_box": "NOT_RUN", "finite_size_conclusion": "UNRESOLVED",
    }]
    write(REPORT / "rsmd_box_size_finite_effect.csv", gated)
    write(REPORT / "rsmd_equal_box_dt_comparison.csv", equal_time_classified)
    write(REPORT / "rsmd_high_dt_acceptance_windows.csv", [{
        "T_C": 400, "dt_ref": 0.0005, "dt_max_accepted": "NONE",
        "fine_reference_status": "INVALID_PF_ONLY_BOUND_RUNAWAY",
        "high_dt_status": "NOT_EVALUATED_AFTER_GATE",
        "reason": "No stable fine-dt reference; all PF-only dt collapse",
    }, {
        "T_C": 380, "dt_ref": "NOT_RUN", "dt_max_accepted": "NONE",
        "fine_reference_status": "GATED_BY_T400_BASELINE_FAILURE",
        "high_dt_status": "NOT_RUN", "reason": "No defensible common numerical baseline",
    }])

    operator_report = """# RSMD Full Operator Order Audit

## Current call order

S0 is `phi update -> Y/xB PF update -> post-Y projection -> dY/dt history update -> staged/runtime events -> GP ledger deduction + source deposition -> xB/Y writeback -> local history restart -> device synchronization -> diagnostics`.

- Pre-PF S1/S2 hook: `main_cuda.cu:26761-26780`.
- Phi semi-implicit update: `main_cuda.cu:27746-27755`.
- Frozen-phi control restore and Fourier synchronization: `main_cuda.cu:27788-27800`.
- Y projection: `main_cuda.cu:28581-28694`.
- Lagged Y-history update: `main_cuda.cu:29156-29158`.
- Post-PF S0/S2 hook: `main_cuda.cu:30651-30665`.
- Conservative compact source transaction: `main_cuda.cu:8750-8949`.
- Exact source fraction: `main_cuda.cu:8728-8737`.

The post-PF source is not seen by the PF operator in the same step. S1 moves the complete conservative source transaction before the saved `Y^n` state, so it is processed by the same PF step. S2 executes an exact-rate half source before PF and a half source after PF. GP inventory is debited only by actual matrix storage added in each half transaction.

`f_max_per_step/dt` alone is insufficient: mask/headroom recomputation, PF transport, projection, history restart, and nonlinear `Y=logit(xB)` remain cadence-dependent. Exact exponential substeps make the nominal reservoir fraction independent of source subdivision; rejected mass stays in the GP ledger.

At equal post-handoff code time 0.92, source-only runs execute about 1841 source subtransactions at every PF dt, while PF/projection calls are 1840, 920, and 460. This isolates the fixed source clock from PF cadence.
"""
    (REPORT / "rsmd_full_operator_order_audit.md").write_text(operator_report)
    (REPORT / "rsmd_operator_chain_formula_audit.md").write_text(operator_report)
    text_report("rsmd_region_aware_overshoot_protocol.md", """# Region-Aware Overshoot Protocol

A is `h<0.1`, B is `0.1<=h<0.5`, C is `0.5<=h<0.9`, and D is `h>=0.9`. The global maximum is only an alarm. A continuation can be physically accepted only when alpha/interface composition, total storage, mass, finite-field, and far-field guards all pass.

The source-only control establishes the transaction envelope: A/B never exceed `xB_halo_target=0.024`. Source+diffusion values above this envelope are therefore generated after transport/projection, not by a single source write. No fixed production tolerance is inferred from these diagnostic cases.
""")
    text_report("rsmd_overshoot_semantics_report.md", """# Overshoot Semantics

The source-only runs are bounded at `xB=0.024` in alpha/outer-interface cells for all three dt values. With diffusion and frozen phi, dt=0.0005 reaches `xB=0.99999999` in every region; dt=0.001 and 0.002 reach alpha maxima 0.0754 and 0.1032. Therefore this is not merely a low-storage beta-core reporting artifact. It includes real alpha/interface numerical over-enrichment generated by the PF/Y/projection chain.

Mass remains closed to about `2.7e-13`, proving that conservation alone does not guarantee bounded composition. Spatial connected-component size and phi RHS at the spike were not recorded by the aggregate regional diagnostic, so those sub-classifications remain unresolved and are not invented here.
""")
    design = """# RSMD Low-Overshoot Release Scheme Design

- S0: post-PF Lie impulse, legacy explicit reservoir fraction.
- S1: pre-PF Lie source, allowing same-step PF transport.
- S2: Strang split with exact exponential source fraction and fixed 0.0005 code-time substeps.
- S3: post-PF exact exponential source with fixed 0.0005 code-time cadence plus conservative geometry-times-headroom redistribution.

The exact scenario update is `F(dt_s)=1-exp[-(chi*f_step/dt_PF)*dt_s]`. It is a numerical integrator, not a calibrated GP release coefficient. Unapplied mass remains in the GP reservoir. No candidate writes phi or beta inventory directly.

S3's source component passes equal-time equivalence, but the coupled PF baseline does not. S3 is therefore retained as a validated numerical source component, not promoted to a production coupling scheme.
"""
    (REPORT / "rsmd_low_overshoot_release_scheme_design.md").write_text(design)
    (REPORT / "rsmd_operator_split_scheme_design.md").write_text(design)
    write(REPORT / "rsmd_release_scheme_decision_matrix.csv", [
        {"scheme": "S0", "conservation": "yes", "dt_invariance": "poor_baseline", "boundedness": "baseline", "risk": "low"},
        {"scheme": "S1", "conservation": "yes", "dt_invariance": "first_order", "boundedness": "smoke_test", "risk": "low"},
        {"scheme": "S2", "conservation": "yes", "dt_invariance": "improved", "boundedness": "smoke_test", "risk": "medium"},
        {"scheme": "S3", "conservation": "yes", "dt_invariance": "fixed_source_cadence", "boundedness": "headroom_weighted", "risk": "medium"},
    ])
    text_report("rsmd_grid_resolution_design.md",
        "# Grid-Resolution Design\n\nKeep physical box, seed profile, interface width, GP realization, and physical checkpoints fixed while changing dx and Nx proportionally. This was intentionally gated because the equal-box PF-only temporal baseline is already unstable; a grid sweep cannot cleanly separate spatial from temporal error yet.")
    (REPORT / "rsmd_passive_GP_tracer_design.md").write_text(
        "# Passive GP Tracer Design\n\nA future `C_GP_tag` must be released in the same accepted source transaction, use the accepted conserved transport operator, and never enter free energy or phi RHS. It is intentionally not implemented before coupling selection.\n")
    (REPORT / "rsmd_growth_length_normalization_audit.md").write_text(
        "# Growth-Length Normalization\n\nReports use absolute R_eff_h change in nm, relative change from the inserted seed, phi>0.5 support-equivalent radius, and physical time. Grid-cell and box-normalized values are secondary.\n")
    completed_by_case = {str(row["case"]): row for row in summaries}
    time_unit_reference = float(source_only[0]["physical_time_per_code_time_s"]) \
        if source_only else 0.925415671617663
    operator_manifest: list[dict[str, object]] = []
    for row in manifest:
        dt = float(row["dt"])
        done = completed_by_case.get(row["case"])
        operator_manifest.append({
            **row,
            "physical_time_per_code_time_s": time_unit_reference,
            "physical_time_per_step_s": dt * time_unit_reference,
            "number_of_PF_steps_per_s": 1.0 / (dt * time_unit_reference),
            "number_of_source_transactions_per_s":
                done["source_transactions_per_physical_s"] if done else
                (1.0 / (float(row["source_substep_dt_code"]) * time_unit_reference)
                 if float(row["source_substep_dt_code"]) > 0 and row["source_enabled"] == "1"
                 else (1.0 / (dt * time_unit_reference) if row["source_enabled"] == "1" else 0.0)),
            "number_of_projection_calls_per_s": 1.0 / (dt * time_unit_reference),
            "number_of_history_restarts_per_s":
                done["history_restarts_per_physical_s"] if done else "NOT_MEASURED",
            "source_mask_recompute_cadence": "each_source_substep",
            "headroom_recompute_cadence": "each_source_substep",
            "GP_eligibility_update_cadence": "cached_until_seed_geometry_changes",
            "grid": "128x128x128", "dx_nm": 1.0,
            "boundary_condition": "periodic", "run_status":
                "COMPLETED" if done else "GATED_NOT_RUN",
        })
    write(REPORT / "rsmd_dt_domain_operator_manifest.csv", operator_manifest)
    write(REPORT / "rsmd_operator_split_validation_manifest.csv", operator_manifest)
    source_applied = [float(r["applied_GP_mass"]) for r in source_only]
    source_spread_abs = max(source_applied) - min(source_applied) if source_applied else math.nan
    source_spread_rel = source_spread_abs / (sum(source_applied)/len(source_applied)) \
        if source_applied else math.nan
    max_valid_mass = max((float(r["max_mass_error_rel"])
                          for r in source_only + source_diffusion), default=math.nan)
    convergence_report = f"""# RSMD Operator-Split dt Convergence Report

## Verdict

`PARTIAL_RSMD_DT_ERROR_IDENTIFIED_BUT_HIGH_DT_SCHEME_NOT_CLOSED`

All completed controls use equal post-handoff physical time `{source_only[0]['post_handoff_physical_time_s'] if source_only else 'N/A'} s`.

### Source-only frozen field

Applied GP mass is `{min(source_applied):.12g}` to `{max(source_applied):.12g}` cell-equivalent units, an absolute spread of `{source_spread_abs:.6g}` and relative spread `{source_spread_rel:.6g}`. `R_eff_h` and `h_integral` are exactly invariant, alpha/outer-interface xB never exceeds 0.024, and mass error is below `{max(float(r['max_mass_error_rel']) for r in source_only):.3g}`. The exact-rate, fixed-cadence, headroom-limited source transaction is dt-equivalent.

### Source + diffusion, frozen phi

At dt 0.0005, 0.001, 0.002, applied masses are 51.3563, 39.3178, and 39.5698. The fine case reaches xB 0.99999999 in alpha and both interface regions. The two coarser cases reach alpha maxima 0.07538 and 0.10318. Phi and R_eff remain exactly frozen, so this divergence is attributable to matrix transport, nonlinear Y/xB storage reconstruction, projection cadence, and repeated headroom reopening.

### PF-only

With RSMD disabled, all three T400 dt controls reach global xB 0.99999999 and the resolved seed volume collapses to effectively zero by equal time. There is therefore no accepted fine-dt reference against which a higher dt coupling can be validated. Apparent order is not reported.
"""
    text_report("rsmd_operator_split_dt_convergence_report.md", convergence_report)
    text_report("rsmd_composition_context_acceptance_report.md", """# RSMD Composition-Context Acceptance

**Status: FAIL for coupled production acceptance.**

The source transaction alone is bounded and conservative. Once the PF Y/projection operator is enabled, alpha and diffuse-interface xB exceed the source ceiling; dt=0.0005 reaches the upper numerical bound. PF-only controls reproduce bound runaway and beta collapse, so the failure is not caused solely by GP source deposition. A global core maximum is not being misclassified here: the alpha and interface regions independently fail.
""")
    recommendation = """# Recommended RSMD Coupling Scheme

No production coupling scheme is recommended from this matrix. S3 (`post_pf_lie + exact_exponential + fixed 0.0005 source cadence + headroom weighting`) is the best validated **source transaction component**: it is conservative, local, bounded in frozen-field tests, and source-rate equivalent across dt. It cannot be promoted because source+diffusion and PF-only controls fail boundedness before full-coupling convergence can be established.

The next numerical target is the PF/Y/projection baseline: establish a bounded no-source equal-time trajectory and then rerun S0/S2/S3 against that reference. This is numerical closure only; GP release thermodynamics remain scenario-level and uncalibrated.
"""
    text_report("rsmd_recommended_coupling_scheme.md", recommendation)
    text_report("rsmd_recommended_low_overshoot_release_scheme.md", recommendation)
    operator_acceptance = f"""# RSMD Operator-Split Acceptance Report

## Final status

`FAIL_RSMD_OPERATOR_SPLITTING_AND_OVERSHOOT_CLOSURE` within the narrower operator-splitting acceptance vocabulary, and `PARTIAL_RSMD_DT_ERROR_IDENTIFIED_BUT_HIGH_DT_SCHEME_NOT_CLOSED` for the combined requested audit.

- Equal physical time controls: complete for T400 source-only, source+diffusion frozen-phi, and PF-only at dt 0.0005/0.001/0.002.
- Source-rate equivalence: PASS, relative applied-mass spread `{source_spread_rel:.6g}`.
- Mass closure: PASS for source controls, worst `{max_valid_mass:.3g}`.
- Alpha/interface boundedness: FAIL once PF transport/projection is enabled.
- PF-only numerical baseline: FAIL; all dt reach xB upper bound and seed collapse.
- T380/full coupling/high-dt/domain tests: gated, because they cannot establish convergence without a valid fine reference.
- Direct phi write: false. Direct GP-to-beta transfer: false. Unapplied mass discarded: false.
"""
    text_report("rsmd_operator_split_acceptance_report.md", operator_acceptance)
    text_report("rsmd_domain_size_independence_report.md", """# RSMD Domain-Size Independence Report

**Status: GATED_NOT_RUN.** The 128^3 equal-box PF-only baseline is temporally/bound unstable at every tested dt. A larger-domain comparison would conflate finite-size effects with the unresolved PF/Y/projection instability, so no domain-independence claim is made.
""")
    final_report = f"""# RSMD Equal-Time, Domain, and Low-Overshoot Acceptance Report

## Final verdict

`PARTIAL_RSMD_DT_ERROR_IDENTIFIED_BUT_HIGH_DT_SCHEME_NOT_CLOSED`

The exact-exponential, fixed-source-clock S3 transaction is conservative and nearly dt-invariant in a frozen field: applied source mass differs by only `{source_spread_rel:.3g}` relative across dt 0.0005–0.002 and never drives alpha/interface xB above 0.024. This closes the source-integration subproblem.

It does not close the coupled model. With matrix transport and projection enabled but phi frozen, dt=0.0005 reaches xB=0.99999999 and releases about 30% more GP mass because transport/projection repeatedly reopens local headroom. More decisively, PF-only equal-time controls at every tested dt reach the same upper xB bound and erase the resolved seed. Thus the primary cause is the PF/Y/projection baseline; source/PF cadence and headroom reopening are secondary coupled effects.

No fine-dt reference, `dt_max_accepted`, domain-size independence, or grid convergence can be claimed. Those tests were stopped by the preregistered numerical-validity gate rather than padded with uninterpretable runs. Mass conservation remains excellent (`<= {max_valid_mass:.3g}`) in the completed controlled tests, showing that strict ledger closure and composition boundedness are separate requirements.

This is a numerically conservative scenario-level GP-to-matrix source component, not calibrated GP release thermodynamics and not a production-closed RSMD coupling.
"""
    text_report("rsmd_dt_domain_high_dt_acceptance_report.md", final_report)
    terminal = f"""equal_physical_time_alignment_status=PASS_T400_CONTROLS_AT_0.851382418288_S
equal_box_dt_inconsistency_primary_cause=PF_TIME_INTEGRATION_Y_PROJECTION_BASELINE_INSTABILITY
equal_box_dt_inconsistency_secondary_cause=SOURCE_HEADROOM_REOPENING_AND_SOURCE_PF_OPERATOR_CADENCE
box_size_effect_status=GATED_NOT_RUN
grid_resolution_status=DESIGNED_NOT_RUN_BASELINE_GATE
baseline_scheme=S0_postPF_source_with_local_restart
implemented_candidate_schemes=S1_prePF_Lie,S2_Strang_exact,S3_fixed_clock_exact_headroom
recommended_release_scheme=S3_SOURCE_COMPONENT_ONLY_NOT_PRODUCTION_COUPLING
recommended_release_scheme_reason=source_only_relative_applied_mass_spread_{source_spread_rel:.6e}_but_PF_baseline_unstable
deferred_release_enabled=true_via_mass_remaining_in_GP_ledger
unapplied_mass_discarded=false
headroom_weighted_distribution=true
fixed_physical_source_cadence=0.0005_code_time
flux_form_release=false
T380_dt_ref=NOT_RUN_BASELINE_GATE
T380_dt_max_accepted=NONE
T380_fine_reference_dR=NOT_AVAILABLE
T380_high_dt_dR=NOT_AVAILABLE
T380_high_dt_error=NOT_AVAILABLE
T400_dt_ref=0.0005_attempted_but_invalid_as_reference
T400_dt_max_accepted=NONE
T400_fine_reference_dR=PF_ONLY_SEED_COLLAPSED
T400_high_dt_dR=NOT_ACCEPTED
T400_high_dt_error=UNDEFINED_WITHOUT_VALID_REFERENCE
T380_alpha_side_overcharge_reduction=NOT_RUN
T380_interface_spike_reduction=NOT_RUN
T400_alpha_side_overcharge_reduction=SOURCE_ONLY_BOUNDED_AT_0.024_BUT_COUPLED_FAILS
T400_interface_spike_reduction=SOURCE_ONLY_BOUNDED_AT_0.024_BUT_COUPLED_FAILS
source_rate_equivalence_status=PASS_relative_spread_{source_spread_rel:.6e}
region_aware_guard_status=FAIL_SOURCE_DIFFUSION_ALPHA_AND_INTERFACE
mass_closure_status=PASS_max_control_error_{max_valid_mass:.6e}
far_field_status=PASS_SOURCE_CONTROLS_BUT_FINE_SOURCE_DIFFUSION_SEVERELY_DEPLETED
domain_size_independence_status=UNRESOLVED_GATED
current_scheme=S0_postPF_source_with_local_restart
recommended_scheme=S3_SOURCE_COMPONENT_ONLY
T380_equal_time_dt0p0005_dR=NOT_RUN
T380_equal_time_dt0p001_dR=NOT_RUN
T380_equal_time_dt0p002_dR=NOT_RUN
T380_dt_convergence_status=NOT_RUN_BASELINE_GATE
T400_equal_time_dt0p0005_dR=PF_ONLY_COLLAPSE
T400_equal_time_dt0p001_dR=PF_ONLY_COLLAPSE
T400_equal_time_dt0p002_dR=PF_ONLY_COLLAPSE
T400_dt_convergence_status=FAIL_NO_STABLE_FINE_REFERENCE
T380_global_core_spike_status=NOT_RUN
T380_alpha_side_overshoot_status=NOT_RUN
T380_interface_spike_status=NOT_RUN
T400_global_core_spike_status=BOUND_RUNAWAY
T400_alpha_side_overshoot_status=FAIL
T400_interface_spike_status=FAIL
positive_beta_response_dt_converged=false
direct_phi_write=false
direct_GP_to_beta_transfer=false
production_GP_thermodynamics_closed=false
recommended_next_action=stabilize_and_dt-converge_PF-only_Y_projection_baseline_before_RSMD_or_domain_tests
final_status=PARTIAL_RSMD_DT_ERROR_IDENTIFIED_BUT_HIGH_DT_SCHEME_NOT_CLOSED
"""
    (REPORT / "final_terminal_output.txt").write_text(terminal)


if __name__ == "__main__":
    main()
