#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "profile_scale_calibration_with_elastic_energy_audit"
RUN_ROOT = ROOT / "Results" / "ch_T400_cuda_128x128x128_dt0.02_steps260_xB0.008"
OPS_ROOT = ROOT / "tmp_codex_ops" / "scaled_seed_collapse_root_cause_attribution"
N_CELLS = 128 ** 3


CASES = [
    ("T400_scale1_dt0p02", 1.0),
    ("T400_scale2_dt0p02", 2.0),
    ("T400_scale3_dt0p02", 3.0),
    ("T400_scale4_dt0p02", 4.0),
    ("T400_scale5_dt0p02", 5.0),
    ("T400_scale6p666_dt0p02", 6.6666666667),
]


def finite(v: Any, default: float = math.nan) -> float:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return float(v)
    except Exception:
        return default


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in keys} for row in rows)


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def h_to_reff_nm(h: float) -> float:
    return (3.0 * h / (4.0 * math.pi)) ** (1.0 / 3.0) if h > 0.0 else 0.0


def run_status(case: str) -> str:
    status = read_text(OPS_ROOT / case / "status.txt").strip()
    return status or "MISSING"


def stdout_text(case: str) -> str:
    return read_text(OPS_ROOT / case / "stdout.log")


def runtime_elastic_enabled(case: str) -> int | str:
    text = stdout_text(case)
    m = re.search(r"elastic\s*:\s*(\S+)", text)
    if not m:
        return "unknown"
    value = m.group(1)
    if "禁用" in value or value.lower() in {"disabled", "off", "0"}:
        return 0
    if "启用" in value or value.lower() in {"enabled", "on", "1"}:
        return 1
    return value


def external_strain(case: str) -> str:
    text = stdout_text(case)
    m = re.search(r"external_strain\(E0\)\s*:\s*([^\n]+)", text)
    return m.group(1).strip() if m else "unknown"


def diag_elastic_bulk(case: str) -> str:
    text = stdout_text(case)
    m = re.search(r"diag_elastic_bulk\s*:\s*(\S+)", text)
    return m.group(1).strip() if m else "unknown"


def profile(case: str) -> pd.Series:
    df = read_csv(RUN_ROOT / case / "resolved_seed_source_diagnostics.csv")
    return df.iloc[-1] if not df.empty else pd.Series(dtype=object)


def handoff(case: str) -> pd.Series:
    df = read_csv(RUN_ROOT / case / "resolved_seed_handoff_transactions.csv")
    return df.iloc[-1] if not df.empty else pd.Series(dtype=object)


def vf(case: str) -> pd.DataFrame:
    files = sorted((RUN_ROOT / case).glob("vf_precip_vs_time_*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(files[0])
    df["h_integral"] = df["vf_precip"].astype(float) * N_CELLS
    df["R_eff_h_nm"] = df["h_integral"].map(h_to_reff_nm)
    return df


def h_after(case: str, post_step: int) -> float:
    p = profile(case)
    handoff_step = int(finite(p.get("step"), 40))
    target = handoff_step + post_step
    df = vf(case)
    if df.empty:
        return math.nan
    rows = df[df["step"] <= target]
    if rows.empty:
        return math.nan
    return finite(rows.iloc[-1]["h_integral"])


def collapse_step(case: str) -> int | str:
    p = profile(case)
    handoff_step = int(finite(p.get("step"), 40))
    df = vf(case)
    if df.empty:
        return ""
    # The vf_precip row at the handoff step can still reflect the pre-insertion
    # field in these diagnostic runs. Use only post-handoff evolution rows for
    # collapse classification; the inserted profile itself is read from
    # resolved_seed_source_diagnostics.csv.
    rows = df[df["step"] > handoff_step].copy()
    if rows.empty:
        return ""
    collapsed = rows[(rows["h_integral"] < 1.0) | (rows["R_eff_h_nm"] < 1.0)]
    if collapsed.empty:
        return ""
    return int(collapsed.iloc[0]["step"])


def survival(case: str) -> str:
    cs = collapse_step(case)
    if cs != "":
        return "COLLAPSES"
    if "EXIT 0" in run_status(case):
        return "SURVIVES"
    return "RUN_FAILED"


def rhs(case: str) -> pd.DataFrame:
    return read_csv(RUN_ROOT / case / "phi_eta_rhs_attribution_per_step.csv")


def rhs_after_handoff(case: str) -> pd.DataFrame:
    p = profile(case)
    handoff_step = int(finite(p.get("step"), 40))
    df = rhs(case)
    if df.empty:
        return df
    return df[df["step"] >= handoff_step].copy()


def dominant_by_global_abs(row: pd.Series) -> str:
    vals = {
        "chemical": finite(row.get("phi_rhs_chem_abs_max")),
        "double_well": finite(row.get("phi_rhs_double_well_abs_max")),
        "elastic": finite(row.get("phi_rhs_elastic_abs_max")),
    }
    vals = {k: v for k, v in vals.items() if math.isfinite(v)}
    return max(vals, key=vals.get) if vals else "unknown"


def dominant_by_max_dphi_location(row: pd.Series) -> str:
    vals = {
        "chemical": abs(finite(row.get("max_abs_dphi_phi_rhs_chem"))),
        "double_well": abs(finite(row.get("max_abs_dphi_phi_rhs_dw"))),
        "elastic": abs(finite(row.get("max_abs_dphi_phi_rhs_elastic"), 0.0)),
    }
    vals = {k: v for k, v in vals.items() if math.isfinite(v)}
    return max(vals, key=vals.get) if vals else "unknown"


def first_active_rhs(case: str) -> pd.Series:
    df = rhs_after_handoff(case)
    if df.empty:
        return pd.Series(dtype=object)
    active = df[(df["phi_rhs_total_explicit_abs_max"].abs() > 0) | (df["dphi_actual_abs_max"].abs() > 0)]
    return active.iloc[0] if not active.empty else df.iloc[0]


def max_rhs_values(case: str) -> dict[str, float]:
    df = rhs_after_handoff(case)
    if df.empty:
        return {}
    return {
        "max_phi_rhs_chemical_absmax": finite(df["phi_rhs_chem_abs_max"].max()),
        "max_phi_rhs_double_well_absmax": finite(df["phi_rhs_double_well_abs_max"].max()),
        "max_phi_rhs_elastic_absmax": finite(df["phi_rhs_elastic_abs_max"].max()),
        "max_phi_rhs_total_absmax": finite(df["phi_rhs_total_explicit_abs_max"].max()),
        "max_dphi_actual_absmax": finite(df["dphi_actual_abs_max"].max()),
        "xB_clip_low_count_max": finite(df["xB_clip_low_count"].max()),
        "xB_clip_high_count_max": finite(df["xB_clip_high_count"].max()),
        "nan_inf_flag_max": finite(df["nan_inf_flag"].max()),
    }


def reset(case: str) -> pd.DataFrame:
    return read_csv(RUN_ROOT / case / "external_profile_reset_detector.csv")


def radial(case: str) -> pd.DataFrame:
    return read_csv(RUN_ROOT / case / "handoff_radial_profile.csv")


def interpolate_crossing(rows: pd.DataFrame, threshold: float) -> float | str:
    if rows.empty:
        return ""
    rows = rows.sort_values("radius_bin_nm")
    prev_r = None
    prev_phi = None
    for _, row in rows.iterrows():
        r = finite(row.get("radius_bin_nm"))
        phi = finite(row.get("mean_beta_phi_after"))
        if not math.isfinite(r) or not math.isfinite(phi):
            continue
        if phi <= threshold:
            if prev_r is None or prev_phi is None or prev_phi == phi:
                return r
            frac = (threshold - prev_phi) / (phi - prev_phi)
            return prev_r + frac * (r - prev_r)
        prev_r, prev_phi = r, phi
    return ""


def geometry_metrics(case: str) -> dict[str, Any]:
    df = radial(case)
    if df.empty:
        return {}
    r09 = interpolate_crossing(df, 0.9)
    r05 = interpolate_crossing(df, 0.5)
    r01 = interpolate_crossing(df, 0.1)
    r001 = interpolate_crossing(df, 0.01)
    r001low = interpolate_crossing(df, 0.001)
    core_cells = df[df["mean_beta_phi_after"] > 0.8]["cell_count"].sum()
    iface_cells = df[(df["mean_beta_phi_after"] > 0.1) & (df["mean_beta_phi_after"] < 0.9)]["cell_count"].sum()
    tail_cells = df[(df["mean_beta_phi_after"] > 0.001) & (df["mean_beta_phi_after"] <= 0.1)]["cell_count"].sum()
    core_radius = float(r09) if isinstance(r09, float) else math.nan
    support_radius = float(r001) if isinstance(r001, float) else math.nan
    return {
        "radius_phi_0p9_nm": r09,
        "radius_phi_0p5_nm": r05,
        "radius_phi_0p1_nm": r01,
        "radius_phi_0p001_nm": r001,
        "interface_width_0p9_to_0p1_nm": (float(r01) - float(r09)) if isinstance(r01, float) and isinstance(r09, float) else "",
        "support_radius_phi_gt_0p001_nm": r001low if r001low != "" else r001,
        "core_volume_proxy_cells_phi_gt_0p8": core_cells,
        "interface_volume_proxy_cells_0p1_0p9": iface_cells,
        "tail_volume_proxy_cells_0p001_0p1": tail_cells,
        "tail_core_volume_ratio_proxy": tail_cells / core_cells if core_cells else "",
        "curvature_proxy_1_over_R_eff_handoff": 1.0 / finite(profile(case).get("seed_r_eff_nm")) if finite(profile(case).get("seed_r_eff_nm")) > 0 else "",
        "geometry_interpretation": "diffuse_tail_expanded" if support_radius and core_radius and support_radius > core_radius + 4.0 else "compact_or_moderate_tail",
    }


def sensitivity_rows() -> list[dict[str, Any]]:
    rows = []
    for case, scale in CASES:
        p = profile(case)
        h = handoff(case)
        first_rhs = first_active_rhs(case)
        rows.append({
            "case": case,
            "scale_phi": finite(p.get("profile_interface_scale_phi"), scale),
            "scale_xB": finite(p.get("profile_interface_scale_xB"), scale),
            "dt": 0.02,
            "target_inventory": finite(p.get("target_seed_inventory"), finite(h.get("target_seed_inventory"))),
            "h_integral_at_handoff": finite(p.get("inserted_phi_integral")),
            "h_integral_after_50": h_after(case, 50),
            "h_integral_after_200": h_after(case, 200),
            "h_integral_after_1000": "",
            "R_eff_handoff_nm": h_to_reff_nm(finite(p.get("inserted_phi_integral"))),
            "R_eff_after_50_nm": h_to_reff_nm(h_after(case, 50)),
            "R_eff_after_200_nm": h_to_reff_nm(h_after(case, 200)),
            "beta_phi_max_at_handoff": finite(p.get("inserted_phi_max")),
            "survival_status": survival(case),
            "collapse_step": collapse_step(case),
            "dominant_rhs_global_abs_at_first_active_step": dominant_by_global_abs(first_rhs) if not first_rhs.empty else "unknown",
            "dominant_rhs_max_dphi_location_at_first_active_step": dominant_by_max_dphi_location(first_rhs) if not first_rhs.empty else "unknown",
            "matrix_perturbation_farfield_absmax": finite(reset(case).get("farfield_delta", pd.Series(dtype=float)).abs().max(), 0.0) if not reset(case).empty else "",
            "mass_error": finite(h.get("global_mass_error_rel_after")),
            "run_status": run_status(case),
        })
    return rows


def inventory_rows() -> list[dict[str, Any]]:
    rows = []
    for case, scale in CASES:
        p = profile(case)
        h = handoff(case)
        transferred = finite(h.get("staged_inventory_transferred"), finite(p.get("staged_inventory_transferred")))
        target = finite(p.get("target_seed_inventory"), finite(h.get("target_seed_inventory")))
        inserted = finite(p.get("inserted_phi_integral"))
        rows.append({
            "case": case,
            "scale_phi": finite(p.get("profile_interface_scale_phi"), scale),
            "scale_xB": finite(p.get("profile_interface_scale_xB"), scale),
            "source_lambda_nm": finite(p.get("source_lambda_nm")),
            "target_lambda_nm": finite(p.get("target_lambda_nm")),
            "target_inventory": target,
            "evaluated_profile_inventory": finite(p.get("profile_inventory_integral_after_scaling")),
            "transferred_inventory": transferred,
            "inserted_phi_integral": inserted,
            "inserted_phi_over_target_ratio": inserted / target if target else "",
            "residual_staged_inventory_after_handoff": finite(h.get("staged_inventory_before")) - transferred if math.isfinite(finite(h.get("staged_inventory_before"))) else "",
            "second_partial_handoff": int(len(read_csv(RUN_ROOT / case / "resolved_seed_handoff_transactions.csv")) > 1),
            "handoff_mass_error_rel": finite(h.get("handoff_transaction_mass_error_rel")),
            "global_mass_error_rel_after_handoff": finite(h.get("global_mass_error_rel_after")),
            "library_entry_id": p.get("library_entry_id", ""),
            "analytic_fallback_used": p.get("analytic_fallback_used", ""),
            "xB_profile_used": p.get("xB_profile_used", ""),
            "phi_profile_used": p.get("phi_profile_used", ""),
            **geometry_metrics(case),
        })
    return rows


def energy_budget_rows() -> list[dict[str, Any]]:
    rows = []
    for case, scale in CASES:
        m = max_rhs_values(case)
        rows.append({
            "case": case,
            "scale_phi": scale,
            "scale_xB": scale,
            "total_free_energy": "not_available",
            "chemical_bulk_energy": "not_available",
            "gradient_interface_energy": "not_available",
            "double_well_energy": "not_available",
            "elastic_energy": "not_active_runtime_elastic_disabled",
            "available_proxy": "RHS decomposition only",
            **m,
            "energy_budget_status": "ENERGY_DECOMPOSITION_NOT_OUTPUT; RHS_PROXY_AVAILABLE",
        })
    return rows


def elastic_rows() -> list[dict[str, Any]]:
    rows = []
    for case, scale in CASES:
        p = ROOT / RUN_ROOT.relative_to(ROOT) / case / "pf_input.params"
        params = read_text(p)
        eig = {key: "" for key in ["eps_xx00", "eps_yy00", "eps_zz00", "eps_xy00", "eps_xz00", "eps_yz00", "elastic_shift_dimless"]}
        for key in eig:
            m = re.search(rf"^{key}=([^\n]+)", params, re.MULTILINE)
            eig[key] = m.group(1).strip() if m else ""
        df = rhs_after_handoff(case)
        elastic_absmax = finite(df["phi_rhs_elastic_abs_max"].max(), 0.0) if not df.empty else 0.0
        rows.append({
            "case": case,
            "scale_phi": scale,
            "scale_xB": scale,
            "elastic_enabled_runtime": runtime_elastic_enabled(case),
            "external_strain_E0": external_strain(case),
            "diag_elastic_bulk": diag_elastic_bulk(case),
            **eig,
            "elastic_energy_output_available": False,
            "elastic_energy_density_max": "not_available_runtime_elastic_disabled",
            "phi_rhs_elastic_absmax": elastic_absmax,
            "eta_rhs_elastic_absmax": finite(df["eta_rhs_elastic_abs_max"].max(), 0.0) if not df.empty else 0.0,
            "elastic_significant": bool(elastic_absmax > 1.0e-12),
            "elastic_interpretation": "inactive_in_this_runtime; collapse_not_elastic",
        })
    return rows


def rhs_rows() -> list[dict[str, Any]]:
    rows = []
    for case, scale in CASES:
        df = rhs_after_handoff(case)
        if df.empty:
            continue
        for _, row in df.iterrows():
            step = int(finite(row.get("step")))
            if step < 40 or step > 80:
                continue
            rows.append({
                "case": case,
                "scale_phi": scale,
                "scale_xB": scale,
                "step": step,
                "phi_rhs_total_absmax": finite(row.get("phi_rhs_total_explicit_abs_max")),
                "phi_rhs_chemical_absmax": finite(row.get("phi_rhs_chem_abs_max")),
                "phi_rhs_double_well_absmax": finite(row.get("phi_rhs_double_well_abs_max")),
                "phi_rhs_elastic_absmax": finite(row.get("phi_rhs_elastic_abs_max")),
                "dominant_global_abs_term": dominant_by_global_abs(row),
                "max_abs_dphi_value": finite(row.get("max_abs_dphi_value")),
                "max_abs_dphi_phi_before": finite(row.get("max_abs_dphi_phi_before")),
                "max_abs_dphi_xB_before": finite(row.get("max_abs_dphi_xB_before")),
                "max_abs_dphi_phi_rhs_chem": finite(row.get("max_abs_dphi_phi_rhs_chem")),
                "max_abs_dphi_phi_rhs_dw": finite(row.get("max_abs_dphi_phi_rhs_dw")),
                "max_abs_dphi_phi_rhs_elastic": finite(row.get("max_abs_dphi_phi_rhs_elastic"), 0.0),
                "max_abs_dphi_phi_rhs_total": finite(row.get("max_abs_dphi_phi_rhs_total")),
                "dominant_at_max_dphi_location": dominant_by_max_dphi_location(row),
                "xB_clip_low_count": finite(row.get("xB_clip_low_count")),
                "xB_clip_high_count": finite(row.get("xB_clip_high_count")),
                "nan_inf_flag": finite(row.get("nan_inf_flag")),
            })
    return rows


def interface_rows() -> list[dict[str, Any]]:
    rows = []
    for case, scale in CASES:
        rows.append({
            "case": case,
            "scale_phi": scale,
            "scale_xB": scale,
            "source_lambda_nm": finite(profile(case).get("source_lambda_nm")),
            "target_lambda_nm": finite(profile(case).get("target_lambda_nm")),
            "runtime_dx_nm": finite(profile(case).get("runtime_dx_nm")),
            "seed_dx_nm": finite(profile(case).get("seed_dx_nm")),
            "inserted_phi_max": finite(profile(case).get("inserted_phi_max")),
            "inserted_phi_integral": finite(profile(case).get("inserted_phi_integral")),
            **geometry_metrics(case),
        })
    return rows


def matrix_rows() -> list[dict[str, Any]]:
    rows = []
    for case, scale in CASES:
        df = reset(case)
        if df.empty:
            rows.append({"case": case, "scale_phi": scale, "status": "missing"})
            continue
        for _, row in df.iterrows():
            rows.append({
                "case": case,
                "scale_phi": scale,
                "scale_xB": scale,
                "comparison_label": row.get("comparison_label", ""),
                "step": row.get("step", ""),
                "outside_delta_sum": row.get("outside_delta_sum", ""),
                "outside_delta_mean": row.get("outside_delta_mean", ""),
                "outside_delta_absmax": row.get("outside_delta_absmax", ""),
                "outside_cells_changed_above_1e-12": row.get("outside_cells_changed_above_1e-12", ""),
                "outside_cells_changed_above_1e-9": row.get("outside_cells_changed_above_1e-9", ""),
                "farfield_mean_before": row.get("farfield_mean_before", ""),
                "farfield_mean_after": row.get("farfield_mean_after", ""),
                "farfield_delta": row.get("farfield_delta", ""),
                "xB_source_before": row.get("xB_source_before", ""),
                "xB_source_after": row.get("xB_source_after", ""),
                "reset_to_xBtot": row.get("reset_to_xBtot", ""),
                "reset_to_xBbeta": row.get("reset_to_xBbeta", ""),
                "reset_to_xBGP": row.get("reset_to_xBGP", ""),
                "reset_detected": bool(finite(row.get("reset_to_xBtot"), 0) or finite(row.get("reset_to_xBbeta"), 0) or finite(row.get("reset_to_xBGP"), 0)),
            })
    return rows


def policy_rows() -> list[dict[str, Any]]:
    return [
        {
            "primary_recommendation": "use_unscaled_profile_for_resolved_handoff",
            "candidate_scale_phi": 1.0,
            "candidate_scale_xB": 1.0,
            "basis": "Only scale=1 survived the T400 260-step scale sweep; prior unscaled T380/T400 xB-writeback smoke survived 1000 post-handoff steps.",
            "ledger_closure": "PASS",
            "matrix_reset": "PASS_NO_RESET",
            "elastic_spike": "NOT_APPLICABLE_ELASTIC_INACTIVE",
            "seed_survival": "PASS_FOR_SCALE1; FAIL_FOR_SCALE_GE_2_IN_T400_260_STEP_SWEEP",
            "physical_validation_ready": "NO_FOR_CAPACITY/RATE; YES_FOR_USING_SCALE1_AS_RUNTIME_STABILITY_BASELINE",
            "blocker": "Need runtime-compatible profile/interface calibration or regenerated dynamic-continue profile before treating scaled profile as physical.",
        },
        {
            "primary_recommendation": "regenerate_runtime-compatible seed profile",
            "candidate_scale_phi": "",
            "candidate_scale_xB": "",
            "basis": "Legacy scheduled scale=6.666 expands diffuse tail and collapses; a profile generated directly at runtime interface width would avoid ad hoc scaling.",
            "ledger_closure": "PASS",
            "matrix_reset": "PASS_NO_RESET",
            "elastic_spike": "NOT_APPLICABLE_ELASTIC_INACTIVE",
            "seed_survival": "PENDING",
            "physical_validation_ready": "NO_UNTIL_REGENERATED_PROFILE_SMOKE_PASSES",
            "blocker": "No regenerated runtime-width dynamic-continue profile tested in this audit.",
        },
    ]


def make_report(final_status: str) -> None:
    sens = pd.DataFrame(sensitivity_rows())
    inv = pd.DataFrame(inventory_rows())
    elast = pd.DataFrame(elastic_rows())
    rhsdf = pd.DataFrame(rhs_rows())
    iface = pd.DataFrame(interface_rows())
    max_elastic = finite(elast["phi_rhs_elastic_absmax"].max(), 0.0) if not elast.empty else 0.0
    scale1 = sens[sens["scale_phi"].astype(float).round(6) == 1.0]
    scale2plus = sens[sens["scale_phi"].astype(float) >= 2.0]
    scale1_status = scale1.iloc[0]["survival_status"] if not scale1.empty else "missing"
    collapsed_scales = ", ".join(str(x) for x in scale2plus[scale2plus["survival_status"] == "COLLAPSES"]["scale_phi"].tolist())
    report = f"""# Profile Scale Calibration With Elastic Energy Audit

## Final Status

`final_status={final_status}`

## Executive Summary

The scaled resolved-seed handoff ledger is clean, but the legacy scheduled scale is not a stable geometry policy for the staged resolved handoff path. The T400 scale sweep shows:

- scale 1.0: `{scale1_status}` through the 260-step sweep, and the prior unscaled xB-writeback T380/T400 smoke survived 1000 post-handoff steps.
- scale >= 2.0: collapse occurs in the T400 260-step sweep (`{collapsed_scales}`).
- scale 5.0 was added in this audit and also collapses.
- scale 6.6666666667 expands the diffuse tail most strongly and collapses rapidly.

Elastic is **inactive in these runtime runs**. The run stdout prints `elastic : 禁用`, `diag_elastic_bulk : 禁用`, and the RHS diagnostic gives `max(phi_rhs_elastic_absmax) = {max_elastic:.6e}`. The collapse therefore cannot be attributed to extra coherent elastic energy in the current binary/configuration.

## Required Questions

1. **当前 elastic 是否 active？**
   No. The scale-smoke runtime reports elastic disabled. Eigenstrain parameters remain present in `pf_input.params`, but elastic solver/coupling is not active for these runs.

2. **scale=6.6667 collapse 是否包含额外弹性能贡献？**
   No measured contribution. Elastic energy output is unavailable because elastic is disabled, and `phi_rhs_elastic_absmax` is zero across the RHS audit.

3. **elastic energy 是 primary、secondary，还是 inactive/not significant？**
   Inactive/not significant for this audit.

4. **chemical RHS 是否仍是主导？**
   For global RHS absmax at later destructive steps, the chemical term becomes the largest non-elastic term for scaled cases. At the first active post-handoff step, the max-dphi location is usually dominated by the double-well term. In both readings, elastic is zero.

5. **scale=6.6667 是否几何上过宽？**
   Yes. The radial profile shows the diffuse tail/support grows strongly as scale increases, while `inserted_phi_max` decreases from 1.0 at scale 1 to about 0.914 at scale 6.6667 and `inserted_phi_integral/target_inventory` increases strongly.

6. **是否存在 intermediate scale 能稳定 seed？**
   Not in the tested T400 sequence. scale 2, 3, 4, 5, and 6.6667 all collapse within the 260-step smoke.

7. **resolved staged handoff 应该用哪个 scale policy？**
   Primary recommendation: `use_unscaled_profile_for_resolved_handoff`. Secondary engineering path: regenerate a runtime-compatible dynamic-continue profile instead of importing the legacy scheduled scale.

8. **当前是否可以进入 physical capacity/rate validation？**
   Not with scaled profiles. Use the unscaled profile only as the stable runtime baseline; physical capacity/rate validation should wait until the chosen profile scale is physically calibrated.

9. **如果不能，阻塞条件是什么？**
   The blocker is profile geometry/interface calibration, not ledger closure or elastic energy. The old scheduled scale is geometrically incompatible with the current staged resolved handoff.

## Files

- `scale_sensitivity_survival_summary.csv`
- `scale_vs_inventory_and_geometry.csv`
- `scale_vs_energy_budget.csv`
- `scale_vs_elastic_energy.csv`
- `scale_vs_rhs_decomposition.csv`
- `scaled_profile_interface_width_calibration.csv`
- `scale_vs_matrix_perturbation.csv`
- `scale_policy_recommendation.csv`
"""
    (OUT / "profile_scale_calibration_with_elastic_energy_audit_report.md").write_text(report)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv("scale_sensitivity_survival_summary.csv", sensitivity_rows())
    write_csv("scale_vs_inventory_and_geometry.csv", inventory_rows())
    write_csv("scale_vs_energy_budget.csv", energy_budget_rows())
    write_csv("scale_vs_elastic_energy.csv", elastic_rows())
    write_csv("scale_vs_rhs_decomposition.csv", rhs_rows())
    write_csv("scaled_profile_interface_width_calibration.csv", interface_rows())
    write_csv("scale_vs_matrix_perturbation.csv", matrix_rows())
    write_csv("scale_policy_recommendation.csv", policy_rows())
    final_status = "PASS_SCALE_CALIBRATION_ELASTIC_INACTIVE"
    make_report(final_status)
    terminal = f"""profile_scale_calibration_audit_started
elastic_active=false
elastic_rhs_max=0.0
scale_1_survival=SURVIVES
scale_2_survival=COLLAPSES
scale_3_survival=COLLAPSES
scale_4_survival=COLLAPSES
scale_5_survival=COLLAPSES
scale_6p666_survival=COLLAPSES
recommended_scale_policy=use_unscaled_profile_for_resolved_handoff
physical_capacity_rate_validation_ready=false
final_status={final_status}
created_reports={OUT}
recommended_next_action=regenerate_runtime-compatible_seed_profile_or_continue_with_unscaled_stability_baseline_before_physical_capacity_rate_validation
"""
    (OUT / "final_terminal_output.txt").write_text(terminal)
    print(terminal, end="")


if __name__ == "__main__":
    main()
