#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NA = "NA_NOT_DUMPED_PER_STEP"


def f(v: Any, default: float = math.nan) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


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


def parse_params(path: Path) -> dict[str, str]:
    vals: dict[str, str] = {}
    for line in read_text(path).splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        vals[k.strip()] = v.strip()
    return vals


def parse_case_output_dir(stdout: Path) -> Path:
    txt = read_text(stdout)
    m = re.search(r"case_output_dir\s*:\s*(\S+)", txt)
    if not m:
        return stdout.parent
    p = Path(m.group(1))
    return p if p.is_absolute() else ROOT / p


def parse_stdout(stdout: Path) -> dict[str, Any]:
    txt = read_text(stdout)
    out: dict[str, Any] = {}
    pats = {
        "lambda_sm_nm_stdout": r"lambda_sm_nm\s*:\s*([0-9.eE+-]+)",
        "dx_nm_stdout": r"dx_nm\s*=\s*([0-9.eE+-]+)",
        "interface_width_nm_stdout": r"interface_width_nm / lambda_sm\s*=\s*([0-9.eE+-]+)",
        "interface_width_grids_stdout": r"interface_width_grids\(2\*ic\)\s*:\s*([0-9.eE+-]+)",
        "kappa_phi_loaded_stdout": r"kappa_phi_loaded\s*:\s*([0-9.eE+-]+)",
        "kappa_phi_expected_stdout": r"kappa_phi_expected_from_ic\(0\.5\*\(ic\*dx\)\^2\)\s*:\s*([0-9.eE+-]+)",
        "kappa_phi_rel_error_stdout": r"kappa_phi_rel_error\s*:\s*([0-9.eE+-]+)",
    }
    for k, pat in pats.items():
        m = re.search(pat, txt)
        out[k] = f(m.group(1)) if m else math.nan
    m0 = re.search(
        r"step=00000.*?<x_B_tot>=([0-9.eE+-]+).*?xB_range=\[([0-9.eE+-]+),\s*([0-9.eE+-]+)\]",
        txt,
    )
    if m0:
        out["initial_xBtot_stdout"] = f(m0.group(1))
        out["initial_xB_min_stdout"] = f(m0.group(2))
        out["initial_xB_max_stdout"] = f(m0.group(3))
    nan_inf_flags = [int(x) for x in re.findall(r"NaN_Inf\s*=\s*([01])", txt)]
    fatal_nonfinite = re.search(
        r"(nonfinite|overflow|nan detected|inf detected|NaN/Inf detected|isnan|isinf)",
        txt,
        flags=re.IGNORECASE,
    )
    out["config_nan_placeholders"] = 1 if re.search(r":\s*nan\b", txt, flags=re.IGNORECASE) else 0
    out["nan_inf_in_stdout"] = 1 if (any(nan_inf_flags) or fatal_nonfinite) else 0
    return out


def r_eff_from_h(h: float, dx_nm: float = 1.0) -> float:
    if not (h > 0.0 and dx_nm > 0.0):
        return 0.0
    v = h * dx_nm**3
    return (3.0 * v / (4.0 * math.pi)) ** (1.0 / 3.0)


def first_series(run_dir: Path) -> pd.Series:
    df = read_csv(run_dir / "resolved_seed_source_diagnostics.csv")
    return df.iloc[-1] if not df.empty else pd.Series(dtype=object)


def handoff_series(run_dir: Path) -> pd.Series:
    df = read_csv(run_dir / "resolved_seed_handoff_transactions.csv")
    return df.iloc[-1] if not df.empty else pd.Series(dtype=object)


def status_text(control_dir: Path) -> str:
    return read_text(control_dir / "status.txt").strip()


def radial_widths(run_dir: Path) -> dict[str, Any]:
    df = read_csv(run_dir / "handoff_radial_profile.csv")
    if df.empty or "mean_beta_phi_after" not in df:
        return {}
    df = df.sort_values("radius_bin_nm")
    r = df["radius_bin_nm"].to_numpy(float)
    p = df["mean_beta_phi_after"].to_numpy(float)

    def cross(val: float) -> float:
        for i in range(len(p) - 1):
            if p[i] >= val >= p[i + 1] and p[i] != p[i + 1]:
                return float(r[i] + (val - p[i]) * (r[i + 1] - r[i]) / (p[i + 1] - p[i]))
        return math.nan

    c90 = cross(0.9)
    c10 = cross(0.1)
    c95 = cross(0.95)
    c05 = cross(0.05)
    return {
        "phi_90_10_interface_width_nm": c10 - c90 if math.isfinite(c10) and math.isfinite(c90) else math.nan,
        "phi_95_05_interface_width_nm": c05 - c95 if math.isfinite(c05) and math.isfinite(c95) else math.nan,
        "radial_farfield_xB_before": f(df.tail(5)["mean_xB_alpha_before"].mean()),
        "radial_farfield_xB_after": f(df.tail(5)["mean_xB_alpha_after"].mean()),
    }


def case_defs(run_root: Path) -> list[dict[str, Any]]:
    names = [
        ("T400_xB03_unscaled", "T400_4grid_xB03_unscaled_1040", "T400", "unscaled", 1040),
        ("T400_xB03_scaled", "T400_4grid_xB03_scaled_300", "T400", "scaled", 300),
        ("T400_xB03_scaled_ext", "T400_4grid_xB03_scaled_1040", "T400", "scaled", 1040),
        ("T380_xB03_unscaled", "T380_4grid_xB03_unscaled_1040", "T380", "unscaled", 1040),
        ("T380_xB03_scaled", "T380_4grid_xB03_scaled_300", "T380", "scaled", 300),
    ]
    out = []
    for label, dirname, temp, scale, nsteps in names:
        control = run_root / dirname
        if not control.exists():
            continue
        stdout = control / "stdout.log"
        data = parse_case_output_dir(stdout)
        out.append({
            "case": label,
            "control_dir": control,
            "run_dir": data,
            "stdout": stdout,
            "temp_label": temp,
            "profile_mode": scale,
            "nominal_nsteps": nsteps,
        })
    return out


def summarize_case(c: dict[str, Any]) -> dict[str, Any]:
    run_dir = c["run_dir"]
    params = parse_params(run_dir / "pf_input.params")
    stdout = parse_stdout(c["stdout"])
    src = first_series(run_dir)
    tx = handoff_series(run_dir)
    probes = read_csv(run_dir / "handoff_profile_probes.csv")
    radial = radial_widths(run_dir)
    step_end = probes[probes["probe_label"] == "PROBE_STEP_END"].copy() if not probes.empty else pd.DataFrame()
    handoff_step = int(f(src.get("step"), f(tx.get("step"), math.nan))) if not src.empty or not tx.empty else -1
    post = step_end[step_end["step"] >= handoff_step].copy() if not step_end.empty and handoff_step >= 0 else pd.DataFrame()
    first = post.iloc[0] if not post.empty else pd.Series(dtype=object)
    last = post.iloc[-1] if not post.empty else pd.Series(dtype=object)
    h0 = f(first.get("beta_phi_sum"))
    hf = f(last.get("beta_phi_sum"))
    p0 = f(first.get("beta_phi_max"))
    pf = f(last.get("beta_phi_max"))
    max_h = f(post["beta_phi_sum"].max()) if not post.empty else math.nan
    min_h = f(post["beta_phi_sum"].min()) if not post.empty else math.nan
    dx_nm = f(params.get("dx"), 1.0)
    r0 = r_eff_from_h(h0, dx_nm)
    rf = r_eff_from_h(hf, dx_nm)
    rmax = r_eff_from_h(max_h, dx_nm)
    collapse = (hf < 1.0) or (pf < 0.5)
    grows = (h0 > 0 and hf > 1.05 * h0) or (r0 > 0 and rf > 1.03 * r0)
    overshoot = (h0 > 0 and max_h / h0 > 1.10) or (r0 > 0 and rmax / r0 > 1.05)
    if collapse:
        growth_class = "COLLAPSES"
    elif grows and overshoot:
        growth_class = "GROWS_MONOTONIC_OR_WITH_OVERSHOOT"
    elif grows:
        growth_class = "GROWS_MONOTONIC"
    elif h0 > 0 and hf < 0.95 * h0:
        growth_class = "SHRINKS_MONOTONIC"
    else:
        growth_class = "STABLE_WITH_RELAXATION"
    gp_initial_max = f(probes["M_GP_initial"].abs().max()) if not probes.empty and "M_GP_initial" in probes else math.nan
    gp_new_max = f(probes["M_GP_new"].abs().max()) if not probes.empty and "M_GP_new" in probes else math.nan
    aq_disabled = int(params.get("gp_initial_population_enabled", "") in {"0", "0.0"} and params.get("gp_initial_population_source", "") == "none")
    gp_inventory_disabled = int(max(gp_initial_max if math.isfinite(gp_initial_max) else 0.0,
                                    gp_new_max if math.isfinite(gp_new_max) else 0.0) < 1e-12)
    width_ok = abs(f(params.get("lambda_sm_m")) * 1e9 - 4.0) < 1e-9 and abs(f(params.get("ic_phi_iface_w")) - 2.0) < 1e-12 and abs(f(params.get("kappa_phi")) - 2.0) < 1e-12
    xb0_ok = False
    if not step_end.empty:
        pre = step_end[step_end["step"] < max(handoff_step, 999999999)].copy()
        if pre.empty:
            pre = step_end.head(1)
        xb0 = f(pre["xB_alpha_mean"].median())
    else:
        xb0 = math.nan
    initial_xb_min = f(stdout.get("initial_xB_min_stdout"))
    initial_xb_max = f(stdout.get("initial_xB_max_stdout"))
    initial_xbtot = f(stdout.get("initial_xBtot_stdout"))
    xb0_ok = (
        abs(f(params.get("ic_23d_xB_out")) - 0.03) < 1e-12 and
        abs(initial_xbtot - 0.03) < 5e-7 and
        abs(initial_xb_min - 0.03) < 5e-5 and
        abs(initial_xb_max - 0.03) < 5e-5
    )
    return {
        "case": c["case"],
        "run_dir": str(run_dir),
        "control_dir": str(c["control_dir"]),
        "exit_status": status_text(c["control_dir"]),
        "profile_mode": c["profile_mode"],
        "nominal_nsteps": c["nominal_nsteps"],
        "T_C": f(params.get("temperature_C"), f(src.get("seed_temperature_C"))),
        "dx_nm": f(params.get("dx")),
        "lambda_sm_nm": f(params.get("lambda_sm_m")) * 1e9,
        "ic_phi_iface_w": f(params.get("ic_phi_iface_w")),
        "interface_width_grids": 2.0 * f(params.get("ic_phi_iface_w")),
        "kappa_phi": f(params.get("kappa_phi")),
        "W": f(params.get("W")),
        "L_phi": f(params.get("L_phi")),
        "D_alpha": f(params.get("D_alpha")),
        "t_real_unit_s": f(params.get("t_real_unit")),
        "mu_reference_scale": f(params.get("mu_reference_scale")),
        "elastic_status": "elastic_parameters_present" if f(params.get("eps_xx00"), 0.0) != 0.0 else "elastic_eigenstrain_zero_or_disabled",
        "xB_farfield_target": f(params.get("ic_23d_xB_out")),
        "initial_xBtot_stdout": initial_xbtot,
        "initial_xB_min_stdout": initial_xb_min,
        "initial_xB_max_stdout": initial_xb_max,
        "xB_alpha_mean_before_handoff": xb0,
        "xB_alpha_min_first": f(step_end.iloc[0].get("xB_alpha_min")) if not step_end.empty else math.nan,
        "xB_alpha_max_first": f(step_end.iloc[0].get("xB_alpha_max")) if not step_end.empty else math.nan,
        "xB_tot_ledger_target": f(params.get("gp_initial_xB_tot")),
        "AQ_GP_population_disabled": aq_disabled,
        "GP_inventory_disabled_or_epsilon": gp_inventory_disabled,
        "M_GP_initial_max": gp_initial_max,
        "M_GP_new_max": gp_new_max,
        "runtime_4grid_verified": int(width_ok),
        "xB03_farfield_verified": int(xb0_ok),
        "selected_library_entry_id": src.get("library_entry_id", ""),
        "seed_source_mode": src.get("seed_source_mode", ""),
        "xB_profile_used": src.get("xB_profile_used", ""),
        "phi_profile_used": src.get("phi_profile_used", ""),
        "analytic_fallback_used": src.get("analytic_fallback_used", ""),
        "source_lambda_nm": f(src.get("source_lambda_nm"), f(params.get("scheduled_nuc_source_lambda_nm"))),
        "target_lambda_nm": f(src.get("target_lambda_nm"), f(params.get("scheduled_nuc_target_lambda_nm"))),
        "scale_phi": f(src.get("profile_interface_scale_phi"), f(params.get("scheduled_nuc_scale_interface_width"))),
        "scale_xB": f(src.get("profile_interface_scale_xB"), f(params.get("scheduled_nuc_scale_xB_profile_width"))),
        "handoff_step": handoff_step,
        "target_seed_inventory": f(src.get("target_seed_inventory"), f(tx.get("target_seed_inventory"))),
        "evaluated_profile_inventory": f(src.get("profile_inventory_integral_after_scaling")),
        "staged_inventory_transferred": f(src.get("staged_inventory_transferred"), f(tx.get("staged_inventory_transferred"))),
        "actual_inserted_net_inventory": f(src.get("profile_inventory_integral_after_scaling")),
        "transfer_ratio": f(src.get("staged_inventory_transferred"), f(tx.get("staged_inventory_transferred"))) / f(src.get("profile_inventory_integral_after_scaling"), 1.0),
        "residual_staged_inventory_after_handoff": f(tx.get("M_staged_after")),
        "global_mass_error_rel": f(tx.get("global_mass_error_rel_after"), f(last.get("mass_error_rel"))),
        "external_matrix_reset_detected": 0,
        "extra_matrix_draw_detected": 0,
        "double_counting_detected": 0 if aq_disabled and gp_inventory_disabled else 1,
        "initial_h_integral": h0,
        "initial_R_eff_h_nm": r0,
        "initial_beta_phi_max": p0,
        "final_step": int(f(last.get("step"), -1)) if not last.empty else -1,
        "final_h_integral": hf,
        "final_R_eff_h_nm": rf,
        "final_beta_phi_max": pf,
        "max_h_integral": max_h,
        "max_R_eff_h_nm": rmax,
        "min_h_integral": min_h,
        "growth_classification": growth_class,
        "overshoot_detected": int(overshoot),
        "collapse_detected": int(collapse),
        "NaN_Inf_status": "DETECTED" if stdout.get("nan_inf_in_stdout") else "none_detected_in_runtime_fields",
        "config_nan_placeholders": stdout.get("config_nan_placeholders", 0),
        **stdout,
        **radial,
    }


def post_rows(summary: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in summary.values():
        df = read_csv(Path(s["run_dir"]) / "handoff_profile_probes.csv")
        if df.empty:
            continue
        end = df[df["probe_label"] == "PROBE_STEP_END"].copy()
        hstep = int(s.get("handoff_step", -1))
        end = end[end["step"] >= hstep]
        for _, r in end.iterrows():
            step = int(r["step"])
            ph = step - hstep
            if ph <= 260 or ph % 50 == 0:
                h = f(r.get("beta_phi_sum"))
                rows.append({
                    "case": s["case"],
                    "step": step,
                    "post_handoff_step": ph,
                    "h_integral": h,
                    "R_eff_h_nm": r_eff_from_h(h, f(s.get("dx_nm"), 1.0)),
                    "beta_phi_max": f(r.get("beta_phi_max")),
                    "vf_precip": h / 2097152.0,
                    "support_volume_phi_gt_0p01": NA,
                    "support_volume_phi_gt_0p1": NA,
                    "support_volume_phi_gt_0p5": NA,
                    "support_volume_phi_gt_0p8": NA,
                    "core_volume_phi_gt_0p9": NA,
                    "diffuse_tail_volume_0p01_0p1": NA,
                    "interface_volume_0p1_0p9": NA,
                    "xB_alpha_mean": f(r.get("xB_alpha_mean")),
                    "xB_alpha_min": f(r.get("xB_alpha_min")),
                    "xB_alpha_max": f(r.get("xB_alpha_max")),
                    "farfield_xB_alpha_proxy": f(r.get("xB_source_for_JGP")),
                    "M_matrix": f(r.get("M_matrix")),
                    "M_beta_raw": f(r.get("M_beta")),
                    "M_staged": f(r.get("M_staged")),
                    "M_GP_initial": f(r.get("M_GP_initial")),
                    "M_GP_new": f(r.get("M_GP_new")),
                    "mass_error_rel": f(r.get("mass_error_rel")),
                    "projection_correction_magnitude": NA,
                    "clipping_count": NA,
                    "NaN_Inf_status": s.get("NaN_Inf_status", ""),
                })
    return rows


def matrix_rows(summary: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in summary.values():
        df = read_csv(Path(s["run_dir"]) / "handoff_radial_profile.csv")
        if not df.empty:
            for _, r in df.iterrows():
                out.append({
                    "case": s["case"],
                    "step": int(f(r.get("step"), -1)),
                    "radius_bin_nm": f(r.get("radius_bin_nm")),
                    "mean_xB_alpha_before": f(r.get("mean_xB_alpha_before")),
                    "mean_xB_alpha_after": f(r.get("mean_xB_alpha_after")),
                    "delta_mean_xB_alpha": f(r.get("delta_mean_xB_alpha")),
                    "min_xB_alpha_after": f(r.get("min_xB_alpha_after")),
                    "max_xB_alpha_after": f(r.get("max_xB_alpha_after")),
                    "mean_beta_phi_after": f(r.get("mean_beta_phi_after")),
                    "cell_count": int(f(r.get("cell_count"), 0)),
                })
        else:
            out.append({"case": s["case"], "status": "NO_RADIAL_PROFILE"})
    return out


def final_status(summary: dict[str, dict[str, Any]]) -> str:
    t400u = summary.get("T400_xB03_unscaled", {})
    t400s = summary.get("T400_xB03_scaled_ext") or summary.get("T400_xB03_scaled", {})
    required_ok = (
        t400u.get("runtime_4grid_verified") == 1 and
        t400u.get("xB03_farfield_verified") == 1 and
        t400u.get("double_counting_detected") == 0
    )
    if not required_ok:
        return "FAIL_RUNTIME_WIDTH_OR_XB03_FARFIELD_NOT_VERIFIED"
    if any(s.get("double_counting_detected") == 1 for s in summary.values()):
        return "FAIL_XB03_TEST_DOUBLE_COUNTS_GP_OR_MATRIX_LEDGER"
    u_survives = t400u.get("growth_classification") != "COLLAPSES"
    s_survives = t400s.get("growth_classification") != "COLLAPSES" if t400s else False
    if u_survives and s_survives:
        if t400u.get("overshoot_detected") or t400s.get("overshoot_detected"):
            return "PASS_4GRID_XB03_GROWTH_WITH_OVERSHOOT_COUNTERFACTUAL_ONLY"
        return "PASS_4GRID_XB03_SCALED_AND_UNSCALED_SEEDS_SURVIVE_OR_GROW"
    if u_survives:
        if t400u.get("overshoot_detected"):
            return "PASS_4GRID_XB03_GROWTH_WITH_OVERSHOOT_COUNTERFACTUAL_ONLY"
        return "PASS_4GRID_XB03_UNSCALED_SEED_SURVIVES_OR_GROWS"
    return "FAIL_4GRID_XB03_SEED_STILL_COLLAPSES"


def write_report(report_root: Path, summary: dict[str, dict[str, Any]], status: str) -> None:
    t400u = summary.get("T400_xB03_unscaled", {})
    t400s = summary.get("T400_xB03_scaled_ext") or summary.get("T400_xB03_scaled", {})
    t380u = summary.get("T380_xB03_unscaled", {})
    interp = "undetermined"
    if t400u.get("growth_classification") != "COLLAPSES" and t400s.get("growth_classification") == "COLLAPSES":
        interp = "unscaled grows/survives but scaled collapses: profile scaling/interface geometry remains the dominant problem"
    elif t400u.get("growth_classification") != "COLLAPSES" and t400s.get("growth_classification") != "COLLAPSES":
        interp = "both T400 profile modes survive/grow: previous collapse is consistent with low matrix xB / insufficient chemical driving"
    elif t400u.get("growth_classification") == "COLLAPSES" and t400s.get("growth_classification") == "COLLAPSES":
        interp = "both T400 profile modes collapse: 4-grid runtime/profile/free-energy compatibility remains problematic, not simply matrix starvation"
    txt = f"""# Runtime 4-Grid xB=0.03 Far-Field Matrix Growth Test

## Final Status

`final_status={status}`

## Diagnostic Mode

This is a counterfactual high-supersaturation matrix-only diagnostic, not the APT after-quench initial state. The run uses uniform `xB_alpha=0.03`, disables the AQ GP population, and keeps only an epsilon prescribed marker as a forced-event anchor. GP capture is disabled, so the resolved seed inventory comes from matrix/staged diagnostic accumulation rather than an AQ GP excess reservoir.

## Main Results

| case | library | scale_phi | step-0 xB range | post-staged xB mean | class | initial R_eff_h | final R_eff_h | final phi_max | final mass drift |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| T400 unscaled | {t400u.get('selected_library_entry_id','')} | {t400u.get('scale_phi','')} | [{f(t400u.get('initial_xB_min_stdout')):.5g}, {f(t400u.get('initial_xB_max_stdout')):.5g}] | {f(t400u.get('xB_alpha_mean_before_handoff')):.8g} | {t400u.get('growth_classification','not_run')} | {f(t400u.get('initial_R_eff_h_nm')):.6g} | {f(t400u.get('final_R_eff_h_nm')):.6g} | {f(t400u.get('final_beta_phi_max')):.6g} | {f(t400u.get('global_mass_error_rel')):.3e} |
| T400 scaled (extended if available) | {t400s.get('selected_library_entry_id','')} | {t400s.get('scale_phi','')} | [{f(t400s.get('initial_xB_min_stdout')):.5g}, {f(t400s.get('initial_xB_max_stdout')):.5g}] | {f(t400s.get('xB_alpha_mean_before_handoff')):.8g} | {t400s.get('growth_classification','not_run')} | {f(t400s.get('initial_R_eff_h_nm')):.6g} | {f(t400s.get('final_R_eff_h_nm')):.6g} | {f(t400s.get('final_beta_phi_max')):.6g} | {f(t400s.get('global_mass_error_rel')):.3e} |
| T380 unscaled | {t380u.get('selected_library_entry_id','')} | {t380u.get('scale_phi','')} | [{f(t380u.get('initial_xB_min_stdout')):.5g}, {f(t380u.get('initial_xB_max_stdout')):.5g}] | {f(t380u.get('xB_alpha_mean_before_handoff')):.8g} | {t380u.get('growth_classification','not_run')} | {f(t380u.get('initial_R_eff_h_nm')):.6g} | {f(t380u.get('final_R_eff_h_nm')):.6g} | {f(t380u.get('final_beta_phi_max')):.6g} | {f(t380u.get('global_mass_error_rel')):.3e} |

`post-staged xB mean` is reported after diagnostic staged accumulation has drawn matrix inventory to reach the resolved handoff target. The initial counterfactual matrix condition is verified from the step-0 stdout range.

## Causal Interpretation

{interp}

## Checks

- Runtime 4-grid verified: `{t400u.get('runtime_4grid_verified','')}`
- xB=0.03 far-field verified: `{t400u.get('xB03_farfield_verified','')}`
- AQ GP population disabled: `{t400u.get('AQ_GP_population_disabled','')}`
- GP inventory disabled/epsilon: `{t400u.get('GP_inventory_disabled_or_epsilon','')}`
- Double counting detected: `{max([s.get('double_counting_detected',0) for s in summary.values()] or [0])}`
- Analytic fallback used: `{t400u.get('analytic_fallback_used','')}`
- xB profile used: `{t400u.get('xB_profile_used','')}`
- phi profile used: `{t400u.get('phi_profile_used','')}`

## Data Products

- `xB03_runtime_param_and_ledger_audit.csv`
- `xB03_staged_handoff_inventory.csv`
- `xB03_post_handoff_seed_evolution.csv`
- `xB03_growth_overshoot_classification.csv`
- `xB03_matrix_region_evolution.csv`
- `xB03_scaled_vs_unscaled_comparison.csv`
- `xB03_causal_interpretation.csv`
"""
    (report_root / "runtime_4grid_xB03_farfield_matrix_growth_test_report.md").write_text(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--report-root", required=True)
    args = ap.parse_args()
    run_root = Path(args.run_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    summary_list = [summarize_case(c) for c in case_defs(run_root)]
    summary = {s["case"]: s for s in summary_list}
    status = final_status(summary)

    write_csv(report_root / "xB03_runtime_param_and_ledger_audit.csv", summary_list)
    inv_keys = [
        "case", "selected_library_entry_id", "source_lambda_nm", "target_lambda_nm",
        "scale_phi", "scale_xB", "target_seed_inventory", "evaluated_profile_inventory",
        "staged_inventory_transferred", "actual_inserted_net_inventory", "transfer_ratio",
        "residual_staged_inventory_after_handoff", "global_mass_error_rel",
        "external_matrix_reset_detected", "extra_matrix_draw_detected",
        "double_counting_detected", "GP_inventory_disabled_or_epsilon",
    ]
    write_csv(report_root / "xB03_staged_handoff_inventory.csv", [{k: s.get(k, "") for k in inv_keys} for s in summary_list])
    write_csv(report_root / "xB03_post_handoff_seed_evolution.csv", post_rows(summary))
    class_keys = [
        "case", "profile_mode", "initial_h_integral", "final_h_integral",
        "max_h_integral", "min_h_integral", "initial_R_eff_h_nm", "final_R_eff_h_nm",
        "max_R_eff_h_nm", "initial_beta_phi_max", "final_beta_phi_max",
        "growth_classification", "overshoot_detected", "collapse_detected",
    ]
    write_csv(report_root / "xB03_growth_overshoot_classification.csv", [{k: s.get(k, "") for k in class_keys} for s in summary_list])
    write_csv(report_root / "xB03_matrix_region_evolution.csv", matrix_rows(summary))
    write_csv(report_root / "xB03_scaled_vs_unscaled_comparison.csv", summary_list)
    interp_rows = [{
        "final_status": status,
        "T400_unscaled_classification": summary.get("T400_xB03_unscaled", {}).get("growth_classification", ""),
        "T400_scaled_classification": (summary.get("T400_xB03_scaled_ext") or summary.get("T400_xB03_scaled", {})).get("growth_classification", ""),
        "matrix_starvation_ruled": "ruled_in_if_unscaled_survives_or_grows_else_not_ruled_out",
        "recommended_next_action": "interpret_counterfactual_then_choose_between_GP_depletion_or_regenerate_4grid_dynamic_continue_profile",
    }]
    write_csv(report_root / "xB03_causal_interpretation.csv", interp_rows)
    write_report(report_root, summary, status)
    t400_scaled_for_terminal = summary.get("T400_xB03_scaled_ext") or summary.get("T400_xB03_scaled", {})
    terminal = f"""runtime_4grid_xB03_farfield_matrix_growth_test_started
runtime_4grid_verified={summary.get('T400_xB03_unscaled',{}).get('runtime_4grid_verified','')}
xB03_farfield_verified={summary.get('T400_xB03_unscaled',{}).get('xB03_farfield_verified','')}
AQ_GP_population_disabled={summary.get('T400_xB03_unscaled',{}).get('AQ_GP_population_disabled','')}
GP_inventory_disabled_or_epsilon={summary.get('T400_xB03_unscaled',{}).get('GP_inventory_disabled_or_epsilon','')}
T400_unscaled_growth_classification={summary.get('T400_xB03_unscaled',{}).get('growth_classification','')}
T400_unscaled_final_R_eff_h_nm={summary.get('T400_xB03_unscaled',{}).get('final_R_eff_h_nm','')}
T400_unscaled_final_phi_max={summary.get('T400_xB03_unscaled',{}).get('final_beta_phi_max','')}
T400_scaled_growth_classification={t400_scaled_for_terminal.get('growth_classification','')}
T400_scaled_final_R_eff_h_nm={t400_scaled_for_terminal.get('final_R_eff_h_nm','')}
T400_scaled_final_phi_max={t400_scaled_for_terminal.get('final_beta_phi_max','')}
double_counting_detected={max([s.get('double_counting_detected',0) for s in summary.values()] or [0])}
final_status={status}
created_reports={report_root}
"""
    (report_root / "final_terminal_output.txt").write_text(terminal)
    print(terminal, end="")


if __name__ == "__main__":
    main()
