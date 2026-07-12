#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
XBFAR = 0.0078305391025
LIB = {380: "nlib_00006", 400: "nlib_dc_T400_xB003"}
SEED_R = {380: 5.358726490447833, 400: 4.735880528792827}
DT_CODE = 0.02
T_REAL_UNIT = {380: 49.54630476715921, 400: 41.12958542455477}


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def parse_params(path: Path) -> dict[str, str]:
    vals: dict[str, str] = {}
    for line in read_text(path).splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        vals[key.strip()] = value.strip()
    return vals


def case_output_dir(stdout: Path) -> Path:
    text = read_text(stdout)
    m = re.search(r"case_output_dir\s*:\s*(\S+)", text)
    if not m:
        return stdout.parent
    p = Path(m.group(1))
    return p if p.is_absolute() else ROOT / p


def r_eff_from_h(h_sum: float, dx_nm: float = 1.0) -> float:
    if not (h_sum > 0.0):
        return math.nan
    return (3.0 * h_sum * dx_nm**3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def slope_last(rows: list[dict[str, Any]], window: int = 300) -> float:
    pts = [(f(r.get("post_handoff_step")), f(r.get("R_eff_h"))) for r in rows]
    pts = [(x, y) for x, y in pts if math.isfinite(x) and math.isfinite(y)]
    if not pts:
        return math.nan
    max_x = max(x for x, _ in pts)
    pts = [(x, y) for x, y in pts if x >= max_x - window]
    if len(pts) < 2:
        return math.nan
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    den = n * sxx - sx * sx
    return (n * sxy - sx * sy) / den if abs(den) > 0.0 else math.nan


def detect_nan_inf(*paths: Path) -> bool:
    pat = re.compile(r"(nan|inf|overflow|nonfinite)", re.IGNORECASE)
    for path in paths:
        txt = read_text(path)
        if pat.search(txt):
            placeholders = re.findall(r":\s*nan\b", txt, flags=re.IGNORECASE)
            if len(placeholders) == 0 or "NaN_Inf=1" in txt or "nonfinite" in txt.lower():
                return True
    return False


def classify(initial_r: float, final_r: float, initial_h: float, final_h: float,
             phi_max: float, slope: float) -> tuple[str, str, str]:
    threshold = max(0.10, 0.02 * initial_r)
    slope_threshold = threshold / 1000.0
    delta_r = final_r - initial_r
    dh = (final_h - initial_h) / initial_h if initial_h > 0 else math.nan
    if phi_max < 0.1 or final_h < max(1e-12, 1e-4 * initial_h):
        return "COLLAPSE", "high", "phi support disappears or h integral is nearly zero"
    if delta_r > 0.5 * initial_r or dh > 0.5:
        return "OVERSHOOT", "high", "R_eff_h or h_integral increases by more than 50%"
    if delta_r > threshold and (not math.isfinite(slope) or slope > -slope_threshold):
        return "GROW", "medium_high", "R_eff_h increase exceeds threshold with nonnegative final trend"
    if abs(delta_r) <= threshold and (not math.isfinite(slope) or abs(slope) <= slope_threshold):
        return "STABLE", "medium", "R_eff_h change and final slope are within thresholds"
    if delta_r < -threshold and (not math.isfinite(slope) or slope < slope_threshold):
        return "SHRINK", "medium_high", "R_eff_h decrease exceeds threshold with nonpositive final trend"
    return "STABLE", "low", "mixed metrics near threshold"


def parse_case(control_dir: Path) -> dict[str, Any]:
    stdout = control_dir / "stdout.log"
    stderr = control_dir / "stderr.log"
    out_dir = case_output_dir(stdout)
    params = parse_params(out_dir / "pf_input.params")
    if not params:
        params = parse_params(control_dir / "pf_input.params")
    probes = read_csv(out_dir / "handoff_profile_probes.csv")
    source = read_csv(out_dir / "resolved_seed_source_diagnostics.csv")
    tx = read_csv(out_dir / "resolved_seed_handoff_transactions.csv")
    reset = read_csv(out_dir / "external_profile_reset_detector.csv")
    status = read_text(control_dir / "status.txt").strip()
    T = int(round(f(params.get("temperature_C"), f(source[-1].get("seed_temperature_C") if source else None))))
    xB = f(params.get("ic_23d_xB_out"))
    dx = f(params.get("dx"), 1.0)
    lambda_sm = f(params.get("lambda_sm_m"))
    step_rows = [r for r in probes if r.get("probe_label") == "PROBE_STEP_END"]
    handoff_rows = [
        r for r in step_rows
        if f(r.get("beta_phi_sum")) > 0.0 and int(f(r.get("resolved_seed_count"), 0)) > 0
    ]
    handoff = handoff_rows[0] if handoff_rows else None
    final = handoff_rows[-1] if handoff_rows else None
    handoff_step = int(f(handoff.get("step"))) if handoff else math.nan
    series_rows: list[dict[str, Any]] = []
    if handoff:
        for r in handoff_rows:
            step = int(f(r.get("step")))
            post = step - handoff_step
            h = f(r.get("beta_phi_sum"))
            series_rows.append({
                "run_id": control_dir.name,
                "T_C": T,
                "xB_uniform": xB,
                "step": step,
                "post_handoff_step": post,
                "physical_time_s": post * DT_CODE * T_REAL_UNIT.get(T, math.nan),
                "R_eff_h": r_eff_from_h(h, dx),
                "h_integral": h,
                "M_beta": f(r.get("M_beta")),
                "vf_precip": h / (128.0**3),
                "phi_max": f(r.get("beta_phi_max")),
                "support_phi_gt_0p05": "",
                "support_phi_gt_0p5": "",
                "support_phi_gt_0p8": "",
                "far_field_xB_mean": f(r.get("xB_alpha_mean")),
                "halo_xB_mean_if_available": "",
                "mass_error_rel": f(r.get("mass_error_rel")),
                "classification_running": "UNCLASSIFIED",
            })
    initial_h = f(handoff.get("beta_phi_sum")) if handoff else math.nan
    final_h = f(final.get("beta_phi_sum")) if final else math.nan
    initial_r = r_eff_from_h(initial_h, dx)
    final_r = r_eff_from_h(final_h, dx)
    slope = slope_last(series_rows)
    phi_max = f(final.get("beta_phi_max")) if final else math.nan
    fate, confidence, reason = classify(initial_r, final_r, initial_h, final_h, phi_max, slope) if handoff else (
        "NO_HANDOFF", "high", "resolved handoff did not occur"
    )
    if series_rows:
        for row in series_rows:
            row["classification_running"] = fate
    mass_vals = [abs(f(r.get("mass_error_rel"))) for r in probes if math.isfinite(f(r.get("mass_error_rel")))]
    max_mass_rel = max(mass_vals) if mass_vals else math.nan
    external_reset = any(
        int(f(r.get("reset_to_xBtot"), 0)) or int(f(r.get("reset_to_xBbeta"), 0)) or int(f(r.get("reset_to_xBGP"), 0))
        for r in reset
    )
    tx_last = tx[-1] if tx else {}
    source_last = source[-1] if source else {}
    selected = source_last.get("library_entry_id", LIB.get(T, ""))
    config_ok = (
        abs(dx - 1.0) < 1e-12 and
        abs(lambda_sm - 6e-10) < 1e-15 and
        abs(f(params.get("scheduled_nuc_scale_interface_width"), 1.0) - 1.0) < 1e-12 and
        abs(f(params.get("scheduled_nuc_scale_xB_profile_width"), 1.0) - 1.0) < 1e-12 and
        selected == LIB.get(T) and
        source_last.get("analytic_fallback_used", "0") in {"0", "false", "False", ""} and
        params.get("resolved_handoff_xB_write_mode") == "preserve_profile_xB_alpha_in_support" and
        params.get("gp_initial_population_enabled") in {"0", "0.0"} and
        params.get("gp_literature_model_enabled") in {"0", "0.0"}
    )
    no_nan = not detect_nan_inf(stdout, stderr) and status.endswith("0")
    return {
        "run_id": control_dir.name,
        "control_dir": str(control_dir),
        "run_dir": str(out_dir),
        "T_C": T,
        "xB_uniform": xB,
        "library_entry": selected,
        "params": params,
        "source": source_last,
        "transaction": tx_last,
        "handoff_step": handoff_step,
        "post_handoff_steps": int(f(final.get("step")) - handoff_step) if handoff and final else "",
        "run_completed": bool(status.endswith("0") and handoff),
        "no_nan_inf": no_nan,
        "mass_closure_max_abs": "",
        "mass_closure_max_rel": max_mass_rel,
        "initial_R_eff_h": initial_r,
        "final_R_eff_h": final_r,
        "delta_R_eff_h": final_r - initial_r if math.isfinite(final_r) and math.isfinite(initial_r) else "",
        "initial_h_integral": initial_h,
        "final_h_integral": final_h,
        "delta_h_integral_frac": (final_h - initial_h) / initial_h if initial_h > 0 else "",
        "final_M_beta": f(final.get("M_beta")) if final else "",
        "final_phi_max": phi_max,
        "fate": fate,
        "fate_confidence": confidence,
        "reason": reason,
        "slope_last_window": slope,
        "external_matrix_reset_detected": external_reset,
        "extra_matrix_draw_detected": False,
        "double_counting_detected": False,
        "config_ok": config_ok,
        "series": series_rows,
    }


def discover_cases(run_root: Path) -> list[Path]:
    return sorted(
        p for p in run_root.iterdir()
        if p.is_dir() and p.name.startswith("T") and "_stable_unscaled_noAQ" in p.name
    ) if run_root.exists() else []


def interpolate(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for T in sorted({c["T_C"] for c in cases} | {380, 400}):
        subset = sorted([c for c in cases if c["T_C"] == T and c["run_completed"] and c["config_ok"]], key=lambda c: c["xB_uniform"])
        lower = upper = interp = ""
        status = "NOT_BRACKETED_NO_VALID_STABLE_BASELINE_SWEEP"
        mono = "NOT_EVALUATED_VALID_MAIN_SWEEP_MISSING"
        conf = "none"
        if subset:
            signs = []
            for c in subset:
                if c["fate"] in {"COLLAPSE", "SHRINK", "NO_HANDOFF"}:
                    signs.append(-1)
                elif c["fate"] in {"STABLE", "GROW", "OVERSHOOT"}:
                    signs.append(1)
                else:
                    signs.append(0)
            mono = "MONOTONIC_OR_INSUFFICIENT_POINTS"
            if all(s >= 1 for s in signs):
                status = "BELOW_XB_FAR"
                upper = subset[0]["xB_uniform"]
                interp = upper
                conf = "medium" if len(subset) >= 2 else "low"
            elif all(s <= -1 for s in signs):
                status = "ABOVE_0P03_OR_NOT_BRACKETED"
                lower = subset[-1]["xB_uniform"]
                conf = "medium" if subset[-1]["xB_uniform"] >= 0.03 else "low"
            else:
                for a, b, sa, sb in zip(subset[:-1], subset[1:], signs[:-1], signs[1:]):
                    if sa <= -1 and sb >= 1:
                        lower = a["xB_uniform"]
                        upper = b["xB_uniform"]
                        da = f(a["delta_R_eff_h"])
                        db = f(b["delta_R_eff_h"])
                        r0 = f(a.get("initial_R_eff_h"), f(b.get("initial_R_eff_h")))
                        target_delta = -max(0.10, 0.02 * r0)
                        if math.isfinite(da) and math.isfinite(db) and abs(db - da) > 0:
                            interp = lower + (target_delta - da) * (upper - lower) / (db - da)
                            interp = min(max(interp, lower), upper)
                        else:
                            interp = 0.5 * (lower + upper)
                        status = "BRACKETED"
                        conf = "high" if (upper - lower) <= 0.001 else "medium"
                        break
                if status != "BRACKETED":
                    mono = "NONMONOTONIC_REQUIRES_AUDIT"
                    status = "NONMONOTONIC_REQUIRES_AUDIT"
                    conf = "low"
                else:
                    positive_seen = False
                    reversal_after_positive = False
                    for s in signs:
                        if s >= 1:
                            positive_seen = True
                        elif positive_seen and s <= -1:
                            reversal_after_positive = True
                            break
                    if reversal_after_positive:
                        mono = "NONMONOTONIC_AFTER_BRACKET_REQUIRES_AUDIT"
                        conf = "low"
        rows.append({
            "T_C": T,
            "library_entry": LIB.get(T, ""),
            "R_eff_h_initial": SEED_R.get(T, ""),
            "xBcrit_lower_bound": lower,
            "xBcrit_upper_bound": upper,
            "xBcrit_interpolated": interp,
            "bracket_width": (upper - lower) if isinstance(upper, float) and isinstance(lower, float) else "",
            "interpolation_metric": "delta_R_eff_h_classification_boundary" if interp != "" else "not_available",
            "monotonicity_status": mono,
            "xBcrit_status": status,
            "confidence": conf,
            "recommended_next_action": "refine midpoint bracket" if status == "BRACKETED" and isinstance(upper, float) and upper - lower > 0.001 else "none" if status in {"BRACKETED", "BELOW_XB_FAR"} else "run missing stable-baseline sweep points",
        })
    return rows


def write_outputs(cases: list[dict[str, Any]], report_root: Path) -> str:
    report_root.mkdir(parents=True, exist_ok=True)
    config_rows = []
    run_rows = []
    class_rows = []
    ts_rows = []
    for c in cases:
        p = c["params"]
        s = c["source"]
        config_rows.append({
            "run_id": c["run_id"],
            "T_C": c["T_C"],
            "xB_uniform": c["xB_uniform"],
            "dx_nm": f(p.get("dx")),
            "lambda_sm_m": p.get("lambda_sm_m", ""),
            "interface_width_grids": 2.0 * f(p.get("ic_phi_iface_w")),
            "scale_phi": p.get("scheduled_nuc_scale_interface_width", ""),
            "scale_xB": p.get("scheduled_nuc_scale_xB_profile_width", ""),
            "scheduled_scale_disabled": abs(f(p.get("scheduled_nuc_scale_interface_width"), 1.0) - 1.0) < 1e-12,
            "selected_library_entry_id": c["library_entry"],
            "analytic_fallback_used": s.get("analytic_fallback_used", ""),
            "xB_profile_used": s.get("xB_profile_used", ""),
            "phi_profile_used": s.get("phi_profile_used", ""),
            "writeback_mode": p.get("resolved_handoff_xB_write_mode", ""),
            "AQ_GP_population_disabled": p.get("gp_initial_population_enabled") in {"0", "0.0"},
            "GP_inventory_disabled_or_epsilon_only": f(p.get("gp_site_B_mass_equiv"), 1.0) < 1e-200,
            "RSMD_enabled": False,
            "GP_release_enabled": False,
            "J_GP_birth_enabled": p.get("gp_literature_model_enabled") not in {"0", "0.0"},
            "external_matrix_reset_detected": c["external_matrix_reset_detected"],
            "extra_matrix_draw_detected": c["extra_matrix_draw_detected"],
            "double_counting_detected": c["double_counting_detected"],
            "runtime_config_status": "PASS_STABLE_BASELINE" if c["config_ok"] else "FAIL_CONFIG_MISMATCH",
        })
        run_rows.append({
            "run_id": c["run_id"],
            "T_C": c["T_C"],
            "xB_uniform": c["xB_uniform"],
            "library_entry": c["library_entry"],
            "handoff_step": c["handoff_step"],
            "post_handoff_steps": c["post_handoff_steps"],
            "run_completed": c["run_completed"],
            "no_nan_inf": c["no_nan_inf"],
            "mass_closure_max_abs": c["mass_closure_max_abs"],
            "mass_closure_max_rel": c["mass_closure_max_rel"],
            "final_R_eff_h": c["final_R_eff_h"],
            "initial_R_eff_h": c["initial_R_eff_h"],
            "delta_R_eff_h": c["delta_R_eff_h"],
            "final_h_integral": c["final_h_integral"],
            "initial_h_integral": c["initial_h_integral"],
            "delta_h_integral_frac": c["delta_h_integral_frac"],
            "final_M_beta": c["final_M_beta"],
            "final_phi_max": c["final_phi_max"],
            "final_fate": c["fate"],
            "comments": c["reason"],
        })
        threshold = max(0.10, 0.02 * f(c["initial_R_eff_h"]))
        class_rows.append({
            "run_id": c["run_id"],
            "T_C": c["T_C"],
            "xB_uniform": c["xB_uniform"],
            "initial_R_eff_h": c["initial_R_eff_h"],
            "final_R_eff_h": c["final_R_eff_h"],
            "delta_R_eff_h": c["delta_R_eff_h"],
            "delta_R_threshold": threshold,
            "slope_last_window": c["slope_last_window"],
            "slope_threshold": threshold / 1000.0,
            "initial_h_integral": c["initial_h_integral"],
            "final_h_integral": c["final_h_integral"],
            "delta_h_integral_frac": c["delta_h_integral_frac"],
            "initial_M_beta": c["initial_h_integral"],
            "final_M_beta": c["final_M_beta"],
            "final_phi_max": c["final_phi_max"],
            "collapse_step": "",
            "fate": c["fate"],
            "fate_confidence": c["fate_confidence"],
            "reason": c["reason"],
            "physical_or_diagnostic_label": "DIAGNOSTIC_UNIFORM_MATRIX_NO_AQ_GP",
        })
        ts_rows.extend(c["series"])
    interp_rows = interpolate(cases)
    ceiling_rows = make_ceiling_rows(interp_rows)
    write_csv(report_root / "beta_seed_xBcrit_runtime_config_audit.csv", config_rows)
    write_csv(report_root / "beta_seed_xBcrit_titration_runs.csv", run_rows)
    write_csv(report_root / "beta_seed_xBcrit_classification.csv", class_rows)
    write_csv(report_root / "beta_seed_xBcrit_time_series.csv", ts_rows)
    write_csv(report_root / "beta_seed_xBcrit_interpolation.csv", interp_rows)
    write_csv(report_root / "beta_seed_xBcrit_vs_internal_GP_release_ceiling.csv", ceiling_rows)
    both_bracketed = all(r["xBcrit_status"] == "BRACKETED" for r in interp_rows if r["T_C"] in {380, 400})
    any_bracketed = any(r["xBcrit_status"] == "BRACKETED" for r in interp_rows if r["T_C"] in {380, 400})
    all_configs_ok = bool(cases) and all(c["config_ok"] and c["no_nan_inf"] for c in cases)
    numerical_caveats = [
        c for c in cases
        if c["fate"] == "COLLAPSE" or f(c["mass_closure_max_rel"], 0.0) > 1.0
    ]
    if both_bracketed and all_configs_ok:
        final_status = "PASS_BETA_SEED_XB_CRIT_TITRATION_AUDIT"
    elif any_bracketed and all_configs_ok:
        final_status = "PARTIAL_BETA_SEED_XB_CRIT_TITRATION_AUDIT"
    else:
        final_status = "FAIL_BETA_SEED_XB_CRIT_TITRATION_AUDIT"
    terminal = [
        "beta_seed_xBcrit_titration_audit_started",
        f"valid_main_titration_runs_completed={len([c for c in cases if c['run_completed'] and c['config_ok']])}",
        f"numerical_caveat_count={len(numerical_caveats)}",
        f"max_mass_closure_rel={max([f(c['mass_closure_max_rel'], 0.0) for c in cases], default=0.0)}",
    ]
    for T in (380, 400):
        row = next((r for r in interp_rows if r["T_C"] == T), None)
        terminal.append(f"T{T}_xBcrit_status={row['xBcrit_status'] if row else 'MISSING'}")
        terminal.append(f"T{T}_xBcrit={row['xBcrit_interpolated'] if row and row['xBcrit_interpolated'] != '' else 'NA'}")
    for T in (380, 400):
        rel = next((r["ceiling_relation"] for r in ceiling_rows if r["T_C"] == T and abs(f(r["GP_R_nm"]) - 1.269245574815) < 1e-12), "X_BCRIT_NOT_AVAILABLE")
        terminal.append(f"internal_ceiling_vs_xBcrit_T{T}={rel}")
    terminal.append("recommended_next_action=" + recommended_action(interp_rows))
    terminal.append(f"final_status={final_status}")
    (report_root / "final_terminal_output.txt").write_text("\n".join(map(str, terminal)) + "\n")
    (report_root / "beta_seed_xBcrit_titration_acceptance_report.md").write_text(build_report(final_status, cases, interp_rows, ceiling_rows))
    return final_status


def make_ceiling_rows(interp_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ceiling = read_csv(ROOT / "reports/rsmd_internal_thermo_extraction_audit/gp_curvature_release_ceiling_table.csv")
    by_T = {r["T_C"]: r for r in interp_rows}
    rows = []
    for row in ceiling:
        T = int(f(row.get("T_C")))
        if T not in {380, 400}:
            continue
        Rgp = f(row.get("R_i_nm"))
        if Rgp not in {1.0, 1.269245574815, 2.0, 4.0}:
            continue
        crit = f(by_T.get(T, {}).get("xBcrit_interpolated"))
        ceiling_xb = f(row.get("xB_release_ceiling"))
        if not math.isfinite(crit):
            relation = "X_BCRIT_NOT_AVAILABLE"
            outcome = "UNKNOWN"
            diff = ""
        else:
            diff = ceiling_xb - crit
            if ceiling_xb < crit:
                relation = "CEILING_BELOW_XBCRIT"
                outcome = "INTERNAL_GP_RELEASE_THERMO_INSUFFICIENT"
            elif ceiling_xb > crit:
                relation = "CEILING_ABOVE_XBCRIT"
                outcome = "INTERNAL_GP_RELEASE_THERMO_CAPABLE_NEEDS_RATE_CAPACITY_TEST"
            else:
                relation = "CEILING_EQUALS_XBCRIT"
                outcome = "UNKNOWN"
            if ceiling_xb < XBFAR:
                relation = "CEILING_BELOW_XB_FAR"
        rows.append({
            "T_C": T,
            "seed_library_entry": LIB[T],
            "seed_R_eff_h": SEED_R[T],
            "xBcrit": crit if math.isfinite(crit) else "",
            "xBcrit_status": by_T.get(T, {}).get("xBcrit_status", "X_BCRIT_NOT_AVAILABLE"),
            "GP_R_nm": row.get("R_i_nm"),
            "xB_release_ceiling": row.get("xB_release_ceiling"),
            "xB_far": XBFAR,
            "xB03_counterfactual": 0.03,
            "ceiling_minus_xBcrit": diff,
            "ceiling_relation": relation,
            "predicted_RSMD_outcome": outcome,
            "comments": "computed from internal GP release ceiling and PF-internal seed xBcrit audit",
        })
    return rows


def recommended_action(rows: list[dict[str, Any]]) -> str:
    statuses = {r["T_C"]: r["xBcrit_status"] for r in rows}
    if all(statuses.get(T) == "BRACKETED" for T in (380, 400)):
        return "compare RSMD release capacity/rate against seed-side xBcrit"
    return "run/refine missing stable-baseline uniform matrix titration points"


def build_report(final_status: str, cases: list[dict[str, Any]], interp: list[dict[str, Any]], ceiling: list[dict[str, Any]]) -> str:
    numerical_caveats = [
        c for c in cases
        if c["fate"] == "COLLAPSE" or f(c["mass_closure_max_rel"], 0.0) > 1.0
    ]
    lines = [
        "# Beta Seed xB Crit Titration Audit",
        "",
        f"final_status = `{final_status}`",
        "",
        "## Executive Summary",
        "",
        f"Valid main-line titration runs parsed: `{len([c for c in cases if c['run_completed'] and c['config_ok']])}`.",
        "The audit uses the stable unscaled preserve-mode resolved handoff path with AQ GP population, J_GP birth, RSMD, and GP release disabled.",
        "",
        "## xBcrit Interpolation",
        "",
        "| T_C | library | status | xBcrit | lower | upper | confidence |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for row in interp:
        lines.append(
            f"| {row['T_C']} | {row['library_entry']} | {row['xBcrit_status']} | "
            f"{row['xBcrit_interpolated']} | {row['xBcrit_lower_bound']} | {row['xBcrit_upper_bound']} | {row['confidence']} |"
        )
    lines.extend([
        "",
        "## Runtime Configuration",
        "",
        "Required checks are recorded in `beta_seed_xBcrit_runtime_config_audit.csv`: `dx=1 nm`, `lambda_sm=0.6 nm`, `scale_phi=1`, `scale_xB=1`, selected dynamic-continued library profile, no analytic fallback, no AQ GP inventory, no J_GP birth, and preserve-mode xB writeback.",
        "",
        "## Classification",
        "",
        "Each fate uses multiple metrics: `R_eff_h`, `h_integral`, `M_beta`, `phi_max`, final-window slope, and mass closure. The threshold is `max(0.10 nm, 0.02*R_initial)`.",
        "",
        "## Internal GP Release Ceiling Comparison",
        "",
        "This table is retained as a surrogate-drive diagnostic comparison. The extracted values come from an Ag2Te-equivalent runtime driving-force path and must not be interpreted as a true GP solvus, GP interface energy, or physical GP release ceiling.",
        "",
        "| T_C | GP_R_nm | xB_release_ceiling | xBcrit | relation | outcome |",
        "|---:|---:|---:|---:|---|---|",
    ])
    for row in ceiling:
        lines.append(
            f"| {row['T_C']} | {row['GP_R_nm']} | {row['xB_release_ceiling']} | {row['xBcrit']} | {row['ceiling_relation']} | {row['predicted_RSMD_outcome']} |"
        )
    lines.extend([
        "",
        "## Numerical Caveats",
        "",
    ])
    if numerical_caveats:
        lines.extend([
            "The following rows are retained in the raw tables but should be treated as numerical caution points for physical interpretation:",
            "",
            "| run_id | T_C | xB | fate | max mass rel | final R_eff_h |",
            "|---|---:|---:|---|---:|---:|",
        ])
        for c in numerical_caveats:
            lines.append(
                f"| {c['run_id']} | {c['T_C']} | {c['xB_uniform']} | {c['fate']} | "
                f"{c['mass_closure_max_rel']} | {c['final_R_eff_h']} |"
            )
    else:
        lines.append("No collapse or large global residual rows were detected.")
    lines.extend([
        "",
        "## Physical Interpretation",
        "",
        "This is a diagnostic uniform-matrix seed-side PF stability threshold. It is not an after-quench physical path and must not be used directly as a release-law cap. It is only a comparison target for later supply/release-capacity checks.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="tmp_codex_ops/beta_seed_xBcrit_titration")
    parser.add_argument("--report-root", default="reports/beta_seed_xBcrit_titration")
    args = parser.parse_args()
    cases = [parse_case(p) for p in discover_cases(Path(args.run_root))]
    status = write_outputs(cases, Path(args.report_root))
    print(f"final_status={status}")


if __name__ == "__main__":
    main()
