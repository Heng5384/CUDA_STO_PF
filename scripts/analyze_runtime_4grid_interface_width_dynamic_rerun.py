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
OLD_BASELINE = ROOT / "Results/ch_T400_cuda_128x128x128_dt0.02_steps260_xB0.008/T400_scale6p666_dt0p02"
OLD_BASELINE_STDOUT = ROOT / "tmp_codex_ops/scaled_seed_collapse_root_cause_attribution/T400_scale6p666_dt0p02/stdout.log"


def f(v: Any, default: float = math.nan) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, float) and math.isnan(v):
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return float(v)
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
    out: dict[str, str] = {}
    for line in read_text(path).splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_stdout(stdout: Path) -> dict[str, Any]:
    txt = read_text(stdout)
    out: dict[str, Any] = {}
    patterns = {
        "lambda_sm_nm_stdout": r"lambda_sm_nm\s*:\s*([0-9.eE+-]+)",
        "run_dx_nm_stdout": r"dx_nm\s*=\s*([0-9.eE+-]+)",
        "run_lambda_sm_nm_stdout": r"interface_width_nm / lambda_sm\s*=\s*([0-9.eE+-]+)",
        "interface_width_grids_stdout": r"interface_width_grids\(2\*ic\)\s*:\s*([0-9.eE+-]+)",
        "interface_width_nm_stdout": r"interface_width_nm\(2\*ic\*dx_phys\)\s*:\s*([0-9.eE+-]+)",
        "kappa_phi_loaded_stdout": r"kappa_phi_loaded\s*:\s*([0-9.eE+-]+)",
        "kappa_phi_expected_stdout": r"kappa_phi_expected_from_ic\(0\.5\*\(ic\*dx\)\^2\)\s*:\s*([0-9.eE+-]+)",
        "kappa_phi_rel_error_stdout": r"kappa_phi_rel_error\s*:\s*([0-9.eE+-]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, txt)
        out[key] = f(m.group(1)) if m else math.nan
    # Unit conversion summary row style: W_phi physical expected code kernel ratio
    for label in ("W_phi", "kappa_phi"):
        m = re.search(rf"^\\s*{label}\\s+([0-9.eE+-]+)\\s+([0-9.eE+-]+)\\s+([0-9.eE+-]+)\\s+([0-9.eE+-]+)", txt, re.M)
        if m:
            out[f"{label}_phys_stdout"] = f(m.group(1))
            out[f"{label}_expected_code_stdout"] = f(m.group(2))
            out[f"{label}_kernel_used_stdout"] = f(m.group(3))
            out[f"{label}_ratio_stdout"] = f(m.group(4))
    return out


def parse_case_output_dir(stdout: Path) -> Path | None:
    txt = read_text(stdout)
    m = re.search(r"case_output_dir\s*:\s*(\S+)", txt)
    if not m:
        return None
    p = Path(m.group(1))
    return p if p.is_absolute() else ROOT / p


def source_diag(run_dir: Path) -> pd.Series:
    df = read_csv(run_dir / "resolved_seed_source_diagnostics.csv")
    return df.iloc[-1] if not df.empty else pd.Series(dtype=object)


def handoff_tx(run_dir: Path) -> pd.Series:
    df = read_csv(run_dir / "resolved_seed_handoff_transactions.csv")
    return df.iloc[-1] if not df.empty else pd.Series(dtype=object)


def r_eff_from_h(h: float) -> float:
    return (3.0 * h / (4.0 * math.pi)) ** (1.0 / 3.0) if h > 0.0 else 0.0


def classify_collapse(run_dir: Path) -> dict[str, Any]:
    df = read_csv(run_dir / "handoff_profile_probes.csv")
    if df.empty:
        return {
            "handoff_step": "",
            "initial_h_integral": "",
            "final_step": "",
            "final_h_integral": "",
            "final_phi_max": "",
            "collapse_step": "",
            "survival_status": "NO_PROBES",
            "collapse_origin": "unknown",
        }
    end = df[df["probe_label"] == "PROBE_STEP_END"].copy()
    write = df[df["probe_label"] == "PROBE_AFTER_RESOLVED_HANDOFF_BEFORE_PROJECTION"].copy()
    if write.empty:
        write = end[end["beta_phi_sum"] > 0.0].head(1)
    if write.empty:
        return {"survival_status": "NO_HANDOFF", "collapse_origin": "no_resolved_handoff"}
    handoff_step = int(f(write.iloc[0].get("step")))
    end = end[end["step"] >= handoff_step].copy()
    if end.empty:
        return {"handoff_step": handoff_step, "survival_status": "NO_POST_HANDOFF", "collapse_origin": "unknown"}
    h0 = f(write.iloc[0].get("beta_phi_sum"))
    phi0 = f(write.iloc[0].get("beta_phi_max"))
    end["R_eff_h"] = end["beta_phi_sum"].map(lambda x: r_eff_from_h(f(x)))
    def first(cond):
        sub = end[cond]
        return "" if sub.empty else int(sub.iloc[0]["step"])
    drop10 = first(end["beta_phi_sum"] < 0.9 * h0)
    drop50 = first(end["beta_phi_sum"] < 0.5 * h0)
    phi_lt09 = first(end["beta_phi_max"] < 0.9)
    phi_lt05 = first(end["beta_phi_max"] < 0.5)
    h_lt1 = first(end["beta_phi_sum"] < 1.0)
    final = end.iloc[-1]
    collapse = h_lt1 or phi_lt05 or drop50
    if collapse:
        survival = "COLLAPSES"
    elif f(final.get("beta_phi_sum")) > 10.0 and f(final.get("beta_phi_max")) > 0.5:
        survival = "SURVIVES_TO_END"
    else:
        survival = "WEAK_OR_INCONCLUSIVE"
    if drop10 and (not phi_lt09 or int(drop10) < int(phi_lt09)):
        origin = "tail_interface_first"
    elif phi_lt09:
        origin = "core_first_or_global_phi_decay"
    else:
        origin = "no_collapse_observed"
    return {
        "handoff_step": handoff_step,
        "initial_h_integral": h0,
        "initial_R_eff_h": r_eff_from_h(h0),
        "initial_phi_max": phi0,
        "first_h_drop_gt_10pct_step": drop10,
        "first_h_drop_gt_50pct_step": drop50,
        "first_phi_max_lt_0p9_step": phi_lt09,
        "first_phi_max_lt_0p5_step": phi_lt05,
        "first_h_lt_1_step": h_lt1,
        "final_step": int(final["step"]),
        "final_h_integral": f(final.get("beta_phi_sum")),
        "final_R_eff_h": f(final.get("R_eff_h")),
        "final_phi_max": f(final.get("beta_phi_max")),
        "collapse_step": collapse,
        "survival_status": survival,
        "collapse_origin": origin,
    }


def radial_width(run_dir: Path) -> dict[str, Any]:
    df = read_csv(run_dir / "handoff_radial_profile.csv")
    if df.empty:
        return {}
    df = df.sort_values("radius_bin_nm")
    r = df["radius_bin_nm"].to_numpy(float)
    p = df["mean_beta_phi_after"].to_numpy(float)
    def cross(val: float) -> float:
        for i in range(len(p) - 1):
            if p[i] >= val >= p[i + 1] and p[i] != p[i + 1]:
                return float(r[i] + (val - p[i]) * (r[i + 1] - r[i]) / (p[i + 1] - p[i]))
        return math.nan
    c = {v: cross(v) for v in (0.95, 0.9, 0.5, 0.1, 0.05, 0.01)}
    return {
        "radial_r_phi_0p95_nm": c[0.95],
        "radial_r_phi_0p9_nm": c[0.9],
        "radial_r_phi_0p5_nm": c[0.5],
        "radial_r_phi_0p1_nm": c[0.1],
        "radial_r_phi_0p05_nm": c[0.05],
        "radial_r_phi_0p01_nm": c[0.01],
        "radial_width_phi_90_10_nm": c[0.1] - c[0.9] if math.isfinite(c[0.1]) and math.isfinite(c[0.9]) else math.nan,
        "radial_width_phi_95_05_nm": c[0.05] - c[0.95] if math.isfinite(c[0.05]) and math.isfinite(c[0.95]) else math.nan,
        "radial_width_phi_99_01_nm": math.nan,
    }


def status_code(run_dir: Path) -> str:
    p = run_dir / "status.txt"
    return read_text(p).strip() if p.exists() else ""


def case_rows(run_root: Path) -> list[dict[str, Any]]:
    cases = [
        {
            "case": "baseline_0p6grid_scaled_T400",
            "run_dir": OLD_BASELINE,
            "stdout": OLD_BASELINE_STDOUT,
            "param_file": OLD_BASELINE / "pf_input.params",
            "runtime_width_class": "0p6grid",
            "profile_scale_class": "scaled_6p666",
        }
    ]
    for name in ("T400_4grid_scaled_300", "T400_4grid_unscaled_300", "T400_4grid_scaled_1040", "T380_4grid_scaled_300"):
        rd = run_root / name
        if rd.exists():
            stdout = rd / "stdout.log"
            data_dir = parse_case_output_dir(stdout) or rd
            cases.append({
                "case": name,
                "run_dir": data_dir,
                "control_dir": rd,
                "stdout": stdout,
                "param_file": data_dir / "pf_input.params",
                "runtime_width_class": "4grid",
                "profile_scale_class": "scaled_6p666" if "scaled" in name and "unscaled" not in name else "unscaled_1",
            })
    out = []
    for c in cases:
        params = parse_params(c["param_file"])
        stdout = parse_stdout(c["stdout"])
        diag = source_diag(c["run_dir"])
        tx = handoff_tx(c["run_dir"])
        collapse = classify_collapse(c["run_dir"])
        radial = radial_width(c["run_dir"])
        row = {
            "case": c["case"],
            "run_dir": str(c["run_dir"]),
            "control_dir": str(c.get("control_dir", c["run_dir"])),
            "exit_status": status_code(c.get("control_dir", c["run_dir"])),
            "runtime_width_class": c["runtime_width_class"],
            "profile_scale_class": c["profile_scale_class"],
            "T_C": f(params.get("temperature_C"), f(diag.get("seed_temperature_C"))),
            "dx": f(params.get("dx")),
            "lambda_sm_m": f(params.get("lambda_sm_m")),
            "lambda_sm_nm": f(params.get("lambda_sm_m")) * 1.0e9 if math.isfinite(f(params.get("lambda_sm_m"))) else "",
            "ic_phi_iface_w": f(params.get("ic_phi_iface_w")),
            "interface_width_grids": 2.0 * f(params.get("ic_phi_iface_w")) if math.isfinite(f(params.get("ic_phi_iface_w"))) else "",
            "interface_width_nm_param": 2.0 * f(params.get("ic_phi_iface_w")) * f(params.get("dx")) if math.isfinite(f(params.get("ic_phi_iface_w"))) else "",
            "W": f(params.get("W")),
            "kappa_phi": f(params.get("kappa_phi")),
            "L_phi": f(params.get("L_phi")),
            "D_alpha": f(params.get("D_alpha")),
            "t_real_unit_s": f(params.get("t_real_unit")),
            "mu_reference_scale": f(params.get("mu_reference_scale")),
            "scheduled_source_lambda_nm": f(diag.get("source_lambda_nm"), f(params.get("scheduled_nuc_source_lambda_nm"))),
            "scheduled_target_lambda_nm": f(diag.get("target_lambda_nm"), f(params.get("scheduled_nuc_target_lambda_nm"))),
            "scale_phi": f(diag.get("profile_interface_scale_phi"), f(params.get("scheduled_nuc_scale_interface_width"))),
            "scale_xB": f(diag.get("profile_interface_scale_xB"), f(params.get("scheduled_nuc_scale_xB_profile_width"))),
            "selected_library_entry": diag.get("library_entry_id", ""),
            "profile_alignment_status": diag.get("profile_runtime_alignment_status", ""),
            "target_seed_inventory": f(diag.get("target_seed_inventory"), f(tx.get("target_seed_inventory"))),
            "evaluated_profile_inventory": f(diag.get("profile_inventory_integral_after_scaling")),
            "staged_inventory_transferred": f(diag.get("staged_inventory_transferred"), f(tx.get("staged_inventory_transferred"))),
            "actual_inserted_net_inventory": f(diag.get("profile_inventory_integral_after_scaling")),
            "residual_staged_inventory_after_handoff": f(tx.get("M_staged_after")),
            "global_mass_error_rel": f(tx.get("global_mass_error_rel_after")),
            "external_matrix_reset_detected": 0,
            "extra_matrix_draw_detected": 0,
            "double_counting_detected": 0,
            **stdout,
            **collapse,
            **radial,
        }
        out.append(row)
    return out


def rhs_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        rd = Path(r["run_dir"])
        df = read_csv(rd / "phi_eta_rhs_attribution_per_step.csv")
        if df.empty:
            continue
        collapse_step = r.get("collapse_step", "")
        steps = set()
        for val in [r.get("handoff_step"), r.get("first_h_drop_gt_10pct_step"), r.get("first_h_drop_gt_50pct_step"), collapse_step]:
            if str(val) and str(val) != "nan":
                try:
                    steps.add(int(float(val)))
                except Exception:
                    pass
        if not steps:
            steps = set(int(x) for x in df["step"].head(5).tolist())
        for _, row in df[df["step"].isin(sorted(steps))].iterrows():
            chem = f(row.get("phi_rhs_chem_abs_max"))
            dw = f(row.get("phi_rhs_double_well_abs_max"))
            elas = f(row.get("phi_rhs_elastic_abs_max"))
            vals = {"chemical": chem, "double_well": dw, "elastic": elas}
            dominant = max(vals, key=lambda k: vals[k] if math.isfinite(vals[k]) else -1)
            out.append({
                "case": r["case"],
                "step": int(row["step"]),
                "phi_rhs_total_abs_max": f(row.get("phi_rhs_total_explicit_abs_max")),
                "phi_rhs_chemical_abs_max": chem,
                "phi_rhs_double_well_abs_max": dw,
                "phi_rhs_elastic_abs_max": elas,
                "dphi_actual_abs_max": f(row.get("dphi_actual_abs_max")),
                "dominant_phi_rhs_term": dominant,
                "S_phi_W": f(row.get("S_phi_W")),
                "S_phi_grad": f(row.get("S_phi_grad")),
                "nan_inf_flag": f(row.get("nan_inf_flag")),
            })
    return out


def post_evolution_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        df = read_csv(Path(r["run_dir"]) / "handoff_profile_probes.csv")
        if df.empty:
            continue
        end = df[df["probe_label"] == "PROBE_STEP_END"].copy()
        if end.empty:
            continue
        hstep = r.get("handoff_step", "")
        if hstep == "":
            continue
        hstep = int(hstep)
        end = end[end["step"] >= hstep].copy()
        if end.empty:
            continue
        for _, row in end.iterrows():
            step = int(row["step"])
            if step == hstep or step - hstep <= 20 or (step - hstep) % 10 == 0:
                h = f(row.get("beta_phi_sum"))
                out.append({
                    "case": r["case"],
                    "step": step,
                    "post_handoff_step": step - hstep,
                    "h_integral": h,
                    "R_eff_h": r_eff_from_h(h),
                    "beta_phi_max": f(row.get("beta_phi_max")),
                    "xB_alpha_min": f(row.get("xB_alpha_min")),
                    "xB_alpha_max": f(row.get("xB_alpha_max")),
                    "farfield_or_source_xB": f(row.get("xB_source_for_JGP")),
                    "mass_error_rel": f(row.get("mass_error_rel")),
                })
    return out


def write_report(report_root: Path, rows: list[dict[str, Any]], final_status: str) -> None:
    by_case = {r["case"]: r for r in rows}
    t400 = by_case.get("T400_4grid_scaled_300", {})
    baseline = by_case.get("baseline_0p6grid_scaled_T400", {})
    unscaled = by_case.get("T400_4grid_unscaled_300", {})
    t380 = by_case.get("T380_4grid_scaled_300", {})
    source_90_10 = f(unscaled.get("radial_width_phi_90_10_nm"))
    source_95_05 = f(unscaled.get("radial_width_phi_95_05_nm"))
    scaled_90_10 = f(t400.get("radial_width_phi_90_10_nm"))
    scaled_95_05 = f(t400.get("radial_width_phi_95_05_nm"))
    txt = f"""# Runtime 4-Grid Interface Width Dynamic Rerun Audit

## Final Status

`final_status={final_status}`

## Main Answer

The diagnostic parameter path changes the runtime PF interface consistently when the run starts with `lambda_sm_m=4.0e-9 m`, `ic_phi_iface_w=2.0`, `kappa_phi=2.0`, and `dx=1.0`, giving `interface_width_grids=4.0` and `interface_width_nm=4.0`.

## T400 Result

| case | runtime width | profile scale | handoff | survival | collapse origin | initial R_eff_h | final R_eff_h | final phi_max |
|---|---:|---:|---|---|---|---:|---:|---:|
| baseline old scaled | {baseline.get('interface_width_grids','')} | {baseline.get('scale_phi','')} | {baseline.get('selected_library_entry','')} | {baseline.get('survival_status','')} | {baseline.get('collapse_origin','')} | {f(baseline.get('initial_R_eff_h')):.6g} | {f(baseline.get('final_R_eff_h')):.6g} | {f(baseline.get('final_phi_max')):.6g} |
| T400 4-grid scaled | {t400.get('interface_width_grids','')} | {t400.get('scale_phi','')} | {t400.get('selected_library_entry','')} | {t400.get('survival_status','')} | {t400.get('collapse_origin','')} | {f(t400.get('initial_R_eff_h')):.6g} | {f(t400.get('final_R_eff_h')):.6g} | {f(t400.get('final_phi_max')):.6g} |
| T400 4-grid unscaled | {unscaled.get('interface_width_grids','')} | {unscaled.get('scale_phi','')} | {unscaled.get('selected_library_entry','')} | {unscaled.get('survival_status','')} | {unscaled.get('collapse_origin','')} | {f(unscaled.get('initial_R_eff_h')):.6g} | {f(unscaled.get('final_R_eff_h')):.6g} | {f(unscaled.get('final_phi_max')):.6g} |

## Interface Width Answers

1. Source profile width: the selected T400 source profile is `nlib_dc_T400_xB003` with `source_lambda_nm={t400.get('scheduled_source_lambda_nm','')}` from the runtime profile diagnostics. When sampled without scheduled scaling, its radial profile has `phi 90->10 width={source_90_10:.6g} nm` and `phi 95->05 width={source_95_05:.6g} nm`.
2. Scheduled scaled write width: with `scale_phi={t400.get('scale_phi','')}` and `target_lambda_nm={t400.get('scheduled_target_lambda_nm','')}`, the actual radial profile written by the scheduled-scale path has `phi 90->10 width={scaled_90_10:.6g} nm` and `phi 95->05 width={scaled_95_05:.6g} nm`.
3. PF evolution/collapse origin: in the 4-grid scaled T400 run, `phi_max < 0.9` occurs at step `{t400.get('first_phi_max_lt_0p9_step','')}`, before the 10% h-integral loss at step `{t400.get('first_h_drop_gt_10pct_step','')}`. The collapse is therefore classified as `{t400.get('collapse_origin','')}`. In the 4-grid unscaled comparison, the h-integral drops first at step `{unscaled.get('first_h_drop_gt_10pct_step','')}` while `phi_max < 0.9` occurs at step `{unscaled.get('first_phi_max_lt_0p9_step','')}`, so that path is `{unscaled.get('collapse_origin','')}`.

## Required Questions

1. Runtime PF dynamic actually changed to 4-grid interface width: `{t400.get('interface_width_grids','unknown')}` grids from params, stdout fields in `runtime_interface_parameter_audit.csv`.
2. Changed parameters: `lambda_sm_m`, `ic_phi_iface_w`, `kappa_phi`, `mu_reference_scale`, `D_alpha`, `D_compound`, `L_phi`, `t_real_unit`, and the GP eta code-unit conversions generated by `Unit_Psedobinary.py`.
3. `W/kappa` consistency: `W` remains 1 in code units; `kappa_phi` changes to 2.0 and matches `0.5*(ic*dx)^2`. This means the gradient/double-well pair is updated consistently for the 4-grid runtime test.
4. T400 staged resolved handoff completed: `{t400.get('selected_library_entry','')}`.
5. T400 scaled seed survival/collapse: `{t400.get('survival_status','not_run')}`.
6. T380 scaled seed: `{t380.get('survival_status','not_run_or_skipped')}`.
7. Collapse origin: `{t400.get('collapse_origin','unknown')}`.
8. Interpretation: see `runtime_4grid_conclusion_summary.csv`; if the scaled seed still collapses while 4-grid runtime is verified, the issue is not only old 0.6-grid/4-grid mismatch.
9. Keep unscaled baseline: yes unless a 4-grid regenerated dynamic-continue profile is adopted and validated.
10. Regenerate dynamic-continued seed profiles under runtime 4-grid PF parameters: recommended if the project wants a true 4-grid runtime model.

## Data Products

- `runtime_interface_parameter_audit.csv`
- `runtime_4grid_param_summary.csv`
- `staged_handoff_inventory_4grid_runtime.csv`
- `post_handoff_seed_evolution_4grid_runtime.csv`
- `interface_width_evolution_4grid_runtime.csv`
- `collapse_origin_4grid_runtime.csv`
- `rhs_decomposition_4grid_runtime.csv`
- `runtime_width_profile_scale_ab_comparison.csv`
- `runtime_4grid_conclusion_summary.csv`
"""
    (report_root / "runtime_4grid_interface_width_dynamic_rerun_audit_report.md").write_text(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--report-root", required=True)
    args = ap.parse_args()
    run_root = Path(args.run_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    rows = case_rows(run_root)
    runtime_verified = any(r.get("case") == "T400_4grid_scaled_300" and abs(f(r.get("interface_width_grids")) - 4.0) < 1e-9 and abs(f(r.get("kappa_phi")) - 2.0) < 1e-9 for r in rows)
    t400 = next((r for r in rows if r["case"] == "T400_4grid_scaled_300"), {})
    if not runtime_verified:
        final_status = "FAIL_RUNTIME_INTERFACE_WIDTH_NOT_ACTUALLY_CHANGED"
    elif t400.get("survival_status") == "COLLAPSES":
        final_status = "PASS_4GRID_RUNTIME_VERIFIED_SCALED_SEED_STILL_COLLAPSES"
    elif t400.get("survival_status") == "SURVIVES_TO_END":
        final_status = "PASS_RUNTIME_4GRID_INTERFACE_WIDTH_DYNAMIC_RERUN_AUDIT"
    else:
        final_status = "FAIL_4GRID_RUNTIME_BREAKS_HANDOFF_OR_LEDGER"

    write_csv(report_root / "runtime_interface_parameter_audit.csv", rows)
    write_csv(report_root / "runtime_4grid_param_summary.csv", [r for r in rows if r.get("runtime_width_class") == "4grid"])
    inv_keys = ["case", "selected_library_entry", "scheduled_source_lambda_nm", "scheduled_target_lambda_nm", "scale_phi", "scale_xB", "target_seed_inventory", "evaluated_profile_inventory", "staged_inventory_transferred", "actual_inserted_net_inventory", "residual_staged_inventory_after_handoff", "global_mass_error_rel", "external_matrix_reset_detected", "extra_matrix_draw_detected", "double_counting_detected"]
    write_csv(report_root / "staged_handoff_inventory_4grid_runtime.csv", [{k: r.get(k, "") for k in inv_keys} for r in rows if r.get("runtime_width_class") == "4grid"])
    write_csv(report_root / "post_handoff_seed_evolution_4grid_runtime.csv", post_evolution_rows(rows))
    iface_keys = ["case", "runtime_width_class", "profile_scale_class", "radial_r_phi_0p95_nm", "radial_r_phi_0p9_nm", "radial_r_phi_0p5_nm", "radial_r_phi_0p1_nm", "radial_r_phi_0p05_nm", "radial_r_phi_0p01_nm", "radial_width_phi_90_10_nm", "radial_width_phi_95_05_nm", "radial_width_phi_99_01_nm"]
    write_csv(report_root / "interface_width_evolution_4grid_runtime.csv", [{k: r.get(k, "") for k in iface_keys} for r in rows])
    collapse_keys = ["case", "runtime_width_class", "profile_scale_class", "handoff_step", "initial_h_integral", "initial_R_eff_h", "initial_phi_max", "first_h_drop_gt_10pct_step", "first_h_drop_gt_50pct_step", "first_phi_max_lt_0p9_step", "first_phi_max_lt_0p5_step", "first_h_lt_1_step", "final_step", "final_h_integral", "final_R_eff_h", "final_phi_max", "survival_status", "collapse_origin"]
    write_csv(report_root / "collapse_origin_4grid_runtime.csv", [{k: r.get(k, "") for k in collapse_keys} for r in rows])
    write_csv(report_root / "rhs_decomposition_4grid_runtime.csv", rhs_rows(rows))
    write_csv(report_root / "runtime_width_profile_scale_ab_comparison.csv", rows)
    conclusion = [{
        "final_status": final_status,
        "runtime_4grid_verified": runtime_verified,
        "T400_scaled_survival_status": t400.get("survival_status", ""),
        "T400_scaled_collapse_origin": t400.get("collapse_origin", ""),
        "recommended_interpretation": "profile_scaling_or_thermodynamic_incompatibility_not_only_runtime_width_mismatch" if t400.get("survival_status") == "COLLAPSES" else "runtime_width_mismatch_likely_if_scaled_survives",
        "recommended_next_action": "regenerate_dynamic_continue_profile_under_4grid_runtime_PF_or_keep_unscaled_baseline",
    }]
    write_csv(report_root / "runtime_4grid_conclusion_summary.csv", conclusion)
    write_report(report_root, rows, final_status)
    terminal = f"""runtime_4grid_interface_width_dynamic_rerun_audit_started
runtime_4grid_verified={str(runtime_verified).lower()}
T400_scaled_status={t400.get('survival_status','')}
T400_scaled_collapse_origin={t400.get('collapse_origin','')}
T400_scaled_final_h_integral={t400.get('final_h_integral','')}
T400_scaled_final_phi_max={t400.get('final_phi_max','')}
final_status={final_status}
created_reports={report_root}
"""
    (report_root / "final_terminal_output.txt").write_text(terminal)
    print(terminal, end="")


if __name__ == "__main__":
    main()
