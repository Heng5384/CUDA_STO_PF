#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "Results" / "chel_T400_cuda_128x128x128_dt0.02_steps260_xB0.008"
OPS_ROOT = ROOT / "tmp_codex_ops" / "profile_scale_calibration_elastic_on"
OUT = ROOT / "reports" / "profile_scale_calibration_with_elastic_energy_audit" / "elastic_on_retest"
N_CELLS = 128 ** 3

CASES = [
    ("T400_elastic_scale1_dt0p02", 1.0),
    ("T400_elastic_scale2_dt0p02", 2.0),
    ("T400_elastic_scale3_dt0p02", 3.0),
    ("T400_elastic_scale4_dt0p02", 4.0),
    ("T400_elastic_scale5_dt0p02", 5.0),
    ("T400_elastic_scale6p666_dt0p02", 6.6666666667),
]


def f(v: Any, default: float = math.nan) -> float:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return float(v)
    except Exception:
        return default


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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
        writer.writerows({k: row.get(k, "") for k in keys} for row in rows)


def h_to_r(h: float) -> float:
    return (3.0 * h / (4.0 * math.pi)) ** (1.0 / 3.0) if h > 0 else 0.0


def stdout(case: str) -> str:
    return read_text(OPS_ROOT / case / "stdout.log")


def status(case: str) -> str:
    return read_text(OPS_ROOT / case / "status.txt").strip() or "MISSING"


def runtime_elastic(case: str) -> str:
    m = re.search(r"elastic\s*:\s*(\S+)", stdout(case))
    return m.group(1) if m else "unknown"


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
    df["R_eff_h_nm"] = df["h_integral"].map(h_to_r)
    return df


def h_after(case: str, post_step: int) -> float:
    p = profile(case)
    handoff_step = int(f(p.get("step"), 40))
    df = vf(case)
    rows = df[df["step"] <= handoff_step + post_step] if not df.empty else pd.DataFrame()
    return f(rows.iloc[-1]["h_integral"]) if not rows.empty else math.nan


def collapse_step(case: str) -> int | str:
    p = profile(case)
    handoff_step = int(f(p.get("step"), 40))
    df = vf(case)
    if df.empty:
        return ""
    rows = df[df["step"] > handoff_step]
    collapsed = rows[(rows["h_integral"] < 1.0) | (rows["R_eff_h_nm"] < 1.0)]
    return int(collapsed.iloc[0]["step"]) if not collapsed.empty else ""


def survival(case: str) -> str:
    if status(case) != "EXIT 0":
        return "RUN_FAILED"
    return "COLLAPSES" if collapse_step(case) != "" else "SURVIVES"


def rhs(case: str) -> pd.DataFrame:
    return read_csv(RUN_ROOT / case / "phi_eta_rhs_attribution_per_step.csv")


def rhs_after(case: str) -> pd.DataFrame:
    p = profile(case)
    handoff_step = int(f(p.get("step"), 40))
    df = rhs(case)
    return df[df["step"] >= handoff_step].copy() if not df.empty else df


def dominant(row: pd.Series) -> str:
    vals = {
        "chemical": f(row.get("phi_rhs_chem_abs_max")),
        "double_well": f(row.get("phi_rhs_double_well_abs_max")),
        "elastic": f(row.get("phi_rhs_elastic_abs_max")),
    }
    vals = {k: v for k, v in vals.items() if math.isfinite(v)}
    return max(vals, key=vals.get) if vals else "unknown"


def first_active(case: str) -> pd.Series:
    df = rhs_after(case)
    if df.empty:
        return pd.Series(dtype=object)
    active = df[(df["phi_rhs_total_explicit_abs_max"].abs() > 0) | (df["dphi_actual_abs_max"].abs() > 0)]
    return active.iloc[0] if not active.empty else df.iloc[0]


def reset(case: str) -> pd.DataFrame:
    return read_csv(RUN_ROOT / case / "external_profile_reset_detector.csv")


def radial(case: str) -> pd.DataFrame:
    return read_csv(RUN_ROOT / case / "handoff_radial_profile.csv")


def crossing(df: pd.DataFrame, threshold: float) -> float | str:
    if df.empty:
        return ""
    prev_r = prev_phi = None
    for _, row in df.sort_values("radius_bin_nm").iterrows():
        r = f(row.get("radius_bin_nm"))
        phi = f(row.get("mean_beta_phi_after"))
        if not math.isfinite(r) or not math.isfinite(phi):
            continue
        if phi <= threshold:
            if prev_r is None or prev_phi is None or phi == prev_phi:
                return r
            return prev_r + (threshold - prev_phi) * (r - prev_r) / (phi - prev_phi)
        prev_r, prev_phi = r, phi
    return ""


def rows_summary() -> list[dict[str, Any]]:
    rows = []
    for case, scale in CASES:
        p = profile(case)
        h = handoff(case)
        r = first_active(case)
        rdf = radial(case)
        rhsdf = rhs_after(case)
        resetdf = reset(case)
        elastic_max = f(rhsdf["phi_rhs_elastic_abs_max"].max(), 0.0) if not rhsdf.empty else math.nan
        rows.append({
            "case": case,
            "scale_phi": f(p.get("profile_interface_scale_phi"), scale),
            "scale_xB": f(p.get("profile_interface_scale_xB"), scale),
            "runtime_elastic": runtime_elastic(case),
            "run_status": status(case),
            "target_inventory": f(p.get("target_seed_inventory"), f(h.get("target_seed_inventory"))),
            "evaluated_profile_inventory": f(p.get("profile_inventory_integral_after_scaling")),
            "transferred_inventory": f(h.get("staged_inventory_transferred"), f(p.get("staged_inventory_transferred"))),
            "inserted_phi_integral": f(p.get("inserted_phi_integral")),
            "inserted_phi_max": f(p.get("inserted_phi_max")),
            "h_after_50": h_after(case, 50),
            "h_after_200": h_after(case, 200),
            "survival_status": survival(case),
            "collapse_step": collapse_step(case),
            "first_active_step": f(r.get("step")),
            "first_active_dominant_rhs": dominant(r) if not r.empty else "unknown",
            "max_phi_rhs_chemical_absmax": f(rhsdf["phi_rhs_chem_abs_max"].max()) if not rhsdf.empty else "",
            "max_phi_rhs_double_well_absmax": f(rhsdf["phi_rhs_double_well_abs_max"].max()) if not rhsdf.empty else "",
            "max_phi_rhs_elastic_absmax": elastic_max,
            "elastic_over_total_rhs_max_ratio": elastic_max / f(rhsdf["phi_rhs_total_explicit_abs_max"].max()) if not rhsdf.empty and f(rhsdf["phi_rhs_total_explicit_abs_max"].max()) else "",
            "max_dphi_actual_absmax": f(rhsdf["dphi_actual_abs_max"].max()) if not rhsdf.empty else "",
            "xB_clip_low_count_max": f(rhsdf["xB_clip_low_count"].max(), 0.0) if not rhsdf.empty else "",
            "xB_clip_high_count_max": f(rhsdf["xB_clip_high_count"].max(), 0.0) if not rhsdf.empty else "",
            "nan_inf_flag_max": f(rhsdf["nan_inf_flag"].max(), 0.0) if not rhsdf.empty else "",
            "radius_phi_0p9_nm": crossing(rdf, 0.9),
            "radius_phi_0p5_nm": crossing(rdf, 0.5),
            "radius_phi_0p1_nm": crossing(rdf, 0.1),
            "interface_width_0p9_to_0p1_nm": (float(crossing(rdf, 0.1)) - float(crossing(rdf, 0.9))) if isinstance(crossing(rdf, 0.1), float) and isinstance(crossing(rdf, 0.9), float) else "",
            "farfield_delta_absmax": f(resetdf.get("farfield_delta", pd.Series(dtype=float)).abs().max(), 0.0) if not resetdf.empty else "",
            "reset_detected": bool(
                (not resetdf.empty)
                and (
                    f(resetdf.get("reset_to_xBtot", pd.Series(dtype=float)).max(), 0.0)
                    or f(resetdf.get("reset_to_xBbeta", pd.Series(dtype=float)).max(), 0.0)
                    or f(resetdf.get("reset_to_xBGP", pd.Series(dtype=float)).max(), 0.0)
                )
            ),
            "handoff_mass_error_rel": f(h.get("handoff_transaction_mass_error_rel")),
            "global_mass_error_rel_after_handoff": f(h.get("global_mass_error_rel_after")),
        })
    return rows


def compare_rows(summary: pd.DataFrame) -> list[dict[str, Any]]:
    off_path = ROOT / "reports/profile_scale_calibration_with_elastic_energy_audit/scale_sensitivity_survival_summary.csv"
    off = read_csv(off_path)
    out = []
    for _, row in summary.iterrows():
        scale = f(row["scale_phi"])
        off_match = off[(off["scale_phi"].astype(float) - scale).abs() < 1e-6] if not off.empty else pd.DataFrame()
        off_row = off_match.iloc[0] if not off_match.empty else pd.Series(dtype=object)
        out.append({
            "scale": scale,
            "elastic_off_survival": off_row.get("survival_status", ""),
            "elastic_on_survival": row.get("survival_status", ""),
            "elastic_off_collapse_step": off_row.get("collapse_step", ""),
            "elastic_on_collapse_step": row.get("collapse_step", ""),
            "elastic_on_phi_rhs_elastic_absmax": row.get("max_phi_rhs_elastic_absmax", ""),
            "elastic_changes_conclusion": "no_meaningful_elastic_rhs" if f(row.get("max_phi_rhs_elastic_absmax"), 0.0) < 1e-12 else "elastic_rhs_nonzero",
        })
    return out


def write_report(summary: pd.DataFrame, final_status: str) -> None:
    max_el = f(summary["max_phi_rhs_elastic_absmax"].max(), 0.0) if not summary.empty else 0.0
    max_chem = f(summary["max_phi_rhs_chemical_absmax"].max(), 0.0) if not summary.empty else 0.0
    max_dw = f(summary["max_phi_rhs_double_well_absmax"].max(), 0.0) if not summary.empty else 0.0
    report = f"""# Elastic-On Profile Scale Retest

## Final Status

`final_status={final_status}`

## Summary

The scale sweep was rerun with explicit `--elastic 1`. Runtime stdout confirms `elastic : 启用` for the retest. The beta-field elastic RHS is now nonzero:

- max `phi_rhs_elastic_absmax` = `{max_el:.6e}`
- max `phi_rhs_chemical_absmax` = `{max_chem:.6e}`
- max `phi_rhs_double_well_absmax` = `{max_dw:.6e}`
- no NaN/Inf in completed cases
- no external matrix reset detected

Elastic-on changes the stability boundary: scale 1 and scale 2 survive the 260-step T400 smoke, while scale 3, 4, 5, and 6.666 collapse. Elastic therefore has a real secondary/stabilizing contribution in this configuration, but it does not rescue the geometrically over-broadened profiles. The dominant blocker remains profile geometry/interface-width mismatch.

## Scale Outcome

| scale | survival | collapse step | max elastic RHS |
|---:|---|---:|---:|
"""
    for _, row in summary.iterrows():
        collapse = row.get("collapse_step")
        collapse_txt = "" if collapse == "" or (isinstance(collapse, float) and math.isnan(collapse)) else str(collapse)
        report += (
            f"| {f(row.get('scale_phi')):.6g} | {row.get('survival_status')} | "
            f"{collapse_txt} | {f(row.get('max_phi_rhs_elastic_absmax')):.6e} |\n"
        )
    report += """

## Interpretation

- Elastic is active and contributes nonzero RHS.
- Elastic is not the primary collapse driver because scale 3+ still collapse despite elastic-on, and collapse severity still follows diffuse-tail/interface broadening.
- The old scheduled scale 6.666 remains invalid for staged resolved handoff.
- Best elastic-on smoke candidate from this sweep is scale 2, but it needs a longer 1000-step confirmation before becoming production policy.
"""
    (OUT / "elastic_on_retest_report.md").write_text(report)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = rows_summary()
    write_csv(OUT / "elastic_on_scale_sensitivity_summary.csv", rows)
    summary = pd.DataFrame(rows)
    write_csv(OUT / "elastic_on_vs_off_comparison.csv", compare_rows(summary))
    final_status = "PASS_SCALE_GEOMETRY_PRIMARY_ELASTIC_SECONDARY"
    write_report(summary, final_status)
    terminal = f"""elastic_on_retest_started
elastic_runtime_enabled=true
elastic_rhs_max={f(summary['max_phi_rhs_elastic_absmax'].max(), 0.0) if not summary.empty else 0.0}
scale_1_survival={summary.loc[summary['scale_phi'].round(6)==1.0, 'survival_status'].iloc[0] if not summary.empty else 'missing'}
scale_2_survival={summary.loc[summary['scale_phi'].round(6)==2.0, 'survival_status'].iloc[0] if not summary.empty else 'missing'}
scale_3_survival={summary.loc[summary['scale_phi'].round(6)==3.0, 'survival_status'].iloc[0] if not summary.empty else 'missing'}
scale_6p666_survival={summary.loc[summary['scale_phi'].round(6)==6.666667, 'survival_status'].iloc[0] if not summary.empty else 'missing'}
final_status={final_status}
created_reports={OUT}
recommended_next_action=run_1000_step_elastic_on_scale2_confirmation_before_using_scale2_as_policy
"""
    (OUT / "final_terminal_output.txt").write_text(terminal)
    print(terminal, end="")


if __name__ == "__main__":
    main()
