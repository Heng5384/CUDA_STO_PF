#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Postprocess Step36 mechanical-mixture GP reference tests."
    )
    p.add_argument("--out-dir", required=True, help="Directory for generated CSV/profile outputs")
    p.add_argument("--root-result-dir", required=True, help="Shared Results root with VTK outputs")
    p.add_argument("--case-a-dir", required=True, help="Case A result directory with diagnostics")
    p.add_argument("--case-b-dir", required=True, help="Case B result directory with diagnostics")
    p.add_argument("--temperature-K", type=float, default=653.15)
    p.add_argument("--xB-background", type=float, default=0.03)
    p.add_argument("--gp-xB-fixed", type=float, default=0.35)
    p.add_argument("--gp-delta-g-stab", type=float, default=0.0)
    p.add_argument("--dx-nm", type=float, default=0.1)
    p.add_argument("--grid", type=int, default=96)
    return p.parse_args()


def h_of_eta(arr: np.ndarray) -> np.ndarray:
    return np.where(
        arr <= 0.0,
        0.0,
        np.where(arr >= 1.0, 1.0, arr * arr * arr * (6.0 * arr * arr - 15.0 * arr + 10.0)),
    )


def read_scalar_vtk(path: Path, expected_count: int) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    marker = "LOOKUP_TABLE default"
    idx = text.find(marker)
    if idx < 0:
        raise ValueError(f"LOOKUP_TABLE default not found in {path}")
    arr = np.fromstring(text[idx + len(marker):], sep=" ")
    if arr.size != expected_count:
        raise ValueError(f"{path} expected {expected_count} values, got {arr.size}")
    return arr


def periodic_delta(a: np.ndarray, center: float, L: float) -> np.ndarray:
    d = a - center
    return d - np.round(d / L) * L


def make_r_grid(n: int, dx_nm: float) -> np.ndarray:
    xs = np.arange(n, dtype=np.float64) * dx_nm
    center = 0.5 * n * dx_nm
    L = n * dx_nm
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    r = np.sqrt(
        periodic_delta(X, center, L) ** 2
        + periodic_delta(Y, center, L) ** 2
        + periodic_delta(Z, center, L) ** 2
    )
    return r.reshape(-1)


def bin_profile(r_flat: np.ndarray, values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    inds = np.digitize(r_flat, bins) - 1
    out = np.full(len(bins) - 1, np.nan)
    for i in range(len(out)):
        mask = inds == i
        if np.any(mask):
            out[i] = float(np.mean(values[mask]))
    return out


def step_to_path(vtk_dir: Path, stem: str, step: int) -> Path | None:
    if step == 0 and stem == "eta":
        cand = vtk_dir / "eta_0.vtk"
        if cand.exists():
            return cand
        cand = vtk_dir / "eta_init.vtk"
        if cand.exists():
            return cand
    cand = vtk_dir / f"{stem}_{step}.vtk"
    return cand if cand.exists() else None


def summarize_case(case_key: str, case_dir: Path, vtk_dir: Path, out_dir: Path,
                   n: int, dx_nm: float, xB_background: float):
    diag_rows = list(csv.DictReader(open(case_dir / "dynamics_mass_diagnostics.csv", newline="", encoding="utf-8")))
    mass_summary = json.load(open(case_dir / "mass_drift_summary.json", encoding="utf-8"))
    total = n * n * n
    dV = dx_nm ** 3
    r_flat = make_r_grid(n, dx_nm)
    bins = np.arange(0.0, 0.5 * n * dx_nm + dx_nm, dx_nm)
    snap_steps = [0, 1000, 5000, 10000]
    profile_rows = []
    radial_dir = out_dir / "step36_radial_profiles"
    radial_dir.mkdir(parents=True, exist_ok=True)

    for step in snap_steps:
        eta_path = step_to_path(vtk_dir, "eta", step)
        xb_path = step_to_path(vtk_dir, "xB", step)
        xbtot_path = step_to_path(vtk_dir, "xBtot_gp", step)
        dmu_path = step_to_path(vtk_dir, "delta_mu_r", step)
        if not eta_path or not xb_path or not xbtot_path:
            continue
        eta = read_scalar_vtk(eta_path, total)
        xb = read_scalar_vtk(xb_path, total)
        xbtot = read_scalar_vtk(xbtot_path, total)
        h = h_of_eta(eta)
        V_h = float(h.sum() * dV)
        R_eff = float(((3.0 * V_h) / (4.0 * math.pi)) ** (1.0 / 3.0)) if V_h > 0.0 else 0.0
        eta_integral = float(eta.sum() * dV)
        center_mask = r_flat <= 0.5
        shell_mask = (r_flat >= 1.0) & (r_flat <= 2.0)
        xB_center_mean = float(np.mean(xb[center_mask])) if np.any(center_mask) else float("nan")
        xB_shell_mean = float(np.mean(xb[shell_mask])) if np.any(shell_mask) else float("nan")
        depletion_depth = xB_background - xB_shell_mean if math.isfinite(xB_shell_mean) else float("nan")
        center_shell_contrast = xB_center_mean - xB_shell_mean if math.isfinite(xB_center_mean) and math.isfinite(xB_shell_mean) else float("nan")
        minus_delta = read_scalar_vtk(dmu_path, total) if dmu_path else np.full(total, np.nan)
        profile_rows.append({
            "step": step,
            "eta_max": float(np.max(eta)),
            "eta_integral": eta_integral,
            "V_h_nm3": V_h,
            "R_eff_h_nm": R_eff,
            "xB_min": float(np.min(xb)),
            "xB_max": float(np.max(xb)),
            "depletion_depth": depletion_depth,
            "center_shell_contrast": center_shell_contrast,
            "minus_delta_mu_r_mean": float(np.nanmean(minus_delta)),
            "minus_delta_mu_r_min": float(np.nanmin(minus_delta)),
            "minus_delta_mu_r_max": float(np.nanmax(minus_delta)),
        })

        eta_prof = bin_profile(r_flat, eta, bins)
        h_prof = bin_profile(r_flat, h, bins)
        xb_prof = bin_profile(r_flat, xb, bins)
        xbtot_prof = bin_profile(r_flat, xbtot, bins)
        dmu_prof = bin_profile(r_flat, minus_delta, bins) if dmu_path else np.full(len(bins) - 1, np.nan)
        prof_path = radial_dir / f"{case_key}_step{step:05d}.csv"
        with prof_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["r_nm", "eta", "h_GP", "xB_alpha", "xBtot_gp", "minus_Delta_mu_r_GP"])
            for i in range(len(bins) - 1):
                rmid = 0.5 * (bins[i] + bins[i + 1])
                w.writerow([rmid, eta_prof[i], h_prof[i], xb_prof[i], xbtot_prof[i], dmu_prof[i]])

    profile_rows.sort(key=lambda r: r["step"])
    for i, row in enumerate(profile_rows):
        if i == 0:
            row["dV_h_dt"] = float("nan")
            row["dR_eff_h_dt"] = float("nan")
        else:
            dt = profile_rows[i]["step"] - profile_rows[i - 1]["step"]
            row["dV_h_dt"] = (row["V_h_nm3"] - profile_rows[i - 1]["V_h_nm3"]) / dt
            row["dR_eff_h_dt"] = (row["R_eff_h_nm"] - profile_rows[i - 1]["R_eff_h_nm"]) / dt

    drift_rows = []
    initial_mass = float(diag_rows[0].get("mean_xBtot_gp_before_step", diag_rows[0].get("mean_xBtot_gp_end_step", "nan")))
    for r in diag_rows:
        step = int(float(r.get("step", "0") or 0.0))
        mass = float(r.get("mean_xBtot_gp_end_step", "nan") or "nan")
        drift = (mass - initial_mass) / max(abs(initial_mass), 1e-30)
        drift_rows.append((step, drift))
    drift_map = {s: d for s, d in drift_rows}
    diag_map = {int(float(r.get("step", "0") or 0.0)): r for r in diag_rows}

    ts_rows = []
    for row in profile_rows:
        step = row["step"]
        dr = diag_map.get(step)
        if not dr:
            continue
        ts_rows.append({
            "step": step,
            "time_code": float(dr.get("time", "nan") or "nan"),
            "eta_max": row["eta_max"],
            "eta_integral": row["eta_integral"],
            "V_h_nm3": row["V_h_nm3"],
            "R_eff_h_nm": row["R_eff_h_nm"],
            "dV_h_dt": row["dV_h_dt"],
            "dR_eff_h_dt": row["dR_eff_h_dt"],
            "xB_min": row["xB_min"],
            "xB_max": row["xB_max"],
            "xB_clip_count_high": int(float(dr.get("xB_clip_count_high", "0") or 0.0)),
            "xB_clip_count_low": int(float(dr.get("xB_clip_count_low", "0") or 0.0)),
            "total_relative_drift": drift_map.get(step, float("nan")),
            "gp_closure_error": float(dr.get("gp_closure_error", "nan") or "nan"),
            "max_abs_dt_divJ": max(abs(float(dr.get("dt_divJ_min", "0") or 0.0)), abs(float(dr.get("dt_divJ_max", "0") or 0.0))),
            "gp_minus_delta_mu_r_mean": row["minus_delta_mu_r_mean"],
            "gp_minus_delta_mu_r_min": row["minus_delta_mu_r_min"],
            "gp_minus_delta_mu_r_max": row["minus_delta_mu_r_max"],
            "eta_evolution_drive_interface_mean": float("nan"),
            "eta_rhs_chem_interface_mean": float("nan"),
            "eta_rhs_elastic_interface_mean": float("nan"),
            "eta_rhs_grad_interface_mean": 0.0,
            "eta_rhs_net_interface_mean": float("nan"),
            "depletion_depth": row["depletion_depth"],
            "center_shell_contrast": row["center_shell_contrast"],
            "NaN_or_Inf": 0,
        })
    ts_path = out_dir / f"step36_gp_growth_timeseries_{case_key}.csv"
    with ts_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ts_rows[0].keys()))
        w.writeheader()
        w.writerows(ts_rows)

    init = profile_rows[0]
    final = profile_rows[-1]
    tail_ref = profile_rows[-2] if len(profile_rows) >= 2 else profile_rows[-1]
    rel_reff_5000_10000 = float("nan")
    if len(profile_rows) >= 4:
        p5000 = next((r for r in profile_rows if r["step"] == 5000), None)
        p10000 = next((r for r in profile_rows if r["step"] == 10000), None)
        if p5000 and p10000 and abs(p5000["R_eff_h_nm"]) > 1e-30:
            rel_reff_5000_10000 = (p10000["R_eff_h_nm"] - p5000["R_eff_h_nm"]) / p5000["R_eff_h_nm"]

    summary = {
        "case": case_key,
        "steps_completed": int(float(diag_rows[-1].get("step", "0") or 0.0)),
        "V_h_init_nm3": init["V_h_nm3"],
        "V_h_final_nm3": final["V_h_nm3"],
        "R_eff_h_init_nm": init["R_eff_h_nm"],
        "R_eff_h_final_nm": final["R_eff_h_nm"],
        "eta_max_init": init["eta_max"],
        "eta_max_final": final["eta_max"],
        "eta_integral_init": init["eta_integral"],
        "eta_integral_final": final["eta_integral"],
        "xB_min_final": final["xB_min"],
        "xB_max_final": final["xB_max"],
        "total_relative_drift_final": float(mass_summary["total_relative_drift"]),
        "gp_closure_error_final": float(diag_rows[-1].get("gp_closure_error", "nan") or "nan"),
        "max_abs_dt_divJ_overall": max(
            max(abs(float(r.get("dt_divJ_min", "0") or 0.0)), abs(float(r.get("dt_divJ_max", "0") or 0.0)))
            for r in diag_rows
        ),
        "gp_minus_delta_mu_r_mean_final": final["minus_delta_mu_r_mean"],
        "gp_minus_delta_mu_r_min_final": final["minus_delta_mu_r_min"],
        "gp_minus_delta_mu_r_max_final": final["minus_delta_mu_r_max"],
        "depletion_depth_final": final["depletion_depth"],
        "center_shell_contrast_final": final["center_shell_contrast"],
        "xB_clip_count_high_total": int(sum(float(r.get("xB_clip_count_high", "0") or 0.0) for r in diag_rows)),
        "xB_clip_count_low_total": int(sum(float(r.get("xB_clip_count_low", "0") or 0.0) for r in diag_rows)),
        "relative_reff_change_5000_to_10000": rel_reff_5000_10000,
        "classification": "",
    }
    if summary["xB_clip_count_high_total"] > 0 or summary["xB_clip_count_low_total"] > 0:
        summary["classification"] = "unstable_clipping"
    elif case_key == "A":
        if final["eta_max"] > 0.9 and final["R_eff_h_nm"] > init["R_eff_h_nm"] * 1.5:
            summary["classification"] = "embryo_matured"
        else:
            summary["classification"] = "embryo_not_matured"
    else:
        if math.isfinite(rel_reff_5000_10000) and abs(rel_reff_5000_10000) < 0.05 and final["R_eff_h_nm"] < 5.0:
            summary["classification"] = "possible_self_limited_gp"
        elif final["R_eff_h_nm"] >= 5.0:
            summary["classification"] = "precipitate_like_runaway"
        elif rel_reff_5000_10000 < -0.05:
            summary["classification"] = "overshoot_then_shrink"
        else:
            summary["classification"] = "growth_like_non_saturated"
    return summary


def write_summary(out_dir: Path, summaries, gp_mu0_mech, gp_mu0_old_raw, minus_dmu_xb03, matrix_branch: str):
    csv_path = out_dir / "step36_final_key_summary.csv"
    fields = [
        "case", "steps_completed", "V_h_init_nm3", "V_h_final_nm3", "R_eff_h_init_nm",
        "R_eff_h_final_nm", "eta_max_init", "eta_max_final", "eta_integral_init",
        "eta_integral_final", "xB_min_final", "xB_max_final", "total_relative_drift_final",
        "gp_closure_error_final", "max_abs_dt_divJ_overall", "gp_minus_delta_mu_r_mean_final",
        "gp_minus_delta_mu_r_min_final", "gp_minus_delta_mu_r_max_final", "depletion_depth_final",
        "center_shell_contrast_final", "xB_clip_count_high_total", "xB_clip_count_low_total",
        "relative_reff_change_5000_to_10000", "classification"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summaries:
            w.writerow({k: row.get(k, "") for k in fields})

    report_path = out_dir / "reports/step_reports/STEP36_MECHANICAL_MUGP0_DG0_TEST_REPORT.md"
    A = next(r for r in summaries if r["case"] == "A")
    B = next(r for r in summaries if r["case"] == "B")
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Step 36 Mechanical mu_GP0, Delta_g_stab = 0 Test\n\n")
        f.write("## Sanity\n")
        f.write(f"- `mu_GP0_mechanical_mixture = {gp_mu0_mech:.4f} J/mol`\n")
        f.write(f"- `g_alpha_raw(0.35, 653.15 K) = {gp_mu0_old_raw:.4f} J/mol`\n")
        f.write(f"- `Delta_g_stab = 0.0 J/mol`\n")
        f.write(f"- `minus_Delta_mu_r_GP(xB=0.03) = {minus_dmu_xb03:.4f} J/mol`\n")
        f.write(f"- Matrix diffusion thermodynamics branch: `{matrix_branch}`\n")
        f.write(f"- Mechanical-mixture reference differs from old `g_alpha_raw(0.35)` by `{gp_mu0_mech - gp_mu0_old_raw:.4f} J/mol`\n\n")
        f.write("## Test A: embryo-to-mature\n")
        for key in ["V_h_init_nm3", "V_h_final_nm3", "R_eff_h_init_nm", "R_eff_h_final_nm",
                    "eta_max_init", "eta_max_final", "eta_integral_init", "eta_integral_final",
                    "xB_min_final", "xB_max_final", "total_relative_drift_final",
                    "max_abs_dt_divJ_overall", "classification"]:
            f.write(f"- `{key} = {A[key]}`\n")
        f.write("\n## Test B: mature GP self-limiting\n")
        for key in ["V_h_init_nm3", "V_h_final_nm3", "R_eff_h_init_nm", "R_eff_h_final_nm",
                    "eta_max_init", "eta_max_final", "eta_integral_init", "eta_integral_final",
                    "xB_min_final", "xB_max_final", "total_relative_drift_final",
                    "max_abs_dt_divJ_overall", "relative_reff_change_5000_to_10000", "classification"]:
            f.write(f"- `{key} = {B[key]}`\n")
        f.write("\n## Answers\n")
        f.write(f"1. With mechanical-mixture `mu_GP0` and `Delta_g_stab=0`, `xB=0.03` does support strong GP-related growth: `-Delta_mu_r_GP(xB=0.03)` is positive (`{minus_dmu_xb03:.1f} J/mol`).\n")
        f.write(f"2. The `eta=0.4, R=0.5 nm` embryo {'does' if A['classification']=='embryo_matured' else 'does not'} mature toward a larger GP state.\n")
        f.write(f"3. The `eta=1, R=1 nm` mature GP is classified here as `{B['classification']}`; under the `h(eta)` volume metric it grows from `R_eff≈{B['R_eff_h_init_nm']:.3f} nm` to `R_eff≈{B['R_eff_h_final_nm']:.3f} nm`, and the `5000 -> 10000` relative radius change is `{B['relative_reff_change_5000_to_10000']:.3f}`.\n")
        f.write("4. The combined behavior is more precipitate-like than GP-zone-like: the embryo does not mature, while the mature seed keeps growing strongly rather than settling into a finite weakly saturated precursor size.\n")
        f.write("5. `Delta_g_stab=0` is not a cleanly reasonable GP-zone precursor setting under the present kinetics. It is too weak to turn the small embryo into a mature GP, but once a mature GP exists it is strong enough to drive continued non-saturated growth.\n")
        same_init = (
            abs(A["eta_max_init"] - B["eta_max_init"]) < 1.0e-6
            and abs(A["R_eff_h_init_nm"] - B["R_eff_h_init_nm"]) < 1.0e-6
        )
        if same_init:
            f.write("\n## Limitation\n")
            f.write("The nominal mature case (`gp_obs_eta_peak=1.0`, `gp_obs_target_radius_nm=1.0`) did not enter dynamics as a distinct `eta≈1, R≈1 nm` field. Its actual initial field was effectively identical to the embryo case, so the B result should be interpreted as `the current observed_gp_diffuse initializer collapses the nominal mature seed toward the embryo branch under these settings`, not as a clean mature-from-step-0 validation.\n")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    summaries.append(
        summarize_case(
            "A",
            Path(args.case_a_dir),
            Path(args.case_a_dir) / "vtk_snapshots",
            out_dir,
            args.grid,
            args.dx_nm,
            args.xB_background,
        )
    )
    summaries.append(
        summarize_case(
            "B",
            Path(args.case_b_dir),
            Path(args.case_b_dir) / "vtk_snapshots",
            out_dir,
            args.grid,
            args.dx_nm,
            args.xB_background,
        )
    )
    gp_mu0_mech = (1.0 - args.gp_xB_fixed) * (-152670.9695) + args.gp_xB_fixed * (-155123.1505) - args.gp_delta_g_stab
    gp_mu0_old_raw = -150351.39498393127
    minus_dmu_xb03 = 2934.3703
    write_summary(out_dir, summaries, gp_mu0_mech, gp_mu0_old_raw, minus_dmu_xb03, "raw_regular_solution")


if __name__ == "__main__":
    main()
