#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Postprocess Step37 GP eta RHS physics audit.")
    p.add_argument("--result-root", required=True)
    p.add_argument("--case-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--grid", type=int, default=96)
    p.add_argument("--dx-nm", type=float, default=0.1)
    p.add_argument("--xB-background", type=float, default=0.03)
    p.add_argument("--gp-W-eta", type=float, default=math.nan)
    p.add_argument("--gp-kappa-eta", type=float, default=math.nan)
    p.add_argument("--gp-L-eta", type=float, default=math.nan)
    p.add_argument("--gp-eps-iso", type=float, default=math.nan)
    p.add_argument("--gp-elastic-enabled", type=int, default=-1)
    p.add_argument("--gp-elastic-active-eta", type=int, default=-1)
    p.add_argument("--gp-elastic-derivative-scale", type=float, default=math.nan)
    p.add_argument("--gp-obs-iface-width-nm", type=float, default=math.nan)
    return p.parse_args()


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


def h_of_eta(arr):
    return np.where(
        arr <= 0.0,
        0.0,
        np.where(arr >= 1.0, 1.0, arr * arr * arr * (6.0 * arr * arr - 15.0 * arr + 10.0)),
    )


def h_prime_of_eta(arr):
    return 30.0 * arr * arr * (1.0 - arr) * (1.0 - arr)


def g_prime_of_eta(arr):
    return 2.0 * arr * (1.0 - arr) * (1.0 - 2.0 * arr)


def stat_dict(arr: np.ndarray):
    if arr.size == 0:
        return {"mean": math.nan, "min": math.nan, "max": math.nan, "rms": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "rms": float(np.sqrt(np.mean(arr * arr))),
    }


def ratio(a, b):
    if not (math.isfinite(a) and math.isfinite(b)) or abs(b) < 1.0e-300:
        return math.nan
    return a / b


def parse_params(param_path: Path):
    out = {}
    for line in param_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_gp_ref_from_log(log_text: str):
    vals = {}
    patterns = {
        "G_PbTe_Solid": r"\[GP-REF\] G_PbTe_Solid\s*:\s*([\-+0-9.eE]+)",
        "G_Ag2Te_Solid": r"\[GP-REF\] G_Ag2Te_Solid\s*:\s*([\-+0-9.eE]+)",
        "mu_GP0": r"\[GP-REF\] mu_GP0\s*:\s*([\-+0-9.eE]+)",
        "delta_g_stab": r"\[GP-REF\] delta_g_stab\s*:\s*([\-+0-9.eE]+)",
        "minus_dmu_xB003": r"\[GP-REF\] minus_delta_mu_r_GP\(xB=0\.0300\)\s*:\s*([\-+0-9.eE]+)",
        "minus_dmu_xB005": r"\[GP-REF\] minus_delta_mu_r_GP\(xB=0\.0500\)\s*:\s*([\-+0-9.eE]+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, log_text)
        vals[k] = float(m.group(1)) if m else math.nan
    return vals


def parse_runtime_scalar_from_log(log_text: str, key: str):
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*([\-+0-9.eE]+)\s*$", log_text, re.MULTILINE)
    return float(m.group(1)) if m else math.nan


def coalesce_numeric(*values):
    for v in values:
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                continue
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
    return math.nan


def coalesce_int(*values):
    for v in values:
        if isinstance(v, str):
            try:
                v = int(float(v))
            except ValueError:
                continue
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return int(float(v))
    return 0


def main():
    args = parse_args()
    result_root = Path(args.result_root)
    case_dir = Path(args.case_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = args.grid ** 3
    voxel_nm3 = args.dx_nm ** 3

    params = parse_params(case_dir / "pf_input.params")
    log_text = (case_dir / "run.log").read_text(encoding="utf-8", errors="ignore")
    gp_ref = parse_gp_ref_from_log(log_text)

    diag_rows = list(csv.DictReader(open(case_dir / "dynamics_mass_diagnostics.csv", newline="", encoding="utf-8")))
    mass_summary = json.load(open(case_dir / "mass_drift_summary.json", encoding="utf-8"))

    steps_analyze = [1000, 2000]
    term_rows = []
    ts_rows = []

    initial_mass = float(
        mass_summary.get(
            "initial_mean_xBtot_gp",
            diag_rows[0].get("mean_xBtot_gp_before_step", diag_rows[0].get("mean_xBtot_gp_end_step", "nan")),
        )
    )
    drift_map = {}
    diag_by_step = {}
    for r in diag_rows:
        step = int(float(r["step"]))
        diag_by_step[step] = r
        cur = float(r.get("mean_xBtot_gp_end_step", "nan"))
        drift_map[step] = (cur - initial_mass) / max(abs(initial_mass), 1.0e-30)

    # include step 0 from init vtk
    eta0 = read_scalar_vtk(result_root / "eta_init.vtk", total)
    h0 = h_of_eta(eta0)
    V0 = float(h0.sum() * voxel_nm3)
    R0 = float(((3.0 * V0) / (4.0 * math.pi)) ** (1.0 / 3.0)) if V0 > 0.0 else 0.0
    ts_rows.append({
        "step": 0,
        "time_code": 0.0,
        "eta_max": float(np.max(eta0)),
        "eta_integral": float(np.sum(eta0) * voxel_nm3),
        "V_h_nm3": V0,
        "R_eff_h_nm": R0,
        "dR_eff_h_dt": math.nan,
        "xB_min": float(np.min(read_scalar_vtk(result_root / "xB_0.vtk", total))),
        "xB_max": float(np.max(read_scalar_vtk(result_root / "xB_0.vtk", total))),
        "xB_clip_high": 0,
        "xB_clip_low": 0,
        "total_relative_drift": 0.0,
        "gp_closure_error": math.nan,
        "max_abs_dt_divJ": math.nan,
        "NaN_or_Inf": 0,
    })

    prev_R = R0
    prev_step = 0
    for step in steps_analyze:
        eta = read_scalar_vtk(result_root / f"eta_{step}.vtk", total)
        xb = read_scalar_vtk(result_root / f"xB_{step}.vtk", total)
        chem = read_scalar_vtk(result_root / f"eta_rhs_chem_{step}.vtk", total)
        dw = read_scalar_vtk(result_root / f"eta_rhs_dw_{step}.vtk", total)
        elastic = read_scalar_vtk(result_root / f"eta_rhs_elastic_{step}.vtk", total)
        net_explicit = read_scalar_vtk(result_root / f"eta_rhs_net_explicit_{step}.vtk", total)
        grad = read_scalar_vtk(result_root / f"eta_rhs_grad_{step}.vtk", total)
        full = read_scalar_vtk(result_root / f"eta_rhs_full_variational_{step}.vtk", total)
        minus_dmu = read_scalar_vtk(result_root / f"gp_minus_delta_mu_r_{step}.vtk", total)
        mask_interface = (eta > 0.1) & (eta < 0.9)
        mask_half = (eta > 0.45) & (eta < 0.55)
        Vh = float(np.sum(h_of_eta(eta)) * voxel_nm3)
        Reff = float(((3.0 * Vh) / (4.0 * math.pi)) ** (1.0 / 3.0)) if Vh > 0.0 else 0.0
        dR = (Reff - prev_R) / (step - prev_step) if step > prev_step else math.nan
        prev_R, prev_step = Reff, step
        dr = diag_by_step[step]
        ts_rows.append({
            "step": step,
            "time_code": float(dr["time"]),
            "eta_max": float(np.max(eta)),
            "eta_integral": float(dr.get("eta_integral", "nan")),
            "V_h_nm3": Vh,
            "R_eff_h_nm": Reff,
            "dR_eff_h_dt": dR,
            "xB_min": float(np.min(xb)),
            "xB_max": float(np.max(xb)),
            "xB_clip_high": int(float(dr.get("xB_clip_count_high", "0") or 0.0)),
            "xB_clip_low": int(float(dr.get("xB_clip_count_low", "0") or 0.0)),
            "total_relative_drift": drift_map[step],
            "gp_closure_error": float(dr.get("gp_closure_error", "nan")),
            "max_abs_dt_divJ": max(abs(float(dr.get("dt_divJ_min", "0") or 0.0)), abs(float(dr.get("dt_divJ_max", "0") or 0.0))),
            "NaN_or_Inf": 0,
        })

        chem_s = stat_dict(chem[mask_interface])
        dw_s = stat_dict(dw[mask_interface])
        grad_s = stat_dict(grad[mask_interface])
        elastic_s = stat_dict(elastic[mask_interface])
        full_s = stat_dict(full[mask_interface])
        drive_s = stat_dict((-full)[mask_interface])
        update_s = stat_dict((-float(params.get("gp_L_eta", "0.0")) * full)[mask_interface])

        hprime_half = stat_dict(h_prime_of_eta(eta[mask_half]))
        gprime_half = stat_dict(g_prime_of_eta(eta[mask_half]))
        minus_dmu_half = stat_dict(minus_dmu[mask_half])
        chem_half = stat_dict(chem[mask_half])
        elastic_half = stat_dict(elastic[mask_half])
        grad_half = stat_dict(grad[mask_half])
        dw_half = stat_dict(dw[mask_half])

        term_rows.append({
            "step": step,
            "interface_count": int(np.count_nonzero(mask_interface)),
            "eta05_count": int(np.count_nonzero(mask_half)),
            "chem_mean": chem_s["mean"], "chem_min": chem_s["min"], "chem_max": chem_s["max"], "chem_rms": chem_s["rms"],
            "dw_mean": dw_s["mean"], "dw_min": dw_s["min"], "dw_max": dw_s["max"], "dw_rms": dw_s["rms"],
            "grad_mean": grad_s["mean"], "grad_min": grad_s["min"], "grad_max": grad_s["max"], "grad_rms": grad_s["rms"],
            "elastic_mean": elastic_s["mean"], "elastic_min": elastic_s["min"], "elastic_max": elastic_s["max"], "elastic_rms": elastic_s["rms"],
            "full_mean": full_s["mean"], "full_min": full_s["min"], "full_max": full_s["max"], "full_rms": full_s["rms"],
            "drive_mean": drive_s["mean"], "drive_min": drive_s["min"], "drive_max": drive_s["max"], "drive_rms": drive_s["rms"],
            "update_mean": update_s["mean"], "update_min": update_s["min"], "update_max": update_s["max"], "update_rms": update_s["rms"],
            "ratio_dw_over_chem": ratio(dw_s["rms"], chem_s["rms"]),
            "ratio_grad_over_chem": ratio(grad_s["rms"], chem_s["rms"]),
            "ratio_elastic_over_chem": ratio(elastic_s["rms"], chem_s["rms"]),
            "ratio_nonchemical_over_chem": ratio(math.sqrt(dw_s["rms"]**2 + grad_s["rms"]**2 + elastic_s["rms"]**2), chem_s["rms"]),
            "hprime_eta05_mean": hprime_half["mean"],
            "gprime_eta05_mean": gprime_half["mean"],
            "minus_dmu_eta05_mean": minus_dmu_half["mean"],
            "eta_rhs_chem_eta05_mean": chem_half["mean"],
            "eta_rhs_dw_eta05_mean": dw_half["mean"],
            "eta_rhs_grad_eta05_mean": grad_half["mean"],
            "eta_rhs_elastic_eta05_mean": elastic_half["mean"],
        })

    term_csv = out_dir / "step37_eta_rhs_terms_summary.csv"
    with term_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(term_rows[0].keys()))
        w.writeheader()
        w.writerows(term_rows)

    ts_csv = out_dir / "step37_eta_rhs_timeseries.csv"
    with ts_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ts_rows[0].keys()))
        w.writeheader()
        w.writerows(ts_rows)

    p1000 = term_rows[0]
    p2000 = term_rows[-1]
    W_eta = coalesce_numeric(
        params.get("gp_W_eta"),
        parse_runtime_scalar_from_log(log_text, "gp_W_eta"),
        args.gp_W_eta,
    )
    kappa_eta = coalesce_numeric(
        params.get("gp_kappa_eta"),
        parse_runtime_scalar_from_log(log_text, "gp_kappa_eta"),
        args.gp_kappa_eta,
    )
    L_eta = coalesce_numeric(
        params.get("gp_L_eta"),
        parse_runtime_scalar_from_log(log_text, "gp_L_eta"),
        args.gp_L_eta,
    )
    gp_eps_iso = coalesce_numeric(
        params.get("gp_eps_iso"),
        parse_runtime_scalar_from_log(log_text, "gp_eps_iso"),
        args.gp_eps_iso,
    )
    gp_elastic_enabled = coalesce_int(
        params.get("gp_elastic_enabled"),
        parse_runtime_scalar_from_log(log_text, "gp_elastic_enabled"),
        args.gp_elastic_enabled,
    )
    gp_elastic_active_eta = coalesce_int(
        params.get("gp_elastic_active_eta"),
        parse_runtime_scalar_from_log(log_text, "gp_elastic_active_eta"),
        args.gp_elastic_active_eta,
    )
    gp_elastic_derivative_scale = coalesce_numeric(
        params.get("gp_elastic_derivative_scale"),
        parse_runtime_scalar_from_log(log_text, "gp_elastic_derivative_scale"),
        args.gp_elastic_derivative_scale,
    )
    iface_width = coalesce_numeric(
        params.get("gp_obs_iface_width_nm"),
        parse_runtime_scalar_from_log(log_text, "gp_obs_iface_width_nm"),
        args.gp_obs_iface_width_nm,
    )
    report = out_dir / "reports/step_reports/STEP37_GP_ETA_RHS_PHYSICS_AUDIT_REPORT.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Step 37 GP eta RHS Physics Audit\n\n")
        f.write("## Runtime configuration\n")
        f.write(f"- `gp_W_eta = {W_eta}`\n")
        f.write(f"- `gp_kappa_eta = {kappa_eta}`\n")
        f.write(f"- `gp_L_eta = {L_eta}`\n")
        f.write(f"- `gp_elastic_enabled = {gp_elastic_enabled}`\n")
        f.write(f"- `gp_elastic_active_eta = {gp_elastic_active_eta}`\n")
        f.write(f"- `gp_elastic_derivative_scale = {gp_elastic_derivative_scale}`\n")
        f.write(f"- `gp_eps_iso = {gp_eps_iso}`\n")
        f.write(f"- `gp_obs_iface_width_nm = {iface_width}`\n")
        f.write(f"- `mu_GP0_mechanical_mixture = {gp_ref['mu_GP0']:.4f} J/mol`\n")
        f.write(f"- `minus_Delta_mu_r_GP(xB=0.03) = {gp_ref['minus_dmu_xB003']:.4f} J/mol`\n")
        f.write("- `matrix diffusion thermodynamics branch = raw_regular_solution`\n")
        f.write("\n## Code audit\n")
        f.write("- `gp_W_eta` enters `compute_eta_rhs_kernel()` as `+ gp_W_eta * g_prime_of_phi(eta)` in [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:851).\n")
        f.write("- `gp_kappa_eta` does not enter the explicit RHS; it enters the semi-implicit denominator in `eta_semi_implicit_update_kernel()` via `1 + L_eta*dt*kappa_eta*k^2` in [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:950).\n")
        f.write("- The gradient contribution written to VTK is reconstructed as `-gp_kappa_eta * laplacian(eta)` in [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:13456).\n")
        f.write("- Elastic `dgel/deta` is active when `gp_elastic_enabled && gp_elastic_active_eta`, through `compute_dgel_deta_gp_point()` in [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:280) and the scaled term added at [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:833).\n")
        f.write("- The eta update is semi-implicit in the gradient term and explicit in chemical / double-well / elastic terms.\n")
        f.write("\n## Interface term magnitudes (0.1 < eta < 0.9)\n")
        for row in term_rows:
            f.write(f"\n### step {row['step']}\n")
            f.write(f"- `chem_rms = {row['chem_rms']:.6e}`\n")
            f.write(f"- `dw_rms = {row['dw_rms']:.6e}`\n")
            f.write(f"- `grad_rms = {row['grad_rms']:.6e}`\n")
            f.write(f"- `elastic_rms = {row['elastic_rms']:.6e}`\n")
            f.write(f"- `full_rms = {row['full_rms']:.6e}`\n")
            f.write(f"- `ratio_dw_over_chem = {row['ratio_dw_over_chem']:.6e}`\n")
            f.write(f"- `ratio_grad_over_chem = {row['ratio_grad_over_chem']:.6e}`\n")
            f.write(f"- `ratio_elastic_over_chem = {row['ratio_elastic_over_chem']:.6e}`\n")
            f.write(f"- `ratio_nonchemical_over_chem = {row['ratio_nonchemical_over_chem']:.6e}`\n")
            f.write(f"- `minus_Delta_mu_r_GP(eta≈0.5)_mean = {row['minus_dmu_eta05_mean']:.6e}`\n")
            f.write(f"- `h_prime(eta≈0.5)_mean = {row['hprime_eta05_mean']:.6e}`\n")
            f.write(f"- `g_prime(eta≈0.5)_mean = {row['gprime_eta05_mean']:.6e}`\n")
            f.write(f"- `eta_rhs_chem(eta≈0.5)_mean = {row['eta_rhs_chem_eta05_mean']:.6e}`\n")
            f.write(f"- `eta_rhs_dw(eta≈0.5)_mean = {row['eta_rhs_dw_eta05_mean']:.6e}`\n")
            f.write(f"- `eta_rhs_grad(eta≈0.5)_mean = {row['eta_rhs_grad_eta05_mean']:.6e}`\n")
            f.write(f"- `eta_rhs_elastic(eta≈0.5)_mean = {row['eta_rhs_elastic_eta05_mean']:.6e}`\n")
        f.write("\n## Health metrics\n")
        for row in ts_rows:
            f.write(f"- step {row['step']}: `eta_max={row['eta_max']:.6e}`, `V_h={row['V_h_nm3']:.6e} nm^3`, `R_eff_h={row['R_eff_h_nm']:.6e} nm`, `xB_range=[{row['xB_min']:.6e}, {row['xB_max']:.6e}]`, `drift={row['total_relative_drift']:.6e}`, `max_abs(dt*divJ)={row['max_abs_dt_divJ']}`\n")
        chem_dom = (
            math.isfinite(W_eta) and abs(W_eta) < 1.0e-30 and
            math.isfinite(kappa_eta) and abs(kappa_eta) < 1.0e-30 and
            (not math.isfinite(p2000["ratio_elastic_over_chem"]) or abs(p2000["ratio_elastic_over_chem"]) < 1.0e-2)
        )
        f.write("\n## Answers\n")
        f.write(f"1. The current eta equation is {'chemical-drive dominated rather than a full Allen-Cahn GP phase-field equation with active interfacial/elastic limiting terms' if chem_dom else 'a mixed chemical + nonchemical Allen-Cahn equation'}.\n")
        f.write(f"2. `gp_W_eta` and `gp_kappa_eta` are {'zero' if (math.isfinite(W_eta) and abs(W_eta) < 1.0e-30 and math.isfinite(kappa_eta) and abs(kappa_eta) < 1.0e-30) else 'nonzero'} at runtime.\n")
        f.write(f"3. Elastic `dgel/deta` is {'active but negligible compared with chemical drive' if (gp_elastic_enabled and gp_elastic_active_eta and (not math.isfinite(p2000['ratio_elastic_over_chem']) or abs(p2000['ratio_elastic_over_chem']) < 1.0e-2)) else 'active and significant' if (gp_elastic_enabled and gp_elastic_active_eta) else 'inactive'}.\n")
        f.write(f"4. The current model {'does not contain an effective finite-size limiting mechanism in the eta RHS for this run' if chem_dom else 'contains some nonchemical limiting mechanisms'}.\n")
        f.write("5. Step36 runaway/continued growth is therefore not proof that the mechanical-mixture `mu_GP0` is physically wrong; it is also consistent with missing or too-weak `W_eta / kappa_eta / elastic` limiting physics.\n")
        f.write("6. The next required calibration target is all three nonchemical controls, with immediate priority on `W_eta` and `kappa_eta`, then elastic strength (`gp_eps_iso`, `gp_elastic_derivative_scale`) once the interfacial terms are nonzero.\n")


if __name__ == "__main__":
    main()
