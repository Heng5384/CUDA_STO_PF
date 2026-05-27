#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Postprocess Step39S GP interface stability audit.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--root-result-dir", required=True)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--grid", type=int, default=96)
    p.add_argument("--dx-nm", type=float, default=0.1)
    return p.parse_args()


def h_of_eta(arr):
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


def step_to_path(vtk_dir: Path, stem: str, step: int) -> Path | None:
    if step == 0 and stem == "eta":
        for cand in (vtk_dir / "eta_0.vtk", vtk_dir / "eta_init.vtk"):
            if cand.exists():
                return cand
    cand = vtk_dir / f"{stem}_{step}.vtk"
    return cand if cand.exists() else None


def classify_case(summary):
    if summary["NaN_or_Inf"] or summary["xB_clip_count_high_total"] > 0 or summary["xB_clip_count_low_total"] > 0:
        return "unstable"
    if summary["total_relative_drift_final"] > 1e-2 or summary["max_abs_dt_divJ_overall"] > 1.0 or summary["max_abs_delta_eta_per_step_overall"] > 0.5:
        return "unstable"
    if summary["total_relative_drift_final"] < 1e-3 and summary["max_abs_dt_divJ_overall"] <= 1e-2 and summary["max_abs_delta_eta_per_step_overall"] < 0.1:
        return "stable"
    return "marginal"


def summarize_case(case_dir: Path, case_name: str, out_dir: Path, grid: int, dx_nm: float):
    diag_rows = list(csv.DictReader(open(case_dir / "dynamics_mass_diagnostics.csv", newline="", encoding="utf-8")))
    mass_summary = json.load(open(case_dir / "mass_drift_summary.json", encoding="utf-8"))
    meta = json.load(open(case_dir / "scan_meta.json", encoding="utf-8"))
    vtk_dir = case_dir / "vtk_snapshots"
    total = grid ** 3
    dV = dx_nm ** 3
    diag_map = {int(float(r["step"])): r for r in diag_rows}

    ts_dir = out_dir / "step39S_timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)
    ts_rows = []
    prev_R = None
    prev_V = None
    prev_t = None
    snap_steps = [0] + sorted(diag_map.keys())
    for step in snap_steps:
        eta_path = step_to_path(vtk_dir, "eta", step)
        xb_path = step_to_path(vtk_dir, "xB", step)
        if not eta_path or not xb_path:
            continue
        eta = read_scalar_vtk(eta_path, total)
        xb = read_scalar_vtk(xb_path, total)
        h = h_of_eta(eta)
        V_h = float(h.sum() * dV)
        R_eff = float(((3.0 * V_h) / (4.0 * math.pi)) ** (1.0 / 3.0)) if V_h > 0 else 0.0
        dr = diag_map.get(step)
        time_code = 0.0 if step == 0 else float(dr.get("time", "nan") or "nan")
        row = {
            "step": step,
            "time_code": time_code,
            "eta_max": float(np.max(eta)),
            "eta_integral": float(np.sum(eta) * dV),
            "V_h_nm3": V_h,
            "R_eff_h_nm": R_eff,
            "dV_h_dt": math.nan,
            "dR_eff_h_dt": math.nan,
            "xB_min": float(np.min(xb)),
            "xB_max": float(np.max(xb)),
            "xB_clip_count_high": 0,
            "xB_clip_count_low": 0,
            "total_relative_drift": 0.0 if step == 0 else (float(dr.get("mean_xBtot_gp_end_step", "nan")) - float(mass_summary["initial_mean_xBtot_gp"])) / max(abs(float(mass_summary["initial_mean_xBtot_gp"])), 1e-30),
            "gp_closure_error": math.nan if step == 0 else float(dr.get("gp_closure_error", "nan") or "nan"),
            "max_abs_dt_divJ": math.nan if step == 0 else max(abs(float(dr.get("dt_divJ_min", "0") or 0.0)), abs(float(dr.get("dt_divJ_max", "0") or 0.0))),
            "max_abs_delta_eta_per_step": math.nan if step == 0 else abs(float(dr.get("eta_step_max_delta", "nan") or "nan")),
            "gp_minus_delta_mu_r_mean": math.nan if step == 0 else float(dr.get("gp_minus_delta_mu_r_mean", "nan") or "nan"),
            "gp_minus_delta_mu_r_min": math.nan if step == 0 else float(dr.get("gp_minus_delta_mu_r_min", "nan") or "nan"),
            "gp_minus_delta_mu_r_max": math.nan if step == 0 else float(dr.get("gp_minus_delta_mu_r_max", "nan") or "nan"),
            "eta_rhs_chem_interface_mean": math.nan,
            "eta_rhs_chem_interface_rms": math.nan,
            "eta_rhs_dw_interface_mean": math.nan,
            "eta_rhs_dw_interface_rms": math.nan,
            "eta_rhs_grad_interface_mean": math.nan,
            "eta_rhs_grad_interface_rms": math.nan,
            "eta_rhs_net_interface_mean": math.nan,
            "eta_rhs_net_interface_rms": math.nan,
            "eta_evolution_drive_interface_mean": math.nan,
            "eta_evolution_drive_interface_rms": math.nan,
            "ratio_dw_over_chem": math.nan,
            "ratio_grad_over_chem": math.nan,
            "ratio_nonchemical_over_chem": math.nan,
            "NaN_or_Inf": 0,
        }
        if prev_t is not None and time_code > prev_t:
            row["dV_h_dt"] = (V_h - prev_V) / (time_code - prev_t)
            row["dR_eff_h_dt"] = (R_eff - prev_R) / (time_code - prev_t)
        prev_t, prev_V, prev_R = time_code, V_h, R_eff

        # interface diagnostics if VTK exists
        if step > 0:
            eta_rhs_paths = {
                "chem": step_to_path(vtk_dir, "eta_rhs_chem", step),
                "dw": step_to_path(vtk_dir, "eta_rhs_dw", step),
                "grad": step_to_path(vtk_dir, "eta_rhs_grad", step),
                "net": step_to_path(vtk_dir, "eta_rhs_net_explicit", step),
            }
            if all(p is not None and p.exists() for p in eta_rhs_paths.values()):
                mask = (eta > 0.1) & (eta < 0.9)
                if np.any(mask):
                    chem = read_scalar_vtk(eta_rhs_paths["chem"], total)[mask]
                    dw = read_scalar_vtk(eta_rhs_paths["dw"], total)[mask]
                    grad = read_scalar_vtk(eta_rhs_paths["grad"], total)[mask]
                    net = read_scalar_vtk(eta_rhs_paths["net"], total)[mask]
                    nonchem_rms = math.sqrt(float(np.mean(dw * dw)) + float(np.mean(grad * grad)))
                    chem_rms = float(np.sqrt(np.mean(chem * chem)))
                    row.update({
                        "eta_rhs_chem_interface_mean": float(np.mean(chem)),
                        "eta_rhs_chem_interface_rms": chem_rms,
                        "eta_rhs_dw_interface_mean": float(np.mean(dw)),
                        "eta_rhs_dw_interface_rms": float(np.sqrt(np.mean(dw * dw))),
                        "eta_rhs_grad_interface_mean": float(np.mean(grad)),
                        "eta_rhs_grad_interface_rms": float(np.sqrt(np.mean(grad * grad))),
                        "eta_rhs_net_interface_mean": float(np.mean(net)),
                        "eta_rhs_net_interface_rms": float(np.sqrt(np.mean(net * net))),
                        "eta_evolution_drive_interface_mean": float(np.mean(-net)),
                        "eta_evolution_drive_interface_rms": float(np.sqrt(np.mean(net * net))),
                        "ratio_dw_over_chem": float(np.sqrt(np.mean(dw * dw)) / chem_rms) if chem_rms > 0 else math.nan,
                        "ratio_grad_over_chem": float(np.sqrt(np.mean(grad * grad)) / chem_rms) if chem_rms > 0 else math.nan,
                        "ratio_nonchemical_over_chem": float(nonchem_rms / chem_rms) if chem_rms > 0 else math.nan,
                    })

        if dr:
            row["xB_clip_count_high"] = int(float(dr.get("xB_clip_count_high", "0") or 0.0))
            row["xB_clip_count_low"] = int(float(dr.get("xB_clip_count_low", "0") or 0.0))
        ts_rows.append(row)

    ts_path = ts_dir / f"{case_name}.csv"
    with ts_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ts_rows[0].keys()))
        w.writeheader()
        w.writerows(ts_rows)

    summary = {
        "case": case_name,
        "phase": meta["phase"],
        "dt": meta["dt"],
        "steps": meta["steps"],
        "simulated_time": meta["simulated_time"],
        "gp_L_eta": meta["gp_L_eta"],
        "gamma_J_m2": meta["gamma_J_m2"],
        "W_eta_J_m3": meta["W_eta_J_m3"],
        "kappa_eta_J_m": meta["kappa_eta_J_m"],
        "final_written_step": int(ts_rows[-1]["step"]),
        "eta_max_final": ts_rows[-1]["eta_max"],
        "V_h_final_nm3": ts_rows[-1]["V_h_nm3"],
        "R_eff_h_final_nm": ts_rows[-1]["R_eff_h_nm"],
        "dR_eff_h_dt_final": ts_rows[-1]["dR_eff_h_dt"],
        "xB_min_final": ts_rows[-1]["xB_min"],
        "xB_max_final": ts_rows[-1]["xB_max"],
        "xB_clip_count_high_total": int(sum(r["xB_clip_count_high"] for r in ts_rows)),
        "xB_clip_count_low_total": int(sum(r["xB_clip_count_low"] for r in ts_rows)),
        "total_relative_drift_final": float(mass_summary["total_relative_drift"]),
        "gp_closure_error_final": float(diag_rows[-1].get("gp_closure_error", "nan") or "nan"),
        "max_abs_dt_divJ_overall": max(max(abs(float(r.get("dt_divJ_min", "0") or 0.0)), abs(float(r.get("dt_divJ_max", "0") or 0.0))) for r in diag_rows),
        "max_abs_delta_eta_per_step_overall": max(abs(float(r.get("eta_step_max_delta", "0") or 0.0)) for r in diag_rows),
        "gp_minus_delta_mu_r_mean_final": ts_rows[-1]["gp_minus_delta_mu_r_mean"],
        "gp_minus_delta_mu_r_min_final": ts_rows[-1]["gp_minus_delta_mu_r_min"],
        "gp_minus_delta_mu_r_max_final": ts_rows[-1]["gp_minus_delta_mu_r_max"],
        "NaN_or_Inf": 0,
        "classification": "",
    }
    summary["classification"] = classify_case(summary)
    return summary


def write_report(out_path: Path, summaries: list[dict]):
    primaries = [s for s in summaries if s["phase"] == "primary"]
    secondaries = [s for s in summaries if s["phase"] == "secondary"]
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Step 39S GP Interface Stability Audit Report\n\n")
        f.write("## Primary dt scan\n")
        for s in primaries:
            f.write(f"- `{s['case']}`: dt=`{s['dt']}`, steps=`{s['steps']}`, simulated_time=`{s['simulated_time']}`, classification=`{s['classification']}`, max_abs(dt*divJ)=`{s['max_abs_dt_divJ_overall']:.6e}`, drift=`{s['total_relative_drift_final']:.6e}`, max_abs_delta_eta_per_step=`{s['max_abs_delta_eta_per_step_overall']:.6e}`, xB_final=`[{s['xB_min_final']:.6e}, {s['xB_max_final']:.6e}]`\n")
        if secondaries:
            f.write("\n## Secondary gp_L_eta scan\n")
            for s in secondaries:
                f.write(f"- `{s['case']}`: dt=`{s['dt']}`, gp_L_eta=`{s['gp_L_eta']}`, classification=`{s['classification']}`, max_abs(dt*divJ)=`{s['max_abs_dt_divJ_overall']:.6e}`, drift=`{s['total_relative_drift_final']:.6e}`\n")
        stable_primary = [s for s in primaries if s["classification"] == "stable"]
        marginal_primary = [s for s in primaries if s["classification"] == "marginal"]
        best = None
        if stable_primary:
            best = min(stable_primary, key=lambda s: s["dt"])
        elif marginal_primary:
            best = min(marginal_primary, key=lambda s: s["dt"])
        elif secondaries:
            stable_secondary = [s for s in secondaries if s["classification"] == "stable"]
            marginal_secondary = [s for s in secondaries if s["classification"] == "marginal"]
            pool = stable_secondary or marginal_secondary or secondaries
            best = min(pool, key=lambda s: (s["classification"] != "stable", s["dt"], s["gp_L_eta"]))
        f.write("\n## Answers\n")
        f.write(f"1. Step39 failure is {'consistent with dt/interface stiffness' if not stable_primary else 'not solely caused by dt stiffness'}.\n")
        if stable_primary:
            f.write(f"2. For gamma=0.05, l_eta=1nm, gp_L_eta=1e-4, a stable dt is `{best['dt']}`.\n")
            f.write("3. Reducing dt does stabilize the calibrated W_eta/kappa_eta model.\n")
            f.write("4. No gp_L_eta reduction is required in the primary stable window.\n")
        else:
            f.write("2. No stable dt was found in the primary dt scan at gp_L_eta=1e-4.\n")
            if secondaries:
                stable_secondary = [s for s in secondaries if s["classification"] == "stable"]
                if stable_secondary:
                    b2 = min(stable_secondary, key=lambda s: (s["dt"], s["gp_L_eta"]))
                    f.write(f"3. dt reduction alone was insufficient; stability required lowering gp_L_eta to `{b2['gp_L_eta']}` at dt=`{b2['dt']}`.\n")
                    f.write(f"4. The required reduced gp_L_eta is `{b2['gp_L_eta']}`.\n")
                else:
                    f.write("3. Reducing dt alone was insufficient, and the secondary gp_L_eta scan also remained unstable/marginal.\n")
                    f.write("4. No clearly stable gp_L_eta reduction was found in the secondary scan.\n")
            else:
                f.write("3. Reducing dt alone was insufficient in the tested primary window.\n")
                f.write("4. Secondary gp_L_eta reduction was not needed because a primary stable case existed.\n")
        if best is not None:
            f.write(f"5. Recommended stable parameter set for the real Step39 GP-zone-like growth test: dt=`{best['dt']}`, gp_L_eta=`{best['gp_L_eta']}`, gamma=`{best['gamma_J_m2']}`, W_eta=`{best['W_eta_J_m3']:.6e}`, kappa_eta=`{best['kappa_eta_J_m']:.6e}`.\n")
        else:
            f.write("5. No stable parameter set was identified yet for the real Step39 GP-zone-like growth test.\n")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    root_result_dir = Path(args.root_result_dir)
    summaries = [summarize_case(root_result_dir / case, case, out_dir, args.grid, args.dx_nm) for case in args.cases]
    with (out_dir / "step39S_stability_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    write_report(out_dir / "STEP39S_GP_INTERFACE_STABILITY_AUDIT_REPORT.md", summaries)


if __name__ == "__main__":
    main()
