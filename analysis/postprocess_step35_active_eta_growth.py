#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np


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


def periodic_radius_grid(n: int, dx_nm: float) -> np.ndarray:
    coords = np.arange(n, dtype=np.float64) * dx_nm
    center = 0.5 * n * dx_nm
    box = n * dx_nm
    q = coords - center
    q = q - np.round(q / box) * box
    rx = q[:, None, None]
    ry = q[None, :, None]
    rz = q[None, None, :]
    return np.sqrt(rx * rx + ry * ry + rz * rz).reshape(-1)


def h_of_eta(arr: np.ndarray) -> np.ndarray:
    return np.where(
        arr <= 0.0,
        0.0,
        np.where(arr >= 1.0, 1.0, arr * arr * arr * (6.0 * arr * arr - 15.0 * arr + 10.0)),
    )


def step_key(path: Path):
    if path.name == "eta_init.vtk":
        return 0
    m = re.fullmatch(r"eta_(\d+)\.vtk", path.name)
    return int(m.group(1)) if m else None


def parse_case_l_eta(case_name: str) -> float:
    m = re.search(r"Leta([0-9eE\+\-\.]+)_", case_name)
    if not m:
        return float("nan")
    token = m.group(1)
    try:
        return float(token)
    except ValueError:
        return float("nan")


def read_param_value(case_params_path: Path, key: str) -> float:
    if not case_params_path.exists():
        return float("nan")
    for line in case_params_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        if k.strip() == key:
            try:
                return float(v.strip())
            except ValueError:
                return float("nan")
    return float("nan")


def read_total_relative_drift(result_dir: Path) -> float:
    json_path = result_dir / "mass_drift_summary.json"
    if json_path.exists():
        import json
        try:
            data = json.load(json_path.open())
            return float(data.get("total_relative_drift", float("nan")))
        except Exception:
            return float("nan")
    csv_path = result_dir / "mass_drift_summary.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
        if rows:
            last = rows[-1]
            for key in ("total_relative_drift", "relative_drift", "drift"):
                if key in last:
                    try:
                        return float(last[key])
                    except ValueError:
                        pass
    return float("nan")


def list_step_files(result_dir: Path, prefix: str) -> list[tuple[int, Path]]:
    out = []
    for p in result_dir.iterdir():
        if not p.is_file():
            continue
        if p.name == f"{prefix}_init.vtk":
            out.append((0, p))
            continue
        m = re.fullmatch(rf"{re.escape(prefix)}_(\d+)\.vtk", p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def classify(summary: dict) -> str:
    if summary["nan_or_inf_detected"] or summary["xB_clip_count_high_total"] > 0 or summary["xB_clip_count_low_total"] > 0:
        return "unstable"
    if abs(summary["total_relative_drift_final"]) > 1.0e-2:
        return "unstable"
    if summary["max_abs_dt_divJ"] > 1.0e-1:
        return "unstable"
    vh = summary["relative_change_V_h"]
    reff = summary["relative_change_R_eff_h"]
    if math.isfinite(vh) and math.isfinite(reff):
        if vh > 0.02 and reff > 0.01:
            return "growth"
        if vh < -0.02 or reff < -0.01:
            return "relaxation/shrink"
        if abs(vh) < 0.01 and abs(reff) < 0.005:
            return "plateau"
    return "weak_change"


def main():
    ap = argparse.ArgumentParser(description="Summarize Step35 active eta growth scan.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--dx-nm", type=float, default=0.1)
    ap.add_argument("--xB-background", type=float, default=0.05)
    ap.add_argument("--eta-interface-min", type=float, default=0.1)
    ap.add_argument("--eta-interface-max", type=float, default=0.9)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    runs_dir = out_root / "runs"
    analysis_dir = out_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    total = args.grid ** 3
    radius = periodic_radius_grid(args.grid, args.dx_nm)

    summary_rows = []
    timeseries_rows = []

    for case_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        result_dirs = [p for p in sorted(case_dir.glob("Results/*/*")) if p.is_dir() and (p / "dynamics_mass_diagnostics.csv").exists()]
        if not result_dirs:
            continue
        result_dir = result_dirs[0]
        diag_path = result_dir / "dynamics_mass_diagnostics.csv"
        case_params_path = case_dir / "case.params"
        log_text = (case_dir / "run.log").read_text(encoding="utf-8", errors="ignore")
        diag_rows = list(csv.DictReader(diag_path.open(newline="", encoding="utf-8")))
        if not diag_rows:
            continue
        diag_by_step = {int(float(r["step"])): r for r in diag_rows}
        last = diag_rows[-1]
        gp_l_eta = read_param_value(case_params_path, "gp_L_eta")
        if not math.isfinite(gp_l_eta):
            gp_l_eta = parse_case_l_eta(case_dir.name)

        eta_paths = list_step_files(result_dir, "eta")
        prev_eta = None
        profile_rows = []
        for step, eta_path in eta_paths:
            xb_path = result_dir / f"xB_{step}.vtk"
            if step == 0 and not xb_path.exists():
                xb_path = result_dir / "xB_init.vtk"
            chem_path = result_dir / f"eta_rhs_chem_{step}.vtk"
            elastic_path = result_dir / f"eta_rhs_elastic_{step}.vtk"
            grad_path = result_dir / f"eta_rhs_grad_{step}.vtk"
            net_path = result_dir / f"eta_rhs_net_explicit_{step}.vtk"
            full_path = result_dir / f"eta_rhs_full_variational_{step}.vtk"
            if not xb_path.exists():
                continue
            eta = read_scalar_vtk(eta_path, total)
            xb = read_scalar_vtk(xb_path, total)
            h = h_of_eta(eta)
            V_h = float(h.sum() * (args.dx_nm ** 3))
            R_eff = float(((3.0 * V_h) / (4.0 * math.pi)) ** (1.0 / 3.0)) if V_h > 0 else 0.0
            interface_mask = (eta >= args.eta_interface_min) & (eta <= args.eta_interface_max)
            row = {
                "case": case_dir.name,
                "gp_L_eta": gp_l_eta,
                "step": step,
                "eta_max": float(np.max(eta)),
                "eta_integral": float(np.sum(eta) * (args.dx_nm ** 3)),
                "V_h_nm3": V_h,
                "R_eff_h_nm": R_eff,
                "xB_min": float(np.min(xb)),
                "xB_max": float(np.max(xb)),
                "max_abs_delta_eta_vs_prev_sample": float(np.max(np.abs(eta - prev_eta))) if prev_eta is not None else float("nan"),
            }
            prev_eta = eta
            if chem_path.exists():
                chem = read_scalar_vtk(chem_path, total)
                row["eta_rhs_chem_interface_mean"] = float(np.mean(chem[interface_mask])) if np.any(interface_mask) else float("nan")
            else:
                row["eta_rhs_chem_interface_mean"] = float("nan")
            if elastic_path.exists():
                elastic = read_scalar_vtk(elastic_path, total)
                row["eta_rhs_elastic_interface_mean"] = float(np.mean(elastic[interface_mask])) if np.any(interface_mask) else float("nan")
            else:
                row["eta_rhs_elastic_interface_mean"] = float("nan")
            if grad_path.exists():
                grad = read_scalar_vtk(grad_path, total)
                row["eta_rhs_grad_interface_mean"] = float(np.mean(grad[interface_mask])) if np.any(interface_mask) else float("nan")
            else:
                row["eta_rhs_grad_interface_mean"] = float("nan")
            if net_path.exists():
                net = read_scalar_vtk(net_path, total)
                row["eta_rhs_net_interface_mean"] = float(np.mean(net[interface_mask])) if np.any(interface_mask) else float("nan")
            else:
                row["eta_rhs_net_interface_mean"] = float("nan")
            if full_path.exists():
                full = read_scalar_vtk(full_path, total)
                row["eta_evolution_drive_interface_mean"] = float(-np.mean(full[interface_mask])) if np.any(interface_mask) else float("nan")
            else:
                row["eta_evolution_drive_interface_mean"] = float("nan")
            diag = diag_by_step.get(step)
            if diag:
                row["gp_minus_delta_mu_r_mean"] = float(diag.get("gp_minus_delta_mu_r_mean", "nan") or "nan")
                row["gp_minus_delta_mu_r_min"] = float(diag.get("gp_minus_delta_mu_r_min", "nan") or "nan")
                row["gp_minus_delta_mu_r_max"] = float(diag.get("gp_minus_delta_mu_r_max", "nan") or "nan")
                row["xB_clip_count_high"] = int(float(diag.get("xB_clip_count_high", "0") or 0.0))
                row["xB_clip_count_low"] = int(float(diag.get("xB_clip_count_low", "0") or 0.0))
                row["gp_closure_error"] = float(diag.get("gp_closure_error", "nan") or "nan")
                row["dt_divJ_min"] = float(diag.get("dt_divJ_min", "nan") or "nan")
                row["dt_divJ_max"] = float(diag.get("dt_divJ_max", "nan") or "nan")
            profile_rows.append(row)
            timeseries_rows.append(row)

        if not profile_rows:
            continue
        init = profile_rows[0]
        final = profile_rows[-1]
        drift = float(last.get("total_relative_drift_final", "nan") or "nan")
        summary = {
            "case": case_dir.name,
            "gp_L_eta": gp_l_eta,
            "steps_completed": int(float(last.get("step", "0") or 0.0)),
            "eta_max_initial": init["eta_max"],
            "eta_max_final": final["eta_max"],
            "eta_integral_initial": init["eta_integral"],
            "eta_integral_final": final["eta_integral"],
            "V_h_initial_nm3": init["V_h_nm3"],
            "V_h_final_nm3": final["V_h_nm3"],
            "R_eff_h_initial_nm": init["R_eff_h_nm"],
            "R_eff_h_final_nm": final["R_eff_h_nm"],
            "relative_change_V_h": (final["V_h_nm3"] - init["V_h_nm3"]) / init["V_h_nm3"] if init["V_h_nm3"] != 0 else float("nan"),
            "relative_change_R_eff_h": (final["R_eff_h_nm"] - init["R_eff_h_nm"]) / init["R_eff_h_nm"] if init["R_eff_h_nm"] != 0 else float("nan"),
            "relative_change_eta_integral": (final["eta_integral"] - init["eta_integral"]) / init["eta_integral"] if init["eta_integral"] != 0 else float("nan"),
            "max_abs_delta_eta_per_sample": max(abs(r["max_abs_delta_eta_vs_prev_sample"]) for r in profile_rows if math.isfinite(r["max_abs_delta_eta_vs_prev_sample"])) if any(math.isfinite(r["max_abs_delta_eta_vs_prev_sample"]) for r in profile_rows) else float("nan"),
            "xB_min_final": float(last.get("gp_min_xB_alpha", "nan") or "nan"),
            "xB_max_final": float(last.get("gp_max_xB_alpha", "nan") or "nan"),
            "xB_clip_count_high_total": sum(int(float(r.get("xB_clip_count_high", "0") or 0.0)) for r in diag_rows),
            "xB_clip_count_low_total": sum(int(float(r.get("xB_clip_count_low", "0") or 0.0)) for r in diag_rows),
            "total_relative_drift_final": read_total_relative_drift(result_dir),
            "gp_closure_error_final": float(last.get("gp_closure_error", "nan") or "nan"),
            "max_abs_dt_divJ": max(
                max(abs(float(r.get("dt_divJ_min", "nan") or "nan")), abs(float(r.get("dt_divJ_max", "nan") or "nan")))
                for r in diag_rows
            ),
            "gp_minus_delta_mu_r_mean_last": float(last.get("gp_minus_delta_mu_r_mean", "nan") or "nan"),
            "gp_minus_delta_mu_r_min_last": float(last.get("gp_minus_delta_mu_r_min", "nan") or "nan"),
            "gp_minus_delta_mu_r_max_last": float(last.get("gp_minus_delta_mu_r_max", "nan") or "nan"),
            "eta_rhs_chem_interface_mean_final": final["eta_rhs_chem_interface_mean"],
            "eta_rhs_elastic_interface_mean_final": final["eta_rhs_elastic_interface_mean"],
            "eta_rhs_grad_interface_mean_final": final["eta_rhs_grad_interface_mean"],
            "eta_rhs_net_interface_mean_final": final["eta_rhs_net_interface_mean"],
            "eta_evolution_drive_interface_mean_final": final["eta_evolution_drive_interface_mean"],
            "nan_or_inf_detected": int(("nan" in log_text.lower()) or bool(re.search(r'(^|[^a-z])inf([^a-z]|$)', log_text.lower()))),
        }
        summary["classification"] = classify(summary)
        summary_rows.append(summary)

    timeseries_fields = []
    if timeseries_rows:
        seen = set()
        for row in timeseries_rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    timeseries_fields.append(key)
    if timeseries_rows:
        with (analysis_dir / "gp_eta_growth_timeseries.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=timeseries_fields)
            writer.writeheader()
            for row in timeseries_rows:
                writer.writerow({k: row.get(k, "") for k in timeseries_fields})

    if summary_rows:
        with (analysis_dir / "final_key_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    report = analysis_dir / "reports/step_reports/STEP35_ACTIVE_ETA_GROWTH_TEST_REPORT.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Step 35 Active Eta Growth Test Report\n\n")
        f.write("- Step 34 plateau was obtained with frozen eta dynamics: `gp_L_eta = 0`, `gp_W_eta = 0`, `gp_kappa_eta = 0`.\n")
        f.write("- Step 35 is the first actual deterministic GP growth test using the raw stoichiometric GP reaction drive with `gp_L_eta > 0`.\n\n")
        for row in summary_rows:
            f.write(f"## {row['case']}\n\n")
            for key in [
                "gp_L_eta",
                "classification",
                "eta_max_initial",
                "eta_max_final",
                "V_h_initial_nm3",
                "V_h_final_nm3",
                "R_eff_h_initial_nm",
                "R_eff_h_final_nm",
                "relative_change_V_h",
                "relative_change_R_eff_h",
                "relative_change_eta_integral",
                "max_abs_delta_eta_per_sample",
                "max_abs_dt_divJ",
                "xB_clip_count_high_total",
                "xB_clip_count_low_total",
                "total_relative_drift_final",
                "gp_minus_delta_mu_r_mean_last",
                "eta_evolution_drive_interface_mean_final",
            ]:
                f.write(f"- `{key} = {row[key]}`\n")
            f.write("\n")
        growth_cases = [r for r in summary_rows if r["classification"] == "growth"]
        plateau_cases = [r for r in summary_rows if r["classification"] == "plateau"]
        stable_cases = [r for r in summary_rows if r["classification"] not in ("unstable",)]
        f.write("## Main Answers\n\n")
        f.write(f"- Cases run: {len(summary_rows)}\n")
        f.write(f"- Stable cases: {len(stable_cases)}\n")
        f.write(f"- Growth cases: {len(growth_cases)}\n")
        f.write(f"- Plateau cases: {len(plateau_cases)}\n")
        if growth_cases:
            smallest = min(growth_cases, key=lambda r: r["gp_L_eta"])
            f.write(f"- Smallest `gp_L_eta` showing measurable growth: `{smallest['gp_L_eta']}`\n")
        else:
            f.write("- No scanned `gp_L_eta` produced clear growth in this test.\n")


if __name__ == "__main__":
    main()
