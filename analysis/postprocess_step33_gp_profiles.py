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
        description="Rebuild Step33 GP profile timeseries from existing VTK outputs."
    )
    p.add_argument("--out-root", required=True, help="Step33 output root directory")
    p.add_argument(
        "--case-name",
        default="step33_raw_gp_drive_only_xB05_eta1_R1_dep2_N96",
        help="Case directory name under runs/",
    )
    p.add_argument("--grid", type=int, default=96, help="Grid size in each dimension")
    p.add_argument("--dx-nm", type=float, default=0.1, help="Physical dx in nm")
    p.add_argument(
        "--xB-background",
        type=float,
        default=0.05,
        help="Background xB used for depletion metrics",
    )
    p.add_argument(
        "--result-dir",
        default="",
        help="Optional explicit result directory containing eta_*.vtk/xB_*.vtk",
    )
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
    data = text[idx + len(marker) :]
    arr = np.fromstring(data, sep=" ")
    if arr.size != expected_count:
        raise ValueError(f"{path} expected {expected_count} values, got {arr.size}")
    return arr


def find_result_dir(case_dir: Path, override: str) -> Path:
    if override:
        result_dir = Path(override)
        if not result_dir.is_dir():
            raise FileNotFoundError(f"explicit result dir missing: {result_dir}")
        return result_dir
    candidates = [p for p in sorted(case_dir.glob("Results/*/*")) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no result directory found under {case_dir / 'Results'}")
    return candidates[0]


def build_masks(n: int, dx_nm: float):
    coords = np.arange(n, dtype=np.float64) * dx_nm
    center_nm = 0.5 * n * dx_nm
    L_nm = n * dx_nm
    periodic = lambda c: c - center_nm - np.round((c - center_nm) / L_nm) * L_nm
    rx = periodic(coords)[:, None, None]
    ry = periodic(coords)[None, :, None]
    rz = periodic(coords)[None, None, :]
    r = np.sqrt(rx * rx + ry * ry + rz * rz)
    center_mask = r <= 0.5
    shell_mask = (r >= 1.0) & (r <= 2.0)
    return center_mask.reshape(-1), shell_mask.reshape(-1)


def step_key(path: Path):
    if path.name == "eta_init.vtk":
        return 0
    m = re.search(r"_(\d+)\.vtk$", path.name)
    return int(m.group(1)) if m else None


def collect_profiles(result_dir: Path, n: int, dx_nm: float, xB_background: float):
    total = n * n * n
    voxel_nm3 = dx_nm ** 3
    center_mask, shell_mask = build_masks(n, dx_nm)
    eta_paths = []
    for p in result_dir.glob("eta_*.vtk"):
        step = step_key(p)
        if step is not None:
            eta_paths.append((step, p))
    eta_paths.sort(key=lambda item: item[0])

    rows = []
    for step, eta_path in eta_paths:
        xb_path = result_dir / ("xB_init.vtk" if step == 0 else f"xB_{step}.vtk")
        if not xb_path.exists():
            continue
        eta_vals = read_scalar_vtk(eta_path, total)
        xb_vals = read_scalar_vtk(xb_path, total)
        h_vals = h_of_eta(eta_vals)
        V_h = float(h_vals.sum() * voxel_nm3)
        eta_integral = float(eta_vals.sum() * voxel_nm3)
        R_eff = float(((3.0 * V_h) / (4.0 * math.pi)) ** (1.0 / 3.0)) if V_h > 0.0 else 0.0
        center_mean = float(xb_vals[center_mask].mean()) if center_mask.any() else float("nan")
        shell_mean = float(xb_vals[shell_mask].mean()) if shell_mask.any() else float("nan")
        depletion_depth = xB_background - shell_mean if math.isfinite(shell_mean) else float("nan")
        center_shell_contrast = (
            center_mean - shell_mean if math.isfinite(center_mean) and math.isfinite(shell_mean) else float("nan")
        )
        rows.append(
            {
                "step": step,
                "V_h_nm3": V_h,
                "R_eff_h_nm": R_eff,
                "eta_integral_nm3": eta_integral,
                "xB_center_mean": center_mean,
                "xB_shell_mean": shell_mean,
                "depletion_depth": depletion_depth,
                "center_shell_contrast": center_shell_contrast,
            }
        )
    return rows


def rel_change(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)) or abs(a) < 1e-30:
        return float("nan")
    return (b - a) / a


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    case_dir = out_root / "runs" / args.case_name
    result_dir = find_result_dir(case_dir, args.result_dir)

    profile_rows = collect_profiles(result_dir, args.grid, args.dx_nm, args.xB_background)
    write_csv(
        out_root / "gp_profile_timeseries.csv",
        profile_rows,
        ["step", "V_h_nm3", "R_eff_h_nm", "eta_integral_nm3", "xB_center_mean", "xB_shell_mean", "depletion_depth", "center_shell_contrast"],
    )

    diag_rows = list(csv.DictReader(open(result_dir / "dynamics_mass_diagnostics.csv", newline="", encoding="utf-8")))
    mass_summary = json.load(open(result_dir / "mass_drift_summary.json", encoding="utf-8"))
    last = diag_rows[-1]
    init_prof = profile_rows[0] if profile_rows else {}
    final_prof = profile_rows[-1] if profile_rows else {}

    def col(name):
        return [float(r.get(name, "nan") or "nan") for r in diag_rows]

    finite_mu_mean = [v for v in col("gp_minus_delta_mu_r_mean") if math.isfinite(v)]

    classification = "plateau_or_weak_relaxation"
    vh_change = rel_change(init_prof.get("V_h_nm3", float("nan")), final_prof.get("V_h_nm3", float("nan")))
    reff_change = rel_change(init_prof.get("R_eff_h_nm", float("nan")), final_prof.get("R_eff_h_nm", float("nan")))
    eta_change = rel_change(init_prof.get("eta_integral_nm3", float("nan")), final_prof.get("eta_integral_nm3", float("nan")))
    contrast_change = (
        final_prof.get("center_shell_contrast", float("nan")) - init_prof.get("center_shell_contrast", float("nan"))
        if profile_rows
        else float("nan")
    )
    if math.isfinite(vh_change) and math.isfinite(eta_change) and vh_change > 0.05 and eta_change > 0.05:
        classification = "growth_like"
    if sum(int(float(r.get("xB_clip_count_high", "0") or 0.0)) for r in diag_rows) > 0:
        classification = "xB_clipping_failure"

    summary = {
        "case": args.case_name,
        "T_K": 653.15,
        "xB_background": args.xB_background,
        "gp_init_mode": "observed_gp_diffuse",
        "gp_y_update_mode": "conservative_y_rhs",
        "gp_raw_reaction_drive_only": 1,
        "steps_completed": int(float(last.get("step", "0") or 0.0)),
        "gp_mu_reference_raw": float(last.get("gp_mu_reference_raw", "nan") or "nan"),
        "gp_minus_delta_mu_r_mean_last": float(last.get("gp_minus_delta_mu_r_mean", "nan") or "nan"),
        "gp_minus_delta_mu_r_min_last": float(last.get("gp_minus_delta_mu_r_min", "nan") or "nan"),
        "gp_minus_delta_mu_r_max_last": float(last.get("gp_minus_delta_mu_r_max", "nan") or "nan"),
        "gp_minus_delta_mu_r_mean_series_min": min(finite_mu_mean) if finite_mu_mean else float("nan"),
        "gp_minus_delta_mu_r_mean_series_max": max(finite_mu_mean) if finite_mu_mean else float("nan"),
        "V_h_init_nm3": init_prof.get("V_h_nm3", float("nan")),
        "V_h_final_nm3": final_prof.get("V_h_nm3", float("nan")),
        "R_eff_h_init_nm": init_prof.get("R_eff_h_nm", float("nan")),
        "R_eff_h_final_nm": final_prof.get("R_eff_h_nm", float("nan")),
        "eta_integral_init_nm3": init_prof.get("eta_integral_nm3", float("nan")),
        "eta_integral_final_nm3": final_prof.get("eta_integral_nm3", float("nan")),
        "center_shell_contrast_init": init_prof.get("center_shell_contrast", float("nan")),
        "center_shell_contrast_final": final_prof.get("center_shell_contrast", float("nan")),
        "depletion_depth_init": init_prof.get("depletion_depth", float("nan")),
        "depletion_depth_final": final_prof.get("depletion_depth", float("nan")),
        "relative_change_V_h": vh_change,
        "relative_change_R_eff_h": reff_change,
        "relative_change_eta_integral": eta_change,
        "delta_center_shell_contrast": contrast_change,
        "xB_min_final": float(last.get("gp_min_xB_alpha", "nan") or "nan"),
        "xB_max_final": float(last.get("gp_max_xB_alpha", "nan") or "nan"),
        "xB_clip_count_high_total": sum(int(float(r.get("xB_clip_count_high", "0") or 0.0)) for r in diag_rows),
        "xB_clip_count_low_total": sum(int(float(r.get("xB_clip_count_low", "0") or 0.0)) for r in diag_rows),
        "total_relative_drift_final": float(mass_summary.get("total_relative_drift", float("nan"))),
        "gp_closure_error_final": float(last.get("gp_closure_error", "nan") or "nan"),
        "classification": classification,
    }
    write_csv(out_root / "final_key_summary.csv", [summary], list(summary.keys()))

    report = out_root / "reports/step_reports/STEP33_RAW_GP_DRIVE_ONLY_REPORT.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Step 33 Raw GP Drive Only Report\n\n")
        f.write("## Raw GP reaction-drive diagnostics\n\n")
        for k in [
            "gp_mu_reference_raw",
            "gp_minus_delta_mu_r_mean_last",
            "gp_minus_delta_mu_r_min_last",
            "gp_minus_delta_mu_r_max_last",
            "gp_minus_delta_mu_r_mean_series_min",
            "gp_minus_delta_mu_r_mean_series_max",
        ]:
            f.write(f"- `{k} = {summary[k]:.10e}`\n")
        f.write("\n## Growth indicators\n\n")
        for k in [
            "V_h_init_nm3",
            "V_h_final_nm3",
            "R_eff_h_init_nm",
            "R_eff_h_final_nm",
            "eta_integral_init_nm3",
            "eta_integral_final_nm3",
            "center_shell_contrast_init",
            "center_shell_contrast_final",
            "depletion_depth_init",
            "depletion_depth_final",
            "relative_change_V_h",
            "relative_change_R_eff_h",
            "relative_change_eta_integral",
            "delta_center_shell_contrast",
        ]:
            f.write(f"- `{k} = {summary[k]:.10e}`\n")
        f.write("\n## Required conclusion\n\n")
        if summary["classification"] == "growth_like":
            f.write("mature GP plateau behavior does not persist under the raw regular-solution GP reaction reference.\n")
        else:
            f.write("mature GP plateau behavior persists, or at least autonomous GP growth is still not established, even when the GP reaction reference is computed from raw regular-solution thermodynamics.\n")


if __name__ == "__main__":
    main()
