#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def parse_simple_params(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def read_scalar_vtk(path: Path, expected_count: int | None = None) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    marker = "LOOKUP_TABLE default"
    idx = text.find(marker)
    if idx < 0:
        raise ValueError(f"LOOKUP_TABLE default not found in {path}")
    arr = np.fromstring(text[idx + len(marker):], sep=" ")
    if expected_count is not None and arr.size != expected_count:
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


def latest_step(result_dir: Path) -> int:
    steps = []
    for p in result_dir.glob("eta_*.vtk"):
        m = re.search(r"_(\d+)\.vtk$", p.name)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        raise FileNotFoundError("no eta_<step>.vtk found")
    return max(steps)


def mean_or_nan(arr: np.ndarray) -> float:
    return float(arr.mean()) if arr.size else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Analyze Step34 GP interface driving-balance diagnostics.")
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--dx-nm", type=float, default=0.1)
    ap.add_argument("--eta-interface-min", type=float, default=0.1)
    ap.add_argument("--eta-interface-max", type=float, default=0.9)
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = latest_step(result_dir)
    n = args.grid
    total = n * n * n
    radius = periodic_radius_grid(n, args.dx_nm)

    fields = {}
    for stem in [
        "eta",
        "xB",
        "eta_rhs_chem",
        "eta_rhs_dw",
        "eta_rhs_elastic",
        "eta_rhs_grad",
        "eta_rhs_net_explicit",
        "eta_rhs_full_variational",
    ]:
        fields[stem] = read_scalar_vtk(result_dir / f"{stem}_{step}.vtk", total)

    eta = fields["eta"]
    interface_mask = (eta >= args.eta_interface_min) & (eta <= args.eta_interface_max)
    core_mask = eta >= 0.95
    shell_mask = interface_mask

    rows = []
    summary = {}
    for name in [
        "eta_rhs_chem",
        "eta_rhs_dw",
        "eta_rhs_elastic",
        "eta_rhs_grad",
        "eta_rhs_net_explicit",
        "eta_rhs_full_variational",
    ]:
        arr = fields[name]
        summary[f"{name}_interface_mean"] = mean_or_nan(arr[interface_mask])
        summary[f"{name}_core_mean"] = mean_or_nan(arr[core_mask])
        summary[f"{name}_global_mean"] = mean_or_nan(arr)
    summary["eta_evolution_drive_interface_mean"] = -summary["eta_rhs_full_variational_interface_mean"]
    summary["eta_evolution_drive_core_mean"] = -summary["eta_rhs_full_variational_core_mean"]
    summary["interface_voxel_count"] = int(interface_mask.sum())
    summary["core_voxel_count"] = int(core_mask.sum())
    summary["step"] = step
    case_params = parse_simple_params(result_dir.parents[2] / "case.params")
    gp_W_eta = float(case_params.get("gp_W_eta", "0") or "0")
    gp_kappa_eta = float(case_params.get("gp_kappa_eta", "0") or "0")
    gp_L_eta = float(case_params.get("gp_L_eta", "0") or "0")
    summary["gp_W_eta"] = gp_W_eta
    summary["gp_kappa_eta"] = gp_kappa_eta
    summary["gp_L_eta"] = gp_L_eta

    r_edges = np.arange(0.0, 3.05, 0.05)
    for lo, hi in zip(r_edges[:-1], r_edges[1:]):
        mask = (radius >= lo) & (radius < hi)
        if not np.any(mask):
            continue
        row = {"r_nm_lo": lo, "r_nm_hi": hi, "count": int(mask.sum())}
        for name in [
            "eta",
            "xB",
            "eta_rhs_chem",
            "eta_rhs_dw",
            "eta_rhs_elastic",
            "eta_rhs_grad",
            "eta_rhs_net_explicit",
            "eta_rhs_full_variational",
        ]:
            row[name] = mean_or_nan(fields[name][mask])
        row["eta_evolution_drive"] = -row["eta_rhs_full_variational"]
        rows.append(row)

    with (out_dir / "gp_interface_driving_balance_profile.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "gp_interface_driving_balance_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    rc = np.array([0.5 * (r["r_nm_lo"] + r["r_nm_hi"]) for r in rows])
    plt.figure(figsize=(7, 4.5))
    for name, label in [
        ("eta_rhs_chem", "chemical"),
        ("eta_rhs_dw", "double-well"),
        ("eta_rhs_elastic", "elastic"),
        ("eta_rhs_grad", "gradient"),
        ("eta_rhs_full_variational", "full variational"),
    ]:
        plt.plot(rc, [r[name] for r in rows], label=label)
    plt.axhline(0.0, color="k", lw=0.8)
    plt.xlabel("r (nm)")
    plt.ylabel("Contribution to δη free-energy derivative")
    plt.title("GP interface driving-balance components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gp_interface_driving_balance_profile.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(rc, [r["eta"] for r in rows], label="eta")
    plt.plot(rc, [r["xB"] for r in rows], label="xB_alpha")
    plt.xlabel("r (nm)")
    plt.ylabel("Field value")
    plt.title("GP profile used for interface-balance diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gp_interface_profile.png", dpi=180)
    plt.close()

    dominant = min(
        [("chemical", abs(summary["eta_rhs_chem_interface_mean"])),
         ("double-well", abs(summary["eta_rhs_dw_interface_mean"])),
         ("elastic", abs(summary["eta_rhs_elastic_interface_mean"])),
         ("gradient", abs(summary["eta_rhs_grad_interface_mean"]))],
        key=lambda x: -x[1],
    )[0]

    report = out_dir / "reports/step_reports/STEP34_GP_INTERFACE_DRIVING_BALANCE_REPORT.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Step 34 GP Interface Driving Balance Report\n\n")
        f.write(f"- `result_dir = {result_dir}`\n")
        f.write(f"- `step = {step}`\n")
        f.write(f"- `interface eta window = [{args.eta_interface_min}, {args.eta_interface_max}]`\n")
        f.write(f"- `interface_voxel_count = {summary['interface_voxel_count']}`\n\n")
        f.write("## GP Eta Parameters\n\n")
        f.write(f"- `gp_W_eta = {gp_W_eta:.10e}`\n")
        f.write(f"- `gp_kappa_eta = {gp_kappa_eta:.10e}`\n")
        f.write(f"- `gp_L_eta = {gp_L_eta:.10e}`\n\n")
        f.write("## Interface Means\n\n")
        for key in [
            "eta_rhs_chem_interface_mean",
            "eta_rhs_dw_interface_mean",
            "eta_rhs_elastic_interface_mean",
            "eta_rhs_grad_interface_mean",
            "eta_rhs_net_explicit_interface_mean",
            "eta_rhs_full_variational_interface_mean",
            "eta_evolution_drive_interface_mean",
        ]:
            f.write(f"- `{key} = {summary[key]:.10e}`\n")
        f.write("\n## Interpretation\n\n")
        f.write(f"- Dominant interface-balance term by magnitude: `{dominant}`.\n")
        f.write("- `eta_rhs_full_variational` is the local total free-energy derivative entering gradient flow.\n")
        f.write("- The actual local eta evolution drive is `-eta_rhs_full_variational`.\n")
        if gp_L_eta == 0.0:
            f.write("- `gp_L_eta = 0`, so even a nonzero local eta driving force does not produce meaningful deterministic eta evolution in this run.\n")
        if gp_W_eta == 0.0 and gp_kappa_eta == 0.0:
            f.write("- `gp_W_eta = 0` and `gp_kappa_eta = 0`, so this test has no eta double-well or gradient regularization; the interface-balance reduces almost entirely to the chemical term.\n")
        if abs(summary["eta_evolution_drive_interface_mean"]) < 1e-3 * max(1.0, abs(summary["eta_rhs_chem_interface_mean"])):
            f.write("- The interface is effectively pinned because the positive reaction-drive benefit is nearly canceled by opposing double-well/gradient/elastic terms.\n")
        else:
            f.write("- The interface net drive is not near zero; check radial profile for where cancellation fails.\n")


if __name__ == "__main__":
    main()
