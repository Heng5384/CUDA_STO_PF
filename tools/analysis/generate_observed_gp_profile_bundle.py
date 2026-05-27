#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


X_B_GP = 0.35


def h_switch(x: np.ndarray) -> np.ndarray:
    xc = np.clip(x, 0.0, 1.0)
    return xc * xc * xc * (6.0 * xc * xc - 15.0 * xc + 10.0)


def periodic_delta(a: np.ndarray, c: float, L: float) -> np.ndarray:
    d = a - c
    return d - np.round(d / L) * L


def eta_profile_value(r: np.ndarray, radius_nm: float, peak: float, width_nm: float, profile_type: str) -> np.ndarray:
    w = max(width_nm, 1.0e-12)
    if profile_type == "gaussian":
        return peak * np.exp(-(r / max(radius_nm, 1.0e-12)) ** 2)
    if profile_type == "compact_smooth":
        x = np.clip(r / max(radius_nm, 1.0e-12), 0.0, 1.0)
        core = 1.0 - (10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5)
        tail = 0.5 * (1.0 - np.tanh((r - radius_nm) / w))
        return peak * np.where(r <= radius_nm, core, tail * core * 0.0)
    return peak * 0.5 * (1.0 - np.tanh((r - radius_nm) / w))


def depletion_weight(r: np.ndarray, radius_nm: float, dep_radius_nm: float, smooth_width_nm: float,
                     profile_type: str) -> np.ndarray:
    w = max(smooth_width_nm, 1.0e-12)
    protected = 0.5 * (1.0 + np.tanh((r - radius_nm) / w))
    if profile_type == "gaussian":
        return protected * np.exp(-(r / max(dep_radius_nm, 1.0e-12)) ** 2)
    if profile_type == "compact_smooth":
        x = np.clip(r / max(dep_radius_nm, 1.0e-12), 0.0, 1.0)
        shell = 1.0 - (10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5)
        return protected * shell
    return protected * 0.5 * (1.0 - np.tanh((r - dep_radius_nm) / w))


def max_abs_grad_periodic(field: np.ndarray, dx_nm: float) -> float:
    gx = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2.0 * dx_nm)
    gy = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2.0 * dx_nm)
    gz = (np.roll(field, -1, axis=2) - np.roll(field, 1, axis=2)) / (2.0 * dx_nm)
    return float(np.max(np.sqrt(gx * gx + gy * gy + gz * gz)))


def estimate_mu_from_xb(xb: np.ndarray) -> np.ndarray:
    eps = 1.0e-8
    x = np.clip(xb, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def radial_profile_csv(path: Path, r: np.ndarray, eta: np.ndarray, xb: np.ndarray, xbtot: np.ndarray,
                       weight: np.ndarray, dx_nm: float) -> None:
    bins = np.arange(0.0, float(np.max(r)) + dx_nm, dx_nm)
    inds = np.digitize(r.ravel(), bins) - 1
    eta_f = eta.ravel()
    h_f = h_switch(eta_f)
    xb_f = xb.ravel()
    xbtot_f = xbtot.ravel()
    w_f = weight.ravel()
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r_nm", "eta", "h_eta", "xB_alpha", "xBtot_gp", "weight", "count"])
        for b in range(len(bins) - 1):
            mask = inds == b
            if not np.any(mask):
                continue
            writer.writerow([
                0.5 * (bins[b] + bins[b + 1]),
                float(np.mean(eta_f[mask])),
                float(np.mean(h_f[mask])),
                float(np.mean(xb_f[mask])),
                float(np.mean(xbtot_f[mask])),
                float(np.mean(w_f[mask])),
                int(np.sum(mask)),
            ])


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate offline observed-GP diffuse profile bundle")
    ap.add_argument("--Nx", type=int, required=True)
    ap.add_argument("--Ny", type=int, required=True)
    ap.add_argument("--Nz", type=int, required=True)
    ap.add_argument("--dx-nm", type=float, required=True)
    ap.add_argument("--xB-background", type=float, required=True)
    ap.add_argument("--xB-GP", type=float, default=X_B_GP)
    ap.add_argument("--R-GP-nm", type=float, required=True)
    ap.add_argument("--eta-peak", type=float, required=True)
    ap.add_argument("--eta-interface-width-nm", type=float, required=True)
    ap.add_argument("--depletion-radius-factor", type=float, required=True)
    ap.add_argument("--depletion-smooth-width-factor", type=float, default=0.5)
    ap.add_argument("--profile-type", choices=["tanh", "gaussian", "compact_smooth"], default="tanh")
    ap.add_argument("--depletion-profile-type", choices=["tanh", "gaussian", "compact_smooth"], default="gaussian")
    ap.add_argument("--xB-min", type=float, default=1.0e-6)
    ap.add_argument("--xB-max", type=float, default=0.999999)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = args.Nx, args.Ny, args.Nz
    dx_nm = args.dx_nm
    xbbg = args.xB_background
    xbgp = args.xB_GP
    r_target = args.R_GP_nm
    eta_peak = args.eta_peak
    w_eta = args.eta_interface_width_nm
    dep_factor = args.depletion_radius_factor
    dep_smooth_factor = args.depletion_smooth_width_factor
    xB_min = args.xB_min
    xB_max = args.xB_max

    xs = np.arange(nx) * dx_nm
    ys = np.arange(ny) * dx_nm
    zs = np.arange(nz) * dx_nm
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    Lx, Ly, Lz = nx * dx_nm, ny * dx_nm, nz * dx_nm
    cx, cy, cz = 0.5 * Lx, 0.5 * Ly, 0.5 * Lz
    r = np.sqrt(
        periodic_delta(X, cx, Lx) ** 2 +
        periodic_delta(Y, cy, Ly) ** 2 +
        periodic_delta(Z, cz, Lz) ** 2
    )
    dV = dx_nm ** 3

    v_target = 4.0 * math.pi * r_target ** 3 / 3.0

    def h_volume_for_radius(r_profile: float) -> float:
        eta_trial = np.clip(eta_profile_value(r, r_profile, eta_peak, w_eta, args.profile_type), 0.0, 1.0)
        return float(np.sum(h_switch(eta_trial)) * dV)

    lo, hi = 0.0, max(r_target * 4.0, 0.2)
    while h_volume_for_radius(hi) < v_target and hi < 10.0 * max(Lx, Ly, Lz):
        hi *= 1.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if h_volume_for_radius(mid) < v_target:
            lo = mid
        else:
            hi = mid
    r_profile = 0.5 * (lo + hi)

    eta = np.clip(eta_profile_value(r, r_profile, eta_peak, w_eta, args.profile_type), 0.0, 1.0)
    h_eta = h_switch(eta)
    dep_radius = dep_factor * r_target
    dep_smooth_width = max(dep_smooth_factor * r_target, dx_nm)
    weight = depletion_weight(r, r_target, dep_radius, dep_smooth_width, args.depletion_profile_type)

    xbtot_before = np.full((nx, ny, nz), xbbg, dtype=np.float64)
    xbtot_gp_bg = xbbg + h_eta * (xbgp - xbbg)
    mass_before = float(np.sum(xbtot_before) * dV)
    mass_gp_bg = float(np.sum(xbtot_gp_bg) * dV)
    mass_error = mass_gp_bg - mass_before

    xB = np.full((nx, ny, nz), xbbg, dtype=np.float64)
    if abs(mass_error) > 0.0:
        sign = 1.0 if mass_error > 0.0 else -1.0
        if sign > 0.0:
            a_hi = np.min((xbbg - xB_min) / np.maximum(weight[weight > 0.0], 1.0e-30)) if np.any(weight > 0.0) else 0.0
        else:
            a_hi = np.min((xB_max - xbbg) / np.maximum(weight[weight > 0.0], 1.0e-30)) if np.any(weight > 0.0) else 0.0
        a_lo = 0.0

        def mass_after_amp(amp: float) -> float:
            xb_trial = np.clip(xbbg - sign * amp * weight, xB_min, xB_max)
            xbtot_trial = xb_trial + h_eta * (xbgp - xb_trial)
            return float(np.sum(xbtot_trial) * dV)

        target = mass_before
        if a_hi <= 0.0:
            raise SystemExit("Infeasible depletion/enrichment cloud: zero amplitude capacity")
        m_hi = mass_after_amp(a_hi)
        if (m_hi - target) * (mass_gp_bg - target) > 0.0:
            raise SystemExit("Infeasible depletion/enrichment cloud: amplitude bracket failed")
        for _ in range(80):
            amid = 0.5 * (a_lo + a_hi)
            m_mid = mass_after_amp(amid)
            if (m_mid - target) * (mass_gp_bg - target) > 0.0:
                a_lo = amid
            else:
                a_hi = amid
        amp = 0.5 * (a_lo + a_hi)
        xB = np.clip(xbbg - sign * amp * weight, xB_min, xB_max)
    else:
        amp = 0.0

    xbtot_after = xB + h_eta * (xbgp - xB)
    mass_after = float(np.sum(xbtot_after) * dV)
    mass_error_final = mass_after - mass_before

    phi = np.zeros((nx, ny, nz), dtype=np.float32)
    phi.tofile(out / "phi_init.raw")
    np.asarray(xB, dtype=np.float32).tofile(out / "xB_init.raw")
    np.asarray(eta, dtype=np.float32).tofile(out / "eta_init.raw")
    np.asarray(xbtot_after, dtype=np.float32).tofile(out / "xBtot_gp_init.raw")

    meta = {
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "dx_nm": dx_nm,
        "interface_width_nm": 0.6,
        "dtype": "float32",
        "order": "C",
        "dt_recommended": None,
        "mean_xBtot": float(np.mean(xbtot_after)),
        "xB_max_safe": float(np.max(xB)),
        "reference_type": "observed_gp_profile_bundle",
        "mean_phi": 0.0,
        "mean_h": float(np.mean(h_eta)),
        "voxel_count": int(np.sum(h_eta > 1.0e-12)),
    }
    (out / "init_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    mu_est = estimate_mu_from_xb(xB)
    summary = {
        "mass_before": mass_before,
        "mass_after": mass_after,
        "mass_error": mass_error_final,
        "xB_min": float(np.min(xB)),
        "xB_max": float(np.max(xB)),
        "max_abs_grad_xB": max_abs_grad_periodic(xB, dx_nm),
        "estimated_max_abs_grad_mu": max_abs_grad_periodic(mu_est, dx_nm),
        "V_h": float(np.sum(h_eta) * dV),
        "R_eff_h": float((3.0 * np.sum(h_eta) * dV / (4.0 * math.pi)) ** (1.0 / 3.0)),
        "eta_max": float(np.max(eta)),
        "eta_integral": float(np.sum(eta) * dV),
        "target_radius_nm": r_target,
        "profile_radius_nm": r_profile,
        "eta_peak": eta_peak,
        "match_error": float(np.sum(h_eta) * dV - v_target),
        "depletion_radius_nm": dep_radius,
        "compensation_amplitude": float(amp),
    }
    (out / "init_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    radial_profile_csv(out / "radial_profile_init.csv", r, eta, xB, xbtot_after, weight, dx_nm)


if __name__ == "__main__":
    main()
