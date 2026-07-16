#!/usr/bin/env python3
"""Prepare a periodic planar beta slab for the one-sided Ctot benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402


def parse_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def sharp_similarity_parameter(x_inf: float, x_eq: float, x_beta: float = 1.0) -> float:
    ratio = (x_inf - x_eq) / (x_beta - x_eq)
    if not (0.0 < ratio < 1.0):
        raise ValueError("sharp-interface supersaturation ratio must lie in (0,1)")
    def residual(value: float) -> float:
        return (math.sqrt(math.pi) * value * math.exp(value * value) *
                math.erfc(value) - ratio)
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dx-nm", type=float, required=True)
    parser.add_argument("--domain-nm", type=float, default=19.2)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--nz", type=int, default=2)
    parser.add_argument("--matrix-xB", type=float, default=0.05)
    parser.add_argument("--sharp-start-time-s", type=float, default=0.1)
    parser.add_argument("--lphi-factor", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--nsteps", type=int, default=10)
    parser.add_argument("--mode", choices=("ctot_mimetic_be", "ctot_fv_be",
                                            "ctot_spectral_be"),
                        default="ctot_fv_be")
    parser.add_argument("--finite-interface-antitrapping", action="store_true")
    parser.add_argument("--outer-max-iter", type=int, default=16)
    args = parser.parse_args()
    if not (args.dx_nm > 0.0 and args.domain_nm > 0.0 and
            0.0 < args.matrix_xB < 0.09 and args.lphi_factor > 0.0 and
            args.outer_max_iter > 0):
        raise ValueError("invalid benchmark geometry/composition/Lphi factor")

    values = parse_params(args.base_params)
    if values.get("L_phi_calibration_mode") != "one_sided_diffusion_controlled":
        raise ValueError("base params must use one_sided_diffusion_controlled")
    temperature_K = float(values["temperature_C"]) + 273.15
    D_phys = unit.D_Ag_in_PbTe_m2_per_s(temperature_K)
    D_code = float(values["D_alpha"])
    t0 = float(values["t_real_unit"])
    dx_ref_nm = math.sqrt(D_phys * t0 / D_code) * 1.0e9
    dx_code = args.dx_nm / dx_ref_nm
    lambda_nm = float(values["lambda_sm_m"]) * 1.0e9
    half_width_nm = 0.5 * lambda_nm
    nx = int(round(args.domain_nm / args.dx_nm))
    if nx < 32 or nx % 2:
        raise ValueError("domain/dx must produce an even Nx >= 32")
    actual_domain_nm = nx * args.dx_nm

    x = np.arange(nx, dtype=np.float64) * args.dx_nm
    left = 0.35 * actual_domain_nm
    right = 0.65 * actual_domain_nm
    phi_1d = 0.5 * (
        np.tanh((x - left) / half_width_nm) -
        np.tanh((x - right) / half_width_nm)
    )
    phi_1d = np.clip(phi_1d, 0.0, 1.0)
    phi = np.broadcast_to(phi_1d[:, None, None],
                          (nx, args.ny, args.nz)).copy()
    x_eq = unit.xAg2Te_eq_from_T(temperature_K)
    sharp_eta = sharp_similarity_parameter(args.matrix_xB, x_eq)
    if not (args.sharp_start_time_s > 0.0):
        raise ValueError("sharp-start-time-s must be positive")
    diffusion_length_nm = math.sqrt(D_phys * args.sharp_start_time_s) * 1.0e9
    xB_1d = np.full(nx, x_eq, dtype=np.float64)
    for idx, coordinate in enumerate(x):
        if coordinate < left:
            distance = left - coordinate
        elif coordinate > right:
            distance = coordinate - right
        else:
            continue
        eta = sharp_eta + distance / (2.0 * diffusion_length_nm)
        normalized = (math.erf(eta) - math.erf(sharp_eta)) / math.erfc(sharp_eta)
        xB_1d[idx] = x_eq + (args.matrix_xB - x_eq) * normalized
    xB = np.broadcast_to(xB_1d[:, None, None],
                         (nx, args.ny, args.nz)).copy()
    h = phi ** 3 * (6.0 * phi ** 2 - 15.0 * phi + 10.0)
    Ctot = (1.0 - h) * xB + h

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    phi.astype(np.float64).tofile(out / "phi_init.raw")
    xB.astype(np.float64).tofile(out / "xB_init.raw")
    Ctot.astype(np.float64).tofile(out / "Ctot_init.raw")
    meta = {
        "Nx": nx,
        "Ny": args.ny,
        "Nz": args.nz,
        "dx_nm": args.dx_nm,
        "interface_width_nm": lambda_nm,
        "dtype": "float64",
        "order": "C",
        "dt_recommended": args.dt,
        "mean_xBtot": float(Ctot.mean()),
        "xB_max_safe": args.matrix_xB,
        "reference_type": "periodic_planar_beta_slab_one_sided",
        "matrix_xB": args.matrix_xB,
        "xB_eq": x_eq,
        "sharp_start_time_s": args.sharp_start_time_s,
        "sharp_similarity_parameter": sharp_eta,
        "left_interface_nm": left,
        "right_interface_nm": right,
        "beta_slab_width_nm": right - left,
    }
    (out / "init_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    text = args.base_params.read_text()
    if not text.endswith("\n"):
        text += "\n"
    selected_lphi = float(values["L_phi_code_value"]) * args.lphi_factor
    selected_lphi_phys = float(values["L_phi_physical_value"]) * args.lphi_factor
    overlay = {
        "dx": dx_code,
        "dy": dx_code,
        "dz": dx_code,
        "ic_phi_iface_w": lambda_nm / (2.0 * args.dx_nm),
        "dt": args.dt,
        "composition_evolution_mode": args.mode,
        "L_phi": selected_lphi,
        "L_phi_code_value": selected_lphi,
        "L_phi_physical_value": selected_lphi_phys,
        "ctot_nonlinear_max_iter": 500,
        "ctot_step_max_retries": 8,
        "ctot_elastic_validation_enabled": 0,
        "elastic_enabled": 0,
        "ctot_residual_abs_tol": 1.0e-10,
        "ctot_residual_rel_tol": 1.0e-8,
        "ctot_automatic_dt_growth": 0,
        "ctot_outer_max_iter": args.outer_max_iter,
        "ctot_finite_interface_antitrapping_enabled":
            int(args.finite_interface_antitrapping),
    }
    text += "\n# One-sided planar benchmark numerical overlay\n"
    for key, value in overlay.items():
        if isinstance(value, str):
            text += f"{key}={value}\n"
        elif isinstance(value, int):
            text += f"{key}={value}\n"
        else:
            text += f"{key}={float(value):.17e}\n"
    (out / "benchmark.params").write_text(text)
    manifest = {
        "grid": [nx, args.ny, args.nz],
        "dx_nm": args.dx_nm,
        "dx_code": dx_code,
        "domain_nm": actual_domain_nm,
        "lambda_nm": lambda_nm,
        "interface_resolution": lambda_nm / args.dx_nm,
        "matrix_xB": args.matrix_xB,
        "xB_eq": x_eq,
        "sharp_start_time_s": args.sharp_start_time_s,
        "sharp_similarity_parameter": sharp_eta,
        "sharp_diffusion_length_nm": diffusion_length_nm,
        "L_phi_factor": args.lphi_factor,
        "L_phi_code": selected_lphi,
        "dt_code": args.dt,
        "nsteps": args.nsteps,
        "final_code_time": args.dt * args.nsteps,
        "final_physical_time_s": args.dt * args.nsteps * t0,
        "mode": args.mode,
        "finite_interface_antitrapping_enabled":
            args.finite_interface_antitrapping,
        "ctot_outer_max_iter": args.outer_max_iter,
        "D_beta": 0.0,
        "provenance": "one_sided_planar_numerical_benchmark_not_growth_tuning",
    }
    (out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)
    print(f"Nx={nx}")
    print(f"interface_resolution={lambda_nm / args.dx_nm:.17e}")
    print(f"L_phi_code={selected_lphi:.17e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
