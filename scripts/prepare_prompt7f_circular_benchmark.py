#!/usr/bin/env python3
"""Prepare a mass-consistent 2D circular fixed-Ctot correction benchmark."""

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


def even_grid_at_least(domain_nm: float, dx_nm: float) -> int:
    count = int(math.ceil(domain_nm / dx_nm - 1.0e-12))
    return count if count % 2 == 0 else count + 1


def exp1_positive(values: np.ndarray, quadrature_order: int = 96) -> np.ndarray:
    """Evaluate E1(x)=exp(-x)*integral_0^inf exp(-u)/(x+u) du for x>0."""
    if np.any(values <= 0.0):
        raise ValueError("E1 arguments must be positive")
    nodes, weights = np.polynomial.laguerre.laggauss(quadrature_order)
    flat = values.reshape(-1)
    result = np.empty_like(flat)
    chunk = 16384
    for start in range(0, flat.size, chunk):
        x = flat[start:start + chunk]
        result[start:start + chunk] = (
            np.exp(-x) * np.sum(weights[None, :] / (x[:, None] + nodes[None, :]), axis=1)
        )
    return result.reshape(values.shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--interface-points", type=float, required=True)
    parser.add_argument("--mode", choices=("ctot_mimetic_be", "ctot_fv_be",
                                            "ctot_spectral_be"),
                        required=True)
    parser.add_argument("--finite-interface-correction", type=int,
                        choices=(0, 1), default=1)
    parser.add_argument("--domain-nm", type=float, default=16.0)
    parser.add_argument("--radius-nm", type=float, default=3.0)
    parser.add_argument("--matrix-xB", type=float, default=0.05)
    parser.add_argument("--radial-diffusion-interface-xB", type=float)
    parser.add_argument("--radial-diffusion-start-time-code", type=float, default=0.0)
    parser.add_argument("--phi-raw-source", type=Path)
    parser.add_argument("--dt", type=float, default=6.25e-6)
    parser.add_argument("--nsteps", type=int, default=100)
    parser.add_argument("--test-only-resolution-override", action="store_true")
    args = parser.parse_args()

    if not (args.interface_points > 0.0 and args.domain_nm > 0.0 and
            args.radius_nm > 0.0 and 0.0 < args.matrix_xB < 1.0 and
            args.dt > 0.0 and args.nsteps > 0):
        raise ValueError("invalid circular benchmark inputs")
    radial_diffusion = args.radial_diffusion_start_time_code > 0.0
    if radial_diffusion != (args.radial_diffusion_interface_xB is not None):
        raise ValueError("radial diffusion requires both interface xB and positive start time")
    if radial_diffusion and not (0.0 < args.radial_diffusion_interface_xB < 1.0):
        raise ValueError("invalid radial-diffusion interface xB")
    if args.interface_points < 12.0 and not args.test_only_resolution_override:
        raise ValueError("sub-12-point correction benchmark requires explicit test-only override")

    values = parse_params(args.base_params)
    lambda_nm = float(values["lambda_sm_m"]) * 1.0e9
    dx_nm = lambda_nm / args.interface_points
    nx = even_grid_at_least(args.domain_nm, dx_nm)
    nz = nx
    ny = 2
    actual_domain_nm = nx * dx_nm
    if 2.0 * (args.radius_nm + 2.0 * lambda_nm) >= actual_domain_nm:
        raise ValueError("circle and diffuse-interface buffer do not fit in domain")

    temperature_K = float(values["temperature_C"]) + 273.15
    D_phys = unit.D_Ag_in_PbTe_m2_per_s(temperature_K)
    D_code = float(values["D_alpha"])
    t0 = float(values["t_real_unit"])
    dx_ref_nm = math.sqrt(D_phys * t0 / D_code) * 1.0e9
    dx_code = dx_nm / dx_ref_nm

    coordinates = np.arange(nx, dtype=np.float64) * dx_nm
    cx = 0.5 * actual_domain_nm
    cz = 0.5 * actual_domain_nm
    xx, zz = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.sqrt((xx - cx) ** 2 + (zz - cz) ** 2)
    half_width_nm = 0.5 * lambda_nm
    if args.phi_raw_source is not None:
        phi = np.fromfile(args.phi_raw_source, dtype=np.float64)
        if phi.size != nx * ny * nz:
            raise ValueError("phi raw source size does not match benchmark grid")
        phi = phi.reshape((nx, ny, nz))
        if not np.all(np.isfinite(phi)) or np.min(phi) < 0.0 or np.max(phi) > 1.0:
            raise ValueError("phi raw source is nonfinite or outside [0,1]")
    else:
        phi_2d = 0.5 * (1.0 - np.tanh((radius - args.radius_nm) / half_width_nm))
        phi = np.repeat(phi_2d[:, None, :], ny, axis=1)
    expected_sharp_velocity_code = math.nan
    if radial_diffusion:
        radius_code = radius * (dx_code / dx_nm)
        seed_radius_code = args.radius_nm * (dx_code / dx_nm)
        u = np.maximum(radius_code, seed_radius_code) ** 2 / (
            4.0 * D_code * args.radial_diffusion_start_time_code
        )
        u_seed = seed_radius_code ** 2 / (
            4.0 * D_code * args.radial_diffusion_start_time_code
        )
        E1 = exp1_positive(u)
        E1_seed = float(exp1_positive(np.array([u_seed]))[0])
        interface_xB = float(args.radial_diffusion_interface_xB)
        xB_2d = args.matrix_xB + (interface_xB - args.matrix_xB) * E1 / E1_seed
        xB_2d[radius <= args.radius_nm] = interface_xB
        gradient_at_seed = ((interface_xB - args.matrix_xB) *
                            (-2.0 * math.exp(-u_seed) / seed_radius_code) /
                            E1_seed)
        expected_sharp_velocity_code = (
            D_code * gradient_at_seed / (1.0 - interface_xB)
        )
        xB = np.repeat(xB_2d[:, None, :], ny, axis=1)
    else:
        xB = np.full_like(phi, args.matrix_xB)
    h = phi ** 3 * (6.0 * phi ** 2 - 15.0 * phi + 10.0)
    Ctot = (1.0 - h) * xB + h

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    phi.astype(np.float64).tofile(out / "phi_init.raw")
    xB.astype(np.float64).tofile(out / "xB_init.raw")
    Ctot.astype(np.float64).tofile(out / "Ctot_init.raw")

    production_allowed = args.interface_points >= 12.0 and not args.test_only_resolution_override
    meta = {
        "Nx": nx, "Ny": ny, "Nz": nz,
        "dx_nm": dx_nm,
        "interface_width_nm": lambda_nm,
        "dtype": "float64", "order": "C",
        "dt_recommended": args.dt,
        "mean_xBtot": float(Ctot.mean()),
        "xB_max_safe": args.matrix_xB,
        "reference_type": "periodic_2d_circular_beta_fixed_ctot",
        "radius_nm": args.radius_nm,
        "R_over_lambda": args.radius_nm / lambda_nm,
        "matrix_xB": args.matrix_xB,
        "radial_diffusion_enabled": int(radial_diffusion),
        "radial_diffusion_interface_xB": args.radial_diffusion_interface_xB,
        "radial_diffusion_start_time_code": args.radial_diffusion_start_time_code,
        "expected_sharp_velocity_code": expected_sharp_velocity_code,
        "phi_source_mode": ("pre_relaxed_raw" if args.phi_raw_source is not None
                            else "analytic_tanh_circle"),
        "phi_raw_source": (str(args.phi_raw_source.resolve())
                           if args.phi_raw_source is not None else None),
        "runtime_lambda_over_dx": args.interface_points,
        "finite_interface_calibration_min_points": 10.0,
        "finite_interface_calibration_max_points": 12.0,
        "finite_interface_production_min_points": 12.0,
        "finite_interface_resolution_test_override": int(
            args.test_only_resolution_override),
        "production_acceptance_allowed": int(production_allowed),
        "provenance": "prompt7f_curved_interface_diagnostic_not_3d_production",
    }
    (out / "init_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    text = args.base_params.read_text()
    if not text.endswith("\n"):
        text += "\n"
    overlay = {
        "dx": dx_code, "dy": dx_code, "dz": dx_code,
        "ic_phi_iface_w": 0.5 * args.interface_points,
        "dt": args.dt,
        "composition_evolution_mode": args.mode,
        "ctot_finite_interface_antitrapping_enabled":
            args.finite_interface_correction,
        "finite_interface_calibration_min_points": 10.0,
        "finite_interface_calibration_max_points": 12.0,
        "finite_interface_production_min_points": 12.0,
        "finite_interface_resolution_test_override": int(
            args.test_only_resolution_override),
        "ctot_nonlinear_max_iter": 500,
        "ctot_step_max_retries": 0,
        "ctot_automatic_dt_growth": 0,
        "ctot_outer_max_iter": 24,
        "ctot_elastic_validation_enabled": 0,
        "elastic_enabled": 0,
    }
    text += "\n# Prompt 7f circular correction benchmark overlay\n"
    for key, value in overlay.items():
        if isinstance(value, int):
            text += f"{key}={value}\n"
        elif isinstance(value, str):
            text += f"{key}={value}\n"
        else:
            text += f"{key}={float(value):.17e}\n"
    (out / "benchmark.params").write_text(text)

    manifest = dict(meta)
    manifest.update({
        "grid": [nx, ny, nz],
        "actual_domain_nm": actual_domain_nm,
        "dx_code": dx_code,
        "mode": args.mode,
        "nsteps": args.nsteps,
        "dt_code": args.dt,
        "final_code_time": args.dt * args.nsteps,
        "final_physical_time_s": args.dt * args.nsteps * t0,
        "finite_interface_correction": args.finite_interface_correction,
        "D_beta": 0.0,
        "selected_L_phi_code": float(values["L_phi_code_value"]),
    })
    (out / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(out)
    print(f"grid={nx}x{ny}x{nz}")
    print(f"runtime_lambda_over_dx={args.interface_points:.17e}")
    print(f"production_acceptance_allowed={int(production_allowed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
