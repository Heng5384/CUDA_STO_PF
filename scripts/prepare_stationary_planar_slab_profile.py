#!/usr/bin/env python3
"""Prepare a nonuniform stationary periodic slab for cubic cost profiling."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = ROOT / "scripts" / "prepare_stationary_curved_equilibrium.py"
SPEC = importlib.util.spec_from_file_location("stationary_oracle", ORACLE_PATH)
oracle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(oracle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(32, 64), required=True)
    args = parser.parse_args()
    params = oracle.parse_params(args.base_params)
    n = args.size
    dx_code = float(params["dx"])
    dx_nm = float(params["lambda_sm_m"]) * 1.0e9 / 12.0
    temperature_k = float(params["temperature_C"]) + 273.15
    energy_scale = float(params["mu_reference_scale"])
    x_eq = oracle.unit.xAg2Te_eq_from_T(temperature_k)
    mu0 = oracle.unit.mu_Ag2Te(temperature_k, x_eq) / energy_scale
    coordinate = (np.arange(n, dtype=np.float64) - n // 2) * dx_code
    distance = np.abs(coordinate)
    slab_half_width = 0.25 * n * dx_code
    diffuse_half_width = math.sqrt(
        2.0 * float(params["kappa_phi"]) / float(params["W"])
    )
    line = 0.5 * (
        1.0 - np.tanh((distance - slab_half_width) / diffuse_half_width)
    )
    plane = np.repeat(line[:, None], n, axis=1)
    target_h_cells = float(np.sum(oracle.h_stable(plane)))
    plane, delta_mu, core, refinement = (
        oracle.refine_production_spectral_equilibrium(
            plane,
            0.0,
            x_eq,
            target_h_cells,
            dx_code,
            float(params["W"]),
            float(params["kappa_phi"]),
            3.0e-8,
            max_newton=100,
        )
    )
    matrix_x = oracle.matrix_x_from_delta_mu(
        delta_mu, temperature_k, energy_scale, mu0
    )
    line = plane[:, 0]
    phi = np.broadcast_to(line[:, None, None], (n, n, n)).copy()
    x_field = np.full_like(phi, matrix_x)
    alpha = oracle.alpha_stable(phi)
    ctot = np.where(
        phi > 0.5,
        1.0 - alpha * (1.0 - matrix_x),
        oracle.h_stable(phi) + alpha * matrix_x,
    )
    residual = (
        float(params["W"]) * oracle.gp(plane)
        + delta_mu * oracle.hp(plane)
        - float(params["kappa_phi"]) *
          oracle.spectral_laplacian_2d(plane, dx_code)
    )
    projected_kkt = oracle.projected_kkt_linf(
        plane,
        np.where(
            plane > 0.5,
            1.0 - oracle.alpha_stable(plane) * (1.0 - matrix_x),
            oracle.h_stable(plane) + oracle.alpha_stable(plane) * matrix_x,
        ),
        residual,
        6.25e-6,
        float(params["L_phi"]),
        1.0,
        1.0e-8,
        1.0 - 1.0e-8,
    )
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    phi.tofile(out / "phi_init.raw")
    x_field.tofile(out / "xB_init.raw")
    ctot.tofile(out / "Ctot_init.raw")
    metadata = {
        "schema": "ctot_checkpoint_v1",
        "Nx": n,
        "Ny": n,
        "Nz": n,
        "dx_nm": dx_nm,
        "interface_width_nm": float(params["lambda_sm_m"]) * 1.0e9,
        "dtype": "float64",
        "order": "C",
        "authoritative_state": "Ctot",
        "transport_operator_name": "mimetic_shared_face_v1",
        "transport_operator_version": "1",
        "gradient_operator_name": "periodic_positive_face_difference_v1",
        "divergence_operator_name": "periodic_face_incidence_v1",
        "adjoint_identity_mode": "D_equals_negative_G_star_exact",
        "face_mobility_mode": "symmetric_harmonic_zero_endpoint_v1",
        "phase_solver_name": "semismooth_pdas_v1",
        "phase_solver_version": "1",
        "finite_interface_correction_enabled": 0,
        "step": 0,
        "time_code": 0.0,
        "migrated_from_legacy_restart": False,
        "reference_type": "stationary_periodic_planar_slab_profile_only",
    }
    (out / "init_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metrics = {
        "grid": [n, n, n],
        "profile_role": "nonuniform_stationary_cubic_cost_profile_only",
        "matrix_xB": matrix_x,
        "delta_mu": delta_mu,
        "phi_min": float(np.min(phi)),
        "phi_max": float(np.max(phi)),
        "free_residual_Linf": float(np.max(np.abs(residual[~core]))),
        "projected_KKT_Linf": projected_kkt,
        "storage_reconstruction_Linf": 0.0,
        "refinement": refinement,
        "physical_benchmark_evidence": False,
    }
    (out / "profile_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"stationary_planar_profile_out={out}")
    print(f"grid={n}x{n}x{n}")
    print(f"projected_KKT_Linf={projected_kkt:.17e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
