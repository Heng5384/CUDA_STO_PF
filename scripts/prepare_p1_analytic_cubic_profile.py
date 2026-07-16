#!/usr/bin/env python3
"""Regenerate the accepted P1 analytic seed for cubic cost profiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import Unit_Psedobinary as unit


def h_poly(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def alpha_stable(phi: np.ndarray) -> np.ndarray:
    return np.where(phi > 0.5, h_poly(1.0 - phi), 1.0 - h_poly(phi))


def build_fields(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    coordinate = np.arange(size, dtype=np.float64)
    delta = coordinate - 0.5 * size
    delta -= np.rint(delta / size) * size
    radius = np.sqrt(
        delta[:, None, None] ** 2
        + delta[None, :, None] ** 2
        + delta[None, None, :] ** 2
    )
    seed_radius = 3.0
    diffuse_half_width = 0.3
    phi = np.clip(
        0.5 * (1.0 + np.tanh((seed_radius - radius) / diffuse_half_width)),
        0.0,
        1.0,
    )
    h_value = h_poly(phi)
    x_b_out = 0.007830539083594734
    x_b_eq = float(unit.xAg2Te_eq_from_T(400.0 + 273.15))
    x_b_alpha = (1.0 - h_value) * x_b_out + h_value * x_b_eq
    alpha = alpha_stable(phi)
    ctot = (1.0 - alpha) + alpha * x_b_alpha
    return phi, x_b_alpha, ctot, x_b_eq


def metadata(size: int) -> dict[str, object]:
    return {
        "schema": "ctot_checkpoint_v1",
        "Nx": size,
        "Ny": size,
        "Nz": size,
        "dx_nm": 1.0,
        "interface_width_nm": 0.6,
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
        "reference_type": "regenerated_accepted_p1_analytic_seed_cost_profile_only",
        "physical_benchmark_evidence": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(32, 64), required=True)
    args = parser.parse_args()
    phi, x_b_alpha, ctot, x_b_eq = build_fields(args.size)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    phi.tofile(out / "phi_init.raw")
    x_b_alpha.tofile(out / "xB_init.raw")
    ctot.tofile(out / "Ctot_init.raw")
    (out / "init_meta.json").write_text(json.dumps(metadata(args.size), indent=2) + "\n")
    metrics = {
        "grid": [args.size, args.size, args.size],
        "profile_role": "representative_3d_cost_profile_only",
        "source_formula": "accepted_P1_R3_w0p3_analytic_sphere",
        "seed_radius_internal": 3.0,
        "diffuse_half_width_internal": 0.3,
        "xB_out": 0.007830539083594734,
        "xB_eq_T400": x_b_eq,
        "phi_min": float(np.min(phi)),
        "phi_max": float(np.max(phi)),
        "xB_min": float(np.min(x_b_alpha)),
        "xB_max": float(np.max(x_b_alpha)),
        "Ctot_min": float(np.min(ctot)),
        "Ctot_max": float(np.max(ctot)),
        "storage_reconstruction_Linf": float(
            np.max(np.abs(ctot - ((1.0 - alpha_stable(phi)) + alpha_stable(phi) * x_b_alpha)))
        ),
        "physical_benchmark_evidence": False,
    }
    (out / "profile_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"p1_analytic_cubic_profile_out={out}")
    print(f"grid={args.size}x{args.size}x{args.size}")
    print(f"xB_eq_T400={x_b_eq:.17e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
