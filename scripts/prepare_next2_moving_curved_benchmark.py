#!/usr/bin/env python3
"""Prepare identical OFF/ON and dt/dt2 moving curved PF initial states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.next2_finite_box_sharp_oracle import (  # noqa: E402
    add_far_inventory_correction,
    aged_no_flux_profile,
    parse_params,
    profile_inventory,
    solve_gibbs_thomson_x,
    square_equivalent_outer_radius,
)
import Unit_Psedobinary as unit  # noqa: E402


RATIOS = (5, 8, 10, 15, 20)


def h_stable(phi: np.ndarray) -> np.ndarray:
    direct = phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)
    reflected_phi = 1.0 - phi
    reflected = reflected_phi**3 * (
        6.0 * reflected_phi**2 - 15.0 * reflected_phi + 10.0
    )
    return np.where(phi > 0.5, 1.0 - reflected, direct)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_params(base: Path, output: Path, values: dict[str, object]) -> None:
    text = base.read_text()
    if not text.endswith("\n"):
        text += "\n"
    text += "\n# Next2 finite-box moving curved benchmark overlay\n"
    for key, value in values.items():
        if isinstance(value, str):
            text += f"{key}={value}\n"
        elif isinstance(value, int):
            text += f"{key}={value}\n"
        else:
            text += f"{key}={float(value):.17e}\n"
    output.write_text(text)


def hardlink(source: Path, target: Path) -> None:
    if target.exists():
        target.unlink()
    os.link(source, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stationary-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--far-xB", type=float, default=0.0078305391025)
    parser.add_argument("--diffusion-age-code", type=float, default=4.0 / 9.0)
    parser.add_argument("--dt", type=float, default=6.25e-6)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--profile-nodes", type=int, default=2048)
    args = parser.parse_args()
    if args.dt <= 0.0 or args.steps <= 0 or args.profile_nodes < 128:
        raise ValueError("invalid moving benchmark controls")
    final_time = args.dt * args.steps
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    shared_root = root / "shared"
    shared_root.mkdir()
    cases: list[dict[str, object]] = []

    for ratio in RATIOS:
        source = args.stationary_root.resolve() / f"r{ratio}"
        stationary = json.loads((source / "benchmark_manifest.json").read_text())
        params = parse_params(source / "benchmark.params")
        nx, ny, nz = (int(value) for value in stationary["grid"])
        shape = (nx, ny, nz)
        dx_nm = float(stationary["dx_nm"])
        box_nm = nx * dx_nm
        radius_nm = float(stationary["h_volume_radius_nm"])
        lambda_nm = float(stationary["target_radius_nm"]) / ratio
        temperature_k = float(params["temperature_C"]) + 273.15
        surface_x = solve_gibbs_thomson_x(
            temperature_k,
            radius_nm,
            float(params["gamma_Jm2"]),
            unit.USER_PHYSICAL_INPUTS.Vm_compound,
            float(params["mu_reference_scale"]),
        )
        outer_radius = square_equivalent_outer_radius(box_nm)
        radial_nodes, base_profile = aged_no_flux_profile(
            radius_nm,
            outer_radius,
            float(params["D_alpha"]),
            surface_x,
            args.far_xB,
            args.diffusion_age_code,
            args.profile_nodes,
        )
        phi_values = np.fromfile(source / "phi_init.raw", dtype=np.float64)
        if phi_values.size != math.prod(shape):
            raise ValueError(f"stationary phi shape mismatch for r{ratio}")
        phi = phi_values.reshape(shape)
        h_phi = h_stable(phi)
        coordinates = np.arange(nx, dtype=np.float64) * dx_nm
        xx, zz = np.meshgrid(coordinates, coordinates, indexing="ij")
        rr = np.sqrt(
            (xx - 0.5 * box_nm) ** 2 + (zz - 0.5 * box_nm) ** 2
        )
        x_2d = np.interp(
            np.minimum(rr, outer_radius), radial_nodes, base_profile
        )
        x_2d[rr <= radius_nm] = surface_x
        x_field = np.repeat(x_2d[:, None, :], ny, axis=1)
        ctot = h_phi + (1.0 - h_phi) * x_field
        pf_inventory_nm2 = float(np.sum(ctot)) * dx_nm**2 / ny
        corrected_profile, inventory_delta_x, matched_inventory = (
            add_far_inventory_correction(
                radius_nm,
                radial_nodes,
                base_profile,
                pf_inventory_nm2,
                protected_width=2.0 * lambda_nm,
            )
        )
        initial_sharp_inventory = profile_inventory(
            radius_nm, radial_nodes, base_profile
        )
        if abs(matched_inventory - pf_inventory_nm2) > 5.0e-12 * max(
            1.0, abs(pf_inventory_nm2)
        ):
            raise RuntimeError(f"finite-box inventory match failed for r{ratio}")
        if np.min(x_field) <= 0.0 or np.max(x_field) >= 1.0:
            raise RuntimeError(f"moving xB field out of bounds for r{ratio}")

        shared = shared_root / f"r{ratio}"
        shared.mkdir()
        phi.astype(np.float64).tofile(shared / "phi_init.raw")
        x_field.astype(np.float64).tofile(shared / "xB_init.raw")
        ctot.astype(np.float64).tofile(shared / "Ctot_init.raw")
        np.savez_compressed(
            shared / "sharp_initial_profile.npz",
            radius_nm=radial_nodes,
            xB_uncorrected=base_profile,
            xB_inventory_matched=corrected_profile,
        )
        shared_manifest = {
            "R_over_lambda": ratio,
            "grid": [nx, ny, nz],
            "dx_nm": dx_nm,
            "dx_code": float(params["dx"]),
            "box_nm": box_nm,
            "lambda_nm": lambda_nm,
            "radius_h_nm": radius_nm,
            "sharp_outer_radius_equal_area_nm": outer_radius,
            "surface_xB_Gibbs_Thomson": surface_x,
            "far_xB_requested": args.far_xB,
            "diffusion_age_code": args.diffusion_age_code,
            "pf_inventory_nm2": pf_inventory_nm2,
            "sharp_inventory_before_match_nm2": initial_sharp_inventory,
            "sharp_inventory_after_match_nm2": matched_inventory,
            "sharp_inventory_match_delta_xB": inventory_delta_x,
            "sharp_inventory_protected_width_nm": 2.0 * lambda_nm,
            "phi_sha256": sha256(shared / "phi_init.raw"),
            "xB_sha256": sha256(shared / "xB_init.raw"),
            "Ctot_sha256": sha256(shared / "Ctot_init.raw"),
            "provenance": (
                "finite_box_equal_area_cylindrical_no_flux_sharp_profile_"
                "matched_to_periodic_PF_total_inventory_no_PF_velocity_fit"
            ),
        }
        (shared / "moving_state_manifest.json").write_text(
            json.dumps(shared_manifest, indent=2) + "\n"
        )

        source_meta = json.loads((source / "init_meta.json").read_text())
        for correction in (0, 1):
            for dt_label, dt, steps in (
                ("dt", args.dt, args.steps),
                ("dt2", 0.5 * args.dt, 2 * args.steps),
            ):
                label = f"Rlambda{ratio}_corr{correction}_{dt_label}"
                case = root / label
                input_dir = case / "input"
                input_dir.mkdir(parents=True)
                for field in ("phi", "xB", "Ctot"):
                    hardlink(
                        shared / f"{field}_init.raw",
                        input_dir / f"{field}_init.raw",
                    )
                metadata = dict(source_meta)
                metadata.update({
                    "finite_interface_correction_enabled": correction,
                    "step": 0,
                    "time_code": 0.0,
                    "mean_Ctot": float(np.mean(ctot)),
                    "reference_type": (
                        "next2_moving_curved_fixed_total_Ctot_periodic_PF_"
                        "matched_to_independent_finite_box_sharp_oracle"
                    ),
                    "sharp_reference_ensemble": (
                        "equal_area_finite_outer_cylinder_no_flux"
                    ),
                    "sharp_initial_inventory_nm2": matched_inventory,
                    "moving_state_manifest_sha256": sha256(
                        shared / "moving_state_manifest.json"
                    ),
                })
                (input_dir / "init_meta.json").write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )
                append_params(
                    source / "benchmark.params",
                    input_dir / "benchmark.params",
                    {
                        "dt": dt,
                        "composition_evolution_mode": "ctot_mimetic_be",
                        "ctot_phase_semismooth_pdas_enabled": 1,
                        "ctot_finite_interface_antitrapping_enabled": correction,
                        "finite_interface_resolution_test_override": 0,
                        "ctot_step_max_retries": 0,
                        "ctot_automatic_dt_growth": 0,
                        "ctot_elastic_validation_enabled": 0,
                        "elastic_enabled": 0,
                        "ctot_performance_profile_enabled": 0,
                    },
                )
                manifest = {
                    **shared_manifest,
                    "case": label,
                    "mode": "ctot_mimetic_be",
                    "runtime_lambda_over_dx": lambda_nm / dx_nm,
                    "interface_width_nm": lambda_nm,
                    "finite_interface_correction": correction,
                    "production_acceptance_allowed": 1,
                    "dt_code": dt,
                    "nsteps": steps,
                    "output_every": max(1, steps // 20),
                    "final_code_time": final_time,
                    "final_physical_time_s": final_time * float(params["t_real_unit"]),
                    "initial_state_raw_hashes_identical_across_pair": True,
                    "source_stationary_manifest_sha256": sha256(
                        source / "benchmark_manifest.json"
                    ),
                    "provenance": (
                        "next2_moving_curved_P2_no_fit_same_raw_state_OFF_ON_dt_dt2"
                    ),
                }
                (input_dir / "benchmark_manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
                cases.append(manifest)
    (root / "next2_moving_manifest.json").write_text(
        json.dumps(cases, indent=2) + "\n"
    )
    print(f"moving_cases_prepared={len(cases)}")
    print(f"moving_root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
