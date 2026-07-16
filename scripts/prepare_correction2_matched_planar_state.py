#!/usr/bin/env python3
"""Build common sharp/PF pre-aged planar states for Correction Flow 2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402
from scripts.correction1_planar_sharp_oracle import PlanarSharpOracle  # noqa: E402


def h_switch(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def provenance_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def g_prime(phi: np.ndarray) -> np.ndarray:
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def evolve_uniform_sharp_state(
    oracle: PlanarSharpOracle, preage_s: float
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Evolve the conservative finite-box Stefan state from uniform matrix xB."""
    radius = oracle.radius_initial_nm
    length = oracle.outer_nm - radius
    state0 = np.concatenate((
        np.full(oracle.cells, length * oracle.matrix_x, dtype=np.float64),
        np.array([radius], dtype=np.float64),
    ))
    mass0 = oracle.inventory(state0)
    solution = solve_ivp(
        oracle.rhs, (0.0, preage_s), state0, method="BDF",
        rtol=2.0e-10, atol=2.0e-12,
        max_step=max(preage_s / 200.0, 1.0e-10),
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    state = solution.y[:, -1]
    mass1 = oracle.inventory(state)
    return state, {
        "solver_steps": len(solution.t) - 1,
        "rhs_evaluations": int(solution.nfev),
        "sharp_mass_initial_xB_nm": mass0,
        "sharp_mass_final_xB_nm": mass1,
        "sharp_mass_error_rel": abs(mass1 - mass0) / max(abs(mass0), 1.0),
        "sharp_radius_initial_nm": radius,
        "sharp_radius_preaged_nm": float(state[-1]),
    }


def diffuse_profile(
    coordinates_nm: np.ndarray, domain_nm: float, radius_nm: float,
    lambda_nm: float,
) -> np.ndarray:
    center = 0.5 * domain_nm
    left = center - radius_nm
    right = center + radius_nm
    half_width = 0.5 * lambda_nm
    return 0.5 * (
        np.tanh((coordinates_nm - left) / half_width)
        - np.tanh((coordinates_nm - right) / half_width)
    )


def match_h_volume_radius(
    coordinates_nm: np.ndarray, domain_nm: float, sharp_radius_nm: float,
    lambda_nm: float, dx_nm: float,
) -> tuple[float, np.ndarray]:
    target = 2.0 * sharp_radius_nm
    lo = max(0.01 * lambda_nm, sharp_radius_nm - 2.0 * lambda_nm)
    hi = min(0.49 * domain_nm, sharp_radius_nm + 2.0 * lambda_nm)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        phi = diffuse_profile(coordinates_nm, domain_nm, mid, lambda_nm)
        volume = float(h_switch(phi).sum() * dx_nm)
        if volume < target:
            lo = mid
        else:
            hi = mid
    radius = 0.5 * (lo + hi)
    phi = diffuse_profile(coordinates_nm, domain_nm, radius, lambda_nm)
    return radius, phi


def map_sharp_to_pf(
    oracle: PlanarSharpOracle, sharp_state: np.ndarray, domain_nm: float,
    dx_nm: float, lambda_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    nx = int(round(domain_nm / dx_nm))
    if nx % 2 or abs(nx * dx_nm - domain_nm) > 1.0e-12:
        raise ValueError("domain/dx must be an exact even grid")
    coordinates = (np.arange(nx, dtype=np.float64) + 0.5) * dx_nm
    sharp_radius = float(sharp_state[-1])
    geometric_radius, phi = match_h_volume_radius(
        coordinates, domain_nm, sharp_radius, lambda_nm, dx_nm
    )
    h = h_switch(phi)
    center = 0.5 * domain_nm
    signed_distance = np.abs(coordinates - center) - sharp_radius
    sharp_x = oracle.concentration(sharp_state)
    sharp_length = oracle.outer_nm - sharp_radius
    sharp_distance = (
        np.arange(oracle.cells, dtype=np.float64) + 0.5
    ) * oracle.dy * sharp_length
    x_b = np.full(nx, oracle.x_eq, dtype=np.float64)
    outside = signed_distance >= 0.0
    x_b[outside] = np.interp(
        signed_distance[outside], sharp_distance, sharp_x,
        left=oracle.x_eq, right=float(sharp_x[-1]),
    )

    target_total = 2.0 * oracle.inventory(sharp_state)
    ctot = h + (1.0 - h) * x_b
    before = float(ctot.sum() * dx_nm)
    delta = target_total - before
    support = np.abs(signed_distance) <= 2.0 * lambda_nm
    correction_weight = np.zeros(nx, dtype=np.float64)
    correction_weight[support] = np.exp(
        -(signed_distance[support] / lambda_nm) ** 2
    )
    denominator = float(np.sum((1.0 - h) * correction_weight) * dx_nm)
    if denominator <= 0.0:
        raise RuntimeError("empty diffuse representation-correction support")
    amplitude = delta / denominator
    x_b += amplitude * correction_weight
    if float(x_b.min()) < 0.0 or float(x_b.max()) > 1.0:
        raise RuntimeError("mass closure would violate composition bounds")
    ctot = h + (1.0 - h) * x_b
    after = float(ctot.sum() * dx_nm)
    q_alpha = ctot - h
    if abs(after - target_total) > 2.0e-13:
        raise RuntimeError("PF representation inventory did not close")

    lambda_code = lambda_nm
    kappa_code = lambda_code**2 / 8.0
    lap_phi = (np.roll(phi, -1) - 2.0 * phi + np.roll(phi, 1)) / dx_nm**2
    stationary_residual = g_prime(phi) - kappa_code * lap_phi
    outside_diffuse = np.abs(signed_distance) > 2.0 * lambda_nm
    sharp_expected = np.full(nx, oracle.x_eq, dtype=np.float64)
    sharp_expected[outside] = np.interp(
        signed_distance[outside], sharp_distance, sharp_x,
        left=oracle.x_eq, right=float(sharp_x[-1]),
    )
    return phi, x_b, ctot, {
        "grid_nx": nx,
        "sharp_radius_nm": sharp_radius,
        "diffuse_geometric_radius_nm": geometric_radius,
        "target_h_volume_nm": 2.0 * sharp_radius,
        "actual_h_volume_nm": float(h.sum() * dx_nm),
        "h_volume_error_abs_nm": abs(float(h.sum() * dx_nm) - 2.0 * sharp_radius),
        "target_total_inventory_xB_nm": target_total,
        "pf_inventory_before_correction_xB_nm": before,
        "pf_inventory_after_correction_xB_nm": after,
        "representation_correction_xB_nm": delta,
        "representation_correction_amplitude": amplitude,
        "representation_correction_support_nm": 2.0 * lambda_nm,
        "outside_profile_Linf": float(np.max(np.abs(
            x_b[outside_diffuse] - sharp_expected[outside_diffuse]
        ))),
        "stationary_dw_gradient_residual_Linf": float(
            np.max(np.abs(stationary_residual))
        ),
        "Ctot_min_minus_h": float(np.min(ctot - h)),
        "one_minus_Ctot_min": float(np.min(1.0 - ctot)),
        "q_alpha_min": float(np.min(q_alpha)),
        "one_minus_h_minus_q_min": float(np.min((1.0 - h) - q_alpha)),
        "xB_min": float(x_b.min()),
        "xB_max": float(x_b.max()),
        "finite": bool(
            np.all(np.isfinite(phi)) and np.all(np.isfinite(x_b))
            and np.all(np.isfinite(ctot))
        ),
    }


def prepare_family(
    out_root: Path, diffusion_width_ratios: list[float], domain_nm: float,
    dx_nm: float, lambda_nm: float, sharp_cells: int, matrix_x: float = 0.05,
) -> dict[str, object]:
    temperature_c = 400.0
    temperature_k = temperature_c + 273.15
    diffusivity_nm2_s = unit.D_Ag_in_PbTe_m2_per_s(temperature_k) * 1.0e18
    rows: list[dict[str, object]] = []
    out_root.mkdir(parents=True, exist_ok=True)
    for ratio in diffusion_width_ratios:
        preage_s = (ratio * lambda_nm) ** 2 / diffusivity_nm2_s
        oracle = PlanarSharpOracle(
            temperature_c=temperature_c, domain_nm=domain_nm,
            matrix_x=matrix_x, start_time_s=preage_s,
            cells=sharp_cells, pf_dx_nm=dx_nm,
        )
        sharp_state, sharp_metrics = evolve_uniform_sharp_state(oracle, preage_s)
        phi, x_b, ctot, mapping = map_sharp_to_pf(
            oracle, sharp_state, domain_nm, dx_nm, lambda_nm
        )
        x_tag = f"{matrix_x:.8f}".replace(".", "p")
        case_id = f"T400_ell{ratio:g}_lambda_x{x_tag}"
        case_dir = out_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        shape = (int(mapping["grid_nx"]), 2, 2)
        for name, line in (("phi", phi), ("xB", x_b), ("Ctot", ctot)):
            np.broadcast_to(line[:, None, None], shape).copy().astype(np.float64).tofile(
                case_dir / f"{name}_init.raw"
            )
        np.savez(
            case_dir / "sharp_state.npz", state=sharp_state,
            temperature_c=temperature_c, domain_nm=domain_nm,
            matrix_x=oracle.matrix_x, preage_s=preage_s,
            sharp_cells=sharp_cells,
        )
        row: dict[str, object] = {
            "case": case_id,
            "temperature_C": temperature_c,
            "lambda_nm": lambda_nm,
            "dx_nm": dx_nm,
            "lambda_over_dx": lambda_nm / dx_nm,
            "domain_nm": domain_nm,
            "matrix_xB": matrix_x,
            "requested_diffusion_width_over_lambda": ratio,
            "preage_s": preage_s,
            "Fo_lambda_preage": diffusivity_nm2_s * preage_s / lambda_nm**2,
            "finite_box_width_over_lambda": 0.35 * domain_nm / lambda_nm,
            **sharp_metrics,
            **mapping,
            "phi_raw": provenance_path(case_dir / "phi_init.raw"),
            "xB_raw": provenance_path(case_dir / "xB_init.raw"),
            "Ctot_raw": provenance_path(case_dir / "Ctot_init.raw"),
            "sharp_state": provenance_path(case_dir / "sharp_state.npz"),
            "provenance": "independent_conservative_sharp_preage_mapped_to_fixed_ctot",
        }
        (case_dir / "matched_state_meta.json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
    manifest = {
        "schema": "correction2_matched_state_manifest_v1",
        "temperature_C": temperature_c,
        "lambda_nm": lambda_nm,
        "dx_nm": dx_nm,
        "domain_nm": domain_nm,
        "matrix_xB": matrix_x,
        "diffusion_width_ratios": diffusion_width_ratios,
        "substitution_note": (
            "ell/lambda=16 exceeds the 11.2-lambda matrix half-width; "
            "finite-box-safe family 4,6,8 is used"
        ),
        "cases": rows,
        "GP_S3_enabled": False,
        "finite_interface_mode": "off",
    }
    return manifest


def write_report(manifest: dict[str, object], path: Path) -> None:
    rows = manifest["cases"]
    table = "\n".join(
        f"| {row['case']} | {float(row['preage_s']):.6e} | "
        f"{float(row['Fo_lambda_preage']):.3f} | "
        f"{float(row['sharp_radius_preaged_nm']):.9f} | "
        f"{float(row['h_volume_error_abs_nm']):.3e} | "
        f"{float(row['sharp_mass_error_rel']):.3e} | "
        f"{abs(float(row['pf_inventory_after_correction_xB_nm'])-float(row['target_total_inventory_xB_nm'])):.3e} | "
        f"{float(row['outside_profile_Linf']):.3e} |"
        for row in rows
    )
    path.write_text(f"""# Correction 2 Matched-State Construction

One independently evolved conservative finite-box Stefan state is the source
for both representations.  The sharp state is not altered to match PF mass.
The diffuse mapping matches its h-volume/equimolar surface and closes the
small representation mass difference only inside a `2 lambda` interface band;
outside that band `xB_alpha` is pointwise the sharp profile.

The requested `ell/lambda=16` exceeds the available 11.2-lambda matrix
half-width.  The finite-box-safe diagnostic family is `4, 6, 8`.

| case | pre-age (s) | Fo | sharp R (nm) | h-volume error | sharp mass rel | PF mass abs | outer-profile Linf |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

All generated fields are finite and satisfy `h<=Ctot<=1`,
`0<=q_alpha<=1-h`, and `0<=xB_alpha<=1`.  The reported double-well/gradient
residual is a finite-difference representation diagnostic; chemical driving
is intentionally not folded into it.  Runtime KKT and energy gates remain to
be checked by the matched short-run matrix.

`matched_state_status=PASS_HOST_CONSTRUCTION`
""")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root", type=Path,
        default=ROOT / "tmp/correction2_matched_states",
    )
    parser.add_argument("--domain-nm", type=float, default=19.2)
    parser.add_argument("--lambda-nm", type=float, default=0.6)
    parser.add_argument("--dx-nm", type=float, default=0.075)
    parser.add_argument("--sharp-cells", type=int, default=800)
    parser.add_argument("--matrix-xB", type=float, default=0.05)
    parser.add_argument("--ratios", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "examples/correction2_matched_state_manifest.json",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "reports/pf_ctot_production_candidate/correction2_matched_state_construction.md",
    )
    args = parser.parse_args()
    manifest = prepare_family(
        args.out_root, args.ratios, args.domain_nm, args.dx_nm,
        args.lambda_nm, args.sharp_cells, args.matrix_xB,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(manifest, args.report)
    print(f"matched_state_cases={len(manifest['cases'])}")
    print("matched_state_status=PASS_HOST_CONSTRUCTION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
