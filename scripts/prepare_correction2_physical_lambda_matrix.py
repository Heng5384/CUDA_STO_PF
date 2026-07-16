#!/usr/bin/env python3
"""Prepare a fixed-outer-state physical-lambda refinement matrix."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402
from scripts.correction1_ji_chen_mapping import corrected_limit  # noqa: E402
from scripts.correction1_planar_sharp_oracle import PlanarSharpOracle  # noqa: E402
from scripts.prepare_correction2_matched_planar_state import (  # noqa: E402
    map_sharp_to_pf,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_physical_inputs(path: Path) -> unit.PhysicalInputs:
    payload = json.loads(path.read_text())
    allowed = {item.name for item in fields(unit.PhysicalInputs)}
    return unit.PhysicalInputs(**{
        key: value for key, value in payload.items() if key in allowed
    })


def append_overlay(source: Path, destination: Path,
                   values: dict[str, object]) -> None:
    text = source.read_text()
    if not text.endswith("\n"):
        text += "\n"
    text += "\n# Correction2 physical-lambda refinement overlay\n"
    for key, value in values.items():
        if isinstance(value, str):
            text += f"{key}={value}\n"
        elif isinstance(value, int):
            text += f"{key}={value}\n"
        else:
            text += f"{key}={float(value):.17e}\n"
    destination.write_text(text)


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def prepare_matrix(
    root: Path,
    source_state_dir: Path,
    base_params: Path,
    physical_inputs_path: Path,
    lambdas_nm: list[float],
    lambda_over_dx: float,
    elapsed_s: float,
    dt_code_requested: float,
    lphi_ratio: float,
) -> dict[str, object]:
    if not (0.0 < lphi_ratio < 1.0):
        raise ValueError("Lphi ratio must be strictly below one")
    if lambda_over_dx not in (8.0, 12.0):
        raise ValueError("lambda/dx must be the declared 8 or 12")
    source_meta = json.loads(
        (source_state_dir / "matched_state_meta.json").read_text()
    )
    sharp_payload = np.load(source_state_dir / "sharp_state.npz")
    sharp_state = np.asarray(sharp_payload["state"], dtype=np.float64)
    temperature_c = float(sharp_payload["temperature_c"])
    domain_nm = float(sharp_payload["domain_nm"])
    matrix_x = float(sharp_payload["matrix_x"])
    preage_s = float(sharp_payload["preage_s"])
    sharp_cells = int(sharp_payload["sharp_cells"])
    physical_inputs = parse_physical_inputs(physical_inputs_path)
    diffusivity_m2_s = unit.D_Ag_in_PbTe_m2_per_s(temperature_c + 273.15)
    diffusivity_nm2_s = diffusivity_m2_s * 1.0e18
    source_sharp_hash = sha256(source_state_dir / "sharp_state.npz")

    cases: list[dict[str, object]] = []
    case_root = root / "cases"
    case_root.mkdir(parents=True, exist_ok=True)
    for lambda_nm in lambdas_nm:
        dx_nm = lambda_nm / lambda_over_dx
        nx_float = domain_nm / dx_nm
        nx = int(round(nx_float))
        if nx % 2 or abs(nx_float - nx) > 1.0e-10:
            raise ValueError(f"lambda={lambda_nm} does not tile the physical box")
        oracle = PlanarSharpOracle(
            temperature_c=temperature_c,
            domain_nm=domain_nm,
            matrix_x=matrix_x,
            start_time_s=preage_s,
            cells=sharp_cells,
            pf_dx_nm=dx_nm,
        )
        phi, x_b, ctot, mapping = map_sharp_to_pf(
            oracle, sharp_state, domain_nm, dx_nm, lambda_nm
        )
        case_id = f"lambda_{tag(lambda_nm)}nm_dx{tag(dx_nm)}"
        case_dir = case_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        shape = (nx, 2, 2)
        for name, line in (("phi", phi), ("xB", x_b), ("Ctot", ctot)):
            np.broadcast_to(line[:, None, None], shape).copy().astype(
                np.float64
            ).tofile(case_dir / f"{name}_init.raw")
        shutil.copy2(source_state_dir / "sharp_state.npz", case_dir / "sharp_state.npz")

        current_inputs = replace(
            physical_inputs,
            temperature_C=temperature_c,
            lambda_sm=lambda_nm * 1.0e-9,
            dx=dx_nm * 1.0e-9,
            pf_dx=dx_nm * 1.0e-9,
            L_phi_calibration_mode="one_sided_diffusion_controlled",
        )
        converted = unit.PFParamConverter().convert(current_inputs)
        strict = corrected_limit(current_inputs, temperature_c)
        gamma = current_inputs.gamma
        lambda_m = current_inputs.lambda_sm
        w_physical = 12.0 * gamma / lambda_m
        kappa_physical = 1.5 * gamma * lambda_m
        lphi_code = lphi_ratio * strict["L_phi_diff_code"]
        lphi_phys = lphi_ratio * strict["L_phi_diff_physical_m3_J_s"]
        t0_s = strict["time_scale_s"]
        final_code_time = elapsed_s / t0_s
        nsteps = max(1, int(math.ceil(final_code_time / dt_code_requested)))
        dt_code = final_code_time / nsteps
        dx_code = dx_nm / (current_inputs.phys_dx_ref * 1.0e9)

        init_meta = {
            "Nx": nx, "Ny": 2, "Nz": 2,
            "dx_nm": dx_nm,
            "interface_width_nm": lambda_nm,
            "dtype": "float64", "order": "C",
            "dt_recommended": dt_code,
            "mean_xBtot": float(ctot.mean()),
            "xB_max_safe": float(mapping["xB_max"]),
            "reference_type": "correction2_same_physical_outer_state_lambda_refinement",
            "source_state": source_meta["case"],
        }
        (case_dir / "init_meta.json").write_text(
            json.dumps(init_meta, indent=2) + "\n"
        )
        overlay: dict[str, object] = {
            "dx": dx_code, "dy": dx_code, "dz": dx_code,
            "dt": dt_code,
            "W": 1.0,
            "kappa_phi": converted.kappa_phi,
            "D_alpha": converted.D_alpha,
            "D_compound": converted.D_compound,
            # GP is disabled, but its retained physical/code parameter pairs
            # must remain dimensionally consistent with the changed beta
            # energy/time references so the runtime can fail closed correctly.
            "gp_W_eta_code": converted.gp_W_eta_code,
            "gp_W_eta_phys": converted.gp_W_eta_phys,
            "gp_kappa_eta_code": converted.gp_kappa_eta_code,
            "gp_kappa_eta_phys": converted.gp_kappa_eta_phys,
            "gp_L_eta_code": converted.gp_L_eta_code,
            "gp_L_eta_phys": converted.gp_L_eta_phys,
            "zeta_eta": converted.gp_zeta_eta,
            "zeta0_eta": converted.gp_zeta0_eta,
            "D_beta_for_calibration": 0.0,
            "zeta_phi": strict["zeta_J_mol"],
            "zeta0_phi": 1.0,
            "L_phi": lphi_code,
            "L_phi_code_value": lphi_code,
            "L_phi_physical_value": lphi_phys,
            "L_phi_calibration_mode": "one_sided_diffusion_controlled",
            "gamma_Jm2": gamma,
            "lambda_sm_m": lambda_m,
            "mu_reference_scale": strict["mu_reference_J_mol"],
            "t_real_unit": t0_s,
            "t_real_unit_s": t0_s,
            "ic_phi_iface_w": lambda_nm / (2.0 * dx_nm),
            "composition_evolution_mode": "ctot_mimetic_be",
            "ctot_nonlinear_max_iter": 500,
            "ctot_step_max_retries": 8,
            "ctot_automatic_dt_growth": 0,
            "ctot_outer_max_iter": 20,
            "ctot_finite_interface_antitrapping_enabled": 0,
            "elastic_enabled": 0,
            "ctot_elastic_validation_enabled": 0,
        }
        append_overlay(base_params, case_dir / "runtime.params", overlay)
        state_meta = {
            "case": case_id,
            "temperature_C": temperature_c,
            "lambda_nm": lambda_nm,
            "dx_nm": dx_nm,
            "lambda_over_dx": lambda_over_dx,
            "domain_nm": domain_nm,
            "matrix_xB": matrix_x,
            "requested_diffusion_width_over_lambda": math.sqrt(
                diffusivity_nm2_s * preage_s
            ) / lambda_nm,
            "preage_s": preage_s,
            "Fo_lambda_preage": diffusivity_nm2_s * preage_s / lambda_nm**2,
            **mapping,
            "source_sharp_state_sha256": source_sharp_hash,
            "provenance": "same_frozen_sharp_state_remapped_at_physical_lambda",
        }
        (case_dir / "matched_state_meta.json").write_text(
            json.dumps(state_meta, indent=2) + "\n"
        )
        manifest: dict[str, object] = {
            "schema": "correction2_physical_lambda_runtime_case_v1",
            "state_case": case_id,
            "study": "physical_lambda_refinement",
            "drive_amplitude": 0.25,
            "grid": [nx, 2, 2],
            "dx_nm": dx_nm,
            "dx_code": dx_code,
            "lambda_nm": lambda_nm,
            "lambda_over_dx": lambda_over_dx,
            "domain_nm": domain_nm,
            "matrix_xB": matrix_x,
            "preage_s": preage_s,
            "preage_Fo": diffusivity_nm2_s * preage_s / lambda_nm**2,
            "target_incremental_Fo": diffusivity_nm2_s * elapsed_s / lambda_nm**2,
            "elapsed_s": elapsed_s,
            "final_code_time": final_code_time,
            "dt_code": dt_code,
            "nsteps": nsteps,
            "L_phi_ratio": lphi_ratio,
            "L_phi_diff_code": strict["L_phi_diff_code"],
            "L_phi_code": lphi_code,
            "L_phi_diff_physical_m3_J_s": strict["L_phi_diff_physical_m3_J_s"],
            "L_phi_physical_m3_J_s": lphi_phys,
            "gamma_J_m2": gamma,
            "w_physical_J_m3": w_physical,
            "kappa_physical_J_m": kappa_physical,
            "kappa_code": converted.kappa_phi,
            "D_alpha_m2_s": diffusivity_m2_s,
            "D_alpha_code": converted.D_alpha,
            "D_compound_code_retained_but_inactive_in_one_sided_ctot_flux": converted.D_compound,
            "zeta_J_mol": strict["zeta_J_mol"],
            "zeta0": 1.0,
            "time_scale_s": t0_s,
            "mu_reference_J_mol": strict["mu_reference_J_mol"],
            "same_outer_state_source": str(source_state_dir.relative_to(ROOT)),
            "same_outer_state_sha256": source_sharp_hash,
            "finite_interface_mode": "off",
            "elasticity": "off",
            "GP_S3": "off",
            "D_beta_for_calibration": 0.0,
            "actual_ctot_flux": "(1-h)*M_alpha*grad(mu)",
            "provenance": "physical_lambda_only_not_numerical_compensation",
        }
        (case_dir / "runtime_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        cases.append({"case": case_id, **manifest})

    payload = {
        "schema": "correction2_physical_lambda_matrix_v1",
        "temperature_C": temperature_c,
        "lambdas_nm": lambdas_nm,
        "lambda_over_dx": lambda_over_dx,
        "physical_box_nm": domain_nm,
        "same_preage_s": preage_s,
        "same_elapsed_s": elapsed_s,
        "same_matrix_xB": matrix_x,
        "same_sharp_state_sha256": source_sharp_hash,
        "Lphi_ratio": lphi_ratio,
        "cases": cases,
        "finite_interface_correction": "off",
        "spectral_transport": "off",
        "GP_S3": "off",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "matrix_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=ROOT / "tmp/correction2_physical_lambda"
    )
    parser.add_argument(
        "--source-state-dir", type=Path,
        default=ROOT / "tmp/correction2_long_time/states/T400_ell4_lambda_x0p01693096",
    )
    parser.add_argument(
        "--base-params", type=Path,
        default=ROOT / "tmp/correction1_Lphi_below_limit/shared/corrected_T400_base.params",
    )
    parser.add_argument(
        "--physical-inputs", type=Path, default=ROOT / "physical_inputs.example.json"
    )
    parser.add_argument("--lambdas-nm", type=float, nargs="+", default=[0.6, 0.45, 0.3])
    parser.add_argument("--lambda-over-dx", type=float, default=12.0)
    parser.add_argument("--elapsed-s", type=float, default=0.3701662688209928)
    parser.add_argument("--dt-code", type=float, default=1.0e-3)
    parser.add_argument("--lphi-ratio", type=float, default=0.9)
    args = parser.parse_args()
    payload = prepare_matrix(
        args.root.resolve(), args.source_state_dir.resolve(),
        args.base_params.resolve(), args.physical_inputs.resolve(),
        args.lambdas_nm, args.lambda_over_dx, args.elapsed_s,
        args.dt_code, args.lphi_ratio,
    )
    print(f"physical_lambda_cases={len(payload['cases'])}")
    for case in payload["cases"]:
        print(
            f"{case['case']}: grid={case['grid']} steps={case['nsteps']} "
            f"Lphi={case['L_phi_code']:.12g} Fo={case['target_incremental_Fo']:.8g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
