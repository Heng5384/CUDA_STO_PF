#!/usr/bin/env python3
"""Prepare equal-time JC4 interface-width and grid-sensitivity cases.

Only the frozen ``dx=1 nm, lambda=4 nm`` point selects the JC4 research
model.  Refinement/width variants deliberately run as provenance-labelled
comparators, so this study cannot weaken the production selector contract.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402
from scripts.correction1_ji_chen_mapping import corrected_limit  # noqa: E402
from scripts.jc4_ji_chen_1d_oracle import h_switch  # noqa: E402
from scripts.prepare_jc4_long_time_planar import (  # noqa: E402
    CASE_SPECS,
    provenance_hashes,
)
from scripts.prepare_jc4_research_model import (  # noqa: E402
    LPHI_RATIO,
    MODEL_NAME,
    runtime_params,
    write_params,
)


BASE_TIME_SCALE_S = 41.12958542455477
PHYSICAL_END_TIME_S = {
    "growth": 100.0 * BASE_TIME_SCALE_S,
    "dissolution": 5000.0 * BASE_TIME_SCALE_S,
}
PHYSICAL_STEP_S = {
    "growth": 0.1 * BASE_TIME_SCALE_S,
    "dissolution": 5.0 * BASE_TIME_SCALE_S,
}
VARIANTS = (
    {
        "variant": "production_dx1_lambda4",
        "dx_nm": 1.0,
        "lambda_nm": 4.0,
        "matrix": "production_baseline",
        "selected_model": True,
    },
    {
        "variant": "refined_dx0p5_lambda4",
        "dx_nm": 0.5,
        "lambda_nm": 4.0,
        "matrix": "fixed_lambda_grid_refinement",
        "selected_model": False,
    },
    {
        "variant": "width_lambda3_dx1",
        "dx_nm": 1.0,
        "lambda_nm": 3.0,
        "matrix": "fixed_dx_width_sensitivity",
        "selected_model": False,
    },
    {
        "variant": "width_lambda5_dx1",
        "dx_nm": 1.0,
        "lambda_nm": 5.0,
        "matrix": "fixed_dx_width_sensitivity",
        "selected_model": False,
    },
)


def physical_payload(lambda_nm: float, dx_nm: float) -> tuple[unit.PhysicalInputs, dict[str, object]]:
    inputs = copy.deepcopy(
        unit.load_physical_inputs(str(ROOT / "physical_inputs.example.json"))
    )
    inputs.gamma = 0.168
    inputs.lambda_sm = lambda_nm * 1.0e-9
    inputs.dx = dx_nm * 1.0e-9
    inputs.pf_dx = dx_nm * 1.0e-9
    inputs.phys_dx_ref = 1.0e-9
    inputs.temperature_C = 400.0
    inputs.dt = 1.0e-6
    inputs.D_ratio = 0.0
    inputs.L_phi_calibration_mode = "one_sided_diffusion_controlled"
    inputs.vf_init = 0.0
    inputs.vf_target = 0.04
    return inputs, unit.generate_payload(inputs)


def planar_profile(cells: int, dx_nm: float, lambda_nm: float,
                   half_width_nm: float) -> np.ndarray:
    length_nm = cells * dx_nm
    coordinate = (np.arange(cells, dtype=np.float64) + 0.5) * dx_nm
    center = 0.5 * length_nm
    return 0.5 * (
        np.tanh((coordinate - (center - half_width_nm)) / (0.5 * lambda_nm))
        - np.tanh((coordinate - (center + half_width_nm)) / (0.5 * lambda_nm))
    )


def write_state(case: Path, phi_line: np.ndarray, matrix_xB: float,
                dx_nm: float, lambda_nm: float,
                target_line_inventory: float) -> dict[str, object]:
    h = h_switch(phi_line)
    length_nm = phi_line.size * dx_nm
    h_integral = float(np.sum(h) * dx_nm)
    matrix_x_adjusted = (
        target_line_inventory - h_integral
    ) / (length_nm - h_integral)
    if not 0.0 < matrix_x_adjusted < 1.0:
        raise ValueError("inventory matching produced an invalid matrix composition")
    x_line = np.full(phi_line.size, matrix_x_adjusted, dtype=np.float64)
    C_line = h + (1.0 - h) * x_line
    shape = (phi_line.size, 2, 2)
    for name, line in (("phi", phi_line), ("xB", x_line), ("Ctot", C_line)):
        np.broadcast_to(line[:, None, None], shape).copy().astype(np.float64).tofile(
            case / f"{name}_init.raw"
        )
    actual_inventory = float(np.sum(C_line) * dx_nm)
    metadata = {
        "schema": "jc4_width_grid_initial_state_v1",
        "Nx": shape[0],
        "Ny": shape[1],
        "Nz": shape[2],
        "dx_nm": dx_nm,
        "interface_width_nm": lambda_nm,
        "dtype": "float64",
        "order": "C",
        "authoritative_state": "Ctot",
        "geometry": "periodic_planar_beta_slab_two_interfaces",
        "nominal_matrix_xB": matrix_xB,
        "inventory_matched_matrix_xB": matrix_x_adjusted,
        "matrix_xB_adjustment": matrix_x_adjusted - matrix_xB,
        "h_line_integral_nm": h_integral,
        "target_C_line_integral": target_line_inventory,
        "actual_C_line_integral": actual_inventory,
        "inventory_match_abs": abs(actual_inventory - target_line_inventory),
    }
    (case / "init_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def variant_params(dx_nm: float, lambda_nm: float, dt_code: float,
                   selected_model: bool, case_id: str,
                   contract_hash: str, reference_hash: str) -> tuple[dict[str, object], dict[str, float]]:
    inputs, payload = physical_payload(lambda_nm, dx_nm)
    strict = corrected_limit(inputs, 400.0)
    values = runtime_params(400.0, contract_hash, reference_hash)
    values.update(payload["main_cuda_overrides"])
    selected_code = LPHI_RATIO * strict["L_phi_diff_code"]
    selected_physical = LPHI_RATIO * strict["L_phi_diff_physical_m3_J_s"]
    values.update({
        "PF_RESEARCH_MODEL": MODEL_NAME if selected_model else "off",
        "coarse_model_name": (
            MODEL_NAME if selected_model else "jc4_sensitivity_comparator_v1"
        ),
        "coarse_uncertainty_version": (
            "JI_CHEN_LONG_TIME_UNQUALIFIED_V1" if selected_model
            else "JC4_SENSITIVITY_COMPARATOR_NOT_SELECTED_MODEL"
        ),
        "dt": dt_code,
        "L_phi": selected_code,
        "L_phi_code_value": selected_code,
        "L_phi_physical_value": selected_physical,
        "L_phi_calibration_mode": "one_sided_diffusion_controlled",
        "D_beta_for_calibration": 0.0,
        "D_compound": 0.0,
        "ctot_finite_interface_antitrapping_enabled": 0,
        "coarse_interface_mobility_mode": "off",
        "coarse_interface_mobility_a_M": 0.0,
        "GP_population_mode": "OFF",
        "elastic_enabled": 0,
        "init_case_tag": case_id,
    })
    derived = {
        "W_physical_J_m3": strict["W_physical_J_m3"],
        "kappa_physical_J_m": 1.5 * inputs.gamma * inputs.lambda_sm,
        "L_phi_diff_code": strict["L_phi_diff_code"],
        "L_phi_selected_code": selected_code,
        "L_phi_diff_physical_m3_J_s": strict["L_phi_diff_physical_m3_J_s"],
        "L_phi_selected_physical_m3_J_s": selected_physical,
        "time_scale_s": strict["time_scale_s"],
        "W_code": float(payload["pf_params"]["W"]),
        "kappa_code": float(payload["pf_params"]["kappa_phi"]),
        "D_alpha_code": float(payload["pf_params"]["D_alpha"]),
    }
    return values, derived


def build(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    contract_hash, reference_hash = provenance_hashes()
    cases: list[dict[str, object]] = []
    for direction, spec in CASE_SPECS.items():
        physical_length_nm = float(spec["cells"])
        half_width_nm = float(spec["half_width_nm"])
        nominal_matrix_xB = float(spec["matrix_xB"])
        target_inventory = (
            2.0 * half_width_nm
            + (physical_length_nm - 2.0 * half_width_nm) * nominal_matrix_xB
        )
        for variant in VARIANTS:
            dx_nm = float(variant["dx_nm"])
            lambda_nm = float(variant["lambda_nm"])
            cells_float = physical_length_nm / dx_nm
            cells = int(round(cells_float))
            if not math.isclose(cells, cells_float, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("physical domain is not an integer grid")
            final_time_s = PHYSICAL_END_TIME_S[direction]
            inputs, payload = physical_payload(lambda_nm, dx_nm)
            time_scale_s = float(payload["metadata"]["scales"]["t0_diff_s"])
            final_time_code = final_time_s / time_scale_s
            desired_dt_code = PHYSICAL_STEP_S[direction] / time_scale_s
            step_ratio = final_time_code / desired_dt_code
            nearest_steps = int(round(step_ratio))
            nsteps = max(1, (
                nearest_steps if math.isclose(
                    step_ratio, nearest_steps, rel_tol=1.0e-12, abs_tol=1.0e-12
                ) else int(math.ceil(step_ratio))
            ))
            dt_code = final_time_code / nsteps
            case_id = f"T400_{direction}_{variant['variant']}"
            case = root / "cases" / case_id
            case.mkdir(parents=True, exist_ok=True)
            phi = planar_profile(cells, dx_nm, lambda_nm, half_width_nm)
            state_meta = write_state(
                case, phi, nominal_matrix_xB, dx_nm, lambda_nm, target_inventory
            )
            params, derived = variant_params(
                dx_nm, lambda_nm, dt_code, bool(variant["selected_model"]),
                case_id, contract_hash, reference_hash,
            )
            write_params(case / "runtime.params", params)
            manifest = {
                "schema": "jc4_width_grid_runtime_case_v1",
                "case_id": case_id,
                "direction": direction,
                "study_mode": "width_grid_sensitivity",
                "sensitivity_matrix": variant["matrix"],
                "grid": [cells, 2, 2],
                "grid_role": "one_dimensional_thin_slab_not_3d_production",
                "dx_nm": dx_nm,
                "lambda_nm": lambda_nm,
                "lambda_over_dx": lambda_nm / dx_nm,
                "temperature_C": 400.0,
                "matrix_xB": float(state_meta["inventory_matched_matrix_xB"]),
                "nominal_matrix_xB": nominal_matrix_xB,
                "initial_beta_half_width_nm": half_width_nm,
                "dt_code": dt_code,
                "nsteps": nsteps,
                "final_time_code": final_time_code,
                "elapsed_s": final_time_s,
                "out_every": max(1, nsteps // 20),
                "selected_research_model": bool(variant["selected_model"]),
                "runtime_selector": MODEL_NAME if variant["selected_model"] else "off",
                "comparator_provenance": (
                    "SELECTED_JC4_PRODUCTION_POINT" if variant["selected_model"]
                    else "JC4_SENSITIVITY_COMPARATOR_NOT_SELECTED_MODEL"
                ),
                "same_physical_time": True,
                "same_physical_box": True,
                "same_total_inventory": True,
                "elasticity": "off",
                "GP": "off",
                "finite_interface_mode": "off",
                "a_M": 0.0,
                "model_contract_hash": contract_hash,
                "reference_evidence_hash": reference_hash,
                "physical_mapping": derived,
                "initial_state": state_meta,
            }
            (case / "runtime_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            cases.append(manifest)
    top = {
        "schema": "jc4_width_grid_sensitivity_matrix_v1",
        "cases": cases,
        "selected_point": {"dx_nm": 1.0, "lambda_nm": 4.0},
        "formal_3d_production": False,
        "cluster_used": False,
    }
    (root / "manifest.json").write_text(
        json.dumps(top, indent=2) + "\n", encoding="utf-8"
    )
    return top


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=ROOT / "tmp/jc4_width_grid_sensitivity",
    )
    args = parser.parse_args()
    manifest = build(args.root.resolve())
    print(f"jc4_width_grid_cases={len(manifest['cases'])}")
    for case in manifest["cases"]:
        print(
            f"{case['case_id']} grid={case['grid']} "
            f"lambda_over_dx={case['lambda_over_dx']:.9g} "
            f"steps={case['nsteps']} selector={case['runtime_selector']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
