#!/usr/bin/env python3
"""Prepare matched fine/coarse planar cases for coarse4 calibration.

The initial sharp state is constructed once per driving condition and then
mapped independently to the frozen fine and coarse diffuse representations.
No PF trajectory is used to construct either representation.
"""

from __future__ import annotations

import argparse
import copy
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
from scripts.correction1_planar_sharp_oracle import PlanarSharpOracle  # noqa: E402
from scripts.prepare_correction2_matched_planar_state import (  # noqa: E402
    evolve_uniform_sharp_state,
    map_sharp_to_pf,
)
from scripts.prepare_coarse4_multifidelity import (  # noqa: E402
    CALIBRATION_PROTOCOL,
    MECHANICS_ACCEPTANCE_MODE,
    MECHANICS_DOUBLE_ORACLE_CONTRACT_HASH,
    MECHANICS_ETA_ACCEPT,
    MECHANICS_ETA_FLOOR_VERSION,
    MECHANICS_PRECISION_MODE,
    MECHANICS_RESIDUAL_NORMALIZATION_VERSION,
    MODEL_NAME,
    MODEL_VERSION,
    UNCERTAINTY_VERSION,
    canonical_hash,
    sha256_files,
    write_params,
)


FINE_LAMBDA_NM = 0.30
FINE_DX_NM = 0.025
COARSE_LAMBDA_NM = 4.0
COARSE_DX_NM = 1.0
TEMPERATURE_C = 400.0
DEFAULT_DOMAIN_NM = 64.0
DEFAULT_PREAGE_FO_FINE = 16.0
# The accepted calibration references were generated over Fo_lambda=3.  Keep
# this window explicit so a fresh matrix cannot silently shorten the frozen
# comparison interval.
DEFAULT_OBSERVATION_FO_FINE = 3.0
OBSERVATION_WINDOW_VERSION = "COARSE4_T400_PLANAR_FO3_V1"
CALIBRATION_WEIGHTS = CALIBRATION_PROTOCOL["weights"]

# Frozen before the v2 calibration rerun.  The refinement points resolve the
# only low-a_M interval where the broad scan's flux gate can plausibly hold;
# the tail points verify that inventory convergence cannot recover without
# violating flux.  This is a deterministic scan, not result-adaptive fitting.
A_M_BROAD_SCAN = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
A_M_FEASIBILITY_REFINEMENT = (
    0.0625, 0.125, 0.1875, 0.25, 0.375, 0.625, 0.75,
)
A_M_TAIL_BRACKET = (1.5, 3.0, 6.0, 12.0, 24.0, 32.0, 64.0)
A_M_DETERMINISTIC_SCAN = tuple(sorted(set(
    A_M_BROAD_SCAN + A_M_FEASIBILITY_REFINEMENT + A_M_TAIL_BRACKET
)))
A_M_SCAN_PROTOCOL_VERSION = "COARSE4_AM_DETERMINISTIC_BRACKET_REFINEMENT_V2"


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def physical_payload(lambda_nm: float, dx_nm: float, temperature_c: float,
                     dt_code: float) -> tuple[unit.PhysicalInputs, dict[str, object]]:
    inputs = copy.deepcopy(
        unit.load_physical_inputs(str(ROOT / "physical_inputs.example.json"))
    )
    inputs.gamma = 0.168
    inputs.lambda_sm = lambda_nm * 1.0e-9
    inputs.dx = dx_nm * 1.0e-9
    inputs.pf_dx = dx_nm * 1.0e-9
    inputs.phys_dx_ref = 1.0e-9
    inputs.temperature_C = temperature_c
    inputs.dt = dt_code
    inputs.L_phi_calibration_mode = "one_sided_diffusion_controlled"
    inputs.vf_init = 0.0
    inputs.vf_target = 0.04
    return inputs, unit.generate_payload(inputs)


def common_runtime_values(payload: dict[str, object]) -> dict[str, object]:
    values = dict(payload["main_cuda_overrides"])
    values.update({
        key: value for key, value in payload["pf_params"].items()
        if isinstance(value, (str, int, float, bool))
    })
    values.update({
        "pf_params_schema_version": 2,
        "model_mode": "two_phase",
        "composition_evolution_mode": "ctot_mimetic_be",
        "pf_composition_mode": "legacy",
        "pf_y_update_mode": "lagged_rhs",
        "PHASE_KINETICS_MODE": "FINITE_LPHI_BE",
        "ctot_phase_semismooth_pdas_enabled": 1,
        "ctot_phase_restart_solver_migration_allowed": 0,
        "ctot_nonlinear_max_iter": 300,
        "ctot_phase_linear_max_iter": 500,
        "ctot_phase_linear_rel_tol": 1.0e-10,
        "ctot_residual_abs_tol": 1.0e-10,
        "ctot_residual_rel_tol": 1.0e-8,
        "ctot_outer_max_iter": 30,
        "ctot_step_max_retries": 8,
        "ctot_retry_shrink_factor": 0.5,
        "ctot_dt_min_ratio": 1.0 / 256.0,
        "ctot_automatic_dt_growth": 0,
        "ctot_finite_interface_antitrapping_enabled": 0,
        "finite_interface_resolution_test_override": 0,
        "thermo_convex_extrapolation_enabled": 1,
        "coarse_interface_mobility_mode": "off",
        "coarse_interface_mobility_a_M": 0.0,
        "D_beta": 0.0,
        "D_beta_for_calibration": 0.0,
        "elastic_enabled": 0,
        "ctot_elastic_validation_enabled": 0,
        "y_update_mass_projection_enabled": 0,
        "diagnostic_rsmd_enabled": 0,
        "gp_nuc_enabled": 0,
        "enable_gp_assisted_beta_nucleation": 0,
        "gp_literature_model_enabled": 0,
        "gp_initial_population_enabled": 0,
        "gp_stochastic_enabled": 0,
        "gp_to_beta_enabled": 0,
        "scheduled_nuc_enabled": 0,
        "enable_runtime_nucleus_library": 0,
        "beta_staged_conversion_enabled": 0,
        "gp_growth_enabled": 0,
        "gp_radius_evolution_enabled": 0,
        "gp_inventory_growth_enabled": 0,
        "enable_legacy_gp_storage_coupling": 0,
        "GP_population_mode": "OFF",
        "ctot_diagnostics_enabled": 1,
    })
    return values


def t0_s(payload: dict[str, object]) -> float:
    return float(payload["metadata"]["scales"]["t0_diff_s"])


def write_state(case: Path, phi_line: np.ndarray, x_line: np.ndarray,
                c_line: np.ndarray, dx_nm: float, lambda_nm: float,
                state_id: str, source_hash: str) -> tuple[int, int, int]:
    shape = (phi_line.size, 2, 2)
    for name, line in (("phi", phi_line), ("xB", x_line), ("Ctot", c_line)):
        np.broadcast_to(line[:, None, None], shape).copy().astype(np.float64).tofile(
            case / f"{name}_init.raw"
        )
    metadata = {
        "schema": "coarse4_matched_planar_initial_state_v1",
        "Nx": shape[0], "Ny": shape[1], "Nz": shape[2],
        "dx_nm": dx_nm,
        "interface_width_nm": lambda_nm,
        "dtype": "float64", "order": "C",
        "authoritative_state": "Ctot",
        "source_state": state_id,
        "common_sharp_state_sha256": source_hash,
        "mean_xBtot": float(np.broadcast_to(c_line[:, None, None], shape).mean()),
        "xB_max_safe": float(x_line.max()),
    }
    (case / "init_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return shape


def build_matrix(root: Path, a_m_values: tuple[float, ...], domain_nm: float,
                 observation_fo: float) -> dict[str, object]:
    if domain_nm % COARSE_DX_NM != 0.0:
        raise ValueError("domain must be an integer number of coarse cells")
    fine_nx = int(round(domain_nm / FINE_DX_NM))
    if fine_nx % 2:
        raise ValueError("fine planar grid must be even")
    if any(value < 0.0 or not math.isfinite(value) for value in a_m_values):
        raise ValueError("a_M scan must be finite and nonnegative")

    reference_assets = [
        ROOT / "examples/correction2_selected_matrix_to_beta_mode.json",
        ROOT / "reports/pf_ctot_production_candidate/correction2_physical_lambda_metrics.csv",
        ROOT / "reports/pf_ctot_production_candidate/correction2_diffusion_limit_decision.md",
    ]
    fine_reference_hash = sha256_files(reference_assets)
    calibration_hash = canonical_hash(CALIBRATION_PROTOCOL)
    temperature_k = TEMPERATURE_C + 273.15
    x_eq = unit.xAg2Te_eq_from_T(temperature_k)
    driving = {
        "growth": x_eq + 0.25 * (0.05 - x_eq),
        "dissolution": 0.5 * x_eq,
    }
    diffusivity_nm2_s = unit.D_Ag_in_PbTe_m2_per_s(temperature_k) * 1.0e18
    preage_s = DEFAULT_PREAGE_FO_FINE * FINE_LAMBDA_NM**2 / diffusivity_nm2_s
    elapsed_s = observation_fo * FINE_LAMBDA_NM**2 / diffusivity_nm2_s

    fine_inputs, fine_payload = physical_payload(
        FINE_LAMBDA_NM, FINE_DX_NM, TEMPERATURE_C, 1.0e-3
    )
    coarse_inputs, coarse_payload = physical_payload(
        COARSE_LAMBDA_NM, COARSE_DX_NM, TEMPERATURE_C, 1.0e-4
    )
    payloads = {"fine": (fine_inputs, fine_payload),
                "coarse": (coarse_inputs, coarse_payload)}
    cases: list[dict[str, object]] = []
    sources: dict[str, object] = {}
    root.mkdir(parents=True, exist_ok=True)

    for direction, matrix_x in driving.items():
        # The historical oracle constructor computes an auxiliary growth-only
        # similarity target.  The conservative finite-volume equations used
        # below also support dissolution, so initialize that unused target on
        # the growth side and then set the requested matrix composition.
        oracle = PlanarSharpOracle(
            temperature_c=TEMPERATURE_C, domain_nm=domain_nm,
            matrix_x=max(matrix_x, x_eq + 1.0e-12),
            start_time_s=preage_s, cells=1200,
            pf_dx_nm=FINE_DX_NM,
        )
        oracle.matrix_x = matrix_x
        sharp_state, sharp_metrics = evolve_uniform_sharp_state(oracle, preage_s)
        source_payload = {
            "temperature_C": TEMPERATURE_C,
            "direction": direction,
            "matrix_xB": matrix_x,
            "xB_eq": x_eq,
            "domain_nm": domain_nm,
            "preage_s": preage_s,
            "sharp_state_sha256": hashlib.sha256(sharp_state.tobytes()).hexdigest(),
        }
        source_hash = canonical_sha256(source_payload)
        source_id = f"T400_planar_small_driving_{direction}"
        sources[direction] = {**source_payload, **sharp_metrics,
                              "source_id": source_id,
                              "common_state_hash": source_hash}
        mapped: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = {}
        for resolution, dx_nm, lambda_nm in (
            ("fine", FINE_DX_NM, FINE_LAMBDA_NM),
            ("coarse", COARSE_DX_NM, COARSE_LAMBDA_NM),
        ):
            mapped[resolution] = map_sharp_to_pf(
                oracle, sharp_state, domain_nm, dx_nm, lambda_nm
            )

        variants: list[tuple[str, str, float]] = [
            ("fine_reference", "FINITE_LPHI_BE", 0.0),
        ]
        for phase_id, phase_mode in (
            ("finite_lphi", "FINITE_LPHI_BE"),
            ("quasi_equilibrium", "QUASI_EQUILIBRIUM_FAST_INTERFACE_V1"),
        ):
            variants.extend(
                (phase_id, phase_mode, value) for value in a_m_values
            )

        for variant, phase_mode, a_m in variants:
            resolution = "fine" if variant == "fine_reference" else "coarse"
            inputs, payload = payloads[resolution]
            dx_nm = FINE_DX_NM if resolution == "fine" else COARSE_DX_NM
            lambda_nm = FINE_LAMBDA_NM if resolution == "fine" else COARSE_LAMBDA_NM
            phi, x_b, ctot, mapping = mapped[resolution]
            a_tag = str(a_m).replace(".", "p")
            case_id = f"cal_{direction}_{variant}_aM{a_tag}"
            case = root / "cases" / case_id
            case.mkdir(parents=True, exist_ok=True)
            grid = write_state(
                case, phi, x_b, ctot, dx_nm, lambda_nm, source_id, source_hash
            )
            np.savez(
                case / "sharp_state.npz", state=sharp_state,
                temperature_c=TEMPERATURE_C, domain_nm=domain_nm,
                matrix_x=matrix_x, preage_s=preage_s, sharp_cells=oracle.cells,
            )

            dt_target = (
                1.0e-3 if resolution == "fine" else
                1.5625e-6 if phase_mode == "QUASI_EQUILIBRIUM_FAST_INTERFACE_V1" else
                1.0e-4
            )
            code_time = elapsed_s / t0_s(payload)
            nsteps = max(1, int(math.ceil(code_time / dt_target)))
            dt_code = code_time / nsteps
            values = common_runtime_values(payload)
            lphi_diff = float(payload["pf_params"]["L_phi"])
            values.update({
                "dt": dt_code,
                "PHASE_KINETICS_MODE": phase_mode,
                "L_phi": 0.9 * lphi_diff,
                "L_phi_code_value": 0.9 * lphi_diff,
                "L_phi_physical_value": 0.9 * float(payload["pf_params"]["L_phi_phys"]),
                "L_phi_calibration_mode": "one_sided_diffusion_controlled",
                "init_case_tag": case_id,
            })
            if resolution == "coarse":
                values.update({
                    "PF_RESEARCH_MODEL": MODEL_NAME,
                    "coarse_model_name": MODEL_NAME,
                    "coarse_model_version": MODEL_VERSION,
                    "fine_reference_hash": fine_reference_hash,
                    "coarse_calibration_hash": calibration_hash,
                    "coarse_uncertainty_version": UNCERTAINTY_VERSION,
                    "mechanics_precision_mode": MECHANICS_PRECISION_MODE,
                    "mechanics_acceptance_mode": MECHANICS_ACCEPTANCE_MODE,
                    "eta_floor_version": MECHANICS_ETA_FLOOR_VERSION,
                    "eta_accept": MECHANICS_ETA_ACCEPT,
                    "double_oracle_contract_hash": MECHANICS_DOUBLE_ORACLE_CONTRACT_HASH,
                    "residual_normalization_version": MECHANICS_RESIDUAL_NORMALIZATION_VERSION,
                    "coarse_interface_mobility_mode": (
                        "off" if a_m == 0.0 else "INTERFACE_BAND_BOOST_V1"
                    ),
                    "coarse_interface_mobility_a_M": a_m,
                })
            else:
                values.update({
                    "PF_RESEARCH_MODEL": "off",
                    "coarse_interface_mobility_mode": "off",
                    "coarse_interface_mobility_a_M": 0.0,
                })
            write_params(case / "runtime.params", values)
            manifest = {
                "schema": "coarse4_calibration_runtime_case_v1",
                "case_id": case_id,
                "state_case": source_id,
                "study": "coarse4_one_parameter_calibration",
                "calibration_role": "reference" if resolution == "fine" else "candidate",
                "direction": direction,
                "temperature_C": TEMPERATURE_C,
                "drive_amplitude": 0.25 if direction == "growth" else -0.5,
                "grid": list(grid),
                "dx_nm": dx_nm,
                "dx_code": dx_nm,
                "lambda_nm": lambda_nm,
                "lambda_over_dx": lambda_nm / dx_nm,
                "domain_nm": domain_nm,
                "preage_s": preage_s,
                "preage_Fo": DEFAULT_PREAGE_FO_FINE,
                "matrix_xB": matrix_x,
                "L_phi_ratio": 0.9,
                "phase_candidate": variant,
                "PHASE_KINETICS_MODE": phase_mode,
                "a_M": a_m,
                "target_incremental_Fo": observation_fo,
                "elapsed_s": elapsed_s,
                "final_code_time": code_time,
                "dt_code": dt_code,
                "nsteps": nsteps,
                "finite_interface_mode": "off",
                "elasticity": "off",
                "GP_S3": "off",
                "D_beta": 0.0,
                "common_sharp_state_sha256": source_hash,
                "fine_reference_hash": fine_reference_hash,
                "coarse_calibration_hash": calibration_hash,
                "mapping": mapping,
                "provenance": "one_common_sharp_state_mapped_to_fine_and_coarse_fixed_ctot",
            }
            (case / "runtime_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            cases.append(manifest)

    matrix = {
        "schema": "coarse4_calibration_matrix_v1",
        "status": "PREPARED_NOT_RUN",
        "temperature_C": TEMPERATURE_C,
        "fine_reference": {"lambda_nm": FINE_LAMBDA_NM, "dx_nm": FINE_DX_NM,
                           "lambda_over_dx": 12.0,
                           "fine_reference_hash": fine_reference_hash},
        "coarse_model": {"name": MODEL_NAME, "lambda_nm": COARSE_LAMBDA_NM,
                         "dx_nm": COARSE_DX_NM, "lambda_over_dx": 4.0},
        "domain_nm": domain_nm,
        "observation_Fo_fine": observation_fo,
        "observation_window_version": OBSERVATION_WINDOW_VERSION,
        "elapsed_s": elapsed_s,
        "weights_frozen_before_scan": CALIBRATION_WEIGHTS,
        "a_M_scan_frozen_before_results": list(a_m_values),
        "a_M_scan_protocol": {
            "version": A_M_SCAN_PROTOCOL_VERSION,
            "frozen_before_new_results": True,
            "broad_scan": list(A_M_BROAD_SCAN),
            "feasibility_refinement": list(A_M_FEASIBILITY_REFINEMENT),
            "tail_bracket": list(A_M_TAIL_BRACKET),
            "ordered_union": list(A_M_DETERMINISTIC_SCAN),
            "adaptive_refit": False,
            "free_parameter_count": 1,
        },
        "fit_cases": CALIBRATION_PROTOCOL["fit_cases"],
        "heldout_refit_allowed": False,
        "sources": sources,
        "cases": cases,
        "GP_source_reintegrated": False,
        "S3_reintegrated": False,
    }
    (root / "matrix_manifest.json").write_text(
        json.dumps(matrix, indent=2) + "\n", encoding="utf-8"
    )
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=ROOT / "tmp/coarse4_calibration_v1"
    )
    parser.add_argument("--domain-nm", type=float, default=DEFAULT_DOMAIN_NM)
    parser.add_argument("--observation-Fo", type=float,
                        default=DEFAULT_OBSERVATION_FO_FINE)
    parser.add_argument("--a-M", type=float, nargs="+",
                        default=list(A_M_DETERMINISTIC_SCAN))
    args = parser.parse_args()
    matrix = build_matrix(
        args.root.resolve(), tuple(args.a_M), args.domain_nm, args.observation_Fo
    )
    print(f"coarse4_calibration_cases={len(matrix['cases'])}")
    print(f"fine_reference_hash={matrix['fine_reference']['fine_reference_hash']}")
    print("GP_source_reintegrated=false")
    print("cluster_used=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
