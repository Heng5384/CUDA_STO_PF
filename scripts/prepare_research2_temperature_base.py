#!/usr/bin/env python3
"""Generate a temperature-consistent source-free Research2 PF base file."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402


NUMERICAL_OVERLAY: dict[str, object] = {
    "pf_params_schema_version": 2,
    "model_mode": "two_phase",
    "composition_evolution_mode": "ctot_mimetic_be",
    "pf_composition_mode": "legacy",
    "pf_y_update_mode": "lagged_rhs",
    "y_update_mass_projection_enabled": 0,
    "thermo_convex_extrapolation_enabled": 1,
    "ctot_initialization_dry_run": 0,
    "ctot_phase_only_dry_run": 0,
    "ctot_nonlinear_max_iter": 500,
    "ctot_residual_abs_tol": 1.0e-10,
    "ctot_residual_rel_tol": 1.0e-8,
    "ctot_matrix_support_eps": 1.0e-8,
    "ctot_line_search_min": 9.5367431640625e-7,
    "ctot_elastic_validation_enabled": 0,
    "elastic_enabled": 0,
    "ctot_step_max_retries": 8,
    "ctot_outer_max_iter": 16,
    "ctot_automatic_dt_growth": 0,
    "ctot_finite_interface_antitrapping_enabled": 0,
    "ctot_phase_semismooth_pdas_enabled": 1,
    "ctot_phase_linear_max_iter": 200,
    "ctot_phase_linear_rel_tol": 1.0e-10,
    "ctot_phase_fd_rel_step": 1.0e-6,
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
}


def encode(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.17e}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-input", type=Path, required=True)
    parser.add_argument("--temperature-C", type=float, required=True)
    parser.add_argument("--input-out", type=Path, required=True)
    parser.add_argument("--conversion-out", type=Path, required=True)
    parser.add_argument("--params-out", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_input.read_text())
    source = {key: value for key, value in source.items()
              if not key.startswith("_")}
    source["temperature_C"] = args.temperature_C
    source["L_phi_calibration_mode"] = "one_sided_diffusion_controlled"
    for path in (args.input_out, args.conversion_out, args.params_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.input_out.write_text(json.dumps(source, indent=2) + "\n")

    inputs = unit.load_physical_inputs(str(args.input_out))
    payload = unit.generate_payload(inputs)
    args.conversion_out.write_text(json.dumps(payload, indent=2) + "\n")
    unit.write_main_cuda_override_file(
        str(args.params_out), payload["main_cuda_overrides"]
    )
    with args.params_out.open("a") as handle:
        handle.write("\n# Research2 source-free accepted numerical overlay\n")
        for key, value in NUMERICAL_OVERLAY.items():
            handle.write(f"{key}={encode(value)}\n")

    generated = asdict(inputs)
    if generated["temperature_C"] != args.temperature_C:
        raise RuntimeError("temperature override was not preserved")
    if generated["L_phi_calibration_mode"] != "one_sided_diffusion_controlled":
        raise RuntimeError("one-sided L_phi calibration was not preserved")
    print(f"temperature_C={args.temperature_C:.17g}")
    print(f"L_phi_reference_code={payload['pf_params']['L_phi']:.17g}")
    print(f"t_real_unit_s={payload['metadata']['scales']['t0_diff_s']:.17g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
