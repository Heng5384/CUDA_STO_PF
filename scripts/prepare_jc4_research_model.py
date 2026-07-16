#!/usr/bin/env python3
"""Prepare the default-off Ji--Chen coarse4 fixed-Ctot research contract."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import Unit_Psedobinary as unit  # noqa: E402
from scripts.correction1_ji_chen_mapping import corrected_limit  # noqa: E402


MODEL_NAME = "fixed_ctot_ji_chen_coarse4_gp_v1"
MODEL_VERSION = "1"
UNCERTAINTY_VERSION = "JI_CHEN_LONG_TIME_UNQUALIFIED_V1"
DX_NM = 1.0
LAMBDA_NM = 4.0
LPHI_RATIO = 0.90
MECHANICS_PRECISION_MODE = "FP32_SPECTRAL"
MECHANICS_ACCEPTANCE_MODE = "FP32_NORMALIZED_BACKWARD_ERROR_V1"
MECHANICS_ETA_FLOOR_VERSION = "COARSE4_FP32_ETA_FLOOR_16_32_V1"
MECHANICS_ETA_ACCEPT = 2.569922949844634e-7
MECHANICS_DOUBLE_ORACLE_CONTRACT_HASH = (
    "cc4cad8955684d45d34ca9db7d1b300dd82acc1fc7473ff52b23cd0f5cd3ae3e"
)
MECHANICS_RESIDUAL_NORMALIZATION_VERSION = (
    "DEALIASED_REAL_DIVSIGMA_OVER_KMAX_STRESS_V1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def base_inputs(temperature_c: float) -> unit.PhysicalInputs:
    inputs = copy.deepcopy(
        unit.load_physical_inputs(str(ROOT / "physical_inputs.example.json"))
    )
    inputs.gamma = 0.168
    inputs.lambda_sm = LAMBDA_NM * 1.0e-9
    inputs.dx = DX_NM * 1.0e-9
    inputs.pf_dx = DX_NM * 1.0e-9
    inputs.phys_dx_ref = DX_NM * 1.0e-9
    inputs.temperature_C = temperature_c
    inputs.dt = 1.0e-6
    inputs.D_ratio = 0.0
    inputs.L_phi_calibration_mode = "one_sided_diffusion_controlled"
    inputs.vf_init = 0.0
    inputs.vf_target = 0.04
    return inputs


def model_contract() -> dict[str, object]:
    return {
        "schema": "jc4_model_contract_v1",
        "PF_RESEARCH_MODEL": MODEL_NAME,
        "classification": [
            "JI_CHEN_STYLE_DIAGONAL_REACTION_DIFFUSION",
            "FIXED_CTOT_CONSERVATIVE_IMPLEMENTATION",
            "COARSE_GRAINED_RESEARCH_MODEL",
        ],
        "dx_nm": DX_NM,
        "lambda_nm": LAMBDA_NM,
        "lambda_over_dx": LAMBDA_NM / DX_NM,
        "phase_kinetics": "FINITE_LPHI_BE",
        "Lphi_over_strict_diffusion_limit": LPHI_RATIO,
        "finite_interface_mode": "off",
        "coarse_interface_mobility_mode": "off",
        "coarse_interface_mobility_a_M": 0.0,
        "D_beta": 0.0,
        "authoritative_conserved_state": "C_B_tot",
        "transport": "ctot_mimetic_be/mimetic_shared_face_v1",
        "transport_nonlinear_coordinate": "adaptive_logit_feasible_ctot_v1",
        "phase_constraint": "semismooth_pdas_v1",
        "mechanics_precision_mode": MECHANICS_PRECISION_MODE,
        "mechanics_acceptance_mode": MECHANICS_ACCEPTANCE_MODE,
        "GP_source_reintegrated": False,
        "S3_reintegrated": False,
        "physical_interface_width_claimed": False,
        "formal_production_executed": False,
    }


def parameter_row(temperature_c: float) -> tuple[dict[str, float], dict[str, object]]:
    inputs = base_inputs(temperature_c)
    payload = unit.generate_payload(inputs)
    strict = corrected_limit(inputs, temperature_c)
    pf = payload["pf_params"]
    selected_code = LPHI_RATIO * strict["L_phi_diff_code"]
    selected_physical = LPHI_RATIO * strict["L_phi_diff_physical_m3_J_s"]
    row = {
        "T_C": temperature_c,
        "T_K": strict["T_K"],
        "dx_nm": DX_NM,
        "lambda_nm": LAMBDA_NM,
        "lambda_over_dx": LAMBDA_NM / DX_NM,
        "gamma_J_m2": inputs.gamma,
        "W_physical_J_m3": strict["W_physical_J_m3"],
        "kappa_physical_J_m": 1.5 * inputs.gamma * inputs.lambda_sm,
        "W_code": float(pf["W"]),
        "kappa_code": float(pf["kappa_phi"]),
        "D_alpha_m2_s": strict["D_alpha_m2_s"],
        "D_alpha_code": float(pf["D_alpha"]),
        "D_beta_code": 0.0,
        "xB_eq": strict["x_eq"],
        "zeta0_strict": strict["zeta0"],
        "zeta_J_mol": strict["zeta_J_mol"],
        "L_phi_diff_physical_m3_J_s": strict[
            "L_phi_diff_physical_m3_J_s"
        ],
        "L_phi_diff_code": strict["L_phi_diff_code"],
        "L_phi_ratio": LPHI_RATIO,
        "L_phi_selected_physical_m3_J_s": selected_physical,
        "L_phi_selected_code": selected_code,
        "time_scale_s": strict["time_scale_s"],
        "historical_converter_L_phi_code": strict["historical_reference_L_phi_code"],
        "historical_to_strict_limit_ratio": strict[
            "historical_reference_ratio_to_strict_limit"
        ],
    }
    return row, payload


def runtime_params(temperature_c: float, contract_hash: str,
                   reference_hash: str) -> dict[str, object]:
    row, payload = parameter_row(temperature_c)
    values = dict(payload["main_cuda_overrides"])
    values.update({
        "pf_params_schema_version": 2,
        "model_mode": "two_phase",
        "PF_RESEARCH_MODEL": MODEL_NAME,
        "coarse_model_name": MODEL_NAME,
        "coarse_model_version": MODEL_VERSION,
        "fine_reference_hash": reference_hash,
        # This legacy field stores the immutable model-contract hash here; no
        # fitted coarse calibration is claimed by the Ji--Chen route.
        "coarse_calibration_hash": contract_hash,
        "coarse_uncertainty_version": UNCERTAINTY_VERSION,
        "mechanics_precision_mode": MECHANICS_PRECISION_MODE,
        "mechanics_acceptance_mode": MECHANICS_ACCEPTANCE_MODE,
        "eta_floor_version": MECHANICS_ETA_FLOOR_VERSION,
        "eta_accept": MECHANICS_ETA_ACCEPT,
        "double_oracle_contract_hash": MECHANICS_DOUBLE_ORACLE_CONTRACT_HASH,
        "residual_normalization_version": MECHANICS_RESIDUAL_NORMALIZATION_VERSION,
        "composition_evolution_mode": "ctot_mimetic_be",
        "ctot_transport_nonlinear_coordinate": "adaptive_logit_feasible_ctot_v1",
        "ctot_outer_acceleration": "OFF",
        "pf_composition_mode": "legacy",
        "pf_y_update_mode": "lagged_rhs",
        "PHASE_KINETICS_MODE": "FINITE_LPHI_BE",
        "L_phi": row["L_phi_selected_code"],
        "L_phi_code_value": row["L_phi_selected_code"],
        "L_phi_physical_value": row["L_phi_selected_physical_m3_J_s"],
        "L_phi_calibration_mode": "one_sided_diffusion_controlled",
        "zeta0_phi": 1.0,
        "zeta_phi": row["zeta_J_mol"],
        "D_beta_for_calibration": 0.0,
        "calibration_script_hash": sha256(
            ROOT / "scripts/correction1_ji_chen_mapping.py"
        ),
        "ctot_phase_semismooth_pdas_enabled": 1,
        "ctot_phase_restart_solver_migration_allowed": 0,
        "ctot_finite_interface_antitrapping_enabled": 0,
        "finite_interface_resolution_test_override": 0,
        "coarse_interface_mobility_mode": "off",
        "coarse_interface_mobility_a_M": 0.0,
        "thermo_convex_extrapolation_enabled": 1,
        "y_update_mass_projection_enabled": 0,
        "GP_population_mode": "OFF",
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
        "elastic_enabled": 0,
        "ctot_elastic_validation_enabled": 0,
        "ctot_diagnostics_enabled": 1,
    })
    return values


def write_params(path: Path, values: dict[str, object]) -> None:
    lines = [
        "# Generated by scripts/prepare_jc4_research_model.py",
        "# lambda=4 nm is a numerical mesoscale interface, not an atomistic claim.",
    ]
    for key in sorted(values):
        value = values[key]
        if isinstance(value, str) or isinstance(value, int):
            lines.append(f"{key}={value}")
        else:
            lines.append(f"{key}={float(value):.17e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(report_root: Path, rows: list[dict[str, object]],
                  contract_hash: str, reference_hash: str) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    write_csv(report_root / "jc4_parameter_contract.csv", rows)
    table = "\n".join(
        f"| {int(row['T_C'])} | {row['L_phi_diff_code']:.12g} | "
        f"{row['L_phi_selected_code']:.12g} | {row['D_alpha_code']:.12g} | "
        f"{row['xB_eq']:.12g} |" for row in rows
    )
    (report_root / "jc4_model_definition.md").write_text(f"""# JC4 model definition

## Selected default-off candidate

`PF_RESEARCH_MODEL={MODEL_NAME}` selects a Ji--Chen-style diagonal
reaction--diffusion model implemented with authoritative fixed `C_B_tot`.
It is a coarse-grained research model: `dx=1 nm`, `lambda=4 nm`, four cells
across the interface parameter. The 4 nm value is not an atomistic interface
width claim.

The selected path is `FINITE_LPHI_BE` at `0.90 Lphi_diff`, one-sided
`D_beta=0`, `ctot_mimetic_be`, and semismooth PDAS. Finite-interface
corrections, quasi-equilibrium phase response, interface mobility boost,
RSMD, GP release/growth, direct beta injection, and S3 are all off.

The earlier short-time calibration remains failed evidence: best flux error
13.419%, inventory error 78.760%, and trajectory error 82.249%. This candidate
therefore makes no strict short-time fine-interface-equivalence claim.

`model_contract_hash={contract_hash}`
`reference_evidence_hash={reference_hash}`
""", encoding="utf-8")
    (report_root / "jc4_parameter_contract.md").write_text(f"""# JC4 parameter contract

The strict one-sided Ji--Chen diffusion limit uses the Supplement S2.18--S2.20
outer domain and `zeta0=1`. It is recomputed for every temperature and must not
be replaced by the historical finite-window converter value.

| T (C) | Lphi_diff code | selected 0.90 limit | D_alpha code | xB_eq |
|---:|---:|---:|---:|---:|
{table}

For both rows, `W=1`, `kappa=2` in code units, corresponding to
`W=12 gamma/lambda=5.04e8 J/m^3` and
`kappa=3 gamma lambda/2=1.008e-9 J/m`. `a_M=0`, finite-interface correction is
off, and `D_beta=0`.
""", encoding="utf-8")
    (report_root / "jc4_mass_and_phase_ownership.md").write_text("""# JC4 mass and phase ownership

The sole PF mass owner is `M_PF = sum(C_B_tot_i V_i)`, with
`C_B_tot = h(phi) v_B + (1-h(phi)) xB_alpha` and `v_B=1`.
The phase-only transaction keeps local `C_B_tot` fixed. No independently
evolved beta-mass field exists. Future fixed GP reservoirs add a separate
inventory only through `M_system = M_PF + sum(N_GP)` and are not active in the
qualification stages.

`V_beta^h = integral h(phi) dV` and `N_beta^eq = c v_B V_beta^h` are phase
diagnostics, not a second conserved ledger. Raw diffuse-band inventory is not
used as a conservation owner.
""", encoding="utf-8")
    (report_root / "jc4_beta_observable_contract.md").write_text("""# JC4 beta observable contract

Planar comparisons report subcell equimolar interface position, `h(phi)`
volume, its change from the initial state, total PF mass, matrix profile,
depletion width, finite-box equilibrium composition, and growth-law exponent.
Curved cases report `R_eff=(3 V_h/4 pi)^(1/3)` in 3D (or the geometry-specific
equivalent radius). Multi-particle work requires connected-component h-volume,
subcell geometry, centroid, area, and restart-safe object matching.

Metadata radius, support-threshold radius, h-weighted equivalent radius, and
inventory-equivalent amount are distinct quantities and must not be mixed.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "tmp/jc4_model_contract"
    )
    parser.add_argument(
        "--report-root", type=Path,
        default=ROOT / "reports/pf_ctot_production_candidate",
    )
    args = parser.parse_args()

    contract = model_contract()
    contract_hash = canonical_hash(contract)
    evidence_paths = [
        ROOT / "scripts/correction1_ji_chen_mapping.py",
        ROOT / "Unit_Psedobinary.py",
        ROOT / "thermo_utils.h",
        ROOT / "phase_functions.h",
        Path("/Users/heng/Documents/STO_SM_with_elastic_notes.pdf"),
    ]
    reference_hash = canonical_hash(
        {path.name: sha256(path) for path in evidence_paths}
    )
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for temperature_c in (380.0, 400.0):
        row, _ = parameter_row(temperature_c)
        rows.append(row)
        write_params(
            out / f"jc4_T{int(temperature_c)}_pf_only.params",
            runtime_params(temperature_c, contract_hash, reference_hash),
        )
    manifest = {
        **contract,
        "model_contract_hash": contract_hash,
        "reference_evidence_hash": reference_hash,
        "parameters": rows,
        "params": [
            f"jc4_T{int(row['T_C'])}_pf_only.params" for row in rows
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "examples/jc4_model_contract.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    write_reports(args.report_root.resolve(), rows, contract_hash, reference_hash)
    print(f"model={MODEL_NAME}")
    print(f"contract_hash={contract_hash}")
    for row in rows:
        print(
            f"T{int(row['T_C'])}_L_phi_diff_code={row['L_phi_diff_code']:.17e} "
            f"selected={row['L_phi_selected_code']:.17e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
