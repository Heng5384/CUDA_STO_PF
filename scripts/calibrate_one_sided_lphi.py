#!/usr/bin/env python3
"""Calibrate beta-phi kinetics for the one-sided PF Ctot candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import Unit_Psedobinary as unit  # noqa: E402


def parse_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_float(values: dict[str, str], key: str) -> float:
    if key not in values:
        raise KeyError(f"missing required parameter: {key}")
    value = float(values[key])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite parameter: {key}")
    return value


def calibrate(base: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    values = parse_params(base)
    temperature_C = required_float(values, "temperature_C")
    temperature_K = temperature_C + 273.15
    D_alpha_phys = unit.D_Ag_in_PbTe_m2_per_s(temperature_K)
    D_alpha_code = required_float(values, "D_alpha")
    t_real_unit = required_float(values, "t_real_unit")
    dx_code = required_float(values, "dx")
    lambda_sm = required_float(values, "lambda_sm_m")
    gamma = required_float(values, "gamma_Jm2")
    Vm_alpha = required_float(values, "Vm_alpha_0_phys_m3mol")
    v_A = required_float(values, "v_A")
    v_B = required_float(values, "v_B")
    if not (lambda_sm > 0.0 and gamma > 0.0 and Vm_alpha > 0.0):
        raise ValueError("invalid interface or molar-volume input")

    # Verify that the runtime code diffusivity and physical time use the same
    # length scale before using them to convert L_phi.
    dx_phys = math.sqrt(D_alpha_phys * t_real_unit / D_alpha_code) * dx_code
    D_reconstructed = D_alpha_code * (dx_phys / dx_code) ** 2 / t_real_unit
    if abs(D_reconstructed - D_alpha_phys) / D_alpha_phys > 1.0e-12:
        raise ValueError("D_alpha physical/code reconstruction failed")

    w_phys, kappa_phys, _ = unit.compute_interface_params(gamma, lambda_sm)
    x_eq = unit.xAg2Te_eq_from_T(temperature_K)
    zeta_phi = unit.compute_beta_phi_zeta(temperature_K, x_eq, v_A, v_B)
    zeta0_analytic = unit.compute_beta_phi_one_sided_zeta0_analytic(
        lambda_sm, w_phys, kappa_phys)
    resolutions = (101, 201, 501, 1001, 2001, 4001, 8001, 16001, 32001)
    convergence: list[dict[str, object]] = []
    for n_quad in resolutions:
        numerical = unit.compute_beta_phi_one_sided_zeta0_quadrature(
            lambda_sm, w_phys, kappa_phys, n_quad=n_quad)
        convergence.append({
            "T_C": temperature_C,
            "n_quad": n_quad,
            "zeta0_phi_analytic": zeta0_analytic,
            "zeta0_phi_numerical": numerical,
            "abs_error": abs(numerical - zeta0_analytic),
            "rel_error": abs(numerical - zeta0_analytic) /
                         max(abs(zeta0_analytic), 1.0e-300),
        })
    if float(convergence[-1]["abs_error"]) > 5.0e-10:
        raise ValueError("one-sided zeta0 quadrature has not converged")

    c_tot = 1.0 / Vm_alpha
    zprod_phi = zeta_phi * zeta0_analytic
    L_phi_phys = 4.0 * (v_A + v_B) * D_alpha_phys / (
        3.0 * c_tot * lambda_sm * lambda_sm * abs(zprod_phi))
    L_phi_code = L_phi_phys * w_phys * t_real_unit

    legacy_L_phi_code = required_float(values, "L_phi")
    legacy_L_phi_phys = legacy_L_phi_code / (w_phys * t_real_unit)
    gp_nu_A = required_float(values, "gp_reaction_nu_A")
    gp_nu_B = required_float(values, "gp_reaction_nu_B")
    gp_x_eq = required_float(values, "gp_xB_eq_alpha_for_eta")
    dmuA, dmuB = unit.component_mu_slopes_from_scalar_backend(
        temperature_K, x_eq)
    zeta_eta = unit.compute_zeta(dmuA, dmuB, gp_nu_A, gp_nu_B, gp_x_eq)
    gp_l_eta = required_float(values, "gp_l_eta_nm") * 1.0e-9
    gp_W_eta = required_float(values, "gp_W_eta_phys")
    gp_kappa_eta = required_float(values, "gp_kappa_eta_phys")
    gp_D_ratio = required_float(values, "gp_D_ratio")
    zeta0_eta = unit.compute_gp_eta_zeta0(
        gp_l_eta, gp_W_eta, gp_kappa_eta,
        D_alpha_phys, D_alpha_phys * gp_D_ratio, n_quad=2001)

    result: dict[str, object] = {
        "T_C": temperature_C,
        "xB_eq_matrix": x_eq,
        "v_A_phi": v_A,
        "v_B_phi": v_B,
        "D_alpha_phys_m2_s": D_alpha_phys,
        "D_alpha_code": D_alpha_code,
        "D_beta_candidate_m2_s": 0.0,
        "lambda_sm_m": lambda_sm,
        "w_phys_J_m3": w_phys,
        "kappa_phys_J_m": kappa_phys,
        "t_real_unit_s": t_real_unit,
        "zeta_phi": zeta_phi,
        "zeta0_phi_analytic": zeta0_analytic,
        "zeta0_phi_numerical_32001": convergence[-1]["zeta0_phi_numerical"],
        "zeta0_phi_abs_error_32001": convergence[-1]["abs_error"],
        "legacy_zeta_eta": zeta_eta,
        "legacy_zeta0_eta": zeta0_eta,
        "legacy_L_phi_phys": legacy_L_phi_phys,
        "legacy_L_phi_code": legacy_L_phi_code,
        "one_sided_L_phi_phys": L_phi_phys,
        "one_sided_L_phi_code": L_phi_code,
        "ratio_new_over_legacy": L_phi_code / legacy_L_phi_code,
        "thermodynamic_backend_hash": sha256(ROOT / "thermo_utils.h"),
        "calibration_script_hash": sha256(Path(__file__).resolve()),
        "unit_conversion_script_hash": sha256(ROOT / "Unit_Psedobinary.py"),
        "base_parameter_hash": sha256(base),
    }
    return result, convergence


def write_overlay(base: Path, output: Path, result: dict[str, object]) -> None:
    text = base.read_text()
    if not text.endswith("\n"):
        text += "\n"
    fields = {
        "L_phi": result["one_sided_L_phi_code"],
        "L_phi_calibration_mode": "one_sided_diffusion_controlled",
        "L_phi_physical_value": result["one_sided_L_phi_phys"],
        "L_phi_code_value": result["one_sided_L_phi_code"],
        "zeta_phi": result["zeta_phi"],
        "zeta0_phi": result["zeta0_phi_analytic"],
        "D_beta_for_calibration": 0.0,
        "zeta_eta": result["legacy_zeta_eta"],
        "zeta0_eta": result["legacy_zeta0_eta"],
        "thermodynamic_backend_hash": result["thermodynamic_backend_hash"],
        "calibration_script_hash": result["calibration_script_hash"],
    }
    text += "\n# Prompt-7d one-sided beta-phi kinetic calibration overlay\n"
    for key, value in fields.items():
        if isinstance(value, str):
            text += f"{key}={value}\n"
        else:
            text += f"{key}={float(value):.17e}\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--output-params", type=Path, required=True)
    parser.add_argument("--values-csv", type=Path, required=True)
    parser.add_argument("--quadrature-csv", type=Path, required=True)
    args = parser.parse_args()
    result, convergence = calibrate(args.base_params)
    write_overlay(args.base_params, args.output_params, result)
    write_csv(args.values_csv, [result])
    write_csv(args.quadrature_csv, convergence)
    print(f"T_C={result['T_C']:.17g}")
    print(f"one_sided_L_phi_phys={result['one_sided_L_phi_phys']:.17e}")
    print(f"one_sided_L_phi_code={result['one_sided_L_phi_code']:.17e}")
    print(f"L_phi_ratio_new_over_legacy={result['ratio_new_over_legacy']:.17e}")
    print(f"zeta_phi={result['zeta_phi']:.17e}")
    print(f"zeta0_phi={result['zeta0_phi_analytic']:.17e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
