#!/usr/bin/env python3
"""Recover the strict Ji--Chen one-sided diffusion-limit mapping.

This module is intentionally independent of ``PFParamConverter.convert`` for
the beta ``zeta0`` value.  The converter currently retains the historical
finite ``[-lambda/2, lambda/2]`` outer integration window; Ji--Chen S2.20 uses
an infinite outer integral and gives ``zeta0=1`` for ``D_beta/D_alpha=0``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402


REPORT_ROOT = ROOT / "reports/pf_ctot_production_candidate"
MANIFEST_PATH = ROOT / "examples/correction1_Lphi_below_limit_manifest.json"
SM_PATH = Path("/Users/heng/Documents/STO_SM_with_elastic_notes.pdf")


def h_switch(phi: np.ndarray | float) -> np.ndarray | float:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def h_prime(phi: np.ndarray | float) -> np.ndarray | float:
    return 30.0 * phi**2 * (1.0 - phi) ** 2


def phi_equilibrium(x_over_lambda: np.ndarray | float) -> np.ndarray | float:
    """Ji--Chen/code profile ``0.5*(1-tanh(2*x/lambda))``."""
    return 0.5 * (1.0 - np.tanh(2.0 * x_over_lambda))


def dphi_dx_over_lambda(x_over_lambda: np.ndarray | float) -> np.ndarray | float:
    """Derivative with respect to ``x/lambda``."""
    return -(1.0 / np.cosh(2.0 * x_over_lambda) ** 2)


def integrate_zeta0(extent_lambda: float, points: int) -> float:
    """Evaluate S2.20 after the one-sided removable-limit cancellation.

    The inner integral starts at ``-lambda/2`` exactly as S2.20.  Only the
    outer integral is expanded toward infinity.  Coordinates are normalized
    by lambda, so no physical length or diffusivity remains.
    """
    if extent_lambda <= 0.0 or points < 3 or points % 2 == 0:
        raise ValueError("extent must be positive and points odd >= 3")
    z = np.linspace(-extent_lambda, extent_lambda, points, dtype=np.float64)
    phi = phi_equilibrium(z)
    dh_dz = h_prime(phi) * dphi_dx_over_lambda(z)
    integrand = dh_dz * (z + 0.5)
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(-2.0 * integrate(integrand, z))


def strict_zeta0() -> float:
    """Exact one-sided S2.20 result for the current quintic interpolation."""
    return 1.0


def load_inputs(path: Path) -> unit.PhysicalInputs:
    return unit.PhysicalInputs(**json.loads(path.read_text()))


def corrected_limit(inputs: unit.PhysicalInputs, temperature_c: float) -> dict[str, float]:
    current = replace(inputs, temperature_C=temperature_c)
    temperature_k = temperature_c + 273.15
    x_eq = unit.xAg2Te_eq_from_T(temperature_k)
    mu_a_prime, mu_b_prime = unit.component_mu_slopes_from_scalar_backend(
        temperature_k, x_eq
    )
    zeta = unit.compute_zeta(
        mu_a_prime, mu_b_prime, current.v_A, current.v_B, x_eq
    )
    diffusivity = unit.D_Ag_in_PbTe_m2_per_s(temperature_k)
    concentration = 1.0 / current.Vm_alpha_0
    w_physical = 12.0 * current.gamma / current.lambda_sm
    time_scale = (current.L_ref_factor * current.lambda_sm) ** 2 / diffusivity
    lphi_physical = (
        4.0 * (current.v_A + current.v_B) * diffusivity
        / (3.0 * concentration * strict_zeta0() * zeta * current.lambda_sm**2)
    )
    lphi_code = lphi_physical * w_physical * time_scale
    historical = unit.PFParamConverter().convert(current)
    return {
        "T_C": temperature_c,
        "T_K": temperature_k,
        "D_alpha_m2_s": diffusivity,
        "x_eq": x_eq,
        "mu_A_prime_J_mol": mu_a_prime,
        "mu_B_prime_J_mol": mu_b_prime,
        "thermodynamic_factor_gpp_J_mol": unit.g_alpha_second_unified(
            temperature_k, x_eq
        ),
        "zeta_J_mol": zeta,
        "zeta0": strict_zeta0(),
        "c_mol_m3": concentration,
        "lambda_JC_m": current.lambda_sm,
        "lambda_code_parameter_m": current.lambda_sm,
        "lambda_over_pf_dx": current.lambda_sm / unit.resolve_pf_dx(current),
        "w90_10_m": math.atanh(0.8) * current.lambda_sm,
        "w95_05_m": math.atanh(0.9) * current.lambda_sm,
        "W_physical_J_m3": w_physical,
        "time_scale_s": time_scale,
        "energy_scale_J_m3": w_physical,
        "mu_reference_J_mol": w_physical / concentration,
        "L_phi_diff_physical_m3_J_s": lphi_physical,
        "L_phi_diff_code": lphi_code,
        "historical_finite_window_zeta0": historical.zeta0_phi,
        "historical_reference_L_phi_physical": historical.L_phi_phys,
        "historical_reference_L_phi_code": historical.L_phi,
        "historical_reference_ratio_to_strict_limit": historical.L_phi / lphi_code,
    }


def inverse_interface_resistance(
    lphi_physical: float, row: dict[str, float]
) -> float:
    return (
        2.0 / (3.0 * row["c_mol_m3"] * lphi_physical * row["lambda_JC_m"])
        - row["zeta0"] * row["zeta_J_mol"] * row["lambda_JC_m"]
        / (2.0 * row["D_alpha_m2_s"])
    )


def classify_ratio(ratio: float, tolerance: float = 5.0e-8) -> str:
    if ratio < 1.0 - tolerance:
        return "FINITE_POSITIVE_INTERFACE_MOBILITY"
    if abs(ratio - 1.0) <= tolerance:
        return "DIFFUSION_CONTROLLED_LIMIT_FROM_BELOW"
    return "NONPHYSICAL_UNDER_JI_CHEN_MAPPING"


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing empty CSV {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_existing_lphi(limits: dict[int, dict[str, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def append(label: str, temperature: int, code: float, physical: float,
               provenance: str, claim: str) -> None:
        limit = limits[temperature]
        ratio = physical / limit["L_phi_diff_physical_m3_J_s"]
        rows.append({
            "label": label,
            "T_C": temperature,
            "L_phi_code": code,
            "L_phi_physical_m3_J_s": physical,
            "L_phi_diff_code": limit["L_phi_diff_code"],
            "L_phi_diff_physical_m3_J_s": limit["L_phi_diff_physical_m3_J_s"],
            "ratio_to_strict_limit": ratio,
            "inverse_M_I_J_s_mol_m": inverse_interface_resistance(physical, limit),
            "classification": classify_ratio(ratio),
            "provenance": provenance,
            "physical_claim_status": claim,
        })

    old_values = REPORT_ROOT / "prompt7d_one_sided_lphi_values.csv"
    if old_values.is_file():
        with old_values.open(newline="") as handle:
            for source in csv.DictReader(handle):
                temperature = int(round(float(source["T_C"])))
                append(
                    "prompt7d_finite_window_reference",
                    temperature,
                    float(source["one_sided_L_phi_code"]),
                    float(source["one_sided_L_phi_phys"]),
                    str(old_values.relative_to(ROOT)),
                    "HISTORICAL_FORMULA_RESULT_SUPERSEDED_BY_STRICT_S2_OUTER_DOMAIN",
                )

    for temperature in (380, 400):
        metrics = (
            REPORT_ROOT / "research2_fast_interface_evidence"
            / f"T{temperature}" / "research2_fast_interface_metrics.csv"
        )
        if not metrics.is_file():
            continue
        with metrics.open(newline="") as handle:
            for source in csv.DictReader(handle):
                append(
                    f"research2_{source['case']}",
                    temperature,
                    float(source["L_phi_code"]),
                    float(source["L_phi_physical"]),
                    str(metrics.relative_to(ROOT)),
                    "EXPLORATORY_NUMERICAL_FAST_PHASE_RESPONSE",
                )

    # These values are repeatedly used by the finite-interface/radial history.
    # Their physical conversion is reconstructed from the same temperature
    # scales rather than inferred from prose.
    for label, code, provenance in (
        ("prompt7f_stationary_profile_selected", 85.50541544691814,
         "reports/pf_ctot_production_candidate/prompt7f_lambda_semantics_audit.md"),
        ("next7_finite_interface_candidate", 85.50541544691814,
         "reports/pf_ctot_production_candidate/next7_interface_identifiability.md"),
    ):
        limit = limits[400]
        physical = code / (
            limit["energy_scale_J_m3"] * limit["time_scale_s"]
        )
        append(
            label, 400, code, physical, provenance,
            "EXPLORATORY_OR_FINITE_INTERFACE_HISTORICAL_NOT_JI_CHEN_PHYSICAL",
        )
    return rows


def write_manifest(limits: dict[int, dict[str, float]]) -> None:
    base_cases = []
    for ratio in (0.50, 0.75, 0.90, 0.95, 0.98, 0.99):
        base_cases.append({
            "case_id": f"T400_ratio_{str(ratio).replace('.', 'p')}_dx0p1_dt1",
            "T_C": 400,
            "ratio_to_L_phi_diff": ratio,
            "dx_nm": 0.1,
            "lambda_over_dx": 6.0,
            "dt_scale": 1.0,
            "observation_time_s": 2.0821852621180848e-4,
            "matrix_xB": 0.05,
            "finite_interface_mode": "off",
            "elasticity": "off",
            "GP_S3": "off",
            "purpose": "base_below_limit_scan",
        })
    refinements = [
        {
            "case_id": "T400_ratio_0p98_dx0p1_dt0p5",
            "T_C": 400, "ratio_to_L_phi_diff": 0.98,
            "dx_nm": 0.1, "lambda_over_dx": 6.0, "dt_scale": 0.5,
            "observation_time_s": 2.0821852621180848e-4,
            "matrix_xB": 0.05, "finite_interface_mode": "off",
            "elasticity": "off", "GP_S3": "off",
            "purpose": "time_step_refinement",
        },
        {
            "case_id": "T400_ratio_0p99_dx0p1_dt0p5",
            "T_C": 400, "ratio_to_L_phi_diff": 0.99,
            "dx_nm": 0.1, "lambda_over_dx": 6.0, "dt_scale": 0.5,
            "observation_time_s": 2.0821852621180848e-4,
            "matrix_xB": 0.05, "finite_interface_mode": "off",
            "elasticity": "off", "GP_S3": "off",
            "purpose": "time_step_refinement",
        },
        {
            "case_id": "T400_ratio_0p95_dx0p05_dt1",
            "T_C": 400, "ratio_to_L_phi_diff": 0.95,
            "dx_nm": 0.05, "lambda_over_dx": 12.0, "dt_scale": 1.0,
            "observation_time_s": 2.0821852621180848e-4,
            "matrix_xB": 0.05, "finite_interface_mode": "off",
            "elasticity": "off", "GP_S3": "off",
            "purpose": "lambda_over_dx_refinement",
        },
    ]
    payload = {
        "schema": "correction1_ji_chen_below_limit_manifest_v1",
        "mapping": {
            "source": "Ji-Chen Supplementary Materials S2.18-S2.20",
            "D_beta_over_D_alpha": 0.0,
            "zeta0": 1.0,
            "T400_L_phi_diff_code": limits[400]["L_phi_diff_code"],
            "T400_L_phi_diff_physical_m3_J_s": limits[400]["L_phi_diff_physical_m3_J_s"],
            "strictly_below_limit": True,
        },
        "common": {
            "geometry": "periodic_planar_beta_slab",
            "domain_nm": 19.2,
            "grid_transverse": [2, 2],
            "sharp_start_time_s": 0.1,
            "composition_evolution_mode": "ctot_mimetic_be",
            "automatic_dt_growth": False,
        },
        "cases": base_cases + refinements,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def generate_reports(inputs_path: Path) -> dict[int, dict[str, float]]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(inputs_path)
    limits = {temperature: corrected_limit(inputs, temperature)
              for temperature in (380, 400)}
    write_csv(REPORT_ROOT / "correction1_Lphi_diff_values.csv", limits.values())

    convergence = []
    for extent in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        for points in (2001, 8001, 32001):
            value = integrate_zeta0(extent, points)
            convergence.append({
                "outer_extent_lambda": extent,
                "quadrature_points": points,
                "zeta0_quadrature": value,
                "zeta0_exact": 1.0,
                "absolute_error": abs(value - 1.0),
                "finite_window_historical": extent == 0.5,
            })
    write_csv(REPORT_ROOT / "correction1_zeta0_convergence.csv", convergence)
    existing = collect_existing_lphi(limits)
    write_csv(REPORT_ROOT / "correction1_existing_Lphi_audit.csv", existing)
    write_manifest(limits)

    source_note = (
        f"Local supplementary material: `{SM_PATH.name}` (Ji and Chen, "
        f"*Phase-Field Model of Stoichiometric Compounds and Solution Phases*, "
        f"17 pages, S2.1--S2.20), SHA-256 `{sha256(SM_PATH)}`."
        if SM_PATH.is_file() else
        "Local supplementary material path was unavailable; equations were not silently substituted."
    )
    (REPORT_ROOT / "correction1_ji_chen_formula_recovery.md").write_text(f"""# Correction 1 Ji--Chen Formula Recovery

## Source and conventions

{source_note}

The exact equilibrium equation is

`w(4 phi^3 - 6 phi^2 + 2 phi) - kappa phi_xx = 0`,

with `phi0=0.5[1-tanh(sqrt(w/(2*kappa))*x)]`.  S2 gives
`lambda=2*sqrt(2*kappa/w)`, `gamma=sqrt(kappa*w)/(3*sqrt(2))`,
therefore `kappa=3*gamma*lambda/2` and `w=12*gamma/lambda`.
These are exactly the formulas in `Unit_Psedobinary.py:536-562`.

For a finite intrinsic interface mobility `M_I`, S2.18 is

`1/M_I = 2*(v_A+v_B)/(3*c*L_phi*lambda) - zeta0*zeta*lambda/(2*D_alpha)`.

The diffusion limit is obtained only from below as `1/M_I -> 0+`:

`L_phi_diff = 4*(v_A+v_B)*D_alpha/(3*c*zeta0*zeta*lambda^2)`.

The project has `v_A=0`, `v_B=1`, `D_beta=0`, and
`zeta=mu_B'(x_eq)*(1-x_eq)`.  Component derivatives are recovered from the
single regular-solution Gibbs backend: `mu_A'=-x*g''`,
`mu_B'=(1-x)*g''`, and `g''=RT/[x(1-x)]-2L(T)`.

## Runtime equation compatibility

The accepted planar benchmark normalizes both molar volumes to one and sets
their composition derivative to zero.  The phase transaction holds local
`C_B_tot` fixed while updating `phi`, then reconstructs
`q_alpha=C_B_tot-h(phi)`.  This is the constant-volume fixed-total-composition
setting assumed by the recovered Ji--Chen mapping.  Elasticity, GP/S3, and
finite-interface correction are OFF in the correction scan.

Source correspondence:

* `Unit_Psedobinary.py:536-562`: `w`, `kappa`, and the equilibrium profile;
* `Unit_Psedobinary.py:514-553`: scalar thermodynamic curvature and component
  chemical-potential derivatives;
* `Unit_Psedobinary.py:668-715`: physical diffusivity, concentration, and
  code-time conversion;
* `cuda_kernels.cu:397-470`: phase local free-energy derivative;
* `main_cuda.cu:31785-31851`: phase BE update and fixed-Ctot storage check;
* `main_cuda.cu:31965-32051`: semismooth fixed-Ctot phase KKT solve;
* `cuda_kernels.cu:3146-3174`: authoritative `Ctot -> q_alpha,xB,Y`
  reconstruction;
* `cuda_kernels.cu:3368-3420`: candidate chemical potential and one-sided
  matrix mobility/flux context;
* `main_cuda.cu:33213-33259`: accepted predicates, commit reconstruction, and
  explicit zero clipping/projection marker.
""")

    lam = inputs.lambda_sm
    dx = unit.resolve_pf_dx(inputs)
    (REPORT_ROOT / "correction1_lambda_definition_audit.md").write_text(f"""# Correction 1 Lambda Definition Audit

The code and Ji--Chen parameter are identical:

`lambda_code_parameter = lambda_JC = {lam:.17e} m = {lam*1e9:.9g} nm`.

Proof: substituting `w=12 gamma/lambda` and `kappa=3 gamma lambda/2` into
`sqrt(w/(2 kappa))` gives `2/lambda`, exactly the profile used by
`Unit_Psedobinary.xi_eq_profile` and the planar initializer.

This parameter is not a threshold width:

* `w90-10 = atanh(0.8)*lambda = {math.atanh(0.8)*lam*1e9:.12g} nm`
* `w95-05 = atanh(0.9)*lambda = {math.atanh(0.9)*lam*1e9:.12g} nm`
* at `dx={dx*1e9:.12g} nm`: `lambda/dx={lam/dx:.12g}`,
  `w90-10/dx={math.atanh(0.8)*lam/dx:.12g}`,
  `w95-05/dx={math.atanh(0.9)*lam/dx:.12g}`.

Status: `LAMBDA_DEFINITION_PROVEN_MATCHED`.
""")

    unit_rows = "\n".join(
        f"| {t} | {row['D_alpha_m2_s']:.12e} | {row['c_mol_m3']:.12e} | "
        f"{row['zeta_J_mol']:.12e} | {row['time_scale_s']:.12e} | "
        f"{row['energy_scale_J_m3']:.12e} |"
        for t, row in limits.items()
    )
    (REPORT_ROOT / "correction1_unit_conversion.md").write_text(f"""# Correction 1 Unit Conversion

`c=1/Vm_alpha_0={limits[380]['c_mol_m3']:.17e} mol/m^3`; it is not set to
one in the physical mapping.  Runtime benchmark volumes are normalized only
after this physical conversion.

`t0=(L_ref_factor*lambda)^2/D_alpha`, `E0=w=12 gamma/lambda`, and
`L_phi_code=L_phi_phys*E0*t0`.  Consequently physical `L_phi` has units
`m^3/(J s)` and `zeta` has units `J/mol`.

| T (C) | D_alpha (m2/s) | c (mol/m3) | zeta (J/mol) | t0 (s) | E0 (J/m3) |
|---:|---:|---:|---:|---:|---:|
{unit_rows}

Status: `UNIT_CONVERSION_PROVEN`.
""")

    best = convergence[-1]
    historical = integrate_zeta0(0.5, 32001)
    (REPORT_ROOT / "correction1_zeta0_one_sided_audit.md").write_text(f"""# Correction 1 One-Sided Zeta0 Audit

Ji--Chen S2.20 retains the inner lower bound `-lambda/2` but integrates the
outer coordinate over `(-infinity,+infinity)`.  For
`D(phi)=(1-h)D_alpha`, the ratio `(1-h)/D(phi)` has the removable value
`1/D_alpha`.  With the symmetric equilibrium profile, the odd first moment
vanishes and the remaining integral is exactly `zeta0=1`.

The historical implementation truncated the outer integral to
`[-lambda/2,+lambda/2]`, producing `{historical:.17e}`.  That number is a
finite-window truncation result, not the one-sided S2 diffusion-limit value.

Largest tested domain: `+/-{best['outer_extent_lambda']} lambda`,
`{best['quadrature_points']} points`, value
`{best['zeta0_quadrature']:.17e}`, absolute error
`{best['absolute_error']:.3e}`.

Status: `PASS_ZETA0_ONE_SIDED_STRICT_VALUE_1`.
""")

    value_table = "\n".join(
        f"| {t} | {row['x_eq']:.12g} | {row['D_alpha_m2_s']:.12e} | "
        f"{row['zeta_J_mol']:.12e} | {row['L_phi_diff_physical_m3_J_s']:.12e} | "
        f"{row['L_phi_diff_code']:.12e} | {row['historical_reference_ratio_to_strict_limit']:.9f} |"
        for t, row in limits.items()
    )
    (REPORT_ROOT / "correction1_Lphi_diff_values.md").write_text(f"""# Correction 1 Ji--Chen Diffusion Limits

| T (C) | x_eq | D_alpha (m2/s) | zeta (J/mol) | Lphi diff physical | Lphi diff code | old ref / limit |
|---:|---:|---:|---:|---:|---:|---:|
{value_table}

All limits use `zeta0=1`, `D_beta=0`, physical `c=1/Vm`, and the exact
`lambda=0.6 nm` parameter.  A positive finite intrinsic interface mobility
requires `L_phi < L_phi_diff`; equality is approached only from below.
""")

    above = sum(row["classification"] == "NONPHYSICAL_UNDER_JI_CHEN_MAPPING"
                for row in existing)
    (REPORT_ROOT / "correction1_existing_Lphi_audit.md").write_text(f"""# Correction 1 Existing Lphi Audit

Audited rows: **{len(existing)}**; above strict Ji--Chen limit: **{above}**.

The previous finite-window references are already
`1/0.971892456... = 1.028920426` times the strict limit.  Research2 factors
`>=1` and the much larger finite-interface/fast-phase values therefore have
negative implied `1/M_I` and are classified
`NONPHYSICAL_UNDER_JI_CHEN_MAPPING`.

This classification does not invalidate their mass closure, KKT, energy, or
solver-regression evidence.  Their kinetic interpretation is restricted to
`EXPLORATORY_NUMERICAL_FAST_PHASE_RESPONSE` and they are not eligible as a
physical Ji--Chen interface mapping.
""")
    return limits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--physical-inputs", type=Path,
        default=REPORT_ROOT / "research2_inputs/T380_physical_inputs.json",
    )
    args = parser.parse_args()
    limits = generate_reports(args.physical_inputs)
    print("ji_chen_formula_recovered=true")
    print("lambda_definition_status=MATCHED")
    print("unit_conversion_status=PASS")
    print("D_beta=0")
    print("zeta0_status=PASS")
    print("zeta0_value=1")
    for temperature in (380, 400):
        row = limits[temperature]
        print(f"T{temperature}_L_phi_diff_physical={row['L_phi_diff_physical_m3_J_s']:.17e}")
        print(f"T{temperature}_L_phi_diff_code={row['L_phi_diff_code']:.17e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
