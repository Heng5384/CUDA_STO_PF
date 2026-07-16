#!/usr/bin/env python3
"""Derive fixed-Ctot dividing-surface mappings without fitting moving data."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.integrate import solve_bvp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402


def h(phi: np.ndarray | float) -> np.ndarray | float:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def hp(phi: np.ndarray | float) -> np.ndarray | float:
    return 30.0 * phi**2 * (1.0 - phi) ** 2


def gp(phi: np.ndarray | float) -> np.ndarray | float:
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def logistic(value: float) -> float:
    if value >= 0.0:
        inv = math.exp(-value)
        return 1.0 / (1.0 + inv)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def parse_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def radius_at_level(radius: np.ndarray, profile: np.ndarray,
                    level: float) -> float:
    below = np.flatnonzero(profile <= level)
    if below.size == 0:
        return math.nan
    hi = int(below[0])
    if hi == 0:
        return float(radius[0])
    lo = hi - 1
    fraction = (level - profile[lo]) / (profile[hi] - profile[lo])
    return float(radius[lo] + fraction * (radius[hi] - radius[lo]))


def universal_planar_moments(lambda_nm: float) -> dict[str, float]:
    half_width = 0.5 * lambda_nm

    def excess(s: float) -> float:
        phi = 0.5 * (1.0 - math.tanh(s / half_width))
        sharp = 1.0 if s < 0.0 else 0.0
        return float(h(phi)) - sharp

    return {
        f"I{power}_nm{power + 1}": quad(
            lambda s, p=power: s**p * excess(s),
            -20.0 * lambda_nm, 20.0 * lambda_nm,
            epsabs=1.0e-14, epsrel=1.0e-13, limit=400,
            points=[0.0],
        )[0]
        for power in range(4)
    }


def solve_radial_profile(
    dimension: int,
    target_radius_nm: float,
    lambda_nm: float,
    far_radius_nm: float,
    params: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, float]:
    if dimension not in (2, 3):
        raise ValueError("radial profile dimension must be 2 or 3")
    temperature_k = float(params["temperature_C"]) + 273.15
    energy_scale = float(params["mu_reference_scale"])
    diffusion_phys = unit.D_Ag_in_PbTe_m2_per_s(temperature_k)
    dx_reference_nm = math.sqrt(
        diffusion_phys * float(params["t_real_unit"]) / float(params["D_alpha"])
    ) * 1.0e9
    radius_target = target_radius_nm / dx_reference_nm
    radius_far = far_radius_nm / dx_reference_nm
    lambda_code = lambda_nm / dx_reference_nm
    w = float(params["W"])
    kappa = float(params["kappa_phi"])
    half_width = math.sqrt(2.0 * kappa / w)
    points = max(1601, int(math.ceil(radius_far / (lambda_code / 48.0))) + 1)
    radius = np.linspace(0.0, radius_far, points)
    phi = 0.5 * (1.0 - np.tanh((radius - radius_target) / half_width))
    derivative = -(0.5 / half_width) / np.cosh(
        np.clip((radius - radius_target) / half_width, -350.0, 350.0)
    ) ** 2
    surface_factor = 2.0 * math.pi if dimension == 2 else 4.0 * math.pi
    metric = surface_factor * radius ** (dimension - 1)
    volume = np.concatenate(([0.0], cumulative_trapezoid(metric * h(phi), radius)))
    state = np.vstack([phi, derivative, volume])
    target_volume = surface_factor / dimension * radius_target**dimension
    x_eq = unit.xAg2Te_eq_from_T(temperature_k)
    x_guess = min(max(x_eq + 0.003, 1.0e-8), 1.0 - 1.0e-8)
    mu0_compound = unit.mu_Ag2Te(temperature_k, x_eq) / energy_scale

    def equations(r: np.ndarray, y: np.ndarray,
                  parameter: np.ndarray) -> np.ndarray:
        matrix_x = logistic(float(parameter[0]))
        delta_mu = mu0_compound - unit.mu_Ag2Te(
            temperature_k, matrix_x
        ) / energy_scale
        return np.vstack([
            y[1],
            (w * gp(y[0]) + delta_mu * hp(y[0])) / kappa,
            surface_factor * r ** (dimension - 1) * h(y[0]),
        ])

    def boundary(left: np.ndarray, right: np.ndarray,
                 parameter: np.ndarray) -> np.ndarray:
        del parameter
        return np.array([left[1], right[1], left[2], right[2] - target_volume])

    singular = np.zeros((3, 3), dtype=np.float64)
    singular[1, 1] = -(dimension - 1.0)
    solution = solve_bvp(
        equations, boundary, radius, state,
        p=np.array([math.log(x_guess / (1.0 - x_guess))]),
        S=singular, tol=1.0e-8, max_nodes=200000,
    )
    if not solution.success:
        raise RuntimeError(
            f"{dimension}D radial BVP failed at R/lambda="
            f"{target_radius_nm / lambda_nm:g}: {solution.message}"
        )
    dense_radius = np.linspace(0.0, radius_far, max(points, 12001))
    dense_phi = np.clip(solution.sol(dense_radius)[0], 0.0, 1.0)
    return dense_radius * dx_reference_nm, dense_phi, logistic(float(solution.p[0]))


def surface_row(
    geometry: str,
    dimension: int,
    ratio: float,
    lambda_nm: float,
    radius: np.ndarray,
    phi: np.ndarray,
    matrix_x: float,
    moments: dict[str, float],
) -> dict[str, object]:
    surface_factor = 2.0 * math.pi if dimension == 2 else 4.0 * math.pi
    metric = surface_factor * radius ** (dimension - 1)
    vh = float(np.trapezoid(metric * h(phi), radius))
    r_h = (dimension * vh / surface_factor) ** (1.0 / dimension)
    r_half = radius_at_level(radius, phi, 0.5)
    volume_half = surface_factor / dimension * r_half**dimension
    area_half = surface_factor * r_half ** (dimension - 1)
    gamma_c_half = (1.0 - matrix_x) * (vh - volume_half) / area_half
    gamma_beta_half = (vh - volume_half) / area_half
    i1 = moments["I1_nm2"]
    if dimension == 2:
        mapped_h = math.sqrt(r_half**2 + 2.0 * i1)
        velocity_jacobian = r_half / mapped_h
    else:
        mapped_h = (r_half**3 + 6.0 * i1 * r_half) ** (1.0 / 3.0)
        velocity_jacobian = (r_half**2 + 2.0 * i1) / mapped_h**2
    return {
        "record_type": "stationary_surface_mapping",
        "geometry": geometry,
        "dimension": dimension,
        "R_over_lambda": ratio,
        "lambda_nm": lambda_nm,
        "curvature_nm_inv": (dimension - 1.0) / r_h,
        "kappa_lambda": (dimension - 1.0) * lambda_nm / r_h,
        "matrix_xB": matrix_x,
        "phi_half_radius_nm": r_half,
        "h_volume_radius_nm": r_h,
        "total_C_equimolar_radius_nm": r_h,
        "beta_storage_equimolar_radius_nm": r_h,
        "control_volume_beta_radius_nm": r_h,
        "h_minus_phi_half_nm": r_h - r_half,
        "moment_mapped_h_radius_nm": mapped_h,
        "moment_mapping_residual_nm": mapped_h - r_h,
        "Gamma_C_at_phi_half_xB_nm": gamma_c_half,
        "Gamma_C_at_h_surface_xB_nm": 0.0,
        "Gamma_beta_at_phi_half_nm": gamma_beta_half,
        "Gamma_beta_at_h_surface_nm": 0.0,
        "dR_h_dR_phi": velocity_jacobian,
        "surface_status": "TOTAL_C_EQUIMOLAR_EQUALS_H_VOLUME",
    }


def fit_velocity(rows: list[dict[str, str]], key: str) -> float:
    time = np.array([float(row["time_code"]) for row in rows])
    radius = np.array([float(row[key]) for row in rows])
    return float(np.polyfit(time, radius, 1)[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-dir", type=Path,
        default=ROOT / "reports/pf_ctot_production_candidate",
    )
    parser.add_argument(
        "--params", type=Path,
        default=ROOT / "reports/pf_ctot_production_candidate/"
        "prompt8_inputs/tiny_base.params",
    )
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    params = parse_params(args.params.resolve())
    lambda_nm = float(params["lambda_sm_m"]) * 1.0e9
    moments = universal_planar_moments(lambda_nm)
    rows: list[dict[str, object]] = [{
        "record_type": "planar_moment",
        "geometry": "planar",
        "dimension": 1,
        "R_over_lambda": math.inf,
        "lambda_nm": lambda_nm,
        **moments,
        "phi_half_radius_nm": 0.0,
        "h_volume_radius_nm": 0.0,
        "total_C_equimolar_radius_nm": 0.0,
        "beta_storage_equimolar_radius_nm": 0.0,
        "Gamma_C_at_phi_half_xB_nm": 0.0,
        "Gamma_C_at_h_surface_xB_nm": 0.0,
        "surface_status": "ALL_PLANAR_SURFACES_COINCIDE",
    }]

    stationary_text = (report_dir / "next_stationary_equilibrium_results.md").read_text()
    matrix_x_by_ratio: dict[float, float] = {}
    for match in re.finditer(
        r"\|\s*(5|8|10|15|20)\s*\|[^|]*\|\s*([0-9.]+)\s*\|",
        stationary_text,
    ):
        matrix_x_by_ratio[float(match.group(1))] = float(match.group(2))
    ratios = [5.0, 8.0, 10.0, 15.0, 20.0]
    for dimension, geometry in ((2, "cylindrical"), (3, "spherical")):
        for ratio in ratios:
            target = ratio * lambda_nm
            far = max(8.0, target + 5.0 * lambda_nm)
            radius, phi, matrix_x = solve_radial_profile(
                dimension, target, lambda_nm, far, params
            )
            if dimension == 2 and ratio in matrix_x_by_ratio:
                matrix_x = matrix_x_by_ratio[ratio]
            rows.append(surface_row(
                geometry, dimension, ratio, lambda_nm,
                radius, phi, matrix_x, moments,
            ))

    timeseries = read_csv(report_dir / "next2_curved_radius_timeseries.csv")
    sharp_rows = read_csv(report_dir / "next2_curved_velocity_metrics.csv")
    sharp_by_case = {
        row["case"]: float(row["sharp_velocity_nm_per_code_time"])
        for row in sharp_rows if row["window"] == "full"
    }
    i1 = moments["I1_nm2"]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in timeseries:
        grouped.setdefault(row["case"], []).append(row)
    remap_rows: list[dict[str, object]] = []
    for case, case_rows in sorted(grouped.items()):
        case_rows.sort(key=lambda row: int(row["step"]))
        for row in case_rows:
            r_phi = float(row["phi_half_radius_nm"])
            row["stationary_moment_mapped_equimolar_radius_nm"] = str(
                math.sqrt(r_phi**2 + 2.0 * i1)
            )
        direct_h_velocity = fit_velocity(case_rows, "h_volume_radius_nm")
        mapped_velocity = fit_velocity(
            case_rows, "stationary_moment_mapped_equimolar_radius_nm"
        )
        phi_velocity = fit_velocity(case_rows, "phi_half_radius_nm")
        sharp_velocity = sharp_by_case[case]
        for surface, velocity in (
            ("phi_half_surface_of_tension", phi_velocity),
            ("stationary_moment_mapped_total_C_equimolar", mapped_velocity),
            ("direct_h_volume_total_C_equimolar", direct_h_velocity),
            ("beta_storage_equimolar", direct_h_velocity),
        ):
            remap_rows.append({
                "record_type": "moving_velocity_remap",
                "case": case,
                "geometry": "cylindrical",
                "dimension": 2,
                "R_over_lambda": float(case_rows[0]["R_over_lambda"]),
                "correction": int(case_rows[0]["correction"]),
                "dt_code": float(case_rows[0]["dt_code"]),
                "surface_name": surface,
                "PF_velocity_nm_per_code_time": velocity,
                "sharp_velocity_nm_per_code_time": sharp_velocity,
                "velocity_error_rel": abs(velocity - sharp_velocity)
                / max(abs(sharp_velocity), 1.0e-300),
                "velocity_direction_match": int(
                    velocity == 0.0 or sharp_velocity == 0.0
                    or math.copysign(1.0, velocity)
                    == math.copysign(1.0, sharp_velocity)
                ),
            })
    rows.extend(remap_rows)
    metrics_path = report_dir / "next3_dividing_surface_metrics.csv"
    write_csv(metrics_path, rows)

    fine_h = [row for row in remap_rows
              if row["surface_name"] == "direct_h_volume_total_C_equimolar"
              and str(row["case"]).endswith("dt2")]
    fine_h.sort(key=lambda row: (int(row["correction"]), float(row["R_over_lambda"])))
    off_errors = [float(row["velocity_error_rel"])
                  for row in fine_h if int(row["correction"]) == 0]
    remap_monotone = all(
        off_errors[i + 1] <= off_errors[i] for i in range(len(off_errors) - 1)
    )
    max_cyl_offset = max(
        abs(float(row["h_minus_phi_half_nm"])) for row in rows
        if row.get("record_type") == "stationary_surface_mapping"
        and row.get("geometry") == "cylindrical"
    )
    max_sph_offset = max(
        abs(float(row["h_minus_phi_half_nm"])) for row in rows
        if row.get("record_type") == "stationary_surface_mapping"
        and row.get("geometry") == "spherical"
    )
    derivation = f"""# Next3 Dividing-Surface Derivation

## Fixed-Ctot identity

The audited model is

```text
C_B_tot = h(phi) v_B + (1-h(phi)) x_B_alpha,
q_alpha = (1-h(phi)) x_B_alpha,
v_B = 1.
```

For every stationary profile used here, `x_B_alpha=x_m` is spatially
constant. Relative to a sharp beta/matrix profile at radius `R_s`,

```text
Gamma_C(R_s) A_s = (v_B-x_m) [integral h dV - V_beta(R_s)].
```

Consequently the total-C equimolar, beta-storage equimolar,
control-volume-beta, and h-volume surfaces are exactly the same surface.
No fitted velocity data enter this statement.

## Surface of tension versus equimolar surface

The symmetric planar phase profile makes the interfacial-energy first moment
zero at `phi=0.5`; it is therefore the planar surface-of-tension convention.
The quintic interpolation gives the planar excess moments

```text
I0 = {moments['I0_nm1']:.17e} nm
I1 = {moments['I1_nm2']:.17e} nm^2
I2 = {moments['I2_nm3']:.17e} nm^3
I3 = {moments['I3_nm4']:.17e} nm^4.
```

With `s=r-R_phi` and `delta h=h-H(-s)`, metric expansion gives

```text
cylinder: R_e^2 = R_phi^2 + 2 I1 + O(lambda^3/R),
sphere:   R_e^3 = R_phi^3 + 6 R_phi I1 + O(lambda^4/R).
```

Thus `R_e-R_phi=O(lambda^2/R)`, not `O(lambda)`. The maximum independently
computed offsets in the requested range are `{max_cyl_offset:.9e} nm`
(cylinder) and `{max_sph_offset:.9e} nm` (sphere).

## Authoritative convention

The conserved Stefan jump is evaluated on the **total-C equimolar surface**,
which is exactly the h-volume surface for the accepted storage definition.
The Gibbs-Thomson surface of tension is represented by `phi=0.5` in the
planar limit and must be transformed to the equimolar convention at curved
order. Existing moving benchmarks already measured the direct h-volume
radius, so this choice does not introduce a fitted or post-hoc velocity
correction.
"""
    (report_dir / "next3_dividing_surface_derivation.md").write_text(derivation)

    decision = f"""# Next3 Velocity Remap Decision

The 20 completed moving cases were re-expressed using four observables:
`phi=0.5`, stationary-moment-mapped equimolar, direct h-volume equimolar, and
beta-storage equimolar. The latter two are algebraically identical.

The fine-step correction-OFF direct-equimolar relative errors are
`{', '.join(f'{value:.9g}' for value in off_errors)}` for
`R/lambda=5,8,10,15,20`. Monotone convergence is
`{'PASS' if remap_monotone else 'FAIL'}`. Because the original Next2 analysis
already used direct h-volume velocity, choosing the physically authoritative
surface leaves the rejected curved-kinetics conclusion unchanged.

`authoritative_dividing_surface=total_C_equimolar_equals_h_volume`

`dividing_surface_mapping_status=PASS_DERIVED_NO_MOVING_ERROR_FIT`

`velocity_remap_status={'PASS_DIVIDING_SURFACE_RECONCILIATION' if remap_monotone else 'FAIL_REMAP_ALONE_DOES_NOT_RECONCILE'}`

`next_stage={'STOP_NO_NEW_CORRECTION' if remap_monotone else 'ENTER_MATCHED_ASYMPTOTIC_DERIVATION'}`
"""
    (report_dir / "next3_velocity_remap_decision.md").write_text(decision)
    print(f"metrics={metrics_path}")
    print("authoritative_dividing_surface=total_C_equimolar_equals_h_volume")
    print(f"velocity_remap_monotone={str(remap_monotone).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
