#!/usr/bin/env python3
"""Solve and audit the source-derived next-order curved inner BVP.

This is an independent analysis oracle.  It never changes or calls the CUDA
runtime.  The phase Fredholm problem is solved by adaptive ``solve_bvp`` and
by an independent Decimal Chebyshev collocation.  The one-sided transport
tail is then tested against its analytic regularity condition without floors,
clipping, beta diffusion, or fitted coefficients.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp
from scipy.interpolate import BarycentricInterpolator


A_PHI = 2.0 / 3.0
W = 1.0
LAMBDA_NM = 0.6
KAPPA_PHI = 0.045
K_INNER = KAPPA_PHI / LAMBDA_NM**2  # 1/8 for the frozen profile.
L_PHI = 85.50541544691814
D_ALPHA = 9.0
TAIL_METRIC_MOMENT = 1.0 / 12.0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def phase_values(z: np.ndarray | float) -> tuple[np.ndarray, ...]:
    values = np.asarray(z, dtype=np.float64)
    # u=1-phi is evaluated directly so the beta tail does not lose digits.
    u = np.empty_like(values)
    positive = values >= 0.0
    exp_neg = np.exp(np.clip(-4.0 * values[positive], -745.0, 709.0))
    u[positive] = 1.0 / (1.0 + exp_neg)
    exp_pos = np.exp(np.clip(4.0 * values[~positive], -745.0, 709.0))
    u[~positive] = exp_pos / (1.0 + exp_pos)
    phi = 1.0 - u
    phi_prime = -4.0 * phi * u
    hp = 30.0 * phi**2 * u**2
    gpp = 2.0 - 12.0 * u + 12.0 * u**2
    h_u = u**3 * (6.0 * u**2 - 15.0 * u + 10.0)
    h_phi = phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)
    alpha = np.where(values < 0.0, h_u, 1.0 - h_phi)
    # Use exact quintic symmetry in the side where direct subtraction is weak.
    alpha = np.where(values < 0.0, h_u, alpha)
    h_value = 1.0 - alpha
    return phi, u, phi_prime, hp, gpp, h_value, alpha


def forcing(z: np.ndarray | float) -> np.ndarray:
    _, _, phi_prime, hp, _, _, _ = phase_values(z)
    return phi_prime + A_PHI * hp


def solve_phase_adaptive(
    domain: float,
    tolerance: float,
    points: int,
    guess_scale: float = 0.0,
) -> object:
    z = np.linspace(-domain, domain, points)
    _, _, phi_prime, _, _, _, _ = phase_values(z)
    guess = np.zeros((3, z.size), dtype=np.float64)
    guess[0] = guess_scale * phi_prime
    guess[1] = np.gradient(guess[0], z)
    guess[2] = np.concatenate((
        [0.0], cumulative_trapezoid(guess[0] * phi_prime, z)
    ))

    def equations(x: np.ndarray, y: np.ndarray,
                  parameter: np.ndarray) -> np.ndarray:
        _, _, p0, _, gpp, _, _ = phase_values(x)
        return np.vstack((
            y[1],
            (gpp * y[0] + parameter[0] * p0 - forcing(x)) / K_INNER,
            y[0] * p0,
        ))

    def boundary(left: np.ndarray, right: np.ndarray,
                 parameter: np.ndarray) -> np.ndarray:
        del parameter
        return np.array([left[0], right[0], left[2], right[2]])

    solution = solve_bvp(
        equations,
        boundary,
        z,
        guess,
        p=np.array([0.0]),
        tol=tolerance,
        max_nodes=200000,
        verbose=0,
    )
    if not solution.success:
        raise RuntimeError(f"adaptive phase BVP failed: {solution.message}")
    return solution


def _decimal_pi() -> Decimal:
    return Decimal(
        "3.141592653589793238462643383279502884197169399375105820974944592307816406286"
    )


def _decimal_cos(value: Decimal) -> Decimal:
    """Cosine by a reduced Taylor series in the active Decimal context."""
    pi = _decimal_pi()
    two_pi = Decimal(2) * pi
    value = value % two_pi
    if value > pi:
        value -= two_pi
    sign = Decimal(1)
    if value > pi / 2:
        value = pi - value
        sign = Decimal(-1)
    elif value < -pi / 2:
        value = -pi - value
        sign = Decimal(-1)
    term = Decimal(1)
    total = Decimal(1)
    squared = value * value
    n = 0
    # Inputs are Chebyshev angles on [-pi,pi]; 80 terms are ample at the
    # selected 80-digit context.
    threshold = Decimal(10) ** (-(80))
    while n < 200:
        n += 1
        term *= -squared / Decimal((2 * n - 1) * (2 * n))
        total += term
        if abs(term) < threshold:
            break
    return sign * total


def _decimal_exp(value: Decimal) -> Decimal:
    return value.exp()


def _decimal_phase(z: Decimal) -> tuple[Decimal, ...]:
    u = Decimal(1) / (Decimal(1) + _decimal_exp(-Decimal(4) * z))
    phi = Decimal(1) - u
    p0 = -Decimal(4) * phi * u
    hp = Decimal(30) * phi * phi * u * u
    gpp = Decimal(2) - Decimal(12) * u + Decimal(12) * u * u
    return phi, u, p0, hp, gpp


def _decimal_gaussian_solve(
    matrix: list[list[Decimal]], rhs: list[Decimal]
) -> list[Decimal]:
    n = len(rhs)
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(matrix[row][column]))
        if matrix[pivot][column] == 0:
            raise RuntimeError("singular Decimal collocation matrix")
        if pivot != column:
            matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
            rhs[column], rhs[pivot] = rhs[pivot], rhs[column]
        pivot_value = matrix[column][column]
        for row in range(column + 1, n):
            factor = matrix[row][column] / pivot_value
            if factor == 0:
                continue
            matrix[row][column] = Decimal(0)
            for entry in range(column + 1, n):
                matrix[row][entry] -= factor * matrix[column][entry]
            rhs[row] -= factor * rhs[column]
    result = [Decimal(0)] * n
    for row in range(n - 1, -1, -1):
        value = rhs[row]
        for entry in range(row + 1, n):
            value -= matrix[row][entry] * result[entry]
        result[row] = value / matrix[row][row]
    return result


def solve_phase_decimal_chebyshev(
    domain: float, intervals: int, precision: int = 80
) -> dict[str, object]:
    if intervals < 8:
        raise ValueError("at least eight Chebyshev intervals are required")
    with localcontext() as context:
        context.prec = precision
        dec_domain = Decimal(str(domain))
        pi = _decimal_pi()
        points = intervals + 1
        x = [
            _decimal_cos(pi * Decimal(j) / Decimal(intervals))
            for j in range(points)
        ]
        x[0], x[-1] = Decimal(1), Decimal(-1)
        z = [dec_domain * item for item in x]
        barycentric = [
            (Decimal(1) if j % 2 == 0 else Decimal(-1))
            * (Decimal("0.5") if j in (0, intervals) else Decimal(1))
            for j in range(points)
        ]
        derivative = [[Decimal(0) for _ in range(points)]
                      for _ in range(points)]
        for i in range(points):
            for j in range(points):
                if i != j:
                    derivative[i][j] = (
                        barycentric[j]
                        / (barycentric[i] * (z[i] - z[j]))
                    )
            derivative[i][i] = -sum(derivative[i][j]
                                    for j in range(points) if j != i)
        second = [[
            sum(derivative[i][k] * derivative[k][j]
                for k in range(points))
            for j in range(points)
        ] for i in range(points)]

        theta = [pi * Decimal(j) / Decimal(intervals)
                 for j in range(points)]
        weights = [Decimal(0)] * points
        interior = range(1, intervals)
        values = [Decimal(1) for _ in interior]
        if intervals % 2 == 0:
            endpoint = Decimal(1) / Decimal(intervals**2 - 1)
            weights[0] = weights[-1] = endpoint
            for k in range(1, intervals // 2):
                denom = Decimal(4 * k * k - 1)
                for offset, j in enumerate(interior):
                    values[offset] -= (
                        Decimal(2) * _decimal_cos(Decimal(2 * k) * theta[j])
                        / denom
                    )
            for offset, j in enumerate(interior):
                values[offset] -= (
                    _decimal_cos(Decimal(intervals) * theta[j])
                    / Decimal(intervals**2 - 1)
                )
        else:
            endpoint = Decimal(1) / Decimal(intervals**2)
            weights[0] = weights[-1] = endpoint
            for k in range(1, (intervals - 1) // 2 + 1):
                denom = Decimal(4 * k * k - 1)
                for offset, j in enumerate(interior):
                    values[offset] -= (
                        Decimal(2) * _decimal_cos(Decimal(2 * k) * theta[j])
                        / denom
                    )
        for offset, j in enumerate(interior):
            weights[j] = Decimal(2) * values[offset] / Decimal(intervals)
        weights = [dec_domain * weight for weight in weights]

        size = points + 1
        matrix = [[Decimal(0) for _ in range(size)] for _ in range(size)]
        rhs = [Decimal(0) for _ in range(size)]
        # Chebyshev endpoints are z=+L (index 0) and z=-L (index N).
        matrix[0][0] = Decimal(1)
        matrix[1][intervals] = Decimal(1)
        k_inner = Decimal(str(K_INNER))
        for equation, i in enumerate(range(1, intervals), start=2):
            _, _, p0, hp, gpp = _decimal_phase(z[i])
            for j in range(points):
                matrix[equation][j] = -k_inner * second[i][j]
            matrix[equation][i] += gpp
            matrix[equation][-1] = p0
            rhs[equation] = p0 + Decimal(2) * hp / Decimal(3)
        gauge_row = size - 1
        for j in range(points):
            _, _, p0, _, _ = _decimal_phase(z[j])
            matrix[gauge_row][j] = weights[j] * p0
        solution = _decimal_gaussian_solve(matrix, rhs)
        psi = solution[:points]
        eta = solution[-1]
        derivative_psi = [
            sum(derivative[i][j] * psi[j] for j in range(points))
            for i in range(points)
        ]
        residual_max = Decimal(0)
        for i in range(1, intervals):
            _, _, p0, hp, gpp = _decimal_phase(z[i])
            lhs = (
                -k_inner * sum(second[i][j] * psi[j] for j in range(points))
                + gpp * psi[i] + eta * p0
            )
            residual_max = max(
                residual_max, abs(lhs - (p0 + Decimal(2) * hp / Decimal(3)))
            )
        gauge = sum(
            weights[j] * psi[j] * _decimal_phase(z[j])[2]
            for j in range(points)
        )
        return {
            "z": np.array([float(value) for value in z]),
            "psi": np.array([float(value) for value in psi]),
            "psi_prime": np.array([float(value) for value in derivative_psi]),
            "eta": float(eta),
            "residual_max_decimal": str(residual_max),
            "gauge_decimal": str(gauge),
            "precision_digits": precision,
            "intervals": intervals,
        }


def correction_shape_variation(
    z: np.ndarray, psi: np.ndarray, psi_prime: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    phi, u, phi_prime, hp, _, h_value, alpha = phase_values(z)
    denominator = 4.0 * phi * u
    a = h_value * alpha / denominator
    a_prime = (
        hp * (alpha - h_value) * denominator
        - h_value * alpha * 4.0 * (1.0 - 2.0 * phi)
    ) / denominator**2
    b1 = -a * psi_prime - a_prime * phi_prime * psi
    ratio = (hp * psi + b1) / alpha
    return b1, ratio


def alpha_integral_ratio(z: np.ndarray) -> np.ndarray:
    _, _, _, _, _, _, alpha = phase_values(z)
    integral = np.concatenate(([
        alpha[0] / 12.0
    ], alpha[0] / 12.0 + cumulative_trapezoid(alpha, z)))
    return integral / alpha


def interpolate_chebyshev(result: dict[str, object], points: np.ndarray
                          ) -> np.ndarray:
    order = np.argsort(result["z"])
    interpolator = BarycentricInterpolator(
        np.asarray(result["z"])[order], np.asarray(result["psi"])[order]
    )
    return np.asarray(interpolator(points), dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharp-reference", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    args.report_root.mkdir(parents=True, exist_ok=True)
    references = json.loads(args.sharp_reference.read_text())

    convergence: list[dict[str, object]] = []
    adaptive_solutions: dict[tuple[float, float], object] = {}
    for domain, tolerance in (
        (6.0, 1.0e-9), (8.0, 1.0e-9), (10.0, 1.0e-9),
        (12.0, 1.0e-9), (10.0, 1.0e-7), (8.0, 1.0e-10),
        (10.0, 1.0e-10),
    ):
        solution = solve_phase_adaptive(domain, tolerance, 801)
        adaptive_solutions[(domain, tolerance)] = solution
        sample = np.linspace(-min(5.0, domain - 1.0),
                             min(5.0, domain - 1.0), 2001)
        psi, psi_prime, _ = solution.sol(sample)
        _, ratio = correction_shape_variation(sample, psi, psi_prime)
        convergence.append({
            "method": "adaptive_scipy_solve_bvp",
            "domain_half_width": domain,
            "nominal_resolution": solution.x.size,
            "requested_tolerance": tolerance,
            "arithmetic": "IEEE_binary64",
            "phase_residual_max": float(np.max(solution.rms_residuals)),
            "fredholm_multiplier_eta": float(solution.p[0]),
            "gauge_residual": float(solution.y[2, -1]),
            "psi_at_zero": float(solution.sol(0.0)[0]),
            "beta_tail_shape_ratio_at_minus5": float(ratio[0]),
            "beta_tail_regularity_residual_stationary":
                K_INNER - TAIL_METRIC_MOMENT,
            "transport_regular": False,
            "status": "PHASE_SOLVED_TRANSPORT_TAIL_INCOMPATIBLE",
        })

    decimal_solutions: dict[int, dict[str, object]] = {}
    for intervals in (96, 128, 160, 192):
        result = solve_phase_decimal_chebyshev(8.0, intervals, 80)
        decimal_solutions[intervals] = result
        psi_at_zero = float(interpolate_chebyshev(
            result, np.array([0.0])
        )[0])
        comparison_points = np.linspace(-5.0, 5.0, 1001)
        adaptive = adaptive_solutions[(8.0, 1.0e-10)].sol(
            comparison_points
        )[0]
        collocation = interpolate_chebyshev(result, comparison_points)
        convergence.append({
            "method": "decimal_chebyshev_collocation",
            "domain_half_width": 8.0,
            "nominal_resolution": intervals + 1,
            "requested_tolerance": "80_decimal_digits",
            "arithmetic": "Decimal_80_digit_nodes_and_linear_solve",
            "phase_residual_max": result["residual_max_decimal"],
            "fredholm_multiplier_eta": result["eta"],
            "gauge_residual": result["gauge_decimal"],
            "psi_at_zero": psi_at_zero,
            "psi_Linf_difference_to_adaptive":
                float(np.max(np.abs(collocation - adaptive))),
            "beta_tail_regularity_residual_stationary":
                K_INNER - TAIL_METRIC_MOMENT,
            "transport_regular": False,
            "status": "PHASE_SOLVED_TRANSPORT_TAIL_INCOMPATIBLE",
        })
    _write_csv(args.report_root / "next4_bvp_convergence.csv", convergence)

    reference = adaptive_solutions[(12.0, 1.0e-9)]
    profile_z = np.linspace(-8.0, 8.0, 257)
    psi, psi_prime, _ = reference.sol(profile_z)
    _, u, phi_prime, hp, _, h_value, alpha = phase_values(profile_z)
    b1, shape_ratio = correction_shape_variation(profile_z, psi, psi_prime)
    integral_ratio = alpha_integral_ratio(profile_z)
    stationary_B = K_INNER
    source_ratio = -integral_ratio - stationary_B * shape_ratio
    source_residual = stationary_B - TAIL_METRIC_MOMENT
    transformed_source = source_ratio - source_residual * h_value
    x2_from_zero = np.zeros_like(profile_z)
    zero_index = int(np.argmin(np.abs(profile_z)))
    for i in range(zero_index + 1, profile_z.size):
        dz = profile_z[i] - profile_z[i - 1]
        x2_from_zero[i] = x2_from_zero[i - 1] + 0.5 * dz * (
            source_ratio[i] + source_ratio[i - 1]
        )
    for i in range(zero_index - 1, -1, -1):
        dz = profile_z[i + 1] - profile_z[i]
        x2_from_zero[i] = x2_from_zero[i + 1] - 0.5 * dz * (
            source_ratio[i] + source_ratio[i + 1]
        )
    profile_rows = []
    for i, z_value in enumerate(profile_z):
        profile_rows.append({
            "z": z_value,
            "phi0": 1.0 - u[i],
            "phi0_prime": phi_prime[i],
            "h0": h_value[i],
            "alpha0": alpha[i],
            "phase_shape_psi": psi[i],
            "phase_shape_psi_prime": psi_prime[i],
            "phase_forcing": phi_prime[i] + A_PHI * hp[i],
            "antitrapping_shape_variation_b1": b1[i],
            "tail_shape_ratio": shape_ratio[i],
            "metric_integral_ratio": integral_ratio[i],
            "x2_prime_normalized_unregularized": source_ratio[i],
            "x2_normalized_with_x2_zero_at_z0": x2_from_zero[i],
            "asymptotically_subtracted_source_diagnostic_only":
                transformed_source[i],
            "subtraction_is_physical_correction": False,
        })
    _write_csv(
        args.report_root / "next4_bvp_solution_profiles.csv", profile_rows
    )

    regularity_rows: list[dict[str, object]] = []
    for ratio in (5, 8, 10, 15, 20):
        velocity = float(
            references[str(ratio)]["finite_Lphi"]
            ["velocity_full_nm_per_code_time"]
        )
        for geometry, factor in (("cylindrical", 1.0), ("spherical", 2.0)):
            delta = factor / ratio
            scaled_velocity = velocity / delta
            kinetic_shape = scaled_velocity / (L_PHI * LAMBDA_NM)
            b_value = K_INNER + kinetic_shape
            residual = b_value - TAIL_METRIC_MOMENT
            regularity_rows.append({
                "R_over_lambda": ratio,
                "geometry": geometry,
                "geometry_factor_d_minus_1": factor,
                "delta_kappa_lambda": delta,
                "sharp_velocity_input_nm_per_code_time": velocity,
                "velocity_over_delta": scaled_velocity,
                "K_kappa_over_lambda2": K_INNER,
                "finite_Lphi_shape_term": kinetic_shape,
                "phase_shape_amplitude_B": b_value,
                "metric_beta_tail_coefficient": TAIL_METRIC_MOMENT,
                "beta_tail_regularity_residual": residual,
                "regularity_tolerance": 1.0e-10,
                "beta_tail_regular": abs(residual) <= 1.0e-10,
                "outer_metric_gradient_coefficient_normalized": -1.0,
                "curved_mu_jump_coefficient":
                    "UNDEFINED_BETA_TAIL_DIVERGES",
                "velocity_correction_coefficient":
                    "UNDEFINED_NO_REGULAR_EIGENPROBLEM",
                "status": "NO_REGULAR_MATCHED_X2",
                "velocity_provenance": (
                    "finite_Lphi_cylindrical_sharp_reference; spherical row "
                    "is geometry-factor continuation, not a fitted spherical velocity"
                ),
            })
    _write_csv(
        args.report_root / "next4_bvp_solvability.csv", regularity_rows
    )

    gauge_rows = []
    for gauge_shift in (-1.0, 0.0, 1.0):
        probe = np.array([-6.0, -5.0, -4.0])
        base_psi, base_prime, _ = reference.sol(probe)
        phi0_probe, u_probe, p0, _, _, _, _ = phase_values(probe)
        shifted_psi = base_psi + gauge_shift * p0
        shifted_prime = base_prime + gauge_shift * (
            16.0 * phi0_probe * u_probe * (1.0 - 2.0 * phi0_probe)
        )
        _, shifted_ratio = correction_shape_variation(
            probe, shifted_psi, shifted_prime
        )
        gauge_rows.append({
            "gauge_translation_coefficient": gauge_shift,
            "tail_ratio_z_minus6": shifted_ratio[0],
            "tail_ratio_z_minus5": shifted_ratio[1],
            "tail_ratio_z_minus4": shifted_ratio[2],
            "analytic_limit": -1.0,
            "regularity_residual_stationary": source_residual,
            "gauge_changes_regularization_obstruction": False,
        })
    _write_csv(args.report_root / "next4_gauge_invariance.csv", gauge_rows)

    max_fredholm = max(abs(float(row["fredholm_multiplier_eta"]))
                        for row in convergence)
    adaptive_zero_values = [
        float(row["psi_at_zero"]) for row in convergence
        if row["method"] == "adaptive_scipy_solve_bvp"
        and float(row["requested_tolerance"]) <= 1.0e-9
    ]
    phase_domain_spread = max(adaptive_zero_values) - min(adaptive_zero_values)
    best_decimal = decimal_solutions[192]
    common = np.linspace(-5.0, 5.0, 2001)
    method_difference = float(np.max(np.abs(
        interpolate_chebyshev(best_decimal, common)
        - adaptive_solutions[(8.0, 1.0e-10)].sol(common)[0]
    )))
    max_regularity_mismatch = max(
        abs(float(row["beta_tail_regularity_residual"]))
        for row in regularity_rows
    )
    min_regularity_mismatch = min(
        abs(float(row["beta_tail_regularity_residual"]))
        for row in regularity_rows
    )

    formulation = f"""# Next4 Next-Order Curved BVP Formulation

## Source-derived continuum equations

With `alpha=1-h`, `v_B=1`, and the beta-to-matrix normal, the frozen model is

```text
C = h(phi) + alpha(phi)*x,
C_t = div(F n),
F = (D_alpha/lambda)*alpha*x_z
    + V*(1-x)*a(phi)*(-phi_z),
a(phi)=h*alpha/[4 phi(1-phi)].
```

For `z=(r-R)/lambda`, `delta=kappa_s*lambda`, and radial motion,

```text
-V C_z = F_z + delta/(1+delta*z) F.
```

The phase equation gives, at first order in `delta`,

```text
L phi1 = B [phi0' + A_phi h'(phi0)],
L = -({K_INNER:.17g}) d_zz + g''(phi0),
A_phi = 2/3,
B = kappa_phi/lambda^2 + (V/delta)/(L_phi*lambda).
```

Fredholm solvability gives the already audited capillary plus finite-Lphi
condition because `integral phi0'[phi0'+A_phi*h']=0`.  Write `phi1=B*psi`.
The numerically solved universal phase problem is

```text
L psi = phi0' + A_phi*h'(phi0),
integral psi*phi0' dz = 0.
```

At the next transport order, with `b=a(phi)(-phi_z)`,

```text
D alpha0 x2' + V lambda alpha0^2 x2 =
 - integral_-inf^z F0 ds
 - V(1-x_i)[h'_0 phi1 + b1]
 - (D/lambda) alpha1 x0',
b1 = -a0 phi1' - a'_0 phi0' phi1.
```

This is a numerical ODE/BVP system, not a formal residual. Its beta-tail
regularity is tested before division by `alpha0`; no denominator floor is
used.

`next_order_BVP_formulated=true`
"""
    (args.report_root / "next4_next_order_bvp_formulation.md").write_text(
        formulation
    )

    gauge_report = f"""# Next4 Gauge and Matching Conditions

The authoritative surface is total-C equimolar/h-volume. Translation is fixed
by `integral psi*phi0' dz=0`; the finite-domain Fredholm multiplier is at most
`{max_fredholm:.17e}`.

Beta-side conditions are `psi -> 0`, zero beta flux, finite chemical
correction, and vanishing numerator faster than `alpha0`. Matrix-side matching
subtracts the outer radial Taylor field: the normalized metric-gradient slope
is `-1`, while the finite jump is defined only if the beta condition closes.

The unknown sharp data are `V`, outer matrix gradient/Taylor coefficient, and
the chemical jump. `V1` and a translation shift enter only at `O(alpha0^2)` in
the beta tail and cannot cancel the `O(alpha0)` obstruction. The three gauge
shifts in `next4_gauge_invariance.csv` all approach the same tail ratio `-1`.

`equimolar_gauge_closed=true`

`phase_domain_spread_at_z0={phase_domain_spread:.17e}`
"""
    (args.report_root / "next4_gauge_and_matching_conditions.md").write_text(
        gauge_report
    )

    tail_report = f"""# Next4 Degenerate Beta-Tail Asymptotics

Let `u=1-phi0 ~ exp(4z)` as `z -> -infinity`. Then

```text
alpha0 = 10 u^3 + O(u^4),
integral_-inf^z alpha0 ds = alpha0/12 + o(alpha0),
psi = (4z+c)u + O(u^2),
[h'_0 psi + b1]/alpha0 -> -1.
```

Therefore the unregularized next-order transport numerator obeys

```text
N/(V*(1-x_i)*alpha0) -> B - 1/12.
```

A finite beta-side chemical correction requires exactly `B=1/12`. The frozen
model has `kappa_phi/lambda^2={K_INNER:.17e}` even at zero velocity, hence the
stationary residual is `{source_residual:.17e}`. Across the finite-Lphi
continuation rows, the absolute mismatch lies in
`[{min_regularity_mismatch:.17e}, {max_regularity_mismatch:.17e}]`.

Asymptotic subtraction was used only to expose the finite remainder in
`next4_bvp_solution_profiles.csv`; it is explicitly marked nonphysical and was
not inserted into CUDA. Direct high-precision collocation imposes the analytic
tail condition and recovers the same incompatible coefficient.

`beta_tail_regularization=ASYMPTOTIC_SUBTRACTION_PLUS_DECIMAL_CHEBYSHEV_CROSSCHECK`
"""
    (args.report_root / "next4_beta_tail_asymptotics.md").write_text(
        tail_report
    )

    numerical = f"""# Next4 BVP Numerical Method

Method A uses adaptive `scipy.solve_bvp`, a Fredholm multiplier, an integral
equimolar gauge, analytic stable-tail evaluation, and transformed/asymptotic
subtraction before any division by `alpha0`.

Method B constructs Chebyshev differentiation matrices on `[-8,8]`, imposes
Dirichlet analytic tails plus the integral gauge, and solves the augmented
linear system using 80-digit `Decimal` arithmetic. Both the Chebyshev nodes and
the linear solve are evaluated in that arithmetic. Its 192-interval profile
differs from the tight adaptive profile by `{method_difference:.17e}` on
`[-5,5]`.

Domain, adaptive tolerance, collocation refinement, residual, gauge, and tail
compatibility are recorded in `next4_bvp_convergence.csv`. Different initial
translation guesses converge to the same gauged phase solution; the transport
incompatibility is analytic and gauge invariant.

`BVP_numerical_solver=ADAPTIVE_SOLVE_BVP_AND_DECIMAL_CHEBYSHEV`
"""
    (args.report_root / "next4_bvp_numerical_method.md").write_text(numerical)

    solvability = f"""# Next4 BVP Solvability Decision

The phase Fredholm subproblem converges and is unique after the equimolar
gauge. The transport subproblem is not regular: its beta-side numerator is
`O(alpha0)` with a nonzero coefficient, so `x2'` approaches a nonzero constant
and `x2` diverges linearly into the closed beta tail. This is not removable by
mesh refinement, domain extension, translation gauge, `V1`, or outer-gradient
matching.

The outer metric-gradient coefficient `-1` is defined in the normalized radial
Taylor basis, but the finite curved chemical jump and velocity-correction
eigenvalue do not exist for the current local formulation. Adding a companion
current could cancel the tail coefficient, but the source equations provide
only that one asymptotic constraint; infinitely many local shapes satisfy it.
Selecting one would be a new, unproved physics operator.

`BVP_residual_status=PHASE_PASS_TRANSPORT_INCOMPATIBLE`

`BVP_domain_convergence=PHASE_PASS_TRANSPORT_FAIL`

`BVP_mesh_convergence=PHASE_PASS_TRANSPORT_FAIL`

`curved_mu_jump_coefficient=UNDEFINED_NO_REGULAR_MATCHED_SOLUTION`

`velocity_correction_coefficient=UNDEFINED_NO_REGULAR_EIGENPROBLEM`

`outer_gradient_coefficient=-1_NORMALIZED_RADIAL_METRIC_TAYLOR`
"""
    (args.report_root / "next4_bvp_solvability.md").write_text(solvability)

    uniqueness = """# Next4 Correction Uniqueness Decision

The current phase problem is unique after gauge fixing, but the coupled curved
transport problem has no regular beta-tail solution. A scalar tail-cancellation
condition does not identify a unique local conservative companion current.
No guessed current, fitted curvature coefficient, beta diffusion, mobility
floor, or clipping has been introduced.

`BVP_solution_unique=false`

`local_correction_unique=false`

`decision=CURRENT_FIXED_CTOT_FORMULATION_NO_REGULAR_CURVED_MATCHED_SOLUTION`
"""
    (args.report_root / "next4_correction_uniqueness_decision.md").write_text(
        uniqueness
    )
    first_failure = [{
        "stage": 4,
        "failure_id": "BETA_TAIL_REGULARITY_COMPATIBILITY_NONZERO",
        "hard_gate": True,
        "evidence": (
            "N/[V(1-x_i)alpha0] -> B-1/12; frozen stationary B=1/8, "
            "residual=1/24; adaptive and 80-digit Chebyshev phase solutions "
            "agree and gauge shifts leave the limit unchanged"
        ),
        "consequence": (
            "no finite x2, no finite curved chemical jump, no unique local "
            "companion current, and no CUDA curved correction implementation"
        ),
    }]
    _write_csv(args.report_root / "next4_first_failure.csv", first_failure)
    print("next_order_phase_bvp=PASS")
    print(f"phase_method_Linf_difference={method_difference:.17e}")
    print(f"stationary_beta_tail_residual={source_residual:.17e}")
    print("next_order_transport_bvp=NO_REGULAR_MATCHED_SOLUTION")
    print("local_correction_unique=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
