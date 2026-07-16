#!/usr/bin/env python3
"""Construct a fixed-mass stationary cylindrical diffuse equilibrium.

The oracle solves the nonelastic equal-molar-volume Euler-Lagrange equation
in cylindrical radius, then evaluates the resulting profile with the exact
periodic production spectral Laplacian. Ordinary PF time evolution is not
used to construct the equilibrium.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.integrate import solve_bvp
from scipy.sparse.linalg import LinearOperator, gmres

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402


def parse_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def alpha_stable(phi: np.ndarray) -> np.ndarray:
    direct = 1.0 - h(phi)
    reflected = h(1.0 - phi)
    return np.where(phi > 0.5, reflected, direct)


def h_stable(phi: np.ndarray) -> np.ndarray:
    return np.where(phi > 0.5, 1.0 - h(1.0 - phi), h(phi))


def hp(phi: np.ndarray) -> np.ndarray:
    return 30.0 * phi**2 * (1.0 - phi) ** 2


def g(phi: np.ndarray) -> np.ndarray:
    return phi**2 * (1.0 - phi) ** 2


def gp(phi: np.ndarray) -> np.ndarray:
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def logistic(value: float) -> float:
    if value >= 0.0:
        inv = math.exp(-value)
        return 1.0 / (1.0 + inv)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def even_grid_at_least(domain_nm: float, dx_nm: float) -> int:
    count = int(math.ceil(domain_nm / dx_nm - 1.0e-12))
    return count if count % 2 == 0 else count + 1


def spectral_laplacian_2d(field: np.ndarray, dx_code: float) -> np.ndarray:
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("spectral oracle expects a square 2D field")
    n = field.shape[0]
    wave = 2.0 * math.pi * np.fft.fftfreq(n, d=dx_code)
    kx, kz = np.meshgrid(wave, wave, indexing="ij")
    spectrum = np.fft.fftn(field)
    return np.fft.ifftn(-(kx * kx + kz * kz) * spectrum).real


def radius_at_level(radius: np.ndarray, profile: np.ndarray,
                    level: float) -> float:
    below = np.flatnonzero(profile <= level)
    if below.size == 0:
        return math.nan
    hi = int(below[0])
    if hi == 0:
        return float(radius[0])
    lo = hi - 1
    p0, p1 = float(profile[lo]), float(profile[hi])
    if p0 == p1:
        return float(radius[hi])
    fraction = (level - p0) / (p1 - p0)
    return float(radius[lo] + fraction * (radius[hi] - radius[lo]))


def invert_h_scalar(target: float) -> float:
    if target <= 0.0:
        return 0.0
    if target >= 1.0:
        return 1.0
    if target > 0.5:
        return 1.0 - invert_h_scalar(1.0 - target)
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        value = mid**3 * (6.0 * mid**2 - 15.0 * mid + 10.0)
        if value < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def invert_h_lower_feasible(target: float) -> float:
    if target <= 0.0:
        return 0.0
    if target >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(h_stable(np.array(mid))) < target:
            lo = mid
        else:
            hi = mid
    return hi


def invert_h_upper_feasible(target: float) -> float:
    if target <= 0.0:
        return 0.0
    if target >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(h_stable(np.array(mid))) < target:
            lo = mid
        else:
            hi = mid
    return lo


def phase_bounds_scalar(ctot: float, v_b: float, x_min: float,
                        x_max: float) -> tuple[float, float]:
    alpha_lower = max(0.0, min(1.0, (v_b - ctot) / (v_b - x_min)))
    alpha_upper = max(0.0, min(1.0, (v_b - ctot) / (v_b - x_max)))
    h_lower = max(0.0, min(1.0, (ctot - x_max) / (v_b - x_max)))
    h_upper = max(0.0, min(1.0, (ctot - x_min) / (v_b - x_min)))
    lower = (
        invert_h_lower_feasible(h_lower) if h_lower <= 0.5
        else 1.0 - invert_h_upper_feasible(alpha_upper)
    )
    upper = (
        invert_h_upper_feasible(h_upper) if h_upper <= 0.5
        else 1.0 - invert_h_lower_feasible(alpha_lower)
    )
    for _ in range(8):
        if lower >= 1.0:
            break
        alpha = float(alpha_stable(np.array(lower)))
        x_value = ((ctot - v_b) + alpha * v_b) / alpha if alpha > 0.0 else math.nan
        if not math.isfinite(x_value) or x_value <= x_max:
            break
        lower = float(np.nextafter(lower, 1.0))
    for _ in range(8):
        if upper <= 0.0:
            break
        alpha = float(alpha_stable(np.array(upper)))
        x_value = ((ctot - v_b) + alpha * v_b) / alpha if alpha > 0.0 else math.nan
        if not math.isfinite(x_value) or x_value >= x_min:
            break
        upper = float(np.nextafter(upper, 0.0))
    return lower, upper


def phase_bounds(ctot: np.ndarray, v_b: float, x_min: float,
                 x_max: float) -> tuple[np.ndarray, np.ndarray]:
    alpha_lower = np.clip((v_b - ctot) / (v_b - x_min), 0.0, 1.0)
    alpha_upper = np.clip((v_b - ctot) / (v_b - x_max), 0.0, 1.0)
    h_lower = np.clip((ctot - x_max) / (v_b - x_max), 0.0, 1.0)
    h_upper = np.clip((ctot - x_min) / (v_b - x_min), 0.0, 1.0)

    def inverse_pair(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = np.zeros_like(target)
        hi = np.ones_like(target)
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            less = h_stable(mid) < target
            lo = np.where(less, mid, lo)
            hi = np.where(less, hi, mid)
        lower_feasible = np.where(
            target <= 0.0, 0.0, np.where(target >= 1.0, 1.0, hi)
        )
        upper_feasible = np.where(
            target <= 0.0, 0.0, np.where(target >= 1.0, 1.0, lo)
        )
        return lower_feasible, upper_feasible

    h_lower_lo, _ = inverse_pair(h_lower)
    _, h_upper_hi = inverse_pair(h_upper)
    _, alpha_upper_hi = inverse_pair(alpha_upper)
    alpha_lower_lo, _ = inverse_pair(alpha_lower)
    lower = np.where(h_lower <= 0.5, h_lower_lo, 1.0 - alpha_upper_hi)
    upper = np.where(h_upper <= 0.5, h_upper_hi, 1.0 - alpha_lower_lo)
    for _ in range(8):
        alpha = alpha_stable(lower)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_value = ((ctot - v_b) + alpha * v_b) / alpha
        move = (lower < 1.0) & np.isfinite(x_value) & (x_value > x_max)
        lower = np.where(move, np.nextafter(lower, 1.0), lower)
    for _ in range(8):
        alpha = alpha_stable(upper)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_value = ((ctot - v_b) + alpha * v_b) / alpha
        move = (upper > 0.0) & np.isfinite(x_value) & (x_value < x_min)
        upper = np.where(move, np.nextafter(upper, 0.0), upper)
    return lower, upper


def projected_kkt_linf(phi: np.ndarray, ctot: np.ndarray,
                       phase_residual: np.ndarray, dt: float, l_phi: float,
                       v_b: float, x_min: float, x_max: float) -> float:
    lower, upper = phase_bounds(ctot, v_b, x_min, x_max)
    rate_residual = l_phi * phase_residual
    projected = phi - np.clip(phi - dt * rate_residual, lower, upper)
    return float(np.max(np.abs(projected)))


def matrix_x_from_delta_mu(delta_mu: float, temperature_k: float,
                           energy_scale: float,
                           mu0_compound: float) -> float:
    x_eq = unit.xAg2Te_eq_from_T(temperature_k)
    lo, hi = x_eq, 0.15
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        value = mu0_compound - unit.mu_Ag2Te(
            temperature_k, mid
        ) / energy_scale
        if value < delta_mu:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def refine_production_spectral_equilibrium(
    phi: np.ndarray,
    delta_mu: float,
    matrix_x: float,
    target_h_cells: float,
    dx_code: float,
    w: float,
    kappa: float,
    free_tolerance: float,
    max_newton: int = 64,
) -> tuple[np.ndarray, float, np.ndarray, dict[str, object]]:
    """Solve the periodic constrained spectral obstacle problem.

    Cells whose matrix capacity is below the representable spacing at Ctot=1
    are exact pure-beta singleton cells. The remaining cells and the volume
    multiplier are solved by a matrix-free Newton/GMRES step using the same
    spectral Laplacian as production.
    """
    if phi.ndim != 2 or phi.shape[0] != phi.shape[1]:
        raise ValueError("spectral refinement requires a square 2D phi")
    n = phi.shape[0]
    wave = 2.0 * math.pi * np.fft.fftfreq(n, d=dx_code)
    kx, kz = np.meshgrid(wave, wave, indexing="ij")
    k2 = kx * kx + kz * kz

    def laplacian(field: np.ndarray) -> np.ndarray:
        return np.fft.ifftn(-k2 * np.fft.fftn(field)).real

    result = np.clip(phi.copy(), 0.0, 1.0)
    representation_support_eps = 64.0 * np.finfo(np.float64).eps
    core = alpha_stable(result) <= representation_support_eps
    result[core] = 1.0
    history: list[dict[str, object]] = []
    for iteration in range(max_newton):
        free = ~core
        indices = np.flatnonzero(free.ravel())
        local_residual = (
            w * gp(result) + delta_mu * hp(result) - kappa * laplacian(result)
        )
        volume_residual = float(np.sum(h_stable(result)) - target_h_cells)
        free_linf = float(np.max(np.abs(local_residual[free])))
        if free_linf <= free_tolerance and abs(volume_residual) <= 1.0e-8:
            break
        local_jacobian = (
            w * (2.0 - 12.0 * result + 12.0 * result**2)
            + delta_mu * 60.0 * result * (1.0 - result) *
              (1.0 - 2.0 * result)
        )
        h_prime = hp(result)
        constraint_scale = 1.0 / math.sqrt(max(target_h_cells, 1.0))
        rhs = np.empty(indices.size + 1, dtype=np.float64)
        rhs[:-1] = -local_residual.ravel()[indices]
        rhs[-1] = -constraint_scale * volume_residual

        def matvec(vector: np.ndarray) -> np.ndarray:
            delta_phi_flat = np.zeros(result.size, dtype=np.float64)
            delta_phi_flat[indices] = vector[:-1]
            delta_phi = delta_phi_flat.reshape(result.shape)
            field = (
                local_jacobian * delta_phi - kappa * laplacian(delta_phi)
                + h_prime * vector[-1]
            )
            output = np.empty_like(vector)
            output[:-1] = field.ravel()[indices]
            output[-1] = constraint_scale * np.sum(h_prime * delta_phi)
            return output

        operator = LinearOperator(
            (rhs.size, rhs.size), matvec=matvec, dtype=np.float64
        )
        precondition_shift = max(
            0.1, float(np.median(local_jacobian[free]))
        )
        precondition_denominator = precondition_shift + kappa * k2

        def approximate_inverse(vector: np.ndarray) -> np.ndarray:
            field_flat = np.zeros(result.size, dtype=np.float64)
            field_flat[indices] = vector[:-1]
            field = field_flat.reshape(result.shape)
            z_field = np.fft.ifftn(
                np.fft.fftn(field) / precondition_denominator
            ).real
            w_field = np.fft.ifftn(
                np.fft.fftn(h_prime) / precondition_denominator
            ).real
            z_field[core] = 0.0
            w_field[core] = 0.0
            unscaled_constraint_rhs = vector[-1] / constraint_scale
            schur = float(np.sum(h_prime * w_field))
            if not math.isfinite(schur) or abs(schur) < 1.0e-20:
                raise RuntimeError("spectral Newton preconditioner Schur failure")
            multiplier = (
                float(np.sum(h_prime * z_field)) - unscaled_constraint_rhs
            ) / schur
            solution = np.empty_like(vector)
            solution[:-1] = (z_field - multiplier * w_field).ravel()[indices]
            solution[-1] = multiplier
            return solution

        preconditioner = LinearOperator(
            (rhs.size, rhs.size), matvec=approximate_inverse,
            dtype=np.float64,
        )
        gmres_history: list[float] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gmres_kwargs = {
                "M": preconditioner,
                "atol": 1.0e-13,
                "restart": 80,
                "maxiter": 20,
                "callback": lambda value: gmres_history.append(float(value)),
            }
            try:
                step, info = gmres(
                    operator, rhs, rtol=1.0e-10,
                    callback_type="pr_norm", **gmres_kwargs,
                )
            except TypeError as error:
                if "unexpected keyword argument" not in str(error):
                    raise
                # SciPy < 1.12 exposes `tol` and has no callback_type.  This is
                # the identical relative convergence target, not a numerical
                # or physical contract change.
                step, info = gmres(
                    operator, rhs, tol=1.0e-10, **gmres_kwargs,
                )
        if info != 0 or not np.all(np.isfinite(step)):
            raise RuntimeError(f"spectral Newton GMRES failed with info={info}")
        delta_phi_flat = np.zeros(result.size, dtype=np.float64)
        delta_phi_flat[indices] = step[:-1]
        delta_phi = delta_phi_flat.reshape(result.shape)
        delta_multiplier = float(step[-1])
        initial_merit = max(
            free_linf, abs(volume_residual) / max(target_h_cells, 1.0)
        )
        line_factor = 1.0
        accepted = False
        for _ in range(30):
            trial = np.clip(result + line_factor * delta_phi, 0.0, 1.0)
            trial[core] = 1.0
            trial_multiplier = delta_mu + line_factor * delta_multiplier
            trial_residual = (
                w * gp(trial) + trial_multiplier * hp(trial)
                - kappa * laplacian(trial)
            )
            trial_volume = float(np.sum(h_stable(trial)) - target_h_cells)
            merit = max(
                float(np.max(np.abs(trial_residual[free]))),
                abs(trial_volume) / max(target_h_cells, 1.0),
            )
            if merit < initial_merit:
                accepted = True
                break
            line_factor *= 0.5
        if not accepted:
            raise RuntimeError("spectral Newton line search failed")
        result = trial
        delta_mu = trial_multiplier
        new_core = (
            alpha_stable(result) <= representation_support_eps
        )
        core |= new_core
        result[core] = 1.0
        history.append(
            {
                "newton_iteration": iteration,
                "free_residual_before": free_linf,
                "volume_residual_before": volume_residual,
                "gmres_iterations": len(gmres_history),
                "gmres_final_relative_residual":
                    (gmres_history[-1] if gmres_history else 0.0),
                "constraint_row_scale": constraint_scale,
                "preconditioner": "spectral_constant_shift_schur_v1",
                "precondition_shift": precondition_shift,
                "line_factor": line_factor,
            }
        )
    final_residual = (
        w * gp(result) + delta_mu * hp(result) - kappa * laplacian(result)
    )
    final_free = ~core
    final_free_linf = float(np.max(np.abs(final_residual[final_free])))
    final_volume = float(np.sum(h_stable(result)) - target_h_cells)
    if final_free_linf > free_tolerance or abs(final_volume) > 1.0e-8:
        raise RuntimeError(
            "spectral obstacle solve did not close: "
            f"free_Linf={final_free_linf:.17e}, volume={final_volume:.17e}, "
            f"iterations={len(history)}, core={int(np.count_nonzero(core))}"
        )
    diagnostics: dict[str, object] = {
        "newton_iterations": len(history),
        "newton_history": history,
        "representability_core_cells": int(np.count_nonzero(core)),
        "representation_support_eps": representation_support_eps,
        "free_residual_Linf": final_free_linf,
        "volume_residual_cell_units": final_volume,
    }
    return result, delta_mu, core, diagnostics


def solve_radial_equilibrium(
    target_radius_code: float,
    far_radius_code: float,
    temperature_k: float,
    energy_scale: float,
    w: float,
    kappa: float,
    mu0_compound: float,
    radial_spacing: float,
    tolerance: float,
) -> tuple[object, float, float]:
    points = max(401, int(math.ceil(far_radius_code / radial_spacing)) + 1)
    radius = np.linspace(0.0, far_radius_code, points)
    half_width = math.sqrt(2.0 * kappa / w)
    phi = 0.5 * (1.0 - np.tanh((radius - target_radius_code) / half_width))
    derivative = -(0.5 / half_width) / np.cosh(
        (radius - target_radius_code) / half_width
    ) ** 2
    integrand = 2.0 * math.pi * radius * h(phi)
    volume = np.zeros_like(radius)
    volume[1:] = np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(radius)
    )
    state = np.vstack([phi, derivative, volume])
    target_volume = math.pi * target_radius_code**2
    x_eq = unit.xAg2Te_eq_from_T(temperature_k)
    x_guess = min(max(x_eq + 0.003, 1.0e-8), 1.0 - 1.0e-8)
    logit_guess = math.log(x_guess / (1.0 - x_guess))

    def equations(r: np.ndarray, y: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        matrix_x = logistic(float(parameter[0]))
        delta_mu = mu0_compound - unit.mu_Ag2Te(
            temperature_k, matrix_x
        ) / energy_scale
        return np.vstack(
            [
                y[1],
                (w * gp(y[0]) + delta_mu * hp(y[0])) / kappa,
                2.0 * math.pi * r * h(y[0]),
            ]
        )

    def boundary(left: np.ndarray, right: np.ndarray,
                 parameter: np.ndarray) -> np.ndarray:
        del parameter
        return np.array(
            [left[1], right[1], left[2], right[2] - target_volume]
        )

    singular = np.zeros((3, 3), dtype=np.float64)
    singular[1, 1] = -1.0
    solution = solve_bvp(
        equations,
        boundary,
        radius,
        state,
        p=np.array([logit_guess]),
        S=singular,
        tol=tolerance,
        max_nodes=50000,
        verbose=0,
    )
    matrix_x = logistic(float(solution.p[0]))
    delta_mu = mu0_compound - unit.mu_Ag2Te(
        temperature_k, matrix_x
    ) / energy_scale
    return solution, matrix_x, delta_mu


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_runtime_params(base: Path, output: Path, values: dict[str, object]) -> None:
    text = base.read_text()
    if not text.endswith("\n"):
        text += "\n"
    text += "\n# Stationary curved fixed-Ctot hold overlay\n"
    for key, value in values.items():
        if isinstance(value, str):
            text += f"{key}={value}\n"
        elif isinstance(value, int):
            text += f"{key}={value}\n"
        else:
            text += f"{key}={float(value):.17e}\n"
    output.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--R-over-lambda", type=float, required=True)
    parser.add_argument("--lambda-over-dx", type=float, default=12.0)
    parser.add_argument("--domain-nm", type=float)
    parser.add_argument("--dt", type=float, default=6.25e-6)
    parser.add_argument("--hold-steps", type=int, default=1000)
    parser.add_argument("--bvp-tol", type=float, default=1.0e-9)
    parser.add_argument("--spectral-tol", type=float, default=3.0e-8)
    args = parser.parse_args()
    if args.R_over_lambda <= 0.0 or args.lambda_over_dx < 12.0:
        raise ValueError("R/lambda must be positive and lambda/dx must be >= 12")
    if args.dt <= 0.0 or args.hold_steps <= 0:
        raise ValueError("dt and hold steps must be positive")

    base = args.base_params.resolve()
    params = parse_params(base)
    temperature_k = float(params["temperature_C"]) + 273.15
    lambda_nm = float(params["lambda_sm_m"]) * 1.0e9
    dx_nm = lambda_nm / args.lambda_over_dx
    target_radius_nm = args.R_over_lambda * lambda_nm
    requested_domain_nm = args.domain_nm or max(
        16.0, 2.0 * (target_radius_nm + 4.0 * lambda_nm)
    )
    nx = even_grid_at_least(requested_domain_nm, dx_nm)
    nz = nx
    ny = 2
    domain_nm = nx * dx_nm
    far_radius_nm = 0.5 * domain_nm
    if far_radius_nm - target_radius_nm < 3.5 * lambda_nm:
        raise ValueError("stationary circle requires at least 3.5 lambda buffer")

    diffusion_phys = unit.D_Ag_in_PbTe_m2_per_s(temperature_k)
    diffusion_code = float(params["D_alpha"])
    time_unit = float(params["t_real_unit"])
    dx_reference_nm = math.sqrt(diffusion_phys * time_unit / diffusion_code) * 1.0e9
    dx_code = dx_nm / dx_reference_nm
    radius_code = target_radius_nm / dx_reference_nm
    far_radius_code = far_radius_nm / dx_reference_nm
    w = float(params["W"])
    kappa = float(params["kappa_phi"])
    l_phi = float(params["L_phi"])
    energy_scale = float(params["mu_reference_scale"])
    v_a = float(params["v_A"])
    v_b = float(params["v_B"])
    if abs(v_a) > 1.0e-15 or abs(v_b - 1.0) > 1.0e-15:
        raise ValueError("this audited oracle currently requires v_A=0 and v_B=1")
    if abs(float(params["Vm_compound"]) - 1.0) > 1.0e-15 or abs(
        float(params["Vm_alpha_0"]) - 1.0
    ) > 1.0e-15 or abs(float(params["dVm_alpha_dxB"])) > 1.0e-15:
        raise ValueError("this audited oracle requires the accepted equal-volume model")
    x_eq = unit.xAg2Te_eq_from_T(temperature_k)
    mu0_compound = unit.mu_Ag2Te(temperature_k, x_eq) / energy_scale

    solution, matrix_x, delta_mu = solve_radial_equilibrium(
        radius_code,
        far_radius_code,
        temperature_k,
        energy_scale,
        w,
        kappa,
        mu0_compound,
        radial_spacing=max(dx_code * 0.5, far_radius_code / 2000.0),
        tolerance=args.bvp_tol,
    )
    if not solution.success:
        raise RuntimeError(f"radial BVP failed: {solution.message}")

    coordinates_nm = np.arange(nx, dtype=np.float64) * dx_nm
    xx_nm, zz_nm = np.meshgrid(coordinates_nm, coordinates_nm, indexing="ij")
    radius_nm = np.sqrt(
        (xx_nm - 0.5 * domain_nm) ** 2 +
        (zz_nm - 0.5 * domain_nm) ** 2
    )
    radius_eval_code = np.minimum(radius_nm / dx_reference_nm, far_radius_code)
    phi_2d = np.clip(solution.sol(radius_eval_code.ravel())[0].reshape(nx, nz),
                     0.0, 1.0)
    radial_matrix_x = matrix_x
    radial_delta_mu = delta_mu
    target_h_cells = math.pi * radius_code**2 / dx_code**2
    phi_2d, delta_mu, representability_core, refinement = (
        refine_production_spectral_equilibrium(
            phi_2d,
            delta_mu,
            matrix_x,
            target_h_cells,
            dx_code,
            w,
            kappa,
            args.spectral_tol,
        )
    )
    matrix_x = matrix_x_from_delta_mu(
        delta_mu, temperature_k, energy_scale, mu0_compound
    )
    laplacian = spectral_laplacian_2d(phi_2d, dx_code)
    phase_residual = w * gp(phi_2d) + delta_mu * hp(phi_2d) - kappa * laplacian
    residual_linf = float(np.max(np.abs(phase_residual)))
    residual_l2 = float(np.sqrt(np.mean(phase_residual**2)))
    residual_free_linf = float(
        np.max(np.abs(phase_residual[~representability_core]))
    )
    if residual_free_linf > args.spectral_tol:
        raise RuntimeError(
            f"production free-cell spectral residual "
            f"{residual_free_linf:.17e} exceeds "
            f"{args.spectral_tol:.17e}; a 2D Newton refinement is required"
        )

    x_2d = np.full_like(phi_2d, matrix_x)
    h_2d = h_stable(phi_2d)
    alpha_2d = alpha_stable(phi_2d)
    ctot_2d = np.where(
        phi_2d > 0.5,
        v_b - alpha_2d * (v_b - x_2d),
        h_2d * v_b + alpha_2d * x_2d,
    )
    phi = np.repeat(phi_2d[:, None, :], ny, axis=1)
    x_field = np.repeat(x_2d[:, None, :], ny, axis=1)
    ctot = np.repeat(ctot_2d[:, None, :], ny, axis=1)
    h_volume_code2 = float(np.sum(h_2d) * dx_code**2)
    h_radius_code = math.sqrt(h_volume_code2 / math.pi)
    radial_profile = solution.sol(solution.x)[0]
    phi_half_radius_code = radius_at_level(solution.x, radial_profile, 0.5)
    kkt_linf = projected_kkt_linf(
        phi_2d,
        ctot_2d,
        phase_residual,
        args.dt,
        l_phi,
        v_b,
        1.0e-8,
        1.0 - 1.0e-8,
    )
    mu_matrix = (
        unit.mu_Ag2Te(temperature_k, matrix_x) -
        unit.mu_PbTe(temperature_k, matrix_x)
    ) / energy_scale
    mean_ctot = float(np.mean(ctot))
    mass_cell_units = float(np.sum(ctot))
    alpha_3d = alpha_stable(phi)
    reconstruction = np.where(
        phi > 0.5,
        v_b - alpha_3d * (v_b - x_field),
        h_stable(phi) * v_b + alpha_3d * x_field,
    )
    reconstruction_linf = float(np.max(np.abs(reconstruction - ctot)))
    energy_density = w * g(phi_2d)
    gradient_energy = -0.5 * kappa * float(
        np.sum(phi_2d * laplacian) * dx_code**2
    )
    interfacial_energy = float(np.sum(energy_density) * dx_code**2) + gradient_energy

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    phi.astype(np.float64).tofile(out / "phi_init.raw")
    x_field.astype(np.float64).tofile(out / "xB_init.raw")
    ctot.astype(np.float64).tofile(out / "Ctot_init.raw")
    with (out / "radial_profile.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["radius_code", "radius_nm", "phi", "h_phi"])
        for r_value, p_value in zip(solution.x, radial_profile):
            writer.writerow(
                [
                    f"{r_value:.17e}",
                    f"{r_value * dx_reference_nm:.17e}",
                    f"{p_value:.17e}",
                    f"{float(h(np.array(p_value))):.17e}",
                ]
            )

    metrics = {
        "ensemble": "fixed_total_Ctot_periodic_domain",
        "radial_oracle": "cylindrical_discrete_BVP_with_h_volume_constraint",
        "production_grid_oracle": "periodic_2D_spectral_laplacian_v1",
        "temperature_K": temperature_k,
        "R_over_lambda": args.R_over_lambda,
        "lambda_over_dx": args.lambda_over_dx,
        "target_radius_nm": target_radius_nm,
        "phi_half_radius_nm": phi_half_radius_code * dx_reference_nm,
        "h_volume_radius_nm": h_radius_code * dx_reference_nm,
        "matrix_xB_equilibrium": matrix_x,
        "radial_matrix_xB_equilibrium": radial_matrix_x,
        "xB_planar_equilibrium": x_eq,
        "delta_mu_phase_dimless": delta_mu,
        "radial_delta_mu_phase_dimless": radial_delta_mu,
        "mu_matrix_dimless": mu_matrix,
        "mu_variation_Linf": 0.0,
        "radial_BVP_max_rms_residual": float(np.max(solution.rms_residuals)),
        "production_spectral_phase_residual_Linf": residual_linf,
        "production_spectral_phase_residual_free_Linf": residual_free_linf,
        "production_spectral_phase_residual_L2": residual_l2,
        "projected_phase_KKT_Linf": kkt_linf,
        "representability_core_cells": int(
            np.count_nonzero(representability_core)
        ),
        "spectral_refinement": refinement,
        "mass_target_cell_units": mass_cell_units,
        "mean_Ctot": mean_ctot,
        "storage_reconstruction_Linf": reconstruction_linf,
        "interfacial_energy_code": interfacial_energy,
        "grid": [nx, ny, nz],
        "dx_nm": dx_nm,
        "dx_code": dx_code,
        "domain_nm": domain_nm,
        "source_initial_state_changed_from_P1_replay": True,
        "P1_solver_flags_changed": False,
    }
    (out / "stationary_oracle_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    metadata = {
        "schema": "ctot_checkpoint_v1",
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "dx_nm": dx_nm,
        "interface_width_nm": lambda_nm,
        "dtype": "float64",
        "order": "C",
        "authoritative_state": "Ctot",
        "transport_operator_name": "mimetic_shared_face_v1",
        "transport_operator_version": "1",
        "gradient_operator_name": "periodic_positive_face_difference_v1",
        "divergence_operator_name": "periodic_face_incidence_v1",
        "adjoint_identity_mode": "D_equals_negative_G_star_exact",
        "face_mobility_mode": "symmetric_harmonic_zero_endpoint_v1",
        "phase_solver_name": "semismooth_pdas_v1",
        "phase_solver_version": "1",
        "finite_interface_correction_enabled": 0,
        "step": 0,
        "time_code": 0.0,
        "migrated_from_legacy_restart": False,
        "reference_type": "stationary_periodic_2d_cylindrical_fixed_Ctot",
        "ensemble": "fixed_total_Ctot_periodic_domain",
        "mean_Ctot": mean_ctot,
        "matrix_xB_equilibrium": matrix_x,
        "R_over_lambda": args.R_over_lambda,
        "lambda_over_dx": args.lambda_over_dx,
        "oracle_metrics_sha256": sha256(out / "stationary_oracle_metrics.json"),
    }
    (out / "init_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")

    write_runtime_params(
        base,
        out / "benchmark.params",
        {
            "dx": dx_code,
            "dy": dx_code,
            "dz": dx_code,
            "dt": args.dt,
            "composition_evolution_mode": "ctot_mimetic_be",
            "ctot_phase_semismooth_pdas_enabled": 1,
            "ctot_finite_interface_antitrapping_enabled": 0,
            "finite_interface_resolution_test_override": 0,
            "ctot_step_max_retries": 0,
            "ctot_automatic_dt_growth": 0,
            "ctot_outer_max_iter": 24,
            "ctot_elastic_validation_enabled": 0,
            "elastic_enabled": 0,
            "ctot_performance_profile_enabled": 0,
        },
    )
    manifest = {
        **metrics,
        "hold_steps": args.hold_steps,
        "dt_code": args.dt,
        "final_code_time": args.dt * args.hold_steps,
        "final_physical_time_s": args.dt * args.hold_steps * time_unit,
        "finite_interface_correction": 0,
        "P1_reference_binary_sha256":
            "1bc5405ab12a6fd872480d5d6c373b98fc3b8940e5d7044cd7a50e6c3bf6c4b0",
        "base_params_sha256": sha256(base),
        "provenance": "stationary_curved_oracle_not_time_relaxed_not_production_3D",
    }
    (out / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"stationary_equilibrium_out={out}")
    print(f"grid={nx}x{ny}x{nz}")
    print(f"matrix_xB_equilibrium={matrix_x:.17e}")
    print(f"radial_BVP_max_rms_residual={max(solution.rms_residuals):.17e}")
    print(f"production_spectral_phase_residual_Linf={residual_linf:.17e}")
    print(
        "production_spectral_phase_residual_free_Linf="
        f"{residual_free_linf:.17e}"
    )
    print(f"projected_phase_KKT_Linf={kkt_linf:.17e}")
    print(f"storage_reconstruction_Linf={reconstruction_linf:.17e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
