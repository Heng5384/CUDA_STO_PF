#!/usr/bin/env python3
"""Independent finite-box one-sided radial sharp-interface oracle.

The outer cylinder has the same area as the periodic square PF box.  Matrix
diffusion is radial, the outer boundary is no-flux, the beta-side flux is
zero, and the interface obeys the cylindrical Gibbs--Thomson condition and
one-sided Stefan balance.  No PF trajectory is used to calibrate this solver.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags
from scipy.optimize import brentq
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402


def trapezoid(values: np.ndarray, coordinates: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(values, coordinates))


def parse_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def square_equivalent_outer_radius(side_length: float) -> float:
    """Radius of a cylinder with area equal to the square PF cross-section."""
    if side_length <= 0.0:
        raise ValueError("side length must be positive")
    return side_length / math.sqrt(math.pi)


def volume_equivalent_outer_radius(
    side_length: float, radial_dimension: int
) -> float:
    """Equal-measure radial outer boundary for a square/cubic PF box."""
    if side_length <= 0.0:
        raise ValueError("side length must be positive")
    if radial_dimension == 2:
        return square_equivalent_outer_radius(side_length)
    if radial_dimension == 3:
        return (3.0 * side_length**3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    raise ValueError("radial_dimension must be 2 or 3")


def solve_gibbs_thomson_x(
    temperature_k: float,
    radius_nm: float,
    gamma_j_m2: float,
    molar_volume_m3_mol: float,
    chemical_scale_j_mol: float,
) -> float:
    """Solve the accepted cylindrical Gibbs--Thomson composition root."""
    return solve_interface_composition_x(
        temperature_k,
        radius_nm,
        gamma_j_m2,
        molar_volume_m3_mol,
        chemical_scale_j_mol,
        radial_dimension=2,
        kinetic_beta_code=0.0,
        velocity_nm_per_code_time=0.0,
    )


def solve_interface_composition_x(
    temperature_k: float,
    radius_nm: float,
    gamma_j_m2: float,
    molar_volume_m3_mol: float,
    chemical_scale_j_mol: float,
    radial_dimension: int,
    kinetic_beta_code: float,
    velocity_nm_per_code_time: float,
    code_length_unit_nm: float = 1.0,
) -> float:
    """Solve Gibbs--Thomson plus finite-Lphi kinetic boundary composition.

    The normal points from beta to matrix and positive velocity denotes beta
    growth.  For radial dimension ``d``, kappa=(d-1)/R.  The kinetic
    coefficient is in code chemical-potential per code velocity, while this
    oracle stores radii in nm; ``code_length_unit_nm`` performs that explicit
    conversion.
    """
    if radius_nm <= 0.0:
        raise ValueError("radius must be positive")
    if radial_dimension not in (2, 3):
        raise ValueError("radial_dimension must be 2 or 3")
    if not (chemical_scale_j_mol > 0.0 and code_length_unit_nm > 0.0):
        raise ValueError("chemical and code-length scales must be positive")
    if not (math.isfinite(kinetic_beta_code)
            and math.isfinite(velocity_nm_per_code_time)):
        raise ValueError("kinetic boundary inputs must be finite")
    x_eq = unit.xAg2Te_eq_from_T(temperature_k)
    mu_eq = (
        unit.mu_Ag2Te(temperature_k, x_eq)
        - unit.mu_PbTe(temperature_k, x_eq)
    )
    curvature_m_inv = (radial_dimension - 1.0) / (radius_nm * 1.0e-9)
    velocity_code = velocity_nm_per_code_time / code_length_unit_nm
    capillary_target = (
        mu_eq
        + gamma_j_m2 * molar_volume_m3_mol * curvature_m_inv
    )

    def chemical_difference(x: float) -> float:
        return (
            unit.mu_Ag2Te(temperature_k, x)
            - unit.mu_PbTe(temperature_k, x)
        )

    def monotone_root(function, target: float, lower: float,
                      label: str) -> float:
        lo = lower
        hi = 0.15
        value_lo = function(lo) - target
        value_hi = function(hi) - target
        if value_lo > 0.0 or value_hi < 0.0:
            raise RuntimeError(
                f"{label} root is outside the accepted [lo,0.15] bracket: "
                f"residual_lo={value_lo:.17e} "
                f"residual_hi={value_hi:.17e}"
            )
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            if function(mid) < target:
                lo = mid
            else:
                hi = mid
        result = 0.5 * (lo + hi)
        residual = (function(result) - target) / chemical_scale_j_mol
        if abs(residual) > 1.0e-13:
            raise RuntimeError(f"{label} root residual {residual:.17e}")
        return result

    # Preserve the frozen oracle's capillary convention exactly.  The new
    # finite-Lphi term comes from phase solvability and therefore shifts the
    # Ag2Te component chemical potential mu_B at that capillary root; it is not
    # silently reinterpreted as a mu_B-mu_A increment.
    capillary = monotone_root(
        chemical_difference, capillary_target, x_eq,
        "accepted Gibbs--Thomson",
    )
    kinetic_increment = (
        chemical_scale_j_mol * kinetic_beta_code * velocity_code
    )
    if kinetic_increment == 0.0:
        return capillary
    component_target = unit.mu_Ag2Te(temperature_k, capillary) + kinetic_increment
    lower = 1.0e-12 if kinetic_increment < 0.0 else capillary
    return monotone_root(
        lambda x: unit.mu_Ag2Te(temperature_k, x),
        component_target,
        lower,
        "finite-Lphi component-potential",
    )


def finite_lphi_kinetic_beta_code(
    params: dict[str, str], code_length_unit_nm: float = 1.0
) -> float:
    """Return A_phi/(L_phi*lambda_code), with A_phi=2/3."""
    if code_length_unit_nm <= 0.0:
        raise ValueError("code_length_unit_nm must be positive")
    l_phi = float(
        params["L_phi_code_value"]
        if "L_phi_code_value" in params else params["L_phi"]
    )
    if "L_phi_code_value" in params:
        l_phi_runtime = float(params["L_phi"])
        if abs(l_phi_runtime - l_phi) > 1.0e-12 * max(1.0, abs(l_phi)):
            raise RuntimeError("L_phi and L_phi_code_value disagree")
    lambda_nm = float(params["lambda_sm_m"]) * 1.0e9
    lambda_code = lambda_nm / code_length_unit_nm
    if not (l_phi > 0.0 and lambda_code > 0.0):
        raise ValueError("L_phi and lambda_code must be positive")
    return (2.0 / 3.0) / (l_phi * lambda_code)


def _radial_measure(radius: np.ndarray, radial_dimension: int) -> np.ndarray:
    if radial_dimension == 2:
        return 2.0 * math.pi * radius
    if radial_dimension == 3:
        return 4.0 * math.pi * radius**2
    raise ValueError("radial_dimension must be 2 or 3")


def _beta_volume(radius: float, radial_dimension: int) -> float:
    if radial_dimension == 2:
        return math.pi * radius**2
    if radial_dimension == 3:
        return (4.0 / 3.0) * math.pi * radius**3
    raise ValueError("radial_dimension must be 2 or 3")


def _radial_operator(
    radius: float,
    outer_radius: float,
    diffusivity: float,
    nodes: int,
    radial_dimension: int = 2,
) -> tuple[object, np.ndarray]:
    """Linear radial diffusion operator for nodes 1..N with c(0)=0 offset."""
    if not (0.0 < radius < outer_radius and diffusivity > 0.0 and nodes >= 8):
        raise ValueError("invalid radial operator inputs")
    dr = (outer_radius - radius) / nodes
    r = radius + dr * np.arange(1, nodes + 1, dtype=np.float64)
    radial_factor = radial_dimension - 1.0
    if radial_dimension not in (2, 3):
        raise ValueError("radial_dimension must be 2 or 3")
    lower = diffusivity / dr**2 - radial_factor * diffusivity / (2.0 * r * dr)
    upper = diffusivity / dr**2 + radial_factor * diffusivity / (2.0 * r * dr)
    diagonal = np.full(nodes, -2.0 * diffusivity / dr**2)
    # At the no-flux outer node, the reflected ghost value gives c_r=0.
    lower[-1] = 2.0 * diffusivity / dr**2
    upper[-1] = 0.0
    operator = diags(
        (lower[1:], diagonal, upper[:-1]), offsets=(-1, 0, 1), format="csr"
    )
    return operator, r


def aged_no_flux_profile(
    radius: float,
    outer_radius: float,
    diffusivity: float,
    surface_x: float,
    far_x: float,
    age: float,
    nodes: int,
    radial_dimension: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Diffuse an initially uniform matrix at fixed R with exact linear time action."""
    if age <= 0.0:
        raise ValueError("diffusion age must be positive")
    operator, r_inner = _radial_operator(
        radius, outer_radius, diffusivity, nodes, radial_dimension
    )
    offset_initial = np.full(nodes, far_x - surface_x, dtype=np.float64)
    offset = expm_multiply(operator * age, offset_initial)
    radial_nodes = np.concatenate(([radius], r_inner))
    profile = np.concatenate(([surface_x], surface_x + offset))
    return radial_nodes, profile


def profile_inventory(
    radius: float, radial_nodes: np.ndarray, profile: np.ndarray,
    radial_dimension: int = 2,
) -> float:
    if radial_nodes.shape != profile.shape or radial_nodes[0] != radius:
        raise ValueError("profile shape/radius mismatch")
    return _beta_volume(radius, radial_dimension) + trapezoid(
        _radial_measure(radial_nodes, radial_dimension) * profile,
        radial_nodes,
    )


def add_far_inventory_correction(
    radius: float,
    radial_nodes: np.ndarray,
    profile: np.ndarray,
    target_inventory: float,
    protected_width: float,
    radial_dimension: int = 2,
) -> tuple[np.ndarray, float, float]:
    """Match global inventory without changing the interface value/gradient.

    The correction is identically zero through the protected near-interface
    interval and transitions smoothly to a constant far-field offset.  It is
    an ensemble-matching operation for the independent sharp oracle, not a PF
    mass projection or a fitted physical parameter.
    """
    outer = float(radial_nodes[-1])
    start = min(radius + max(protected_width, 0.0), outer)
    shape = np.zeros_like(radial_nodes)
    if start < outer:
        s = np.clip((radial_nodes - start) / (outer - start), 0.0, 1.0)
        shape = s**3 * (10.0 - 15.0 * s + 6.0 * s**2)
    coefficient = trapezoid(
        _radial_measure(radial_nodes, radial_dimension) * shape, radial_nodes
    )
    current = profile_inventory(
        radius, radial_nodes, profile, radial_dimension
    )
    if coefficient <= 0.0:
        if abs(target_inventory - current) > 1.0e-14:
            raise RuntimeError("no support for finite-box inventory correction")
        return profile.copy(), 0.0, current
    delta = (target_inventory - current) / coefficient
    corrected = profile + delta * shape
    return corrected, delta, profile_inventory(
        radius, radial_nodes, corrected, radial_dimension
    )


def _operator_coefficients(
    radius: float,
    outer_radius: float,
    velocity: float,
    diffusivity: float,
    nodes: int,
    radial_dimension: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gap = outer_radius - radius
    dy = 1.0 / nodes
    y = np.arange(1, nodes + 1, dtype=np.float64) * dy
    r = radius + y * gap
    second = diffusivity / gap**2
    if radial_dimension not in (2, 3):
        raise ValueError("radial_dimension must be 2 or 3")
    first = (
        (radial_dimension - 1.0) * diffusivity / (r * gap)
        + (1.0 - y) * velocity / gap
    )
    lower = second / dy**2 - first / (2.0 * dy)
    upper = second / dy**2 + first / (2.0 * dy)
    diagonal = np.full(nodes, -2.0 * second / dy**2)
    lower[-1] = 2.0 * second / dy**2
    upper[-1] = 0.0
    return lower, diagonal, upper


def _surface_gradient(
    surface_x: float, profile_unknown: np.ndarray, dr: float
) -> float:
    if profile_unknown.size < 2:
        raise ValueError("at least two matrix nodes are required")
    return (
        -3.0 * surface_x + 4.0 * profile_unknown[0] - profile_unknown[1]
    ) / (2.0 * dr)


def solve_stefan_surface_state(
    radius: float,
    profile_unknown: np.ndarray,
    dr: float,
    diffusivity: float,
    surface_composition,
    velocity_hint: float = 0.0,
) -> tuple[float, float, float]:
    """Solve the kinetic interface composition and Stefan speed together."""
    if dr <= 0.0 or diffusivity <= 0.0:
        raise ValueError("dr and diffusivity must be positive")

    def residual(velocity: float) -> float:
        surface = float(surface_composition(radius, velocity))
        stefan = diffusivity * _surface_gradient(
            surface, profile_unknown, dr
        ) / (1.0 - surface)
        return velocity - stefan

    width = max(0.02, 4.0 * abs(velocity_hint))
    lo = velocity_hint - width
    hi = velocity_hint + width
    f_lo = residual(lo)
    f_hi = residual(hi)
    for _ in range(12):
        if f_lo <= 0.0 <= f_hi:
            break
        width *= 2.0
        lo = velocity_hint - width
        hi = velocity_hint + width
        f_lo = residual(lo)
        f_hi = residual(hi)
    else:
        raise RuntimeError(
            "failed to bracket kinetic Stefan state: "
            f"lo={lo:.17e} f_lo={f_lo:.17e} "
            f"hi={hi:.17e} f_hi={f_hi:.17e}"
        )
    velocity = brentq(
        residual, lo, hi, xtol=1.0e-14, rtol=1.0e-13, maxiter=200
    )
    surface = float(surface_composition(radius, velocity))
    final_residual = residual(velocity)
    return surface, velocity, final_residual


def _cn_step(
    old: np.ndarray,
    old_surface: float,
    new_surface: float,
    radius_mid: float,
    outer_radius: float,
    velocity_mid: float,
    diffusivity: float,
    dt: float,
    radial_dimension: int = 2,
) -> np.ndarray:
    nodes = old.size
    lower, diagonal, upper = _operator_coefficients(
        radius_mid, outer_radius, velocity_mid, diffusivity, nodes,
        radial_dimension,
    )
    rhs = (1.0 + 0.5 * dt * diagonal) * old
    rhs[1:] += 0.5 * dt * lower[1:] * old[:-1]
    rhs[:-1] += 0.5 * dt * upper[:-1] * old[1:]
    rhs[0] += 0.5 * dt * lower[0] * (old_surface + new_surface)

    band = np.zeros((3, nodes), dtype=np.float64)
    band[1] = 1.0 - 0.5 * dt * diagonal
    band[0, 1:] = -0.5 * dt * upper[:-1]
    band[2, :-1] = -0.5 * dt * lower[1:]
    return solve_banded((1, 1), band, rhs, check_finite=False)


def evolve_moving_interface(
    radius_initial: float,
    outer_radius: float,
    radial_nodes: np.ndarray,
    profile: np.ndarray,
    diffusivity: float,
    surface_composition,
    final_time: float,
    dt: float,
    radial_dimension: int = 2,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    """Crank--Nicolson ALE evolution with trapezoidal Stefan update."""
    if final_time <= 0.0 or dt <= 0.0:
        raise ValueError("final time and dt must be positive")
    nodes = profile.size - 1
    if nodes < 8:
        raise ValueError("insufficient radial nodes")
    y = np.linspace(0.0, 1.0, nodes + 1)
    unknown = profile[1:].copy()
    radius = radius_initial
    time = 0.0
    dr = (outer_radius - radius) / nodes
    surface, velocity, kinetic_residual = solve_stefan_surface_state(
        radius, unknown, dr, diffusivity, surface_composition
    )
    initial_profile = np.concatenate(([surface], unknown))
    initial_inventory = profile_inventory(
        radius, radial_nodes, initial_profile, radial_dimension
    )
    history: list[dict[str, float]] = []
    coupled_iterations = 0

    def append_history(step: int, velocity: float) -> None:
        current_nodes = radius + y * (outer_radius - radius)
        current_profile = np.concatenate(([surface], unknown))
        inventory = profile_inventory(
            radius, current_nodes, current_profile, radial_dimension
        )
        history.append({
            "step": float(step),
            "time_code": time,
            "radius": radius,
            "surface_xB": surface,
            "velocity": velocity,
            "inventory": inventory,
            "inventory_error_rel": abs(inventory - initial_inventory)
            / max(abs(initial_inventory), 1.0e-300),
            "kinetic_boundary_residual": kinetic_residual,
            "coupled_boundary_iterations": float(coupled_iterations),
        })

    append_history(0, velocity)
    step = 0
    while time < final_time - 0.5 * np.finfo(np.float64).eps:
        step += 1
        step_dt = min(dt, final_time - time)
        old_unknown = unknown.copy()
        old_radius = radius
        old_surface = surface
        old_velocity = velocity
        radius_trial = old_radius + step_dt * old_velocity
        trial_unknown = old_unknown
        trial_velocity = old_velocity
        coupled_converged = False
        for coupled_iter in range(20):
            surface_guess = float(surface_composition(
                radius_trial, trial_velocity
            ))
            radius_mid = 0.5 * (old_radius + radius_trial)
            velocity_mid = 0.5 * (old_velocity + trial_velocity)
            trial_unknown = _cn_step(
                old_unknown,
                old_surface,
                surface_guess,
                radius_mid,
                outer_radius,
                velocity_mid,
                diffusivity,
                step_dt,
                radial_dimension,
            )
            dr_trial = (outer_radius - radius_trial) / nodes
            surface_trial, new_velocity, kinetic_residual = (
                solve_stefan_surface_state(
                    radius_trial, trial_unknown, dr_trial, diffusivity,
                    surface_composition, trial_velocity,
                )
            )
            new_radius = old_radius + 0.5 * step_dt * (
                old_velocity + new_velocity
            )
            radius_ok = (
                abs(new_radius - radius_trial)
                <= 1.0e-13 * max(1.0, abs(old_radius))
            )
            velocity_ok = (
                abs(new_velocity - trial_velocity)
                <= 1.0e-11 * max(1.0, abs(new_velocity))
            )
            surface_ok = (
                abs(surface_trial - surface_guess)
                <= 1.0e-13 * max(1.0, abs(surface_trial))
            )
            if radius_ok and velocity_ok and surface_ok:
                radius_trial = new_radius
                trial_velocity = new_velocity
                coupled_iterations = coupled_iter + 1
                coupled_converged = True
                break
            radius_trial = new_radius
            trial_velocity = new_velocity
        if not coupled_converged:
            raise RuntimeError(
                "kinetic ALE/Stefan iteration did not converge at "
                f"step={step} radius={radius_trial:.17e} "
                f"velocity={trial_velocity:.17e}"
            )
        radius = radius_trial
        unknown = trial_unknown
        dr = (outer_radius - radius) / nodes
        surface, velocity, kinetic_residual = solve_stefan_surface_state(
            radius, unknown, dr, diffusivity, surface_composition,
            trial_velocity,
        )
        time += step_dt
        if step == 1 or time >= final_time - 0.5 * step_dt:
            append_history(step, velocity)
    final_nodes = radius + y * (outer_radius - radius)
    final_profile = np.concatenate(([surface], unknown))
    return history, final_nodes, final_profile


def run_case(
    params_path: Path,
    radius_nm: float,
    box_nm: float,
    far_x: float,
    diffusion_age_code: float,
    final_time_code: float,
    sharp_dt_code: float,
    nodes: int,
    target_inventory: float | None = None,
    finite_lphi_kinetics: bool = False,
    radial_dimension: int = 2,
    code_length_unit_nm: float = 1.0,
    l_phi_multiplier: float = 1.0,
    prepared_capillary_state: dict[str, object] | None = None,
) -> dict[str, object]:
    if not (math.isfinite(l_phi_multiplier) and l_phi_multiplier > 0.0):
        raise ValueError("l_phi_multiplier must be positive and finite")
    params = parse_params(params_path)
    temperature_k = float(params["temperature_C"]) + 273.15
    diffusivity = float(params["D_alpha"])
    gamma = float(params["gamma_Jm2"])
    scale = float(params["mu_reference_scale"])
    beta_phi_nominal = finite_lphi_kinetic_beta_code(
        params, code_length_unit_nm
    )
    beta_phi = (
        beta_phi_nominal / l_phi_multiplier
        if finite_lphi_kinetics else 0.0
    )

    def surface(radius: float, velocity: float) -> float:
        return solve_interface_composition_x(
            temperature_k,
            radius,
            gamma,
            unit.USER_PHYSICAL_INPUTS.Vm_compound,
            scale,
            radial_dimension,
            beta_phi,
            velocity,
            code_length_unit_nm,
        )

    outer_radius = volume_equivalent_outer_radius(box_nm, radial_dimension)
    surface_initial_capillary = surface(radius_nm, 0.0)
    if prepared_capillary_state is None:
        radial_nodes, profile = aged_no_flux_profile(
            radius_nm,
            outer_radius,
            diffusivity,
            surface_initial_capillary,
            far_x,
            diffusion_age_code,
            nodes,
            radial_dimension,
        )
        correction = 0.0
        if target_inventory is not None:
            profile, correction, matched = add_far_inventory_correction(
                radius_nm,
                radial_nodes,
                profile,
                target_inventory,
                protected_width=1.2,
                radial_dimension=radial_dimension,
            )
            if abs(matched - target_inventory) > 5.0e-12 * max(
                    1.0, target_inventory):
                raise RuntimeError("sharp/PF inventory match failed")
        inventory_target = (
            float(target_inventory) if target_inventory is not None
            else profile_inventory(
                radius_nm, radial_nodes, profile, radial_dimension
            )
        )
        capillary_state = {
            "radius_nm": radius_nm,
            "outer_radius_nm": outer_radius,
            "nodes": nodes,
            "radial_dimension": radial_dimension,
            "surface_initial_capillary": surface_initial_capillary,
            "radial_nodes": radial_nodes.copy(),
            "profile": profile.copy(),
            "inventory_target": inventory_target,
            "inventory_match_correction_xB": correction,
        }
    else:
        expected = (
            float(prepared_capillary_state["radius_nm"]),
            float(prepared_capillary_state["outer_radius_nm"]),
            int(prepared_capillary_state["nodes"]),
            int(prepared_capillary_state["radial_dimension"]),
        )
        actual = (radius_nm, outer_radius, nodes, radial_dimension)
        if any(
            abs(a - b) > 1.0e-13 * max(1.0, abs(a), abs(b))
            for a, b in zip(expected[:2], actual[:2])
        ) or expected[2:] != actual[2:]:
            raise ValueError("prepared capillary state does not match this case")
        surface_cached = float(
            prepared_capillary_state["surface_initial_capillary"]
        )
        if abs(surface_cached - surface_initial_capillary) > 1.0e-14:
            raise ValueError("prepared capillary state surface mismatch")
        radial_nodes = np.asarray(
            prepared_capillary_state["radial_nodes"], dtype=np.float64
        ).copy()
        profile = np.asarray(
            prepared_capillary_state["profile"], dtype=np.float64
        ).copy()
        inventory_target = float(
            prepared_capillary_state["inventory_target"]
        )
        if target_inventory is not None and abs(
            inventory_target - float(target_inventory)
        ) > 5.0e-12 * max(1.0, abs(inventory_target)):
            raise ValueError("prepared capillary state inventory mismatch")
        correction = float(
            prepared_capillary_state["inventory_match_correction_xB"]
        )
        capillary_state = prepared_capillary_state
    kinetic_inventory_correction = 0.0
    initial_dr = (outer_radius - radius_nm) / nodes
    initial_surface_kinetic, initial_velocity_kinetic, initial_kinetic_residual = (
        solve_stefan_surface_state(
            radius_nm, profile[1:], initial_dr, diffusivity, surface
        )
    )
    profile = profile.copy()
    profile[0] = initial_surface_kinetic
    profile, kinetic_inventory_correction, matched = add_far_inventory_correction(
        radius_nm,
        radial_nodes,
        profile,
        inventory_target,
        protected_width=1.2,
        radial_dimension=radial_dimension,
    )
    if abs(matched - inventory_target) > 5.0e-12 * max(1.0, inventory_target):
        raise RuntimeError("kinetic-boundary inventory rematch failed")
    history, final_nodes, final_profile = evolve_moving_interface(
        radius_nm,
        outer_radius,
        radial_nodes,
        profile,
        diffusivity,
        surface,
        final_time_code,
        sharp_dt_code,
        radial_dimension,
    )
    return {
        "temperature_K": temperature_k,
        "radius_initial_nm": radius_nm,
        "radius_final_nm": history[-1]["radius"],
        "velocity_full_nm_per_code_time":
            (history[-1]["radius"] - radius_nm) / final_time_code,
        "velocity_initial_nm_per_code_time": history[0]["velocity"],
        "velocity_final_nm_per_code_time": history[-1]["velocity"],
        "surface_xB_initial_capillary": surface_initial_capillary,
        "surface_xB_initial": history[0]["surface_xB"],
        "surface_xB_final": history[-1]["surface_xB"],
        "velocity_initial_kinetic_solve": initial_velocity_kinetic,
        "kinetic_boundary_initial_residual": initial_kinetic_residual,
        "max_kinetic_boundary_residual": max(
            abs(row["kinetic_boundary_residual"]) for row in history
        ),
        "finite_lphi_kinetics": bool(finite_lphi_kinetics),
        "l_phi_multiplier": l_phi_multiplier,
        "kinetic_beta_phi_nominal_code": beta_phi_nominal,
        "kinetic_beta_phi_code": beta_phi,
        "radial_dimension": radial_dimension,
        "curvature_factor": radial_dimension - 1,
        "code_length_unit_nm": code_length_unit_nm,
        "outer_radius_nm": outer_radius,
        "nodes": nodes,
        "sharp_dt_code": sharp_dt_code,
        "final_time_code": final_time_code,
        "initial_inventory": history[0]["inventory"],
        "final_inventory": history[-1]["inventory"],
        "max_inventory_error_rel": max(row["inventory_error_rel"] for row in history),
        "inventory_match_correction_xB": correction,
        "kinetic_boundary_inventory_correction_xB":
            kinetic_inventory_correction,
        "radial_nodes_initial": radial_nodes,
        "profile_initial": profile,
        "radial_nodes_final": final_nodes,
        "profile_final": final_profile,
        "prepared_capillary_state": capillary_state,
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--radius-nm", type=float, required=True)
    parser.add_argument("--box-nm", type=float, required=True)
    parser.add_argument("--far-xB", type=float, default=0.0078305391025)
    parser.add_argument("--diffusion-age-code", type=float, default=4.0 / 9.0)
    parser.add_argument("--final-time-code", type=float, required=True)
    parser.add_argument("--dt-code", type=float, required=True)
    parser.add_argument("--nodes", type=int, default=1024)
    parser.add_argument("--target-inventory", type=float)
    parser.add_argument("--finite-Lphi-kinetics", action="store_true")
    parser.add_argument("--radial-dimension", type=int, choices=(2, 3), default=2)
    parser.add_argument("--code-length-unit-nm", type=float, default=1.0)
    parser.add_argument("--Lphi-multiplier", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-profile-csv", type=Path)
    args = parser.parse_args()
    result = run_case(
        args.params,
        args.radius_nm,
        args.box_nm,
        args.far_xB,
        args.diffusion_age_code,
        args.final_time_code,
        args.dt_code,
        args.nodes,
        args.target_inventory,
        args.finite_Lphi_kinetics,
        args.radial_dimension,
        args.code_length_unit_nm,
        args.Lphi_multiplier,
    )
    serializable = {
        key: value for key, value in result.items()
        if key not in {
            "radial_nodes_initial", "profile_initial", "radial_nodes_final",
            "profile_final",
            "prepared_capillary_state",
        }
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(serializable, indent=2) + "\n")
    if args.output_profile_csv:
        args.output_profile_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_profile_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "radius_initial_grid_nm", "xB_initial",
                "radius_final_grid_nm", "xB_final",
            ])
            for values in zip(
                result["radial_nodes_initial"], result["profile_initial"],
                result["radial_nodes_final"], result["profile_final"],
            ):
                writer.writerow([f"{float(value):.17e}" for value in values])
    print(f"sharp_radius_final_nm={result['radius_final_nm']:.17e}")
    print(
        "sharp_velocity_full_nm_per_code_time="
        f"{result['velocity_full_nm_per_code_time']:.17e}"
    )
    print(f"sharp_inventory_error_rel={result['max_inventory_error_rel']:.17e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
