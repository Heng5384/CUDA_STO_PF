#!/usr/bin/env python3
"""Equation-preserving optimized finite-Lphi sharp-interface oracle.

The frozen Python oracle remains in ``next2_finite_box_sharp_oracle.py``.
This module changes only nonlinear solution strategy: analytic derivatives,
strictly bracketed Newton continuation, and a coupled interface-composition /
Stefan-velocity Newton solve with an exact fallback to the frozen nested path.
The Crank--Nicolson ALE update, inventory construction, and physical equations
are unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np

import Unit_Psedobinary as unit
from scripts import next2_finite_box_sharp_oracle as reference


CHECKPOINT_VERSION = "next4_optimized_ALE_checkpoint_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_ale_checkpoint(
    path: Path, state: dict[str, object], input_hash: str
) -> None:
    """Atomically save exact accepted ALE state plus a verified manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **state)
    checksum = _sha256(temporary)
    os.replace(temporary, path)
    metadata = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "input_hash": input_hash,
        "payload_sha256": checksum,
    }
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(json.dumps(metadata, sort_keys=True) + "\n")
    os.replace(metadata_temporary, metadata_path)


def load_ale_checkpoint(path: Path, input_hash: str) -> dict[str, object]:
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise RuntimeError("oracle checkpoint version mismatch")
    if metadata.get("input_hash") != input_hash:
        raise RuntimeError("oracle checkpoint input hash mismatch")
    if metadata.get("payload_sha256") != _sha256(path):
        raise RuntimeError("oracle checkpoint checksum mismatch")
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


@dataclass(frozen=True)
class ThermoBranch:
    temperature_k: float
    rt: float
    l0: float
    mu_b_base: float
    mu_a_base: float
    x_lower: float
    x_upper_full: float
    x_upper_monotone: float


@dataclass
class RootMetrics:
    iterations: int = 0
    function_evaluations: int = 0
    derivative_evaluations: int = 0
    bisection_steps: int = 0
    fallback_count: int = 0
    residual: float = math.nan

    def add(self, other: "RootMetrics") -> None:
        self.iterations += other.iterations
        self.function_evaluations += other.function_evaluations
        self.derivative_evaluations += other.derivative_evaluations
        self.bisection_steps += other.bisection_steps
        self.fallback_count += other.fallback_count
        self.residual = other.residual


@dataclass
class SolverMetrics:
    capillary: RootMetrics
    kinetic: RootMetrics
    coupled_newton_calls: int = 0
    coupled_newton_iterations: int = 0
    coupled_line_search_reductions: int = 0
    coupled_fallback_count: int = 0
    coupled_f1_abs_max: float = 0.0
    coupled_f2_abs_max: float = 0.0
    ale_coupled_iterations: int = 0

    @classmethod
    def empty(cls) -> "SolverMetrics":
        return cls(RootMetrics(), RootMetrics())

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["capillary"] = asdict(self.capillary)
        values["kinetic"] = asdict(self.kinetic)
        return values


def build_thermo_branch(
    temperature_k: float,
    x_lower: float = 1.0e-12,
    x_upper_full: float = 0.15,
) -> ThermoBranch:
    """Freeze temperature-only terms and identify the low-x monotone branch."""
    rt = unit.R_GAS * temperature_k
    l0 = unit.L0_PseudoBinary(temperature_k)
    monotone_upper = x_upper_full
    if l0 > 0.0:
        discriminant = 1.0 - 2.0 * rt / l0
        if discriminant > 0.0:
            turning = 0.5 * (1.0 - math.sqrt(discriminant))
            monotone_upper = min(x_upper_full, turning)
    if not (x_lower < monotone_upper <= x_upper_full < 1.0):
        raise ValueError("invalid thermodynamic low-composition branch")
    return ThermoBranch(
        temperature_k=temperature_k,
        rt=rt,
        l0=l0,
        mu_b_base=unit.G_Ag2Te_Solid(temperature_k),
        mu_a_base=unit.G_PbTe_Solid(temperature_k),
        x_lower=x_lower,
        x_upper_full=x_upper_full,
        x_upper_monotone=monotone_upper,
    )


def mu_b(branch: ThermoBranch, x: float) -> float:
    return branch.mu_b_base + branch.rt * math.log(x) + branch.l0 * (1.0 - x) ** 2


def mu_a(branch: ThermoBranch, x: float) -> float:
    return branch.mu_a_base + branch.rt * math.log(1.0 - x) + branch.l0 * x * x


def mu_b_minus_mu_a(branch: ThermoBranch, x: float) -> float:
    return mu_b(branch, x) - mu_a(branch, x)


def dmu_b_dx(branch: ThermoBranch, x: float) -> float:
    return branch.rt / x - 2.0 * branch.l0 * (1.0 - x)


def dmu_a_dx(branch: ThermoBranch, x: float) -> float:
    return -branch.rt / (1.0 - x) + 2.0 * branch.l0 * x


def dmu_b_minus_mu_a_dx(branch: ThermoBranch, x: float) -> float:
    return branch.rt / x + branch.rt / (1.0 - x) - 2.0 * branch.l0


def safeguarded_newton_root(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    lower: float,
    upper: float,
    initial: float,
    residual_scale: float,
    residual_tolerance: float = 5.0e-15,
    x_tolerance: float = 2.0e-16,
    max_iterations: int = 64,
) -> tuple[float, RootMetrics]:
    """Monotone-branch Newton with a strict sign bracket and bisection guard."""
    metrics = RootMetrics()
    f_lower = function(lower)
    f_upper = function(upper)
    metrics.function_evaluations += 2
    if not (math.isfinite(f_lower) and math.isfinite(f_upper)):
        raise RuntimeError("non-finite safeguarded-root endpoint")
    if f_lower > 0.0 or f_upper < 0.0:
        raise RuntimeError(
            "safeguarded root is not bracketed on the selected branch: "
            f"f_lower={f_lower:.17e} f_upper={f_upper:.17e}"
        )
    x = min(max(initial, lower), upper)
    for iteration in range(1, max_iterations + 1):
        f = function(x)
        metrics.function_evaluations += 1
        metrics.iterations = iteration
        metrics.residual = f
        if abs(f) <= residual_tolerance * residual_scale:
            return x, metrics
        if f < 0.0:
            lower, f_lower = x, f
        else:
            upper, f_upper = x, f
        if upper - lower <= x_tolerance * max(1.0, abs(x)):
            midpoint = 0.5 * (lower + upper)
            f_mid = function(midpoint)
            metrics.function_evaluations += 1
            metrics.residual = f_mid
            if abs(f_mid) <= 2.0 * residual_tolerance * residual_scale:
                return midpoint, metrics
        slope = derivative(x)
        metrics.derivative_evaluations += 1
        candidate = x - f / slope if math.isfinite(slope) and slope > 0.0 else math.nan
        if not (lower < candidate < upper):
            candidate = 0.5 * (lower + upper)
            metrics.bisection_steps += 1
        x = candidate
    raise RuntimeError(
        "safeguarded Newton did not converge: "
        f"x={x:.17e} residual={metrics.residual:.17e}"
    )


def legacy_full_bracket_bisection(
    function: Callable[[float], float],
    target: float,
    lower: float,
    upper: float,
    residual_scale: float,
) -> tuple[float, RootMetrics]:
    """Exact frozen 120-step low-root bisection over the full bracket."""
    metrics = RootMetrics(fallback_count=1)
    value_lower = function(lower) - target
    value_upper = function(upper) - target
    metrics.function_evaluations = 2
    if value_lower > 0.0 or value_upper < 0.0:
        raise RuntimeError("legacy full-bracket fallback is not bracketed")
    for _ in range(120):
        midpoint = 0.5 * (lower + upper)
        value = function(midpoint) - target
        metrics.function_evaluations += 1
        metrics.iterations += 1
        metrics.bisection_steps += 1
        if value < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    result = 0.5 * (lower + upper)
    metrics.residual = function(result) - target
    metrics.function_evaluations += 1
    if abs(metrics.residual) > 1.0e-13 * residual_scale:
        raise RuntimeError("legacy full-bracket fallback residual failed")
    return result, metrics


def solve_capillary_root(
    branch: ThermoBranch,
    radius_nm: float,
    gamma_j_m2: float,
    molar_volume_m3_mol: float,
    radial_dimension: int,
    initial: float | None = None,
    compiled: bool = False,
) -> tuple[float, RootMetrics]:
    if radius_nm <= 0.0 or radial_dimension not in (2, 3):
        raise ValueError("invalid capillary-root geometry")
    x_eq = unit.xAg2Te_eq_from_T(branch.temperature_k)
    mu_eq = mu_b_minus_mu_a(branch, x_eq)
    target = mu_eq + gamma_j_m2 * molar_volume_m3_mol * (
        radial_dimension - 1.0
    ) / (radius_nm * 1.0e-9)
    guess = x_eq if initial is None else initial
    if compiled:
        from scripts.next4_numba_root_kernels import capillary_root_jit
        result = capillary_root_jit(
            branch.mu_b_base, branch.mu_a_base, branch.rt, branch.l0,
            x_eq, target, x_eq, branch.x_upper_monotone, guess, 1.0e5,
        )
        if result[4] == 0:
            return result[0], RootMetrics(
                iterations=result[1], function_evaluations=result[2],
                derivative_evaluations=result[1],
                bisection_steps=result[3], residual=result[5],
            )
    try:
        return safeguarded_newton_root(
            lambda x: mu_b_minus_mu_a(branch, x) - target,
            lambda x: dmu_b_minus_mu_a_dx(branch, x),
            x_eq,
            branch.x_upper_monotone,
            guess,
            residual_scale=max(abs(target), 1.0e5),
        )
    except RuntimeError:
        return legacy_full_bracket_bisection(
            lambda x: mu_b_minus_mu_a(branch, x), target,
            x_eq, branch.x_upper_full, 1.0e5,
        )


def solve_kinetic_root(
    branch: ThermoBranch,
    capillary_x: float,
    kinetic_increment_j_mol: float,
    initial: float | None = None,
    compiled: bool = False,
) -> tuple[float, RootMetrics]:
    if kinetic_increment_j_mol == 0.0:
        return capillary_x, RootMetrics(
            iterations=0, function_evaluations=0, derivative_evaluations=0,
            bisection_steps=0, residual=0.0,
        )
    target = mu_b(branch, capillary_x) + kinetic_increment_j_mol
    lower = branch.x_lower if kinetic_increment_j_mol < 0.0 else capillary_x
    upper = capillary_x if kinetic_increment_j_mol < 0.0 else branch.x_upper_monotone
    guess = capillary_x if initial is None else initial
    if compiled:
        from scripts.next4_numba_root_kernels import kinetic_root_jit
        result = kinetic_root_jit(
            branch.mu_b_base, branch.rt, branch.l0, capillary_x,
            kinetic_increment_j_mol, lower, upper, guess, 1.0e5,
        )
        if result[4] == 0:
            return result[0], RootMetrics(
                iterations=result[1], function_evaluations=result[2],
                derivative_evaluations=result[1],
                bisection_steps=result[3], residual=result[5],
            )
    try:
        return safeguarded_newton_root(
            lambda x: mu_b(branch, x) - target,
            lambda x: dmu_b_dx(branch, x),
            lower,
            upper,
            guess,
            residual_scale=max(abs(target), 1.0e5),
        )
    except RuntimeError:
        legacy_lower = branch.x_lower if kinetic_increment_j_mol < 0.0 else capillary_x
        return legacy_full_bracket_bisection(
            lambda x: mu_b(branch, x), target,
            legacy_lower, branch.x_upper_full, 1.0e5,
        )


def surface_gradient(surface_x: float, unknown: np.ndarray, dr: float) -> float:
    return (-3.0 * surface_x + 4.0 * unknown[0] - unknown[1]) / (2.0 * dr)


def coupled_surface_velocity_newton_v1(
    branch: ThermoBranch,
    radius_nm: float,
    unknown: np.ndarray,
    dr: float,
    diffusivity: float,
    gamma_j_m2: float,
    molar_volume_m3_mol: float,
    chemical_scale_j_mol: float,
    radial_dimension: int,
    kinetic_beta_code: float,
    code_length_unit_nm: float,
    capillary_initial: float | None,
    interface_initial: float | None,
    velocity_initial: float,
    metrics: SolverMetrics,
    compiled: bool = False,
) -> tuple[float, float, float, float, float]:
    """Solve interface composition and Stefan velocity with an analytic 2x2 Jacobian."""
    capillary, cap_metrics = solve_capillary_root(
        branch, radius_nm, gamma_j_m2, molar_volume_m3_mol,
        radial_dimension, capillary_initial, compiled,
    )
    metrics.capillary.add(cap_metrics)
    x = capillary if interface_initial is None else interface_initial
    velocity = velocity_initial
    kinetic_slope = chemical_scale_j_mol * kinetic_beta_code / code_length_unit_nm
    metrics.coupled_newton_calls += 1
    if compiled:
        from scripts.next4_numba_root_kernels import coupled_surface_velocity_jit
        result = coupled_surface_velocity_jit(
            branch.mu_b_base, branch.rt, branch.l0, capillary,
            kinetic_slope, float(unknown[0]), float(unknown[1]), dr,
            diffusivity, x, velocity, branch.x_lower,
            branch.x_upper_monotone, chemical_scale_j_mol,
        )
        metrics.coupled_newton_iterations += int(result[2])
        metrics.coupled_line_search_reductions += int(result[3])
        if result[4] == 0:
            metrics.coupled_f1_abs_max = max(
                metrics.coupled_f1_abs_max, abs(float(result[5]))
            )
            metrics.coupled_f2_abs_max = max(
                metrics.coupled_f2_abs_max, abs(float(result[6]))
            )
            return capillary, result[0], result[1], result[5], result[6]
        metrics.coupled_fallback_count += 1
    converged = False
    f1 = f2 = math.nan
    for _ in range(16):
        gradient = surface_gradient(x, unknown, dr)
        f1 = mu_b(branch, x) - mu_b(branch, capillary) - kinetic_slope * velocity
        f2 = velocity - diffusivity * gradient / (1.0 - x)
        norm = max(abs(f1) / chemical_scale_j_mol, abs(f2))
        metrics.coupled_f1_abs_max = max(metrics.coupled_f1_abs_max, abs(f1))
        metrics.coupled_f2_abs_max = max(metrics.coupled_f2_abs_max, abs(f2))
        if norm <= 5.0e-14:
            converged = True
            break
        j11 = dmu_b_dx(branch, x)
        j12 = -kinetic_slope
        dgradient_dx = -3.0 / (2.0 * dr)
        j21 = -diffusivity * (
            dgradient_dx / (1.0 - x) + gradient / (1.0 - x) ** 2
        )
        determinant = j11 - j12 * j21
        if not math.isfinite(determinant) or abs(determinant) < 1.0e-30:
            break
        dx = (-f1 + j12 * f2) / determinant
        dv = (-j11 * f2 + j21 * f1) / determinant
        accepted = False
        damping = 1.0
        for reduction in range(13):
            x_trial = x + damping * dx
            v_trial = velocity + damping * dv
            if branch.x_lower < x_trial < branch.x_upper_monotone:
                g_trial = surface_gradient(x_trial, unknown, dr)
                f1_trial = (
                    mu_b(branch, x_trial) - mu_b(branch, capillary)
                    - kinetic_slope * v_trial
                )
                f2_trial = v_trial - diffusivity * g_trial / (1.0 - x_trial)
                trial_norm = max(
                    abs(f1_trial) / chemical_scale_j_mol, abs(f2_trial)
                )
                if trial_norm < norm:
                    x, velocity = x_trial, v_trial
                    metrics.coupled_line_search_reductions += reduction
                    accepted = True
                    break
            damping *= 0.5
        metrics.coupled_newton_iterations += 1
        if not accepted:
            break
    if converged:
        return capillary, x, velocity, f1, f2

    metrics.coupled_fallback_count += 1

    def frozen_surface(value: float, trial_velocity: float) -> float:
        return reference.solve_interface_composition_x(
            branch.temperature_k, value, gamma_j_m2,
            molar_volume_m3_mol, chemical_scale_j_mol,
            radial_dimension, kinetic_beta_code, trial_velocity,
            code_length_unit_nm,
        )

    x_fallback, velocity_fallback, f2_fallback = reference.solve_stefan_surface_state(
        radius_nm, unknown, dr, diffusivity, frozen_surface, velocity_initial
    )
    capillary_fallback = frozen_surface(radius_nm, 0.0)
    f1_fallback = (
        mu_b(branch, x_fallback) - mu_b(branch, capillary_fallback)
        - kinetic_slope * velocity_fallback
    )
    return (
        capillary_fallback, x_fallback, velocity_fallback,
        f1_fallback, f2_fallback,
    )


def _surface_at_velocity(
    branch: ThermoBranch,
    radius_nm: float,
    velocity: float,
    gamma_j_m2: float,
    molar_volume_m3_mol: float,
    chemical_scale_j_mol: float,
    radial_dimension: int,
    kinetic_beta_code: float,
    code_length_unit_nm: float,
    capillary_initial: float | None,
    interface_initial: float | None,
    metrics: SolverMetrics,
    compiled: bool = False,
) -> tuple[float, float]:
    capillary, cap_metrics = solve_capillary_root(
        branch, radius_nm, gamma_j_m2, molar_volume_m3_mol,
        radial_dimension, capillary_initial, compiled,
    )
    metrics.capillary.add(cap_metrics)
    kinetic_increment = (
        chemical_scale_j_mol * kinetic_beta_code
        * velocity / code_length_unit_nm
    )
    interface, kinetic_metrics = solve_kinetic_root(
        branch, capillary, kinetic_increment, interface_initial, compiled
    )
    metrics.kinetic.add(kinetic_metrics)
    return capillary, interface


def evolve_moving_interface_optimized(
    radius_initial: float,
    outer_radius: float,
    radial_nodes: np.ndarray,
    profile: np.ndarray,
    diffusivity: float,
    branch: ThermoBranch,
    gamma_j_m2: float,
    molar_volume_m3_mol: float,
    chemical_scale_j_mol: float,
    radial_dimension: int,
    kinetic_beta_code: float,
    code_length_unit_nm: float,
    final_time: float,
    dt: float,
    compiled_roots: bool = False,
    checkpoint_path: Path | None = None,
    checkpoint_interval_steps: int = 0,
    resume_checkpoint: Path | None = None,
    checkpoint_input_hash: str = "",
    stop_at_step: int | None = None,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, SolverMetrics]:
    """Reference ALE/CN update with optimized nonlinear interface solves."""
    if final_time <= 0.0 or dt <= 0.0:
        raise ValueError("final time and dt must be positive")
    nodes = profile.size - 1
    y = np.linspace(0.0, 1.0, nodes + 1)
    metrics = SolverMetrics.empty()
    if resume_checkpoint is None:
        unknown = profile[1:].copy()
        radius = radius_initial
        time_value = 0.0
        dr = (outer_radius - radius) / nodes
        capillary, surface, velocity, f1, f2 = coupled_surface_velocity_newton_v1(
            branch, radius, unknown, dr, diffusivity, gamma_j_m2,
            molar_volume_m3_mol, chemical_scale_j_mol, radial_dimension,
            kinetic_beta_code, code_length_unit_nm, None, profile[0], 0.0,
            metrics, compiled_roots,
        )
        initial_profile = np.concatenate(([surface], unknown))
        initial_inventory = reference.profile_inventory(
            radius, radial_nodes, initial_profile, radial_dimension
        )
        step = 0
        coupled_iterations = 0
        capillary_previous = capillary
        surface_previous = surface
        capillary_older = capillary
        surface_older = surface
        velocity_older = velocity
    else:
        saved = load_ale_checkpoint(resume_checkpoint, checkpoint_input_hash)
        saved_nodes = int(saved["nodes"])
        saved_outer = float(saved["outer_radius"])
        if saved_nodes != nodes or abs(saved_outer-outer_radius) > 1.0e-13:
            raise RuntimeError("oracle checkpoint geometry mismatch")
        unknown = np.asarray(saved["unknown"], dtype=np.float64)
        radius = float(saved["radius"])
        time_value = float(saved["time_value"])
        step = int(saved["step"])
        surface = float(saved["surface"])
        velocity = float(saved["velocity"])
        capillary = float(saved["capillary"])
        f1 = float(saved["f1"])
        f2 = float(saved["f2"])
        initial_inventory = float(saved["initial_inventory"])
        coupled_iterations = int(saved["coupled_iterations"])
        capillary_previous = float(saved["capillary_previous"])
        surface_previous = float(saved["surface_previous"])
        capillary_older = float(saved["capillary_older"])
        surface_older = float(saved["surface_older"])
        velocity_older = float(saved["velocity_older"])
    history: list[dict[str, float]] = []

    def append_history(step: int) -> None:
        current_nodes = radius + y * (outer_radius - radius)
        current_profile = np.concatenate(([surface], unknown))
        inventory = reference.profile_inventory(
            radius, current_nodes, current_profile, radial_dimension
        )
        history.append({
            "step": float(step),
            "time_code": time_value,
            "radius": radius,
            "surface_xB": surface,
            "capillary_xB": capillary,
            "velocity": velocity,
            "inventory": inventory,
            "inventory_error_rel": abs(inventory - initial_inventory)
            / max(abs(initial_inventory), 1.0e-300),
            "kinetic_boundary_residual": max(
                abs(f1) / chemical_scale_j_mol, abs(f2)
            ),
            "coupled_boundary_iterations": float(coupled_iterations),
        })

    append_history(step)
    while time_value < final_time - 0.5 * np.finfo(np.float64).eps:
        step += 1
        step_dt = min(dt, final_time - time_value)
        old_unknown = unknown.copy()
        old_radius = radius
        old_surface = surface
        old_velocity = velocity
        velocity_predictor = old_velocity + (old_velocity - velocity_older)
        radius_trial = old_radius + step_dt * velocity_predictor
        trial_unknown = old_unknown
        trial_velocity = velocity_predictor
        trial_capillary = capillary_previous + (
            capillary_previous - capillary_older
        )
        trial_surface = surface_previous + (surface_previous - surface_older)
        converged = False
        for coupled_iter in range(20):
            cap_guess, surface_guess = _surface_at_velocity(
                branch, radius_trial, trial_velocity, gamma_j_m2,
                molar_volume_m3_mol, chemical_scale_j_mol,
                radial_dimension, kinetic_beta_code, code_length_unit_nm,
                trial_capillary, trial_surface, metrics,
                compiled_roots,
            )
            radius_mid = 0.5 * (old_radius + radius_trial)
            velocity_mid = 0.5 * (old_velocity + trial_velocity)
            trial_unknown = reference._cn_step(
                old_unknown, old_surface, surface_guess, radius_mid,
                outer_radius, velocity_mid, diffusivity, step_dt,
                radial_dimension,
            )
            dr_trial = (outer_radius - radius_trial) / nodes
            cap_new, surface_new, velocity_new, f1, f2 = (
                coupled_surface_velocity_newton_v1(
                    branch, radius_trial, trial_unknown, dr_trial, diffusivity,
                    gamma_j_m2, molar_volume_m3_mol,
                    chemical_scale_j_mol, radial_dimension,
                    kinetic_beta_code, code_length_unit_nm,
                    cap_guess, surface_guess, trial_velocity, metrics,
                    compiled_roots,
                )
            )
            radius_new = old_radius + 0.5 * step_dt * (
                old_velocity + velocity_new
            )
            radius_ok = abs(radius_new - radius_trial) <= 1.0e-13 * max(
                1.0, abs(old_radius)
            )
            velocity_ok = abs(velocity_new - trial_velocity) <= 1.0e-11 * max(
                1.0, abs(velocity_new)
            )
            surface_ok = abs(surface_new - surface_guess) <= 1.0e-13 * max(
                1.0, abs(surface_new)
            )
            trial_capillary, trial_surface = cap_new, surface_new
            if radius_ok and velocity_ok and surface_ok:
                radius_trial = radius_new
                trial_velocity = velocity_new
                coupled_iterations = coupled_iter + 1
                metrics.ale_coupled_iterations += coupled_iterations
                converged = True
                break
            radius_trial = radius_new
            trial_velocity = velocity_new
        if not converged:
            raise RuntimeError(
                "optimized kinetic ALE/Stefan iteration did not converge at "
                f"step={step} radius={radius_trial:.17e} "
                f"velocity={trial_velocity:.17e}"
            )
        radius = radius_trial
        unknown = trial_unknown
        dr = (outer_radius - radius) / nodes
        capillary, surface, velocity, f1, f2 = coupled_surface_velocity_newton_v1(
            branch, radius, unknown, dr, diffusivity, gamma_j_m2,
            molar_volume_m3_mol, chemical_scale_j_mol, radial_dimension,
            kinetic_beta_code, code_length_unit_nm, trial_capillary,
            trial_surface, trial_velocity, metrics,
            compiled_roots,
        )
        capillary_older, surface_older, velocity_older = (
            capillary_previous, surface_previous, old_velocity
        )
        capillary_previous, surface_previous = capillary, surface
        time_value += step_dt
        should_checkpoint = (
            checkpoint_path is not None
            and checkpoint_interval_steps > 0
            and (
                step % checkpoint_interval_steps == 0
                or time_value >= final_time - 0.5 * step_dt
            )
        )
        if should_checkpoint:
            save_ale_checkpoint(
                checkpoint_path,
                {
                    "nodes": np.int64(nodes),
                    "outer_radius": np.float64(outer_radius),
                    "unknown": unknown,
                    "radius": np.float64(radius),
                    "time_value": np.float64(time_value),
                    "step": np.int64(step),
                    "surface": np.float64(surface),
                    "velocity": np.float64(velocity),
                    "capillary": np.float64(capillary),
                    "f1": np.float64(f1),
                    "f2": np.float64(f2),
                    "initial_inventory": np.float64(initial_inventory),
                    "coupled_iterations": np.int64(coupled_iterations),
                    "capillary_previous": np.float64(capillary_previous),
                    "surface_previous": np.float64(surface_previous),
                    "capillary_older": np.float64(capillary_older),
                    "surface_older": np.float64(surface_older),
                    "velocity_older": np.float64(velocity_older),
                },
                checkpoint_input_hash,
            )
        if step == 1 or time_value >= final_time - 0.5 * step_dt:
            append_history(step)
        if stop_at_step is not None and step >= stop_at_step:
            if history[-1]["step"] != float(step):
                append_history(step)
            break
    final_nodes = radius + y * (outer_radius - radius)
    final_profile = np.concatenate(([surface], unknown))
    return history, final_nodes, final_profile, metrics


def run_case_optimized(
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
    root_backend: str = "python",
    checkpoint_path: Path | None = None,
    checkpoint_interval_steps: int = 0,
    resume_checkpoint: Path | None = None,
    checkpoint_input_hash: str = "",
    stop_at_step: int | None = None,
) -> dict[str, object]:
    """Run the unchanged finite-box/ALE equations with optimized roots."""
    if root_backend not in ("python", "numba"):
        raise ValueError("root_backend must be python or numba")
    compiled_roots = root_backend == "numba"
    params = reference.parse_params(params_path)
    temperature_k = float(params["temperature_C"]) + 273.15
    diffusivity = float(params["D_alpha"])
    gamma = float(params["gamma_Jm2"])
    scale = float(params["mu_reference_scale"])
    molar_volume = unit.USER_PHYSICAL_INPUTS.Vm_compound
    beta_nominal = reference.finite_lphi_kinetic_beta_code(
        params, code_length_unit_nm
    )
    beta_phi = beta_nominal / l_phi_multiplier if finite_lphi_kinetics else 0.0
    branch = build_thermo_branch(temperature_k)
    outer_radius = reference.volume_equivalent_outer_radius(
        box_nm, radial_dimension
    )
    capillary, _ = solve_capillary_root(
        branch, radius_nm, gamma, molar_volume, radial_dimension,
        compiled=compiled_roots,
    )
    if prepared_capillary_state is None:
        radial_nodes, profile = reference.aged_no_flux_profile(
            radius_nm, outer_radius, diffusivity, capillary, far_x,
            diffusion_age_code, nodes, radial_dimension,
        )
        correction = 0.0
        if target_inventory is not None:
            profile, correction, matched = reference.add_far_inventory_correction(
                radius_nm, radial_nodes, profile, target_inventory, 1.2,
                radial_dimension,
            )
            if abs(matched - target_inventory) > 5.0e-12 * max(
                1.0, target_inventory
            ):
                raise RuntimeError("optimized sharp/PF inventory match failed")
        inventory_target = (
            float(target_inventory) if target_inventory is not None
            else reference.profile_inventory(
                radius_nm, radial_nodes, profile, radial_dimension
            )
        )
        capillary_state = {
            "radius_nm": radius_nm,
            "outer_radius_nm": outer_radius,
            "nodes": nodes,
            "radial_dimension": radial_dimension,
            "surface_initial_capillary": capillary,
            "radial_nodes": radial_nodes.copy(),
            "profile": profile.copy(),
            "inventory_target": inventory_target,
            "inventory_match_correction_xB": correction,
        }
    else:
        radial_nodes = np.asarray(
            prepared_capillary_state["radial_nodes"], dtype=np.float64
        ).copy()
        profile = np.asarray(
            prepared_capillary_state["profile"], dtype=np.float64
        ).copy()
        inventory_target = float(prepared_capillary_state["inventory_target"])
        correction = float(
            prepared_capillary_state["inventory_match_correction_xB"]
        )
        capillary_state = prepared_capillary_state
        if abs(float(prepared_capillary_state["surface_initial_capillary"])
               - capillary) > 2.0e-14:
            raise ValueError("optimized prepared capillary state mismatch")
    initial_metrics = SolverMetrics.empty()
    initial_dr = (outer_radius - radius_nm) / nodes
    _, initial_surface, initial_velocity, initial_f1, initial_f2 = (
        coupled_surface_velocity_newton_v1(
            branch, radius_nm, profile[1:], initial_dr, diffusivity,
            gamma, molar_volume, scale, radial_dimension, beta_phi,
            code_length_unit_nm, capillary, profile[0], 0.0,
            initial_metrics, compiled_roots,
        )
    )
    profile = profile.copy()
    profile[0] = initial_surface
    profile, kinetic_correction, matched = reference.add_far_inventory_correction(
        radius_nm, radial_nodes, profile, inventory_target, 1.2,
        radial_dimension,
    )
    if abs(matched - inventory_target) > 5.0e-12 * max(1.0, inventory_target):
        raise RuntimeError("optimized kinetic inventory rematch failed")
    history, final_nodes, final_profile, metrics = evolve_moving_interface_optimized(
        radius_nm, outer_radius, radial_nodes, profile, diffusivity, branch,
        gamma, molar_volume, scale, radial_dimension, beta_phi,
        code_length_unit_nm, final_time_code, sharp_dt_code,
        compiled_roots,
        checkpoint_path,
        checkpoint_interval_steps,
        resume_checkpoint,
        checkpoint_input_hash,
        stop_at_step,
    )
    metrics.capillary.add(initial_metrics.capillary)
    metrics.kinetic.add(initial_metrics.kinetic)
    metrics.coupled_newton_calls += initial_metrics.coupled_newton_calls
    metrics.coupled_newton_iterations += initial_metrics.coupled_newton_iterations
    metrics.coupled_line_search_reductions += initial_metrics.coupled_line_search_reductions
    metrics.coupled_fallback_count += initial_metrics.coupled_fallback_count
    return {
        "temperature_K": temperature_k,
        "radius_initial_nm": radius_nm,
        "radius_final_nm": history[-1]["radius"],
        "velocity_full_nm_per_code_time":
            (history[-1]["radius"] - radius_nm) / history[-1]["time_code"],
        "velocity_initial_nm_per_code_time": history[0]["velocity"],
        "velocity_final_nm_per_code_time": history[-1]["velocity"],
        "surface_xB_initial_capillary": capillary,
        "surface_xB_initial": history[0]["surface_xB"],
        "surface_xB_final": history[-1]["surface_xB"],
        "velocity_initial_kinetic_solve": initial_velocity,
        "kinetic_boundary_initial_residual": max(
            abs(initial_f1) / scale, abs(initial_f2)
        ),
        "max_kinetic_boundary_residual": max(
            abs(row["kinetic_boundary_residual"]) for row in history
        ),
        "finite_lphi_kinetics": bool(finite_lphi_kinetics),
        "l_phi_multiplier": l_phi_multiplier,
        "kinetic_beta_phi_nominal_code": beta_nominal,
        "kinetic_beta_phi_code": beta_phi,
        "radial_dimension": radial_dimension,
        "curvature_factor": radial_dimension - 1,
        "code_length_unit_nm": code_length_unit_nm,
        "outer_radius_nm": outer_radius,
        "nodes": nodes,
        "sharp_dt_code": sharp_dt_code,
        "final_time_code": final_time_code,
        "actual_final_time_code": history[-1]["time_code"],
        "initial_inventory": history[0]["inventory"],
        "final_inventory": history[-1]["inventory"],
        "max_inventory_error_rel": max(
            row["inventory_error_rel"] for row in history
        ),
        "inventory_match_correction_xB": correction,
        "kinetic_boundary_inventory_correction_xB": kinetic_correction,
        "radial_nodes_initial": radial_nodes,
        "profile_initial": profile,
        "radial_nodes_final": final_nodes,
        "profile_final": final_profile,
        "prepared_capillary_state": capillary_state,
        "history": history,
        "solver_metrics": metrics.as_dict(),
        "optimized_solver": "safeguarded_Newton_coupled_surface_velocity_v1",
        "root_backend": root_backend,
    }
