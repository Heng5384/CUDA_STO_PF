#!/usr/bin/env python3
"""Numba nopython kernels for the validated Next4 nonlinear root path."""

from __future__ import annotations

import math

from numba import njit


@njit(cache=True)
def mu_b_jit(mu_b_base, rt, l0, x):
    return mu_b_base + rt * math.log(x) + l0 * (1.0 - x) ** 2


@njit(cache=True)
def mu_a_jit(mu_a_base, rt, l0, x):
    return mu_a_base + rt * math.log(1.0 - x) + l0 * x * x


@njit(cache=True)
def dmu_b_dx_jit(rt, l0, x):
    return rt / x - 2.0 * l0 * (1.0 - x)


@njit(cache=True)
def dmu_diff_dx_jit(rt, l0, x):
    return rt / x + rt / (1.0 - x) - 2.0 * l0


@njit(cache=True)
def capillary_root_jit(
    mu_b_base, mu_a_base, rt, l0, x_eq, target,
    lower, upper, initial, residual_scale,
):
    def residual(x):
        return (
            mu_b_jit(mu_b_base, rt, l0, x)
            - mu_a_jit(mu_a_base, rt, l0, x) - target
        )

    f_lower = residual(lower)
    f_upper = residual(upper)
    evaluations = 2
    if f_lower > 0.0 or f_upper < 0.0:
        return math.nan, 0, evaluations, 0, 1, math.nan
    x = min(max(initial, lower), upper)
    bisections = 0
    for iteration in range(1, 65):
        f = residual(x)
        evaluations += 1
        if abs(f) <= 5.0e-15 * residual_scale:
            return x, iteration, evaluations, bisections, 0, f
        if f < 0.0:
            lower = x
        else:
            upper = x
        slope = dmu_diff_dx_jit(rt, l0, x)
        candidate = x - f / slope if math.isfinite(slope) and slope > 0.0 else math.nan
        if not (lower < candidate < upper):
            candidate = 0.5 * (lower + upper)
            bisections += 1
        x = candidate
    return x, 64, evaluations, bisections, 2, residual(x)


@njit(cache=True)
def kinetic_root_jit(
    mu_b_base, rt, l0, capillary_x, kinetic_increment,
    lower, upper, initial, residual_scale,
):
    if kinetic_increment == 0.0:
        return capillary_x, 0, 0, 0, 0, 0.0
    target = mu_b_jit(mu_b_base, rt, l0, capillary_x) + kinetic_increment

    def residual(x):
        return mu_b_jit(mu_b_base, rt, l0, x) - target

    f_lower = residual(lower)
    f_upper = residual(upper)
    evaluations = 2
    if f_lower > 0.0 or f_upper < 0.0:
        return math.nan, 0, evaluations, 0, 1, math.nan
    x = min(max(initial, lower), upper)
    bisections = 0
    for iteration in range(1, 65):
        f = residual(x)
        evaluations += 1
        if abs(f) <= 5.0e-15 * residual_scale:
            return x, iteration, evaluations, bisections, 0, f
        if f < 0.0:
            lower = x
        else:
            upper = x
        slope = dmu_b_dx_jit(rt, l0, x)
        candidate = x - f / slope if math.isfinite(slope) and slope > 0.0 else math.nan
        if not (lower < candidate < upper):
            candidate = 0.5 * (lower + upper)
            bisections += 1
        x = candidate
    return x, 64, evaluations, bisections, 2, residual(x)


@njit(cache=True)
def coupled_surface_velocity_jit(
    mu_b_base, rt, l0, capillary_x, kinetic_slope,
    u0, u1, dr, diffusivity, x_initial, velocity_initial,
    x_lower, x_upper, chemical_scale,
):
    x = min(max(x_initial, x_lower), x_upper)
    velocity = velocity_initial
    line_reductions_total = 0
    mu_capillary = mu_b_jit(mu_b_base, rt, l0, capillary_x)
    f1 = math.nan
    f2 = math.nan
    for iteration in range(17):
        gradient = (-3.0*x + 4.0*u0 - u1)/(2.0*dr)
        f1 = mu_b_jit(mu_b_base, rt, l0, x) - mu_capillary - kinetic_slope*velocity
        f2 = velocity - diffusivity*gradient/(1.0-x)
        norm = max(abs(f1)/chemical_scale, abs(f2))
        if norm <= 5.0e-14:
            return x, velocity, iteration, line_reductions_total, 0, f1, f2
        j11 = dmu_b_dx_jit(rt, l0, x)
        j12 = -kinetic_slope
        dgradient_dx = -3.0/(2.0*dr)
        j21 = -diffusivity*(
            dgradient_dx/(1.0-x) + gradient/(1.0-x)**2
        )
        determinant = j11 - j12*j21
        if not math.isfinite(determinant) or abs(determinant) < 1.0e-30:
            return x, velocity, iteration, line_reductions_total, 1, f1, f2
        dx = (-f1 + j12*f2)/determinant
        dv = (-j11*f2 + j21*f1)/determinant
        damping = 1.0
        accepted = False
        for reduction in range(13):
            x_trial = x + damping*dx
            v_trial = velocity + damping*dv
            if x_lower < x_trial < x_upper:
                g_trial = (-3.0*x_trial + 4.0*u0 - u1)/(2.0*dr)
                f1_trial = (
                    mu_b_jit(mu_b_base, rt, l0, x_trial)
                    - mu_capillary - kinetic_slope*v_trial
                )
                f2_trial = v_trial - diffusivity*g_trial/(1.0-x_trial)
                trial_norm = max(abs(f1_trial)/chemical_scale, abs(f2_trial))
                if trial_norm < norm:
                    x = x_trial
                    velocity = v_trial
                    line_reductions_total += reduction
                    accepted = True
                    break
            damping *= 0.5
        if not accepted:
            return x, velocity, iteration+1, line_reductions_total, 2, f1, f2
    return x, velocity, 16, line_reductions_total, 3, f1, f2


def warmup_numba_kernels():
    """Compile every signature without changing any production state."""
    mu_b_jit(0.0, 5000.0, 30000.0, 0.01)
    mu_a_jit(0.0, 5000.0, 30000.0, 0.01)
    capillary_root_jit(
        0.0, 0.0, 5000.0, 30000.0, 0.008, -5000.0,
        0.008, 0.1, 0.009, 1.0e5,
    )
    kinetic_root_jit(
        0.0, 5000.0, 30000.0, 0.008, 0.1,
        0.008, 0.1, 0.0081, 1.0e5,
    )
    coupled_surface_velocity_jit(
        0.0, 5000.0, 30000.0, 0.008, 1000.0,
        0.0081, 0.0082, 0.01, 9.0, 0.008, 0.0,
        1.0e-12, 0.1, 1.0e5,
    )
