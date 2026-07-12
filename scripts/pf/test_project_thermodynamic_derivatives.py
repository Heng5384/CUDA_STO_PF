#!/usr/bin/env python3
"""Directional derivative checks for the executable equal-volume PF chemistry."""

from __future__ import annotations

import math

R = 8.31446261815324
T = 673.15
SCALE = 1.3779024e5
MU0 = -1.0
VB = 1.0


def h(phi: float) -> float:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def hp(phi: float) -> float:
    return 30.0 * phi**2 * (1.0 - phi) ** 2


def g0_a(t: float) -> float:
    pb = -10531.095 + 154.243182*t - 32.4913959*t*math.log(t) + 0.00154613*t*t + 8.05448e25*t**-9
    te = -10544.679 + 183.372894*t - 35.6687*t*math.log(t) + 0.01583435*t*t - 5.240417e-6*t**3 + 155015.0/t
    return -76063.2138 + 9.67716633*t + pb + te


def g0_b(t: float) -> float:
    ag = -7209.512 + 118.202013*t - 23.8463314*t*math.log(t) - 0.001790585*t*t - 3.98587e-7*t**3 - 12011.0/t
    te = -10544.679 + 183.372894*t - 35.6687*t*math.log(t) + 0.01583435*t*t - 5.240417e-6*t**3 + 155015.0/t
    return 3.0 * (-10128.93 - 12.645115*t + (2.0/3.0)*ag + (1.0/3.0)*te)


def f_alpha(x: float) -> float:
    lval = 41212.9 - 18.05*T
    return (1.0-x)*g0_a(T) + x*g0_b(T) + R*T*((1.0-x)*math.log(1.0-x) + x*math.log(x)) + lval*x*(1.0-x)


def mu_a(x: float) -> float:
    lval = 41212.9 - 18.05*T
    return (g0_a(T) + R*T*math.log(1.0-x) + lval*x*x) / SCALE


def mu_b(x: float) -> float:
    lval = 41212.9 - 18.05*T
    return (g0_b(T) + R*T*math.log(x) + lval*(1.0-x)**2) / SCALE


def energy(c: float, phi: float) -> float:
    hh = h(phi)
    x = (c - hh*VB) / (1.0-hh)
    return ((1.0-hh)*f_alpha(x)/SCALE + hh*MU0)


def central(fun, value: float, eps: float = 1.0e-7) -> float:
    return (fun(value+eps) - fun(value-eps)) / (2.0*eps)


def main() -> int:
    x, phi = 0.03, 0.31
    c = (1.0-h(phi))*x + h(phi)*VB
    mu_fd = central(lambda cc: energy(cc, phi), c)
    mu_kernel = mu_b(x) - mu_a(x)
    phase_fd = central(lambda pp: energy(c, pp), phi)
    phase_kernel = hp(phi) * (MU0 - mu_b(x))
    mu_err = abs(mu_fd-mu_kernel)
    phase_err = abs(phase_fd-phase_kernel)
    print(f"mu_fd={mu_fd:.17e}")
    print(f"mu_kernel={mu_kernel:.17e}")
    print(f"mu_abs_error={mu_err:.17e}")
    print(f"phase_fixed_C_fd={phase_fd:.17e}")
    print(f"phase_kernel={phase_kernel:.17e}")
    print(f"phase_abs_error={phase_err:.17e}")
    passed = mu_err <= 2.0e-8 and phase_err <= 2.0e-8
    print(f"thermodynamic_directional_derivative_tests={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
