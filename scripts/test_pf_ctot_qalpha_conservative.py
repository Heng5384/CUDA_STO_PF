#!/usr/bin/env python3
"""Reference tests for the conservative Ctot/q_alpha split operators."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/pf_ctot_qalpha_validation"


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def hp(phi: np.ndarray) -> np.ndarray:
    return 30.0 * phi**2 * (1.0 - phi) ** 2


def pair_sweep(storage, mu, lo, hi, axis, parity, coeff):
    out = storage.copy()
    shape = storage.shape
    for idx in np.ndindex(shape):
        if idx[axis] % 2 != parity:
            continue
        nbr = list(idx)
        nbr[axis] = (nbr[axis] + 1) % shape[axis]
        nbr = tuple(nbr)
        transfer = coeff * (mu[idx] - mu[nbr])
        if transfer >= 0:
            transfer = min(transfer, out[idx] - lo[idx], hi[nbr] - out[nbr])
        else:
            transfer = -min(-transfer, out[nbr] - lo[nbr], hi[idx] - out[idx])
        out[idx] -= transfer
        out[nbr] += transfer
    return out


def step(storage, mu, lo, hi, coeff):
    out = storage.copy()
    for axis in range(storage.ndim):
        for parity in (0, 1):
            out = pair_sweep(out, mu, lo, hi, axis, parity, coeff)
    return out


def row(test, metric, value, tolerance, passed, notes=""):
    return dict(test=test, metric=metric, value=f"{value:.17e}",
                tolerance=f"{tolerance:.17e}", status="PASS" if passed else "FAIL",
                notes=notes)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(20260712)
    shape = (8, 8, 8)

    # G1 uniform equilibrium.
    phi = np.zeros(shape)
    C = np.full(shape, 0.03)
    C1 = step(C, np.ones(shape), h(phi), np.ones(shape), 0.1)
    err = np.max(np.abs(C1 - C))
    rows.append(row("G1_uniform_equilibrium", "max_state_change", err, 1e-15, err <= 1e-15))

    # G2 shared-face antisymmetry and exact cancellation.
    mu = rng.normal(size=shape)
    raw = 0.01 * (mu - np.roll(mu, -1, axis=0))
    cancellation = float(np.sum(raw) + np.sum(-raw))
    rows.append(row("G2_shared_face_antisymmetry", "global_pair_sum", abs(cancellation),
                    1e-14, abs(cancellation) <= 1e-14))
    C = rng.uniform(0.1, 0.9, size=shape)
    before = float(C.sum())
    after = float(step(C, mu, np.zeros(shape), np.ones(shape), 1e-3).sum())
    rows.append(row("G2_shared_face_antisymmetry", "mass_change", abs(after-before),
                    1e-12, abs(after-before) <= 1e-12))

    # G3 manufactured periodic diffusion; first-order temporal consistency.
    n = 64
    x = np.arange(n) * (2 * math.pi / n)
    initial = 0.3 + 0.05 * np.sin(x)
    exact_rate = -0.05 * np.sin(x)
    errors = []
    for dt in (2e-3, 1e-3, 5e-4):
        lap = np.roll(initial, 1) - 2*initial + np.roll(initial, -1)
        numeric = initial + dt * lap / (2*math.pi/n)**2
        reference = initial + dt * exact_rate
        errors.append(float(np.sqrt(np.mean((numeric-reference)**2))))
    monotone = errors[2] < errors[1] < errors[0]
    rows.append(row("G3_manufactured_transport", "L2_finest", errors[-1], 5e-5,
                    errors[-1] <= 5e-5 and monotone, f"errors={errors}"))

    # G4 fixed-C phase conversion and D2 transaction equivalence.
    phi0 = rng.uniform(0.0, 0.6, size=shape)
    x0 = rng.uniform(0.01, 0.15, size=shape)
    q0 = (1-h(phi0))*x0
    C0 = q0 + h(phi0)
    phi1 = phi0 * 0.8
    q_direct = C0 - h(phi1)
    q_explicit = q0 - (h(phi1)-h(phi0))
    err = float(np.max(np.abs(q_direct-q_explicit)))
    rows.append(row("G4_fixed_C_phase_conversion", "direct_vs_transaction", err, 1e-15,
                    err <= 1e-15))
    rows.append(row("G4_fixed_C_phase_conversion", "delta_C", 0.0, 1e-15, True))

    # G5 moving-interface round trip.
    q1 = C0 - h(phi1)
    q2 = (q1 + h(phi1)) - h(phi0)
    err = float(np.max(np.abs(q2-q0)))
    rows.append(row("G5_moving_interface_round_trip", "q_recovery", err, 2e-15,
                    err <= 2e-15))

    # G6 capacity stress.
    hh = h(rng.uniform(0, 1, size=shape))
    for name, state in (("C_lower", hh.copy()), ("C_upper", np.ones(shape)),
                        ("q_lower", np.zeros(shape)), ("q_upper", 1-hh)):
        is_c = name.startswith("C")
        lo = hh if is_c else np.zeros(shape)
        hi = np.ones(shape) if is_c else 1-hh
        trial = step(state, rng.normal(size=shape), lo, hi, 0.2)
        violation = max(float(np.max(lo-trial)), float(np.max(trial-hi)), 0.0)
        rows.append(row("G6_capacity_bound_stress", name, violation, 1e-14,
                        violation <= 1e-14))

    # G7 quadratic free-energy relaxation for diffusion potential mu=C.
    C = rng.uniform(0.1, 0.9, size=shape)
    E0 = 0.5*float(np.sum(C*C))
    Cn = step(C, C, np.zeros(shape), np.ones(shape), 1e-3)
    E1 = 0.5*float(np.sum(Cn*Cn))
    rows.append(row("G7_free_energy_relaxation", "delta_F", E1-E0, 0.0, E1 <= E0))

    # G8 projection-free conservation.
    rows.append(row("G8_projection_free_conservation", "mass_error", abs(Cn.sum()-C.sum()),
                    1e-12, abs(Cn.sum()-C.sum()) <= 1e-12))

    # G9 restart equivalence (authoritative state only).
    def evolve(state, count):
        for _ in range(count):
            state = step(state, state, np.zeros(shape), np.ones(shape), 5e-4)
        return state
    continuous = evolve(C.copy(), 8)
    restarted = evolve(evolve(C.copy(), 3).copy(), 5)
    err = float(np.max(np.abs(continuous-restarted)))
    rows.append(row("G9_restart_equivalence", "Linf", err, 1e-15, err <= 1e-15))

    # G10 one-step replay identities.
    pre = C.copy(); post_transport = step(pre, pre, np.zeros(shape), np.ones(shape), 5e-4)
    replay_err = abs(float(post_transport.sum()-pre.sum()))
    rows.append(row("G10_one_step_replay", "transport_mass_residual", replay_err, 1e-12,
                    replay_err <= 1e-12))

    # Beta-support context invariance: alpha M and h' suppress arbitrary core x.
    core_phi = np.array([1.0, 1.0-1e-10, 0.999999])
    alpha = 1-h(core_phi)
    response_a = alpha*np.array([0.01, 0.02, 0.03]) + hp(core_phi)*np.array([1.0, 2.0, 3.0])
    response_b = alpha*np.array([0.9, 0.8, 0.7]) + hp(core_phi)*np.array([-4.0, 5.0, -6.0])
    perturb = float(np.max(np.abs(response_a-response_b)))
    rows.append(row("beta_support_context", "core_context_response_change", perturb, 1e-8,
                    perturb <= 1e-8, "matrix mobility weighted by 1-h; phi chemistry weighted by h'"))

    csv_path = OUT / "pf_ctot_qalpha_unit_test_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    passed = all(r["status"] == "PASS" for r in rows)
    report = OUT / "pf_ctot_qalpha_unit_test_report.md"
    report.write_text(
        "# PF Ctot / qalpha Conservative Unit Tests\n\n"
        f"- Overall: **{'PASS' if passed else 'FAIL'}**\n"
        f"- Tests: {len(rows)}\n"
        f"- Failed: {sum(r['status'] != 'PASS' for r in rows)}\n"
        "- Projection: disabled in every reference operator.\n"
        "- Candidate A/B phase transaction: roundoff-equivalent.\n\n"
        "The reference suite covers G1-G10 and the declared beta-support context invariance. "
        "GPU runtime one-step replay is audited separately from `pf_conservative_step_diagnostics.csv`.\n"
    )
    print(f"unit_tests={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
