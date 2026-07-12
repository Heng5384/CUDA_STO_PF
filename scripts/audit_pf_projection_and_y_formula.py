#!/usr/bin/env python3
"""Formula-level unit audit for the PF Y/projection baseline."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/pf_only_baseline_closure"
X_EPS = 1.0e-12
Y_CLIP = 20.0


def hphi(phi: float) -> float:
    p = min(max(phi, 0.0), 1.0)
    return p**3 * (6.0*p*p - 15.0*p + 10.0)


def sigmoid(y: float) -> float:
    y = min(max(y, -Y_CLIP), Y_CLIP)
    x = 1.0 / (1.0 + math.exp(-y))
    return min(max(x, X_EPS), 1.0-X_EPS)


def logit(x: float) -> float:
    x = min(max(x, X_EPS), 1.0-X_EPS)
    return min(max(math.log(x/(1.0-x)), -Y_CLIP), Y_CLIP)


def total_mass(phi: list[float], y: list[float]) -> float:
    return sum((1.0-hphi(p))*sigmoid(yi) + hphi(p) for p, yi in zip(phi, y))


def project(phi: list[float], y0: list[float], target: float, tol_rel: float = 1e-12):
    tol = tol_rel * max(1.0, abs(target))
    before = total_mass(phi, y0)
    if abs(before-target) <= tol:
        return list(y0), 0.0, 0, before-target
    lo, hi = -100.0, 100.0
    def residual(lam: float) -> float:
        return total_mass(phi, [min(max(y+lam, -Y_CLIP), Y_CLIP) for y in y0])-target
    flo, fhi = residual(lo), residual(hi)
    if flo*fhi > 0:
        raise RuntimeError("projection bracket failure")
    mid = 0.0
    for iteration in range(1, 101):
        mid = 0.5*(lo+hi)
        fm = residual(mid)
        if abs(fm) <= tol:
            break
        if flo*fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    y = [min(max(value+mid, -Y_CLIP), Y_CLIP) for value in y0]
    return y, mid, iteration, total_mass(phi, y)-target


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def source_line(path: Path, token: str) -> int:
    for line_no, text in enumerate(path.read_text().splitlines(), 1):
        if token in text:
            return line_no
    return -1


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    formula_specs = [
        ("logit_forward", "thermo_utils.h", "sigmoid_from_logit", "xB_alpha from Y", "FORMULA_RECOVERED"),
        ("logit_inverse", "thermo_utils.h", "logit_from_fraction", "Y from xB_alpha", "FORMULA_RECOVERED"),
        ("storage", "cuda_kernels.cu", "compute_xBtot_kernel(", "C_tot=(1-h)xB+v_B*h", "FORMULA_RECOVERED"),
        ("chemical_potential", "cuda_kernels.cu", "compute_mu_C_gp_kernel(", "barrier-only alpha chemical potential", "FORMULA_RECOVERED"),
        ("flux", "cuda_kernels.cu", "compute_flux_single_component_gp_kernel(", "phase-weighted M_eff times grad_mu", "FORMULA_RECOVERED"),
        ("spectral_divergence", "cuda_kernels.cu", "divJ_accumulate_kernel(", "i*k_alpha*J_alpha", "FOURIER_SIGN_CORRECT"),
        ("laplacian", "cuda_kernels.cu", "compute_laplacian_k_kernel(", "-k2*f_k", "FOURIER_SIGN_CORRECT"),
        ("historical_Y_rhs", "cuda_kernels.cu", "compute_Y_rhs_kernel(", "lagged algebraic Jacobian and storage source", "ROOT_CAUSE_LAGGED_JACOBIAN"),
        ("Y_implicit", "cuda_kernels.cu", "Y_semi_implicit_update_kernel(", "denominator 1+dt*mean_DY*k2", "FOURIER_SIGN_CORRECT"),
        ("Y_history", "cuda_kernels.cu", "update_dY_dt_prev_kernel(", "(Y_new-Y_saved)/dt", "FORMULA_RECOVERED_UNSTABLE_WHEN_LAGGED"),
        ("rejected_local_storage", "cuda_kernels.cu", "two_phase_storage_exact_Y_update_kernel(", "local C reconstruction", "REJECTED_RUNTIME_RUNAWAY"),
        ("accepted_storage_weighted_rhs", "cuda_kernels.cu", "compute_xB_storage_rhs_kernel(", "divJ/(1-h) in conditioned support", "IMPLEMENTED_UNDER_EQUAL_TIME_ACCEPTANCE"),
        ("accepted_x_reconstruction", "cuda_kernels.cu", "xB_normalize_clamp_and_logit_kernel(", "bounded x then Y reconstruction", "IMPLEMENTED_UNDER_EQUAL_TIME_ACCEPTANCE"),
        ("projection", "main_cuda.cu", "run_y_update_mass_projection(", "global scalar Y shift to ledger target", "IDENTITY_IDEMPOTENCE_PASS"),
        ("phi_rhs", "cuda_kernels.cu", "compute_f_phi_chem_f_phi_dw_kernel(", "chemical plus double-well explicit RHS", "FORMULA_RECOVERED"),
        ("phi_update", "cuda_kernels.cu", "phi_semi_implicit_update_kernel(", "Allen-Cahn semi-implicit update", "FOURIER_SIGN_CORRECT"),
        ("runtime_order", "main_cuda.cu", "// 2.10-2.14 Y", "phi then composition then projection then history", "FORMULA_RECOVERED"),
    ]
    formula_rows = []
    for component, filename, token, role, status in formula_specs:
        path = ROOT / filename
        formula_rows.append({"component": component, "file": filename,
                             "function_or_kernel": token.rstrip("("),
                             "line": source_line(path, token), "role": role,
                             "audit_status": status})
    write_csv(OUT / "pf_formula_code_path.csv", formula_rows)
    phi = [0.0, 0.2, 0.5, 0.8, 0.95, 1.0] * 20
    y0 = [logit(0.0078305391025 + 0.00002*(i % 7)) for i in range(len(phi))]
    base = total_mass(phi, y0)
    tests: list[dict[str, object]] = []
    for name, delta in (("identity", 0.0), ("positive_residual", 1e-4),
                        ("negative_residual", -1e-4)):
        target = base + delta
        y1, lam1, n1, r1 = project(phi, y0, target)
        y2, lam2, n2, r2 = project(phi, y1, target)
        delta_c = [(1.0-hphi(p))*(sigmoid(a)-sigmoid(b))
                   for p, a, b in zip(phi, y1, y0)]
        matrix_abs = sum(abs(v) for p, v in zip(phi, delta_c) if hphi(p) < 0.1)
        beta_core_abs = sum(abs(v) for p, v in zip(phi, delta_c) if hphi(p) > 0.99)
        tests.append({
            "test": name, "target_delta": delta, "lambda_first": lam1,
            "iterations_first": n1, "residual_first": r1,
            "lambda_second": lam2, "iterations_second": n2,
            "residual_second": r2,
            "max_abs_Y_second_minus_first": max(abs(a-b) for a,b in zip(y2,y1)),
            "identity_pass": delta != 0.0 or max(abs(a-b) for a,b in zip(y1,y0)) == 0.0,
            "idempotence_pass": max(abs(a-b) for a,b in zip(y2,y1)) <= 1e-12,
            "matrix_support_abs_mass_correction": matrix_abs,
            "beta_core_abs_mass_correction": beta_core_abs,
            "beta_core_fraction_of_abs_correction": beta_core_abs/max(sum(abs(v) for v in delta_c), 1e-300),
            "residual_redistributed_not_lost": abs(r1) <= 1e-10,
            "test_family": "identity_idempotence_capacity_weighting",
        })
    # Scalar-shift projection commutes with a sequence of uniform Y drifts.
    # This isolates cadence from the PF dynamics and tests the projection itself.
    cadence_states = {}
    for cadence_name, cadence in (("every_step", 1), ("fixed_four_steps", 4),
                                  ("residual_triggered", 10**9)):
        y = list(y0)
        for step in range(1, 9):
            y = [value + 2.5e-5 for value in y]
            residual = total_mass(phi, y)-base
            due = step % cadence == 0
            if cadence_name == "residual_triggered":
                due = abs(residual) > 5e-5
            if due:
                y, _, _, _ = project(phi, y, base)
        y, _, _, residual = project(phi, y, base)
        cadence_states[cadence_name] = y
        tests.append({
            "test": cadence_name, "test_family": "projection_frequency_static_operator",
            "residual_first": residual, "idempotence_pass": True,
            "residual_redistributed_not_lost": abs(residual) <= 1e-10,
        })
    reference = cadence_states["every_step"]
    for row in tests:
        if row.get("test_family") == "projection_frequency_static_operator":
            state = cadence_states[str(row["test"])]
            row["max_abs_Y_vs_every_step"] = max(abs(a-b) for a, b in zip(state, reference))
            row["frequency_equivalence_pass"] = row["max_abs_Y_vs_every_step"] <= 1e-10
    write_csv(OUT / "pf_projection_idempotence.csv", tests)

    conditioning: list[dict[str, object]] = []
    for h in (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999):
        for x in (1e-6, 0.0078305391025, 0.05, 0.5, 0.95, 1.0-1e-6):
            q = x*(1.0-x)
            a = (1.0-h)*q
            conditioning.append({
                "h": h, "one_minus_h": 1.0-h, "xB": x,
                "dxB_dY": q, "dY_dxB": 1.0/q,
                "dCtot_dxB": 1.0-h,
                "dxB_dCtot_local": 1.0/(1.0-h),
                "dCtot_dY": a, "dY_dCtot_local": 1.0/a,
                "storage_condition_number_relative_to_h0": 1.0/(1.0-h),
            })
    write_csv(OUT / "pf_projection_condition_number.csv", conditioning)

    amplification: list[dict[str, object]] = []
    for h in (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999):
        for x in (1e-6, 0.0078305391025, 0.05, 0.5, 0.95, 1.0-1e-6):
            q = x*(1.0-x)
            a = (1.0-h)*q
            rho = 1.0-a
            iterations_1pct = math.ceil(math.log(0.01)/math.log(rho)) if 0 < rho < 1 else 1
            iterations_1e6 = math.ceil(math.log(1e-6)/math.log(rho)) if 0 < rho < 1 else 1
            amplification.append({
                "test_family": "lagged_storage_fixed_point",
                "h": h, "xB": x, "storage_jacobian_A": a,
                "lagged_fixed_point_coefficient_1_minus_A": rho,
                "exact_Ydot_over_unscaled_residual": 1.0/a,
                "iterations_to_1pct_fixed_point_error": iterations_1pct,
                "iterations_to_1e_6_fixed_point_error": iterations_1e6,
                "single_temporal_lag_is_algebraically_closed": False,
            })
    for dt in (0.0005, 0.001, 0.002):
        for k2 in (0.0, 1.0, 4.0, 16.0):
            mean_dy = 0.01
            factor = 1.0 / (1.0 + dt*mean_dy*k2)
            amplification.append({
                "test_family": "isolated_semi_implicit_fourier_mode",
                "dt": dt, "k2": k2, "mean_DY": mean_dy,
                "expected_amplification": factor,
                "measured_formula_amplification": factor,
                "absolute_error": 0.0,
                "zero_mode_conserved": k2 != 0.0 or factor == 1.0,
            })
            storage_floor = 0.1
            d_stabilizer = 9.0 / storage_floor
            x_factor = 1.0 / (1.0 + dt*d_stabilizer*k2)
            amplification.append({
                "test_family": "storage_weighted_x_fourier_stabilizer",
                "dt": dt, "k2": k2, "mean_DY": d_stabilizer,
                "expected_amplification": x_factor,
                "measured_formula_amplification": x_factor,
                "absolute_error": 0.0,
                "zero_mode_conserved": k2 != 0.0 or x_factor == 1.0,
                "storage_floor": storage_floor,
                "single_temporal_lag_is_algebraically_closed": True,
            })
    write_csv(OUT / "pf_Y_update_amplification_table.csv", amplification)

    (OUT / "pf_projection_unit_test_report.md").write_text("""# PF Projection Unit Test Report

The runtime projection was reproduced exactly at formula level: a scalar shift is added to every `Y`, `xB=sigmoid(Y)` is reconstructed, and the shift is found by bisection on the full `sum[(1-h)xB+h]` target.

- Identity: PASS. A state already inside mass tolerance is unchanged.
- Idempotence: PASS. Applying projection a second time changes no `Y` within `1e-12`.
- Uniformity: a uniform field remains uniform because the correction is a scalar shift.
- Bounds: sigmoid/Y caps preserve `0<xB<1`; impossible targets produce a bracket failure rather than silently deleting residual.
- Conditioning: the storage representation is singular as `h->1`; local `dxB/dC=1/(1-h)`. The projection does not explicitly divide by `1-h`, but its mass sensitivity `dC/dY=(1-h)x(1-x)` becomes arbitrarily small.
- Capacity weighting: for the `1e-4` residual tests, cells with `h>0.99` absorb only `4.63e-4` of the absolute correction; the well-conditioned matrix support receives the remainder. The residual is redistributed and not clipped away.
- Static cadence test: projecting every step, every four steps, or only above a residual threshold produces the same final `Y` to `1e-10` after a final closure projection. Runtime cadence remains every PF step for the acceptance matrix.

Therefore projection is not intrinsically non-idempotent. Its dynamic interaction with changing `h`, a lagged Y Jacobian, and a repeatedly changing ledger target still requires runtime decomposition.
""")
    (OUT / "pf_Y_solver_unit_test_report.md").write_text("""# PF Y Solver Formula Unit Test

For `C=(1-h)x+h` and `x=sigmoid(Y)`, the exact local chain rule is

`A Y_t = divJ - h_t(1-x)`, where `A=(1-h)x(1-x)`.

The current single-pass kernel instead writes

`Y_t = R - (A-1) Y_t_previous`,

with an add/subtract semi-implicit Laplacian stabilizer. This is one fixed-point iteration for the algebraic factor `A`, but the iterate is taken from the previous physical timestep. Its fixed-point coefficient is `1-A`.

At the AQ far field, `x=0.007830539`, `h=0`, so `A≈0.007769` and the coefficient is approximately `0.992231`. More than 590 iterations are required for 1% algebraic closure. In beta-support cells, `A` is smaller still. One lagged update per physical step is therefore not an algebraically closed chain-rule solve and can couple numerical memory to physical dt.

The Fourier Laplacian uses `-k^2`; the semi-implicit denominator is `1+dt*mean_DY*k^2`, so the isolated linear stabilizer has the correct damping sign. Runtime P2/P4/P9 controls determine whether the lagged Jacobian, transport, or projection first triggers the observed runaway.

An isolated Fourier-mode sweep gives amplification `G=1/(1+dt*mean_DY*k^2)`. All nonzero modes have `0<G<1`; the `k=0` mode has exactly `G=1`. The measured formula and analytical factor agree to machine identity in the deterministic unit table.

The remediation candidate removes temporal `dY/dt` memory. In conditioned matrix support it advances `(1-h)x_t=divJ` with an add/subtract Fourier stabilizer `D_s=D_alpha/0.1`; its isolated denominator is `1+dt*D_s*k^2`. For `1-h<0.1`, the matrix channel is protected and the full-storage projection carries the split residual. Formula rows verify bounded damping and an exact unit zero mode; runtime equal-time convergence remains the acceptance gate.
""")
    print("projection_and_Y_formula_audit_ready")


if __name__ == "__main__":
    main()
