#!/usr/bin/env python3
"""Independent host-double Ji--Chen-style 1D reaction--diffusion oracle.

The state is the phase/reaction extent ``phi`` and matrix logit composition
``Y``.  The conserved quantity is reconstructed only for auditing:

    C = h(phi) + (1-h(phi))*x,  x = sigmoid(Y).

The direct Ji--Chen source identity is used for Y, so ``dC/dt = div(J)``
algebraically.  This module does not call, import, or fit a CUDA trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import csv
import math
from pathlib import Path
import sys
from typing import Iterable
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402
from scripts.prepare_jc4_research_model import parameter_row  # noqa: E402


def h_switch(phi: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def h_prime(phi: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return 30.0 * phi**2 * (1.0 - phi) ** 2


def g_prime(phi: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def sigmoid(logit: np.ndarray) -> np.ndarray:
    z = np.exp(-np.abs(logit))
    return np.where(logit >= 0.0, 1.0 / (1.0 + z), z / (1.0 + z))


def logit(fraction: float) -> float:
    if not 0.0 < fraction < 1.0:
        raise ValueError("composition must lie strictly inside (0,1)")
    return math.log(fraction / (1.0 - fraction))


@dataclass(frozen=True)
class OracleConfig:
    temperature_c: float = 400.0
    cells: int = 96
    dx_nm: float = 1.0
    lambda_nm: float = 4.0
    initial_beta_half_width_nm: float = 10.0
    initial_matrix_xB: float = 0.03
    lphi_ratio: float = 0.90
    final_time_code: float = 2.0
    output_points: int = 21
    rtol: float = 2.0e-11
    atol: float = 2.0e-13
    max_step_code: float = 0.02
    matrix_support_eps: float = 1.0e-5
    kkt_activation_tol: float = 1.0e-11

    @property
    def length_nm(self) -> float:
        return self.cells * self.dx_nm


class JiChen1DOracle:
    """Periodic planar slab oracle in project code units."""

    def __init__(self, config: OracleConfig):
        if config.cells < 16 or config.dx_nm <= 0.0:
            raise ValueError("oracle grid is too small or invalid")
        if config.initial_beta_half_width_nm <= config.lambda_nm:
            raise ValueError("beta slab must contain a resolved core")
        if 2.0 * config.initial_beta_half_width_nm >= config.length_nm - 4.0 * config.lambda_nm:
            raise ValueError("matrix region is too small for two independent interfaces")
        self.config = config
        row, _ = parameter_row(config.temperature_c)
        self.temperature_k = float(row["T_K"])
        self.x_eq = float(row["xB_eq"])
        self.W = float(row["W_code"])
        self.kappa = float(row["kappa_code"])
        self.D_alpha = float(row["D_alpha_code"])
        self.L_phi_diff = float(row["L_phi_diff_code"])
        self.L_phi = config.lphi_ratio * self.L_phi_diff
        self.mu_reference = float(row["W_physical_J_m3"]) * 4.1009e-5
        self.mu0_compound = self._mu_b(np.array([self.x_eq]))[0]
        self._jac_sparsity = self._build_jacobian_sparsity(config.cells)

    @staticmethod
    def _build_jacobian_sparsity(cells: int) -> csr_matrix:
        pattern = np.zeros((2 * cells, 2 * cells), dtype=bool)
        for i in range(cells):
            for offset in (-1, 0, 1):
                j = (i + offset) % cells
                pattern[i, j] = True
                pattern[i, cells + j] = True
                pattern[cells + i, j] = True
                pattern[cells + i, cells + j] = True
        return csr_matrix(pattern)

    def _mu_a(self, x: np.ndarray) -> np.ndarray:
        T = self.temperature_k
        L = unit.L0_PseudoBinary(T)
        return (
            unit.G_PbTe_Solid(T) + unit.R_GAS * T * np.log1p(-x) + L * x**2
        ) / self.mu_reference

    def _mu_b(self, x: np.ndarray) -> np.ndarray:
        T = self.temperature_k
        L = unit.L0_PseudoBinary(T)
        return (
            unit.G_Ag2Te_Solid(T) + unit.R_GAS * T * np.log(x)
            + L * (1.0 - x) ** 2
        ) / self.mu_reference

    def _g_second(self, x: np.ndarray) -> np.ndarray:
        T = self.temperature_k
        return (
            unit.R_GAS * T / (x * (1.0 - x))
            - 2.0 * unit.L0_PseudoBinary(T)
        ) / self.mu_reference

    def initial_state(self) -> np.ndarray:
        cfg = self.config
        coordinate = (np.arange(cfg.cells, dtype=np.float64) + 0.5) * cfg.dx_nm
        center = 0.5 * cfg.length_nm
        half = cfg.initial_beta_half_width_nm
        interface_half_parameter = 0.5 * cfg.lambda_nm
        phi = 0.5 * (
            np.tanh((coordinate - (center - half)) / interface_half_parameter)
            - np.tanh((coordinate - (center + half)) / interface_half_parameter)
        )
        Y = np.full(cfg.cells, logit(cfg.initial_matrix_xB), dtype=np.float64)
        return np.concatenate((phi, Y))

    def unpack(self, state: np.ndarray) -> dict[str, np.ndarray]:
        n = self.config.cells
        phi = np.asarray(state[:n], dtype=np.float64)
        Y = np.asarray(state[n:], dtype=np.float64)
        x = sigmoid(Y)
        h = h_switch(phi)
        # Evaluate the exact polynomial complement directly.  Forming 1-h
        # loses every active-matrix bit in a saturated beta core long before
        # h'(phi) vanishes; that numerical cancellation previously forced the
        # oracle to freeze such cells and created an artificial dissolution
        # stop at the initial ``alpha < matrix_support_eps`` contour.
        alpha = h_switch(1.0 - phi)
        with np.errstate(over="ignore", invalid="ignore"):
            C = h + alpha * x
        return {"phi": phi, "Y": Y, "x": x, "h": h, "alpha": alpha, "C": C}

    def differential_terms(self, state: np.ndarray) -> dict[str, np.ndarray]:
        cfg = self.config
        values = self.unpack(state)
        phi = values["phi"]
        x = values["x"]
        h = values["h"]
        alpha = values["alpha"]
        # Constitutive evaluation remains finite for solver trial states. The
        # accepted physical state itself is represented by an unbounded logit,
        # so this guard is not a post-step composition clip or projection.
        x_eval = np.clip(x, 1.0e-14, 1.0 - 1.0e-14)
        mu_a = self._mu_a(x_eval)
        mu_b = self._mu_b(x_eval)
        mu_exchange = mu_b - mu_a
        gamma = self._g_second(x_eval)
        mobility = np.maximum(alpha, 0.0) * self.D_alpha / gamma
        mobility_right = np.roll(mobility, -1)
        face_mobility = np.divide(
            2.0 * mobility * mobility_right,
            mobility + mobility_right,
            out=np.zeros_like(mobility),
            where=(mobility + mobility_right) > 1.0e-300,
        )
        face_flux = face_mobility * (
            np.roll(mu_exchange, -1) - mu_exchange
        ) / cfg.dx_nm
        div_flux = (face_flux - np.roll(face_flux, 1)) / cfg.dx_nm
        with np.errstate(over="ignore", invalid="ignore"):
            lap_phi = (
                np.roll(phi, -1) - 2.0 * phi + np.roll(phi, 1)
            ) / cfg.dx_nm**2
        with np.errstate(over="ignore", invalid="ignore"):
            phase_gradient = (
                self.W * g_prime(phi)
                + h_prime(phi) * (self.mu0_compound - mu_b)
                - self.kappa * lap_phi
            )
        phi_rate = -self.L_phi * phase_gradient
        denominator = alpha * x_eval * (1.0 - x_eval)
        Y_rate = np.zeros_like(phi)
        # In the one-sided limit, exact phi=1 has alpha=h'=0 and requires no
        # composition equation.  Every finite alpha remains part of the local
        # Ji--Chen reaction identity, including the diffuse beta-side tail.
        active = denominator > np.finfo(np.float64).tiny
        with np.errstate(over="ignore", invalid="ignore"):
            reaction_source = h_prime(phi) * (1.0 - x_eval) * phi_rate
        Y_rate[active] = (
            div_flux[active] - reaction_source[active]
        ) / denominator[active]
        with np.errstate(over="ignore", invalid="ignore"):
            C_rate_from_chain = reaction_source + denominator * Y_rate
        return {
            **values,
            "mu_exchange": mu_exchange,
            "mobility": mobility,
            "face_flux": face_flux,
            "div_flux": div_flux,
            "phase_gradient": phase_gradient,
            "phi_rate": phi_rate,
            "Y_rate": Y_rate,
            "reaction_source": reaction_source,
            "C_rate_from_chain": C_rate_from_chain,
        }

    def rhs(self, _time: float, state: np.ndarray) -> np.ndarray:
        terms = self.differential_terms(state)
        return np.concatenate((terms["phi_rate"], terms["Y_rate"]))

    def run(self) -> dict[str, object]:
        cfg = self.config
        initial = self.initial_state()
        times = np.linspace(0.0, cfg.final_time_code, cfg.output_points)
        # BDF finite-difference Jacobian probes can temporarily leave the
        # admissible state and overflow before the step is rejected. Accepted
        # states are checked below; suppress only those internal trial warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                    module=r"scipy\.integrate\._ivp\..*")
            solution = solve_ivp(
                self.rhs,
                (0.0, cfg.final_time_code),
                initial,
                method="BDF",
                t_eval=times,
                rtol=cfg.rtol,
                atol=cfg.atol,
                max_step=cfg.max_step_code,
                jac_sparsity=self._jac_sparsity,
            )
        if not solution.success:
            raise RuntimeError(solution.message)
        records = [self.metrics(solution.y[:, i], float(time))
                   for i, time in enumerate(solution.t)]
        return {"solution": solution, "records": records}

    def metrics(self, state: np.ndarray, time_code: float) -> dict[str, float]:
        values = self.unpack(state)
        terms = self.differential_terms(state)
        h_volume = float(np.sum(values["h"]) * self.config.dx_nm)
        matrix_weight = float(np.sum(values["alpha"]))
        matrix_mean = float(np.sum(values["alpha"] * values["x"]) / matrix_weight)
        source_residual = terms["C_rate_from_chain"] - terms["div_flux"]
        return {
            "time_code": time_code,
            "time_s": time_code * float(parameter_row(self.config.temperature_c)[0]["time_scale_s"]),
            "h_volume_nm": h_volume,
            "beta_half_width_nm": 0.5 * h_volume,
            "matrix_mean_xB": matrix_mean,
            "matrix_min_xB": float(np.min(values["x"][values["alpha"] >= self.config.matrix_support_eps])),
            "matrix_max_xB": float(np.max(values["x"][values["alpha"] >= self.config.matrix_support_eps])),
            "total_C_integral": float(np.sum(values["C"]) * self.config.dx_nm),
            "phase_min": float(np.min(values["phi"])),
            "phase_max": float(np.max(values["phi"])),
            "source_identity_Linf": float(np.max(np.abs(source_residual))),
            "domain_divergence_sum": float(np.sum(terms["div_flux"]) * self.config.dx_nm),
        }


class FixedCtotJiChen1DOracle(JiChen1DOracle):
    """Independent host-double long-time realization in ``(phi,Ctot)``.

    The continuous model is exactly the direct Ji--Chen source identity used
    above, but the linear conserved state is integrated explicitly:

        Ctot_t = div(J_B),  x=(Ctot-h(phi))/(1-h(phi)).

    A local tangent-cone condition enforces the stoichiometric storage bound
    ``h(phi)<=Ctot``.  This is the host analogue of a phase KKT condition; it
    is neither a composition clip nor a domain-wide mass projection.
    """

    def initial_state(self) -> np.ndarray:
        direct = super().initial_state()
        values = super().unpack(direct)
        return np.concatenate((values["phi"].copy(), values["C"].copy()))

    def unpack(self, state: np.ndarray) -> dict[str, np.ndarray]:
        n = self.config.cells
        phi = np.asarray(state[:n], dtype=np.float64)
        C = np.asarray(state[n:], dtype=np.float64)
        h = h_switch(phi)
        alpha = h_switch(1.0 - phi)
        x = np.ones_like(phi)
        active = alpha > np.finfo(np.float64).tiny
        x[active] = (C[active] - h[active]) / alpha[active]
        x_eval = np.clip(x, 1.0e-300, 1.0 - 1.0e-15)
        Y = np.log(x_eval) - np.log1p(-x_eval)
        return {"phi": phi, "Y": Y, "x": x, "h": h,
                "alpha": alpha, "C": C}

    def differential_terms(self, state: np.ndarray) -> dict[str, np.ndarray]:
        cfg = self.config
        values = self.unpack(state)
        phi = values["phi"]
        C = values["C"]
        h = values["h"]
        alpha = values["alpha"]
        x = values["x"]
        x_eval = np.clip(x, 1.0e-14, 1.0 - 1.0e-14)
        mu_a = self._mu_a(x_eval)
        mu_b = self._mu_b(x_eval)
        mu_exchange = mu_b - mu_a
        gamma = self._g_second(x_eval)
        mobility = np.maximum(alpha, 0.0) * self.D_alpha / gamma
        mobility_right = np.roll(mobility, -1)
        face_mobility = np.divide(
            2.0 * mobility * mobility_right,
            mobility + mobility_right,
            out=np.zeros_like(mobility),
            where=(mobility + mobility_right) > 1.0e-300,
        )
        face_flux = face_mobility * (
            np.roll(mu_exchange, -1) - mu_exchange
        ) / cfg.dx_nm
        div_flux = (face_flux - np.roll(face_flux, 1)) / cfg.dx_nm
        lap_phi = (
            np.roll(phi, -1) - 2.0 * phi + np.roll(phi, 1)
        ) / cfg.dx_nm**2
        phase_gradient = (
            self.W * g_prime(phi)
            + h_prime(phi) * (self.mu0_compound - mu_b)
            - self.kappa * lap_phi
        )
        phi_rate = -self.L_phi * phase_gradient
        hp = h_prime(phi)
        C_rate = div_flux
        storage_slack = C - h
        storage_rate_free = C_rate - hp * phi_rate
        active_upper = (
            (storage_slack <= cfg.kkt_activation_tol)
            & (storage_rate_free < 0.0)
            & (hp > np.finfo(np.float64).tiny)
        )
        phi_rate[active_upper] = C_rate[active_upper] / hp[active_upper]
        phi_rate[(phi <= cfg.kkt_activation_tol) & (phi_rate < 0.0)] = 0.0
        phi_rate[(phi >= 1.0 - cfg.kkt_activation_tol) & (phi_rate > 0.0)] = 0.0
        storage_rate = C_rate - hp * phi_rate
        return {
            **values,
            "mu_exchange": mu_exchange,
            "mobility": mobility,
            "face_flux": face_flux,
            "div_flux": div_flux,
            "phase_gradient": phase_gradient,
            "phi_rate": phi_rate,
            "C_rate": C_rate,
            "storage_slack": storage_slack,
            "storage_rate": storage_rate,
            "active_upper": active_upper,
            "reaction_source": hp * (1.0 - x_eval) * phi_rate,
            "C_rate_from_chain": C_rate,
        }

    def rhs(self, _time: float, state: np.ndarray) -> np.ndarray:
        terms = self.differential_terms(state)
        return np.concatenate((terms["phi_rate"], terms["C_rate"]))

    def metrics(self, state: np.ndarray, time_code: float) -> dict[str, float]:
        row = super().metrics(state, time_code)
        values = self.unpack(state)
        slack = values["C"] - values["h"]
        row.update({
            "storage_slack_min": float(np.min(slack)),
            "C_min": float(np.min(values["C"])),
            "C_max": float(np.max(values["C"])),
            "accepted_bound_violation_Linf": float(max(
                0.0,
                -float(np.min(values["phi"])),
                float(np.max(values["phi"])) - 1.0,
                -float(np.min(values["C"])),
                float(np.max(values["C"])) - 1.0,
                -float(np.min(slack)),
            )),
        })
        return row


def run_limiting_suite() -> list[dict[str, object]]:
    cases = [
        ("diffusion_controlled_growth", 0.90, 0.03, 1.0, "growth"),
        ("reaction_controlled_growth", 0.05, 0.03, 1.0, "growth"),
        ("mixed_controlled_growth", 0.30, 0.03, 1.0, "growth"),
        ("dissolution", 0.90, 0.003, 1.0, "dissolution"),
        ("finite_box_equilibrium", 0.90, unit.xAg2Te_eq_from_T(673.15), 0.2,
         "stationary"),
    ]
    rows: list[dict[str, object]] = []
    for case, ratio, x0, final_time, expected in cases:
        cfg = OracleConfig(
            cells=64,
            initial_beta_half_width_nm=8.0,
            initial_matrix_xB=x0,
            lphi_ratio=ratio,
            final_time_code=final_time,
            output_points=11,
            max_step_code=0.01,
        )
        oracle = JiChen1DOracle(cfg)
        result = oracle.run()
        first = result["records"][0]
        last = result["records"][-1]
        displacement = float(last["beta_half_width_nm"] - first["beta_half_width_nm"])
        direction = (
            "growth" if displacement > 1.0e-3
            else "dissolution" if displacement < -1.0e-3
            else "stationary"
        )
        mass_scale = max(abs(float(first["total_C_integral"])), 1.0)
        rows.append({
            "case": case,
            "control_regime": expected,
            "Lphi_ratio": ratio,
            "initial_xB": x0,
            "xB_eq": oracle.x_eq,
            "initial_half_width_nm": first["beta_half_width_nm"],
            "final_half_width_nm": last["beta_half_width_nm"],
            "displacement_nm": displacement,
            "observed_direction": direction,
            "direction_pass": direction == expected,
            "mass_error_rel": abs(
                float(last["total_C_integral"] - first["total_C_integral"])
            ) / mass_scale,
            "source_identity_Linf": max(
                float(record["source_identity_Linf"])
                for record in result["records"]
            ),
            "domain_divergence_sum_max": max(
                abs(float(record["domain_divergence_sum"]))
                for record in result["records"]
            ),
            "status": "PASS" if direction == expected else "FAIL",
        })
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-root", type=Path,
        default=ROOT / "reports/pf_ctot_production_candidate",
    )
    args = parser.parse_args()
    rows = run_limiting_suite()
    report_root = args.report_root.resolve()
    write_csv(report_root / "jc4_ji_chen_oracle_metrics.csv", rows)
    status = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    table = "\n".join(
        f"| {row['case']} | {row['Lphi_ratio']:.2f} | "
        f"{row['displacement_nm']:.6g} | {row['observed_direction']} | "
        f"{row['mass_error_rel']:.3e} | {row['status']} |" for row in rows
    )
    (report_root / "jc4_ji_chen_oracle.md").write_text(f"""# JC4 independent Ji--Chen 1D oracle

## Equation and independence

The host-double oracle evolves `phi` and `Y=logit(xB_alpha)` with diagonal
finite-`Lphi` reaction kinetics and one-sided matrix diffusion. It enforces the
direct source identity

`(1-h)x(1-x) Y_t = div(M grad(mu_B-mu_A)) - h'(1-x) phi_t`,

which gives `C_B_tot,t = div(M grad(mu_B-mu_A))` algebraically. It neither calls
the CUDA runtime nor uses a runtime trajectory as input. The PbTe--Ag2Te
regular-solution thermodynamics, strict `D_beta=0` Ji--Chen limit, `W`, `kappa`,
and `D_alpha` are instantiated from the project physical mapping.

| case | Lphi/Ldiff | displacement (nm) | direction | mass error | status |
|---|---:|---:|---|---:|---|
{table}

The equilibrium row is a finite periodic box initialized at planar coexistence;
its purpose is to verify the stationary limiting state rather than a growth
trajectory. Overall oracle status: **{status}**.
""", encoding="utf-8")
    print(f"Ji_Chen_1D_oracle_status={status}")
    for row in rows:
        print(f"{row['case']}={row['status']} displacement_nm={row['displacement_nm']:.12e}")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
