#!/usr/bin/env python3
"""Finite-domain planar diffusion-controlled sharp-interface oracle.

The oracle solves one half of the periodic planar slab.  Matrix diffusion is
one-sided, the outer symmetry boundary is no-flux, the interface composition
is fixed at matrix equilibrium, and the interface speed obeys Stefan balance.
The ALE finite-volume state is conservative by construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402
from scripts.prepare_one_sided_planar_benchmark import (  # noqa: E402
    sharp_similarity_parameter,
)


REPORT_ROOT = ROOT / "reports/pf_ctot_production_candidate"


def h_switch(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def pf_half_inventory(
    temperature_c: float,
    domain_nm: float,
    dx_nm: float,
    matrix_x: float,
    start_time_s: float,
) -> float:
    """Exact cell-sum inventory of the accepted planar initializer per half box."""
    nx = int(round(domain_nm / dx_nm))
    if nx % 2 or abs(nx * dx_nm - domain_nm) > 1.0e-12:
        raise ValueError("domain/dx must produce an even exact grid")
    x = np.arange(nx, dtype=np.float64) * dx_nm
    left = 0.35 * domain_nm
    right = 0.65 * domain_nm
    lambda_nm = 0.6
    phi = 0.5 * (
        np.tanh((x - left) / (0.5 * lambda_nm))
        - np.tanh((x - right) / (0.5 * lambda_nm))
    )
    phi = np.clip(phi, 0.0, 1.0)
    temperature_k = temperature_c + 273.15
    x_eq = unit.xAg2Te_eq_from_T(temperature_k)
    eta = sharp_similarity_parameter(matrix_x, x_eq)
    diffusion_length = math.sqrt(
        unit.D_Ag_in_PbTe_m2_per_s(temperature_k) * start_time_s
    ) * 1.0e9
    matrix = np.full(nx, x_eq, dtype=np.float64)
    for index, coordinate in enumerate(x):
        if coordinate < left:
            distance = left - coordinate
        elif coordinate > right:
            distance = coordinate - right
        else:
            continue
        coordinate_eta = eta + distance / (2.0 * diffusion_length)
        normalized = (
            math.erf(coordinate_eta) - math.erf(eta)
        ) / math.erfc(eta)
        matrix[index] = x_eq + (matrix_x - x_eq) * normalized
    total = (1.0 - h_switch(phi)) * matrix + h_switch(phi)
    return float(total.sum() * dx_nm / 2.0)


def similarity_profile(
    distance_nm: np.ndarray,
    temperature_c: float,
    matrix_x: float,
    start_time_s: float,
) -> np.ndarray:
    temperature_k = temperature_c + 273.15
    x_eq = unit.xAg2Te_eq_from_T(temperature_k)
    diffusivity_nm2_s = unit.D_Ag_in_PbTe_m2_per_s(temperature_k) * 1.0e18
    eta = sharp_similarity_parameter(matrix_x, x_eq)
    coordinate = eta + distance_nm / (
        2.0 * math.sqrt(diffusivity_nm2_s * start_time_s)
    )
    normalized = (np.vectorize(math.erf)(coordinate) - math.erf(eta)) / math.erfc(eta)
    return x_eq + (matrix_x - x_eq) * normalized


class PlanarSharpOracle:
    """Conservative moving-grid finite-volume Stefan solver."""

    def __init__(
        self,
        temperature_c: float = 400.0,
        domain_nm: float = 19.2,
        matrix_x: float = 0.05,
        start_time_s: float = 0.1,
        cells: int = 400,
        pf_dx_nm: float = 0.1,
    ) -> None:
        if cells < 16:
            raise ValueError("at least 16 matrix cells are required")
        self.temperature_c = temperature_c
        self.temperature_k = temperature_c + 273.15
        self.domain_nm = domain_nm
        self.outer_nm = 0.5 * domain_nm
        self.radius_initial_nm = 0.15 * domain_nm
        self.matrix_x = matrix_x
        self.start_time_s = start_time_s
        self.cells = cells
        self.dy = 1.0 / cells
        self.v_beta = 1.0
        self.x_eq = unit.xAg2Te_eq_from_T(self.temperature_k)
        self.diffusivity_nm2_s = (
            unit.D_Ag_in_PbTe_m2_per_s(self.temperature_k) * 1.0e18
        )
        self.target_inventory_nm = pf_half_inventory(
            temperature_c, domain_nm, pf_dx_nm, matrix_x, start_time_s
        )

    def initial_state(self, mode: str = "matched_growth") -> np.ndarray:
        radius = self.radius_initial_nm
        length = self.outer_nm - radius
        y = (np.arange(self.cells, dtype=np.float64) + 0.5) * self.dy
        distance = y * length
        if mode == "matched_growth":
            concentration = similarity_profile(
                distance, self.temperature_c, self.matrix_x, self.start_time_s
            )
            # Match the exact diffuse-PF initial total inventory without using
            # any PF trajectory.  The correction vanishes at the sharp
            # interface and is a small far-field ensemble adjustment.
            weight = 1.0 - np.exp(-distance / 0.6)
            current = radius * self.v_beta + length * float(concentration.mean())
            correction = (self.target_inventory_nm - current) / (
                length * float(weight.mean())
            )
            concentration = concentration + correction * weight
        elif mode == "equilibrium":
            concentration = np.full(self.cells, self.x_eq)
        elif mode == "dissolution":
            concentration = np.full(self.cells, 0.5 * self.x_eq)
        else:
            raise ValueError(f"unknown initial mode {mode}")
        if np.min(concentration) < 0.0 or np.max(concentration) > 1.0:
            raise RuntimeError("initial sharp concentration outside [0,1]")
        q = length * concentration
        return np.concatenate((q, np.array([radius], dtype=np.float64)))

    def inventory(self, state: np.ndarray) -> float:
        return float(state[-1] * self.v_beta + self.dy * np.sum(state[:-1]))

    def concentration(self, state: np.ndarray) -> np.ndarray:
        length = self.outer_nm - float(state[-1])
        return np.asarray(state[:-1]) / length

    def velocity(self, state: np.ndarray) -> float:
        radius = float(state[-1])
        length = self.outer_nm - radius
        concentration = self.concentration(state)
        gradient = (concentration[0] - self.x_eq) / (0.5 * self.dy * length)
        return self.diffusivity_nm2_s * gradient / (self.v_beta - self.x_eq)

    def rhs(self, _time: float, state: np.ndarray) -> np.ndarray:
        radius = float(state[-1])
        length = self.outer_nm - radius
        if not (0.0 < radius < self.outer_nm):
            raise RuntimeError("sharp interface left finite domain")
        concentration = self.concentration(state)
        velocity = self.velocity(state)
        flux = np.zeros(self.cells + 1, dtype=np.float64)
        # This equals J(0)-V*x_eq by Stefan balance and closes the combined
        # beta+matrix inventory exactly in the semi-discrete equations.
        flux[0] = -velocity * self.v_beta
        for face in range(1, self.cells):
            diffusive = -self.diffusivity_nm2_s * (
                concentration[face] - concentration[face - 1]
            ) / (self.dy * length)
            grid_velocity = (1.0 - face * self.dy) * velocity
            advected = (
                concentration[face]
                if grid_velocity >= 0.0 else concentration[face - 1]
            )
            flux[face] = diffusive - grid_velocity * advected
        flux[-1] = 0.0  # exact outer no-flux/symmetry boundary
        q_rate = -(flux[1:] - flux[:-1]) / self.dy
        return np.concatenate((q_rate, np.array([velocity])))

    def run(
        self,
        final_time_s: float,
        max_step_s: float,
        mode: str = "matched_growth",
        rtol: float = 2.0e-10,
        atol: float = 2.0e-12,
    ) -> dict[str, float | str | int]:
        state0 = self.initial_state(mode)
        mass0 = self.inventory(state0)
        velocity0 = self.velocity(state0)
        solution = solve_ivp(
            self.rhs,
            (0.0, final_time_s),
            state0,
            method="BDF",
            rtol=rtol,
            atol=atol,
            max_step=max_step_s,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        state1 = solution.y[:, -1]
        mass1 = self.inventory(state1)
        concentration1 = self.concentration(state1)
        mass_error = abs(mass1 - mass0) / max(abs(mass0), 1.0)
        return {
            "mode": mode,
            "cells": self.cells,
            "max_step_s": max_step_s,
            "final_time_s": final_time_s,
            "solver_steps": len(solution.t) - 1,
            "rhs_evaluations": solution.nfev,
            "radius_initial_nm": float(state0[-1]),
            "radius_final_nm": float(state1[-1]),
            "displacement_nm": float(state1[-1] - state0[-1]),
            "velocity_average_nm_s": float((state1[-1] - state0[-1]) / final_time_s),
            "velocity_initial_nm_s": velocity0,
            "velocity_final_nm_s": self.velocity(state1),
            "inventory_initial_nm": mass0,
            "inventory_final_nm": mass1,
            "mass_error_rel": mass_error,
            "x_min": float(np.min(concentration1)),
            "x_max": float(np.max(concentration1)),
            "x_outer": float(concentration1[-1]),
            "x_interface_cell": float(concentration1[0]),
            "finite": bool(np.all(np.isfinite(state1))),
        }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative_change(left: float, right: float) -> float:
    return abs(right - left) / max(abs(left), abs(right), 1.0e-300)


def run_convergence(report_root: Path = REPORT_ROOT) -> dict[str, object]:
    observation = 2.0821852621180848e-4
    rows: list[dict[str, object]] = []
    for cells in (100, 200, 400, 800):
        oracle = PlanarSharpOracle(cells=cells)
        row = oracle.run(observation, observation / 40.0)
        row["study"] = "mesh"
        rows.append(row)
    for divisor in (5, 10, 20, 40, 80):
        oracle = PlanarSharpOracle(cells=400)
        row = oracle.run(observation, observation / divisor)
        row["study"] = "time_step"
        rows.append(row)

    equilibrium = PlanarSharpOracle(cells=200)
    equilibrium_row = equilibrium.run(1.0, 0.02, mode="equilibrium")
    equilibrium_row["study"] = "zero_velocity_equilibrium"
    rows.append(equilibrium_row)
    dissolution = PlanarSharpOracle(cells=200)
    dissolution_row = dissolution.run(1.0e-3, 1.0e-5, mode="dissolution")
    dissolution_row["study"] = "direction"
    rows.append(dissolution_row)

    fine = next(row for row in rows if row["study"] == "mesh" and row["cells"] == 800)
    medium = next(row for row in rows if row["study"] == "mesh" and row["cells"] == 400)
    fine_dt = next(row for row in rows if row["study"] == "time_step"
                   and math.isclose(float(row["max_step_s"]), observation / 80.0))
    coarse_dt = next(row for row in rows if row["study"] == "time_step"
                     and math.isclose(float(row["max_step_s"]), observation / 40.0))
    mesh_error = relative_change(
        float(medium["velocity_average_nm_s"]), float(fine["velocity_average_nm_s"])
    )
    dt_error = relative_change(
        float(coarse_dt["velocity_average_nm_s"]),
        float(fine_dt["velocity_average_nm_s"]),
    )
    max_mass_error = max(float(row["mass_error_rel"]) for row in rows)
    status = (
        "PASS_DIFFUSION_CONTROLLED_PLANAR_SHARP_ORACLE"
        if mesh_error <= 0.01 and dt_error <= 1.0e-5
        and max_mass_error <= 1.0e-10
        and float(fine["velocity_average_nm_s"]) > 0.0
        and float(dissolution_row["velocity_average_nm_s"]) < 0.0
        and abs(float(equilibrium_row["velocity_average_nm_s"])) <= 1.0e-12
        else "FAIL_DIFFUSION_CONTROLLED_PLANAR_SHARP_ORACLE"
    )
    write_csv(report_root / "correction1_diffusion_sharp_convergence.csv", rows)
    reference = {
        "status": status,
        "model": "finite_domain_planar_one_sided_diffusion_controlled_ALE_FV",
        "T_C": 400,
        "D_beta": 0.0,
        "interface_xB": PlanarSharpOracle().x_eq,
        "outer_boundary": "zero_flux",
        "finite_inventory_matched_to_PF_initializer": True,
        "observation_time_s": observation,
        "velocity_average_nm_s": fine["velocity_average_nm_s"],
        "displacement_nm": fine["displacement_nm"],
        "mass_error_rel": fine["mass_error_rel"],
        "mesh_relative_change_400_to_800": mesh_error,
        "time_step_relative_change_40_to_80": dt_error,
        "equilibrium_velocity_nm_s": equilibrium_row["velocity_average_nm_s"],
        "dissolution_velocity_nm_s": dissolution_row["velocity_average_nm_s"],
    }
    (report_root / "correction1_diffusion_sharp_reference.json").write_text(
        json.dumps(reference, indent=2) + "\n"
    )
    (report_root / "correction1_diffusion_sharp_oracle.md").write_text(f"""# Correction 1 Diffusion-Controlled Sharp Oracle

The independent reference solves planar matrix Fick diffusion on one half of
the periodic slab, with `x_alpha(R+)=x_eq`, `J_beta=0`, exact no-flux at the
outer symmetry boundary, and
`dR/dt=D_alpha*x_r(R+)/(1-x_eq)`.  It contains no finite interface reaction
resistance and uses no PF trajectory or fitted PF velocity.

The ALE finite-volume variable is `q=(L-R)*x_alpha`.  Its boundary flux is
`F_0=-v_B*dR/dt`, so `d[R*v_B + integral(q)dy]/dt=0` in the semi-discrete
equations.  Initial total inventory is matched to the accepted diffuse-PF
initializer only as an ensemble definition.

* fine reference velocity: `{float(fine['velocity_average_nm_s']):.17e} nm/s`
* fine reference displacement: `{float(fine['displacement_nm']):.17e} nm`
* 400-to-800-cell relative change: `{mesh_error:.3e}`
* max-step refinement relative change: `{dt_error:.3e}`
* maximum inventory error: `{max_mass_error:.3e}`
* zero-driving velocity: `{float(equilibrium_row['velocity_average_nm_s']):.3e} nm/s`
* dissolution-test velocity: `{float(dissolution_row['velocity_average_nm_s']):.3e} nm/s`

`diffusion_sharp_oracle_status={status}`
""")
    return reference


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    reference = run_convergence(args.report_root)
    print(f"diffusion_sharp_oracle_status={reference['status']}")
    print(f"sharp_velocity_nm_s={reference['velocity_average_nm_s']:.17e}")
    print(f"sharp_mass_error_rel={reference['mass_error_rel']:.17e}")
    return 0 if str(reference["status"]).startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
