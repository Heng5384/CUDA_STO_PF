"""N1--N7 numerical qualification plus a global ledger-closure invariant."""

from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np

from kwn_mvp.diagnostics import solver_observables
from kwn_mvp.solver import KWNSolver, SolverConfig


def _config(
    *,
    bins: int = 200,
    beta_diffusivity: float = 1.0e-19,
    matrix_xb: float = 0.0066,
    beta_initial: Dict[str, Any] | None = None,
    gamma: float = 0.01,
    xeq: float = 0.006,
    max_dt_s: float = 200.0,
) -> Dict[str, Any]:
    """Create an explicit synthetic numerical-qualification config in SI units."""

    return {
        "simulation": {
            "temperature_K": 653.15,
            "max_dt_s": max_dt_s,
            "min_dt_s": 1.0e-12,
            "size_cfl": 0.35,
            "cfl_active_inventory_relative_threshold": 1.0e-6,
            "rmax_outflow_relative_tolerance": 1.0e-10,
        },
        "matrix": {
            "molar_volume_m3_mol": 4.1009e-5,
            "initial_xB": matrix_xb,
            "total_b_mol_m3": None,
            "inventory_tolerance_relative": 1.0e-10,
        },
        "radius_grid": {"minimum_m": 5.0e-10, "maximum_m": 2.0e-7, "bins": bins},
        "thermodynamics": {"mode": "approximate_dilute", "planar_reference_xB": xeq},
        "populations": {
            "g": {
                "xB": 0.02,
                "molar_volume_m3_mol": 4.1009e-5,
                "diffusivity_m2_s": 0.0,
                "gamma_j_m2": 0.0,
                "xeq_infinity": 0.005,
                "initial": {"kind": "empty"},
                "nucleation": {"mode": "off"},
            },
            "beta": {
                "xB": 1.0,
                "molar_volume_m3_mol": 4.1009e-5,
                "diffusivity_m2_s": beta_diffusivity,
                "gamma_j_m2": gamma,
                "xeq_infinity": xeq,
                "initial": beta_initial
                or {
                    "kind": "discrete",
                    "entries": [
                        {"radius_m": 1.5e-9, "number_density_m3": 1.0e22},
                        {"radius_m": 7.0e-9, "number_density_m3": 1.0e22},
                    ],
                },
                "nucleation": {"mode": "off"},
            },
        },
    }


class NumericalQualificationTest(unittest.TestCase):
    """Run N1--N7 with fixed synthetic states and one all-state mass check."""

    def _solver(self, **kwargs: Any) -> KWNSolver:
        """Build a solver for one numerical gate."""

        return KWNSolver(SolverConfig.from_mapping(_config(**kwargs)))

    def test_n1_zero_mobility_invariance(self) -> None:
        """N1: D=J=0 leaves PSD, matrix, and inventory unchanged."""

        solver = self._solver(beta_diffusivity=0.0)
        before = solver.state_arrays()
        solver.run_steps(20)
        after = solver.state_arrays()
        for key in ("g_number_density_per_m4", "beta_number_density_per_m4", "matrix_xb"):
            np.testing.assert_array_equal(before[key], after[key])
        self.assertLessEqual(
            solver.ledger.snapshot(matrix_xb=solver.matrix_xb, populations=solver.population_list()).relative_residual,
            1.0e-12,
        )

    def test_n2_pure_dissolution(self) -> None:
        """N2: undersaturated matrix dissolves beta and returns solute to matrix."""

        solver = self._solver(matrix_xb=0.0045, gamma=0.0, xeq=0.006, beta_diffusivity=1.0e-19)
        before = solver_observables(solver)
        solver.run_to_time(5.0e4)
        after = solver_observables(solver)
        self.assertLess(after["beta_volume_fraction"], before["beta_volume_fraction"])
        self.assertGreater(after["matrix_xB"], before["matrix_xB"])
        self.assertGreaterEqual(np.min(solver.population("beta").number_density_per_m4), 0.0)
        self.assertLessEqual(after["inventory_relative_residual"], 1.0e-10)

    def test_n3_pure_growth(self) -> None:
        """N3: supersaturation grows beta and reduces matrix supersaturation."""

        solver = self._solver(matrix_xb=0.008, gamma=0.0, xeq=0.006, beta_diffusivity=1.0e-19)
        before = solver_observables(solver)
        solver.run_to_time(5.0e4)
        after = solver_observables(solver)
        self.assertGreater(after["beta_mean_radius_m"], before["beta_mean_radius_m"])
        self.assertLess(after["matrix_xB"], before["matrix_xB"])
        self.assertLessEqual(after["inventory_relative_residual"], 1.0e-10)

    def test_n4_closed_coarsening_sanity(self) -> None:
        """N4: mixed sub/supercritical beta follows a closed coarsening trend."""

        solver = self._solver(beta_diffusivity=5.0e-20)
        before = solver_observables(solver)
        solver.run_to_time(1.5e5)
        after = solver_observables(solver)
        self.assertLess(after["beta_number_density_m3"], before["beta_number_density_m3"])
        self.assertGreater(after["beta_mean_radius_m"], before["beta_mean_radius_m"])
        self.assertLess(after["beta_specific_surface_area_m_inv"], before["beta_specific_surface_area_m_inv"])
        self.assertLess(
            abs(after["beta_B_inventory_mol_m3"] - before["beta_B_inventory_mol_m3"])
            / before["beta_B_inventory_mol_m3"],
            0.35,
        )
        self.assertLessEqual(after["inventory_relative_residual"], 1.0e-10)

    def test_n5_radius_bin_convergence(self) -> None:
        """N5: 100/200/400-bin results remain within the declared 2% target."""

        results = []
        initial = {"kind": "lognormal", "number_density_m3": 2.0e22, "median_radius_m": 5.0e-9, "log_sigma": 0.2}
        for bins in (100, 200, 400):
            solver = self._solver(bins=bins, beta_initial=initial, gamma=0.002, beta_diffusivity=3.0e-20)
            solver.run_to_time(3.0e4)
            results.append(solver_observables(solver))
        reference = results[-1]
        for candidate in results[:-1]:
            for key in (
                "beta_number_density_m3",
                "beta_mean_radius_m",
                "beta_mean_radius_cubed_m3",
                "beta_specific_surface_area_m_inv",
                "beta_volume_fraction",
                "matrix_xB",
            ):
                denom = max(abs(reference[key]), 1.0e-300)
                self.assertLessEqual(abs(candidate[key] - reference[key]) / denom, 0.02, key)

    def test_n6_timestep_convergence(self) -> None:
        """N6: max_dt and max_dt/2 agree on primary observables within 2%."""

        initial = {"kind": "lognormal", "number_density_m3": 2.0e22, "median_radius_m": 5.0e-9, "log_sigma": 0.2}
        coarse = self._solver(beta_initial=initial, gamma=0.002, beta_diffusivity=3.0e-20, max_dt_s=300.0)
        fine = self._solver(beta_initial=initial, gamma=0.002, beta_diffusivity=3.0e-20, max_dt_s=150.0)
        coarse.run_to_time(3.0e4)
        fine.run_to_time(3.0e4)
        c = solver_observables(coarse)
        f = solver_observables(fine)
        for key in ("beta_number_density_m3", "beta_mean_radius_m", "beta_volume_fraction", "matrix_xB"):
            self.assertLessEqual(abs(c[key] - f[key]) / max(abs(f[key]), 1.0e-300), 0.02, key)

    def test_n7_restart(self) -> None:
        """N7: checkpoint/restart preserves all bins, matrix composition, and ledger."""

        continuous = self._solver()
        continuous.run_steps(80)
        split = self._solver()
        split.run_steps(40)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "restart.npz"
            split.save_checkpoint(checkpoint)
            resumed = KWNSolver.load_checkpoint(config=split.config, path=checkpoint)
            resumed.run_steps(40)
        for key, value in continuous.state_arrays().items():
            np.testing.assert_array_equal(value, resumed.state_arrays()[key])
        self.assertEqual(
            continuous.ledger.component_dict(matrix_xb=continuous.matrix_xb, populations=continuous.population_list()),
            resumed.ledger.component_dict(matrix_xb=resumed.matrix_xb, populations=resumed.population_list()),
        )

    def test_inventory_closure_across_representative_states(self) -> None:
        """Every representative N1--N7-style trajectory remains ledger-closed."""

        states = []
        zero_mobility = self._solver(beta_diffusivity=0.0)
        zero_mobility.run_steps(20)
        states.append(zero_mobility)

        dissolution = self._solver(
            matrix_xb=0.0045,
            gamma=0.0,
            xeq=0.006,
            beta_diffusivity=1.0e-19,
        )
        dissolution.run_to_time(5.0e4)
        states.append(dissolution)

        growth = self._solver(
            matrix_xb=0.008,
            gamma=0.0,
            xeq=0.006,
            beta_diffusivity=1.0e-19,
        )
        growth.run_to_time(5.0e4)
        states.append(growth)

        coarsening = self._solver(beta_diffusivity=5.0e-20)
        coarsening.run_to_time(1.5e5)
        states.append(coarsening)

        grid_style = self._solver(
            bins=100,
            beta_initial={
                "kind": "lognormal",
                "number_density_m3": 2.0e22,
                "median_radius_m": 5.0e-9,
                "log_sigma": 0.2,
            },
            gamma=0.002,
            beta_diffusivity=3.0e-20,
        )
        grid_style.run_to_time(3.0e4)
        states.append(grid_style)

        timestep_style = self._solver(
            bins=400,
            beta_initial={
                "kind": "lognormal",
                "number_density_m3": 2.0e22,
                "median_radius_m": 5.0e-9,
                "log_sigma": 0.2,
            },
            gamma=0.002,
            beta_diffusivity=3.0e-20,
            max_dt_s=150.0,
        )
        timestep_style.run_to_time(3.0e4)
        states.append(timestep_style)

        restart_source = self._solver()
        restart_source.run_steps(40)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "n8_restart.npz"
            restart_source.save_checkpoint(checkpoint)
            restart_resumed = KWNSolver.load_checkpoint(config=restart_source.config, path=checkpoint)
            restart_resumed.run_steps(40)
        states.extend((restart_source, restart_resumed))

        maximum_relative_residual = max(
            diagnostic.inventory.relative_residual
            for solver in states
            for diagnostic in solver.history
        )
        maximum_relative_residual = max(
            maximum_relative_residual,
            *(
                solver.ledger.snapshot(
                    matrix_xb=solver.matrix_xb,
                    populations=solver.population_list(),
                ).relative_residual
                for solver in states
            ),
        )
        self.assertLessEqual(maximum_relative_residual, 1.0e-10)
