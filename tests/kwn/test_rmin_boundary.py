"""Regression tests for the explicit unresolved-sub-1-nm KWN boundary."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np

from kwn_mvp.config import load_config
from kwn_mvp.diagnostics import solver_observables
from kwn_mvp.solver import KWNSolver, RadiusGridOverflowError, SolverConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _dissolution_config() -> Dict[str, Any]:
    """Return a small state whose only transport route is conservative Rmin loss."""

    return {
        "simulation": {
            "temperature_K": 653.15,
            "max_dt_s": 100.0,
            "min_dt_s": 1.0e-12,
            "size_cfl": 0.35,
            "rmax_outflow_relative_tolerance": 1.0e-10,
        },
        "matrix": {
            "molar_volume_m3_mol": 4.1009e-5,
            "initial_xB": 0.0045,
            "total_b_mol_m3": None,
            "inventory_tolerance_relative": 1.0e-10,
        },
        "radius_grid": {"minimum_m": 1.0e-9, "maximum_m": 1.28e-7, "bins": 140},
        "thermodynamics": {"mode": "approximate_dilute", "planar_reference_xB": 0.006},
        "populations": {
            "g": {
                "xB": 0.02,
                "molar_volume_m3_mol": 4.1009e-5,
                "diffusivity_m2_s": 1.0e-19,
                "gamma_j_m2": 0.0,
                "xeq_infinity": 0.006,
                "initial": {
                    "kind": "discrete",
                    "entries": [{"radius_m": 1.01e-9, "number_density_m3": 1.0e22}],
                },
                "nucleation": {"mode": "off"},
            },
            "beta": {
                "xB": 1.0,
                "molar_volume_m3_mol": 4.1009e-5,
                "diffusivity_m2_s": 0.0,
                "gamma_j_m2": 0.0,
                "xeq_infinity": 0.006,
                "initial": {"kind": "empty"},
                "nucleation": {"mode": "off"},
            },
        },
    }


class RminBoundaryTest(unittest.TestCase):
    """Keep the numerical GP cutoff explicit, conservative, and PF-aligned."""

    def test_gp_configs_declare_cutoff_and_exact_handoff_edge(self) -> None:
        """Both GP routes retain 1--3 nm support and put 8 nm at an exact edge."""

        for name in ("gp_prescribed_source_mvp.yaml", "gp_effective_cnt_sweep.yaml"):
            document = load_config(REPOSITORY_ROOT / "configs" / "kwn" / name)
            grid_data = document.data["radius_grid"]
            config = SolverConfig.from_document(document)
            self.assertEqual(grid_data["minimum_m"], 1.0e-9)
            self.assertEqual(grid_data["maximum_m"], 1.28e-7)
            self.assertEqual(grid_data["bins"], 140)
            self.assertEqual(
                grid_data["lower_boundary_policy"],
                "UNRESOLVED_SUB_1NM_DISSOLUTION_NUMERICAL_BOUNDARY_GP_OBSERVATION_SCOPE_1_TO_3_NM",
            )
            self.assertEqual(grid_data["handoff_radius_alignment"], "EXACT_LOG_GRID_EDGE_INDEX_60_OF_140")
            self.assertAlmostEqual(config.grid.edges_m[60], 8.0e-9, places=22)

    def test_rmin_dissolution_returns_inventory_to_matrix(self) -> None:
        """A subcritical first-bin GP exits conservatively under the declared CFL cap."""

        solver = KWNSolver(SolverConfig.from_mapping(_dissolution_config()))
        before = solver_observables(solver)
        diagnostic = solver.advance_one()
        after = solver_observables(solver)
        self.assertGreater(diagnostic.rmin_dissolution_flux_m3_s, 0.0)
        self.assertLess(after["g_number_density_m3"], before["g_number_density_m3"])
        self.assertGreater(after["matrix_xB"], before["matrix_xB"])
        self.assertLessEqual(diagnostic.size_cfl, 0.4)
        self.assertLessEqual(after["inventory_relative_residual"], 1.0e-12)
        self.assertGreaterEqual(np.min(solver.population("g").number_density_per_m4), 0.0)

    def test_beta_rmin_dissolution_returns_its_inventory_to_matrix(self) -> None:
        """The beta Rmin route closes its own matrix-plus-beta inventory change."""

        data = _dissolution_config()
        data["populations"]["g"].update(
            {"diffusivity_m2_s": 0.0, "initial": {"kind": "empty"}}
        )
        data["populations"]["beta"].update(
            {
                "diffusivity_m2_s": 1.0e-19,
                "gamma_j_m2": 0.0,
                "xeq_infinity": 0.006,
                "initial": {
                    "kind": "discrete",
                    "entries": [{"radius_m": 1.01e-9, "number_density_m3": 1.0e22}],
                },
            }
        )
        solver = KWNSolver(SolverConfig.from_mapping(data))
        before = solver_observables(solver)
        diagnostic = solver.advance_one()
        after = solver_observables(solver)
        matrix_gain = after["C_B_matrix_mol_m3"] - before["C_B_matrix_mol_m3"]
        beta_change = after["C_B_beta_mol_m3"] - before["C_B_beta_mol_m3"]
        self.assertGreater(diagnostic.rmin_dissolution_flux_m3_s, 0.0)
        self.assertLess(beta_change, 0.0)
        self.assertGreater(matrix_gain, 0.0)
        self.assertLessEqual(
            abs(matrix_gain + beta_change), 1.0e-12 * before["C_B_total_mol_m3"]
        )
        self.assertLessEqual(after["inventory_relative_residual"], 1.0e-12)
        self.assertGreaterEqual(np.min(solver.population("beta").number_density_per_m4), 0.0)

    def test_rmax_outflow_fails_closed_instead_of_deleting_material(self) -> None:
        """A material upper-boundary flux requests grid expansion through an error."""

        data = _dissolution_config()
        data["matrix"]["initial_xB"] = 0.008
        data["radius_grid"] = {"minimum_m": 1.0e-9, "maximum_m": 1.0e-8, "bins": 4}
        data["populations"]["g"]["diffusivity_m2_s"] = 0.0
        data["populations"]["g"]["initial"] = {"kind": "empty"}
        data["populations"]["beta"].update(
            {
                "diffusivity_m2_s": 1.0e-18,
                "gamma_j_m2": 0.0,
                "xeq_infinity": 0.006,
                "initial": {
                    "kind": "discrete",
                    "entries": [{"radius_m": 9.9e-9, "number_density_m3": 1.0e22}],
                },
            }
        )
        solver = KWNSolver(SolverConfig.from_mapping(data))
        with self.assertRaises(RadiusGridOverflowError):
            solver.advance_one()
