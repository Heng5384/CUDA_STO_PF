"""Immutable regression for the canonical physical KWN lower boundary.

The canonical smooth measure is built from the established radius-grid
qualification runner, deliberately not from the evolving time-accuracy
runner.  This freezes the physical lower edge and its ledger price while
leaving time-policy work free to change only its intended policy controls.
"""

from __future__ import annotations

from copy import deepcopy
import math
import unittest

import numpy as np

from kwn_mvp.lower_boundary import (
    boundary_event_residual,
    boundary_growth_velocity,
    boundary_inventory_diagnostic,
    boundary_number_flux_diagnostic,
    boundary_radius,
    particle_inventory_at_radius,
)
from kwn_mvp.populations import Population
from kwn_mvp.radius_grid import RadiusGrid
from kwn_mvp.solver import KWNSolver, SolverConfig
from scripts.frozen_canonical_smooth_population_v1 import build_frozen_canonical_context


FROZEN_BINS = 3200
FROZEN_CONTRACT_HASH = "d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333"
FROZEN_RMIN_M = 4.724027182871601e-10
FROZEN_RMIN_HEX = "0x1.03b4c5a5fbe5ep-31"
FROZEN_BOUNDARY_GROWTH_M_S = -11767189.594675226
FROZEN_BOUNDARY_VOLUME_M3 = 4.415966530881012e-28
FROZEN_BOUNDARY_BETA_MOLES_MOL = 1.0768286305154996e-23
FROZEN_MATRIX_XB = 0.006220578557312318
FROZEN_TOTAL_B_MOL_M3 = 731.5467336438344

BENCHMARK_NUMBER_M3 = 2.0e20
BENCHMARK_VELOCITY_M_S = -1.0e-12


def _smooth_cdf_fraction(radius_m: float, *, low_m: float, high_m: float) -> float:
    """Compact sin-squared CDF used only by the analytic crossing benchmark."""

    coordinate = (float(radius_m) - low_m) / (high_m - low_m)
    if coordinate <= 0.0:
        return 0.0
    if coordinate >= 1.0:
        return 1.0
    return coordinate - math.sin(2.0 * math.pi * coordinate) / (2.0 * math.pi)


def _smooth_density_per_m4(radius_m: float, *, low_m: float, high_m: float) -> float:
    """Analytic density corresponding to :func:`_smooth_cdf_fraction`."""

    coordinate = (float(radius_m) - low_m) / (high_m - low_m)
    if not 0.0 < coordinate < 1.0:
        return 0.0
    return (
        BENCHMARK_NUMBER_M3
        * 2.0
        / (high_m - low_m)
        * math.sin(math.pi * coordinate) ** 2
    )


class _ConstantVelocityEulerian(KWNSolver):
    """Test-only conservative FV transport with a shared physical lower face."""

    def growth_rates(self) -> dict[str, np.ndarray]:
        return {
            name: np.full(
                population.grid.bins,
                BENCHMARK_VELOCITY_M_S if name == "beta" else 0.0,
                dtype=np.float64,
            )
            for name, population in self.populations.items()
        }

    def lower_boundary_growth_velocity(self, population: Population | str) -> float:
        item = self.population(population) if isinstance(population, str) else population
        return BENCHMARK_VELOCITY_M_S if item.parameters.name == "beta" else 0.0


class FrozenLowerBoundaryContractTests(unittest.TestCase):
    """Freeze canonical Rmin physics and its independent analytic transport gate."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.context = build_frozen_canonical_context()
        cls.solver = cls.context.solver

    def _benchmark_solver(self, *, bins: int, dt_fraction: float) -> _ConstantVelocityEulerian:
        """Build the B4-style test problem at the frozen physical lower edge."""

        grid = RadiusGrid.logarithmic(
            FROZEN_RMIN_M,
            float(self.solver.config.grid.edges_m[-1]),
            bins,
        )
        low_m = 2.0 * FROZEN_RMIN_M
        high_m = 8.0 * FROZEN_RMIN_M
        cell_number = np.asarray(
            [
                BENCHMARK_NUMBER_M3
                * (
                    _smooth_cdf_fraction(upper, low_m=low_m, high_m=high_m)
                    - _smooth_cdf_fraction(lower, low_m=low_m, high_m=high_m)
                )
                for lower, upper in zip(grid.edges_m[:-1], grid.edges_m[1:])
            ],
            dtype=np.float64,
        )
        mapping = deepcopy(self.context.mapping)
        mapping["radius_grid"]["bins"] = bins
        mapping["radius_grid"].pop("edges_m", None)
        mapping["simulation"]["max_dt_s"] = (
            float(dt_fraction) * float(np.min(grid.widths_m)) / abs(BENCHMARK_VELOCITY_M_S)
        )
        mapping["simulation"]["min_dt_s"] = 1.0e-16
        mapping["simulation"]["size_cfl"] = 0.35
        mapping["simulation"]["population_measure"] = "cell_integrated"
        mapping["matrix"]["total_b_mol_m3"] = None
        mapping["populations"]["g"]["initial"] = {"kind": "empty"}
        mapping["populations"]["beta"]["initial"] = {
            "kind": "cell_integrated",
            "radius_edges_m": [float(value) for value in grid.edges_m],
            "cell_number_density_m3": [float(value) for value in cell_number],
        }
        return _ConstantVelocityEulerian(SolverConfig.from_mapping(mapping))

    def _analytic_crossing_errors(self, *, bins: int, dt_fraction: float) -> tuple[float, float, float]:
        target_time_s = 3.0 * FROZEN_RMIN_M / abs(BENCHMARK_VELOCITY_M_S)
        low_m = 2.0 * FROZEN_RMIN_M
        high_m = 8.0 * FROZEN_RMIN_M
        cutoff_m = FROZEN_RMIN_M + abs(BENCHMARK_VELOCITY_M_S) * target_time_s
        expected_m0 = BENCHMARK_NUMBER_M3 * (
            1.0 - _smooth_cdf_fraction(cutoff_m, low_m=low_m, high_m=high_m)
        )
        expected_loss = BENCHMARK_NUMBER_M3 - expected_m0
        expected_flux = abs(BENCHMARK_VELOCITY_M_S) * _smooth_density_per_m4(
            cutoff_m, low_m=low_m, high_m=high_m
        )
        solver = self._benchmark_solver(bins=bins, dt_fraction=dt_fraction)
        solver.run_to_time(target_time_s)
        actual_m0 = solver.population("beta").number_density_m3()
        actual_loss = BENCHMARK_NUMBER_M3 - actual_m0
        actual_flux = solver.history[-1].beta_rmin_number_flux_m3_s
        crossed_from_flux = math.fsum(
            diagnostic.beta_rmin_number_flux_m3_s * diagnostic.dt_s
            for diagnostic in solver.history
        )
        self.assertLessEqual(solver.history[-1].inventory.relative_residual, 1.0e-10)
        self.assertLessEqual(
            abs(crossed_from_flux - actual_loss) / BENCHMARK_NUMBER_M3,
            1.0e-10,
        )
        return (
            abs(actual_m0 - expected_m0) / BENCHMARK_NUMBER_M3,
            abs(actual_loss - expected_loss) / BENCHMARK_NUMBER_M3,
            abs(actual_flux - expected_flux) / expected_flux,
        )

    def test_canonical_physical_boundary_inventory_sign_and_matrix_closure(self) -> None:
        """The 3200-bin smooth state retains its exact physical lower-face contract."""

        solver = self.solver
        beta = solver.population("beta")
        rmin = boundary_radius(beta.grid)
        inventory = particle_inventory_at_radius(
            rmin,
            x_b=beta.parameters.x_b,
            molar_volume_m3_mol=beta.parameters.molar_volume_m3_mol,
        )
        growth = boundary_growth_velocity(
            radius_m=rmin,
            matrix_xb=solver.matrix_xb,
            parameters=beta.parameters,
            equilibrium_adapter=solver.equilibrium_adapter,
        )
        ledger = solver.ledger.snapshot(
            matrix_xb=solver.matrix_xb,
            populations=solver.population_list(),
            beta_resolved_fraction=1.0,
        )

        self.assertEqual(self.context.contract_hash, FROZEN_CONTRACT_HASH)
        self.assertEqual(rmin, FROZEN_RMIN_M)
        self.assertEqual(rmin.hex(), FROZEN_RMIN_HEX)
        self.assertEqual(float(beta.grid.edges_m[0]).hex(), FROZEN_RMIN_HEX)
        self.assertEqual(solver.matrix_xb, FROZEN_MATRIX_XB)
        self.assertEqual(ledger.total_mol_m3, FROZEN_TOTAL_B_MOL_M3)
        self.assertLessEqual(ledger.relative_residual, 1.0e-12)
        self.assertLessEqual(
            abs(solver.ledger.recover_matrix_xb(solver.population_list()) - solver.matrix_xb),
            1.0e-14,
        )

        self.assertEqual(growth, FROZEN_BOUNDARY_GROWTH_M_S)
        self.assertEqual(solver.lower_boundary_growth_velocity("beta"), FROZEN_BOUNDARY_GROWTH_M_S)
        self.assertEqual(inventory.volume_m3, FROZEN_BOUNDARY_VOLUME_M3)
        self.assertEqual(inventory.beta_moles_mol, FROZEN_BOUNDARY_BETA_MOLES_MOL)
        self.assertEqual(inventory.b_moles_mol, FROZEN_BOUNDARY_BETA_MOLES_MOL)
        self.assertEqual(boundary_event_residual(rmin, rmin), 0.0)

        self.assertLess(growth, 0.0)
        number_flux = boundary_number_flux_diagnostic(growth, 1.0)
        self.assertEqual(number_flux, -FROZEN_BOUNDARY_GROWTH_M_S)
        self.assertEqual(boundary_number_flux_diagnostic(-growth, 1.0), 0.0)
        flux = boundary_inventory_diagnostic(number_flux, inventory)
        self.assertGreater(flux.number_flux_out_m3_s, 0.0)
        self.assertGreater(flux.b_mol_flux_out_mol_m3_s, 0.0)
        self.assertEqual(
            flux.beta_volume_flux_out_s,
            number_flux * FROZEN_BOUNDARY_VOLUME_M3,
        )
        self.assertEqual(
            flux.b_mol_flux_out_mol_m3_s,
            number_flux * FROZEN_BOUNDARY_BETA_MOLES_MOL,
        )

    def test_shared_physical_boundary_b4_style_analytic_crossing_refines(self) -> None:
        """The shared FV lower face still converges for smooth analytic crossing."""

        spatial_errors = [
            self._analytic_crossing_errors(bins=bins, dt_fraction=0.1)
            for bins in (64, 128, 256)
        ]
        temporal_errors = [
            self._analytic_crossing_errors(bins=128, dt_fraction=fraction)
            for fraction in (0.2, 0.1, 0.05)
        ]
        for metric_index, label in enumerate(("M0", "number loss", "boundary flux")):
            coarse, medium, fine = (entry[metric_index] for entry in spatial_errors)
            self.assertGreater(coarse, medium, msg=f"spatial {label}: {spatial_errors}")
            self.assertGreater(medium, fine, msg=f"spatial {label}: {spatial_errors}")
            coarse, medium, fine = (entry[metric_index] for entry in temporal_errors)
            self.assertGreater(coarse, medium, msg=f"temporal {label}: {temporal_errors}")
            self.assertGreater(medium, fine, msg=f"temporal {label}: {temporal_errors}")
        self.assertLess(spatial_errors[-1][0], 0.02, msg=f"fine M0={spatial_errors[-1][0]}")
        self.assertLess(
            spatial_errors[-1][1], 0.02, msg=f"fine number loss={spatial_errors[-1][1]}"
        )
        self.assertLess(
            spatial_errors[-1][2], 0.05, msg=f"fine boundary flux={spatial_errors[-1][2]}"
        )


if __name__ == "__main__":
    unittest.main()
