"""B1--B7 regression gates for the shared physical KWN lower boundary.

The constant-velocity solvers in this module are test-only transport
benchmarks.  They exercise the production finite-volume and cohort event
implementations without changing the frozen thermodynamic or growth-law
contract used by scientific runs.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from kwn_mvp.cohort_solver import Cohort, CohortSolver
from kwn_mvp.lower_boundary import (
    boundary_event_residual,
    boundary_growth_velocity,
    boundary_inventory_diagnostic,
    boundary_number_flux_diagnostic,
    boundary_radius,
    particle_inventory_at_radius,
)
from kwn_mvp.population_metrics import positive_cell_quadrature
from kwn_mvp.populations import Population, PopulationParameters
from kwn_mvp.radius_grid import RadiusGrid
from kwn_mvp.solver import KWNSolver, SolverConfig
from kwn_mvp.thermo_adapter import DiluteEquilibriumAdapter


RMIN_M = 5.0e-9
RMAX_M = 2.0e-8
SMOOTH_LOW_M = 7.0e-9
SMOOTH_HIGH_M = 1.2e-8
SMOOTH_NUMBER_M3 = 2.0e20
CONSTANT_DISSOLUTION_M_S = -1.0e-12


def _beta_parameters(*, x_b: float = 0.8) -> PopulationParameters:
    """Return one physical beta parameter set for the focused gates."""

    return PopulationParameters(
        name="beta",
        x_b=x_b,
        molar_volume_m3_mol=4.1009e-5,
        diffusivity_m2_s=1.0e-19,
        gamma_j_m2=0.0,
        xeq_infinity=0.006,
        nucleation={"mode": "off"},
    )


def _smooth_cdf_fraction(radius_m: float) -> float:
    """CDF of the normalized compact smooth sin-squared test population."""

    coordinate = (float(radius_m) - SMOOTH_LOW_M) / (SMOOTH_HIGH_M - SMOOTH_LOW_M)
    if coordinate <= 0.0:
        return 0.0
    if coordinate >= 1.0:
        return 1.0
    return coordinate - math.sin(2.0 * math.pi * coordinate) / (2.0 * math.pi)


def _smooth_density_per_m4(radius_m: float) -> float:
    """Analytic density corresponding to :func:`_smooth_cdf_fraction`."""

    coordinate = (float(radius_m) - SMOOTH_LOW_M) / (SMOOTH_HIGH_M - SMOOTH_LOW_M)
    if not 0.0 < coordinate < 1.0:
        return 0.0
    return (
        SMOOTH_NUMBER_M3
        * 2.0
        / (SMOOTH_HIGH_M - SMOOTH_LOW_M)
        * math.sin(math.pi * coordinate) ** 2
    )


def _smooth_cell_numbers(edges_m: np.ndarray) -> np.ndarray:
    """Integrate the smooth canonical benchmark exactly into radius cells."""

    cdf = np.asarray([_smooth_cdf_fraction(value) for value in edges_m], dtype=np.float64)
    return SMOOTH_NUMBER_M3 * np.diff(cdf)


def _mapping(
    *,
    bins: int,
    beta_initial: dict[str, Any],
    max_dt_s: float,
    x_b: float = 0.8,
) -> dict[str, Any]:
    """Build a small beta-only inventory-closed KWN state for a unit gate."""

    return {
        "simulation": {
            "temperature_K": 653.15,
            "max_dt_s": max_dt_s,
            "min_dt_s": 1.0e-16,
            "size_cfl": 0.35,
            "cfl_active_inventory_relative_threshold": 1.0e-12,
            "rmax_outflow_relative_tolerance": 1.0e-12,
            "population_measure": "cell_integrated",
        },
        "matrix": {
            "molar_volume_m3_mol": 4.1009e-5,
            "initial_xB": 0.005,
            "total_b_mol_m3": None,
            "inventory_tolerance_relative": 1.0e-12,
        },
        "radius_grid": {"minimum_m": RMIN_M, "maximum_m": RMAX_M, "bins": bins},
        "thermodynamics": {"mode": "approximate_dilute", "planar_reference_xB": 0.006},
        "populations": {
            "g": {
                "xB": 0.02,
                "molar_volume_m3_mol": 4.1009e-5,
                "diffusivity_m2_s": 0.0,
                "gamma_j_m2": 0.0,
                "xeq_infinity": 0.006,
                "initial": {"kind": "empty"},
                "nucleation": {"mode": "off"},
            },
            "beta": {
                "xB": x_b,
                "molar_volume_m3_mol": 4.1009e-5,
                "diffusivity_m2_s": 1.0e-19,
                "gamma_j_m2": 0.0,
                "xeq_infinity": 0.006,
                "initial": beta_initial,
                "nucleation": {"mode": "off"},
            },
        },
    }


def _smooth_initial_definition(*, bins: int) -> dict[str, Any]:
    """Return the same physical smooth measure on the requested grid."""

    grid = RadiusGrid.logarithmic(RMIN_M, RMAX_M, bins)
    return {
        "kind": "cell_integrated",
        "radius_edges_m": [float(value) for value in grid.edges_m],
        "cell_number_density_m3": [float(value) for value in _smooth_cell_numbers(grid.edges_m)],
    }


def _first_cell_initial_definition(*, bins: int, number_m3: float) -> dict[str, Any]:
    """Put a finite resolved population in the first physical FV cell."""

    grid = RadiusGrid.logarithmic(RMIN_M, RMAX_M, bins)
    values = np.zeros(bins, dtype=np.float64)
    values[0] = number_m3
    return {
        "kind": "cell_integrated",
        "radius_edges_m": [float(value) for value in grid.edges_m],
        "cell_number_density_m3": [float(value) for value in values],
    }


class _ConstantVelocityEulerian(KWNSolver):
    """Test-only physical-Rmin FV operator with ``dR/dt=constant``."""

    constant_velocity_m_s = CONSTANT_DISSOLUTION_M_S

    def __init__(
        self,
        config: SolverConfig,
        constant_velocity_m_s: float = CONSTANT_DISSOLUTION_M_S,
    ) -> None:
        self.constant_velocity_m_s = float(constant_velocity_m_s)
        super().__init__(config)

    def growth_rates(self) -> dict[str, np.ndarray]:
        return {
            name: np.full(
                population.grid.bins,
                self.constant_velocity_m_s if name == "beta" else 0.0,
                dtype=np.float64,
            )
            for name, population in self.populations.items()
        }

    def lower_boundary_growth_velocity(self, population: Population | str) -> float:
        item = self.population(population) if isinstance(population, str) else population
        return self.constant_velocity_m_s if item.parameters.name == "beta" else 0.0


class _ConstantVelocityCohort(CohortSolver):
    """Test-only characteristic benchmark with an exactly known event time."""

    def __init__(self, *args: Any, constant_velocity_m_s: float, **kwargs: Any) -> None:
        self.constant_velocity_m_s = float(constant_velocity_m_s)
        super().__init__(*args, **kwargs)

    def growth_rates(self) -> np.ndarray:
        return np.full(
            len(self._active_cohorts()), self.constant_velocity_m_s, dtype=np.float64
        )

    def _rhs(self, _time_s: float, radii_m: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(radii_m).shape, self.constant_velocity_m_s, dtype=np.float64)

    def boundary_growth_velocity(self) -> float:
        return self.constant_velocity_m_s


def _cohort_total_inventory(
    *, cohort: Cohort, parameters: PopulationParameters, matrix_xb: float
) -> float:
    """Derive a closed total B inventory from one discrete cohort state."""

    inventory = particle_inventory_at_radius(
        cohort.radius_m,
        x_b=parameters.x_b,
        molar_volume_m3_mol=parameters.molar_volume_m3_mol,
    )
    beta_fraction = cohort.weight_m3 * inventory.volume_m3
    return (
        (1.0 - beta_fraction) * matrix_xb / 4.1009e-5
        + cohort.weight_m3 * inventory.b_moles_mol
    )


def _constant_cohort(
    *,
    radius_m: float = 8.0e-9,
    weight_m3: float = 1.0e17,
    velocity_m_s: float = CONSTANT_DISSOLUTION_M_S,
) -> _ConstantVelocityCohort:
    """Construct a one-cohort event fixture with an algebraically closed matrix."""

    parameters = _beta_parameters()
    item = Cohort("crossing", radius_m, weight_m3)
    return _ConstantVelocityCohort(
        cohorts=[item],
        beta_parameters=parameters,
        matrix_molar_volume_m3_mol=4.1009e-5,
        total_b_mol_m3=_cohort_total_inventory(
            cohort=item, parameters=parameters, matrix_xb=0.005
        ),
        equilibrium_adapter=DiluteEquilibriumAdapter(temperature_k=653.15),
        temperature_k=653.15,
        r_diss_m=RMIN_M,
        inventory_tolerance_relative=1.0e-12,
        rtol=1.0e-11,
        atol_m=1.0e-20,
        constant_velocity_m_s=velocity_m_s,
    )


def _cohort_from_eulerian(solver: KWNSolver) -> CohortSolver:
    """Use positive cell quadrature so the cohort and FV M3 inventories match."""

    beta = solver.population("beta")
    numbers = beta.number_density_per_m4 * beta.grid.widths_m
    radii, weights = positive_cell_quadrature(beta.grid.edges_m, numbers, points_per_cell=2)
    cohorts = [
        Cohort(f"cell_{index:05d}", float(radius), float(weight))
        for index, (radius, weight) in enumerate(zip(radii, weights))
    ]
    return CohortSolver.from_kwn_solver(kwn_solver=solver, cohorts=cohorts, rtol=1.0e-11)


def _constant_eulerian(
    *,
    bins: int,
    initial: dict[str, Any],
    velocity_m_s: float = CONSTANT_DISSOLUTION_M_S,
    dt_fraction_of_smallest_cell: float = 0.1,
) -> _ConstantVelocityEulerian:
    """Construct an FV constant-velocity benchmark with coupled h/dt refinement."""

    grid = RadiusGrid.logarithmic(RMIN_M, RMAX_M, bins)
    max_dt_s = dt_fraction_of_smallest_cell * float(np.min(grid.widths_m)) / abs(velocity_m_s)
    config = SolverConfig.from_mapping(
        _mapping(bins=bins, beta_initial=initial, max_dt_s=max_dt_s)
    )
    return _ConstantVelocityEulerian(config, constant_velocity_m_s=velocity_m_s)


def _relative_error(observed: float, expected: float, *, scale: float | None = None) -> float:
    denominator = max(abs(expected), 1.0e-300) if scale is None else max(abs(scale), 1.0e-300)
    return abs(observed - expected) / denominator


class LowerBoundaryContractTests(unittest.TestCase):
    """B1--B7: physical lower-edge identity, conservation, and restart gates."""

    def assertRelativeEqual(self, left: float, right: float, *, tolerance: float = 1.0e-12) -> None:
        self.assertLessEqual(_relative_error(left, right), tolerance, msg=f"{left:.17e} != {right:.17e}")

    def test_b1_boundary_growth_velocity_identity(self) -> None:
        """Eulerian, helper, and cohort evaluate the same frozen G(Rmin)."""

        initial = _smooth_initial_definition(bins=48)
        solver = KWNSolver(
            SolverConfig.from_mapping(_mapping(bins=48, beta_initial=initial, max_dt_s=1.0))
        )
        cohort = _cohort_from_eulerian(solver)
        beta = solver.population("beta")
        rmin = boundary_radius(beta.grid)
        helper = boundary_growth_velocity(
            radius_m=rmin,
            matrix_xb=solver.matrix_xb,
            parameters=beta.parameters,
            equilibrium_adapter=solver.equilibrium_adapter,
        )
        eulerian = solver.lower_boundary_growth_velocity("beta")
        characteristic = cohort.boundary_growth_velocity()
        self.assertEqual(rmin.hex(), boundary_radius(cohort.r_diss_m).hex())
        self.assertRelativeEqual(eulerian, helper)
        self.assertRelativeEqual(characteristic, helper)
        self.assertRelativeEqual(float(solver.face_velocities("beta")[0]), helper)
        self.assertEqual(boundary_number_flux_diagnostic(1.0, 3.0), 0.0)
        self.assertEqual(boundary_number_flux_diagnostic(0.0, 3.0), 0.0)

    def test_b2_inventory_per_particle_identity(self) -> None:
        """Eulerian flux diagnostics and cohort use one Rmin particle inventory."""

        initial = _first_cell_initial_definition(bins=32, number_m3=1.0e18)
        solver = _constant_eulerian(bins=32, initial=initial)
        diagnostic = solver.advance_one()
        self.assertGreater(diagnostic.beta_rmin_number_flux_m3_s, 0.0)
        beta = solver.population("beta")
        helper = particle_inventory_at_radius(
            boundary_radius(beta.grid),
            x_b=beta.parameters.x_b,
            molar_volume_m3_mol=beta.parameters.molar_volume_m3_mol,
        )
        eulerian_inventory = solver.lower_boundary_particle_inventory("beta")
        cohort = _constant_cohort()
        characteristic = cohort.boundary_particle_inventory()
        eulerian_volume = (
            diagnostic.beta_rmin_volume_flux_s / diagnostic.beta_rmin_number_flux_m3_s
        )
        eulerian_b_moles = (
            diagnostic.beta_rmin_mol_b_flux_mol_m3_s
            / diagnostic.beta_rmin_number_flux_m3_s
        )
        eulerian_beta_moles = eulerian_volume / beta.parameters.molar_volume_m3_mol
        self.assertRelativeEqual(eulerian_volume, helper.volume_m3)
        self.assertRelativeEqual(eulerian_beta_moles, helper.beta_moles_mol)
        self.assertRelativeEqual(eulerian_b_moles, helper.b_moles_mol)
        self.assertRelativeEqual(eulerian_inventory.volume_m3, helper.volume_m3)
        self.assertRelativeEqual(eulerian_inventory.beta_moles_mol, helper.beta_moles_mol)
        self.assertRelativeEqual(eulerian_inventory.b_moles_mol, helper.b_moles_mol)
        self.assertRelativeEqual(characteristic.volume_m3, helper.volume_m3)
        self.assertRelativeEqual(characteristic.beta_moles_mol, helper.beta_moles_mol)
        self.assertRelativeEqual(characteristic.b_moles_mol, helper.b_moles_mol)
        flux = boundary_inventory_diagnostic(diagnostic.beta_rmin_number_flux_m3_s, helper)
        self.assertRelativeEqual(flux.beta_volume_flux_out_s, diagnostic.beta_rmin_volume_flux_s)
        self.assertRelativeEqual(flux.b_mol_flux_out_mol_m3_s, diagnostic.beta_rmin_mol_b_flux_mol_m3_s)

    def test_b3_single_characteristic_crossing(self) -> None:
        """A constant negative characteristic event occurs at analytic physical Rmin."""

        solver = _constant_cohort(radius_m=8.0e-9)
        expected_time_s = (8.0e-9 - RMIN_M) / abs(CONSTANT_DISSOLUTION_M_S)
        solver.advance_to(expected_time_s + 10.0)
        row = solver.cohort_rows()[0]
        self.assertFalse(bool(row["active"]))
        self.assertIsNotNone(row["dissolution_time_s"])
        self.assertLessEqual(abs(float(row["dissolution_time_s"]) - expected_time_s), 1.0e-6)
        self.assertIsNotNone(row["boundary_event_residual_m"])
        self.assertLessEqual(abs(float(row["boundary_event_residual_m"])), 8.0 * np.spacing(RMIN_M))
        self.assertEqual(float(row["radius_m"]).hex(), RMIN_M.hex())
        self.assertEqual(boundary_event_residual(float(row["radius_m"]), RMIN_M), 0.0)
        self.assertLessEqual(solver.snapshot().inventory_relative_residual, 1.0e-10)

    def test_b4_eulerian_absorbing_constant_velocity_refines_monotonically(self) -> None:
        """Smooth M0, crossed number, and Rmin flux converge under h/dt refinement."""

        target_time_s = 4.0e3
        cutoff_m = RMIN_M + abs(CONSTANT_DISSOLUTION_M_S) * target_time_s
        expected_m0 = SMOOTH_NUMBER_M3 * (1.0 - _smooth_cdf_fraction(cutoff_m))
        expected_loss = SMOOTH_NUMBER_M3 - expected_m0
        expected_flux = abs(CONSTANT_DISSOLUTION_M_S) * _smooth_density_per_m4(cutoff_m)
        errors: list[tuple[float, float, float]] = []
        for bins in (64, 128, 256):
            solver = _constant_eulerian(
                bins=bins, initial=_smooth_initial_definition(bins=bins)
            )
            solver.run_to_time(target_time_s)
            actual_m0 = solver.population("beta").number_density_m3()
            actual_loss = SMOOTH_NUMBER_M3 - actual_m0
            actual_flux = solver.history[-1].beta_rmin_number_flux_m3_s
            errors.append(
                (
                    _relative_error(actual_m0, expected_m0, scale=SMOOTH_NUMBER_M3),
                    _relative_error(actual_loss, expected_loss, scale=SMOOTH_NUMBER_M3),
                    _relative_error(actual_flux, expected_flux),
                )
            )
            self.assertLessEqual(
                solver.history[-1].inventory.relative_residual, 1.0e-10
            )
            crossed_from_flux = math.fsum(
                item.beta_rmin_number_flux_m3_s * item.dt_s for item in solver.history
            )
            self.assertLessEqual(
                abs(crossed_from_flux - actual_loss) / SMOOTH_NUMBER_M3, 1.0e-10
            )
        for metric_index, label in enumerate(("M0", "number loss", "boundary flux")):
            coarse, medium, fine = (entry[metric_index] for entry in errors)
            self.assertGreater(coarse, medium, msg=f"{label}: coarse={coarse} medium={medium}")
            self.assertGreater(medium, fine, msg=f"{label}: medium={medium} fine={fine}")
        self.assertLess(errors[-1][0], 0.02, msg=f"M0 fine error={errors[-1][0]}")
        self.assertLess(errors[-1][1], 0.02, msg=f"number-loss fine error={errors[-1][1]}")
        self.assertLess(errors[-1][2], 0.05, msg=f"boundary-flux fine error={errors[-1][2]}")

        # Hold the radius grid fixed and reduce only dt.  This prevents a
        # coupled h/dt ladder from accidentally masking a temporal regression.
        temporal_errors: list[tuple[float, float, float]] = []
        for dt_fraction in (0.2, 0.1, 0.05):
            solver = _constant_eulerian(
                bins=128,
                initial=_smooth_initial_definition(bins=128),
                dt_fraction_of_smallest_cell=dt_fraction,
            )
            solver.run_to_time(target_time_s)
            actual_m0 = solver.population("beta").number_density_m3()
            actual_loss = SMOOTH_NUMBER_M3 - actual_m0
            actual_flux = solver.history[-1].beta_rmin_number_flux_m3_s
            temporal_errors.append(
                (
                    _relative_error(actual_m0, expected_m0, scale=SMOOTH_NUMBER_M3),
                    _relative_error(actual_loss, expected_loss, scale=SMOOTH_NUMBER_M3),
                    _relative_error(actual_flux, expected_flux),
                )
            )
        for metric_index, label in enumerate(("M0", "number loss", "boundary flux")):
            coarse, medium, fine = (entry[metric_index] for entry in temporal_errors)
            self.assertGreater(coarse, medium, msg=f"time {label}: coarse={coarse} medium={medium}")
            self.assertGreater(medium, fine, msg=f"time {label}: medium={medium} fine={fine}")

    def test_b5_inventory_closure_for_boundary_paths(self) -> None:
        """Both event and finite-volume boundary routes remain ledger-closed."""

        cohort = _constant_cohort()
        cohort.advance_to(4.0e3)
        self.assertLessEqual(cohort.snapshot().inventory_relative_residual, 1.0e-10)

        eulerian = _constant_eulerian(
            bins=64, initial=_first_cell_initial_definition(bins=64, number_m3=1.0e18)
        )
        eulerian.run_to_time(4.0e3)
        for diagnostic in eulerian.history:
            self.assertLessEqual(diagnostic.inventory.relative_residual, 1.0e-10)
        final = eulerian.ledger.snapshot(
            matrix_xb=eulerian.matrix_xb, populations=eulerian.population_list()
        )
        self.assertLessEqual(final.relative_residual, 1.0e-10)

    def test_b6_no_double_return(self) -> None:
        """Matrix gain equals beta loss; the positive boundary diagnostic is not a source."""

        solver = _constant_eulerian(
            bins=64, initial=_first_cell_initial_definition(bins=64, number_m3=1.0e18)
        )
        before = solver.ledger.snapshot(
            matrix_xb=solver.matrix_xb,
            populations=solver.population_list(),
            beta_resolved_fraction=1.0,
        )
        solver.run_to_time(4.0e3)
        after = solver.ledger.snapshot(
            matrix_xb=solver.matrix_xb,
            populations=solver.population_list(),
            beta_resolved_fraction=1.0,
        )
        matrix_gain = after.matrix_mol_m3 - before.matrix_mol_m3
        beta_drop = before.beta_resolved_mol_m3 - after.beta_resolved_mol_m3
        boundary_residual_diagnostic = math.fsum(
            item.beta_rmin_mol_b_flux_mol_m3_s * item.dt_s for item in solver.history
        )
        continuous_release = beta_drop - boundary_residual_diagnostic
        self.assertGreater(matrix_gain, 0.0)
        self.assertGreater(beta_drop, 0.0)
        self.assertGreater(boundary_residual_diagnostic, 0.0)
        self.assertGreater(continuous_release, 0.0)
        self.assertLessEqual(
            abs(matrix_gain - beta_drop) / max(abs(before.total_mol_m3), 1.0e-300),
            1.0e-10,
        )
        self.assertGreater(
            matrix_gain + boundary_residual_diagnostic,
            beta_drop,
            msg="a diagnostic-only boundary flux must not be added to the algebraic matrix closure",
        )

    def test_b7_restart_through_boundary_event(self) -> None:
        """Both representations restart before Rmin and cross it reproducibly."""

        initial = _smooth_initial_definition(bins=64)
        continuous = _constant_eulerian(bins=64, initial=initial)
        continuous.run_steps(350)
        split = _constant_eulerian(bins=64, initial=initial)
        split.run_steps(175)
        self.assertGreater(split.population("beta").number_density_m3(), 0.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "eulerian_before_crossing.npz"
            split.save_checkpoint(checkpoint)
            resumed = _ConstantVelocityEulerian.load_checkpoint(config=split.config, path=checkpoint)
        resumed.run_steps(175)
        for key, values in continuous.state_arrays().items():
            np.testing.assert_array_equal(values, resumed.state_arrays()[key])
        self.assertGreater(
            math.fsum(item.beta_rmin_number_flux_m3_s * item.dt_s for item in continuous.history),
            0.0,
        )
        self.assertLessEqual(
            resumed.ledger.snapshot(
                matrix_xb=resumed.matrix_xb, populations=resumed.population_list()
            ).relative_residual,
            1.0e-10,
        )

        cohort_continuous = _constant_cohort()
        cohort_continuous.advance_to(3.5e3)
        cohort_split = _constant_cohort()
        cohort_split.advance_to(1.0e3)
        self.assertTrue(cohort_split.cohorts[0].active)
        checkpoint = cohort_split.checkpoint()
        cohort_resumed = _constant_cohort()
        cohort_resumed.restore_checkpoint(checkpoint)
        cohort_resumed.advance_to(3.5e3)
        for field, value in cohort_continuous.snapshot().as_dict().items():
            self.assertTrue(
                math.isclose(
                    float(value),
                    float(cohort_resumed.snapshot().as_dict()[field]),
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-18,
                ),
                msg=field,
            )
        continuous_row = cohort_continuous.cohort_rows()[0]
        resumed_row = cohort_resumed.cohort_rows()[0]
        self.assertFalse(bool(resumed_row["active"]))
        self.assertIsNotNone(resumed_row["boundary_event_residual_m"])
        self.assertTrue(
            math.isclose(
                float(continuous_row["dissolution_time_s"]),
                float(resumed_row["dissolution_time_s"]),
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            )
        )


if __name__ == "__main__":
    unittest.main()
