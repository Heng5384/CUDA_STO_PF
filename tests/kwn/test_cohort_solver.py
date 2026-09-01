"""Numerical gates for the no-bin discrete-cohort beta comparator."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from kwn_mvp.cohort_solver import Cohort, CohortSolver, sphere_volume_m3
from kwn_mvp.populations import PopulationParameters
from kwn_mvp.thermo_adapter import DiluteEquilibriumAdapter


def _parameters(*, diffusivity_m2_s: float = 5.0e-19) -> PopulationParameters:
    return PopulationParameters(
        name="beta",
        x_b=1.0,
        molar_volume_m3_mol=4.1009e-5,
        diffusivity_m2_s=diffusivity_m2_s,
        gamma_j_m2=5.0e-3,
        xeq_infinity=0.006,
        nucleation={"mode": "off"},
    )


def _total_inventory(*, cohorts: list[Cohort], matrix_xb: float, parameters: PopulationParameters) -> float:
    beta_fraction = sum(item.weight_m3 * sphere_volume_m3(item.radius_m) for item in cohorts)
    beta = beta_fraction * parameters.x_b / parameters.molar_volume_m3_mol
    matrix = (1.0 - beta_fraction) * matrix_xb / parameters.molar_volume_m3_mol
    return matrix + beta


def _solver(
    cohorts: list[Cohort],
    *,
    matrix_xb: float,
    diffusivity_m2_s: float = 5.0e-19,
    rtol: float = 1.0e-10,
) -> CohortSolver:
    parameters = _parameters(diffusivity_m2_s=diffusivity_m2_s)
    return CohortSolver(
        cohorts=cohorts,
        beta_parameters=parameters,
        matrix_molar_volume_m3_mol=4.1009e-5,
        total_b_mol_m3=_total_inventory(
            cohorts=cohorts, matrix_xb=matrix_xb, parameters=parameters
        ),
        equilibrium_adapter=DiluteEquilibriumAdapter(temperature_k=653.15),
        temperature_k=653.15,
        r_diss_m=1.0e-9,
        inventory_tolerance_relative=1.0e-10,
        rtol=rtol,
        atol_m=1.0e-18,
    )


def _mixed_cohorts() -> list[Cohort]:
    return [
        Cohort("small", 2.0e-9, 1.0e20),
        Cohort("large", 8.0e-9, 1.0e20),
    ]


class CohortSolverQualificationTests(unittest.TestCase):
    """C1--C8: finite cohorts remain deterministic and inventory closed."""

    def test_c1_zero_mobility(self) -> None:
        solver = _solver(_mixed_cohorts(), matrix_xb=0.0061, diffusivity_m2_s=0.0)
        before = solver.snapshot()
        before_rows = solver.cohort_rows()
        solver.advance_to(4.8e4)
        after = solver.snapshot()
        for field, value in before.as_dict().items():
            if field != "time_s":
                self.assertEqual(value, after.as_dict()[field])
        self.assertEqual(before_rows, solver.cohort_rows())

    def test_c2_single_growing_cohort(self) -> None:
        solver = _solver([Cohort("grow", 8.0e-9, 1.0e20)], matrix_xb=0.008)
        before = solver.snapshot()
        solver.advance_to(2.0e4)
        after = solver.snapshot()
        self.assertGreater(after.Rmean_m, before.Rmean_m)
        self.assertLess(after.matrix_xB, before.matrix_xB)
        self.assertLessEqual(after.inventory_relative_residual, 1.0e-10)

    def test_c3_single_dissolving_cohort_returns_inventory(self) -> None:
        solver = _solver([Cohort("dissolve", 2.0e-9, 1.0e19)], matrix_xb=0.005)
        before = solver.snapshot()
        solver.advance_to(2.0e4)
        after = solver.snapshot()
        event = solver.cohort_rows()[0]
        self.assertFalse(event["active"])
        self.assertIsNotNone(event["dissolution_time_s"])
        self.assertGreater(float(event["returned_inventory_mol_m3"]), 0.0)
        self.assertGreater(after.matrix_xB, before.matrix_xB)
        self.assertLessEqual(after.inventory_relative_residual, 1.0e-10)

    def test_c4_two_cohort_source_sink(self) -> None:
        solver = _solver(_mixed_cohorts(), matrix_xb=0.0061)
        before = solver.snapshot()
        solver.advance_to(2.0e4)
        after = solver.snapshot()
        rows = {str(row["initial_id"]): row for row in solver.cohort_rows()}
        self.assertFalse(bool(rows["small"]["active"]))
        self.assertGreater(float(rows["large"]["radius_m"]), 8.0e-9)
        self.assertLess(after.N_m0_m3, before.N_m0_m3)
        self.assertLessEqual(after.inventory_relative_residual, 1.0e-10)

    def test_c5_permutation_invariance(self) -> None:
        forward = _solver(_mixed_cohorts(), matrix_xb=0.0061)
        reverse = _solver(list(reversed(_mixed_cohorts())), matrix_xb=0.0061)
        forward.advance_to(2.0e4)
        reverse.advance_to(2.0e4)
        for field, value in forward.snapshot().as_dict().items():
            self.assertAlmostEqual(float(value), float(reverse.snapshot().as_dict()[field]), places=20)
        self.assertEqual(forward.cohort_rows(), reverse.cohort_rows())

    def test_c6_tolerance_convergence(self) -> None:
        snapshots = {}
        events = {}
        for rtol in (1.0e-6, 1.0e-8, 1.0e-10):
            solver = _solver(_mixed_cohorts(), matrix_xb=0.0061, rtol=rtol)
            solver.advance_to(2.0e4)
            snapshots[rtol] = solver.snapshot()
            events[rtol] = next(
                row["dissolution_time_s"]
                for row in solver.cohort_rows()
                if row["initial_id"] == "small"
            )
        fine = snapshots[1.0e-10]
        for field in ("N_m0_m3", "Rmean_m", "Rmean3_m3", "Sv_m_inv", "f_beta", "matrix_xB"):
            observed = float(getattr(snapshots[1.0e-8], field))
            reference = float(getattr(fine, field))
            self.assertLessEqual(abs(observed - reference) / max(abs(reference), 1.0e-300), 1.0e-3)
        self.assertLessEqual(
            abs(float(events[1.0e-8]) - float(events[1.0e-10])), 1.0e-5
        )

    def test_c7_restart(self) -> None:
        continuous = _solver(_mixed_cohorts(), matrix_xb=0.0061)
        continuous.advance_to(2.0e4)
        split = _solver(_mixed_cohorts(), matrix_xb=0.0061)
        split.advance_to(1.0e4)
        checkpoint = split.checkpoint()
        resumed = _solver(_mixed_cohorts(), matrix_xb=0.0061)
        resumed.restore_checkpoint(checkpoint)
        resumed.advance_to(2.0e4)
        for field, value in continuous.snapshot().as_dict().items():
            reference = float(resumed.snapshot().as_dict()[field])
            self.assertTrue(
                math.isclose(float(value), reference, rel_tol=1.0e-10, abs_tol=1.0e-18),
                msg=field,
            )
        for left, right in zip(continuous.cohort_rows(), resumed.cohort_rows()):
            self.assertEqual(left["initial_id"], right["initial_id"])
            self.assertEqual(left["active"], right["active"])
            self.assertTrue(
                math.isclose(float(left["radius_m"]), float(right["radius_m"]), rel_tol=1.0e-10)
            )

    def test_c8_exact_inventory(self) -> None:
        solver = _solver(_mixed_cohorts(), matrix_xb=0.0061)
        for time_s in (0.0, 10.0, 100.0, 1.0e3, 2.0e4):
            solver.advance_to(time_s)
            self.assertLessEqual(solver.snapshot().inventory_relative_residual, 1.0e-10)


if __name__ == "__main__":
    unittest.main()
