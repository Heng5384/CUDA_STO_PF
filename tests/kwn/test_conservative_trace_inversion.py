"""Binary64 contracts for the production local TOF inverse."""

from __future__ import annotations

import unittest

import numpy as np

from kwn_mvp.conservative_remap import (
    ConservativeRemapError,
    _gauss_time_of_flight_intervals,
    _invert_descending_time_of_flight,
    _invert_time_of_flight,
)


def _inverse_cubic_velocity(radii_m: np.ndarray) -> np.ndarray:
    radii = np.asarray(radii_m, dtype=np.float64)
    return radii**-3


class ConservativeTraceInversionContracts(unittest.TestCase):
    """The production inverse must close its existing local GL2 interval."""

    def setUp(self) -> None:
        self.coordinates = np.asarray([1.0, 4.0], dtype=np.float64)
        self.cumulative = np.asarray([0.0, 63.75], dtype=np.float64)

    def _invert(self, targets: np.ndarray) -> np.ndarray:
        return _invert_time_of_flight(
            np.asarray(targets, dtype=np.float64),
            self.cumulative,
            self.coordinates,
            expected_sign=1.0,
            velocity_m_s=_inverse_cubic_velocity,
        )

    def _local_residual(self, radius_m: float, *, target_s: float) -> float:
        return float(
            _gauss_time_of_flight_intervals(
                np.asarray([1.0], dtype=np.float64),
                np.asarray([radius_m], dtype=np.float64),
                expected_sign=1.0,
                velocity_m_s=_inverse_cubic_velocity,
            )[0]
            - target_s
        )

    def test_exact_table_knots_return_the_original_binary64_coordinates(self) -> None:
        coordinates = np.asarray([1.0, 1.25, 2.0, 4.0], dtype=np.float64)
        increments = _gauss_time_of_flight_intervals(
            coordinates[:-1],
            coordinates[1:],
            expected_sign=1.0,
            velocity_m_s=_inverse_cubic_velocity,
        )
        cumulative = np.concatenate((np.asarray([0.0]), np.cumsum(increments, dtype=np.float64)))
        observed = _invert_time_of_flight(
            cumulative,
            cumulative,
            coordinates,
            expected_sign=1.0,
            velocity_m_s=_inverse_cubic_velocity,
        )
        np.testing.assert_array_equal(observed.view(np.uint64), coordinates.view(np.uint64))

    def test_nonlinear_gl2_inverse_is_repeatable_and_matches_the_known_root(self) -> None:
        target = np.asarray([50.0], dtype=np.float64)
        first = self._invert(target)
        second = self._invert(target)
        expected = np.asarray([np.float64(201.0) ** np.float64(0.25)], dtype=np.float64)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, expected)

    def test_binary64_terminal_choice_is_no_worse_than_adjacent_representable_radii(self) -> None:
        target = 50.123456789
        radius = float(self._invert(np.asarray([target], dtype=np.float64))[0])
        residual = abs(self._local_residual(radius, target_s=target))
        below = float(np.nextafter(radius, -np.inf))
        above = float(np.nextafter(radius, np.inf))
        self.assertLessEqual(residual, abs(self._local_residual(below, target_s=target)))
        self.assertLessEqual(residual, abs(self._local_residual(above, target_s=target)))

    def test_adjacent_binary64_interval_chooses_the_lower_residual_endpoint(self) -> None:
        left = np.float64(1.0)
        right = np.nextafter(left, np.inf)
        duration = right - left
        observed = _invert_time_of_flight(
            np.asarray([0.5 * duration], dtype=np.float64),
            np.asarray([0.0, duration], dtype=np.float64),
            np.asarray([left, right], dtype=np.float64),
            expected_sign=1.0,
            velocity_m_s=lambda radii: np.ones(np.asarray(radii).shape, dtype=np.float64),
        )
        self.assertEqual(observed[0], left)

    def test_descending_wrapper_is_the_exact_reflection_of_the_local_inverse(self) -> None:
        decreasing = np.asarray([4.0, 1.0], dtype=np.float64)
        target = np.asarray([50.0], dtype=np.float64)
        observed = _invert_descending_time_of_flight(
            target,
            self.cumulative,
            decreasing,
            expected_sign=1.0,
            velocity_m_s=_inverse_cubic_velocity,
        )
        reflected = _invert_time_of_flight(
            target,
            self.cumulative,
            -decreasing,
            expected_sign=1.0,
            velocity_m_s=lambda reflected_radius: _inverse_cubic_velocity(-reflected_radius),
        )
        np.testing.assert_array_equal(observed, -reflected)
        expected = np.float64(56.0) ** np.float64(0.25)
        self.assertLessEqual(abs(float(observed[0]) - float(expected)), 2.0 * np.spacing(expected))

    def test_target_outside_its_existing_table_still_fails_closed(self) -> None:
        for target in (-np.finfo(np.float64).tiny, np.nextafter(63.75, np.inf)):
            with self.assertRaises(ConservativeRemapError):
                self._invert(np.asarray([target], dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
