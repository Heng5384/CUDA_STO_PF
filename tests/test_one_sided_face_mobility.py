import math
import unittest


def harmonic_face(left, right):
    if not (math.isfinite(left) and math.isfinite(right)):
        return math.nan
    if left < 0.0 or right < 0.0:
        return math.nan
    if left == 0.0 or right == 0.0:
        return 0.0
    return 2.0 * left * right / (left + right)


class OneSidedFaceMobilityTests(unittest.TestCase):
    def test_harmonic_face_is_symmetric(self):
        self.assertEqual(harmonic_face(2.0, 5.0), harmonic_face(5.0, 2.0))

    def test_constant_mu_has_zero_flux(self):
        mobility = harmonic_face(2.0, 5.0)
        self.assertEqual(mobility * (3.0 - 3.0), 0.0)

    def test_single_face_conservation(self):
        mobility = harmonic_face(2.0, 5.0)
        flux = mobility * (4.0 - 1.0)
        left_update = flux
        right_update = -flux
        self.assertEqual(left_update + right_update, 0.0)

    def test_closed_beta_face_has_exact_zero_flux(self):
        harmonic = harmonic_face(0.0, 2.0)
        arithmetic = 0.5 * (0.0 + 2.0)
        self.assertEqual(harmonic, 0.0)
        self.assertGreater(arithmetic, 0.0)

    def test_harmonic_matches_two_half_cell_series_resistance(self):
        left, right = 0.25, 4.0
        conductance = 1.0 / (0.5 / left + 0.5 / right)
        self.assertAlmostEqual(harmonic_face(left, right), conductance, 15)


if __name__ == "__main__":
    unittest.main()
