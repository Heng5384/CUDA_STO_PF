import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_prompt7g_adjoint_identity import (  # noqa: E402
    audit_case,
    face_difference,
    face_divergence,
    harmonic_face_mobility,
    make_cases,
)


class Prompt7gAdjointOperatorTests(unittest.TestCase):
    def test_periodic_difference_and_incidence_are_exact_adjoint(self):
        rng = np.random.default_rng(7)
        mu = rng.normal(size=(5, 7, 6))
        q = [rng.normal(size=mu.shape) for _ in range(3)]
        spacing = (0.3, 0.8, 1.4)
        volume = np.prod(spacing)
        lhs = np.sum(mu * face_divergence(q, spacing)) * volume
        rhs = sum(np.sum(face_difference(mu, a, d)*q[a])*volume
                  for a, d in enumerate(spacing))
        self.assertLess(abs(lhs + rhs), 2.0e-13*max(abs(lhs), abs(rhs), 1.0))

    def test_harmonic_endpoint_is_exactly_zero(self):
        mobility = np.array([0.0, 1.0, 2.0, 0.0]).reshape(4, 1, 1)
        face = harmonic_face_mobility(mobility, 0).ravel()
        np.testing.assert_array_equal(face, [0.0, 4.0/3.0, 0.0, 0.0])

    def test_all_required_host_cases_pass(self):
        rows = [audit_case(case) for case in make_cases()]
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["status"] == "PASS" for row in rows))
        self.assertTrue(all(row["face_dissipation"] >= 0.0 for row in rows))
        self.assertTrue(all(row["closed_face_nonzero_flux_count"] == 0
                            for row in rows))


if __name__ == "__main__":
    unittest.main()
