#!/usr/bin/env python3
import math
import unittest

import numpy as np

from scripts.pf_particle_handoff import extract_handoff, h_of_phi


class PfParticleHandoffTest(unittest.TestCase):
    def test_two_particle_mass_roundtrip(self) -> None:
        shape = (48, 48, 48)
        grid = np.indices(shape, dtype=np.float64)
        phi = np.zeros(shape, dtype=np.float64)
        for center, radius in (((14.0, 24.0, 24.0), 6.0),
                               ((34.0, 24.0, 24.0), 8.0)):
            distance = np.sqrt(sum((grid[i] - center[i])**2 for i in range(3)))
            phi = np.maximum(phi, 0.5 * (1.0 - np.tanh((distance - radius) / 1.5)))
        h = h_of_phi(phi)
        qalpha = (1.0 - h) * 0.012
        ctot = qalpha + h
        result = extract_handoff(
            ctot, phi, dx_nm=1.0, v_B=1.0, gp_inventory=7.25,
        )
        ledger = result["ledger"]
        self.assertEqual(result["particle_count"], 2)
        self.assertLessEqual(abs(ledger["roundtrip_error_rel"]), 1.0e-12)
        self.assertLessEqual(abs(ledger["particle_partition_error_rel"]), 1.0e-12)
        self.assertEqual(result["threshold_stability"]["phi_ge_0.5"]["particle_count"], 2)
        radii = sorted(p["equivalent_radius_nm"] for p in result["particles"])
        self.assertGreater(radii[0], 5.0)
        self.assertGreater(radii[1], radii[0])

    def test_rejects_negative_matrix_storage(self) -> None:
        phi = np.ones((4, 4, 4), dtype=np.float64) * 0.8
        ctot = np.zeros_like(phi)
        with self.assertRaisesRegex(ValueError, "negative q_alpha"):
            extract_handoff(ctot, phi, dx_nm=1.0)


if __name__ == "__main__":
    unittest.main()
