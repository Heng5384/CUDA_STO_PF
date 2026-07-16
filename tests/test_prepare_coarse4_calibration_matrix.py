import json
import pathlib
import tempfile
import unittest

import numpy as np

from scripts.prepare_coarse4_calibration_matrix import (
    A_M_DETERMINISTIC_SCAN,
    A_M_SCAN_PROTOCOL_VERSION,
    DEFAULT_OBSERVATION_FO_FINE,
    OBSERVATION_WINDOW_VERSION,
    build_matrix,
)


class PrepareCoarse4CalibrationMatrixTests(unittest.TestCase):
    def test_matched_sources_and_frozen_single_parameter_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "matrix"
            manifest = build_matrix(root, (0.0, 2.0), 64.0, 0.01)
            self.assertEqual(manifest["weights_frozen_before_scan"], {
                "matrix_flux": 0.4, "beta_inventory": 0.4, "trajectory": 0.2,
            })
            self.assertFalse(manifest["heldout_refit_allowed"])
            self.assertEqual(manifest["a_M_scan_frozen_before_results"], [0.0, 2.0])
            self.assertEqual(manifest["observation_Fo_fine"], 0.01)
            self.assertEqual(
                manifest["observation_window_version"], OBSERVATION_WINDOW_VERSION
            )
            protocol = manifest["a_M_scan_protocol"]
            self.assertEqual(protocol["version"], A_M_SCAN_PROTOCOL_VERSION)
            self.assertEqual(protocol["ordered_union"], list(A_M_DETERMINISTIC_SCAN))
            self.assertTrue(protocol["frozen_before_new_results"])
            self.assertFalse(protocol["adaptive_refit"])
            self.assertEqual(protocol["free_parameter_count"], 1)
            self.assertFalse(manifest["GP_source_reintegrated"])
            self.assertFalse(manifest["S3_reintegrated"])
            self.assertEqual(len(manifest["cases"]), 10)

            for direction in ("growth", "dissolution"):
                rows = [row for row in manifest["cases"]
                        if row["direction"] == direction]
                self.assertEqual(len({row["common_sharp_state_sha256"]
                                      for row in rows}), 1)
                reference = [row for row in rows
                             if row["calibration_role"] == "reference"]
                self.assertEqual(len(reference), 1)
                for row in rows:
                    case = root / "cases" / row["case_id"]
                    phi = np.fromfile(case / "phi_init.raw", dtype=np.float64)
                    x_b = np.fromfile(case / "xB_init.raw", dtype=np.float64)
                    ctot = np.fromfile(case / "Ctot_init.raw", dtype=np.float64)
                    h = phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)
                    np.testing.assert_allclose(
                        ctot, h + (1.0 - h) * x_b, rtol=0.0, atol=0.0
                    )
                    params = (case / "runtime.params").read_text()
                    self.assertIn("diagnostic_rsmd_enabled=0", params)
                    self.assertIn("elastic_enabled=0", params)
                    self.assertIn("ctot_finite_interface_antitrapping_enabled=0", params)
                    self.assertIn("GP_population_mode=OFF", params)

    def test_default_scan_is_sorted_unique_and_contains_all_declared_stages(self):
        self.assertEqual(A_M_DETERMINISTIC_SCAN, tuple(sorted(
            set(A_M_DETERMINISTIC_SCAN)
        )))
        self.assertEqual(A_M_DETERMINISTIC_SCAN[0], 0.0)
        self.assertEqual(A_M_DETERMINISTIC_SCAN[-1], 64.0)
        self.assertIn(0.0625, A_M_DETERMINISTIC_SCAN)
        self.assertIn(16.0, A_M_DETERMINISTIC_SCAN)
        self.assertEqual(DEFAULT_OBSERVATION_FO_FINE, 3.0)


if __name__ == "__main__":
    unittest.main()
