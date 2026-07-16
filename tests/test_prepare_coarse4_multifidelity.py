import json
import pathlib
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_coarse4_multifidelity.py"


class PrepareCoarse4Tests(unittest.TestCase):
    def test_parameter_mapping_and_raw_storage_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "case"
            subprocess.run(
                ["python3", str(SCRIPT), "--output-root", str(output), "--grid", "16"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["coarse_model_name"],
                             "fixed_ctot_gp_coarse4_multifidelity_v1")
            self.assertEqual(manifest["dx_nm"], 1.0)
            self.assertEqual(manifest["lambda_nm"], 4.0)
            self.assertEqual(manifest["lambda_over_dx"], 4.0)
            self.assertAlmostEqual(manifest["W"], 1.0, places=14)
            self.assertAlmostEqual(manifest["kappa"], 2.0, places=13)
            self.assertEqual(len(manifest["fine_reference_hash"]), 64)
            self.assertEqual(len(manifest["coarse_calibration_hash"]), 64)
            self.assertEqual(manifest["mechanics_precision_mode"], "FP32_SPECTRAL")
            self.assertEqual(
                manifest["mechanics_acceptance_mode"],
                "FP32_NORMALIZED_BACKWARD_ERROR_V1",
            )
            self.assertEqual(manifest["mechanics_safety_factor"], 4.0)
            self.assertEqual(
                len(manifest["mechanics_double_oracle_contract_hash"]), 64
            )
            self.assertEqual(
                manifest["mechanics_residual_normalization_version"],
                "DEALIASED_REAL_DIVSIGMA_OVER_KMAX_STRESS_V1",
            )
            self.assertFalse(manifest["GP_source_reintegrated"])
            self.assertFalse(manifest["S3_reintegrated"])
            self.assertEqual(len(manifest["cases"]), 4)

            phi = np.fromfile(output / "phi_init.raw", dtype=np.float64)
            x_b = np.fromfile(output / "xB_init.raw", dtype=np.float64)
            ctot = np.fromfile(output / "Ctot_init.raw", dtype=np.float64)
            h = phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)
            self.assertEqual(phi.size, 16**3)
            np.testing.assert_allclose(ctot, h + (1.0 - h) * x_b,
                                       rtol=0.0, atol=0.0)
            self.assertTrue(np.all(ctot >= h))
            self.assertTrue(np.all(ctot <= 1.0))

            for case in manifest["cases"]:
                if case["elastic_enabled"]:
                    self.assertEqual(case["initial_state"], "planar_slab")
                    self.assertEqual(case["elastic_iter_max"], 40)

            finite = (output / "finite_lphi_r0p90_elastic0.params").read_text()
            quasi = (output / "quasi_equilibrium_elastic0.params").read_text()
            self.assertIn("PHASE_KINETICS_MODE=FINITE_LPHI_BE", finite)
            self.assertIn(
                "PHASE_KINETICS_MODE=QUASI_EQUILIBRIUM_FAST_INTERFACE_V1", quasi
            )
            for text in (finite, quasi):
                self.assertIn("coarse_interface_mobility_mode=off", text)
                self.assertIn("coarse_interface_mobility_a_M=0.00000000000000000e+00", text)
                self.assertIn("GP_population_mode=OFF", text)
                self.assertIn("diagnostic_rsmd_enabled=0", text)
                self.assertIn("mechanics_precision_mode=FP32_SPECTRAL", text)
                self.assertIn(
                    "mechanics_acceptance_mode=FP32_NORMALIZED_BACKWARD_ERROR_V1",
                    text,
                )
                self.assertIn(
                    "eta_floor_version=COARSE4_FP32_ETA_FLOOR_16_32_V1", text
                )


if __name__ == "__main__":
    unittest.main()
