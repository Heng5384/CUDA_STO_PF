import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PrepareJc4WidthGridSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(ROOT))
        from scripts import prepare_jc4_width_grid_sensitivity as module
        cls.module = module

    def test_matrix_is_equal_time_equal_inventory_and_selector_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            manifest = self.module.build(root)
            self.assertEqual(len(manifest["cases"]), 8)
            selected = 0
            by_direction = {}
            for case in manifest["cases"]:
                by_direction.setdefault(case["direction"], []).append(case)
                self.assertTrue(case["same_physical_time"])
                self.assertTrue(case["same_total_inventory"])
                self.assertEqual(case["grid"][1:], [2, 2])
                params = (root / "cases" / case["case_id"] / "runtime.params").read_text()
                self.assertIn("GP_population_mode=OFF", params)
                self.assertIn("ctot_finite_interface_antitrapping_enabled=0", params)
                self.assertIn("coarse_interface_mobility_a_M=0.00000000000000000e+00", params)
                if case["selected_research_model"]:
                    selected += 1
                    self.assertEqual(case["dx_nm"], 1.0)
                    self.assertEqual(case["lambda_nm"], 4.0)
                    self.assertIn(
                        "PF_RESEARCH_MODEL=fixed_ctot_ji_chen_coarse4_gp_v1",
                        params,
                    )
                else:
                    self.assertIn("PF_RESEARCH_MODEL=off", params)
                    self.assertEqual(
                        case["comparator_provenance"],
                        "JC4_SENSITIVITY_COMPARATOR_NOT_SELECTED_MODEL",
                    )
                metadata = json.loads(
                    (root / "cases" / case["case_id"] / "init_meta.json").read_text()
                )
                self.assertLess(metadata["inventory_match_abs"], 1.0e-12)
                self.assertLess(abs(metadata["matrix_xB_adjustment"]), 1.0e-7)
            self.assertEqual(selected, 2)
            for rows in by_direction.values():
                times = [float(row["elapsed_s"]) for row in rows]
                self.assertLess(max(times) - min(times), 1.0e-10)

    def test_strict_lphi_is_recomputed_for_every_width(self):
        contract_hash, reference_hash = self.module.provenance_hashes()
        values = []
        for lambda_nm in (3.0, 4.0, 5.0):
            _, derived = self.module.variant_params(
                1.0, lambda_nm, 0.1, lambda_nm == 4.0,
                f"lambda{lambda_nm}", contract_hash, reference_hash,
            )
            self.assertAlmostEqual(
                derived["L_phi_selected_code"],
                self.module.LPHI_RATIO * derived["L_phi_diff_code"],
                places=14,
            )
            values.append(derived["L_phi_selected_physical_m3_J_s"])
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])


if __name__ == "__main__":
    unittest.main()
