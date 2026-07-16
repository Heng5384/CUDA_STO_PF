import math
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class Jc4ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(ROOT))
        from scripts import prepare_jc4_research_model as model
        cls.model = model

    def test_strict_limits_and_selected_values(self):
        expected = {
            380.0: (0.6260938872648409, 0.5634844985383568),
            400.0: (0.7790818896991268, 0.7011737007292141),
        }
        for temperature, (limit_expected, selected_expected) in expected.items():
            row, _ = self.model.parameter_row(temperature)
            self.assertTrue(math.isclose(row["L_phi_diff_code"], limit_expected,
                                         rel_tol=2.0e-13))
            self.assertTrue(math.isclose(row["L_phi_selected_code"], selected_expected,
                                         rel_tol=2.0e-13))
            self.assertEqual(row["zeta0_strict"], 1.0)
            self.assertEqual(row["D_beta_code"], 0.0)

    def test_runtime_contract_has_no_compensating_operator(self):
        values = self.model.runtime_params(400.0, "a" * 64, "b" * 64)
        self.assertEqual(values["PF_RESEARCH_MODEL"], self.model.MODEL_NAME)
        self.assertEqual(values["PHASE_KINETICS_MODE"], "FINITE_LPHI_BE")
        self.assertEqual(values["coarse_interface_mobility_mode"], "off")
        self.assertEqual(values["coarse_interface_mobility_a_M"], 0.0)
        self.assertEqual(values["ctot_finite_interface_antitrapping_enabled"], 0)
        self.assertEqual(values["GP_population_mode"], "OFF")
        for key in (
            "diagnostic_rsmd_enabled",
            "gp_initial_population_enabled",
            "gp_growth_enabled",
            "enable_legacy_gp_storage_coupling",
            "enable_runtime_nucleus_library",
        ):
            self.assertEqual(values[key], 0)

    def test_generator_writes_default_off_pf_only_params(self):
        row, _ = self.model.parameter_row(380.0)
        values = self.model.runtime_params(380.0, "a" * 64, "b" * 64)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "jc4.params"
            self.model.write_params(path, values)
            text = path.read_text()
        self.assertIn("PF_RESEARCH_MODEL=fixed_ctot_ji_chen_coarse4_gp_v1", text)
        self.assertIn("GP_population_mode=OFF", text)
        self.assertIn("coarse_interface_mobility_a_M=0.00000000000000000e+00", text)
        self.assertIn(f"L_phi={row['L_phi_selected_code']:.17e}", text)


if __name__ == "__main__":
    unittest.main()
