import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_finite_interface_stefan import summarize  # noqa: E402


class FiniteInterfaceStefanSummaryTests(unittest.TestCase):
    def test_doubling_change_uses_only_complete_equal_dt_rows(self):
        def row(case, factor, ratio, retry="0", accepted="4000"):
            return {
                "case": case, "source_csv": "test.csv", "dx_nm": "0.1",
                "interface_resolution": "6", "L_phi_factor": str(factor),
                "finite_interface_correction": "1", "old_step": "3500",
                "new_step": "4000", "accepted_steps": accepted,
                "requested_steps": "4000", "retry_count": retry,
                "equal_requested_dt": str(retry == "0"),
                "phi_velocity_ratio_to_sharp": str(ratio),
                "local_stefan_ratio_phi_over_flux": "1",
                "mu_surface_new_excess": "0", "runtime_mass_error_max": "0",
                "energy_balance_rel_max": "0",
            }
        result = summarize([
            row("L8", 8, 1.02), row("L16", 16, 1.03),
            row("L32_retry", 32, 1.01, retry="1"),
        ])
        by_case = {item["case"]: item for item in result}
        self.assertAlmostEqual(
            by_case["L16"]["doubling_change_from_previous"], 0.01
        )
        self.assertFalse(by_case["L32_retry"]["eligible_equal_dt_complete"])


if __name__ == "__main__":
    unittest.main()
