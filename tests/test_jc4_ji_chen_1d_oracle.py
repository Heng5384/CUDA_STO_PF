import pathlib
import sys
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.jc4_ji_chen_1d_oracle import (  # noqa: E402
    FixedCtotJiChen1DOracle,
    JiChen1DOracle,
    OracleConfig,
    h_switch,
    run_limiting_suite,
)


class JiChenOracleTests(unittest.TestCase):
    def test_direct_source_identity_and_periodic_mass_rate(self):
        oracle = JiChen1DOracle(OracleConfig(cells=48, final_time_code=0.02))
        terms = oracle.differential_terms(oracle.initial_state())
        np.testing.assert_allclose(
            terms["C_rate_from_chain"], terms["div_flux"], rtol=0.0, atol=2.0e-13
        )
        self.assertLess(abs(float(np.sum(terms["div_flux"]))), 2.0e-13)

    def test_logit_state_is_strictly_bounded(self):
        oracle = JiChen1DOracle(OracleConfig(cells=48, final_time_code=0.02))
        state = oracle.initial_state()
        state[48:] = np.linspace(-30.0, 30.0, 48)
        x = oracle.unpack(state)["x"]
        self.assertTrue(np.all(x >= 0.0))
        self.assertTrue(np.all(x <= 1.0))
        self.assertGreater(x[1], 0.0)
        self.assertLess(x[-2], 1.0)

    def test_saturated_beta_complement_does_not_cancel(self):
        phi = np.array([1.0 - 5.0e-9], dtype=np.float64)
        direct_complement = h_switch(1.0 - phi)[0]
        cancelled_complement = 1.0 - h_switch(phi)[0]
        self.assertGreater(direct_complement, 0.0)
        self.assertGreater(cancelled_complement, direct_complement * 1.0e6)
        self.assertAlmostEqual(direct_complement, 1.25e-24, delta=1.0e-31)

    def test_small_alpha_core_is_not_artificially_phase_frozen(self):
        oracle = JiChen1DOracle(OracleConfig(cells=48, final_time_code=0.02))
        state = oracle.initial_state()
        values = oracle.unpack(state)
        core = int(np.argmin(values["alpha"]))
        # A small local perturbation gives a finite discrete phase residual in
        # a cell that the historical alpha<1e-5 guard would have frozen.
        state[core] -= 1.0e-7
        terms = oracle.differential_terms(state)
        self.assertLess(terms["alpha"][core], 1.0e-5)
        self.assertNotEqual(terms["phi_rate"][core], 0.0)
        self.assertAlmostEqual(
            terms["C_rate_from_chain"][core], terms["div_flux"][core],
            delta=2.0e-13,
        )

    def test_fixed_ctot_form_has_linear_mass_identity_and_local_kkt(self):
        oracle = FixedCtotJiChen1DOracle(
            OracleConfig(cells=64, initial_beta_half_width_nm=8.0,
                         initial_matrix_xB=0.03, final_time_code=0.02)
        )
        state = oracle.initial_state()
        terms = oracle.differential_terms(state)
        np.testing.assert_allclose(
            terms["C_rate_from_chain"], terms["div_flux"],
            rtol=0.0, atol=2.0e-13,
        )
        self.assertLess(abs(float(np.sum(terms["C_rate"]))), 2.0e-13)
        self.assertGreaterEqual(float(np.min(terms["storage_slack"])), -2.0e-15)

    def test_limiting_examples_and_control_order(self):
        rows = run_limiting_suite()
        by_name = {row["case"]: row for row in rows}
        self.assertTrue(all(row["status"] == "PASS" for row in rows))
        reaction = by_name["reaction_controlled_growth"]["displacement_nm"]
        mixed = by_name["mixed_controlled_growth"]["displacement_nm"]
        diffusion = by_name["diffusion_controlled_growth"]["displacement_nm"]
        self.assertGreater(diffusion, mixed)
        self.assertGreater(mixed, reaction)
        self.assertLess(max(row["mass_error_rel"] for row in rows), 1.0e-10)
        self.assertLess(max(row["source_identity_Linf"] for row in rows), 2.0e-11)


if __name__ == "__main__":
    unittest.main()
