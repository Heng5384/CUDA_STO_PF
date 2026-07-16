#!/usr/bin/env python3
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "main_cuda.cu").read_text(encoding="utf-8")


def anchored_picard(old_state, coupling, dt, iterations):
    value = old_state
    for _ in range(iterations):
        value = old_state + dt * coupling(value)
    return value


def incorrectly_accumulated_picard(old_state, coupling, dt, iterations):
    value = old_state
    for _ in range(iterations):
        value = value + dt * coupling(value)
    return value


def detects_two_cycle(x_new, x_prev, x_two_back, tol):
    returns_to_two_back = abs(x_new - x_two_back) <= tol
    differs_from_previous = abs(x_new - x_prev) > 10.0 * tol
    return returns_to_two_back and differs_from_previous


class CtotOuterContractTests(unittest.TestCase):
    def test_outer_phase_relative_tolerance_uses_pre_solve_kkt(self):
        self.assertIn(
            "fmax(phase_inner_initial_kkt, 1.0e-300)", SOURCE
        )
        self.assertNotIn(
            "initial_phase_residual = fmax(phase_kkt_linf", SOURCE
        )

    def test_outer_old_state_anchor_reference(self):
        result = anchored_picard(1.0, lambda x: 0.5 * x, 0.1, 8)
        self.assertAlmostEqual(result, 1.0 / 0.95, places=8)

    def test_outer_does_not_accumulate_time(self):
        anchored = anchored_picard(1.0, lambda x: 0.5 * x, 0.1, 8)
        accumulated = incorrectly_accumulated_picard(
            1.0, lambda x: 0.5 * x, 0.1, 8
        )
        self.assertNotAlmostEqual(anchored, accumulated, places=6)

    def test_transport_residual_uses_accepted_C(self):
        self.assertIn(
            "const double *ctot_transport_old_r = d_ctot_saved_r;", SOURCE
        )
        self.assertIn(
            "d_ctot_work_r, ctot_transport_old_r, d_divJ_r", SOURCE
        )

    def test_phase_update_uses_accepted_phi(self):
        self.assertIn(
            "cufftExecD2Z(plan_r2c_phi, d_phi_n_saved, d_Y_k)", SOURCE
        )
        self.assertIn(
            "d_Y_k, d_phi_rhs_k, KS.d_k2, d_phi_k", SOURCE
        )

    def test_final_transport_audit_is_read_only_in_C(self):
        self.assertIn(
            "d_Y_r, 0, ctot_use_spectral, dt_transport,\n"
            "                        &final_transport_res_inf", SOURCE
        )

    def test_two_cycle_detection(self):
        self.assertTrue(detects_two_cycle(1.0, 2.0, 1.0, 1.0e-8))
        self.assertFalse(detects_two_cycle(1.0, 1.0, 1.0, 1.0e-8))
        self.assertIn("CTOT_ELASTIC_OUTER_TWO_CYCLE", SOURCE)


if __name__ == "__main__":
    unittest.main()
