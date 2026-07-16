import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main_cuda.cu").read_text(encoding="utf-8")
PARAMS = (ROOT / "pf_params.h").read_text(encoding="utf-8")
RUNNER = (ROOT / "scripts/run_bounded_retry_equal_time_workstation.py").read_text(
    encoding="utf-8"
)
ANALYZER = (ROOT / "scripts/analyze_bounded_retry_equal_time.py").read_text(
    encoding="utf-8"
)


class BoundedRetryBdf2RuntimeContractTests(unittest.TestCase):
    def test_contract_is_separate_and_default_off(self):
        self.assertIn("ctot_retry_acceptance_contract", PARAMS)
        self.assertIn('"ZERO_REJECT_FIXED_STEP_V1"', MAIN)
        self.assertIn(
            '"ACTIVE_MANIFOLD_BDF2_BOUNDED_RETRY_PRODUCTION_V1"',
            (ROOT / "bounded_retry_bdf2_utils.h").read_text(encoding="utf-8"),
        )

    def test_contract_does_not_select_a_different_integrator(self):
        runtime = MAIN.split("const int ctot_candidate_runtime", 1)[1].split(
            "int ctot_bdf2_history_valid", 1
        )[0]
        self.assertIn("ctot_imex_bdf2_active_manifold_v1_runtime", runtime)
        self.assertIn("ctot_bounded_retry_contract_runtime", runtime)
        self.assertIn("changes_numerical_method=0", runtime)

    def test_specific_transport_failure_survives_iterate_restore(self):
        solve = MAIN.split("double initial_res_inf", 1)[1].split(
            "auto solve_ctot_transport_from_accepted", 1
        )[0]
        self.assertIn("transport_failure_code", solve)
        self.assertIn("CTOT_TRANSPORT_LINE_SEARCH_STAGNATION", solve)
        self.assertIn("CTOT_TRANSPORT_NONLINEAR_ITERATION_LIMIT", solve)
        self.assertIn("CTOT_TRANSPORT_FAILED_ITERATE_RESTORED", solve)

    def test_event_rollback_hashes_state_and_history(self):
        rollback = MAIN.split("verify_ctot_event_macro_rollback", 1)[1].split(
            "auto record_ctot_attempt", 1
        )[0]
        for token in (
            "d_ctot_accepted_r",
            "d_ctot_work_r",
            "d_ctot_saved_r",
            "d_phi_r",
            "d_phi_n_saved",
            "d_Y_r",
            "d_Y_n_saved",
            "d_ctot_history_nm1_r",
            "d_phi_history_nm1_r",
        ):
            self.assertIn(token, rollback)
        self.assertIn("CTOT_BOUNDED_RETRY_ROLLBACK", rollback)

    def test_old_zero_reject_reports_are_not_rewritten(self):
        self.assertNotIn(
            "reports/active_manifold_bdf2_v1/fixed_step_qualification.md",
            MAIN,
        )

    def test_equal_time_runner_uses_one_common_state_and_exact_window(self):
        namespace = {}
        case_block = RUNNER.split("CASES =", 1)[1].split("\n\n\ndef execute", 1)[0]
        exec("CASES =" + case_block, {}, namespace)
        cases = namespace["CASES"]
        self.assertEqual(set(cases), {"dt4", "dt8", "dt16", "fine_dt32"})
        for dt, steps in cases.values():
            self.assertAlmostEqual(dt * steps, 1.5625, places=14)
        self.assertIn("common_input = stage_input(args.host)", RUNNER)
        self.assertIn("run_case(args.host, common_input, name", RUNNER)
        appended = RUNNER.split("# Bounded-retry common-state", 1)[1].split(
            '"""', 1
        )[0]
        for forbidden in ("D_alpha=", "L_phi=", "gamma_Jm2=", "lambda_sm_m="):
            self.assertNotIn(forbidden, appended)

    def test_analyzer_contains_preregistered_gates(self):
        for expression in (
            'row["retry_fraction"] <= 0.01',
            'row["fallback_fraction"] <= 0.01',
            'row["retry_wall_overhead_fraction"] <= 0.05',
            'row["max_consecutive_fallback_macros"] <= 2',
            'row["max_accepted_subcycle_depth"] <= 2',
            'row["accepted_iteration_p99"] < 0.8',
            'row["Ctot_increment_relative_L2_error"] <= 0.02',
            'row["matrix_profile_capacity_weighted_relative_error"] <= 0.03',
            'row["interface_error_dx"] <= 0.25',
        ):
            self.assertIn(expression, ANALYZER)


if __name__ == "__main__":
    unittest.main()
