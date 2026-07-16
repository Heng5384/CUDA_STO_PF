import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main_cuda.cu").read_text(encoding="utf-8")


class CtotJichenLieBeV2ContractTests(unittest.TestCase):
    def test_contract_is_versioned_and_default_off(self):
        defaults = MAIN.split("static void params_default", 1)[1].split(
            "static void compute_nucleus_dimensions_gpu", 1
        )[0]
        self.assertIn("ctot_jichen_lie_be_v2", MAIN)
        self.assertIn('sizeof(P->ctot_numerics_contract), "legacy"', defaults)
        self.assertNotIn('"ctot_jichen_lie_be_v2"', defaults)

    def test_validator_requires_lie_policy_zero_correctors_and_no_acceleration(self):
        validation = MAIN.split("if (production_v2)", 1)[1].split(
            "const int finite_lphi_mode", 1
        )[0]
        for token in (
            "lie_be_v2",
            "LIE_NO_POST_PHASE_POLISH",
            "P->ctot_max_coupling_correctors == 0",
            "lie_be_v2 && policy_lie",
            "staggered_defect1 || lie_be_v2",
            "!outer_acceleration_off",
        ):
            self.assertIn(token, validation)

    def test_cold_transport_evaluator_has_explicit_phase_context(self):
        evaluator = MAIN.split(
            "auto evaluate_ctot_transport_residual_at_phi", 1
        )[1].split("auto evaluate_ctot_transport_residual =", 1)[0]
        self.assertIn("const double *phi_eval_r", evaluator)
        self.assertIn("Y_eval, phi_eval_r", evaluator)
        self.assertIn("d_ctot_work_r, phi_eval_r", evaluator)
        self.assertIn("phi_eval_r, d_xB_r, d_mu_x_r", evaluator)
        self.assertIn("ctot_transport_old_r, phi_eval_r, d_divJ_r", evaluator)
        self.assertNotIn("d_ctot_work_r, d_phi_r", evaluator)

    def test_operator_order_is_transport_cold_audit_then_phase(self):
        outer = MAIN.split("const int outer_limit", 1)[1].split(
            "const int force_outer_elastic_reject", 1
        )[0]
        predictor = outer.index("solve_ctot_transport_from_accepted()")
        cold = outer.index("evaluate_ctot_transport_residual_at_phi")
        phase = outer.index("solve_ctot_phase_semismooth_pdas")
        final_phi = outer.index("CTOT_OUTER_POST_PHASE_TRANSPORT_INVALID")
        self.assertLess(predictor, cold)
        self.assertLess(cold, phase)
        self.assertLess(phase, final_phi)
        cold_call = outer[cold:phase]
        self.assertIn("ctot_transport_phi_r", cold_call)
        self.assertIn("const double *ctot_transport_phi_r = d_phi_r", MAIN)

    def test_lie_branch_never_executes_post_phase_transport_polish(self):
        closure = MAIN.split("if (final_closure_point)", 1)[1].split(
            "outer_anchor_intact", 1
        )[0]
        lie_branch = closure.split(
            "if (ctot_method_consistent_split_runtime)", 1
        )[1].split(
            "} else {", 1
        )[0]
        self.assertIn("CTOT_METHOD_NO_POST_PHASE_TRANSPORT", lie_branch)
        self.assertNotIn("solve_ctot_transport_from_accepted", lie_branch)
        self.assertNotIn("final_fixed_phi_polish_skipped = 1", lie_branch)

    def test_hard_gate_uses_old_phi_method_residual_only_for_lie(self):
        gate = MAIN.split("const int full_transport_contract_pass", 1)[1].split(
            "double cycle_C_l2", 1
        )[0]
        self.assertIn("method_transport_contract_pass", gate)
        self.assertIn("method_transport_res_inf <= transport_tol", gate)
        self.assertIn(
            "ctot_method_consistent_split_runtime\n"
            "                        ? method_transport_contract_pass",
            gate,
        )
        self.assertIn("phase_kkt_linf <= phase_tol", gate)
        self.assertNotIn(
            "final_transport_res_inf <= transport_tol && ctot_lie_be_v2_runtime",
            gate,
        )

    def test_substep_energy_and_split_work_are_audited(self):
        for token in (
            "F_after_method_transport",
            "transport_substep_delta_F",
            "phase_substep_delta_F",
            "splitting_work_residual",
            "CTOT_LIE_BE_SUBSTEP_ENERGY",
            "lie_substep_energy_pass",
        ):
            self.assertIn(token, MAIN)

    def test_metrics_distinguish_method_and_final_phi_residuals(self):
        header = MAIN.split('"physical_step,attempt_id,dt_try,numerics_contract,"', 1)[1]
        self.assertIn("method_transport_residual", header)
        self.assertIn("final_phi_split_residual", header)
        self.assertIn("method_transport_gate_pass", header)


if __name__ == "__main__":
    unittest.main()
