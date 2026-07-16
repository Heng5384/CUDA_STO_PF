import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main_cuda.cu").read_text(encoding="utf-8")
PARAMS = (ROOT / "pf_params.h").read_text(encoding="utf-8")


class CtotStaggeredV1ContractTests(unittest.TestCase):
    def test_versioned_path_is_default_off(self):
        defaults = MAIN.split("static void params_default", 1)[1].split(
            "static void compute_nucleus_dimensions_gpu", 1
        )[0]
        self.assertIn('sizeof(P->ctot_numerics_contract), "legacy"', defaults)
        self.assertIn('sizeof(P->ctot_split_defect_policy), "OFF"', defaults)
        self.assertIn("P->ctot_max_coupling_correctors = 0;", defaults)
        self.assertIn("char ctot_numerics_contract[96]", PARAMS)
        self.assertIn("char ctot_split_defect_policy[40]", PARAMS)

    def test_physics_and_numerics_contracts_are_separate_and_fail_closed(self):
        validation = MAIN.split("if (production_v2)", 1)[1].split(
            "const int finite_lphi_mode", 1
        )[0]
        for token in (
            "pbte_ag2te_gp_coarse4_stoich_rd_v2",
            "ctot_fully_coupled_M3_polish_BE_v1",
            "ctot_jichen_staggered_defect1_BE_v1",
            "ALWAYS_ONE_POLISH",
            "OPTIONAL_ONE_POLISH",
            "ctot_max_coupling_correctors != 1",
            "D_compound != 0.0",
            "D_beta_for_calibration != 0.0",
        ):
            self.assertIn(token, MAIN)
        self.assertIn("production-v2 numerics mismatch", validation)
        self.assertIn("production-v2 physics requires", validation)

    def test_checkpoint_round_trip_records_both_contracts(self):
        loader = MAIN.split("const int contract_fields", 1)[1].split(
            "const int coarse_fields", 1
        )[0]
        validator = MAIN.split("const int current_versioned_contract", 1)[1].split(
            "const int current_coarse", 1
        )[0]
        writer = MAIN.split('\\"schema\\": \\"ctot_checkpoint_v1\\"', 1)[1].split(
            "fclose(meta_fp);", 1
        )[0]
        for token in (
            "PF_RESEARCH_MODEL",
            "ctot_numerics_contract",
            "ctot_split_defect_policy",
            "ctot_max_coupling_correctors",
        ):
            self.assertIn(token, loader)
            self.assertIn(token, validator)
            self.assertIn(token, writer)
        self.assertIn("provenance is missing or partial", validator)
        self.assertIn("mismatch: stored", validator)

    def test_staggered_step_is_predictor_phase_then_at_most_one_polish(self):
        outer = MAIN.split("const int outer_limit", 1)[1].split(
            "const int force_outer_elastic_reject", 1
        )[0]
        self.assertIn("ctot_staggered_v1_runtime\n                ? 1", outer)
        predictor = outer.index("solve_ctot_transport_from_accepted()")
        phase = outer.index("solve_ctot_phase_semismooth_pdas")
        closure = outer.index("const int final_closure_point")
        self.assertLess(predictor, phase)
        self.assertLess(phase, closure)
        final_closure = outer.split("const int final_closure_point", 1)[1].split(
            "outer_anchor_intact", 1
        )[0]
        self.assertEqual(final_closure.count("solve_ctot_transport_from_accepted()"), 1)
        self.assertNotIn("solve_ctot_phase_semismooth_pdas", final_closure)
        self.assertIn("evaluate_final_phase_residual", final_closure)

    def test_staggered_gate_does_not_reuse_outer_increment_as_convergence(self):
        gate = MAIN.split("const int primary_state_delta_pass", 1)[1].split(
            "double cycle_C_l2", 1
        )[0]
        self.assertIn("const int delta_pass = ctot_staggered_v1_runtime", gate)
        self.assertIn("? 1 :", gate)
        self.assertIn("full_transport_contract_pass", gate)
        self.assertIn("skipped_split_contract_pass", gate)
        self.assertIn("ctot_split_defect_skip_threshold", gate)
        self.assertIn("ctot_split_defect_hard_cap", gate)
        self.assertIn("phase_kkt_linf <= phase_tol", gate)
        self.assertIn("mechanical_gate_pass", gate)

    def test_split_metrics_include_atomic_attempt_outcome_and_solve_counts(self):
        self.assertIn("ctot_split_step_metrics.csv", MAIN)
        self.assertIn("transport_solves,phase_solves,mechanics_solves", MAIN)
        self.assertIn("ctot_transport_solves_this_attempt", MAIN)
        self.assertIn("ctot_phase_solves_this_attempt", MAIN)
        self.assertIn("ctot_mechanics_solves_this_attempt", MAIN)
        self.assertIn("energy_audit_pass,accepted", MAIN)


if __name__ == "__main__":
    unittest.main()
