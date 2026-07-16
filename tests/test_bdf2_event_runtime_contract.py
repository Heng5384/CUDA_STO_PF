import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main_cuda.cu").read_text(encoding="utf-8")
PARAMS = (ROOT / "pf_params.h").read_text(encoding="utf-8")


class Bdf2EventRuntimeContractTests(unittest.TestCase):
    def test_event_selectors_are_declared_and_default_off(self):
        self.assertIn("int bdf2_event_preflight_v1", PARAMS)
        self.assertIn("int bdf2_event_be_subcycling_v1", PARAMS)
        defaults = MAIN.split("static void params_default", 1)[1].split(
            "static void compute_nucleus_dimensions_gpu", 1
        )[0]
        self.assertIn("P->bdf2_event_preflight_v1 = 0", defaults)
        self.assertIn("P->bdf2_event_be_subcycling_v1 = 0", defaults)

    def test_subcycling_requires_preflight_and_bdf2_contract(self):
        validation = MAIN
        self.assertIn("P->bdf2_event_be_subcycling_v1 &&", validation)
        self.assertIn("!P->bdf2_event_preflight_v1", validation)
        self.assertIn("!is_ctot_jichen_imex_bdf2_family(P)", validation)

    def test_preflight_uses_accepted_history_before_bdf2_selection(self):
        selection = MAIN.split("const int dt_history_matches", 1)[1].split(
            "int finite_interface_violation_diag_written", 1
        )[0]
        self.assertIn("ctot_bdf2_prepare_context_kernel", selection)
        self.assertIn("ctot_bdf2_event_preflight_kernel", selection)
        self.assertIn("d_ctot_saved_r, d_ctot_history_nm1_r", selection)
        self.assertIn("d_phi_n_saved, d_phi_history_nm1_r", selection)
        self.assertLess(
            selection.index("ctot_bdf2_event_preflight_kernel"),
            selection.index("ctot_bdf2_active_this_attempt = 1"),
        )

    def test_event_retry_is_atomic_and_covers_history_rebuild(self):
        start = MAIN.index(
            "const char *failure_stage =\n"
            "                    force_outer_elastic_reject"
        )
        end = MAIN.index("if (prepare_ctot_retry", start)
        rejection = MAIN[start:end]
        self.assertIn("ctot_bdf2_event_history_rebuild_pending", rejection)
        self.assertIn("restore_ctot_event_macro_start()", rejection)
        self.assertIn("ctot_bdf2_event_subcycle_depth *= 2", rejection)
        self.assertIn("ctot_bdf2_event_subcycle_depth <= 8", rejection)
        self.assertIn("goto ctot_bdf2_event_substep_begin", rejection)

    def test_intermediate_substeps_do_not_commit_history(self):
        accept = MAIN.split(
            "if (ctot_bdf2_event_subcycle_active) {\n"
            "                ctot_bdf2_event_extra_transport_solves",
            1,
        )[1].split("if (P.elastic_enabled)", 1)[0]
        self.assertIn("macro_time_committed=0 history_committed=0", accept)
        self.assertIn("goto ctot_bdf2_event_substep_begin", accept)
        self.assertNotIn("CTOT_IMEX_BDF2_HISTORY_COMMIT", accept)

    def test_event_endpoint_invalidates_history_and_checkpoint_records_it(self):
        commit = MAIN.split("Atomic accepted-history commit", 1)[1].split(
            "CUDA_CHECK(cudaMemcpy(d_ctot_accepted_r", 1
        )[0]
        self.assertIn("ctot_bdf2_history_valid = 0", commit)
        self.assertIn("ctot_bdf2_fallback_pending = 1", commit)
        self.assertIn("ctot_bdf2_event_history_rebuild_pending = 1", commit)
        self.assertIn('\\"bdf2_event_history_rebuild_pending\\"', MAIN)

    def test_stable_endpoint_contract_avoids_mixed_energy_evaluation(self):
        energy = MAIN.split("const int stable_endpoint_work_v2", 1)[1].split(
            "CTOT_IMEX_BDF2_ENERGY_WORK", 1
        )[0]
        self.assertIn("bdf2_stable_endpoint_chain_v2", energy)
        self.assertIn("F_np1_phi_n = stable_endpoint_work_v2", energy)
        self.assertIn("? NAN", energy)
        self.assertNotIn("isfinite ?", energy)


if __name__ == "__main__":
    unittest.main()
