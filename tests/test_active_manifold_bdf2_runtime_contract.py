import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main_cuda.cu").read_text(encoding="utf-8")


class ActiveManifoldBdf2RuntimeContractTests(unittest.TestCase):
    def test_selector_is_distinct_and_old_selector_remains_exact(self):
        selectors = MAIN.split(
            "static int is_ctot_jichen_imex_bdf2_v1", 1
        )[1].split("static const char *ctot_time_integrator_name", 1)[0]
        self.assertIn('"ctot_jichen_imex_bdf2_v1"', selectors)
        self.assertIn('"ctot_jichen_imex_bdf2_active_manifold_v1"', selectors)
        self.assertIn("is_ctot_jichen_imex_bdf2_family", selectors)

    def test_context_is_versioned_and_checkpointed(self):
        self.assertIn(
            "QALPHA_AND_PHI_ENDPOINT_ACTIVE_MANIFOLD_CONTEXT_V1", MAIN
        )
        self.assertIn("ctot_bdf2_phase_context_version(&P)", MAIN)
        self.assertIn("FIXED_STEP_IMEX_BDF2_ACTIVE_MANIFOLD_V1", MAIN)
        self.assertIn(
            "QALPHA_EXACT_H_INVERSE_PLUS_PHI_ENDPOINT_TANGENT_CONE_V1", MAIN
        )
        self.assertIn("STORAGE_ULP64_AND_BOUND_TOL_1E12_V1", MAIN)
        self.assertIn(
            "BDF2_EVENT_PREFLIGHT_AND_ATOMIC_BE_SUBCYCLING_V1", MAIN
        )

    def test_context_kernel_writes_only_coefficient_buffers(self):
        kernel = MAIN.split(
            "__global__ void ctot_bdf2_prepare_active_manifold_context_kernel", 1
        )[1].split(
            "__global__ void ctot_bdf2_active_manifold_unhandled_preflight_kernel", 1
        )[0]
        self.assertIn("C_anchor_r[idx]", kernel)
        self.assertIn("phi_context_r[idx]", kernel)
        self.assertNotIn("C_n_r[idx] =", kernel)
        self.assertNotIn("phi_n_r[idx] =", kernel)
        self.assertNotIn("C_nm1_r[idx] =", kernel)
        self.assertNotIn("phi_nm1_r[idx] =", kernel)

    def test_active_context_bypasses_only_handled_preflight_events(self):
        selection = MAIN.split("const int dt_history_matches", 1)[1].split(
            "int finite_interface_violation_diag_written", 1
        )[0]
        self.assertIn(
            "ctot_bdf2_prepare_active_manifold_context_kernel", selection
        )
        self.assertIn(
            "ctot_bdf2_active_manifold_unhandled_preflight_kernel", selection
        )
        self.assertIn("invalid_context_count <= 0.5", selection)
        self.assertIn("ctot_bdf2_active_this_attempt = 1", selection)

    def test_old_checkpoint_migration_is_field_preserving_and_narrow(self):
        restart = MAIN.split("if (has_ctot_nm1) {", 1)[1].split(
            "} else {\n            printf(\"CTOT_IMEX_BDF2_HISTORY_LOAD", 1
        )[0]
        self.assertIn("FIXED_STEP_IMEX_BDF2_V1", restart)
        self.assertIn("PHI_EXTRAPOLATION_2N_MINUS_NM1_ULP64_V1", restart)
        self.assertIn("is_ctot_jichen_imex_bdf2_active_manifold_v1", restart)
        self.assertIn("authoritative_fields_unchanged=1", MAIN)
        self.assertIn("history_fields_unchanged=1", MAIN)

    def test_event_subcycling_remains_the_unhandled_safety_net(self):
        self.assertIn("ctot_bdf2_event_subcycle_active = 1", MAIN)
        self.assertIn("restore_ctot_event_macro_start()", MAIN)
        self.assertIn("ctot_bdf2_event_subcycle_depth <= 8", MAIN)


if __name__ == "__main__":
    unittest.main()
