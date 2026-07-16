import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main_cuda.cu").read_text()
HEADER = (ROOT / "phase_pdas_reduction.h").read_text()


class P2ExactActiveReductionSourceTests(unittest.TestCase):
    def test_trial_kernel_has_no_global_atomic_contention(self):
        kernel = HEADER.split(
            "__global__ void ctot_phase_pdas_compare_trial_active_kernel", 1
        )[1].split(
            "__global__ void ctot_phase_pdas_reduce_trial_summaries_kernel", 1
        )[0]
        self.assertNotIn("atomicAdd", kernel)
        self.assertNotIn("atomicCAS", kernel)
        self.assertIn("trial_active_code_r[idx]", kernel)
        self.assertIn("block_summaries[blockIdx.x]", kernel)

    def test_second_stage_is_fixed_tree_and_packet_is_device_resident(self):
        reduction = HEADER.split(
            "__global__ void ctot_phase_pdas_reduce_trial_summaries_kernel", 1
        )[1]
        self.assertIn("for (int stride = blockDim.x >> 1", reduction)
        self.assertIn("*packet = result", reduction)
        self.assertIn("active_set_stable", reduction)
        self.assertIn("trial_valid", reduction)

    def test_trial_decision_keeps_existing_kkt_and_merit_semantics(self):
        trial = MAIN.split(
            "ctot_phase_pdas_compare_trial_active_kernel<<<", 1
        )[1].split("if (sufficient_decrease)", 1)[0]
        self.assertIn("trial_pdas_packet.trial_valid", trial)
        self.assertIn("trial_kkt_linf <= nonlinear_tol", trial)
        self.assertIn("merit_after <= merit_before", trial)
        self.assertNotIn("PHASE_PDAS_INVALID_COUNT", trial)

    def test_p1_classification_and_legacy_solver_remain_present(self):
        self.assertIn("ctot_phase_pdas_classify_kernel", MAIN)
        self.assertIn("solve_ctot_phase_from_accepted", MAIN)
        self.assertIn("legacy_lagged_y", MAIN)


if __name__ == "__main__":
    unittest.main()
