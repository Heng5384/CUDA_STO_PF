import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main_cuda.cu").read_text(encoding="utf-8")
KERNELS = (ROOT / "cuda_kernels.cu").read_text(encoding="utf-8")


class PfCtotPerformanceP1ContractTests(unittest.TestCase):
    def test_nonelastic_main_jacobian_skips_independent_fd_path(self):
        solver = MAIN.split("auto solve_ctot_phase_semismooth_pdas", 1)[1]
        solver = solver.split("const int coupled_outer_required", 1)[0]
        elastic = solver.split("if (phi_elastic_coupling_enabled)", 1)[1]
        elastic = elastic.split("const int perf_classify", 1)[0]
        self.assertIn("ctot_phase_independent_phi_perturb_kernel", elastic)
        self.assertIn("ctot_phase_independent_x_perturb_kernel", elastic)
        nonelastic = solver.split(
            "if (!phi_elastic_coupling_enabled)", 1
        )[1].split("const int perf_pcg_solve", 1)[0]
        self.assertIn(
            "ctot_phase_exact_nonelastic_local_jacobian_kernel", nonelastic
        )
        self.assertNotIn("ctot_phase_independent_phi_perturb_kernel", nonelastic)
        self.assertNotIn("ctot_phase_independent_x_perturb_kernel", nonelastic)

    def test_pdas_statistics_use_block_aggregated_atomics(self):
        restore = MAIN.split(
            "__global__ void ctot_phase_restore_continuous_context_kernel", 1
        )[1].split("__global__ void ctot_phase_fd_perturb_kernel", 1)[0]
        classify = MAIN.split(
            "__global__ void ctot_phase_pdas_classify_kernel", 1
        )[1].split("__global__ void ctot_phase_fd_local_jacobian_kernel", 1)[0]
        for kernel in (restore, classify):
            self.assertIn("__shared__", kernel)
            self.assertIn("__syncthreads();", kernel)
            self.assertIn("if (threadIdx.x == 0)", kernel)

    def test_pcg_scalars_and_status_packet_remain_on_device(self):
        solver = MAIN.split("const int perf_pcg_solve", 1)[1]
        solver = solver.split("Full reduced-space Newton", 1)[0]
        self.assertIn("ctot_phase_dot_device", solver)
        self.assertIn("ctot_phase_pcg_prepare_alpha_kernel", solver)
        self.assertIn("ctot_phase_cg_update_device_scalar_kernel", solver)
        self.assertIn("ctot_phase_pcg_prepare_beta_kernel", solver)
        self.assertIn("ctot_phase_cg_direction_device_scalar_kernel", solver)
        self.assertIn("ctot_phase_pcg_pack_status_kernel", solver)
        self.assertNotIn("const double cg_alpha", solver)
        self.assertNotIn("const double cg_beta", solver)

    def test_reduction_workspace_is_persistent_and_tree_is_preserved(self):
        implementation = KERNELS.split(
            "void gpu_reduce_workspace_reserve", 1
        )[1].split("// ============", 1)[0]
        compatibility = implementation.split("double gpu_reduce_sum(", 1)[1]
        self.assertIn("gpu_reduce_sum_to_device", compatibility)
        self.assertNotIn("cudaMalloc", compatibility)
        self.assertNotIn("cudaDeviceSynchronize", compatibility)
        device_reduce = implementation.split(
            "void gpu_reduce_sum_to_device", 1
        )[1].split("double gpu_reduce_sum(", 1)[0]
        self.assertIn("reduce_sum_kernel<<<", device_reduce)
        self.assertIn("cudaMemcpyAsync", device_reduce)

    def test_profiler_is_default_off(self):
        self.assertIn("P->ctot_performance_profile_enabled = 0;", MAIN)
        self.assertIn(
            "if (ctot_candidate_runtime && P.ctot_performance_profile_enabled)",
            MAIN,
        )


if __name__ == "__main__":
    unittest.main()
