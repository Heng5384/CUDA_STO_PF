import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main_cuda.cu").read_text(encoding="utf-8")
KERNELS = (ROOT / "cuda_kernels.cu").read_text(encoding="utf-8")
THERMO = (ROOT / "thermo_utils.h").read_text(encoding="utf-8")


class Coarse4RuntimeContractTests(unittest.TestCase):
    def test_new_research_model_is_default_off(self):
        defaults = MAIN.split("static void params_default", 1)[1].split(
            "static void compute_nucleus_dimensions_gpu", 1
        )[0]
        self.assertIn('P->PF_RESEARCH_MODEL, sizeof(P->PF_RESEARCH_MODEL), "off"', defaults)
        self.assertIn('P->PHASE_KINETICS_MODE, sizeof(P->PHASE_KINETICS_MODE)', defaults)
        self.assertIn('"FINITE_LPHI_BE"', defaults)
        self.assertIn('P->coarse_interface_mobility_a_M = 0.0;', defaults)
        self.assertIn('P->GP_population_mode, sizeof(P->GP_population_mode), "OFF"', defaults)

    def test_coarse_contract_is_explicit_and_fail_closed(self):
        validation = MAIN.split("if (coarse4_model)", 1)[1].split(
            "if (!P->thermo_convex_extrapolation_enabled)", 1
        )[0]
        for token in (
            "4.0e-9",
            "1.0e-9",
            "ctot_mimetic_be",
            "ctot_phase_semismooth_pdas_enabled",
            "ctot_finite_interface_antitrapping_enabled",
            "fine_reference_hash",
            "coarse_calibration_hash",
        ):
            self.assertIn(token, validation)
        self.assertIn("fixed_ctot_gp_coarse4_multifidelity_v1", MAIN)
        self.assertIn("strcmp(P->coarse_model_name", validation)
        self.assertIn("P->PF_RESEARCH_MODEL", validation)

    def test_ji_chen_coarse4_contract_is_separate_and_fail_closed(self):
        selectors = MAIN.split("static int is_legacy_coarse4_research_model", 1)[1].split(
            "static const char *coarse4_mechanics_precision_mode", 1
        )[0]
        validation = MAIN.split("const int jc4_model", 1)[1].split(
            "if (coarse4_model)", 1
        )[0]
        self.assertIn("fixed_ctot_gp_coarse4_multifidelity_v1", selectors)
        self.assertIn("fixed_ctot_ji_chen_coarse4_gp_v1", selectors)
        self.assertIn("is_legacy_coarse4_research_model(P) || is_jc4_research_model(P)", selectors)
        for token in (
            "FINITE_LPHI_BE",
            "mobility_mode=off",
            "a_M=0",
            "GP_population_mode=OFF",
            "adaptive_logit_feasible_ctot_v1",
        ):
            self.assertIn(token, validation)
        self.assertIn("P->coarse_model_name", MAIN)
        self.assertIn("P->PF_RESEARCH_MODEL", MAIN)

    def test_quasi_equilibrium_solves_energy_kkt_at_fixed_ctot(self):
        residual = MAIN.split("__global__ void ctot_phase_final_residual_kernel", 1)[1].split(
            "__global__ void ctot_phase_projected_defect_trial_kernel", 1
        )[0]
        self.assertIn("const double residual = quasi_equilibrium_mode", residual)
        self.assertIn("? energy_gradient", residual)
        self.assertIn("const double metric_step = quasi_equilibrium_mode", residual)
        self.assertIn("dt / (bdf2_active ? 1.5 : 1.0)", residual)
        solver = MAIN.split("auto solve_ctot_phase_semismooth_pdas", 1)[1].split(
            "const int coupled_outer_required", 1
        )[0]
        self.assertIn(
            "ctot_quasi_equilibrium_phase ? 1.0 : phase_rate_dt", solver
        )
        self.assertIn("ctot_phase_reconstruct_x_fixed_C_kernel", solver)
        self.assertNotIn("cudaMemcpy(d_ctot_work_r, ctot_scratch_r", solver)

    def test_quasi_equilibrium_energy_audit_is_inequality_not_compensation(self):
        final_audit = MAIN.split("double energy_balance_residual", 1)[1].split(
            "const int forced_first_attempt_reject", 1
        )[0]
        self.assertIn("ctot_quasi_equilibrium_phase", final_audit)
        self.assertIn("energy_balance_residual <=", final_audit)
        self.assertIn("energy_balance_rel <=", final_audit)
        self.assertIn("W_finite_interface", final_audit)
        self.assertNotIn("coarse_interface_mobility_a_M", final_audit)

    def test_single_parameter_mobility_preserves_harmonic_face_operator(self):
        helper = THERMO.split("matrix_capacity_mobility_coarse_candidate", 1)[1]
        self.assertIn("4.0 * h * (1.0 - h)", helper)
        self.assertIn("1.0 + a_M * b", helper)
        self.assertIn("a_M < 0.0", helper)
        face = KERNELS.split("ctot_fv_positive_face_flux_kernel", 1)[1].split(
            "ctot_positive_face_flux_from_cell_gradient_kernel", 1
        )[0]
        self.assertIn("matrix_capacity_mobility_coarse_candidate", face)
        self.assertIn("2.0 * Mi * Mj / (Mi + Mj)", face)

    def test_checkpoint_round_trip_carries_coarse_provenance(self):
        writer = MAIN.split('\\"schema\\": \\"ctot_checkpoint_v1\\"', 1)[1].split(
            "fclose(meta_fp);", 1
        )[0]
        validation = MAIN.split("const int current_coarse", 1)[1].split(
            "return ok;", 1
        )[0]
        for field in (
            "coarse_model_name",
            "coarse_model_version",
            "lambda_over_dx",
            "gamma",
            "fine_reference_hash",
            "coarse_calibration_hash",
            "coarse_uncertainty_version",
            "coarse_interface_mobility_mode",
            "coarse_interface_mobility_a_M",
        ):
            self.assertIn(field, writer)
            self.assertIn(field, validation)
        self.assertIn("coarse checkpoint provenance mismatch", validation)

    def test_fp32_mechanics_contract_is_coarse4_only_and_checkpointed(self):
        coarse_validation = MAIN.split("if (coarse4_model)", 1)[1].split(
            "if (!P->thermo_convex_extrapolation_enabled)", 1
        )[0]
        gate = MAIN.split("const int normalized_mechanics_gate", 1)[1].split(
            "const int ledger_pass", 1
        )[0]
        writer = MAIN.split('\\"schema\\": \\"ctot_checkpoint_v1\\"', 1)[1].split(
            "fclose(meta_fp);", 1
        )[0]
        restart = MAIN.split("coarse checkpoint mechanics provenance", 1)[1].split(
            "return ok;", 1
        )[0]
        for token in (
            "FP32_SPECTRAL",
            "FP32_NORMALIZED_BACKWARD_ERROR_V1",
            "COARSE4_FP32_ETA_FLOOR_16_32_V1",
            "DEALIASED_REAL_DIVSIGMA_OVER_KMAX_STRESS_V1",
            "cc4cad8955684d45d34ca9db7d1b300dd82acc1fc7473ff52b23cd0f5cd3ae3e",
        ):
            self.assertIn(token, MAIN)
        self.assertIn("coarse4_mechanics_contract_matches", coarse_validation)
        self.assertIn("ctot_mechanical_backward_eta_linf", gate)
        self.assertIn("ctot_mechanical_equilibrium_linf", gate)
        for field in (
            "mechanics_precision_mode",
            "mechanics_acceptance_mode",
            "eta_floor_version",
            "eta_accept",
            "double_oracle_contract_hash",
            "residual_normalization_version",
        ):
            self.assertIn(field, writer)
            self.assertIn(field, restart)

    def test_first_active_set_line_search_is_continuous(self):
        trial = MAIN.split(
            "__global__ void ctot_phase_projected_line_trial_kernel", 1
        )[1].split("__global__ void ctot_local_storage_preconditioner_kernel", 1)[0]
        solver = MAIN.split("auto solve_ctot_phase_semismooth_pdas", 1)[1].split(
            "const int coupled_outer_required", 1
        )[0]
        self.assertIn("damp_active_constraints", trial)
        self.assertIn("phase_kkt_active_line_trial", trial)
        phase_kkt = (ROOT / "phase_kkt_utils.h").read_text()
        self.assertIn(
            "phi + lambda * (active_target - phi)", phase_kkt
        )
        self.assertIn("!previous_active_valid", solver)
        self.assertIn("const int damp_active_constraints = 1;", solver)
        self.assertIn("lambda=1", solver)
        self.assertNotIn(
            "phi_elastic_coupling_enabled &&\n                                    !previous_active_valid",
            solver,
        )
        self.assertIn("use_diagonal_globalization", solver)

    def test_mechanics_timing_is_opt_in_and_elastic_scoped(self):
        mechanics = MAIN.split(
            "auto recompute_elasticity_trial_state", 1
        )[1].split("auto compute_mass_from_Y", 1)[0]
        self.assertIn('if (P.elastic_enabled)', mechanics)
        self.assertIn(
            'CTOT_PERF_SCOPE(&ctot_perf, "mechanics.solve_and_contract"',
            mechanics,
        )

    def test_gp_and_s3_are_not_activated_by_coarse_core_mode(self):
        defaults = MAIN.split("static void params_default", 1)[1].split(
            "static void compute_nucleus_dimensions_gpu", 1
        )[0]
        self.assertIn('"OFF"', defaults)
        coarse_validation = MAIN.split("if (coarse4_model)", 1)[1].split(
            "if (!P->thermo_convex_extrapolation_enabled)", 1
        )[0]
        self.assertNotIn("diagnostic_rsmd_enabled = 1", coarse_validation)
        self.assertNotIn("gp_growth_enabled = 1", coarse_validation)


if __name__ == "__main__":
    unittest.main()
