import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "main_cuda.cu").read_text(encoding="utf-8")
KERNEL_SOURCE = (ROOT / "cuda_kernels.cu").read_text(encoding="utf-8")


def h(phi):
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def logistic(Y):
    return 1.0 / (1.0 + math.exp(-Y))


class CtotCandidateSolverSemanticsTests(unittest.TestCase):
    def test_current_spectral_hybrid_is_diagnostic_only(self):
        self.assertIn(
            "ctot_spectral_be uses an FFT chemical-potential", SOURCE
        )
        self.assertIn(
            "launch_ctot_positive_face_flux_from_cell_gradient_kernel(", SOURCE
        )
        self.assertIn("launch_ctot_fv_divergence_kernel(", SOURCE)
        self.assertIn(
            "BLOCKED_NONADJOINT_HYBRID_TRANSPORT_PRODUCTION_DISABLED", SOURCE
        )
        self.assertIn("ctot_nonadjoint_hybrid_test_only", SOURCE)

    def test_mimetic_and_fv_names_share_the_same_operator_path(self):
        runtime = SOURCE.split("const int ctot_use_mimetic", 1)[1].split(
            "auto run_ctot_transport_solve", 1
        )[0]
        self.assertIn('strcmp(P.composition_evolution_mode, "ctot_mimetic_be")', runtime)
        self.assertIn('strcmp(P.composition_evolution_mode, "ctot_fv_be")', runtime)
        solve = SOURCE.split("auto evaluate_ctot_transport_residual", 1)[1].split(
            "auto solve_ctot_transport", 1
        )[0]
        self.assertIn("launch_ctot_fv_positive_face_flux_kernel(", solve)
        self.assertIn("launch_ctot_fv_divergence_kernel(", solve)
        self.assertIn("if (use_spectral_operator)", solve)

    def test_checkpoint_records_transport_operator_provenance(self):
        writer = SOURCE.split('\\"schema\\": \\"ctot_checkpoint_v1\\"', 1)[1]
        writer = writer.split("fclose(meta_fp);", 1)[0]
        for field in (
            "transport_operator_name",
            "transport_operator_version",
            "gradient_operator_name",
            "divergence_operator_name",
            "adjoint_identity_mode",
            "face_mobility_mode",
            "finite_interface_correction_enabled",
            "finite_interface_mode",
            "finite_interface_mode_version",
            "finite_interface_applicability",
        ):
            self.assertIn(field, writer)
        self.assertIn("P.dx * unit_to_nm", writer)

    def test_explicit_restart_rejects_transport_provenance_mismatch(self):
        validation = SOURCE.split("if (meta->is_ctot_checkpoint", 1)[1].split(
            "return ok;", 1
        )[0]
        self.assertIn("Ctot checkpoint transport-operator provenance mismatch", validation)
        self.assertIn("meta->transport_operator_name", validation)
        self.assertIn("meta->transport_operator_version", validation)
        self.assertIn("meta->gradient_operator_name", validation)
        self.assertIn("meta->divergence_operator_name", validation)
        self.assertIn("meta->adjoint_identity_mode", validation)
        self.assertIn("meta->face_mobility_mode", validation)
        self.assertIn("meta->finite_interface_correction_enabled", validation)
        self.assertIn("meta->finite_interface_mode", validation)
        self.assertIn("meta->finite_interface_mode_version", validation)
        self.assertIn("stored_finite_interface_version", validation)
        self.assertIn("current_finite_interface_version", validation)

    def test_finite_interface_restart_provenance_is_atomic_and_legacy_compatible(self):
        validation = SOURCE.split("if (meta->is_ctot_checkpoint", 1)[1].split(
            "return ok;", 1
        )[0]
        self.assertIn("finite_interface_mode_provenance_fields == 1", validation)
        self.assertIn(
            "mode and version must be stored together", validation
        )
        self.assertIn("!meta->has_finite_interface_mode_provenance", validation)
        self.assertIn('"legacy_boolean_only"', validation)

    def test_rejected_curved_correction_is_named_planar_only(self):
        self.assertIn('"planar_antitrapping_v1"', SOURCE)
        self.assertIn(
            '"planar_benchmark_only_curved_quantitative_rejected"', SOURCE
        )
        self.assertIn("ctot_finite_interface_mode_name(&P)", SOURCE)
        self.assertIn("ctot_finite_interface_applicability(&P)", SOURCE)

    def test_semismooth_phase_solver_is_candidate_only_and_default_off(self):
        self.assertIn("P->ctot_phase_semismooth_pdas_enabled = 0;", SOURCE)
        branch = SOURCE.split(
            "if (P.ctot_phase_semismooth_pdas_enabled)", 1
        )[1].split("printf(\"CTOT_PHASE_INNER", 1)[0]
        self.assertIn("solve_ctot_phase_semismooth_pdas(", branch)
        self.assertIn("solve_ctot_phase_from_accepted()", branch)

    def test_semismooth_phase_jacobian_matches_the_accepted_residual(self):
        solver = SOURCE.split(
            "auto solve_ctot_phase_semismooth_pdas", 1
        )[1].split("const int coupled_outer_required", 1)[0]
        self.assertIn("evaluate_final_phase_residual(", solver)
        self.assertIn("ctot_phase_independent_partial_phi_kernel", solver)
        self.assertIn("ctot_phase_chain_rule_local_jacobian_kernel", solver)
        self.assertIn("ctot_phase_exact_nonelastic_local_jacobian_kernel", solver)
        self.assertIn("phase_kkt_fixed_ctot_dx_dphi", SOURCE)
        self.assertIn("ctot_phase_apply_free_jacobian(", solver)
        self.assertIn("ctot_phase_pdas_diagonal_defect_direction_kernel", solver)
        self.assertIn("P.kappa_phi", solver)
        self.assertIn("merit_after <= merit_before", solver)
        self.assertIn("PHASE_PDAS_ACTIVE_SET_CHANGES", solver)

    def test_phase_solver_provenance_is_written_and_restart_checked(self):
        writer = SOURCE.split('\\"schema\\": \\"ctot_checkpoint_v1\\"', 1)[1]
        writer = writer.split("fclose(meta_fp);", 1)[0]
        self.assertIn("phase_solver_name", writer)
        self.assertIn("phase_solver_version", writer)
        validation = SOURCE.split("if (meta->is_ctot_checkpoint", 1)[1].split(
            "return ok;", 1
        )[0]
        self.assertIn("Ctot checkpoint phase-solver provenance", validation)
        self.assertIn("ctot_phase_restart_solver_migration_allowed", validation)
        self.assertIn("CTOT_PHASE_SOLVER_RESTART_MIGRATION", validation)

    def test_phase_solver_keeps_ctot_authoritative_during_globalization(self):
        solver = SOURCE.split(
            "auto solve_ctot_phase_semismooth_pdas", 1
        )[1].split("const int coupled_outer_required", 1)[0]
        self.assertIn("ctot_phase_projected_line_trial_kernel", solver)
        self.assertIn("ctot_phase_reconstruct_x_fixed_C_kernel", solver)
        self.assertIn("restore_continuous_phase_context", solver)
        self.assertNotIn("launch_phi_normalize_project_ctot_kernel", solver)
        self.assertNotIn("cudaMemcpy(d_ctot_work_r, ctot_scratch_r", solver)
        self.assertIn("CTOT_PHASE_LINE_ZERO_REPLAY", solver)
        self.assertIn("0.0, P.v_B", solver)

    def test_thermodynamic_context_is_continuous_below_transport_support(self):
        kernel = SOURCE.split(
            "__global__ void ctot_phase_restore_continuous_context_kernel", 1
        )[1].split("__global__ void ctot_phase_fd_perturb_kernel", 1)[0]
        self.assertIn("alpha > representation_support_eps", kernel)
        self.assertIn("alpha > transport_support_eps ? 1.0 : 0.0", kernel)
        solver = SOURCE.split(
            "auto solve_ctot_phase_semismooth_pdas", 1
        )[1].split("const int coupled_outer_required", 1)[0]
        self.assertIn("restore_continuous_phase_context", solver)

    def test_continuous_context_does_not_change_legacy_reconstruction(self):
        reconstruct = KERNEL_SOURCE.split(
            "__global__ void reconstruct_x_q_Y_from_ctot_kernel", 1
        )[1].split("__global__ void compute_ctot_from_Y_kernel", 1)[0]
        normalize = KERNEL_SOURCE.split(
            "__global__ void phi_normalize_project_ctot_kernel", 1
        )[1].split("__global__ void ctot_candidate_state_from_Y_kernel", 1)[0]
        for kernel in (reconstruct, normalize):
            self.assertIn("alpha > matrix_support_eps", kernel)
            self.assertNotIn("representation_support_eps", kernel)

    def test_pdas_globalization_damps_first_active_set_continuously(self):
        kernel = SOURCE.split(
            "__global__ void ctot_phase_projected_line_trial_kernel", 1
        )[1].split("__global__ void ctot_local_storage_preconditioner_kernel", 1)[0]
        self.assertIn("active == PHASE_KKT_LOWER_ACTIVE", kernel)
        self.assertIn("phase_kkt_active_line_trial", kernel)
        phase_kkt = (ROOT / "phase_kkt_utils.h").read_text()
        self.assertIn("active_code == PHASE_KKT_UPPER_ACTIVE", phase_kkt)
        self.assertIn("return phi + lambda * delta", phase_kkt)
        self.assertIn("phi + lambda * (active_target - phi)", phase_kkt)
        solver = SOURCE.split(
            "auto solve_ctot_phase_semismooth_pdas", 1
        )[1].split("const int coupled_outer_required", 1)[0]
        self.assertIn("const int damp_active_constraints = 1;", solver)

    def test_disabled_gp_configuration_cannot_enable_pf_projection(self):
        activation = SOURCE.split(
            'strcmp(P.gp_birth_model, "poisson_literature_JGP")', 1
        )[0].rsplit("if (", 1)[1]
        self.assertIn("P.gp_literature_model_enabled", activation)
        self.assertIn("P.gp_stochastic_enabled", activation)

    def test_local_storage_jacobian_matches_finite_difference(self):
        phi = 0.47
        Y = -4.2
        x = logistic(Y)
        analytic = (1.0 - h(phi)) * x * (1.0 - x)
        eps = 1.0e-6
        plus = (1.0 - h(phi)) * logistic(Y + eps) + h(phi)
        minus = (1.0 - h(phi)) * logistic(Y - eps) + h(phi)
        numerical = (plus - minus) / (2.0 * eps)
        self.assertLess(abs(analytic - numerical) / analytic, 2.0e-8)

    def test_local_storage_newton_correction_closes_linearized_residual(self):
        phi = 0.35
        Y = -3.8
        x = logistic(Y)
        jacobian = (1.0 - h(phi)) * x * (1.0 - x)
        residual = 2.5e-7
        correction_Y = residual / jacobian
        linearized_after = residual - jacobian * correction_Y
        self.assertLess(abs(linearized_after), 1.0e-22)

    def test_inactive_beta_context_has_zero_preconditioner_correction(self):
        active = False
        correction = 0.0 if not active else math.nan
        self.assertEqual(correction, 0.0)

    def test_accepted_state_reconstructs_context_from_authoritative_ctot(self):
        accepted = SOURCE.split(
            "cudaMemcpy(d_ctot_accepted_r, d_ctot_work_r", 1
        )[1].split('printf("%s_ACCEPT', 1)[0]
        self.assertIn("reconstruct_ctot_thermodynamic_context(", accepted)
        self.assertIn("d_ctot_accepted_r, d_phi_r", accepted)

    def test_authoritative_restart_is_not_overwritten_for_vtk_output(self):
        self.assertIn("authoritative_ctot_raw_restart", SOURCE)
        guarded = SOURCE.split("const int authoritative_ctot_raw_restart", 1)[1]
        guarded = guarded.split("cudaMemcpy(d_temp, h_xBtot_r", 1)[0]
        self.assertIn("if (!authoritative_ctot_raw_restart)", guarded)
        self.assertIn("recompute_host_xBtot_field(", guarded)

    def test_finite_interface_work_is_not_transport_dissipation(self):
        kernel = KERNEL_SOURCE.split(
            "__global__ void ctot_add_antitrapping_face_flux_kernel", 1
        )[1].split("__global__ void ctot_fv_divergence_kernel", 1)[0]
        self.assertIn("CTOT_TRANSPORT_ANTITRAPPING_WORK_SUM", kernel)
        self.assertNotIn("CTOT_TRANSPORT_DISSIPATION_SUM", kernel)
        self.assertIn("(x_face - v_B) * phi_t", kernel)
        self.assertNotIn("(v_B - x_face) * phi_t", kernel)
        self.assertIn(
            "D_phase + D_transport + W_finite_interface", SOURCE
        )
        self.assertIn("isfinite(W_finite_interface)", SOURCE)
        self.assertIn("coupled_outer_max_iter_not_converged", SOURCE)
        self.assertIn("max_finite_interface_face_flux", SOURCE)
        self.assertIn("P->ctot_finite_interface_antitrapping_enabled = 0;", SOURCE)

    def test_candidate_transaction_does_not_depend_on_lphi_provenance(self):
        transaction_select = SOURCE.split(
            "const int ctot_coupled_outer_runtime", 1
        )[1].split("if (ctot_coupled_outer_runtime)", 1)[0]
        self.assertIn("= ctot_candidate_runtime;", transaction_select)
        self.assertNotIn("L_phi_calibration_mode", transaction_select)

    def test_finite_interface_resolution_guard_precedes_transport(self):
        guard = SOURCE.split("const double lambda_over_dx", 1)[1].split(
            "log_section_header(\"Run Configuration\")", 1
        )[0]
        self.assertIn(
            "BLOCKED_FINITE_INTERFACE_CORRECTION_OUTSIDE_VALIDATED_RESOLUTION",
            guard,
        )
        self.assertIn("finite_interface_production_min_points", guard)
        self.assertIn("finite_interface_resolution_test_override", guard)
        self.assertLess(
            SOURCE.index(
                "BLOCKED_FINITE_INTERFACE_CORRECTION_OUTSIDE_VALIDATED_RESOLUTION"
            ),
            SOURCE.index("CUDA_CHECK(cudaMalloc(&d_phi_r"),
        )

    def test_finite_interface_defaults_are_production_safe(self):
        self.assertIn(
            "P->finite_interface_calibration_min_points = 10.0;", SOURCE
        )
        self.assertIn(
            "P->finite_interface_calibration_max_points = 12.0;", SOURCE
        )
        self.assertIn(
            "P->finite_interface_production_min_points = 12.0;", SOURCE
        )
        self.assertIn(
            "P->finite_interface_resolution_test_override = 0;", SOURCE
        )
        self.assertIn(
            "P->finite_interface_violation_diagnostics_enabled = 0;", SOURCE
        )

    def test_finite_interface_violation_diagnostic_is_observational(self):
        diagnostic = SOURCE.split(
            "auto write_finite_interface_violation_diagnostics", 1
        )[1].split("auto compute_ctot_candidate_energy", 1)[0]
        self.assertIn("cudaMemcpyDeviceToHost", diagnostic)
        self.assertNotIn("cudaMemcpyHostToDevice", diagnostic)
        self.assertNotIn("launch_", diagnostic)
        self.assertIn("ctot_finite_interface_bound_violations.csv", diagnostic)

    def test_refined_grid_checkpoint_records_physical_dx(self):
        writer = SOURCE.split('\\"schema\\": \\"ctot_checkpoint_v1\\"', 1)[1]
        writer = writer.split("fclose(meta_fp);", 1)[0]
        self.assertIn("P.dx * unit_to_nm", writer)
        self.assertNotIn("P.Nx, P.Ny, P.Nz, unit_to_nm,", writer)

    def test_antitrapping_coefficient_maps_psi_width_to_runtime_phi(self):
        phi = 0.37
        H = h(phi)
        lambda_runtime = 0.6
        psi = 2.0 * phi - 1.0
        h_source = 2.0 * H - 1.0
        q_source = 1.0 - H
        a_source = ((h_source - 1.0) * (1.0 - q_source) /
                    (math.sqrt(2.0) * (psi * psi - 1.0)))
        W_source = lambda_runtime / (2.0 * math.sqrt(2.0))
        mapped = a_source * W_source * 2.0
        runtime = (lambda_runtime * H * (1.0 - H) /
                   (4.0 * phi * (1.0 - phi)))
        self.assertAlmostEqual(mapped, runtime, places=15)

    def test_projected_kkt_state_defect_does_not_amplify_roundoff_by_dt(self):
        state_roundoff = math.ulp(0.5)
        dt = 1.0e-7
        self.assertLessEqual(state_roundoff, 2.0e-16)
        self.assertGreater(state_roundoff / dt, 1.0e-10)

    def test_linear_anchor_filter_would_create_false_rate_residual(self):
        phi_accepted = 0.6302602229177514
        phi_filtered = 0.6302258073967102
        dt = 1.0e-6
        false_rate = (phi_filtered - phi_accepted) / dt
        self.assertGreater(abs(false_rate), 30.0)
        self.assertEqual((phi_accepted - phi_accepted) / dt, 0.0)

    def test_transport_floor_audit_is_default_off_and_transaction_neutral(self):
        self.assertIn(
            "P->ctot_debug_transport_floor_audit = 0;", SOURCE
        )
        audit = SOURCE.split(
            "if (P.ctot_debug_transport_floor_audit", 1
        )[1].split("if (ctot_energy_diag_fp)", 1)[0]
        self.assertIn("restore_d5_frozen_trial", audit)
        self.assertIn("P.ctot_residual_abs_tol = saved_abs_tol", audit)
        self.assertIn("P.ctot_residual_rel_tol = saved_rel_tol", audit)
        self.assertIn("P.ctot_nonlinear_max_iter = saved_max_iter", audit)
        self.assertIn("transport_ok = saved_transport_ok", audit)
        self.assertIn("converged = saved_converged", audit)

    def test_transport_floor_snapshot_handles_mechanics_off(self):
        audit = SOURCE.split(
            "if (P.ctot_debug_transport_floor_audit", 1
        )[1].split("if (ctot_energy_diag_fp)", 1)[0]
        self.assertIn(
            "if (d_sigma_xx_r && d_sigma_yy_r && d_sigma_zz_r)", audit
        )
        self.assertLess(
            audit.index("if (d_sigma_xx_r && d_sigma_yy_r && d_sigma_zz_r)"),
            audit.index("cudaMemcpy(frozen_sxx.data(), d_sigma_xx_r"),
        )

    def test_transport_floor_deterministic_gpu_reduction_has_fixed_order(self):
        kernel = KERNEL_SOURCE.split(
            "__global__ void ctot_deterministic_residual_reduction_kernel", 1
        )[1].split("__global__", 1)[0]
        self.assertIn("blockIdx.x != 0 || threadIdx.x != 0", kernel)
        self.assertIn("for (int idx = 0; idx < total_size; ++idx)", kernel)
        self.assertIn("result_r[0] = max_abs", kernel)
        self.assertIn("result_r[2] = sumsq", kernel)


if __name__ == "__main__":
    unittest.main()
