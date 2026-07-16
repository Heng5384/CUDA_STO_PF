import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main_cuda.cu").read_text(encoding="utf-8")
KERNELS = (ROOT / "cuda_kernels.cu").read_text(encoding="utf-8")
PARAMS = (ROOT / "pf_params.h").read_text(encoding="utf-8")
BOUNDS = (ROOT / "ctot_transport_bound_utils.h").read_text(encoding="utf-8")
PREP = (ROOT / "scripts/prepare_jc4_research_model.py").read_text(
    encoding="utf-8"
)


class CtotFeasibleStorageRuntimeTests(unittest.TestCase):
    def test_selector_is_declared_and_legacy_default_is_preserved(self):
        self.assertIn("ctot_transport_nonlinear_coordinate", PARAMS)
        self.assertIn("ctot_outer_acceleration", PARAMS)
        defaults = MAIN.split("static void params_default", 1)[1].split(
            "static void compute_nucleus_dimensions_gpu", 1
        )[0]
        self.assertIn('"legacy_logit_newton"', defaults)
        self.assertIn('sizeof(P->ctot_outer_acceleration), "OFF"', defaults)

    def test_jc4_explicitly_selects_feasible_coordinate(self):
        self.assertIn(
            '"ctot_transport_nonlinear_coordinate": '
            '"adaptive_logit_feasible_ctot_v1"',
            PREP,
        )
        validation = MAIN.split("const int jc4_model", 1)[1].split(
            "if (coarse4_model)", 1
        )[0]
        self.assertIn("adaptive_logit_feasible_ctot_v1", validation)

    def test_feasible_branch_updates_authoritative_ctot_without_y_jacobian(self):
        solve = MAIN.split("auto solve_ctot_transport_operator", 1)[1].split(
            "auto solve_ctot_transport_from_accepted", 1
        )[0]
        branch = solve.split("if (transport_feasible_coordinate_active)", 2)[2]
        branch = branch.split("} else {", 1)[0]
        self.assertIn("launch_ctot_trial_feasible_C_update_kernel", branch)
        self.assertIn("d_ctot_line_base_residual_r", branch)
        self.assertIn("cudaMemcpy(d_ctot_work_r, d_ctot_trial_r", branch)
        self.assertNotIn("ctot_local_storage_preconditioner_kernel", branch)
        self.assertNotIn("launch_ctot_add_active_Y_shift_kernel", branch)

    def test_line_search_freezes_preconditioned_base_direction_before_trials(self):
        solve = MAIN.split("auto solve_ctot_transport_operator", 1)[1].split(
            "auto solve_ctot_transport_from_accepted", 1
        )[0]
        freeze = solve.index("cufftExecZ2D(plan_c2r_xB,")
        trial = solve.index("launch_ctot_trial_feasible_C_update_kernel")
        self.assertLess(freeze, trial)
        precondition = solve.index("launch_apply_ctot_preconditioner_k_kernel")
        zero_mode = solve.index("cudaMemset(ctot_scratch_k, 0")
        self.assertLess(precondition, zero_mode)
        self.assertLess(zero_mode, freeze)
        self.assertIn("launch_ctot_build_mass_tangent_direction_kernel", solve)
        self.assertIn("launch_ctot_subtract_free_direction_mean_kernel", solve)
        self.assertIn("direction_sum / free_count_before_mean", solve)
        self.assertGreaterEqual(
            solve.count("launch_ctot_build_mass_tangent_direction_kernel"), 2
        )
        mean_shift = solve.index("launch_ctot_subtract_free_direction_mean_kernel")
        post_mean_revalidation = solve.index(
            "launch_ctot_build_mass_tangent_direction_kernel", mean_shift
        )
        stable_check = solve.index("free_count_after_mean", post_mean_revalidation)
        self.assertLess(mean_shift, post_mean_revalidation)
        self.assertLess(post_mean_revalidation, stable_check)
        trial_call = solve[trial:solve.index(");", trial) + 2]
        self.assertIn("d_ctot_line_base_residual_r", trial_call)
        self.assertNotIn("d_ctot_residual_r", trial_call)

    def test_spectral_preconditioner_matches_runtime_dealias_support(self):
        kernel = KERNELS.split(
            "__global__ void apply_ctot_preconditioner_k_kernel", 1
        )[1].split("__global__ void ctot_trial_Y_update_kernel", 1)[0]
        self.assertIn("ctot_r2c_mode_retained_by_23", kernel)
        self.assertIn(
            "retained ? dt * D_ref * k2[idx] : 0.0",
            kernel,
        )
        self.assertNotIn(
            "const double denom = a_ref + dt * D_ref * k2[idx]",
            kernel,
        )

    def test_bound_aware_line_search_uses_l2_merit_but_linf_gate_remains(self):
        solve = MAIN.split("auto solve_ctot_transport_operator", 1)[1].split(
            "auto solve_ctot_transport_from_accepted", 1
        )[0]
        self.assertIn(
            "trial_res_l2 <\n                                    res_l2_rel",
            solve,
        )
        self.assertIn("res_inf <= residual_target", solve)
        self.assertIn("fabs(mass_error) <= 1.0e-10", solve)

    def test_storage_active_set_is_not_the_residual_tolerance_band(self):
        solve = MAIN.split("auto solve_ctot_transport_operator", 1)[1].split(
            "auto solve_ctot_transport_from_accepted", 1
        )[0]
        self.assertIn("const double ctot_storage_active_tol = 0.0", MAIN)
        self.assertEqual(solve.count("ctot_storage_active_tol"), 3)
        self.assertIn("ctot_storage_context_q_ulp64", BOUNDS)
        self.assertIn("active_tol + min_lambda * direction", BOUNDS)

    def test_adaptive_outer_backtrack_restores_state_without_changing_dt(self):
        outer = MAIN.split(
            "int transport_block_converged =", 1
        )[1].split("const double transport_solve_residual", 1)[0]
        self.assertIn("ctot_adaptive_feasible_storage_enabled && outer_iter > 0", outer)
        self.assertIn(
            "cudaMemcpy(\n                            d_ctot_work_r, "
            "d_ctot_outer_C_prev_r",
            outer,
        )
        self.assertIn("launch_ctot_outer_convex_blend_kernel", outer)
        self.assertIn("d_ctot_outer_phi_prev2_r", outer)
        self.assertIn("d_ctot_outer_phi_prev_r", outer)
        self.assertIn("reconstruct_ctot_thermodynamic_context", outer)
        self.assertIn(
            "omega = ctot_next_line_search_lambda(\n"
            "                             omega, P.ctot_line_search_min)",
            outer,
        )
        self.assertNotIn("P.dt =", outer)
        self.assertNotIn("cudaMemcpy(d_ctot_saved_r", outer)
        self.assertNotIn("cudaMemcpy(d_phi_n_saved", outer)

    def test_local_outer_filter_uses_be_target_then_resolves_original_blocks(self):
        outer = MAIN.split(
            "int transport_block_converged =", 1
        )[1].split("const double transport_solve_residual", 1)[0]
        local_filter = outer.split(
            "launch_ctot_outer_local_phase_feasibility_filter_kernel", 1
        )[1].split("// The preceding full phase block", 1)[0]
        self.assertIn(
            "d_ctot_saved_r, d_ctot_work_r,\n"
            "                            d_divJ_r, d_phi_r",
            local_filter,
        )
        self.assertIn("dt_transport, P.v_B", local_filter)
        self.assertIn("local_filter_attempt <= P.ctot_outer_max_iter", outer)
        self.assertNotIn("d_ctot_outer_C_prev_r", local_filter)
        self.assertIn("last finite, mass-conservative nonlinear C", local_filter)
        self.assertIn("d_ctot_outer_phi_prev2_r,\n                            d_phi_r,", local_filter)
        self.assertIn("reconstruct_ctot_thermodynamic_context", local_filter)
        self.assertIn("solve_ctot_transport_from_accepted", local_filter)
        self.assertIn("local_filter_state_finite", local_filter)
        self.assertIn("local_filter_made_progress", local_filter)
        self.assertIn("ctot_local_phase_filter_made_progress", local_filter)
        self.assertIn("CTOT_OUTER_PHASE_LOCAL_FILTER_NO_PROGRESS", local_filter)
        self.assertNotIn("P.dt =", local_filter)
        self.assertNotIn("cudaMemcpy(d_ctot_saved_r", local_filter)

        kernel = KERNELS.split(
            "__global__ void ctot_outer_local_phase_feasibility_filter_kernel", 1
        )[1].split("__global__ void apply_ctot_preconditioner_k_kernel", 1)[0]
        self.assertIn("C_be_target", kernel)
        self.assertIn("phase_kkt_h_stable(candidate)", kernel)
        self.assertIn("phase_kkt_h_inverse_upper_feasible(h_target)", kernel)
        self.assertIn("filtered < previous ? 2 : 1", kernel)
        self.assertNotIn("if (filtered < previous) filtered = previous", kernel)
        self.assertIn("filtered_r[idx] = NAN", kernel)

    def test_authoritative_ctot_reconstruction_uses_stable_beta_endpoint_form(self):
        kernel = KERNELS.split(
            "__global__ void reconstruct_x_q_Y_from_ctot_bound_aware_kernel", 1
        )[1].split("__global__ void compute_ctot_from_Y_kernel", 1)[0]
        self.assertIn("phase_kkt_h_stable(phi_r[idx])", kernel)
        self.assertIn("phase_kkt_alpha(phi_r[idx])", kernel)
        self.assertIn("phase_kkt_q_from_ctot(phi_r[idx], C, v_B)", kernel)
        self.assertNotIn("const double alpha = 1.0 - h", kernel)
        self.assertNotIn("const double q = C - h * v_B", kernel)

    def test_adaptive_switch_occurs_only_after_legacy_failure(self):
        solve = MAIN.split("auto solve_ctot_transport_operator", 1)[1].split(
            "auto solve_ctot_transport_from_accepted", 1
        )[0]
        self.assertIn("legacy_initial_evaluation_failed", solve)
        self.assertIn("legacy_line_search_failed", solve)
        self.assertIn("transport_feasible_coordinate_active = 1", solve)
        self.assertIn("ctot_adaptive_feasible_storage_enabled", solve)
        self.assertIn(
            "d_ctot_line_base_residual_r,\n"
            "                                      d_ctot_work_r",
            solve,
        )
        switch = solve.split("reason=legacy_line_search_failed", 1)[0]
        switch = switch.rsplit(
            "transport_feasible_coordinate_active = 1", 1
        )[1]
        self.assertIn("d_Y_r, 1, use_spectral_operator", switch)
        self.assertIn("snapshot_feasible_iterate()", switch)
        self.assertNotIn(
            "d_ctot_work_r, d_ctot_line_base_residual_r", switch
        )
        self.assertIn("source=accepted_legacy_Y_iterate", solve)

    def test_every_failed_adaptive_solve_restores_last_finite_iterate(self):
        solve = MAIN.split("auto solve_ctot_transport_operator", 1)[1].split(
            "auto solve_ctot_transport_from_accepted", 1
        )[0]
        self.assertIn("int feasible_iterate_snapshot_valid = 0", solve)
        self.assertIn("auto snapshot_feasible_iterate", solve)
        self.assertIn("auto restore_feasible_iterate", solve)
        self.assertIn("d_ctot_work_r, ctot_correction_r", solve)
        self.assertIn("CTOT_TRANSPORT_FAILED_ITERATE_RESTORED", solve)
        final_restore = solve.split(
            "const int solve_accepted = transport_ok && converged", 1
        )[1]
        self.assertIn("!solve_accepted", final_restore)
        self.assertIn("restore_feasible_iterate()", final_restore)
        self.assertIn("transport_ok = 0", final_restore)
        self.assertIn("converged = 0", final_restore)
        self.assertNotIn("P.dt =", final_restore)

    def test_feasible_trial_is_rejected_not_clipped(self):
        kernel = KERNELS.split(
            "__global__ void ctot_trial_feasible_C_update_kernel", 1
        )[1].split("__global__ void ctot_active_capacity_kernel", 1)[0]
        self.assertIn("ctot_build_feasible_storage_trial", kernel)
        self.assertIn("CTOT_STORAGE_INVALID", kernel)
        self.assertIn("CTOT_TRANSPORT_CONTEXT_ULP_NORMALIZATION_COUNT", kernel)
        self.assertIn("CTOT_TRANSPORT_CONTEXT_ULP_MAX_DEFECT", kernel)
        self.assertIn("phase_kkt_h_stable(phi_r[idx])", kernel)
        self.assertIn("phase_kkt_alpha(phi_r[idx])", kernel)
        self.assertNotIn("fmax", kernel)
        self.assertNotIn("fmin", kernel)

    def test_bound_aware_reconstruction_keeps_c_and_q_exact(self):
        kernel = KERNELS.split(
            "__global__ void reconstruct_x_q_Y_from_ctot_bound_aware_kernel", 1
        )[1].split("__global__ void compute_ctot_from_Y_kernel", 1)[0]
        self.assertIn(
            "const double q = phase_kkt_q_from_ctot(phi_r[idx], C, v_B)",
            kernel,
        )
        self.assertIn("q_alpha_r[idx] = q", kernel)
        self.assertNotIn("ctot_r[idx] =", kernel)
        self.assertIn("ctot_storage_context_q_ulp64", kernel)
        self.assertIn("const double x = q_context / alpha", kernel)
        self.assertIn("CTOT_TRANSPORT_CONTEXT_ULP_NORMALIZATION_COUNT", kernel)

    def test_adaptive_ctot_tangent_and_capacity_use_stable_beta_endpoint(self):
        tangent = KERNELS.split(
            "__global__ void ctot_build_mass_tangent_direction_kernel", 1
        )[1].split("__global__ void ctot_subtract_free_direction_mean_kernel", 1)[0]
        capacity = KERNELS.split(
            "__global__ void ctot_active_capacity_kernel", 1
        )[1].split("__global__ void ctot_add_active_Y_shift_kernel", 1)[0]
        self.assertIn("phase_kkt_h_stable(phi_r[idx])", tangent)
        self.assertIn("phase_kkt_alpha(phi_r[idx])", capacity)
        self.assertIn("min_lambda", tangent)
        self.assertIn("ctot_storage_context_q_ulp64", BOUNDS)
        self.assertIn("active_tol + min_lambda * direction", BOUNDS)

    def test_mass_tangent_active_set_iterates_to_a_stable_monotone_mask(self):
        solve = MAIN.split("auto solve_ctot_transport_operator", 1)[1].split(
            "auto solve_ctot_transport_from_accepted", 1
        )[0]
        tangent = solve.split("int tangent_set_stable = 0", 1)[1].split(
            "while (transport_ok", 1
        )[0]
        self.assertIn("tangent_pass < 64", tangent)
        self.assertIn("free_count_before_mean", tangent)
        self.assertIn("free_count_after_mean", tangent)
        self.assertIn(
            "fabs(free_count_after_mean -\n"
            "                                 free_count_before_mean) < 0.5",
            tangent,
        )
        mean_shift = tangent.index("launch_ctot_subtract_free_direction_mean_kernel")
        revalidation = tangent.index(
            "launch_ctot_build_mass_tangent_direction_kernel", mean_shift
        )
        stable_check = tangent.index("free_count_after_mean", revalidation)
        self.assertLess(mean_shift, revalidation)
        self.assertLess(revalidation, stable_check)
        self.assertIn("tangent_set_stable = 1", tangent)
        self.assertIn("CTOT_TRANSPORT_TANGENT_ACTIVE_SET_UNRESOLVED", tangent)
        self.assertIn("transport_ok = 0", tangent)
        self.assertIn("P.ctot_line_search_min", tangent)

    def test_line_search_evaluates_exact_minimum_step_before_stopping(self):
        solve = MAIN.split("auto solve_ctot_transport_operator", 1)[1].split(
            "auto solve_ctot_transport_from_accepted", 1
        )[0]
        self.assertIn(
            "lambda = ctot_next_line_search_lambda(\n"
            "                        lambda, P.ctot_line_search_min)",
            solve,
        )
        self.assertIn("halved < minimum ? minimum : halved", BOUNDS)

    def test_step655_capacity_diagnostics_are_read_only_and_authoritative(self):
        self.assertIn("ctot_outer_capacity_diagnostics.csv", MAIN)
        block = MAIN.split("if (ctot_outer_capacity_diag_fp) {", 1)[1].split(
            "if (!isfinite(initial_phase_residual))", 1
        )[0]
        self.assertIn("d_ctot_work_r", block)
        self.assertIn("d_ctot_outer_C_prev_r", block)
        self.assertIn("phase_kkt_q_from_ctot", block)
        self.assertIn("alpha_now * dx", block)
        self.assertIn("x_prev * (alpha_now - alpha_prev)", block)
        self.assertIn("d_ctot_fv_face_x", block)
        self.assertIn("d_ctot_residual_r", block)
        self.assertNotIn("cudaMemcpyHostToDevice", block)
        self.assertNotIn("P.dt =", block)
        self.assertNotIn("fwrite", block)

    def test_raw_x_and_y_are_not_outer_acceptance_quantities(self):
        contract = MAIN.split("const int primary_state_delta_pass =", 1)[1].split(
            "elastic_outer_converged =", 1
        )[0]
        self.assertIn("outer_C_linf", contract)
        self.assertIn("outer_phi_linf", contract)
        self.assertIn("final_transport_res_inf", contract)
        self.assertIn("phase_kkt_linf", contract)
        self.assertNotIn("delta_x", contract)
        self.assertNotIn("delta_Y", contract)

    def test_coarse4_outer_contract_uses_q_and_shared_face_flux(self):
        selector = MAIN.split(
            "const int ctot_capacity_aware_outer_contract", 1
        )[1].split("CtotPerformanceProfiler", 1)[0]
        self.assertIn("is_coarse4_research_model(&P)", selector)
        contract = MAIN.split("const int capacity_delta_pass =", 1)[1].split(
            "const int normalized_mechanics_gate", 1
        )[0]
        self.assertIn("outer_q_linf <= delta_tol", contract)
        self.assertIn("outer_face_flux_linf <= delta_tol", contract)
        self.assertIn("isfinite(outer_q_linf)", contract)
        self.assertIn("isfinite(outer_face_flux_linf)", contract)
        self.assertNotIn("delta_x", contract)
        self.assertNotIn("delta_Y", contract)

        metric = MAIN.split(
            "__global__ void ctot_scaled_face_flux_increment_kernel", 1
        )[1].split("__global__ void ctot_phase_final_residual_kernel", 1)[0]
        self.assertIn("ctot_scaled_face_flux_increment", metric)
        self.assertIn("flux_new_r[idx]", metric)
        self.assertIn("flux_old_r[idx]", metric)
        self.assertIn("dt", metric)
        self.assertIn("spacing", metric)

        outer = MAIN.split(
            "if (ctot_capacity_aware_outer_contract) {", 2
        )[2].split("if (outer_mechanics_required)", 1)[0]
        self.assertIn("d_ctot_fv_face_x", outer)
        self.assertIn("d_ctot_fv_face_y", outer)
        self.assertIn("d_ctot_fv_face_z", outer)
        self.assertIn("d_ctot_outer_face_prev_x", outer)
        self.assertIn("ctot_normalized_q_increment_kernel", outer)
        self.assertIn("ctot_scaled_face_flux_increment_kernel", outer)
        self.assertNotIn("d_xB_r", outer)
        self.assertNotIn("d_Y_r", outer)

    def test_block_aitken_is_default_off_active_set_aware_and_fail_closed(self):
        self.assertIn('"ctot_outer_acceleration": "OFF"', PREP)
        self.assertIn('"BLOCK_AITKEN_V1"', MAIN)
        block = MAIN.split(
            "if (ctot_block_aitken_runtime) {\n"
            "                    const double omega_min", 1
        )[1].split(
            "outer_anchor_intact =", 1
        )[0]
        self.assertIn("ctot_outer_active_code_change_kernel", block)
        self.assertIn("active_set_changes_outer > 0", block)
        self.assertIn("outer_C_aitken_history_valid = 0", block)
        self.assertIn("outer_phi_aitken_history_valid = 0", block)
        self.assertIn("ctot_outer_aitken_C_apply_kernel", block)
        self.assertIn("ctot_outer_aitken_phase_apply_kernel", block)
        self.assertIn("d_ctot_outer_phase_active_current_r", block)
        self.assertIn("acceleration_merit_after <", block)
        self.assertIn("physical_blocks_not_worse", block)
        self.assertIn("physical_block_residual_worsened", block)
        self.assertIn("acceleration_energy_after - F_before", block)
        self.assertIn("d_ctot_outer_unaccelerated_C_r", block)
        self.assertIn("d_ctot_outer_unaccelerated_phi_r", block)
        self.assertIn("reconstruct_ctot_thermodynamic_context", block)
        self.assertNotIn("d_Y_n_saved", block)
        self.assertNotIn("P.dt =", block)

    def test_anderson_m2_uses_physical_residuals_and_is_fail_closed(self):
        self.assertIn('"ctot_outer_acceleration": "OFF"', PREP)
        self.assertIn('"ANDERSON_M2_V1"', MAIN)
        block = MAIN.split(
            "if (ctot_anderson_runtime) {", 1
        )[1].split("outer_anchor_intact =", 1)[0]
        self.assertIn(
            "ctot_outer_anderson_physical_residual_terms_kernel", block
        )
        self.assertIn("transport_gate", block)
        self.assertIn("phase_gate", block)
        self.assertIn("current_mechanics_residual", block)
        self.assertIn("ctot_outer_active_code_change_kernel", block)
        self.assertIn("active_set_changes_outer > 0", block)
        self.assertIn("ctot_outer_anderson_C_apply_kernel", block)
        self.assertIn("ctot_outer_anderson_phase_apply_kernel", block)
        self.assertIn("d_ctot_outer_phase_active_current_r", block)
        self.assertIn("acceleration_merit_after <", block)
        self.assertIn("physical_blocks_not_worse", block)
        self.assertIn("physical_block_residual_worsened", block)
        self.assertIn("acceleration_energy_after - F_before", block)
        self.assertIn("d_ctot_outer_unaccelerated_C_r", block)
        self.assertIn("d_ctot_outer_unaccelerated_phi_r", block)
        self.assertIn("reconstruct_ctot_thermodynamic_context", block)
        self.assertIn("outer_anderson_history_valid = 0", block)
        self.assertIn("d_ctot_residual_r", block)
        self.assertIn("ctot_correction_r", block)
        self.assertNotIn("d_Y_n_saved", block)
        self.assertNotIn("P.dt =", block)

    def test_anderson_m3_uses_three_physical_histories_and_exact_active_bounds(self):
        self.assertIn('"ANDERSON_M3_V1"', MAIN)
        block = MAIN.split(
            "if (ctot_anderson_runtime) {", 1
        )[1].split("outer_anchor_intact =", 1)[0]
        self.assertIn("outer_anderson_history2_valid", block)
        self.assertIn("ctot_outer_scaled_physical_inner_product_kernel", block)
        self.assertIn("ctot_anderson_m3_weights", block)
        self.assertIn("ctot_outer_anderson_m3_C_apply_kernel", block)
        self.assertIn("ctot_outer_anderson_m3_phase_apply_kernel", block)
        self.assertIn("d_ctot_outer_phase_active_current_r", block)
        self.assertIn("outer_anderson_history2_valid = 0", block)
        self.assertIn("physical_blocks_not_worse", block)
        self.assertIn("acceleration_energy_after - F_before", block)
        self.assertNotIn("P.dt =", block)

    def test_outer_kkt_is_replayed_after_transport_context_reconstruction(self):
        transport = MAIN.index(
            "const int final_transport_evaluable =\n"
            "                    evaluate_ctot_transport_residual("
        )
        replay = MAIN.index(
            "Transport residual evaluation reconstructs the authoritative",
            transport,
        )
        acceleration = MAIN.index("if (ctot_block_aitken_runtime) {", replay)
        self.assertLess(transport, replay)
        self.assertLess(replay, acceleration)
        audit = MAIN[replay:acceleration]
        self.assertIn("evaluate_final_phase_residual", audit)

    def test_final_phase_kkt_uses_fixed_ctot_context_not_transport_x_buffer(self):
        block = MAIN.split(
            "auto evaluate_final_phase_residual =", 1
        )[1].split("auto evaluate_ctot_phase_state =", 1)[0]
        self.assertIn("ctot_phase_reconstruct_x_fixed_C_kernel", block)
        self.assertIn("d_ctot_trial_r", block)
        driving = block.split("compute_ctot_phase_local_driving(", 1)[1]
        driving = driving.split(");", 1)[0]
        self.assertIn("d_ctot_trial_r", driving)
        self.assertNotIn("d_xB_r", driving)

    def test_restart_provenance_is_written_and_fail_closed(self):
        writer = MAIN.split('\\"schema\\": \\"ctot_checkpoint_v1\\"', 1)[1]
        writer = writer.split("fclose(meta_fp);", 1)[0]
        self.assertIn("transport_nonlinear_solver_name", writer)
        validation = MAIN.split("if (meta->is_ctot_checkpoint", 1)[1].split(
            "return ok;", 1
        )[0]
        self.assertIn("missing nonlinear-solver", validation)
        self.assertIn("provenance for current solver", validation)
        self.assertIn("nonlinear-solver provenance", validation)
        self.assertIn("mismatch: stored=%s current=%s", validation)


if __name__ == "__main__":
    unittest.main()
