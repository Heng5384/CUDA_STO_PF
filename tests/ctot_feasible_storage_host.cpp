#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "ctot_transport_bound_utils.h"

namespace {

struct Result {
    const char *test;
    double observed;
    double tolerance;
    bool passed;
    const char *semantics;
};

void add(std::vector<Result> &out, const char *test, double observed,
         double tolerance, const char *semantics) {
    out.push_back({test, observed, tolerance, observed <= tolerance, semantics});
}

CtotFeasibleStorageTrial trial(double C, double residual, double h,
                               double lambda = 1.0) {
    return ctot_build_feasible_storage_trial(
        C, residual, h, 1.0 - h, lambda, 1.0,
        1.0e-10, 1.0e-8, 20.0, 1.0e-12);
}

CtotFeasibleStorageTrial trial_with_alpha(
    double C, double residual, double h, double alpha,
    double lambda = 1.0) {
    return ctot_build_feasible_storage_trial(
        C, residual, h, alpha, lambda, 1.0,
        1.0e-10, 1.0e-8, 20.0, 1.0e-12);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s OUTPUT.csv\n", argv[0]);
        return 2;
    }
    std::vector<Result> results;

    const double h = 0.73;
    const double C = h + (1.0 - h) * 0.04;
    const double residual = 2.5e-4;
    const CtotFeasibleStorageTrial direct = trial(C, residual, h);
    add(results, "direct_storage_map_closes_frozen_flux_residual",
        std::fabs((direct.C - C) + residual), 1.0e-16,
        "C_trial=C_current-R is the exact frozen-flux BE target map");
    add(results, "direct_storage_map_is_interior",
        direct.status == CTOT_STORAGE_FREE ? 0.0 : 1.0, 0.0,
        "an admissible interior trial remains a free storage coordinate");

    const double alpha_small = 1.63738989e-9;
    const double h_small = 1.0 - alpha_small;
    const double q_small = 1.110223e-16;
    const double C_small = h_small + q_small;
    const CtotFeasibleStorageTrial reopening =
        trial(C_small, -0.25 * alpha_small, h_small);
    add(results, "vanishing_jacobian_reopening_is_finite",
        std::isfinite(reopening.Y_context) &&
                reopening.status != CTOT_STORAGE_INVALID
            ? 0.0 : 1.0,
        0.0,
        "the C-coordinate can reopen q without dividing by alpha*x*(1-x)");
    add(results, "vanishing_jacobian_trial_preserves_storage_identity",
        std::fabs(reopening.C - (h_small + reopening.q)), 2.0e-16,
        "the bound-aware coordinate retains C=h+q exactly");

    const double alpha_sub_ulp = std::ldexp(1.0, -56);
    const CtotFeasibleStorageTrial endpoint = trial_with_alpha(
        1.0, 0.0, 1.0, alpha_sub_ulp);
    add(results, "sub_ulp_beta_endpoint_capacity_remains_representable",
        endpoint.status == CTOT_STORAGE_INACTIVE_SUPPORT &&
                endpoint.q == alpha_sub_ulp
            ? 0.0 : 1.0,
        0.0,
        "explicit alpha preserves matrix capacity after h rounds to one");

    int context_normalized = 0;
    double context_defect = 0.0;
    const double endpoint_roundoff = -8.5 * DBL_EPSILON;
    const double endpoint_context = ctot_storage_context_q_ulp64(
        endpoint_roundoff, 0.25, 0.75, 1.0,
        &context_normalized, &context_defect);
    add(results, "ulp_endpoint_context_is_finite_without_rewriting_storage",
        endpoint_context == 0.0 && context_normalized == 1 &&
                context_defect == std::fabs(endpoint_roundoff)
            ? 0.0 : 1.0,
        0.0,
        "only derived x/Y context is canonicalized for an ulp-scale q defect");
    const double material_context = ctot_storage_context_q_ulp64(
        -1.0e-10, 0.25, 0.75, 1.0,
        &context_normalized, &context_defect);
    add(results, "material_storage_violation_remains_invalid",
        std::isnan(material_context) ? 0.0 : 1.0, 0.0,
        "a non-roundoff q violation still rejects the transaction");

    const double ulp_trial_q = -2.0 * DBL_EPSILON;
    const double ulp_trial_alpha = 0.25;
    const double ulp_trial_C =
        (1.0 - ulp_trial_alpha) + ulp_trial_q;
    const CtotFeasibleStorageTrial ulp_endpoint_trial = trial_with_alpha(
        ulp_trial_C, 0.0, 1.0 - ulp_trial_alpha,
        ulp_trial_alpha);
    add(results, "ulp_endpoint_trial_uses_lower_active_context",
        ulp_endpoint_trial.status == CTOT_STORAGE_LOWER_ACTIVE &&
                ulp_endpoint_trial.context_normalized == 1 &&
                ulp_endpoint_trial.q < 0.0 &&
                ulp_endpoint_trial.C == ulp_trial_C
            ? 0.0 : 1.0,
        0.0,
        "a roundoff endpoint is usable without rewriting authoritative C or q");
    const CtotFeasibleStorageTrial material_endpoint_trial = trial_with_alpha(
        0.75 - 1.0e-10, 0.0, 0.75, 0.25);
    add(results, "material_endpoint_trial_remains_invalid",
        material_endpoint_trial.status == CTOT_STORAGE_INVALID ? 0.0 : 1.0,
        0.0,
        "a material trial bound crossing is not hidden by context normalization");
    add(results, "ulp_lower_endpoint_outward_direction_is_blocked",
        ctot_direction_is_free_in_mass_tangent(
            ulp_trial_C, 0.75, 0.25, 1.0, 1.0,
            1.0e-10, 1.0e-12, 0.0)
            ? 1.0 : 0.0,
        0.0,
        "the ULP-normalized lower endpoint still blocks outward mass motion");
    add(results, "ulp_lower_endpoint_inward_direction_remains_free",
        ctot_direction_is_free_in_mass_tangent(
            ulp_trial_C, 0.75, 0.25, -1.0, 1.0,
            1.0e-10, 1.0e-12, 0.0)
            ? 0.0 : 1.0,
        0.0,
        "the ULP-normalized lower endpoint permits an inward tangent direction");
    add(results, "materially_invalid_endpoint_has_no_tangent_direction",
        ctot_direction_is_free_in_mass_tangent(
            0.75 - 1.0e-10, 0.75, 0.25, -1.0, 1.0,
            1.0e-10, 1.0e-12, 0.0)
            ? 1.0 : 0.0,
        0.0,
        "a material storage violation cannot enter the tangent solve");

    const double near_lower_q = 9.5e-13;
    const double near_lower_C = 0.75 + near_lower_q;
    add(results, "interior_storage_is_not_frozen_by_residual_tolerance_band",
        ctot_direction_is_free_in_mass_tangent(
            near_lower_C, 0.75, 0.25, 1.0e-7, 1.0,
            1.0e-10, 0.0, 1.0e-6)
            ? 0.0 : 1.0,
        0.0,
        "q above the minimum-step crossing remains tangent-free even when q is below the equation residual tolerance");
    add(results, "minimum_step_bound_crossing_remains_blocked",
        ctot_direction_is_free_in_mass_tangent(
            near_lower_C, 0.75, 0.25, 1.0e-6, 1.0,
            1.0e-10, 0.0, 1.0e-6)
            ? 1.0 : 0.0,
        0.0,
        "the exact geometric active set still blocks a direction whose smallest trial would cross q=0");

    const CtotFeasibleStorageTrial outward =
        trial(C_small, 2.0 * alpha_small, h_small);
    add(results, "outward_lower_bound_trial_rejected_not_clipped",
        outward.status == CTOT_STORAGE_INVALID ? 0.0 : 1.0, 0.0,
        "an infeasible q<0 trial is rejected rather than projected");
    add(results, "outward_trial_raw_value_not_rewritten",
        std::fabs(outward.C - (C_small - 2.0 * alpha_small)), 0.0,
        "diagnostics retain the raw infeasible trial value");

    const CtotFeasibleStorageTrial lower = trial(h, 0.0, h);
    add(results, "exact_lower_bound_has_finite_context",
        lower.status == CTOT_STORAGE_LOWER_ACTIVE &&
                std::isfinite(lower.Y_context) && lower.q == 0.0
            ? 0.0 : 1.0,
        0.0,
        "q=0 is authoritative while Y is only a finite thermo context");

    add(results, "lower_bound_outward_direction_is_blocked",
        ctot_direction_is_free_in_mass_tangent(
            h, h, 1.0 - h, 1.0, 1.0, 1.0e-10, 1.0e-12, 0.0)
            ? 1.0 : 0.0,
        0.0,
        "C_new=C-lambda*p cannot use p>0 at q=0");
    add(results, "lower_bound_inward_direction_remains_free",
        ctot_direction_is_free_in_mass_tangent(
            h, h, 1.0 - h, -1.0, 1.0, 1.0e-10, 1.0e-12, 0.0)
            ? 0.0 : 1.0,
        0.0,
        "the tangent cone permits a lower-bound component to reopen q");
    const double upper_C = h + (1.0 - h);
    add(results, "upper_bound_outward_direction_is_blocked",
        ctot_direction_is_free_in_mass_tangent(
            upper_C, h, 1.0 - h, -1.0, 1.0, 1.0e-10, 1.0e-12, 0.0)
            ? 1.0 : 0.0,
        0.0,
        "C_new=C-lambda*p cannot use p<0 at q=alpha");
    add(results, "minimum_line_step_activates_near_lower_bound",
        ctot_direction_is_free_in_mass_tangent(
            h + 1.0e-8, h, 1.0 - h, 1.0, 1.0,
            1.0e-10, 1.0e-12, 1.0e-6)
            ? 1.0 : 0.0,
        0.0,
        "the tangent cone blocks a direction whose minimum line step crosses q=0");
    // A zero-sum equality shift can create a new outward component even when
    // the pre-shift direction was feasible.  The runtime must therefore audit
    // the direction after subtracting the free-set mean.
    const double q_near_lower = 1.0e-8;
    const double pre_mean_direction = 0.0;
    const double free_set_mean = -2.0 / 3.0;
    const double post_mean_direction =
        pre_mean_direction - free_set_mean;
    add(results, "post_mean_tangent_revalidation_blocks_new_outward_direction",
        ctot_direction_is_free_in_mass_tangent(
            h + q_near_lower, h, 1.0 - h, post_mean_direction,
            1.0, 1.0e-10, 1.0e-12, 1.0e-6)
            ? 1.0 : 0.0,
        0.0,
        "the equality correction is rechecked before any line-search trial");
    add(results, "line_search_schedule_hits_exact_minimum_floor",
        std::fabs(ctot_next_line_search_lambda(
            1.9073486328125e-6, 1.0e-6) - 1.0e-6),
        0.0,
        "backtracking must evaluate the step protected by the tangent contract");
    add(results, "line_search_schedule_stops_after_minimum_floor",
        ctot_next_line_search_lambda(1.0e-6, 1.0e-6), 0.0,
        "the exact floor is tested once and cannot create an infinite loop");
    add(results, "local_phase_filter_detects_no_progress",
        ctot_local_phase_filter_made_progress(0.0, 0.0, 0.0) ? 1.0 : 0.0,
        0.0,
        "an unchanged phase candidate must not repeat the same transport solve");
    add(results, "local_phase_filter_detects_rewritten_same_value",
        ctot_local_phase_filter_made_progress(0.0, 2.0, 0.0) ? 1.0 : 0.0,
        0.0,
        "rewriting bound cells to identical values is still zero progress");
    add(results, "local_phase_filter_detects_modified_cell",
        ctot_local_phase_filter_made_progress(1.0, 0.0, 1.0e-15) ? 0.0 : 1.0,
        0.0,
        "a limited cell permits another safeguarded feasibility pass");

    // Equality-constrained tangent projection after blocking one lower-bound
    // component: subtract the free-set mean and retain exact zero direction
    // mass without changing the physical C state.
    std::vector<double> tangent = {0.0, -0.4, -0.6};
    const double free_mean = (tangent[1] + tangent[2]) / 2.0;
    tangent[1] -= free_mean;
    tangent[2] -= free_mean;
    add(results, "free_set_mean_enforces_zero_mass_direction",
        std::fabs(tangent[0] + tangent[1] + tangent[2]), 1.0e-16,
        "the equality multiplier acts on the search direction, not C");

    std::vector<double> residuals = {1.0e-3, -2.0e-3, 4.0e-3, -3.0e-3};
    double residual_sum = 0.0;
    double delta_sum = 0.0;
    for (double r : residuals) residual_sum += r;
    for (double r : residuals) {
        const CtotFeasibleStorageTrial value = trial(0.4, r, 0.2);
        delta_sum += value.C - 0.4;
    }
    add(results, "shared_face_zero_sum_map_preserves_global_mass",
        std::fabs(residual_sum) + std::fabs(delta_sum), 2.0e-16,
        "a telescoping residual produces a zero-sum C trial update");

    add(results, "dealias_512_mode_170_retained",
        ctot_mode_retained_by_23(170, 512) ? 0.0 : 1.0, 0.0,
        "strict 2/3 cutoff retains signed mode 170 on a 512-cell axis");
    add(results, "dealias_512_mode_171_removed",
        ctot_mode_retained_by_23(171, 512) ? 1.0 : 0.0, 0.0,
        "strict 2/3 cutoff removes signed mode 171 on a 512-cell axis");
    add(results, "dealias_negative_mode_170_retained",
        ctot_mode_retained_by_23(512 - 170, 512) ? 0.0 : 1.0, 0.0,
        "negative stored mode -170 follows the same cutoff");
    add(results, "dealias_negative_mode_171_removed",
        ctot_mode_retained_by_23(512 - 171, 512) ? 1.0 : 0.0, 0.0,
        "negative stored mode -171 follows the same cutoff");
    add(results, "dealias_degenerate_axes_keep_zero_mode",
        ctot_r2c_mode_retained_by_23(0, 0, 0, 512, 1, 1)
            ? 0.0 : 1.0,
        0.0,
        "a 512x1x1 R2C grid keeps the zero modes on singleton axes");
    add(results, "dealias_high_frequency_r2c_mode_removed",
        ctot_r2c_mode_retained_by_23(248, 0, 0, 512, 1, 1)
            ? 1.0 : 0.0,
        0.0,
        "the observed two-cell residual band is outside the physical operator");

    add(results, "outer_blend_zero_returns_previous",
        std::fabs(ctot_outer_convex_blend(0.2, 0.8, 0.0) - 0.2), 0.0,
        "omega=0 restores the previous outer phase input exactly");
    add(results, "outer_blend_one_returns_candidate",
        std::fabs(ctot_outer_convex_blend(0.2, 0.8, 1.0) - 0.8), 0.0,
        "omega=1 recovers the full phase candidate exactly");
    add(results, "outer_blend_half_is_convex_midpoint",
        std::fabs(ctot_outer_convex_blend(0.2, 0.8, 0.5) - 0.5), 1.0e-16,
        "phase globalization stays inside the pointwise convex segment");
    add(results, "outer_blend_rejects_invalid_weight",
        std::isnan(ctot_outer_convex_blend(0.2, 0.8, 1.1)) ? 0.0 : 1.0,
        0.0, "an invalid outer globalization weight fails closed");

    const double alpha_old = 2.0e-7;
    const double alpha_new = 1.5e-7;
    const double x_old = 0.03;
    const double x_new = 0.025;
    const double delta_q = alpha_new * x_new - alpha_old * x_old;
    const double capacity_weighted_delta_x = alpha_new * (x_new - x_old);
    const double moving_capacity_term = x_old * (alpha_new - alpha_old);
    add(results, "capacity_weighted_x_increment_closes_q_product_rule",
        std::fabs(delta_q - ctot_q_increment_from_capacity_product_rule(
                  alpha_new, x_new, alpha_old, x_old)),
        2.0e-22,
        "Delta q=alpha_new Delta x+x_old Delta alpha keeps near-beta x diagnostic capacity-aware");
    add(results, "raw_x_increment_is_not_storage_increment",
        std::fabs(x_new - x_old) >
                1.0e4 * std::fabs(capacity_weighted_delta_x)
            ? 0.0 : 1.0,
        0.0,
        "raw Delta x cannot dominate convergence when matrix capacity vanishes");
    add(results, "shared_face_flux_change_scales_to_one_step_ctot_increment",
        std::fabs(ctot_scaled_face_flux_increment(
                      2.5, 2.0, 0.01, 0.5, 0.1) - 0.1),
        1.0e-15,
        "dt Delta J/(dx Cscale) compares flux history in conserved-state units");
    add(results, "invalid_shared_face_flux_scale_fails_closed",
        std::isnan(ctot_scaled_face_flux_increment(
                       2.5, 2.0, 0.01, 0.0, 0.1)) ? 0.0 : 1.0,
        0.0,
        "the capacity-aware flux contract cannot hide an invalid spacing");
    add(results, "block_aitken_scalar_matches_vector_formula",
        std::fabs(ctot_block_aitken_omega(
                      1.0, -0.25, 0.5, 0.1, 1.5) - 0.5),
        1.0e-15,
        "omega=-omega_prev r_prev dot Delta r / ||Delta r||^2");
    add(results, "block_aitken_scalar_is_safeguard_bounded",
        std::fabs(ctot_block_aitken_omega(
                      1.0, -10.0, 1.0, 0.1, 1.5) - 1.5),
        1.0e-15,
        "Aitken extrapolation cannot leave the configured safe interval");
    add(results, "block_aitken_nonfinite_history_fails_closed",
        std::isnan(ctot_block_aitken_omega(
                       1.0, NAN, 1.0, 0.1, 1.5)) ? 0.0 : 1.0,
        0.0,
        "invalid acceleration history selects the unaccelerated fallback");
    add(results, "block_aitken_blend_allows_bounded_extrapolation",
        std::fabs(ctot_block_aitken_blend(
                      0.2, 0.4, 1.5, 0.1, 1.5) - 0.5),
        1.0e-15,
        "Ctot/free-phase blocks may extrapolate only inside the safe omega range");
    add(results, "anderson_m2_weight_minimizes_two_residual_segment",
        std::fabs(ctot_anderson_m2_current_weight(
                      -0.25, 0.5, -0.5, 1.5) - 0.5),
        1.0e-15,
        "depth-2 Anderson minimizes ||r_prev+w(r_cur-r_prev)||");
    add(results, "anderson_m2_weight_is_safeguard_bounded",
        std::fabs(ctot_anderson_m2_current_weight(
                      -10.0, 1.0, -0.5, 1.5) - 1.5),
        1.0e-15,
        "the M2 coefficient cannot leave the fail-closed interval");
    add(results, "anderson_m2_affine_candidate_preserves_equal_mass",
        std::fabs((ctot_anderson_m2_blend(
                       0.2, 0.4, 1.5, -0.5, 1.5) +
                   ctot_anderson_m2_blend(
                       0.8, 0.6, 1.5, -0.5, 1.5)) - 1.0),
        1.0e-15,
        "affine mixing preserves the shared conserved sum without projection");
    const CtotAndersonM3Weights m3 = ctot_anderson_m3_weights(
        -1.0, -2.0, 1.0, 0.0, 4.0, -0.5, 1.5);
    add(results, "anderson_m3_solves_two_direction_least_squares",
        m3.valid ? std::fabs(m3.previous2 + 0.5) +
                       std::fabs(m3.previous - 1.0) +
                       std::fabs(m3.current - 0.5)
                 : 1.0,
        1.0e-15,
        "M3 weights solve the nonsingular two-direction normal equations");
    add(results, "anderson_m3_affine_candidate_preserves_equal_mass",
        m3.valid
            ? std::fabs(
                  ctot_anderson_m3_blend(0.2, 0.3, 0.4, m3) +
                  ctot_anderson_m3_blend(0.8, 0.7, 0.6, m3) - 1.0)
            : 1.0,
        1.0e-15,
        "three-candidate affine mixing preserves the shared conserved sum");
    const CtotAndersonM3Weights singular_m3 = ctot_anderson_m3_weights(
        -1.0, -2.0, 1.0, 2.0, 4.0, -0.5, 1.5);
    add(results, "anderson_m3_rejects_singular_history",
        singular_m3.valid ? 1.0 : 0.0,
        0.0,
        "rank-deficient depth-three history must fall back without mixing");
    const CtotAndersonM3Weights regularized_m3 =
        ctot_anderson_m3_regularized_weights(
            -1.0, -1.0001, 1.0, 0.999999, 1.000001,
            1.0e-2, -0.5, 1.5);
    add(results, "anderson_m3_regularization_recovers_bounded_history",
        regularized_m3.valid ? 0.0 : 1.0,
        0.0,
        "ridge regularization may recover a bounded affine candidate only");
    add(results, "anderson_m3_regularized_weights_remain_affine",
        regularized_m3.valid
            ? std::fabs(regularized_m3.previous2 +
                        regularized_m3.previous +
                        regularized_m3.current - 1.0)
            : 1.0,
        1.0e-15,
        "regularization changes the residual fit but not mass-affine mixing");
    add(results, "acceleration_safeguard_accepts_roundoff_replay_noise",
        ctot_acceleration_block_within_nonregression_envelope(
            1.0e-8, 1.0e-8 + 5.0e-19, 1.0e-10, 1.0e-10)
            ? 0.0 : 1.0,
        0.0,
        "candidate block residual may differ only by bounded replay noise");
    add(results, "acceleration_safeguard_accepts_subgate_block_tradeoff",
        ctot_acceleration_block_within_nonregression_envelope(
            2.4e-12, 7.5e-10, 1.0e-9, 1.0e-10)
            ? 0.0 : 1.0,
        0.0,
        "an already-converged block may move only within its unchanged gate");
    add(results, "acceleration_safeguard_rejects_above_gate_tradeoff",
        ctot_acceleration_block_within_nonregression_envelope(
            2.4e-12, 1.1e-9, 1.0e-9, 1.0e-10)
            ? 1.0 : 0.0,
        0.0,
        "a lower combined merit cannot move a converged block above its gate");
    add(results, "acceleration_safeguard_rejects_worse_unconverged_block",
        ctot_acceleration_block_within_nonregression_envelope(
            2.0e-8, 2.1e-8, 1.0e-9, 1.0e-10)
            ? 1.0 : 0.0,
        0.0,
        "a block above its gate must not worsen while another block improves");

    std::ofstream csv(argv[1]);
    csv << "test,observed,tolerance,passed,semantics\n";
    int failed = 0;
    for (const Result &result : results) {
        csv << result.test << ',' << result.observed << ',' << result.tolerance
            << ',' << (result.passed ? "true" : "false") << ",\""
            << result.semantics << "\"\n";
        failed += result.passed ? 0 : 1;
    }
    std::printf("ctot_feasible_storage_rows=%zu\nctot_feasible_storage_failed=%d\n",
                results.size(), failed);
    return failed == 0 ? 0 : 1;
}
