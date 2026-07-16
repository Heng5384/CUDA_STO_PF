#ifndef CTOT_TRANSPORT_BOUND_UTILS_H
#define CTOT_TRANSPORT_BOUND_UTILS_H

#include <math.h>
#include <float.h>

#ifndef __CUDACC__
#define CTOT_HD inline
#else
#define CTOT_HD __host__ __device__ inline
#endif

enum CtotFeasibleStorageStatus {
    CTOT_STORAGE_INVALID = 0,
    CTOT_STORAGE_FREE = 1,
    CTOT_STORAGE_LOWER_ACTIVE = 2,
    CTOT_STORAGE_UPPER_ACTIVE = 3,
    CTOT_STORAGE_INACTIVE_SUPPORT = 4
};

struct CtotFeasibleStorageTrial {
    double C;
    double q;
    double x;
    double Y_context;
    double context_normalization_defect;
    int context_normalized;
    int status;
};

CTOT_HD double ctot_transport_logit_context(
    double x, double x_context_eps, double Y_safety_cap)
{
    double context = x;
    if (context < x_context_eps) context = x_context_eps;
    if (context > 1.0 - x_context_eps) context = 1.0 - x_context_eps;
    double Y = log(context / (1.0 - context));
    if (Y > Y_safety_cap) Y = Y_safety_cap;
    if (Y < -Y_safety_cap) Y = -Y_safety_cap;
    return Y;
}

// Build only the nonconserved thermodynamic context at a storage endpoint.
// The authoritative Ctot and the reported raw q are never changed.  A q value
// outside [0,alpha] by at most the registered 64-double-ulp reconstruction
// envelope is canonicalized for x/Y evaluation; a material violation remains
// invalid and must reject the enclosing transaction.
CTOT_HD double ctot_storage_context_q_ulp64(
    double q_raw, double alpha, double C, double v_B,
    int *normalized, double *normalization_defect)
{
    if (normalized) *normalized = 0;
    if (normalization_defect) *normalization_defect = 0.0;
    if (!isfinite(q_raw) || !isfinite(alpha) || !isfinite(C) ||
        !isfinite(v_B) || alpha < 0.0 || alpha > 1.0) return NAN;
    double scale = fabs(C);
    if (fabs(v_B) > scale) scale = fabs(v_B);
    if (fabs(alpha) > scale) scale = fabs(alpha);
    if (scale < 1.0) scale = 1.0;
    const double tolerance = 64.0 * DBL_EPSILON * scale;
    double q_context = q_raw;
    if (q_raw < 0.0) {
        if (q_raw < -tolerance) return NAN;
        q_context = 0.0;
    } else if (q_raw > alpha) {
        if (q_raw - alpha > tolerance) return NAN;
        q_context = alpha;
    }
    const double defect = fabs(q_context - q_raw);
    if (defect > 0.0) {
        if (normalized) *normalized = 1;
        if (normalization_defect) *normalization_defect = defect;
    }
    return q_context;
}

// Construct a line-search trial in the authoritative conserved coordinate.
// No physical state is clipped or projected: an infeasible trial is returned
// as CTOT_STORAGE_INVALID so the caller can reduce the line-search step.
CTOT_HD CtotFeasibleStorageTrial ctot_build_feasible_storage_trial(
    double C_current, double residual, double h, double alpha, double lambda,
    double v_B, double matrix_support_eps, double x_context_eps,
    double Y_safety_cap, double active_tol)
{
    CtotFeasibleStorageTrial out;
    out.C = C_current - lambda * residual;
    // q=C-h*v_B=(C-v_B)+(1-h)*v_B.  Carry alpha explicitly so
    // sub-ulp matrix capacity is not destroyed by a second 1-h subtraction.
    out.q = (out.C - v_B) + alpha * v_B;
    out.x = NAN;
    out.Y_context = NAN;
    out.context_normalization_defect = 0.0;
    out.context_normalized = 0;
    out.status = CTOT_STORAGE_INVALID;

    const double upper_q = alpha;
    if (!isfinite(C_current) || !isfinite(residual) || !isfinite(h) ||
        !isfinite(lambda) || !isfinite(v_B) || !isfinite(alpha) ||
        !isfinite(out.C) || !isfinite(out.q) || !(lambda > 0.0) ||
        alpha < 0.0 || alpha > 1.0) {
        return out;
    }
    const double q_context = ctot_storage_context_q_ulp64(
        out.q, upper_q, out.C, v_B,
        &out.context_normalized, &out.context_normalization_defect);
    if (!isfinite(q_context)) return out;

    if (alpha <= matrix_support_eps) {
        out.x = 0.0;
        out.Y_context = 0.0;
        out.status = CTOT_STORAGE_INACTIVE_SUPPORT;
        return out;
    }

    out.x = q_context / alpha;
    if (!isfinite(out.x) || out.x < 0.0 || out.x > 1.0) return out;
    out.Y_context = ctot_transport_logit_context(
        out.x, x_context_eps, Y_safety_cap);
    if (!isfinite(out.Y_context)) return out;
    if (q_context <= active_tol) out.status = CTOT_STORAGE_LOWER_ACTIVE;
    else if (upper_q - q_context <= active_tol)
        out.status = CTOT_STORAGE_UPPER_ACTIVE;
    else out.status = CTOT_STORAGE_FREE;
    return out;
}

// Return one only for a search-direction component that belongs to the local
// feasible tangent cone. The physical state is never modified here. With the
// update C_new=C-lambda*direction, positive direction is outward at the lower
// bound and negative direction is outward at the upper bound.
CTOT_HD int ctot_direction_is_free_in_mass_tangent(
    double C, double h, double alpha, double direction, double v_B,
    double matrix_support_eps, double active_tol, double min_lambda)
{
    const double q = (C - v_B) + alpha * v_B;
    if (!isfinite(C) || !isfinite(h) || !isfinite(direction) ||
        !isfinite(alpha) || !isfinite(q) || !isfinite(min_lambda) ||
        min_lambda < 0.0 || alpha <= matrix_support_eps ||
        alpha < 0.0 || alpha > 1.0) return 0;
    const double q_context = ctot_storage_context_q_ulp64(
        q, alpha, C, v_B, nullptr, nullptr);
    if (!isfinite(q_context)) return 0;
    // A component is numerically active whenever even the smallest permitted
    // line-search step would cross its local storage bound.  This keeps the
    // tangent cone and line-search contract consistent without clipping C.
    if (direction > 0.0 &&
        q_context <= active_tol + min_lambda * direction) return 0;
    if (direction < 0.0 &&
        alpha - q_context <= active_tol + min_lambda * (-direction)) return 0;
    return 1;
}

// Integer form of the runtime's strict 2/3 de-alias predicate. For a signed
// Fourier mode n, |k| < (2/3) k_Nyquist is exactly 3|n| < N. Keeping this
// predicate beside the nonlinear preconditioner avoids damping modes whose
// diffusive Jacobian contribution has already been removed by de-aliasing.
CTOT_HD int ctot_mode_retained_by_23(int storage_index, int extent)
{
    if (extent <= 0 || storage_index < 0 || storage_index >= extent) return 0;
    const int signed_mode = storage_index <= extent / 2
        ? storage_index : storage_index - extent;
    const int magnitude = signed_mode < 0 ? -signed_mode : signed_mode;
    return 3 * magnitude < extent;
}

CTOT_HD int ctot_r2c_mode_retained_by_23(
    int i, int j, int k, int Nx, int Ny, int Nz)
{
    if (k < 0 || k > Nz / 2) return 0;
    return ctot_mode_retained_by_23(i, Nx) &&
           ctot_mode_retained_by_23(j, Ny) &&
           3 * k < Nz;
}

CTOT_HD double ctot_outer_convex_blend(
    double previous, double candidate, double omega)
{
    if (!isfinite(previous) || !isfinite(candidate) || !isfinite(omega) ||
        omega < 0.0 || omega > 1.0) return NAN;
    return previous + omega * (candidate - previous);
}

// A local phase-feasibility pass has numerical work to continue only when at
// least one cell was changed. Re-solving the identical fixed-phi transport
// problem cannot produce new information and must fall through to the existing
// safeguarded outer backtrack instead.
CTOT_HD int ctot_local_phase_filter_made_progress(
    double limited_count, double extrapolated_lower_count,
    double phi_change_linf)
{
    return isfinite(limited_count) && isfinite(extrapolated_lower_count) &&
           isfinite(phi_change_linf) && phi_change_linf > 0.0 &&
           limited_count + extrapolated_lower_count > 0.5;
}

// Return the next backtracking value while guaranteeing that the exact
// configured floor is evaluated once.  A pure lambda*=0.5 sequence can jump
// from above the floor to below it and silently skip the only step covered by
// the minimum-step tangent-feasibility contract.
CTOT_HD double ctot_next_line_search_lambda(
    double current, double minimum)
{
    if (!isfinite(current) || !isfinite(minimum) ||
        !(current > 0.0) || !(minimum > 0.0) || current <= minimum) {
        return 0.0;
    }
    const double halved = 0.5 * current;
    return halved < minimum ? minimum : halved;
}

// Exact finite-increment product rule for q=alpha*x.  This is diagnostic:
// neither x nor alpha is promoted to the conserved state.
CTOT_HD double ctot_q_increment_from_capacity_product_rule(
    double alpha_new, double x_new, double alpha_old, double x_old)
{
    return alpha_new * (x_new - x_old) +
           x_old * (alpha_new - alpha_old);
}

// Convert a shared-face flux change to the dimensionless one-step C change
// it can induce across one cell.  This keeps every physical face visible in
// the outer contract while avoiding raw flux units in the convergence test.
CTOT_HD double ctot_scaled_face_flux_increment(
    double flux_new, double flux_old, double dt, double spacing,
    double ctot_scale)
{
    if (!isfinite(flux_new) || !isfinite(flux_old) || !isfinite(dt) ||
        !isfinite(spacing) || !isfinite(ctot_scale) || !(dt > 0.0) ||
        !(spacing > 0.0) || !(ctot_scale > 0.0)) return NAN;
    return dt * (flux_new - flux_old) / (spacing * ctot_scale);
}

CTOT_HD double ctot_block_aitken_omega(
    double omega_previous, double previous_dot_difference,
    double difference_norm_sq, double omega_min, double omega_max)
{
    if (!isfinite(omega_previous) ||
        !isfinite(previous_dot_difference) ||
        !isfinite(difference_norm_sq) || !isfinite(omega_min) ||
        !isfinite(omega_max) || !(omega_previous > 0.0) ||
        !(difference_norm_sq > 0.0) || !(omega_min > 0.0) ||
        !(omega_max >= omega_min)) return NAN;
    double omega = -omega_previous * previous_dot_difference /
                   difference_norm_sq;
    if (!isfinite(omega)) return NAN;
    if (omega < omega_min) omega = omega_min;
    if (omega > omega_max) omega = omega_max;
    return omega;
}

CTOT_HD double ctot_block_aitken_blend(
    double previous, double fixed_point_candidate, double omega,
    double omega_min, double omega_max)
{
    if (!isfinite(previous) || !isfinite(fixed_point_candidate) ||
        !isfinite(omega) || omega < omega_min || omega > omega_max)
        return NAN;
    return previous + omega * (fixed_point_candidate - previous);
}

CTOT_HD double ctot_anderson_m2_current_weight(
    double previous_dot_difference, double difference_norm_sq,
    double weight_min, double weight_max)
{
    if (!isfinite(previous_dot_difference) ||
        !isfinite(difference_norm_sq) || !isfinite(weight_min) ||
        !isfinite(weight_max) || !(difference_norm_sq > 0.0) ||
        !(weight_max >= weight_min)) return NAN;
    double weight = -previous_dot_difference / difference_norm_sq;
    if (!isfinite(weight)) return NAN;
    if (weight < weight_min) weight = weight_min;
    if (weight > weight_max) weight = weight_max;
    return weight;
}

CTOT_HD double ctot_anderson_m2_blend(
    double previous_candidate, double current_candidate,
    double current_weight, double weight_min, double weight_max)
{
    if (!isfinite(previous_candidate) || !isfinite(current_candidate) ||
        !isfinite(current_weight) || current_weight < weight_min ||
        current_weight > weight_max) return NAN;
    return previous_candidate +
           current_weight * (current_candidate - previous_candidate);
}

struct CtotAndersonM3Weights {
    double previous2;
    double previous;
    double current;
    int valid;
};

// Minimize ||r0 + a(r1-r0) + b(r2-r0)|| in the two-dimensional
// residual-history subspace.  The three returned affine coefficients sum to
// one; no clipping is permitted because that would silently change the
// least-squares problem.  A coefficient outside the safeguard box invalidates
// the candidate and triggers the unaccelerated fallback.
CTOT_HD CtotAndersonM3Weights ctot_anderson_m3_weights(
    double r0_dot_d1, double r0_dot_d2,
    double d1_dot_d1, double d1_dot_d2, double d2_dot_d2,
    double weight_min, double weight_max)
{
    CtotAndersonM3Weights out = {NAN, NAN, NAN, 0};
    if (!isfinite(r0_dot_d1) || !isfinite(r0_dot_d2) ||
        !isfinite(d1_dot_d1) || !isfinite(d1_dot_d2) ||
        !isfinite(d2_dot_d2) || !isfinite(weight_min) ||
        !isfinite(weight_max) || !(d1_dot_d1 > 0.0) ||
        !(d2_dot_d2 > 0.0) || !(weight_max >= weight_min)) return out;
    const double determinant =
        d1_dot_d1 * d2_dot_d2 - d1_dot_d2 * d1_dot_d2;
    const double gram_scale = d1_dot_d1 * d2_dot_d2;
    if (!isfinite(determinant) || !isfinite(gram_scale) ||
        determinant <= 1.0e-14 * gram_scale) return out;
    const double a =
        (-r0_dot_d1 * d2_dot_d2 + r0_dot_d2 * d1_dot_d2) /
        determinant;
    const double b =
        (-r0_dot_d2 * d1_dot_d1 + r0_dot_d1 * d1_dot_d2) /
        determinant;
    const double w0 = 1.0 - a - b;
    if (!isfinite(w0) || !isfinite(a) || !isfinite(b) ||
        w0 < weight_min || w0 > weight_max ||
        a < weight_min || a > weight_max ||
        b < weight_min || b > weight_max) return out;
    out.previous2 = w0;
    out.previous = a;
    out.current = b;
    out.valid = 1;
    return out;
}

CTOT_HD CtotAndersonM3Weights ctot_anderson_m3_regularized_weights(
    double r0_dot_d1, double r0_dot_d2,
    double d1_dot_d1, double d1_dot_d2, double d2_dot_d2,
    double ridge_relative, double weight_min, double weight_max)
{
    CtotAndersonM3Weights out = {NAN, NAN, NAN, 0};
    if (!isfinite(ridge_relative) || ridge_relative < 0.0 ||
        !isfinite(d1_dot_d1) || !isfinite(d2_dot_d2)) return out;
    const double trace = d1_dot_d1 + d2_dot_d2;
    if (!(trace > 0.0) || !isfinite(trace)) return out;
    const double ridge = ridge_relative * trace;
    return ctot_anderson_m3_weights(
        r0_dot_d1, r0_dot_d2,
        d1_dot_d1 + ridge, d1_dot_d2, d2_dot_d2 + ridge,
        weight_min, weight_max);
}

CTOT_HD double ctot_anderson_m3_blend(
    double previous2_candidate, double previous_candidate,
    double current_candidate, CtotAndersonM3Weights weights)
{
    if (!weights.valid || !isfinite(previous2_candidate) ||
        !isfinite(previous_candidate) || !isfinite(current_candidate))
        return NAN;
    return weights.previous2 * previous2_candidate +
           weights.previous * previous_candidate +
           weights.current * current_candidate;
}

// Acceleration is a numerical wrapper around the frozen nonlinear map.  A
// block that is still above its unchanged final acceptance gate may not get
// worse.  A block already below that gate may move within the same gate while
// another block is reduced; otherwise a tiny, already-converged block can
// prevent a coupled fixed-point cycle from being globalized.  This is an
// intermediate-iterate safeguard only and does not change final acceptance.
CTOT_HD int ctot_acceleration_block_within_nonregression_envelope(
    double before, double after, double acceptance_gate,
    double relative_replay_slack)
{
    if (!isfinite(before) || !isfinite(after) ||
        !isfinite(acceptance_gate) || !isfinite(relative_replay_slack) ||
        before < 0.0 || after < 0.0 || !(acceptance_gate > 0.0) ||
        relative_replay_slack < 0.0) return 0;
    const double envelope = fmax(before, acceptance_gate);
    return after <= envelope + relative_replay_slack * envelope;
}

#undef CTOT_HD

#endif
