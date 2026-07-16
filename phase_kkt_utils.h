#ifndef PHASE_KKT_UTILS_H
#define PHASE_KKT_UTILS_H

#include <math.h>

#if defined(__CUDACC__)
#define PHASE_KKT_HD __host__ __device__
#else
#define PHASE_KKT_HD
#endif

enum PhaseKktActiveCode {
    PHASE_KKT_FREE = 0,
    PHASE_KKT_LOWER_ACTIVE = 1,
    PHASE_KKT_UPPER_ACTIVE = 2,
    PHASE_KKT_INVALID = 3
};

PHASE_KKT_HD static inline double phase_kkt_active_line_trial(
    double phi, double delta, double active_target, double lambda,
    int active_code, int damp_active_constraints) {
    const int bound_active = active_code == PHASE_KKT_LOWER_ACTIVE ||
                             active_code == PHASE_KKT_UPPER_ACTIVE;
    if (!bound_active) return phi + lambda * delta;
    return damp_active_constraints
        ? phi + lambda * (active_target - phi) : active_target;
}

struct PhaseKktBounds {
    double phi_lower;
    double phi_upper;
    double h_lower;
    double h_upper;
    int valid;
};

PHASE_KKT_HD static inline double phase_kkt_h(double phi) {
    const double p2 = phi * phi;
    const double p3 = p2 * phi;
    return p3 * (6.0 * p2 - 15.0 * phi + 10.0);
}

PHASE_KKT_HD static inline double phase_kkt_alpha(double phi) {
    // Quintic smoothstep satisfies 1-h(phi)=h(1-phi).  The symmetric form
    // avoids subtracting two nearly equal doubles in the beta endpoint.
    return phi > 0.5 ? phase_kkt_h(1.0 - phi)
                     : 1.0 - phase_kkt_h(phi);
}

PHASE_KKT_HD static inline double phase_kkt_h_stable(double phi) {
    return phi > 0.5 ? 1.0 - phase_kkt_alpha(phi) : phase_kkt_h(phi);
}

PHASE_KKT_HD static inline double phase_kkt_q_from_ctot(
    double phi, double C, double v_B) {
    const double alpha = phase_kkt_alpha(phi);
    // C-h*v_B = (C-v_B) + (1-h)*v_B.  This form preserves the small
    // matrix inventory when h is within a few ulps of one.
    return (C - v_B) + alpha * v_B;
}

PHASE_KKT_HD static inline double phase_kkt_x_from_ctot(
    double phi, double C, double v_B) {
    const double alpha = phase_kkt_alpha(phi);
    return alpha > 0.0
        ? phase_kkt_q_from_ctot(phi, C, v_B) / alpha : NAN;
}

PHASE_KKT_HD static inline int phase_kkt_h_less_than(
    double phi, double target) {
    if (phi > 0.5 && target > 0.5)
        return phase_kkt_alpha(phi) > 1.0 - target;
    return phase_kkt_h(phi) < target;
}

PHASE_KKT_HD static inline double phase_kkt_h_prime(double phi) {
    const double one_minus = 1.0 - phi;
    return 30.0 * phi * phi * one_minus * one_minus;
}

PHASE_KKT_HD static inline double phase_kkt_fixed_ctot_dx_dphi(
    double phi, double C, double v_B, double matrix_support_eps) {
    const double alpha = phase_kkt_alpha(phi);
    if (!(alpha > matrix_support_eps)) return 0.0;
    const double x = phase_kkt_q_from_ctot(phi, C, v_B) / alpha;
    return phase_kkt_h_prime(phi) * (x - v_B) / alpha;
}

PHASE_KKT_HD static inline double phase_kkt_h_inverse(double target) {
    if (target <= 0.0) return 0.0;
    if (target >= 1.0) return 1.0;
    double lo = 0.0;
    double hi = 1.0;
    for (int iter = 0; iter < 64; ++iter) {
        const double mid = 0.5 * (lo + hi);
        if (phase_kkt_h_less_than(mid, target)) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

PHASE_KKT_HD static inline double phase_kkt_h_inverse_lower_feasible(
    double target) {
    if (target <= 0.0) return 0.0;
    if (target >= 1.0) return 1.0;
    double lo = 0.0;
    double hi = 1.0;
    for (int iter = 0; iter < 64; ++iter) {
        const double mid = 0.5 * (lo + hi);
        if (phase_kkt_h_less_than(mid, target)) lo = mid;
        else hi = mid;
    }
    return hi;
}

PHASE_KKT_HD static inline double phase_kkt_h_inverse_upper_feasible(
    double target) {
    if (target <= 0.0) return 0.0;
    if (target >= 1.0) return 1.0;
    double lo = 0.0;
    double hi = 1.0;
    for (int iter = 0; iter < 64; ++iter) {
        const double mid = 0.5 * (lo + hi);
        if (phase_kkt_h_less_than(mid, target)) lo = mid;
        else hi = mid;
    }
    return lo;
}

PHASE_KKT_HD static inline PhaseKktBounds phase_kkt_bounds_from_ctot(
    double C, double v_B, double x_min, double x_max, double tolerance) {
    PhaseKktBounds result;
    result.phi_lower = 0.0;
    result.phi_upper = 1.0;
    result.h_lower = 0.0;
    result.h_upper = 1.0;
    result.valid = 0;
    if (!isfinite(C) || !isfinite(v_B) || !isfinite(x_min) ||
        !isfinite(x_max) || !(x_min >= 0.0) || !(x_min < x_max) ||
        !(x_max < v_B)) {
        return result;
    }
    const double alpha_lower = fmax(
        0.0, fmin(1.0, (v_B - C) / (v_B - x_min)));
    const double alpha_upper = fmax(
        0.0, fmin(1.0, (v_B - C) / (v_B - x_max)));
    result.h_lower = fmax(0.0, (C - x_max) / (v_B - x_max));
    result.h_upper = fmin(1.0, (C - x_min) / (v_B - x_min));
    result.h_lower = fmax(0.0, fmin(1.0, result.h_lower));
    result.h_upper = fmax(0.0, fmin(1.0, result.h_upper));
    if (result.h_lower > result.h_upper + tolerance ||
        alpha_lower > alpha_upper + tolerance) return result;
    // Near beta, h is numerically indistinguishable from one long before
    // alpha=1-h loses its physical matrix-capacity information.  Invert the
    // smaller of h and alpha=h(1-phi), never a rounded 1-small target.
    result.phi_lower = result.h_lower <= 0.5
        ? phase_kkt_h_inverse_lower_feasible(result.h_lower)
        : 1.0 - phase_kkt_h_inverse_upper_feasible(alpha_upper);
    result.phi_upper = result.h_upper <= 0.5
        ? phase_kkt_h_inverse_upper_feasible(result.h_upper)
        : 1.0 - phase_kkt_h_inverse_lower_feasible(alpha_lower);
    // The subtraction 1-y can itself round by one ulp.  Move only the
    // algorithmic box endpoint inward until its reconstructed composition is
    // representably feasible; Ctot is never altered.
    for (int iter = 0; iter < 8 && result.phi_lower < 1.0; ++iter) {
        const double x = phase_kkt_x_from_ctot(
            result.phi_lower, C, v_B);
        if (!isfinite(x) || x <= x_max) break;
        result.phi_lower = nextafter(result.phi_lower, 1.0);
    }
    for (int iter = 0; iter < 8 && result.phi_upper > 0.0; ++iter) {
        const double x = phase_kkt_x_from_ctot(
            result.phi_upper, C, v_B);
        if (!isfinite(x) || x >= x_min) break;
        result.phi_upper = nextafter(result.phi_upper, 0.0);
    }
    result.valid = isfinite(result.phi_lower) && isfinite(result.phi_upper) &&
                   result.phi_lower <= result.phi_upper + tolerance;
    return result;
}

PHASE_KKT_HD static inline double phase_kkt_projected_defect(
    double phi, double residual, double alpha, const PhaseKktBounds &bounds) {
    if (!bounds.valid || !isfinite(phi) || !isfinite(residual) ||
        !(alpha > 0.0)) return NAN;
    const double z = phi - alpha * residual;
    const double projected = fmin(fmax(z, bounds.phi_lower), bounds.phi_upper);
    return phi - projected;
}

PHASE_KKT_HD static inline int phase_kkt_active_code(
    double phi, double residual, double alpha, const PhaseKktBounds &bounds) {
    if (!bounds.valid || !isfinite(phi) || !isfinite(residual) ||
        !(alpha > 0.0)) return PHASE_KKT_INVALID;
    const double z = phi - alpha * residual;
    if (z <= bounds.phi_lower) return PHASE_KKT_LOWER_ACTIVE;
    if (z >= bounds.phi_upper) return PHASE_KKT_UPPER_ACTIVE;
    return PHASE_KKT_FREE;
}

PHASE_KKT_HD static inline double phase_kkt_richardson_derivative(
    double coarse, double fine, int centered_stencil) {
    if (!isfinite(coarse) || !isfinite(fine)) return NAN;
    // A centered secant has O(h^2) error; a bound-clipped one-sided secant
    // has O(h) error.  Eliminate the corresponding leading term without
    // changing the nonlinear residual used for final KKT acceptance.
    return centered_stencil ? (4.0 * fine - coarse) / 3.0
                            : 2.0 * fine - coarse;
}

PHASE_KKT_HD static inline double phase_kkt_fixed_ctot_fd_step(
    double phi, double C, double v_B, double relative_step) {
    double step = relative_step * fmax(1.0, fabs(phi));
    const double alpha = phase_kkt_alpha(phi);
    if (alpha > 1.0e-14) {
        const double x = phase_kkt_q_from_ctot(phi, C, v_B) / alpha;
        const double dx_dphi =
            phase_kkt_h_prime(phi) * (x - v_B) / alpha;
        if (isfinite(x) && isfinite(dx_dphi) && fabs(dx_dphi) > 0.0) {
            // The fixed-Ctot map becomes singular as matrix support vanishes.
            // Keep the induced composition perturbation local as well as phi.
            step = fmin(step,
                        relative_step * fmax(1.0, fabs(x)) / fabs(dx_dphi));
        }
    }
    const double roundoff_floor = 64.0 * 2.2204460492503131e-16 *
                                  fmax(1.0, fabs(phi));
    return fmax(step, roundoff_floor);
}

#undef PHASE_KKT_HD

#endif
