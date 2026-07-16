#ifndef ACTIVE_MANIFOLD_BDF2_UTILS_H
#define ACTIVE_MANIFOLD_BDF2_UTILS_H

#include <cfloat>
#include <cmath>

#include "phase_kkt_utils.h"

#ifdef __CUDACC__
#define ACTIVE_MANIFOLD_HD __host__ __device__
#else
#define ACTIVE_MANIFOLD_HD
#endif

enum ActiveManifoldBdf2BranchV1 {
    ACTIVE_MANIFOLD_BDF2_FREE = 0,
    ACTIVE_MANIFOLD_BDF2_QALPHA_LOWER = 1,
    ACTIVE_MANIFOLD_BDF2_PHI_LOWER = 2,
    ACTIVE_MANIFOLD_BDF2_ULP_ENDPOINT = 3,
    ACTIVE_MANIFOLD_BDF2_INVALID = 4
};

struct ActiveManifoldBdf2ContextV1 {
    double C_anchor;
    double phi_raw;
    double phi_context;
    double h_raw;
    double h_context;
    double alpha_context;
    double q_n;
    double q_raw;
    double q_context;
    double upper_margin_context;
    double normal_correction_phi;
    int branch;
    int valid;
};

ACTIVE_MANIFOLD_HD inline double active_manifold_ulp64_scale(
    double a, double b, double c) {
    double scale = fmax(fabs(a), fmax(fabs(b), fabs(c)));
    if (scale < 1.0) scale = 1.0;
    return 64.0 * DBL_EPSILON * scale;
}

// Construct a coefficient context only. C_n, C_nm1, phi_n, and phi_nm1 are
// authoritative accepted states and are never changed. For v_B=1 the lower
// storage manifold is q_alpha=C_anchor-h(phi)=0. If the unconstrained BDF2
// phase extrapolation points outside that manifold, the context is evaluated
// at its exact monotone h-inverse. The upper-feasible inverse is used so
// representable roundoff stays on the admissible side of q_alpha=0.
ACTIVE_MANIFOLD_HD inline ActiveManifoldBdf2ContextV1
active_manifold_bdf2_context_v1(
    double C_n, double C_nm1, double phi_n, double phi_nm1,
    double v_B, double bound_tol) {
    ActiveManifoldBdf2ContextV1 out{};
    out.C_anchor = (4.0 * C_n - C_nm1) / 3.0;
    out.phi_raw = 2.0 * phi_n - phi_nm1;
    out.phi_context = out.phi_raw;
    out.branch = ACTIVE_MANIFOLD_BDF2_INVALID;
    out.valid = 0;

    if (!isfinite(C_n) || !isfinite(C_nm1) || !isfinite(phi_n) ||
        !isfinite(phi_nm1) || !isfinite(out.C_anchor) ||
        !isfinite(out.phi_raw) || !isfinite(v_B) ||
        fabs(v_B - 1.0) > 1.0e-14 || !(bound_tol >= 0.0)) {
        return out;
    }

    const double endpoint_tol = 64.0 * DBL_EPSILON;
    const double storage_ulp = active_manifold_ulp64_scale(
        out.C_anchor, v_B, 1.0);
    const double phi_bounded = fmin(fmax(out.phi_raw, 0.0), 1.0);
    out.h_raw = phase_kkt_h_stable(phi_bounded);
    const double alpha_raw = phase_kkt_alpha(phi_bounded);
    out.q_n = phase_kkt_q_from_ctot(phi_n, C_n, v_B);
    out.q_raw = (out.C_anchor - v_B) + alpha_raw * v_B;
    const double upper_raw = alpha_raw - out.q_raw;

    // At the pure-alpha endpoint, phi >= 0 is itself an active constraint.
    // h'(0)=0 makes accepted positive phi values whose h-storage is below the
    // floating-point storage resolution equivalent to phi=0 for the conserved
    // coordinate. If the BDF2 extrapolate points farther out of the feasible
    // tangent cone, evaluate coefficients on that exact endpoint. This changes
    // neither accepted phi history nor the authoritative conserved state.
    const double phi_n_bounded = fmin(fmax(phi_n, 0.0), 1.0);
    const double h_n = phase_kkt_h_stable(phi_n_bounded);
    const int outward_pure_alpha_excursion =
        out.phi_raw < -endpoint_tol &&
        phi_n >= -endpoint_tol && h_n <= storage_ulp &&
        phi_nm1 >= phi_n;
    if (outward_pure_alpha_excursion) {
        if (out.C_anchor < -storage_ulp ||
            out.C_anchor > v_B + storage_ulp) {
            return out;
        }
        out.phi_context = 0.0;
        out.h_context = 0.0;
        out.alpha_context = 1.0;
        out.q_context = out.C_anchor;
        out.upper_margin_context = 1.0 - out.C_anchor;
        out.normal_correction_phi = -out.phi_raw;
        if (!isfinite(out.q_context) ||
            out.q_context < -storage_ulp ||
            out.upper_margin_context < -storage_ulp) {
            return out;
        }
        out.branch = ACTIVE_MANIFOLD_BDF2_PHI_LOWER;
        out.valid = 1;
        return out;
    }

    const int material_lower_excursion = out.q_raw < -storage_ulp;
    const int outward_pure_beta_excursion =
        out.phi_raw > 1.0 + endpoint_tol && out.q_n <= bound_tol;
    if (material_lower_excursion || outward_pure_beta_excursion) {
        if (out.C_anchor < -storage_ulp ||
            out.C_anchor > v_B + storage_ulp) {
            return out;
        }
        const double h_target = fmin(fmax(out.C_anchor / v_B, 0.0), 1.0);
        out.phi_context = phase_kkt_h_inverse_upper_feasible(h_target);
        out.h_context = phase_kkt_h_stable(out.phi_context);
        out.alpha_context = phase_kkt_alpha(out.phi_context);
        out.q_context =
            (out.C_anchor - v_B) + out.alpha_context * v_B;
        out.upper_margin_context = out.alpha_context - out.q_context;
        out.normal_correction_phi = out.phi_context - out.phi_raw;
        const double context_ulp = active_manifold_ulp64_scale(
            out.C_anchor, out.alpha_context, v_B);
        if (!isfinite(out.phi_context) || !isfinite(out.q_context) ||
            out.phi_context < 0.0 || out.phi_context > 1.0 ||
            out.q_context < -context_ulp ||
            out.q_context > out.alpha_context + context_ulp) {
            return out;
        }
        out.branch = ACTIVE_MANIFOLD_BDF2_QALPHA_LOWER;
        out.valid = 1;
        return out;
    }

    if (out.phi_raw < -endpoint_tol || out.phi_raw > 1.0 + endpoint_tol ||
        out.q_raw < -storage_ulp || upper_raw < -storage_ulp) {
        return out;
    }

    out.phi_context = phi_bounded;
    out.h_context = phase_kkt_h_stable(out.phi_context);
    out.alpha_context = phase_kkt_alpha(out.phi_context);
    out.q_context =
        (out.C_anchor - v_B) + out.alpha_context * v_B;
    out.upper_margin_context = out.alpha_context - out.q_context;
    out.normal_correction_phi = out.phi_context - out.phi_raw;
    out.branch =
        (out.phi_context != out.phi_raw || out.q_context < 0.0 ||
         out.upper_margin_context < 0.0)
            ? ACTIVE_MANIFOLD_BDF2_ULP_ENDPOINT
            : ACTIVE_MANIFOLD_BDF2_FREE;
    out.valid = 1;
    return out;
}

ACTIVE_MANIFOLD_HD inline const char *active_manifold_bdf2_branch_name_v1(
    int branch) {
    switch (branch) {
        case ACTIVE_MANIFOLD_BDF2_FREE:
            return "FREE_SECOND_ORDER";
        case ACTIVE_MANIFOLD_BDF2_QALPHA_LOWER:
            return "QALPHA_LOWER_MANIFOLD";
        case ACTIVE_MANIFOLD_BDF2_PHI_LOWER:
            return "PHI_LOWER_MANIFOLD";
        case ACTIVE_MANIFOLD_BDF2_ULP_ENDPOINT:
            return "ULP_ENDPOINT_CONTEXT";
        default:
            return "INVALID_CONTEXT";
    }
}

#undef ACTIVE_MANIFOLD_HD

#endif
