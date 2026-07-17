#ifndef BDF2_EVENT_UTILS_H
#define BDF2_EVENT_UTILS_H

#include <cmath>
#include <cfloat>

#include "variable_bdf2_utils.h"

#ifdef __CUDACC__
#define BDF2_EVENT_HD __host__ __device__
#else
#define BDF2_EVENT_HD
#endif

enum Bdf2EventReasonV1 {
    BDF2_EVENT_NONE = 0,
    BDF2_EVENT_PHI_EXTRAPOLATION_CONTEXT_INFEASIBLE = 1,
    BDF2_EVENT_ANCHOR_CAPACITY_INFEASIBLE = 2,
    BDF2_EVENT_ACTIVE_SET_TRANSITION = 3,
    BDF2_EVENT_ENERGY_CONTEXT_UNDEFINED = 4,
    BDF2_EVENT_OTHER_VERSIONED_REASON = 5
};

struct Bdf2EventCellV1 {
    double C_anchor;
    double phi_E;
    double h_n;
    double h_E;
    double alpha_n;
    double alpha_E;
    double q_n;
    double q_anchor_E;
    double upper_margin_n;
    double upper_margin_E;
    int reason;
};

BDF2_EVENT_HD inline double bdf2_event_smoothstep(double phi) {
    const double p = fmin(fmax(phi, 0.0), 1.0);
    return p * p * p * (6.0 * p * p - 15.0 * p + 10.0);
}

BDF2_EVENT_HD inline double bdf2_event_alpha(double phi) {
    const double p = fmin(fmax(phi, 0.0), 1.0);
    return p > 0.5 ? bdf2_event_smoothstep(1.0 - p)
                   : 1.0 - bdf2_event_smoothstep(p);
}

BDF2_EVENT_HD inline double bdf2_event_h(double phi) {
    const double alpha = bdf2_event_alpha(phi);
    return phi > 0.5 ? 1.0 - alpha : bdf2_event_smoothstep(phi);
}

BDF2_EVENT_HD inline bool bdf2_event_isfinite(double value) {
#ifdef __CUDA_ARCH__
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

BDF2_EVENT_HD inline Bdf2EventCellV1 bdf2_event_classify_v1(
    double C_n, double C_nm1, double phi_n, double phi_nm1,
    double v_B, double bound_tol);

BDF2_EVENT_HD inline Bdf2EventCellV1 bdf2_event_classify_ratio_v1(
    double C_n, double C_nm1, double phi_n, double phi_nm1,
    double v_B, double bound_tol, double step_ratio) {
    Bdf2EventCellV1 out{};
    const VariableBdf2CoefficientsV1 coefficients =
        variable_bdf2_coefficients_v1(step_ratio, 1.0);
    if (!coefficients.valid) {
        out.reason = BDF2_EVENT_OTHER_VERSIONED_REASON;
        return out;
    }
    if (step_ratio == 1.0) {
        out.C_anchor = (4.0 * C_n - C_nm1) / 3.0;
        out.phi_E = 2.0 * phi_n - phi_nm1;
    } else {
        out.C_anchor = coefficients.anchor_n * C_n +
                       coefficients.anchor_nm1 * C_nm1;
        out.phi_E = coefficients.extrap_n * phi_n +
                    coefficients.extrap_nm1 * phi_nm1;
    }
    const double context_tol = 64.0 * DBL_EPSILON;
    const double phi_E_bounded = fmin(fmax(out.phi_E, 0.0), 1.0);
    out.alpha_n = bdf2_event_alpha(phi_n);
    out.alpha_E = bdf2_event_alpha(phi_E_bounded);
    out.h_n = phi_n > 0.5 ? 1.0 - out.alpha_n
                          : bdf2_event_smoothstep(phi_n);
    out.h_E = phi_E_bounded > 0.5 ? 1.0 - out.alpha_E
                                  : bdf2_event_smoothstep(phi_E_bounded);
    out.q_n = (C_n - v_B) + out.alpha_n * v_B;
    out.q_anchor_E = (out.C_anchor - v_B) + out.alpha_E * v_B;
    out.upper_margin_n = out.alpha_n - out.q_n;
    out.upper_margin_E = out.alpha_E - out.q_anchor_E;
    out.reason = BDF2_EVENT_NONE;

    if (!bdf2_event_isfinite(phi_n) ||
        !bdf2_event_isfinite(phi_nm1) ||
        !bdf2_event_isfinite(out.phi_E) ||
        out.phi_E < -context_tol || out.phi_E > 1.0 + context_tol) {
        out.reason = BDF2_EVENT_PHI_EXTRAPOLATION_CONTEXT_INFEASIBLE;
        return out;
    }
    if (!bdf2_event_isfinite(C_n) ||
        !bdf2_event_isfinite(C_nm1) ||
        !bdf2_event_isfinite(out.C_anchor)) {
        out.reason = BDF2_EVENT_ANCHOR_CAPACITY_INFEASIBLE;
        return out;
    }

    // Use the solver's registered active-set tolerance to define a branch.
    // Roundoff motion inside an already-active branch is not a new event.
    const bool lower_crossing =
        (out.q_n > bound_tol && out.q_anchor_E <= bound_tol) ||
        (out.q_n <= bound_tol && out.q_anchor_E > bound_tol);
    const bool upper_crossing =
        (out.upper_margin_n > bound_tol &&
         out.upper_margin_E <= bound_tol) ||
        (out.upper_margin_n <= bound_tol &&
         out.upper_margin_E > bound_tol);
    if (lower_crossing || upper_crossing) {
        out.reason = BDF2_EVENT_ACTIVE_SET_TRANSITION;
        return out;
    }

    if (out.q_anchor_E < -bound_tol ||
        out.q_anchor_E > out.alpha_E + bound_tol) {
        out.reason = BDF2_EVENT_ANCHOR_CAPACITY_INFEASIBLE;
        return out;
    }
    if (!bdf2_event_isfinite(out.h_E) ||
        !bdf2_event_isfinite(out.alpha_E) ||
        !bdf2_event_isfinite(out.q_anchor_E)) {
        out.reason = BDF2_EVENT_ENERGY_CONTEXT_UNDEFINED;
    }
    return out;
}

BDF2_EVENT_HD inline Bdf2EventCellV1 bdf2_event_classify_v1(
    double C_n, double C_nm1, double phi_n, double phi_nm1,
    double v_B, double bound_tol) {
    return bdf2_event_classify_ratio_v1(
        C_n, C_nm1, phi_n, phi_nm1, v_B, bound_tol, 1.0);
}

BDF2_EVENT_HD inline const char *bdf2_event_reason_name_v1(int reason) {
    switch (reason) {
        case BDF2_EVENT_NONE:
            return "NONE";
        case BDF2_EVENT_PHI_EXTRAPOLATION_CONTEXT_INFEASIBLE:
            return "PHI_EXTRAPOLATION_CONTEXT_INFEASIBLE";
        case BDF2_EVENT_ANCHOR_CAPACITY_INFEASIBLE:
            return "BDF2_ANCHOR_CAPACITY_INFEASIBLE";
        case BDF2_EVENT_ACTIVE_SET_TRANSITION:
            return "ACTIVE_SET_TRANSITION";
        case BDF2_EVENT_ENERGY_CONTEXT_UNDEFINED:
            return "ENERGY_CONTEXT_UNDEFINED";
        default:
            return "OTHER_VERSIONED_REASON";
    }
}

// Stable algebraic form of transport_chain + phase_chain + context_work.
// It uses only the two admissible endpoint energies and the method's actual
// transport/phase work terms; no artificial mixed state is constructed.
BDF2_EVENT_HD inline double bdf2_stable_endpoint_chain_v2(
    double F_np1_phi_np1, double F_n_phi_n,
    double mu_E_delta_C_n, double g_delta_phi_n) {
    return (F_np1_phi_np1 - F_n_phi_n) -
           mu_E_delta_C_n - g_delta_phi_n;
}

#undef BDF2_EVENT_HD

#endif
