#ifndef LOW_MEMORY_TRANSPORT_V1_UTILS_H
#define LOW_MEMORY_TRANSPORT_V1_UTILS_H

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define LMT_HD __host__ __device__
#else
#define LMT_HD
#endif

enum CtotTransportEfficiencyFeature : uint32_t {
    CTOT_EFFICIENCY_PLATEAU_ESCALATION = 1u << 0,
    CTOT_EFFICIENCY_EXACT_FRACTION_TO_BOUNDARY = 1u << 1,
    CTOT_EFFICIENCY_HYBRID_L2_LINF_MERIT = 1u << 2,
    CTOT_EFFICIENCY_ADAPTIVE_SPECTRAL_SCALAR = 1u << 3,
    CTOT_EFFICIENCY_INTERNAL_HOMOTOPY = 1u << 4,
};

static constexpr uint32_t CTOT_EFFICIENCY_ALL_FEATURES =
    CTOT_EFFICIENCY_PLATEAU_ESCALATION |
    CTOT_EFFICIENCY_EXACT_FRACTION_TO_BOUNDARY |
    CTOT_EFFICIENCY_HYBRID_L2_LINF_MERIT |
    CTOT_EFFICIENCY_ADAPTIVE_SPECTRAL_SCALAR |
    CTOT_EFFICIENCY_INTERNAL_HOMOTOPY;

static constexpr uint32_t CTOT_EFFICIENCY_IMPLEMENTED_FEATURES =
    CTOT_EFFICIENCY_PLATEAU_ESCALATION |
    CTOT_EFFICIENCY_EXACT_FRACTION_TO_BOUNDARY |
    CTOT_EFFICIENCY_HYBRID_L2_LINF_MERIT |
    CTOT_EFFICIENCY_ADAPTIVE_SPECTRAL_SCALAR |
    CTOT_EFFICIENCY_INTERNAL_HOMOTOPY;

static constexpr int CTOT_LMT_HOMOTOPY_LEVELS = 3;

LMT_HD inline double ctot_lmt_homotopy_theta(int level)
{
    return level == 0 ? 0.25 : (level == 1 ? 0.5 : 1.0);
}

LMT_HD inline double ctot_lmt_clamp(double value, double lower, double upper)
{
    if (value < lower) return lower;
    if (value > upper) return upper;
    return value;
}

LMT_HD inline bool ctot_lmt_isfinite(double value)
{
#if defined(__CUDA_ARCH__)
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

LMT_HD inline bool ctot_lmt_feature_enabled(uint32_t mask, uint32_t feature)
{
    return (mask & feature) != 0u;
}

// q_new = q - lambda*d with q=C-h*vB. Inactive cells and a zero direction do
// not constrain the global step. Invalid local storage fails closed with zero.
LMT_HD inline double ctot_lmt_fraction_to_boundary_local(
    double C, double h, double alpha, double direction, double v_B,
    double matrix_support_eps)
{
    if (!ctot_lmt_isfinite(C) || !ctot_lmt_isfinite(h) ||
        !ctot_lmt_isfinite(alpha) || !ctot_lmt_isfinite(direction) ||
        !ctot_lmt_isfinite(v_B) ||
        !(v_B > 0.0) || !(matrix_support_eps > 0.0) ||
        h < 0.0 || h > 1.0 || alpha < 0.0 || alpha > 1.0) {
        return 0.0;
    }
    if (alpha <= matrix_support_eps || direction == 0.0) return 1.0;
    // Match the authoritative storage reconstruction. This algebraic form
    // retains the small matrix capacity when h is within a few ulps of one.
    const double q = (C - v_B) + alpha * v_B;
    const double roundoff_tol = 64.0 * 2.2204460492503131e-16 *
        fmax(1.0, fmax(fabs(C), fmax(fabs(h * v_B), fabs(alpha))));
    if (q < -roundoff_tol || q > alpha + roundoff_tol) return 0.0;
    const double q_admissible = ctot_lmt_clamp(q, 0.0, alpha);
    double bound = 1.0;
    if (direction > 0.0) {
        bound = q_admissible / direction;
    } else if (direction < 0.0) {
        bound = (alpha - q_admissible) / (-direction);
    }
    if (!ctot_lmt_isfinite(bound) || bound < 0.0) return 0.0;
    return fmin(1.0, bound);
}

LMT_HD inline double ctot_lmt_fraction_to_boundary_initial_lambda(
    double lambda_feasible)
{
    if (!ctot_lmt_isfinite(lambda_feasible) ||
        !(lambda_feasible > 0.0)) return 0.0;
    return fmin(1.0, 0.995 * lambda_feasible);
}

// The final cold Linf gate is unchanged. This function only chooses whether a
// trial may become the next nonlinear iterate.
LMT_HD inline bool ctot_lmt_hybrid_merit_accept(
    double current_l2, double current_linf,
    double trial_l2, double trial_linf,
    double lambda, double production_gate)
{
    if (!ctot_lmt_isfinite(current_l2) ||
        !ctot_lmt_isfinite(current_linf) ||
        !ctot_lmt_isfinite(trial_l2) ||
        !ctot_lmt_isfinite(trial_linf) ||
        !ctot_lmt_isfinite(lambda) ||
        !ctot_lmt_isfinite(production_gate) ||
        !(lambda > 0.0) || !(production_gate > 0.0)) return false;
    if (trial_linf <= production_gate) return true;
    if (current_linf > 100.0 * production_gate) {
        return trial_l2 < current_l2 * (1.0 - 1.0e-4 * lambda);
    }
    if (!(trial_l2 < current_l2)) return false;
    if (current_linf <= 10.0 * production_gate) {
        return trial_linf < current_linf;
    }
    return trial_linf <= current_linf;
}

struct CtotTransportPlateauWindowV1 {
    double residuals[9];
    int count;

    LMT_HD void reset()
    {
        for (int i = 0; i < 9; ++i) residuals[i] = 0.0;
        count = 0;
    }

    LMT_HD void push(double residual)
    {
        if (!ctot_lmt_isfinite(residual) || !(residual >= 0.0)) return;
        if (count < 9) {
            residuals[count++] = residual;
            return;
        }
        for (int i = 0; i < 8; ++i) residuals[i] = residuals[i + 1];
        residuals[8] = residual;
    }

    LMT_HD bool plateau(double production_gate) const
    {
        if (count < 9 || !(production_gate > 0.0) ||
            residuals[8] <= production_gate || !(residuals[0] > 0.0)) {
            return false;
        }
        const double plateau_ratio = residuals[8] / residuals[0];
        if (!ctot_lmt_isfinite(plateau_ratio) ||
            plateau_ratio < 0.98) return false;
        for (int i = 5; i <= 8; ++i) {
            if (!(residuals[i - 1] > 0.0) ||
                residuals[i] < 0.99 * residuals[i - 1]) return false;
        }
        return true;
    }

    LMT_HD double ratio() const
    {
        return count == 9 && residuals[0] > 0.0
            ? residuals[8] / residuals[0] : NAN;
    }
};

struct CtotAdaptiveSpectralScalarV1 {
    double a_ref;
    double D_ref;
    bool used_fallback;
};

// Matrix-free scalar update. The mobility ratio changes only the conditioning
// model and is clipped to the preregistered [0.1,10] baseline interval.
LMT_HD inline CtotAdaptiveSpectralScalarV1
ctot_lmt_adaptive_spectral_scalar(
    double baseline_a_ref, double baseline_D_ref,
    double max_cell_mobility)
{
    CtotAdaptiveSpectralScalarV1 out = {
        baseline_a_ref, baseline_D_ref, true};
    if (!ctot_lmt_isfinite(baseline_a_ref) ||
        !ctot_lmt_isfinite(baseline_D_ref) ||
        !ctot_lmt_isfinite(max_cell_mobility) ||
        !(baseline_a_ref > 0.0) ||
        !(baseline_D_ref > 0.0) || !(max_cell_mobility > 0.0)) return out;
    const double mobility_ratio = max_cell_mobility / baseline_D_ref;
    if (!ctot_lmt_isfinite(mobility_ratio) ||
        !(mobility_ratio > 0.0)) return out;
    const double scale = ctot_lmt_clamp(sqrt(mobility_ratio), 0.1, 10.0);
    out.a_ref = ctot_lmt_clamp(
        baseline_a_ref / scale, 0.1 * baseline_a_ref, 10.0 * baseline_a_ref);
    out.D_ref = ctot_lmt_clamp(
        baseline_D_ref * scale, 0.1 * baseline_D_ref, 10.0 * baseline_D_ref);
    out.used_fallback = false;
    return out;
}

#undef LMT_HD

#endif
