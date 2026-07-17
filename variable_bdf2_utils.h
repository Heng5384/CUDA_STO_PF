#ifndef VARIABLE_BDF2_UTILS_H
#define VARIABLE_BDF2_UTILS_H

#include <cmath>

#ifdef __CUDACC__
#define VARIABLE_BDF2_HD __host__ __device__
#else
#define VARIABLE_BDF2_HD
#endif

struct VariableBdf2CoefficientsV1 {
    double ratio;
    double a0;
    double a1;
    double a2;
    double anchor_n;
    double anchor_nm1;
    double extrap_n;
    double extrap_nm1;
    double effective_dt_factor;
    int valid;
};

VARIABLE_BDF2_HD inline VariableBdf2CoefficientsV1
variable_bdf2_coefficients_v1(double dt, double dt_previous) {
    VariableBdf2CoefficientsV1 out{};
    out.valid = 0;
    if (!(dt > 0.0) || !(dt_previous > 0.0) ||
        !isfinite(dt) || !isfinite(dt_previous)) {
        return out;
    }
    const double r = dt / dt_previous;
    if (!isfinite(r) || r < 0.25 || r > 2.0) return out;
    const double denominator = 1.0 + r;
    out.ratio = r;
    out.a0 = (1.0 + 2.0 * r) / denominator;
    out.a1 = -(1.0 + r);
    out.a2 = r * r / denominator;
    out.anchor_n = -out.a1 / out.a0;
    out.anchor_nm1 = -out.a2 / out.a0;
    out.extrap_n = 1.0 + r;
    out.extrap_nm1 = -r;
    out.effective_dt_factor = 1.0 / out.a0;
    out.valid = isfinite(out.a0) && out.a0 > 0.0 &&
        isfinite(out.anchor_n) && isfinite(out.anchor_nm1) &&
        isfinite(out.effective_dt_factor);
    return out;
}

struct VariableBdf2ControllerDecisionV1 {
    double error_ratio;
    double raw_factor;
    double limited_factor;
    double next_dt;
    int accept;
    int valid;
};

VARIABLE_BDF2_HD inline VariableBdf2ControllerDecisionV1
variable_bdf2_pi_decision_v1(
    double dt, double error_ratio, double previous_error_ratio,
    double dt_min, double dt_max, double safety,
    double proportional_exponent, double integral_exponent) {
    VariableBdf2ControllerDecisionV1 out{};
    out.error_ratio = error_ratio;
    out.valid = 0;
    if (!(dt > 0.0) || !(dt_min > 0.0) || !(dt_max >= dt_min) ||
        !(safety > 0.0) || !isfinite(dt) || !isfinite(error_ratio) ||
        !isfinite(previous_error_ratio) || !isfinite(safety) ||
        !isfinite(proportional_exponent) || !isfinite(integral_exponent)) {
        return out;
    }
    const double current = fmax(error_ratio, 1.0e-12);
    const double previous = fmax(previous_error_ratio, 1.0e-12);
    out.raw_factor = safety * pow(current, -proportional_exponent) *
                     pow(previous, integral_exponent);
    if (!isfinite(out.raw_factor)) return out;
    // Normal accepted growth is deliberately capped at 1.25. Rejected trials
    // may shrink by as much as 4x, matching the registered ratio contract.
    const double upper = error_ratio <= 1.0 ? 1.25 : 1.0;
    out.limited_factor = fmin(fmax(out.raw_factor, 0.25), upper);
    out.next_dt = fmin(fmax(dt * out.limited_factor, dt_min), dt_max);
    out.accept = error_ratio <= 1.0;
    out.valid = isfinite(out.next_dt) && out.next_dt > 0.0;
    return out;
}

#undef VARIABLE_BDF2_HD

#endif
