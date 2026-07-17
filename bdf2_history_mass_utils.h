#ifndef BDF2_HISTORY_MASS_UTILS_H
#define BDF2_HISTORY_MASS_UTILS_H

#include <cfloat>
#include <cmath>
#include <limits>

// These helpers cover algebraic identities assembled from already accepted
// conserved states. They do not replace or relax the source-free accepted-step
// mass and divergence gates.
inline double ctot_bdf2_reduction_equivalence_tolerance_v1(
    double mass_scale) {
    if (!std::isfinite(mass_scale)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double magnitude = std::fabs(mass_scale);
    const double ulp = std::nextafter(
        magnitude, std::numeric_limits<double>::infinity()) - magnitude;
    return std::fmax(1.0e-10, 64.0 * ulp);
}

inline bool ctot_bdf2_reduction_equivalent_v1(
    double residual, double mass_scale) {
    const double tolerance =
        ctot_bdf2_reduction_equivalence_tolerance_v1(mass_scale);
    return std::isfinite(residual) && std::isfinite(tolerance) &&
           std::fabs(residual) <= tolerance;
}

inline double ctot_bdf2_history_mass_tolerance_v1(double mass_scale) {
    return ctot_bdf2_reduction_equivalence_tolerance_v1(mass_scale);
}

inline bool ctot_bdf2_history_mass_consistent_v1(
    double mass_delta, double mass_scale) {
    return ctot_bdf2_reduction_equivalent_v1(mass_delta, mass_scale);
}

#endif
