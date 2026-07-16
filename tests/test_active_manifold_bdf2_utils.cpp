#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "active_manifold_bdf2_utils.h"

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

void check_free_second_order() {
    const auto value = active_manifold_bdf2_context_v1(
        0.21, 0.20, 0.30, 0.29, 1.0, 1.0e-12);
    require(value.valid, "free context must be valid");
    require(value.branch == ACTIVE_MANIFOLD_BDF2_FREE,
            "free context must retain the free branch");
    require(value.phi_context == 0.31,
            "free context must preserve the exact second-order extrapolate");
}

void check_free_to_lower() {
    const auto value = active_manifold_bdf2_context_v1(
        9.999994891683184e-1, 9.999994891229144e-1,
        9.962826389863121e-1, 9.962825517864125e-1,
        1.0, 1.0e-12);
    require(value.valid, "free-to-lower event must be representable");
    require(value.branch == ACTIVE_MANIFOLD_BDF2_QALPHA_LOWER,
            "free-to-lower event must select q-alpha lower manifold");
    require(value.q_raw < 0.0, "manufactured raw context must point outward");
    require(value.q_context >= -64.0 * DBL_EPSILON,
            "manifold context must remain lower-feasible");
    require(std::fabs(value.h_context - value.C_anchor) <=
                64.0 * DBL_EPSILON,
            "q=0 manifold storage identity must close");
}

void check_persistent_lower() {
    const double phi_n = 0.99856276658359988;
    const double alpha_n = phase_kkt_alpha(phi_n);
    const double C_n = 1.0 - alpha_n;
    const auto value = active_manifold_bdf2_context_v1(
        C_n, C_n, phi_n, phi_n - 2.0e-8, 1.0, 1.0e-12);
    require(value.valid, "persistent lower event must be valid");
    require(value.branch == ACTIVE_MANIFOLD_BDF2_QALPHA_LOWER,
            "persistent lower event must stay on lower manifold");
}

void check_stationary_lower() {
    const double phi = 0.83;
    const double C = phase_kkt_h_stable(phi);
    const auto value = active_manifold_bdf2_context_v1(
        C, C, phi, phi, 1.0, 1.0e-12);
    require(value.valid, "stationary q=0 context must be valid");
    require(value.phi_context == phi,
            "stationary q=0 context must not create motion");
}

void check_leaving_lower() {
    const auto value = active_manifold_bdf2_context_v1(
        0.51, 0.50, 0.50, 0.51, 1.0, 1.0e-12);
    require(value.valid, "lower-to-free context must be valid");
    require(value.branch == ACTIVE_MANIFOLD_BDF2_FREE,
            "inward lower-to-free motion must not be suppressed");
}

void check_pure_beta() {
    const auto value = active_manifold_bdf2_context_v1(
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0e-12);
    require(value.valid, "pure beta endpoint must be valid");
    require(value.phi_context == 1.0, "pure beta endpoint must remain exact");
}

void check_pure_alpha_outward_context() {
    const auto value = active_manifold_bdf2_context_v1(
        0.05, 0.05, 6.0e-12, 2.5e-10, 1.0, 1.0e-12);
    require(value.valid, "pure-alpha outward context must be valid");
    require(value.branch == ACTIVE_MANIFOLD_BDF2_PHI_LOWER,
            "pure-alpha outward motion must select the phi lower manifold");
    require(value.phi_raw < 0.0 && value.phi_context == 0.0,
            "pure-alpha endpoint context must be exact");
    require(value.q_context == value.C_anchor,
            "pure-alpha endpoint must preserve the C=q storage identity");
}

void check_pure_alpha_leaving_endpoint() {
    const auto value = active_manifold_bdf2_context_v1(
        0.05, 0.05, 2.0e-7, 0.0, 1.0, 1.0e-12);
    require(value.valid, "pure-alpha inward context must be valid");
    require(value.branch == ACTIVE_MANIFOLD_BDF2_FREE,
            "inward motion from the pure-alpha endpoint must remain free");
    require(value.phi_context == 4.0e-7,
            "inward pure-alpha motion must preserve second-order context");
}

void check_interior_negative_extrapolate_fails_closed() {
    const auto value = active_manifold_bdf2_context_v1(
        0.20, 0.20, 1.0e-3, 3.0e-3, 1.0, 1.0e-12);
    require(!value.valid,
            "material interior phase overshoot must not masquerade as endpoint motion");
}

void check_upper_capacity() {
    const auto value = active_manifold_bdf2_context_v1(
        1.0, 1.0, 0.4, 0.4, 1.0, 1.0e-12);
    require(value.valid, "upper-capacity context must remain valid");
    require(std::fabs(value.upper_margin_context) <= 64.0 * DBL_EPSILON,
            "upper-capacity context must retain its branch");
}

void check_neighbor_face_context() {
    const auto active = active_manifold_bdf2_context_v1(
        0.9999995, 0.9999994, 0.996, 0.9959, 1.0, 1.0e-12);
    const auto free = active_manifold_bdf2_context_v1(
        0.95, 0.949, 0.91, 0.909, 1.0, 1.0e-12);
    require(active.valid && free.valid,
            "neighboring active/free contexts must both be finite");
    const double x_active = active.alpha_context > 0.0
        ? fmax(active.q_context, 0.0) / active.alpha_context : 0.0;
    const double x_free = free.alpha_context > 0.0
        ? fmax(free.q_context, 0.0) / free.alpha_context : 0.0;
    const double mobility_active =
        active.alpha_context * x_active * (1.0 - x_active);
    const double mobility_free =
        free.alpha_context * x_free * (1.0 - x_free);
    require(std::isfinite(mobility_active) && std::isfinite(mobility_free),
            "neighboring face mobility factors must be finite");
}

void check_ulp_only_excursion() {
    const double phi = 0.7;
    const double h = phase_kkt_h_stable(phi);
    const double C = std::nextafter(h, 0.0);
    const auto value = active_manifold_bdf2_context_v1(
        C, C, phi, phi, 1.0, 1.0e-12);
    require(value.valid, "ULP-only lower excursion must remain evaluable");
    require(value.branch == ACTIVE_MANIFOLD_BDF2_ULP_ENDPOINT,
            "ULP-only excursion must use the versioned endpoint context");
}

void check_material_upper_failure() {
    const auto value = active_manifold_bdf2_context_v1(
        1.1, 1.1, 0.9, 0.9, 1.0, 1.0e-12);
    require(!value.valid, "material upper-capacity violation must fail closed");
}

}  // namespace

int main() {
    check_free_second_order();
    check_free_to_lower();
    check_persistent_lower();
    check_stationary_lower();
    check_leaving_lower();
    check_pure_beta();
    check_pure_alpha_outward_context();
    check_pure_alpha_leaving_endpoint();
    check_interior_negative_extrapolate_fails_closed();
    check_upper_capacity();
    check_neighbor_face_context();
    check_ulp_only_excursion();
    check_material_upper_failure();
    std::puts("active_manifold_bdf2_utils=PASS");
    return 0;
}
