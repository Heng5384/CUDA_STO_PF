#include "../low_memory_transport_v1_utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static bool near(double a, double b, double tol = 1.0e-14)
{
    return std::fabs(a - b) <= tol * std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
}

int main()
{
    assert(near(ctot_lmt_homotopy_theta(0), 0.25));
    assert(near(ctot_lmt_homotopy_theta(1), 0.5));
    assert(near(ctot_lmt_homotopy_theta(2), 1.0));

    const double h = 0.2;
    const double alpha = 0.8;
    const double C = h + 0.3;
    assert(near(ctot_lmt_fraction_to_boundary_local(
        C, h, alpha, 0.6, 1.0, 1.0e-10), 0.5));
    assert(near(ctot_lmt_fraction_to_boundary_local(
        C, h, alpha, -1.0, 1.0, 1.0e-10), 0.5));
    assert(near(ctot_lmt_fraction_to_boundary_local(
        C, h, alpha, 0.0, 1.0, 1.0e-10), 1.0));
    assert(ctot_lmt_fraction_to_boundary_local(
        h - 1.0e-5, h, alpha, 1.0, 1.0, 1.0e-10) == 0.0);
    const double tiny_alpha = 2.0e-12;
    const double tiny_q = 0.25 * tiny_alpha;
    const double near_closed_C = (1.0 - tiny_alpha) + tiny_q;
    assert(near(ctot_lmt_fraction_to_boundary_local(
        near_closed_C, 1.0 - tiny_alpha, tiny_alpha,
        tiny_q, 1.0, 1.0e-14), 1.0));
    assert(near(ctot_lmt_fraction_to_boundary_initial_lambda(0.5), 0.4975));

    // Mixed free/lower/upper vector with a zero-mass direction. The fraction
    // to boundary must preserve both the global sum and every local bound.
    const std::vector<double> vector_h = {0.2, 0.2, 0.0, 0.0};
    const std::vector<double> vector_alpha = {0.8, 0.8, 1.0, 1.0};
    const std::vector<double> vector_q = {0.01, 0.79, 0.4, 0.6};
    const std::vector<double> vector_d = {0.02, -0.02, 0.1, -0.1};
    double lambda_feasible = 1.0;
    double sum_before = 0.0;
    for (std::size_t i = 0; i < vector_h.size(); ++i) {
        const double value = ctot_lmt_fraction_to_boundary_local(
            vector_h[i] + vector_q[i], vector_h[i], vector_alpha[i],
            vector_d[i], 1.0, 1.0e-10);
        lambda_feasible = std::fmin(lambda_feasible, value);
        sum_before += vector_h[i] + vector_q[i];
    }
    const double lambda_vector =
        ctot_lmt_fraction_to_boundary_initial_lambda(lambda_feasible);
    double sum_after = 0.0;
    for (std::size_t i = 0; i < vector_h.size(); ++i) {
        const double q_after = vector_q[i] - lambda_vector * vector_d[i];
        assert(q_after >= 0.0 && q_after <= vector_alpha[i]);
        sum_after += vector_h[i] + q_after;
    }
    assert(near(sum_before, sum_after));

    assert(ctot_lmt_hybrid_merit_accept(
        1.0, 1000.0, 0.9, 1100.0, 1.0, 1.0));
    assert(!ctot_lmt_hybrid_merit_accept(
        1.0, 50.0, 0.9, 51.0, 1.0, 1.0));
    assert(ctot_lmt_hybrid_merit_accept(
        1.0, 50.0, 0.9, 50.0, 1.0, 1.0));
    assert(!ctot_lmt_hybrid_merit_accept(
        1.0, 5.0, 0.9, 5.0, 1.0, 1.0));
    assert(ctot_lmt_hybrid_merit_accept(
        1.0, 5.0, 2.0, 0.9, 1.0, 1.0));

    CtotTransportPlateauWindowV1 window;
    window.reset();
    for (int i = 0; i < 9; ++i) window.push(1.0 - 0.001 * i);
    assert(window.plateau(1.0e-10));
    window.reset();
    for (int i = 0; i < 9; ++i) window.push(std::pow(0.9, i));
    assert(!window.plateau(1.0e-10));

    const CtotAdaptiveSpectralScalarV1 adaptive =
        ctot_lmt_adaptive_spectral_scalar(0.1, 1.0, 100.0);
    assert(!adaptive.used_fallback);
    assert(near(adaptive.a_ref, 0.01));
    assert(near(adaptive.D_ref, 10.0));
    const CtotAdaptiveSpectralScalarV1 fallback =
        ctot_lmt_adaptive_spectral_scalar(0.1, 1.0, NAN);
    assert(fallback.used_fallback);
    assert(near(fallback.a_ref, 0.1));
    assert(near(fallback.D_ref, 1.0));

    std::puts("PASS_LOW_MEMORY_TRANSPORT_V1_UTILS");
    return 0;
}
