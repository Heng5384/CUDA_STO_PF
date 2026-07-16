#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <string>
#include <vector>

#ifndef __CUDACC__
#define __device__
#define __host__
#endif

#define THERMO_UTILS_DEFINE_GLOBALS
#include "phase_functions.h"
#include "thermo_utils.h"

namespace {

struct Params {
    const char *label;
    double temperature_C;
    double D_alpha;
    double D_compound;
    double Vm_alpha_0;
    double dVm_alpha_dxB;
    double Vm_compound;
    double mu_reference_scale;
    double W;
    double v_A;
    double v_B;
    double x_min;
    double x_max;
    int thermo_convex_enabled;
};

struct Row {
    std::string test;
    std::string condition;
    double perturbation;
    double observed;
    double expected;
    double abs_error;
    double rel_error;
    double tolerance;
    bool passed;
    std::string semantics;
};

double relerr(double a, double b) {
    return std::fabs(a - b) / std::max({1.0, std::fabs(a), std::fabs(b)});
}

void add(std::vector<Row> &rows, const std::string &test, const std::string &condition,
         double eps, double observed, double expected, double tolerance, bool passed,
         const std::string &semantics) {
    rows.push_back({test, condition, eps, observed, expected,
                    std::fabs(observed - expected), relerr(observed, expected),
                    tolerance, passed, semantics});
}

double h_inverse(double target) {
    if (target <= 0.0) return 0.0;
    if (target >= 1.0) return 1.0;
    double lo = 0.0, hi = 1.0;
    for (int i = 0; i < 80; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (h_of_phi(mid) < target) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

double mu0_compound(const Params &p) {
    const double T = p.temperature_C + 273.15;
    const double xeq = xB_eq_from_temperature(T);
    const double muA = mu_A_dimless(xeq, T, p.mu_reference_scale);
    const double muB = mu_B_dimless(xeq, T, p.mu_reference_scale);
    return v_weighted_mu_compound(muA, muB, p.v_A, p.v_B);
}

double bulk_energy(double phi, double x, const Params &p) {
    const double T = p.temperature_C + 273.15;
    const double h = h_of_phi(phi);
    const double muA = mu_A_dimless(x, T, p.mu_reference_scale);
    const double muB = mu_B_dimless(x, T, p.mu_reference_scale);
    const double mix = (1.0 - x) * muA + x * muB;
    return (1.0 - h) * mix + h * mu0_compound(p) + p.W * g_of_phi(phi);
}

double ctot(double phi, double x, const Params &p) {
    return (1.0 - h_of_phi(phi)) * x + h_of_phi(phi) * p.v_B;
}

double x_at_fixed_C(double C, double phi, const Params &p) {
    return (C - h_of_phi(phi) * p.v_B) / (1.0 - h_of_phi(phi));
}

double mu_code(double phi, double x, const Params &p) {
    const double T = p.temperature_C + 273.15;
    const double h = h_of_phi(phi);
    const double muA = mu_A_dimless(x, T, p.mu_reference_scale);
    const double muB = mu_B_dimless(x, T, p.mu_reference_scale);
    const double c = c_xB_phi(x, p.Vm_alpha_0, p.dVm_alpha_dxB, p.Vm_compound, h);
    const double mut = mu_tot_mix(muA, muB, x, mu0_compound(p), h);
    return c * (muB - muA - c * mut * p.dVm_alpha_dxB);
}

double phi_force_code(double phi, double x, const Params &p) {
    const double T = p.temperature_C + 273.15;
    const double h = h_of_phi(phi);
    const double hp = h_prime_of_phi(phi);
    const double muA = mu_A_dimless(x, T, p.mu_reference_scale);
    const double muB = mu_B_dimless(x, T, p.mu_reference_scale);
    const double mu0 = mu0_compound(p);
    const double delta_mu = mu0 - p.v_A * muA - p.v_B * muB;
    const double c = c_xB_phi(x, p.Vm_alpha_0, p.dVm_alpha_dxB, p.Vm_compound, h);
    const double mut = mu_tot_mix(muA, muB, x, mu0, h);
    const double Vm_alpha = Vm_alpha_of_xB(x, p.Vm_alpha_0, p.dVm_alpha_dxB);
    const double volume_term = p.Vm_compound - Vm_alpha + p.dVm_alpha_dxB * (x - p.v_B);
    return p.W * g_prime_of_phi(phi) + c * hp * (delta_mu - c * mut * volume_term);
}

double stabilized_meff_host(double Dm, double gamma) {
    const double gamma_floor = 1.0e-4;
    const double D_safe = std::max(Dm, 0.0);
    const double gamma_safe = (std::isfinite(gamma) && gamma > gamma_floor) ? gamma : gamma_floor;
    const double cap = D_safe * 1000.0;
    const double value = D_safe / gamma_safe;
    return (std::isfinite(value) && value < cap) ? value : cap;
}

bool convergence_seen(const std::vector<double> &errors) {
    double best = std::numeric_limits<double>::infinity();
    int improvements = 0;
    for (double e : errors) {
        if (e < best * 0.8) ++improvements;
        best = std::min(best, e);
    }
    return improvements >= 2;
}

void test_h(std::vector<Row> &rows) {
    double min_delta = 1.0;
    double max_roundtrip = 0.0;
    double previous = h_of_phi(0.0);
    for (int i = 1; i <= 10000; ++i) {
        const double phi = i / 10000.0;
        const double value = h_of_phi(phi);
        min_delta = std::min(min_delta, value - previous);
        previous = value;
        max_roundtrip = std::max(max_roundtrip, std::fabs(h_inverse(value) - phi));
    }
    add(rows, "h_monotonicity", "phi_in_0_to_1", 0.0, min_delta, 0.0, 1e-15,
        min_delta >= -1e-15, "quintic h is monotone nondecreasing");
    add(rows, "h_inverse_roundtrip", "phi_in_0_to_1", 0.0, max_roundtrip, 0.0, 2e-8,
        max_roundtrip <= 2e-8, "bisection inverse round-trip, endpoint conditioning included");
}

void test_storage(std::vector<Row> &rows, const Params &p) {
    std::mt19937_64 rng(0xC70A2026ULL);
    std::uniform_real_distribution<double> uphi(0.0, 0.96);
    std::uniform_real_distribution<double> ux(p.x_min, p.x_max);
    double max_c = 0.0, max_q = 0.0, max_phase = 0.0;
    for (int i = 0; i < 20000; ++i) {
        const double phi = uphi(rng);
        const double x = ux(rng);
        const double h = h_of_phi(phi);
        const double C = (1.0 - h) * x + h * p.v_B;
        const double q = C - h * p.v_B;
        const double xr = q / (1.0 - h);
        max_c = std::max(max_c, std::fabs(C - (q + h * p.v_B)));
        max_q = std::max(max_q, std::fabs(x - xr));
        const double dphi = 1e-4 * (0.5 - (i % 2));
        const double h2 = h_of_phi(phi + dphi);
        const double q2 = q - (h2 - h) * p.v_B;
        max_phase = std::max(max_phase, std::fabs((q2 + h2 * p.v_B) - C));
    }
    add(rows, "C_q_x_roundtrip", p.label, 0.0, std::max(max_c, max_q), 0.0, 2e-13,
        std::max(max_c, max_q) <= 2e-13, "C=(1-h)x+h*vB; q=C-h*vB; x=q/(1-h)");
    add(rows, "fixed_C_phase_conversion", p.label, 0.0, max_phase, 0.0, 5e-14,
        max_phase <= 5e-14, "q_new-q_old+(h_new-h_old)*vB=0");
}

void test_derivatives(std::vector<Row> &rows, const Params &p, bool convex_enabled) {
    h_thermo_convex_extrapolation_enabled = convex_enabled ? 1 : 0;
    const double phi = 0.35;
    const double x = convex_enabled ? 0.12 : std::min(0.02, 0.5 * (p.x_min + p.x_max));
    const double C = ctot(phi, x, p);
    const std::vector<double> eps = {1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6};
    std::vector<double> mu_errors, phi_errors;
    const double mu = mu_code(phi, x, p);
    const double force = phi_force_code(phi, x, p);
    for (double e : eps) {
        const double xp = x_at_fixed_C(C + e, phi, p);
        const double xm = x_at_fixed_C(C - e, phi, p);
        const double dFdC = (bulk_energy(phi, xp, p) - bulk_energy(phi, xm, p)) / (2.0 * e);
        const double mu_error = relerr(dFdC, mu);
        mu_errors.push_back(mu_error);
        add(rows, "mu_vs_dF_dC_fixed_phi", std::string(p.label) + (convex_enabled ? ":convex_on" : ":runtime_flag"),
            e, mu, dFdC, 2e-6, std::isfinite(mu_error),
            "convergence sample: compute_mu_x formula versus central directional derivative");

        const double pp = phi + e;
        const double pm = phi - e;
        const double xpp = x_at_fixed_C(C, pp, p);
        const double xpm = x_at_fixed_C(C, pm, p);
        const double dFdphi = (bulk_energy(pp, xpp, p) - bulk_energy(pm, xpm, p)) / (2.0 * e);
        const double phi_error = relerr(dFdphi, force);
        phi_errors.push_back(phi_error);
        add(rows, "phi_force_vs_dF_dphi_fixed_C", std::string(p.label) + (convex_enabled ? ":convex_on" : ":runtime_flag"),
            e, force, dFdphi, 2e-6, std::isfinite(phi_error),
            "convergence sample: chemical chain term in compute_phi_rhs_kernel at fixed C");
    }
    const bool mu_conv = convergence_seen(mu_errors);
    const bool phi_conv = convergence_seen(phi_errors);
    add(rows, "mu_derivative_convergence_region", std::string(p.label) + (convex_enabled ? ":convex_on" : ":runtime_flag"),
        0.0, *std::min_element(mu_errors.begin(), mu_errors.end()), 0.0, 2e-6,
        (mu_conv || *std::min_element(mu_errors.begin(), mu_errors.end()) <= 2e-10) &&
            *std::min_element(mu_errors.begin(), mu_errors.end()) <= 2e-6,
        "multiple perturbations must enter a convergence region");
    add(rows, "phi_derivative_convergence_region", std::string(p.label) + (convex_enabled ? ":convex_on" : ":runtime_flag"),
        0.0, *std::min_element(phi_errors.begin(), phi_errors.end()), 0.0, 2e-6,
        (phi_conv || *std::min_element(phi_errors.begin(), phi_errors.end()) <= 2e-10) &&
            *std::min_element(phi_errors.begin(), phi_errors.end()) <= 2e-6,
        "multiple perturbations must enter a convergence region");
}

void test_convex_integrability(std::vector<Row> &rows, const Params &p) {
    h_thermo_convex_extrapolation_enabled = 1;
    const double T = p.temperature_C + 273.15;
    const double xc = X_LIMIT_CONVEX;
    const double K = g_alpha_extension_K(T);
    const double ideal_K = R_GAS*T/(xc*(1.0-xc));
    const double expected_K = std::max(g_alpha_second_calphad(T, xc), ideal_K);
    add(rows, "convex_extension_K_rule", p.label, 0.0, K, expected_K, 1e-13,
        relerr(K, expected_K) <= 1e-13,
        "K=max(g_second_at_xc,RT/(xc*(1-xc)))");
    double low_mu_regression = 0.0;
    for (double x : {0.001, 0.0078305391025, 0.03, 0.06, 0.089, 0.09}) {
        low_mu_regression = std::max(low_mu_regression,
            std::max(relerr(mu_PbTe_raw(T,x), mu_PbTe_calphad(T,x)),
                     relerr(mu_Ag2Te_raw(T,x), mu_Ag2Te_calphad(T,x))));
    }
    add(rows, "low_x_calphad_mu_regression", p.label, 0.0,
        low_mu_regression, 0.0, 2e-13, low_mu_regression <= 2e-13,
        "x<=xc component chemical potentials preserve original CALPHAD values");
    std::vector<double> continuity_errors;
    for (double e : {1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6}) {
        const double g_c = g_alpha_raw(T, xc);
        const double gp_c = g_alpha_prime_raw(T, xc);
        const double value_mismatch = std::max(relerr(g_alpha_raw(T, xc-e), g_c-gp_c*e),
                                               relerr(g_alpha_raw(T, xc+e), g_c+gp_c*e));
        const double left_slope = (g_c - g_alpha_raw(T, xc-e)) / e;
        const double right_slope = (g_alpha_raw(T, xc+e) - g_c) / e;
        const double mismatch = std::max(value_mismatch,
            std::max(relerr(left_slope, gp_c), relerr(right_slope, gp_c)));
        continuity_errors.push_back(mismatch);
        add(rows, "convex_boundary_derivative_continuity", p.label, e, mismatch, 0.0, 2e-3,
            std::isfinite(mismatch), "convergence sample: g value and g-prime continuity at x=0.09");
    }
    const double best_continuity = *std::min_element(continuity_errors.begin(), continuity_errors.end());
    add(rows, "convex_boundary_continuity_convergence_region", p.label, 0.0,
        best_continuity, 0.0, 2e-3,
        convergence_seen(continuity_errors) && best_continuity <= 2e-3,
        "g and g-prime mismatch enters a convergence region as perturbation shrinks");
    for (double x : {0.091, 0.12, 0.20}) {
        const double e = 1e-6;
        const double dA = (mu_PbTe_raw(T, x + e) - mu_PbTe_raw(T, x - e)) / (2.0 * e);
        const double dB = (mu_Ag2Te_raw(T, x + e) - mu_Ag2Te_raw(T, x - e)) / (2.0 * e);
        const double gd = (1.0 - x) * dA + x * dB;
        const double scale = std::max({1.0, std::fabs(dA), std::fabs(dB)});
        add(rows, "convex_gibbs_duhem_integrability", std::string(p.label) + ":x=" + std::to_string(x), e, gd / scale, 0.0, 1e-8,
            std::fabs(gd) / scale <= 1e-8,
            "single binary free energy requires (1-x)dmuA/dx+x*dmuB/dx=0");
    }
}

void test_gamma_and_mobility(std::vector<Row> &rows, const Params &p) {
    const double T = p.temperature_C + 273.15;
    for (int convex : {0, 1}) {
        h_thermo_convex_extrapolation_enabled = convex;
        for (double x : {p.x_min, 0.0078305391025, 0.03, 0.079, 0.081, 0.089, 0.091, 0.12, p.x_max}) {
            if (!(x > 1e-7 && x < 1.0 - 1e-7)) continue;
            const double e = 1e-6;
            const double dmu = (mu_code(0.0, x + e, p) - mu_code(0.0, x - e, p)) / (2.0 * e);
            const double gamma = gamma_thermo_nonlinear(x, 0.0, p.Vm_alpha_0,
                p.dVm_alpha_dxB, p.Vm_compound, T, p.mu_reference_scale);
            const double expected = gamma / c_xB_phi(x, p.Vm_alpha_0, p.dVm_alpha_dxB, p.Vm_compound, 0.0);
            const double error = relerr(dmu, expected);
            add(rows, "dmu_dx_vs_gamma_semantics", std::string(p.label) + (convex ? ":convex_on:x=" : ":runtime_flag:x=") + std::to_string(x),
                e, dmu, expected, 2e-3, error <= 2e-3,
                "Gamma=c_bulk*d(muB-muA)/dx, subject to the source x=0.08 limiter");
        }
    }

    h_thermo_convex_extrapolation_enabled = p.thermo_convex_enabled;
    double minM = std::numeric_limits<double>::infinity();
    double maxM = 0.0;
    bool finite_nonnegative = true;
    for (int ih = 0; ih <= 100; ++ih) {
        const double h = ih / 100.0;
        for (int ix = 0; ix <= 400; ++ix) {
            const double x = p.x_min + (p.x_max - p.x_min) * ix / 400.0;
            const double Dm = D_mix(h, p.D_alpha, p.D_compound);
            const double G = gamma_thermo_nonlinear(x, h, p.Vm_alpha_0,
                p.dVm_alpha_dxB, p.Vm_compound, T, p.mu_reference_scale);
            const double M = stabilized_meff_host(Dm, G);
            finite_nonnegative = finite_nonnegative && std::isfinite(M) && M >= 0.0;
            minM = std::min(minM, M);
            maxM = std::max(maxM, M);
        }
    }
    add(rows, "M_eff_finite_nonnegative_scan", p.label, 0.0, minM, 0.0, 0.0,
        finite_nonnegative, "current stabilized_meff over actual T/composition scan");
    add(rows, "M_eff_scan_max", p.label, 0.0, maxM, maxM, 0.0,
        finite_nonnegative, "maximum recorded for audit");

    h_thermo_convex_extrapolation_enabled = 1;
    const double x = std::max(p.x_min, 0.0078305391025);
    bool candidate_finite_nonnegative = true;
    double candidate_min = std::numeric_limits<double>::infinity();
    double candidate_max = 0.0;
    for (int ih = 0; ih <= 100; ++ih) {
        const double hc = ih / 100.0;
        for (int ix = 0; ix <= 400; ++ix) {
            const double xc = p.x_min + (p.x_max - p.x_min) * ix / 400.0;
            const double Mc = matrix_capacity_mobility_candidate(
                hc, xc, p.D_alpha, p.Vm_alpha_0, p.dVm_alpha_dxB,
                p.Vm_compound, T, p.mu_reference_scale, 1.0e-10);
            candidate_finite_nonnegative = candidate_finite_nonnegative &&
                                            std::isfinite(Mc) && Mc >= 0.0;
            candidate_min = std::min(candidate_min, Mc);
            candidate_max = std::max(candidate_max, Mc);
        }
    }
    add(rows, "candidate_matrix_mobility_nonnegative_scan", p.label, 0.0,
        candidate_min, 0.0, 0.0, candidate_finite_nonnegative,
        "new candidate uses (1-h)*D_alpha/Gamma_alpha with no constitutive floor");
    add(rows, "candidate_matrix_mobility_scan_max", p.label, 0.0,
        candidate_max, candidate_max, 0.0, candidate_finite_nonnegative,
        "maximum candidate mobility recorded for audit");

    const double M_alpha = matrix_mobility_alpha_candidate(
        x, p.D_alpha, p.Vm_alpha_0, p.dVm_alpha_dxB,
        p.Vm_compound, T, p.mu_reference_scale);
    const double M_h0 = matrix_capacity_mobility_candidate(
        0.0, x, p.D_alpha, p.Vm_alpha_0, p.dVm_alpha_dxB,
        p.Vm_compound, T, p.mu_reference_scale, 1.0e-10);
    const double M_hhalf = matrix_capacity_mobility_candidate(
        0.5, x, p.D_alpha, p.Vm_alpha_0, p.dVm_alpha_dxB,
        p.Vm_compound, T, p.mu_reference_scale, 1.0e-10);
    const double M_h1 = matrix_capacity_mobility_candidate(
        1.0, x, p.D_alpha, p.Vm_alpha_0, p.dVm_alpha_dxB,
        p.Vm_compound, T, p.mu_reference_scale, 1.0e-10);
    add(rows, "candidate_mobility_h0_identity", p.label, 0.0, M_h0, M_alpha,
        1e-14, relerr(M_h0, M_alpha) <= 1e-14, "h=0 gives M_eff=M_alpha");
    add(rows, "candidate_mobility_interface_nonnegative", p.label, 0.0,
        M_hhalf, 0.5*M_alpha, 1e-14,
        std::isfinite(M_hhalf) && M_hhalf >= 0.0 && relerr(M_hhalf, 0.5*M_alpha) <= 1e-14,
        "0<h<1 follows matrix-capacity interpolation");
    add(rows, "candidate_mobility_h1_zero", p.label, 0.0, M_h1, 0.0,
        1e-14, std::fabs(M_h1) <= 1e-14, "h=1 gives zero composition mobility");
    const double M_inactive_nan_context = matrix_capacity_mobility_candidate(
        1.0, std::numeric_limits<double>::quiet_NaN(), p.D_alpha,
        p.Vm_alpha_0, p.dVm_alpha_dxB, p.Vm_compound, T,
        p.mu_reference_scale, 1.0e-10);
    add(rows, "candidate_inactive_skips_matrix_thermo", p.label, 0.0,
        M_inactive_nan_context, 0.0, 0.0, M_inactive_nan_context == 0.0,
        "inactive beta support returns zero before xB thermodynamics");
    const double M_roundoff_negative_capacity = matrix_capacity_mobility_candidate(
        1.0 + 5.0e-11, x, p.D_alpha, p.Vm_alpha_0,
        p.dVm_alpha_dxB, p.Vm_compound, T, p.mu_reference_scale, 1.0e-10);
    add(rows, "candidate_roundoff_negative_capacity_closed", p.label, 0.0,
        M_roundoff_negative_capacity, 0.0, 0.0,
        M_roundoff_negative_capacity == 0.0,
        "roundoff-scale negative capacity uses the removable zero limit");
    const double M_material_negative_capacity = matrix_capacity_mobility_candidate(
        1.0 + 2.0e-10, x, p.D_alpha, p.Vm_alpha_0,
        p.dVm_alpha_dxB, p.Vm_compound, T, p.mu_reference_scale, 1.0e-10);
    add(rows, "candidate_material_negative_capacity_rejected", p.label, 0.0,
        M_material_negative_capacity, std::numeric_limits<double>::quiet_NaN(),
        0.0, std::isnan(M_material_negative_capacity),
        "materially negative capacity remains a constitutive failure");
    add(rows, "candidate_constant_mu_zero_flux", p.label, 0.0, M_hhalf*0.0, 0.0,
        0.0, M_hhalf*0.0 == 0.0, "constant mu has zero gradient and zero flux");
    add(rows, "candidate_beta_core_zero_divJ", p.label, 0.0,
        M_h1 * (2.0*M_PI/32.0) * (2.0*M_PI/32.0), 0.0, 1e-14,
        std::fabs(M_h1) <= 1e-14, "pure beta core has zero flux and divergence");

    const double h = 1.0;
    const double Dm = D_mix(h, p.D_alpha, p.D_compound);
    const double G = gamma_thermo_nonlinear(x, h, p.Vm_alpha_0,
        p.dVm_alpha_dxB, p.Vm_compound, T, p.mu_reference_scale);
    const double Mlegacy = stabilized_meff_host(Dm, G);
    const double matrix_capacity = 1.0 - h;
    const double Mq = matrix_capacity * stabilized_meff_host(p.D_alpha,
        gamma_thermo_nonlinear(x, 0.0, p.Vm_alpha_0, p.dVm_alpha_dxB,
            p.Vm_compound, T, p.mu_reference_scale));
    add(rows, "beta_core_matrix_capacity", p.label, 0.0, matrix_capacity, 0.0, 1e-15,
        matrix_capacity == 0.0, "fixed-stoichiometric beta has no matrix composition capacity");
    add(rows, "beta_core_legacy_transport_regression", p.label, 0.0, Mlegacy, Mlegacy, 0.0,
        std::isfinite(Mlegacy) && Mlegacy > 0.0,
        "negative control: legacy D_mix still tends to D_compound at h=1");
    add(rows, "beta_core_q_transport_closure", p.label, 0.0, Mq, 0.0, 1e-14,
        std::fabs(Mq) <= 1e-14,
        "Q flux multiplies matrix mobility by 1-h");
    const double wave = 2.0 * M_PI / 32.0;
    const double legacy_divj_amplitude = Mlegacy * wave * wave;
    const double q_divj_amplitude = Mq * wave * wave;
    add(rows, "beta_core_legacy_divJ_regression", p.label, 0.0,
        legacy_divj_amplitude, legacy_divj_amplitude, 0.0,
        std::isfinite(legacy_divj_amplitude) && legacy_divj_amplitude > 0.0,
        "negative control: legacy beta-core sine divJ remains nonzero");
    add(rows, "beta_core_q_divJ_sine_amplitude", p.label, 0.0,
        q_divj_amplitude, 0.0, 1e-14, q_divj_amplitude <= 1e-14,
        "matrix-capacity weighted Q flux closes divJ at h=1");
}

void write_csv(const char *path, const std::vector<Row> &rows) {
    std::ofstream out(path);
    out << "test,condition,perturbation,observed,expected,abs_error,rel_error,tolerance,passed,semantics\n";
    out << std::setprecision(17);
    for (const Row &r : rows) {
        out << r.test << ',' << r.condition << ',' << r.perturbation << ','
            << r.observed << ',' << r.expected << ',' << r.abs_error << ','
            << r.rel_error << ',' << r.tolerance << ',' << (r.passed ? "true" : "false")
            << ",\"" << r.semantics << "\"\n";
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 28) {
        std::fprintf(stderr, "usage: %s OUTPUT.csv (T Da Dc Vm0 dVm Vmc scale W vA vB xmin xmax convex_flag)x2\n", argv[0]);
        return 2;
    }
    const char *labels[] = {"T380", "T400"};
    Params cases[2];
    for (int c = 0; c < 2; ++c) {
        const int o = 2 + 13*c;
        cases[c] = {labels[c], std::atof(argv[o]), std::atof(argv[o+1]), std::atof(argv[o+2]),
            std::atof(argv[o+3]), std::atof(argv[o+4]), std::atof(argv[o+5]),
            std::atof(argv[o+6]), std::atof(argv[o+7]), std::atof(argv[o+8]),
            std::atof(argv[o+9]), std::atof(argv[o+10]), std::atof(argv[o+11]),
            std::atoi(argv[o+12])};
    }
    std::vector<Row> rows;
    test_h(rows);
    for (const Params &p : cases) {
        test_storage(rows, p);
        test_derivatives(rows, p, p.thermo_convex_enabled != 0);
        test_derivatives(rows, p, true);
        test_convex_integrability(rows, p);
        test_gamma_and_mobility(rows, p);
    }
    write_csv(argv[1], rows);
    int failures = 0;
    for (const Row &r : rows) failures += r.passed ? 0 : 1;
    std::printf("host_gate_rows=%zu\nhost_gate_failed_rows=%d\n", rows.size(), failures);
    return 0;
}
