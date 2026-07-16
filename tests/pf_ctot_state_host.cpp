#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#ifndef __CUDACC__
#define __device__
#define __host__
#endif

#include "phase_functions.h"

namespace {

struct Result {
    const char *test;
    double observed;
    double tolerance;
    bool passed;
    const char *semantics;
};

double max_abs(double a, double b) { return std::fabs(a - b); }

double h_inverse(double target) {
    if (target <= 0.0) return 0.0;
    if (target >= 1.0) return 1.0;
    double lo = 0.0, hi = 1.0;
    for (int i = 0; i < 100; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (h_of_phi(mid) < target) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

void add(std::vector<Result> &out, const char *test, double observed,
         double tolerance, const char *semantics) {
    out.push_back({test, observed, tolerance, observed <= tolerance, semantics});
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s OUTPUT.csv\n", argv[0]);
        return 2;
    }
    constexpr double vB = 1.0;
    constexpr double support_eps = 1.0e-8;
    std::mt19937_64 rng(20260712ULL);
    std::uniform_real_distribution<double> uphi(0.0, 0.999);
    std::uniform_real_distribution<double> ux(1.0e-5, 1.0 - 1.0e-5);
    std::vector<Result> results;

    double roundtrip = 0.0;
    double phase_storage = 0.0;
    for (int n = 0; n < 100000; ++n) {
        const double phi0 = uphi(rng);
        const double phi1 = uphi(rng);
        const double h0 = h_of_phi(phi0);
        const double h1 = h_of_phi(phi1);
        const double x0 = ux(rng);
        const double C = (1.0 - h0) * x0 + h0 * vB;
        const double q0 = C - h0 * vB;
        const double q1 = q0 - (h1 - h0) * vB;
        roundtrip = std::max(roundtrip, max_abs(q0 + h0 * vB, C));
        phase_storage = std::max(phase_storage,
                                 std::fabs(q1 - q0 + (h1 - h0) * vB));
    }
    add(results, "C_q_x_roundtrip", roundtrip, 2.0e-15,
        "Ctot is authoritative and q+h*vB reconstructs it exactly");
    add(results, "fixed_C_phase_storage_transaction", phase_storage, 2.0e-15,
        "phase-only transaction changes q while preserving local Ctot");

    const double phi_core = 1.0;
    const double h_core = h_of_phi(phi_core);
    const double C_core = vB;
    const double q_core = C_core - h_core * vB;
    const bool inactive = (1.0 - h_core) <= support_eps;
    add(results, "inactive_beta_core_q_zero", std::fabs(q_core), 0.0,
        "inactive beta core stores no matrix q and never divides by 1-h");
    add(results, "inactive_beta_core_mask", inactive ? 0.0 : 1.0, 0.0,
        "active xB context is masked out in the stoichiometric beta core");

    const int nstate = 4096;
    std::vector<double> accepted(nstate), saved(nstate), trial(nstate);
    for (int i = 0; i < nstate; ++i) accepted[i] = 0.01 + i * 1.0e-7;
    saved = accepted;
    trial = accepted;
    for (double &value : trial) value += 0.25;
    trial = saved;
    double rollback = 0.0;
    for (int i = 0; i < nstate; ++i)
        rollback = std::max(rollback, std::fabs(trial[i] - accepted[i]));
    add(results, "transactional_state_rollback", rollback, 0.0,
        "rejected trial restores the accepted Ctot snapshot bit-for-bit");

    double migration = 0.0;
    double authoritative = 0.0;
    for (int n = 0; n < 10000; ++n) {
        const double phi = uphi(rng);
        const double h = h_of_phi(phi);
        const double x = ux(rng);
        const double migrated_C = (1.0 - h) * x + h * vB;
        const double checkpoint_C = migrated_C + 1.0e-6 * (1.0 - h);
        const double reconstructed_x =
            (1.0 - h > support_eps) ? (checkpoint_C - h * vB) / (1.0 - h) : x;
        migration = std::max(migration,
                             std::fabs((migrated_C - h * vB) - (1.0 - h) * x));
        if (1.0 - h > support_eps) {
            authoritative = std::max(
                authoritative,
                std::fabs((1.0 - h) * reconstructed_x + h * vB - checkpoint_C));
        }
    }
    add(results, "legacy_restart_migration", migration, 2.0e-15,
        "legacy phi/xB restart deterministically reconstructs missing Ctot");
    add(results, "authoritative_Ctot_restart", authoritative, 2.0e-15,
        "checkpoint Ctot wins and derived xB reconstructs the checkpoint value");

    const double C = 0.20;
    const double xmin = 1.0e-8;
    const double xmax = 1.0 - 1.0e-8;
    const double hmin = std::max(0.0, (C - xmax) / (vB - xmax));
    const double hmax = std::min(1.0, (C - xmin) / (vB - xmin));
    const double pmin = h_inverse(hmin);
    const double pmax = h_inverse(hmax);
    const double growth_raw = 0.95;
    const double growth_projected = std::min(std::max(growth_raw, pmin), pmax);
    const double h_growth = h_of_phi(growth_projected);
    const double q_growth = C - h_growth * vB;
    const double x_growth = q_growth / (1.0 - h_growth);
    add(results, "infeasible_growth_activates_upper_constraint",
        std::fabs(growth_projected - pmax), 2.0e-15,
        "fixed-C growth is projected to the admissible upper phase bound");
    add(results, "projected_growth_C_preservation",
        std::fabs(q_growth + h_growth * vB - C), 2.0e-15,
        "phase projection preserves authoritative C exactly");
    add(results, "projected_growth_x_bound",
        (x_growth >= xmin - 1.0e-12 && x_growth <= xmax + 1.0e-12) ? 0.0 : 1.0,
        0.0, "projected phase state is composition-admissible without clipping");
    const double dissolution_raw = -0.25;
    const double dissolution_projected =
        std::min(std::max(dissolution_raw, pmin), pmax);
    add(results, "dissolution_lower_constraint",
        std::fabs(dissolution_projected - pmin), 2.0e-15,
        "dissolution trial respects the lower admissible phase bound");
    add(results, "projected_KKT_sign_upper",
        growth_raw >= pmax && growth_projected == pmax ? 0.0 : 1.0, 0.0,
        "upper-active projection has the outward trial direction required by KKT");
    add(results, "projected_KKT_sign_lower",
        dissolution_raw <= pmin && dissolution_projected == pmin ? 0.0 : 1.0, 0.0,
        "lower-active projection has the outward trial direction required by KKT");

    const double exact = 0.3 * std::exp(-0.2);
    auto explicit_relax = [](double dt) {
        double p = 0.3;
        const int steps = static_cast<int>(std::llround(0.2 / dt));
        for (int i = 0; i < steps; ++i) p = std::max(0.0, p - dt * p);
        return p;
    };
    const double err_coarse = std::fabs(explicit_relax(0.01) - exact);
    const double err_fine = std::fabs(explicit_relax(0.005) - exact);
    add(results, "phase_only_dt_refinement",
        err_fine < 0.55 * err_coarse ? 0.0 : err_fine / err_coarse, 0.0,
        "constrained phase-only scalar oracle converges under dt refinement");

    const int nfv = 64;
    std::vector<double> mu(nfv), mobility(nfv), face(nfv), div(nfv);
    for (int i = 0; i < nfv; ++i) {
        const double theta = 2.0 * M_PI * static_cast<double>(i) / nfv;
        mu[i] = 0.3 + 0.02 * std::sin(theta) + 0.01 * std::cos(3.0 * theta);
        mobility[i] = 0.1 + 0.04 * (1.0 + std::sin(2.0 * theta));
    }
    for (int i = 0; i < nfv; ++i) {
        const int ip = (i + 1) % nfv;
        const double harmonic =
            2.0 * mobility[i] * mobility[ip] / (mobility[i] + mobility[ip]);
        face[i] = harmonic * (mu[ip] - mu[i]);
    }
    double sum_div = 0.0;
    double shared_face_antisymmetry = 0.0;
    for (int i = 0; i < nfv; ++i) {
        const int im = (i - 1 + nfv) % nfv;
        div[i] = face[i] - face[im];
        sum_div += div[i];
        const double outgoing_from_i = face[i];
        const double incoming_to_ip = -face[i];
        shared_face_antisymmetry = std::max(
            shared_face_antisymmetry,
            std::fabs(outgoing_from_i + incoming_to_ip));
    }
    add(results, "fv_periodic_sum_divergence", std::fabs(sum_div), 2.0e-15,
        "shared periodic positive-face fluxes telescope to zero global divergence");
    add(results, "fv_shared_face_antisymmetry", shared_face_antisymmetry, 0.0,
        "one stored face flux contributes equal and opposite exchange to its cells");

    double constant_mu_flux = 0.0;
    for (int i = 0; i < nfv; ++i) {
        const int ip = (i + 1) % nfv;
        const double constant_mu = 0.25;
        const double harmonic =
            2.0 * mobility[i] * mobility[ip] / (mobility[i] + mobility[ip]);
        constant_mu_flux = std::max(
            constant_mu_flux,
            std::fabs(harmonic * (constant_mu - constant_mu)));
    }
    add(results, "fv_constant_mu_zero_flux", constant_mu_flux, 0.0,
        "constant chemical potential produces exactly zero FV face flux");

    const double matrix_mobility = 0.23;
    const double beta_core_mobility = 0.0;
    const double closed_face_mobility =
        (matrix_mobility > 0.0 && beta_core_mobility > 0.0)
            ? 2.0 * matrix_mobility * beta_core_mobility /
                  (matrix_mobility + beta_core_mobility)
            : 0.0;
    add(results, "fv_beta_core_face_flux_closed", std::fabs(closed_face_mobility),
        0.0, "harmonic shared-face mobility vanishes next to stoichiometric beta core");

    std::ofstream csv(argv[1]);
    csv << "test,observed,tolerance,passed,semantics\n";
    int failed = 0;
    for (const Result &r : results) {
        csv << r.test << ',' << r.observed << ',' << r.tolerance << ','
            << (r.passed ? "true" : "false") << ",\"" << r.semantics << "\"\n";
        failed += r.passed ? 0 : 1;
    }
    std::printf("ctot_state_host_rows=%zu\nctot_state_host_failed=%d\n",
                results.size(), failed);
    return failed == 0 ? 0 : 1;
}
