#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#ifndef __CUDACC__
#define __device__
#define __host__
#endif

#define THERMO_UTILS_DEFINE_GLOBALS
#include "thermo_utils.h"

namespace {

int failures = 0;
FILE *output = nullptr;

void check(const char *name, double observed, double expected,
           double tolerance, bool passed, const char *semantics) {
    std::fprintf(output, "%s,%.17e,%.17e,%.17e,%d,%s\n",
                 name, observed, expected, tolerance, passed ? 1 : 0,
                 semantics);
    if (!passed) ++failures;
}

double mobility(double h, double x, double a_M) {
    return matrix_capacity_mobility_coarse_candidate(
        h, x, D_Ag_in_PbTe_m2_per_s(673.15), 1.0, 0.0, 1.0,
        673.15, 1.0e5, 1.0e-10, a_M);
}

double harmonic(double left, double right) {
    if (left == 0.0 || right == 0.0) return 0.0;
    return 2.0 * left * right / (left + right);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    output = std::fopen(argv[1], "w");
    if (!output) return 2;
    std::fprintf(output,
                 "test,observed,expected,tolerance,passed,semantics\n");

    h_thermo_convex_extrapolation_enabled = 1;
    const double x = 0.02;
    const double a_M = 2.5;
    for (int ih = 0; ih <= 100; ++ih) {
        const double h = ih / 100.0;
        const double base = matrix_capacity_mobility_candidate(
            h, x, D_Ag_in_PbTe_m2_per_s(673.15), 1.0, 0.0, 1.0,
            673.15, 1.0e5, 1.0e-10);
        const double disabled = mobility(h, x, 0.0);
        check("aM_zero_exact_legacy", disabled, base, 0.0,
              disabled == base,
              "a_M=0 returns the legacy expression without extra arithmetic");
    }

    const double matrix_base = mobility(0.0, x, 0.0);
    const double matrix_boost = mobility(0.0, x, a_M);
    check("matrix_bulk_unchanged", matrix_boost, matrix_base, 0.0,
          matrix_boost == matrix_base, "b(0)=0");
    const double beta = mobility(1.0, x, a_M);
    check("pure_beta_zero", beta, 0.0, 0.0, beta == 0.0,
          "one-sided beta mobility remains exactly zero");

    const double middle_base = mobility(0.5, x, 0.0);
    const double middle_boost = mobility(0.5, x, a_M);
    const double middle_expected = middle_base * (1.0 + a_M);
    check("mid_interface_factor", middle_boost, middle_expected,
          1.0e-15 * std::max(1.0, std::fabs(middle_expected)),
          std::fabs(middle_boost - middle_expected) <=
              1.0e-15 * std::max(1.0, std::fabs(middle_expected)),
          "b(1/2)=1");

    bool nonnegative = true;
    double minimum = std::numeric_limits<double>::infinity();
    for (int ih = 0; ih <= 100; ++ih) {
        const double value = mobility(ih / 100.0, x, a_M);
        nonnegative = nonnegative && std::isfinite(value) && value >= 0.0;
        minimum = std::min(minimum, value);
    }
    check("cell_mobility_nonnegative", minimum, 0.0, 0.0, nonnegative,
          "positive interface factor preserves constitutive nonnegativity");

    const double left = mobility(0.2, x, a_M);
    const double right = mobility(0.7, x, a_M);
    const double face = harmonic(left, right);
    const double gradient_mu = -0.37;
    const double dissipation = face * gradient_mu * gradient_mu;
    check("harmonic_face_nonnegative", face, 0.0, 0.0,
          std::isfinite(face) && face >= 0.0,
          "symmetric harmonic shared face remains nonnegative");
    check("face_dissipation_nonnegative", dissipation, 0.0, 0.0,
          std::isfinite(dissipation) && dissipation >= 0.0,
          "-J dot grad(mu)=M_face*|grad(mu)|^2");

    const double bad = mobility(0.5, x, -1.0e-3);
    check("negative_parameter_rejected", bad,
          std::numeric_limits<double>::quiet_NaN(), 0.0, std::isnan(bad),
          "a_M is constrained to be nonnegative");

    std::fclose(output);
    return failures == 0 ? 0 : 1;
}
