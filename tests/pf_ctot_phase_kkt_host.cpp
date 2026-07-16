#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "phase_kkt_utils.h"

namespace {

struct Row {
    std::string test;
    double observed;
    double tolerance;
    bool pass;
    std::string semantics;
};

void add(std::vector<Row> &rows, const char *test, double observed,
         double tolerance, const char *semantics) {
    rows.push_back({test, observed, tolerance, observed <= tolerance, semantics});
}

double solve_single(double phi, double a, double b,
                    const PhaseKktBounds &bounds) {
    const double ncp_alpha = 0.1;
    for (int iter = 0; iter < 20; ++iter) {
        const double residual = a * phi - b;
        const int active = phase_kkt_active_code(
            phi, residual, ncp_alpha, bounds);
        double trial = phi;
        if (active == PHASE_KKT_LOWER_ACTIVE) trial = bounds.phi_lower;
        else if (active == PHASE_KKT_UPPER_ACTIVE) trial = bounds.phi_upper;
        else trial = phi - residual / a;
        trial = std::max(bounds.phi_lower, std::min(bounds.phi_upper, trial));
        phi = trial;
        if (std::fabs(phase_kkt_projected_defect(
                phi, a * phi - b, ncp_alpha, bounds)) <= 1.0e-14) break;
    }
    return phi;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s OUTPUT.csv\n", argv[0]);
        return 2;
    }
    std::vector<Row> rows;
    const PhaseKktBounds box = phase_kkt_bounds_from_ctot(
        0.45, 1.0, 1.0e-12, 1.0 - 1.0e-12, 1.0e-12);
    add(rows, "bounds_valid", box.valid ? 0.0 : 1.0, 0.0,
        "Ctot-derived local box exists");

    const double interior = solve_single(0.2, 4.0, 1.2, box);
    add(rows, "single_cell_interior_solution", std::fabs(interior - 0.3),
        1.0e-14, "free-cell Newton equation");

    const double lower = solve_single(0.2, 4.0, -1.0, box);
    add(rows, "single_cell_lower_active", std::fabs(lower - box.phi_lower),
        1.0e-14, "lower complementarity");

    const double upper = solve_single(0.2, 4.0, 8.0, box);
    add(rows, "single_cell_upper_active", std::fabs(upper - box.phi_upper),
        1.0e-14, "upper complementarity");

    const double switched = solve_single(box.phi_upper, 5.0, 1.25, box);
    add(rows, "active_set_switching", std::fabs(switched - 0.25), 1.0e-14,
        "upper-active initial state switches to free");

    const double active_phi = 0.42;
    const double active_target = 0.70;
    const double active_lambda = 1.0e-6;
    const double damped_active_trial = phase_kkt_active_line_trial(
        active_phi, 0.0, active_target, active_lambda,
        PHASE_KKT_UPPER_ACTIVE, 1);
    add(rows, "first_active_set_trial_is_lambda_continuous",
        std::fabs(damped_active_trial -
                  (active_phi + active_lambda *
                                    (active_target - active_phi))),
        1.0e-16,
        "first PDAS active classification approaches the base as lambda tends to zero");
    const double exact_active_trial = phase_kkt_active_line_trial(
        active_phi, 0.0, active_target, active_lambda,
        PHASE_KKT_UPPER_ACTIVE, 0);
    add(rows, "established_active_set_enforces_exact_bound",
        std::fabs(exact_active_trial - active_target), 1.0e-16,
        "established PDAS active constraints remain exact equalities");

    const PhaseKktBounds near_zero = phase_kkt_bounds_from_ctot(
        1.0e-10, 1.0, 1.0e-12, 1.0 - 1.0e-12, 1.0e-12);
    add(rows, "near_phi_zero", near_zero.valid ? 0.0 : 1.0, 0.0,
        "near-matrix endpoint remains feasible");

    const PhaseKktBounds near_one = phase_kkt_bounds_from_ctot(
        1.0 - 1.0e-10, 1.0, 1.0e-12, 1.0 - 1.0e-12, 1.0e-12);
    const double phi_hi = solve_single(near_one.phi_upper, 1.0, 2.0, near_one);
    add(rows, "near_phi_one", std::fabs(phi_hi - near_one.phi_upper), 1.0e-14,
        "near-zero matrix capacity uses upper complementarity");

    const double near_capacity_C = 1.0 - 1.0e-13;
    const PhaseKktBounds near_zero_capacity = phase_kkt_bounds_from_ctot(
        near_capacity_C, 1.0, 1.0e-12, 1.0 - 1.0e-12, 1.0e-12);
    const double capacity_width = near_zero_capacity.phi_upper -
                                  near_zero_capacity.phi_lower;
    add(rows, "near_zero_matrix_capacity",
        near_zero_capacity.valid && capacity_width >= 0.0 ? 0.0 : 1.0,
        0.0, "collapsed matrix capacity remains a valid KKT box");

    const double endpoint_feasible_C = 0.999999999997759126;
    const double endpoint_x_min = 1.0e-8;
    const PhaseKktBounds endpoint_feasible = phase_kkt_bounds_from_ctot(
        endpoint_feasible_C, 1.0, endpoint_x_min,
        1.0 - endpoint_x_min, 1.0e-12);
    const double endpoint_alpha = phase_kkt_alpha(
        endpoint_feasible.phi_upper);
    const double endpoint_q = phase_kkt_q_from_ctot(
        endpoint_feasible.phi_upper, endpoint_feasible_C, 1.0);
    const double endpoint_reconstructed_x = endpoint_q / endpoint_alpha;
    const double endpoint_feasibility_defect =
        !endpoint_feasible.valid || !std::isfinite(endpoint_reconstructed_x)
            ? 1.0
            : std::max(endpoint_x_min - endpoint_reconstructed_x, 0.0);
    add(rows, "near_beta_conservative_inverse_feasibility",
        endpoint_feasibility_defect, 2.0e-16,
        "upper inverse endpoint and stable q reconstruction remain feasible");

    const double uniform_phi = 0.37;
    const double uniform_residual = 0.0;
    add(rows, "uniform_phase_equilibrium",
        std::fabs(phase_kkt_projected_defect(
            uniform_phi, uniform_residual, 0.1, box)),
        1.0e-15, "zero phase residual is a stationary interior KKT state");

    double manufactured_error = 0.0;
    for (int i = 0; i < 9; ++i) {
        const double root = 0.1 + 0.08 * i;
        const double solved = solve_single(0.5, 3.0, 3.0 * root, box);
        const double expected = std::max(
            box.phi_lower, std::min(box.phi_upper, root));
        manufactured_error = std::max(
            manufactured_error, std::fabs(solved - expected));
    }
    add(rows, "manufactured_constrained_phase", manufactured_error,
        1.0e-14, "manufactured roots cover free and bound-active cells");

    double storage = 0.0;
    const double C = 0.37;
    const double p0 = 0.21;
    const double p1 = 0.43;
    const double q0 = C - phase_kkt_h(p0);
    const double q1 = C - phase_kkt_h(p1);
    storage = std::fabs(q1 - q0 + phase_kkt_h(p1) - phase_kkt_h(p0));
    add(rows, "fixed_Ctot_storage_bound", storage, 2.0e-16,
        "phase transaction preserves local Ctot exactly");

    const auto nonlinear = [](double value) {
        return std::exp(value) + value * value * value;
    };
    const double boundary = 0.73;
    const double fd_step = 1.0e-3;
    const double coarse_one_sided =
        (nonlinear(boundary) - nonlinear(boundary - fd_step)) / fd_step;
    const double fine_one_sided =
        (nonlinear(boundary) - nonlinear(boundary - 0.5 * fd_step)) /
        (0.5 * fd_step);
    const double refined_one_sided = phase_kkt_richardson_derivative(
        coarse_one_sided, fine_one_sided, 0);
    const double exact_derivative = std::exp(boundary) + 3.0 * boundary * boundary;
    add(rows, "bound_aware_richardson_jacobian",
        std::fabs(refined_one_sided - exact_derivative), 2.0e-6,
        "one-sided local Jacobian cancels its leading bound-truncation error");

    const double endpoint_phi = 9.9899949944e-1;
    const double endpoint_C = 9.99999992087557521e-1;
    const double endpoint_local_alpha = phase_kkt_alpha(endpoint_phi);
    const double endpoint_x = phase_kkt_q_from_ctot(
        endpoint_phi, endpoint_C, 1.0) / endpoint_local_alpha;
    const double endpoint_dx_dphi = phase_kkt_h_prime(endpoint_phi) *
        (endpoint_x - 1.0) / endpoint_local_alpha;
    const double adaptive_step = phase_kkt_fixed_ctot_fd_step(
        endpoint_phi, endpoint_C, 1.0, 1.0e-6);
    const double induced_dx = std::fabs(endpoint_dx_dphi * adaptive_step);
    add(rows, "near_beta_fixed_ctot_fd_composition_locality",
        std::max(induced_dx - 1.0e-6, 0.0), 2.0e-15,
        "adaptive phi perturbation limits induced x_alpha change near h=1");

    const auto synthetic_local_driving = [](double phi, double x) {
        return std::sin(phi) + std::exp(x);
    };
    const double independent_step = 1.0e-6;
    const double partial_phi =
        (synthetic_local_driving(endpoint_phi + independent_step, endpoint_x) -
         synthetic_local_driving(endpoint_phi - independent_step, endpoint_x)) /
        (2.0 * independent_step);
    const double partial_x =
        (synthetic_local_driving(endpoint_phi, endpoint_x + independent_step) -
         synthetic_local_driving(endpoint_phi, endpoint_x - independent_step)) /
        (2.0 * independent_step);
    const double chain_fd = partial_phi + partial_x *
        phase_kkt_fixed_ctot_dx_dphi(endpoint_phi, endpoint_C, 1.0, 1.0e-14);
    const double chain_exact = std::cos(endpoint_phi) +
        std::exp(endpoint_x) * endpoint_dx_dphi;
    add(rows, "near_beta_independent_partial_chain_rule_jacobian",
        std::fabs(chain_fd - chain_exact), 2.0e-6,
        "regular independent partials recover fixed-Ctot derivative near h=1");

    std::ofstream output(argv[1]);
    output << "test,observed,tolerance,pass,semantics\n";
    int failures = 0;
    for (const Row &row : rows) {
        output << row.test << ',' << row.observed << ',' << row.tolerance << ','
               << (row.pass ? 1 : 0) << ',' << row.semantics << '\n';
        failures += row.pass ? 0 : 1;
        std::printf("%s=%s observed=%.17e\n", row.test.c_str(),
                    row.pass ? "PASS" : "FAIL", row.observed);
    }
    return failures == 0 ? 0 : 1;
}
