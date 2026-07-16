#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bounded_retry_bdf2_utils.h"

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

void check_versioned_contract_names() {
    require(ctot_retry_acceptance_contract_valid(
                "ZERO_REJECT_FIXED_STEP_V1"),
            "historical zero-reject contract must remain valid");
    require(ctot_bounded_retry_contract_selected(
                "ACTIVE_MANIFOLD_BDF2_BOUNDED_RETRY_PRODUCTION_V1"),
            "bounded-retry contract must be explicit");
    require(!ctot_retry_acceptance_contract_valid("unknown"),
            "unknown acceptance contracts must fail closed");
}

void check_failure_names_are_specific() {
    require(std::strcmp(ctot_transport_failure_name(
                            CTOT_TRANSPORT_LINE_SEARCH_STAGNATION),
                        "transport_line_search_stagnation") == 0,
            "line-search stagnation must not be reported as generic failure");
    require(std::strcmp(ctot_transport_failure_name(
                            CTOT_TRANSPORT_NONLINEAR_ITERATION_LIMIT),
                        "transport_nonlinear_iteration_limit") == 0,
            "iteration-limit failure must retain its own category");
}

void check_gate_boundary() {
    CtotBoundedRetryMetrics metrics = {};
    metrics.macro_steps = 1000;
    metrics.internal_trial_rejects = 10;
    metrics.fallback_macros = 10;
    metrics.retry_wall_overhead_fraction = 0.05;
    metrics.max_consecutive_fallback_macros = 2;
    metrics.max_accepted_subcycle_depth = 2;
    metrics.accepted_iteration_p99 = 399.0;
    metrics.nonlinear_iteration_budget = 500;
    metrics.accepted_state_gates_pass = 1;
    metrics.no_persistent_interface_cell_rejection = 1;
    const auto pass = ctot_bounded_retry_gate_v1(metrics);
    require(pass.pass, "pre-registered inclusive fraction/overhead gates must pass");

    metrics.internal_trial_rejects = 11;
    require(!ctot_bounded_retry_gate_v1(metrics).pass,
            "retry fraction above one percent must fail");
    metrics.internal_trial_rejects = 10;
    metrics.accepted_iteration_p99 = 400.0;
    require(!ctot_bounded_retry_gate_v1(metrics).pass,
            "p99 at eighty percent of budget must fail the strict gate");
}

}  // namespace

int main() {
    check_versioned_contract_names();
    check_failure_names_are_specific();
    check_gate_boundary();
    std::puts("bounded_retry_bdf2_utils=PASS");
    return 0;
}
