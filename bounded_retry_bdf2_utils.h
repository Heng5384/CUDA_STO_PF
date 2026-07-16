#ifndef BOUNDED_RETRY_BDF2_UTILS_H
#define BOUNDED_RETRY_BDF2_UTILS_H

#include <cmath>
#include <cstring>

enum CtotTransportFailureCode {
    CTOT_TRANSPORT_FAILURE_NONE = 0,
    CTOT_TRANSPORT_INITIAL_EVALUATION = 1,
    CTOT_TRANSPORT_TANGENT_ACTIVE_SET_UNRESOLVED = 2,
    CTOT_TRANSPORT_LINE_SEARCH_STAGNATION = 3,
    CTOT_TRANSPORT_NONLINEAR_ITERATION_LIMIT = 4
};

static inline const char *ctot_transport_failure_name(
    CtotTransportFailureCode code) {
    switch (code) {
        case CTOT_TRANSPORT_INITIAL_EVALUATION:
            return "transport_initial_evaluation_failed";
        case CTOT_TRANSPORT_TANGENT_ACTIVE_SET_UNRESOLVED:
            return "transport_tangent_active_set_unresolved";
        case CTOT_TRANSPORT_LINE_SEARCH_STAGNATION:
            return "transport_line_search_stagnation";
        case CTOT_TRANSPORT_NONLINEAR_ITERATION_LIMIT:
            return "transport_nonlinear_iteration_limit";
        default:
            return "none";
    }
}

static inline int ctot_bounded_retry_contract_selected(const char *name) {
    return name && std::strcmp(
        name, "ACTIVE_MANIFOLD_BDF2_BOUNDED_RETRY_PRODUCTION_V1") == 0;
}

static inline int ctot_retry_acceptance_contract_valid(const char *name) {
    return name &&
        (std::strcmp(name, "ZERO_REJECT_FIXED_STEP_V1") == 0 ||
         ctot_bounded_retry_contract_selected(name));
}

struct CtotBoundedRetryMetrics {
    long long macro_steps;
    long long macro_hard_rejects;
    long long internal_trial_rejects;
    long long fallback_macros;
    int max_consecutive_fallback_macros;
    int max_accepted_subcycle_depth;
    double retry_wall_overhead_fraction;
    double accepted_iteration_p99;
    int nonlinear_iteration_budget;
    int accepted_state_gates_pass;
    int no_persistent_interface_cell_rejection;
};

struct CtotBoundedRetryGateResult {
    int pass;
    double retry_fraction;
    double fallback_fraction;
};

static inline CtotBoundedRetryGateResult ctot_bounded_retry_gate_v1(
    const CtotBoundedRetryMetrics &m) {
    CtotBoundedRetryGateResult result = {0, NAN, NAN};
    if (m.macro_steps <= 0 || m.nonlinear_iteration_budget <= 0) return result;
    result.retry_fraction =
        static_cast<double>(m.internal_trial_rejects) /
        static_cast<double>(m.macro_steps);
    result.fallback_fraction =
        static_cast<double>(m.fallback_macros) /
        static_cast<double>(m.macro_steps);
    result.pass =
        m.macro_hard_rejects == 0 && m.accepted_state_gates_pass &&
        result.retry_fraction <= 0.01 && result.fallback_fraction <= 0.01 &&
        m.retry_wall_overhead_fraction <= 0.05 &&
        m.max_consecutive_fallback_macros <= 2 &&
        m.max_accepted_subcycle_depth <= 2 &&
        m.no_persistent_interface_cell_rejection &&
        m.accepted_iteration_p99 < 0.8 * m.nonlinear_iteration_budget;
    return result;
}

#endif
