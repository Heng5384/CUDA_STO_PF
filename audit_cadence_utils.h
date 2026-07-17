#ifndef AUDIT_CADENCE_UTILS_H
#define AUDIT_CADENCE_UTILS_H

#include <math.h>

enum CtotFullAuditReason : unsigned int {
    CTOT_FULL_AUDIT_PERIODIC = 1u << 0,
    CTOT_FULL_AUDIT_FIRST_STEP = 1u << 1,
    CTOT_FULL_AUDIT_FINAL_STEP = 1u << 2,
    CTOT_FULL_AUDIT_RETRY = 1u << 3,
    CTOT_FULL_AUDIT_DT_CHANGE = 1u << 4,
    CTOT_FULL_AUDIT_HISTORY_REBUILD = 1u << 5,
    CTOT_FULL_AUDIT_ACTIVE_MANIFOLD = 1u << 6,
    CTOT_FULL_AUDIT_EVENT = 1u << 7
};

struct CtotFullAuditDecision {
    int due;
    unsigned int reasons;
};

inline CtotFullAuditDecision ctot_full_audit_decision_v1(
    int step, int final_step, int cadence, int retry_count,
    int history_valid, double dt, double previous_dt) {
    CtotFullAuditDecision out = {0, 0u};
    if (cadence < 1) cadence = 1;
    if (step % cadence == 0) out.reasons |= CTOT_FULL_AUDIT_PERIODIC;
    if (step == 1) out.reasons |= CTOT_FULL_AUDIT_FIRST_STEP;
    if (step == final_step) out.reasons |= CTOT_FULL_AUDIT_FINAL_STEP;
    if (retry_count > 0) out.reasons |= CTOT_FULL_AUDIT_RETRY;
    if (!history_valid) out.reasons |= CTOT_FULL_AUDIT_HISTORY_REBUILD;
    if (isfinite(previous_dt) && isfinite(dt) && dt != previous_dt)
        out.reasons |= CTOT_FULL_AUDIT_DT_CHANGE;
    out.due = out.reasons != 0u;
    return out;
}

inline void ctot_force_full_audit_v1(CtotFullAuditDecision *decision,
                                     unsigned int reason) {
    if (!decision) return;
    decision->reasons |= reason;
    decision->due = 1;
}

#endif
