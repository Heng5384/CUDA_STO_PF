#include <cassert>
#include <cmath>

#include "../audit_cadence_utils.h"

int main() {
    CtotFullAuditDecision first = ctot_full_audit_decision_v1(
        1, 100, 50, 0, 1, 0.1, 0.1);
    assert(first.due);
    assert(first.reasons & CTOT_FULL_AUDIT_FIRST_STEP);

    CtotFullAuditDecision quiet = ctot_full_audit_decision_v1(
        2, 100, 50, 0, 1, 0.1, 0.1);
    assert(!quiet.due);

    CtotFullAuditDecision periodic = ctot_full_audit_decision_v1(
        50, 100, 50, 0, 1, 0.1, 0.1);
    assert(periodic.reasons & CTOT_FULL_AUDIT_PERIODIC);

    CtotFullAuditDecision retry = ctot_full_audit_decision_v1(
        7, 100, 50, 1, 1, 0.1, 0.1);
    assert(retry.reasons & CTOT_FULL_AUDIT_RETRY);

    CtotFullAuditDecision changed = ctot_full_audit_decision_v1(
        7, 100, 50, 0, 1, 0.05, 0.1);
    assert(changed.reasons & CTOT_FULL_AUDIT_DT_CHANGE);

    CtotFullAuditDecision rebuild = ctot_full_audit_decision_v1(
        7, 100, 50, 0, 0, 0.1, NAN);
    assert(rebuild.reasons & CTOT_FULL_AUDIT_HISTORY_REBUILD);

    ctot_force_full_audit_v1(&quiet, CTOT_FULL_AUDIT_EVENT);
    assert(quiet.due);
    assert(quiet.reasons & CTOT_FULL_AUDIT_EVENT);
    return 0;
}
