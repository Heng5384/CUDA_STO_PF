#include "../bdf2_event_utils.h"

#include <cassert>
#include <cmath>

int main() {
    const auto dt8 = bdf2_event_classify_v1(
        0.9999994891680709, 0.9999994891321955,
        0.9962826389863121, 0.9962825517870315, 1.0, 1.0e-12);
    assert(dt8.reason == BDF2_EVENT_ACTIVE_SET_TRANSITION);
    assert(dt8.q_n > 0.0);
    assert(dt8.q_anchor_E < 0.0);

    // An ULP-scale sign motion inside the already lower-active branch is not
    // repeatedly subcycled. The stable endpoint audit handles this case.
    const double phi_n = 0.995;
    const double phi_nm1 = 0.9949999;
    const double phi_E = 2.0 * phi_n - phi_nm1;
    const double h_n = bdf2_event_h(phi_n);
    const double h_E = bdf2_event_h(phi_E);
    const double C_n = h_n + 3.0e-16;
    const double C_nm1 = 4.0 * C_n - 3.0 * (h_E - 9.0e-15);
    const auto dt16 = bdf2_event_classify_v1(
        C_n, C_nm1, phi_n, phi_nm1, 1.0, 1.0e-12);
    assert(dt16.reason == BDF2_EVENT_NONE);
    assert(dt16.q_n <= 1.0e-12);
    assert(dt16.q_anchor_E <= 1.0e-12);

    const auto smooth = bdf2_event_classify_v1(
        0.2, 0.199, 0.3, 0.299, 1.0, 1.0e-12);
    assert(smooth.reason == BDF2_EVENT_NONE);

    const double Ft = 1.25;
    const double Fn = 1.10;
    const double mu_work = 0.04;
    const double phase_work = 0.03;
    const double stable = bdf2_stable_endpoint_chain_v2(
        Ft, Fn, mu_work, phase_work);
    const double expanded = (1.18 - Fn - 0.02) +
                            (Ft - 1.18 - phase_work) +
                            (0.02 - mu_work);
    assert(std::fabs(stable - expanded) <= 8.0e-17);
    return 0;
}
