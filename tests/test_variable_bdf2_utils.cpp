#include <cassert>
#include <cmath>
#include <initializer_list>

#include "../variable_bdf2_utils.h"

static bool close(double a, double b, double tol = 1.0e-14) {
    return std::fabs(a - b) <= tol;
}

int main() {
    const auto fixed = variable_bdf2_coefficients_v1(0.5, 0.5);
    assert(fixed.valid);
    assert(close(fixed.ratio, 1.0));
    assert(close(fixed.a0, 1.5));
    assert(close(fixed.a1, -2.0));
    assert(close(fixed.a2, 0.5));
    assert(close(fixed.anchor_n, 4.0 / 3.0));
    assert(close(fixed.anchor_nm1, -1.0 / 3.0));
    assert(close(fixed.extrap_n, 2.0));
    assert(close(fixed.extrap_nm1, -1.0));
    assert(close(fixed.effective_dt_factor, 2.0 / 3.0));

    for (double r : {0.25, 0.5, 1.0, 1.25, 2.0}) {
        const auto c = variable_bdf2_coefficients_v1(r, 1.0);
        assert(c.valid);
        assert(close(c.a0 + c.a1 + c.a2, 0.0));
        assert(close(c.anchor_n + c.anchor_nm1, 1.0));
        // Exact derivative for y=t at the new endpoint.
        const double y_np1 = r;
        const double y_n = 0.0;
        const double y_nm1 = -1.0;
        const double derivative =
            (c.a0 * y_np1 + c.a1 * y_n + c.a2 * y_nm1) / r;
        assert(close(derivative, 1.0));
    }
    assert(!variable_bdf2_coefficients_v1(2.01, 1.0).valid);
    assert(!variable_bdf2_coefficients_v1(0.24, 1.0).valid);

    const auto accepted = variable_bdf2_pi_decision_v1(
        1.0, 0.01, 0.02, 0.1, 4.0, 0.9, 0.35, 0.2);
    assert(accepted.valid && accepted.accept);
    assert(accepted.limited_factor <= 1.25);
    const auto rejected = variable_bdf2_pi_decision_v1(
        1.0, 4.0, 1.0, 0.1, 4.0, 0.9, 0.35, 0.2);
    assert(rejected.valid && !rejected.accept);
    assert(rejected.limited_factor >= 0.25);
    assert(rejected.limited_factor <= 1.0);
    return 0;
}
