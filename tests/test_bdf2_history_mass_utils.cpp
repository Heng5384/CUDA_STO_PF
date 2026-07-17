#include <cassert>
#include <cmath>

#include "../bdf2_history_mass_utils.h"

int main() {
    assert(ctot_bdf2_history_mass_tolerance_v1(50.0) == 1.0e-10);
    const double large_mass = 3.3e7;
    const double tolerance = ctot_bdf2_history_mass_tolerance_v1(large_mass);
    assert(tolerance ==
           ctot_bdf2_reduction_equivalence_tolerance_v1(large_mass));
    assert(tolerance > 1.0e-9);
    assert(tolerance < 1.0e-5);
    assert(ctot_bdf2_history_mass_consistent_v1(1.9e-9, large_mass));
    assert(ctot_bdf2_reduction_equivalent_v1(1.9e-9, large_mass));
    assert(!ctot_bdf2_reduction_equivalent_v1(
        std::nextafter(tolerance, INFINITY), large_mass));
    assert(!ctot_bdf2_history_mass_consistent_v1(1.0e-4, large_mass));
    assert(!ctot_bdf2_history_mass_consistent_v1(NAN, large_mass));
    return 0;
}
