"""N8 observation-mapping regression tests."""

from __future__ import annotations

import unittest

from kwn_mvp.composition_mapping import ag_at_fraction_to_xb, xb_to_ag_at_fraction


class CompositionMappingTest(unittest.TestCase):
    """Verify ideal pseudo-binary mapping across supplied Sheskin values."""

    def test_round_trip_across_primary_constraint_range(self) -> None:
        """AQ, 6 h, and 48 h matrix values must round-trip to machine precision."""

        for ag_value in (0.0078, 0.0062, 0.0062):
            x_b = ag_at_fraction_to_xb(ag_value)
            self.assertAlmostEqual(xb_to_ag_at_fraction(x_b), ag_value, places=15)

    def test_known_formula_value(self) -> None:
        """The 6 h observation uses the documented analytical conversion."""

        self.assertAlmostEqual(ag_at_fraction_to_xb(0.0062), 2.0 * 0.0062 / (2.0 - 0.0062), places=15)
