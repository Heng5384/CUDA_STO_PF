#!/usr/bin/env python3
import math
import unittest

from scripts.particle_coarsening_oracle import OracleParams, run_oracle


def handoff(radii: list[float], matrix_xB: float) -> dict:
    box_cells = 100**3
    beta = sum((4.0 * math.pi / 3.0) * r**3 for r in radii)
    matrix_cells = box_cells - beta
    total = beta + matrix_cells * matrix_xB
    return {
        "schema": "PF_PARTICLE_HANDOFF_LEDGER_V1",
        "grid_shape": [100, 100, 100],
        "dx_nm": 1.0,
        "cell_volume_nm3": 1.0,
        "v_B": 1.0,
        "particles": [
            {"particle_id": i + 1, "equivalent_radius_nm": r}
            for i, r in enumerate(radii)
        ],
        "ledger": {"total_inventory_with_GP": total, "GP_inventory": 0.0},
    }


class ParticleOracleTest(unittest.TestCase):
    def test_two_particle_competition_and_mass(self) -> None:
        trajectory, summary = run_oracle(
            handoff([4.0, 10.0], 0.012),
            OracleParams(0.01, 1.0, 0.2, 0.5),
            100.0,
        )
        self.assertLessEqual(summary["max_mass_error_rel"], 1.0e-12)
        self.assertLess(summary["final_radii_nm"][0], 4.0)
        self.assertGreater(summary["final_radii_nm"][1], 10.0)
        self.assertTrue(trajectory)

    def test_three_particle_ordering(self) -> None:
        _, summary = run_oracle(
            handoff([2.0, 4.0, 10.0], 0.012),
            OracleParams(0.01, 1.0, 1.0, 0.2),
            2000.0,
        )
        self.assertLessEqual(summary["max_mass_error_rel"], 1.0e-12)
        self.assertTrue(summary["extinction_order"])
        self.assertEqual(summary["extinction_order"][0], 1)


if __name__ == "__main__":
    unittest.main()
