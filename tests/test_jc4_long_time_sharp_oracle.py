import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.jc4_long_time_sharp_oracle import (  # noqa: E402
    LongTimePlanarSharpOracle,
    SharpConfig,
)


class Jc4LongTimeSharpOracleTests(unittest.TestCase):
    def test_equilibrium_is_stationary_and_mass_closed(self):
        base = LongTimePlanarSharpOracle(SharpConfig()).x_eq
        oracle = LongTimePlanarSharpOracle(SharpConfig(
            length_nm=128.0, initial_beta_half_width_nm=10.0,
            initial_matrix_xB=base, matrix_cells=64,
            final_time_s=1.0, output_points=3,
        ))
        result = oracle.run()
        self.assertLess(abs(result["records"][-1]["displacement_nm"]), 1.0e-11)
        self.assertLess(result["max_mass_error_rel"], 1.0e-12)

    def test_growth_and_dissolution_directions(self):
        rows = []
        for x0 in (0.03, 0.003):
            oracle = LongTimePlanarSharpOracle(SharpConfig(
                length_nm=128.0, initial_beta_half_width_nm=10.0,
                initial_matrix_xB=x0, matrix_cells=64,
                final_time_s=0.1, output_points=2,
            ))
            rows.append(oracle.run()["records"][-1]["displacement_nm"])
        self.assertGreater(rows[0], 0.0)
        self.assertLess(rows[1], 0.0)


if __name__ == "__main__":
    unittest.main()
