import pathlib
import sys
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_finite_interface_stefan_windows import (  # noqa: E402
    crossing_positions,
    linear_surface_extrapolation,
    matrix_mu_and_mobility,
)
from prepare_one_sided_planar_benchmark import sharp_similarity_parameter  # noqa: E402
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402
from run_finite_interface_stefan_matrix_workstation import (  # noqa: E402
    require_unscaled_base_params,
)

PREP_SOURCE = (ROOT / "scripts/prepare_one_sided_planar_benchmark.py").read_text()
RUNNER_SOURCE = (ROOT / "scripts/run_finite_interface_stefan_matrix_workstation.py").read_text()
ANALYZER_SOURCE = (ROOT / "scripts/analyze_finite_interface_stefan_windows.py").read_text()


class FiniteInterfaceStefanAnalyzerTests(unittest.TestCase):
    def test_runner_rejects_already_scaled_base(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            params = root / "benchmark.params"
            params.write_text("L_phi=1\n")
            (root / "benchmark_manifest.json").write_text(json.dumps({
                "L_phi_factor": 64.0,
            }))
            with self.assertRaisesRegex(ValueError, "already-scaled"):
                require_unscaled_base_params(params)

    def test_retry_cases_are_explicitly_disqualified_from_equal_dt_curves(self):
        self.assertIn('"retry_count": retry_count', ANALYZER_SOURCE)
        self.assertIn('"equal_requested_dt": retry_count == 0', ANALYZER_SOURCE)

    def test_benchmark_correction_is_explicit_opt_in(self):
        self.assertIn(
            'parser.add_argument("--finite-interface-antitrapping", action="store_true")',
            PREP_SOURCE,
        )
        self.assertIn(
            'int(args.finite_interface_antitrapping)', PREP_SOURCE
        )
        self.assertIn('"ctot_outer_max_iter": args.outer_max_iter', PREP_SOURCE)
        self.assertIn(
            '"provenance": "finite_interface_asymptotic_audit_not_solver_compensation"',
            RUNNER_SOURCE,
        )
        self.assertNotIn("ctot_spectral_be", RUNNER_SOURCE)
        self.assertIn('"binary_sha256": sha256(binary)', RUNNER_SOURCE)
        self.assertIn('"retry_count": len(re.findall', RUNNER_SOURCE)

    def test_linear_matrix_side_extrapolation_recovers_surface_value(self):
        surface = 2.0
        positions = np.linspace(2.1, 3.5, 15)
        distances = positions - surface
        values = 7.0 - 0.25 * distances
        extrapolated, count, rms = linear_surface_extrapolation(
            positions, values, surface, distances, 0.1, 1.5
        )
        self.assertEqual(count, 15)
        self.assertAlmostEqual(extrapolated, 7.0, places=13)
        self.assertLess(rms, 1.0e-14)

    def test_cubic_extrapolation_recovers_curved_surface_value(self):
        surface = 2.0
        positions = np.linspace(2.1, 4.0, 20)
        distances = positions - surface
        values = 7.0 - distances + 0.5 * distances**2 - 0.1 * distances**3
        extrapolated, count, rms = linear_surface_extrapolation(
            positions, values, surface, distances, 0.1, 2.0, degree=3
        )
        self.assertEqual(count, 20)
        self.assertAlmostEqual(extrapolated, 7.0, places=12)
        self.assertLess(rms, 1.0e-13)

    def test_crossing_uses_phi_half_not_h_integral(self):
        phi = np.array([0.0, 0.25, 0.75, 1.0, 0.75, 0.25])
        self.assertEqual(crossing_positions(phi, 1.0), [1.5, 4.5])

    def test_crossing_accepts_exact_phi_half_grid_point(self):
        phi = np.array([0.0, 0.5, 1.0, 0.5])
        np.testing.assert_allclose(
            crossing_positions(phi, 0.2), [0.2, 0.6], rtol=0.0, atol=1.0e-15
        )

    def test_cubic_flux_extrapolation_recovers_similarity_oracle(self):
        import math

        temperature = 673.15
        mu_scale = 1.3779024e5
        D_code = 9.0
        dx = 0.1
        nx = 192
        left, right = 0.35 * nx * dx, 0.65 * nx * dx
        coordinates = np.arange(nx) * dx
        phi = 0.5 * (
            np.tanh((coordinates - left) / 0.3) -
            np.tanh((coordinates - right) / 0.3)
        )
        x_eq = unit.xAg2Te_eq_from_T(temperature)
        x_inf = 0.05
        eta0 = sharp_similarity_parameter(x_inf, x_eq)
        D_phys = unit.D_Ag_in_PbTe_m2_per_s(temperature) * 1.0e18
        start_time = 0.1
        diffusion_length = math.sqrt(D_phys * start_time)
        x = np.full(nx, x_eq)
        for index, coordinate in enumerate(coordinates):
            if coordinate < left:
                distance = left - coordinate
            elif coordinate > right:
                distance = coordinate - right
            else:
                continue
            eta = eta0 + distance / (2.0 * diffusion_length)
            normalized = ((math.erf(eta) - math.erf(eta0)) /
                          math.erfc(eta0))
            x[index] = x_eq + (x_inf - x_eq) * normalized
        mu, mobility = matrix_mu_and_mobility(
            phi, x, temperature, mu_scale, D_code
        )
        neighbor = np.roll(np.arange(nx), -1)
        face_mobility = np.zeros(nx)
        active = (mobility > 0.0) & (mobility[neighbor] > 0.0)
        face_mobility[active] = (
            2.0 * mobility[active] * mobility[neighbor][active] /
            (mobility[active] + mobility[neighbor][active])
        )
        flux = face_mobility * (mu[neighbor] - mu) / dx
        surface = crossing_positions(phi, dx)[1]
        positions = (np.arange(nx) + 0.5) * dx
        distances = (positions - surface) % (nx * dx)
        pure_matrix = (phi < 0.01) & (phi[neighbor] < 0.01)
        surface_flux, _, _ = linear_surface_extrapolation(
            positions, np.where(pure_matrix, flux, np.nan), surface,
            distances, 0.1, 2.0, degree=3
        )
        x_face = 0.5 * (x + x[neighbor])
        surface_x, _, _ = linear_surface_extrapolation(
            positions, np.where(pure_matrix, x_face, np.nan), surface,
            distances, 0.1, 2.0, degree=3
        )
        flux_velocity_code = surface_flux / (1.0 - surface_x)
        t0 = 0.9254156720524821
        sharp_velocity_code = (
            eta0 * math.sqrt(D_phys / start_time) * t0
        )
        self.assertLess(
            abs(flux_velocity_code / sharp_velocity_code - 1.0), 0.01
        )


if __name__ == "__main__":
    unittest.main()
