import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_stationary_curved_equilibrium.py"
SPEC = importlib.util.spec_from_file_location("stationary_curved", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StationaryCurvedEquilibriumTests(unittest.TestCase):
    def test_h_symmetry_is_stable(self):
        values = np.linspace(0.0, 1.0, 101)
        np.testing.assert_allclose(
            1.0 - MODULE.h(values), MODULE.h(1.0 - values),
            rtol=0.0, atol=2.0e-15,
        )

    def test_spectral_laplacian_manufactured_mode(self):
        n = 64
        dx = 0.05
        x = np.arange(n) * dx
        wave = 2.0 * np.pi / (n * dx)
        xx, zz = np.meshgrid(x, x, indexing="ij")
        field = np.sin(wave * xx) + 0.25 * np.cos(2.0 * wave * zz)
        expected = -(wave**2) * np.sin(wave * xx)
        expected -= (2.0 * wave) ** 2 * 0.25 * np.cos(2.0 * wave * zz)
        observed = MODULE.spectral_laplacian_2d(field, dx)
        self.assertLess(float(np.max(np.abs(observed - expected))), 2.0e-12)

    def test_vector_phase_bounds_match_production_scalar_algorithm(self):
        values = np.array([
            1.0e-8, 0.01, 0.5, 0.99,
            np.nextafter(1.0, 0.0), 1.0,
        ])
        lower, upper = MODULE.phase_bounds(values, 1.0, 1.0e-8, 1.0 - 1.0e-8)
        scalar = np.array([
            MODULE.phase_bounds_scalar(value, 1.0, 1.0e-8, 1.0 - 1.0e-8)
            for value in values
        ])
        np.testing.assert_array_equal(lower, scalar[:, 0])
        np.testing.assert_array_equal(upper, scalar[:, 1])

    def test_radial_and_production_residual_close(self):
        temperature = 673.15
        scale = 1.3779024e5
        xeq = MODULE.unit.xAg2Te_eq_from_T(temperature)
        mu0 = MODULE.unit.mu_Ag2Te(temperature, xeq) / scale
        solution, matrix_x, delta_mu = MODULE.solve_radial_equilibrium(
            3.0, 8.0, temperature, scale, 1.0, 0.045, mu0,
            radial_spacing=0.025, tolerance=1.0e-8,
        )
        self.assertTrue(solution.success)
        self.assertGreater(matrix_x, xeq)
        self.assertAlmostEqual(
            MODULE.matrix_x_from_delta_mu(delta_mu, temperature, scale, mu0),
            matrix_x,
            places=13,
        )
        n = 320
        dx = 0.05
        coords = np.arange(n) * dx
        xx, zz = np.meshgrid(coords, coords, indexing="ij")
        radius = np.sqrt((xx - 8.0) ** 2 + (zz - 8.0) ** 2)
        phi = solution.sol(np.minimum(radius, 8.0).ravel())[0].reshape(n, n)
        residual = (
            MODULE.gp(phi) + delta_mu * MODULE.hp(phi)
            - 0.045 * MODULE.spectral_laplacian_2d(phi, dx)
        )
        self.assertLess(float(np.max(np.abs(residual))), 2.0e-8)
        h_volume = float(np.sum(MODULE.h(phi)) * dx**2)
        self.assertAlmostEqual(h_volume, np.pi * 3.0**2, places=9)

        refined, refined_delta, core, diagnostics = (
            MODULE.refine_production_spectral_equilibrium(
                phi,
                delta_mu,
                matrix_x,
                np.pi * 3.0**2 / dx**2,
                dx,
                1.0,
                0.045,
                3.0e-8,
            )
        )
        refined_residual = (
            MODULE.gp(refined) + refined_delta * MODULE.hp(refined)
            - 0.045 * MODULE.spectral_laplacian_2d(refined, dx)
        )
        self.assertLess(
            float(np.max(np.abs(refined_residual[~core]))), 3.0e-8
        )
        self.assertLess(abs(float(diagnostics["volume_residual_cell_units"])),
                        1.0e-8)
        self.assertGreater(int(diagnostics["representability_core_cells"]), 0)

    def test_fixed_mass_storage_reconstruction(self):
        phi = np.linspace(0.0, 1.0, 257)
        matrix_x = 0.01
        alpha = np.where(phi > 0.5, MODULE.h(1.0 - phi), 1.0 - MODULE.h(phi))
        ctot = 1.0 - alpha * (1.0 - matrix_x)
        q_alpha = (ctot - 1.0) + alpha
        reconstructed_x = np.zeros_like(phi)
        active = alpha > 1.0e-12
        reconstructed_x[active] = q_alpha[active] / alpha[active]
        self.assertLess(
            float(np.max(np.abs(reconstructed_x[active] - matrix_x))),
            2.0e-11,
        )


if __name__ == "__main__":
    unittest.main()
