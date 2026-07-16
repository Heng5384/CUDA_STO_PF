import math
from pathlib import Path
import unittest

import numpy as np

from scripts.next2_finite_box_sharp_oracle import (
    add_far_inventory_correction,
    aged_no_flux_profile,
    evolve_moving_interface,
    finite_lphi_kinetic_beta_code,
    profile_inventory,
    solve_gibbs_thomson_x,
    solve_interface_composition_x,
    solve_stefan_surface_state,
    square_equivalent_outer_radius,
    volume_equivalent_outer_radius,
    run_case,
)
import Unit_Psedobinary as unit


class FiniteBoxSharpOracleTests(unittest.TestCase):
    def test_equal_area_outer_radius(self):
        radius = square_equivalent_outer_radius(16.0)
        self.assertAlmostEqual(math.pi * radius * radius, 16.0**2, places=13)

    def test_equal_volume_spherical_outer_radius(self):
        radius = volume_equivalent_outer_radius(16.0, 3)
        self.assertLess(
            abs((4.0 / 3.0) * math.pi * radius**3 - 16.0**3)
            / 16.0**3,
            1.0e-15,
        )

    def test_aged_profile_has_fixed_surface_and_no_flux_outer_boundary(self):
        radius, outer = 3.0, square_equivalent_outer_radius(16.0)
        nodes, profile = aged_no_flux_profile(
            radius, outer, 9.0, 0.009, 0.0078, 4.0 / 9.0, 256
        )
        self.assertEqual(profile[0], 0.009)
        self.assertTrue(np.all(np.isfinite(profile)))
        self.assertLess(abs(profile[-1] - profile[-2]), 2.0e-8)

    def test_inventory_correction_preserves_interface_neighborhood(self):
        radius, outer = 3.0, square_equivalent_outer_radius(16.0)
        nodes, profile = aged_no_flux_profile(
            radius, outer, 9.0, 0.009, 0.0078, 4.0 / 9.0, 256
        )
        target = profile_inventory(radius, nodes, profile) + 0.05
        corrected, delta, actual = add_far_inventory_correction(
            radius, nodes, profile, target, 1.2
        )
        self.assertNotEqual(delta, 0.0)
        self.assertAlmostEqual(actual, target, places=11)
        self.assertTrue(np.array_equal(corrected[nodes <= radius + 1.2],
                                       profile[nodes <= radius + 1.2]))

    def test_uniform_equilibrium_remains_stationary(self):
        radius, outer, surface = 3.0, square_equivalent_outer_radius(16.0), 0.009
        nodes = np.linspace(radius, outer, 129)
        profile = np.full(nodes.size, surface)
        history, _, final = evolve_moving_interface(
            radius,
            outer,
            nodes,
            profile,
            9.0,
            lambda value, velocity: surface,
            1.0e-3,
            1.0e-5,
        )
        self.assertLess(abs(history[-1]["radius"] - radius), 1.0e-13)
        self.assertLess(np.max(np.abs(final - surface)), 1.0e-13)

    def test_zero_kinetic_interface_root_is_legacy_gibbs_thomson(self):
        temperature = 673.15
        radius = 6.0
        gamma = 0.168
        volume = unit.USER_PHYSICAL_INPUTS.Vm_compound
        scale = 1.3779024e5
        legacy = solve_gibbs_thomson_x(
            temperature, radius, gamma, volume, scale
        )
        generalized = solve_interface_composition_x(
            temperature, radius, gamma, volume, scale,
            radial_dimension=2, kinetic_beta_code=0.0,
            velocity_nm_per_code_time=0.004,
        )
        self.assertEqual(legacy, generalized)

    def test_finite_lphi_kinetic_boundary_has_expected_sign(self):
        args = dict(
            temperature_k=673.15,
            radius_nm=6.0,
            gamma_j_m2=0.168,
            molar_volume_m3_mol=unit.USER_PHYSICAL_INPUTS.Vm_compound,
            chemical_scale_j_mol=1.3779024e5,
            radial_dimension=2,
            kinetic_beta_code=0.012994628530878145,
        )
        shrinking = solve_interface_composition_x(
            **args, velocity_nm_per_code_time=-0.004
        )
        stationary = solve_interface_composition_x(
            **args, velocity_nm_per_code_time=0.0
        )
        growing = solve_interface_composition_x(
            **args, velocity_nm_per_code_time=0.004
        )
        self.assertLess(shrinking, stationary)
        self.assertLess(stationary, growing)
        expected = args["kinetic_beta_code"] * 0.004
        actual = (
            unit.mu_Ag2Te(args["temperature_k"], growing)
            - unit.mu_Ag2Te(args["temperature_k"], stationary)
        ) / args["chemical_scale_j_mol"]
        self.assertAlmostEqual(actual, expected, places=14)

    def test_spherical_curvature_is_twice_cylindrical(self):
        common = dict(
            temperature_k=673.15,
            radius_nm=6.0,
            gamma_j_m2=0.168,
            molar_volume_m3_mol=unit.USER_PHYSICAL_INPUTS.Vm_compound,
            chemical_scale_j_mol=1.3779024e5,
            kinetic_beta_code=0.0,
            velocity_nm_per_code_time=0.0,
        )
        cylinder = solve_interface_composition_x(
            **common, radial_dimension=2
        )
        sphere = solve_interface_composition_x(
            **common, radial_dimension=3
        )
        self.assertGreater(sphere, cylinder)

    def test_finite_lphi_coefficient_uses_runtime_code_units(self):
        params = {
            "L_phi": "85.50541544691814",
            "L_phi_code_value": "85.50541544691814",
            "lambda_sm_m": "6e-10",
        }
        beta = finite_lphi_kinetic_beta_code(params)
        self.assertAlmostEqual(
            beta, (2.0 / 3.0) / (85.50541544691814 * 0.6), places=15
        )

    def test_lphi_multiplier_rejects_nonpositive_values(self):
        with self.assertRaisesRegex(ValueError, "l_phi_multiplier"):
            run_case(
                Path("/this/file/must/not/be/read.params"),
                6.0, 16.8, 0.0078305391025, 4.0 / 9.0,
                1.0e-6, 1.0e-6, 16, l_phi_multiplier=0.0,
            )

    def test_stefan_and_kinetic_boundary_are_solved_implicitly(self):
        radius = 6.0
        dr = 0.01
        unknown = np.array([0.0082, 0.0083, 0.0084])

        def boundary(value, velocity):
            del value
            return 0.008 + 1.0e-3 * velocity

        surface, velocity, residual = solve_stefan_surface_state(
            radius, unknown, dr, 9.0, boundary
        )
        gradient = (-3.0 * surface + 4.0 * unknown[0] - unknown[1]) / (2.0 * dr)
        self.assertAlmostEqual(
            velocity, 9.0 * gradient / (1.0 - surface), places=12
        )
        self.assertLess(abs(residual), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
