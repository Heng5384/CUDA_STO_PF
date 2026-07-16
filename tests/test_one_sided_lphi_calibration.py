import copy
import math
import unittest

import Unit_Psedobinary as unit


class OneSidedLPhiCalibrationTests(unittest.TestCase):
    def test_zeta0_analytic_matches_high_accuracy_quadrature(self):
        lam = 6.0e-10
        w, kappa, _ = unit.compute_interface_params(0.168, lam)
        analytic = unit.compute_beta_phi_one_sided_zeta0_analytic(lam, w, kappa)
        numerical = unit.compute_beta_phi_one_sided_zeta0_quadrature(
            lam, w, kappa, n_quad=32001)
        self.assertLess(abs(analytic - numerical), 5.0e-10)

    def test_quadrature_converges_under_refinement(self):
        lam = 6.0e-10
        w, kappa, _ = unit.compute_interface_params(0.168, lam)
        analytic = unit.compute_beta_phi_one_sided_zeta0_analytic(lam, w, kappa)
        errors = []
        for n in (1001, 2001, 4001, 8001, 16001):
            value = unit.compute_beta_phi_one_sided_zeta0_quadrature(
                lam, w, kappa, n_quad=n)
            errors.append(abs(value - analytic))
        self.assertTrue(all(b < a for a, b in zip(errors, errors[1:])))
        self.assertGreater(errors[-2] / errors[-1], 3.9)

    def test_beta_zeta_uses_scalar_backend_identity(self):
        T = 673.15
        x = unit.xAg2Te_eq_from_T(T)
        observed = unit.compute_beta_phi_zeta(T, x, 0.0, 1.0)
        expected = (1.0 - x) ** 2 * unit.g_alpha_second_unified(T, x)
        self.assertLess(abs(observed - expected) / abs(expected), 1.0e-15)

    def test_one_sided_beta_calibration_is_independent_of_gp_inputs(self):
        a = copy.deepcopy(unit.USER_PHYSICAL_INPUTS)
        b = copy.deepcopy(unit.USER_PHYSICAL_INPUTS)
        for inputs in (a, b):
            inputs.temperature_C = 400.0
            inputs.L_phi_calibration_mode = "one_sided_diffusion_controlled"
        b.gp_reaction_nu_A = 0.1
        b.gp_reaction_nu_B = 0.9
        b.gp_xB_eq_alpha_for_eta = 0.08
        b.gp_l_eta = 2.0e-9
        b.gp_D_ratio_eta = 0.2
        pa = unit.PFParamConverter().convert(a)
        pb = unit.PFParamConverter().convert(b)
        self.assertEqual(pa.L_phi, pb.L_phi)
        self.assertEqual(pa.zeta_phi, pb.zeta_phi)
        self.assertEqual(pa.zeta0_phi, pb.zeta0_phi)
        self.assertEqual(pa.D_beta_for_calibration, 0.0)
        self.assertEqual(pb.D_beta_for_calibration, 0.0)

    def test_legacy_mode_retains_gp_eta_cross_coupling(self):
        a = copy.deepcopy(unit.USER_PHYSICAL_INPUTS)
        b = copy.deepcopy(unit.USER_PHYSICAL_INPUTS)
        for inputs in (a, b):
            inputs.temperature_C = 400.0
            inputs.L_phi_calibration_mode = "legacy_gp_eta_coupled"
        b.gp_xB_eq_alpha_for_eta = 0.08
        pa = unit.PFParamConverter().convert(a)
        pb = unit.PFParamConverter().convert(b)
        self.assertNotEqual(pa.gp_zeta_eta, pb.gp_zeta_eta)
        self.assertNotEqual(pa.L_phi, pb.L_phi)

    def test_one_sided_lphi_is_positive_and_finite(self):
        for temperature_C in (380.0, 400.0):
            inputs = copy.deepcopy(unit.USER_PHYSICAL_INPUTS)
            inputs.temperature_C = temperature_C
            inputs.L_phi_calibration_mode = "one_sided_diffusion_controlled"
            params = unit.PFParamConverter().convert(inputs)
            self.assertTrue(math.isfinite(params.L_phi))
            self.assertGreater(params.L_phi, 0.0)
            self.assertEqual(params.L_phi_calibration_mode,
                             "one_sided_diffusion_controlled")


if __name__ == "__main__":
    unittest.main()
