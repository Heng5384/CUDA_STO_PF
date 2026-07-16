import importlib.util
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "next3_curved_derivation",
    ROOT / "scripts" / "derive_next3_curved_fixed_ctot_model.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Next3CurvedDerivationTests(unittest.TestCase):
    def test_quintic_storage_profile_has_equimolar_symmetry(self):
        for z in (-8.0, -2.0, -0.7, 0.0, 0.7, 2.0, 8.0):
            self.assertAlmostEqual(
                MODULE.h(MODULE.phi0(-z)),
                1.0 - MODULE.h(MODULE.phi0(z)),
                places=14,
            )

    def test_planar_antitrapping_identity(self):
        for z in (-4.0, -1.0, -0.2, 0.0, 0.2, 1.0, 4.0):
            phi = MODULE.phi0(z)
            h_value = MODULE.h(phi)
            shape = h_value * (1.0 - h_value) / (
                4.0 * phi * (1.0 - phi)
            )
            self.assertAlmostEqual(
                shape * (-MODULE.phi0_prime(z)),
                h_value * (1.0 - h_value),
                places=14,
            )

    def test_profile_norm_and_current_moment_match_exact_values(self):
        profile_norm = MODULE.integrate(lambda z: MODULE.phi0_prime(z) ** 2)
        current_moment = MODULE.integrate(
            lambda z: MODULE.h(MODULE.phi0(z)) * MODULE.alpha(z)
        )
        self.assertAlmostEqual(profile_norm, 2.0 / 3.0, places=13)
        self.assertAlmostEqual(current_moment, 209.0 / 1680.0, places=13)

    def test_total_c_excess_is_storage_excess_times_composition_jump(self):
        matrix_x = 0.013

        def total_c_excess(z):
            h_value = MODULE.h(MODULE.phi0(z))
            c_pf = h_value + (1.0 - h_value) * matrix_x
            c_sharp = 1.0 if z < 0.0 else matrix_x
            return c_pf - c_sharp

        gamma_c = MODULE.integrate(total_c_excess)
        gamma_h = MODULE.integrate(
            lambda z: MODULE.h(MODULE.phi0(z))
            - (1.0 if z < 0.0 else 0.0)
        )
        self.assertAlmostEqual(gamma_c, (1.0 - matrix_x) * gamma_h, places=14)
        self.assertLess(abs(gamma_c), 1.0e-13)

    def test_blocked_curved_mode_is_not_in_runtime_sources(self):
        runtime = "\n".join(
            (ROOT / name).read_text(encoding="utf-8")
            for name in ("main_cuda.cu", "cuda_kernels.cu", "cuda_kernels.h")
        )
        self.assertNotIn("curved_matched_v1", runtime)


if __name__ == "__main__":
    unittest.main()
