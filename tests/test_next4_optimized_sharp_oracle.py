import math
from decimal import Decimal, localcontext
from pathlib import Path
import tempfile
import unittest

import numpy as np

import Unit_Psedobinary as unit
from scripts.next2_finite_box_sharp_oracle import solve_interface_composition_x
from scripts.next4_optimized_sharp_oracle import (
    SolverMetrics,
    build_thermo_branch,
    coupled_surface_velocity_newton_v1,
    dmu_a_dx,
    dmu_b_dx,
    dmu_b_minus_mu_a_dx,
    mu_a,
    mu_b,
    run_case_optimized,
    solve_capillary_root,
    solve_kinetic_root,
)
from scripts.next4_numba_root_kernels import (
    capillary_root_jit,
    coupled_surface_velocity_jit,
    kinetic_root_jit,
)


class Next4OptimizedSharpOracleTests(unittest.TestCase):
    def setUp(self):
        self.temperature = 673.15
        self.branch = build_thermo_branch(self.temperature)
        self.gamma = 0.168
        self.volume = unit.USER_PHYSICAL_INPUTS.Vm_compound
        self.scale = 1.3779024e5
        self.beta = 0.012994628530878145

    def test_analytic_thermo_matches_reference(self):
        for x in np.geomspace(1.0e-9, 0.15, 80):
            self.assertLess(abs(mu_b(self.branch, x) - unit.mu_Ag2Te(
                self.temperature, x)), 2.0e-10)
            self.assertLess(abs(mu_a(self.branch, x) - unit.mu_PbTe(
                self.temperature, x)), 2.0e-10)

    def test_analytic_derivatives_match_80_digit_derivatives(self):
        with localcontext() as context:
            context.prec = 80
            rt = Decimal(str(self.branch.rt))
            l0 = Decimal(str(self.branch.l0))
            one = Decimal(1)
            for x_float in np.geomspace(1.0e-8, 0.15, 60):
                x = Decimal(str(x_float))
                step = max(Decimal("1e-35"), abs(x) * Decimal("1e-28"))

                def mu_b_decimal(y):
                    return rt * y.ln() + l0 * (one-y)**2

                def mu_a_decimal(y):
                    return rt * (one-y).ln() + l0 * y**2

                expected_b = float(
                    (mu_b_decimal(x+step)-mu_b_decimal(x-step))/(2*step)
                )
                expected_a = float(
                    (mu_a_decimal(x+step)-mu_a_decimal(x-step))/(2*step)
                )
                actual_b = dmu_b_dx(self.branch, x_float)
                actual_a = dmu_a_dx(self.branch, x_float)
                self.assertLess(
                    abs(actual_b-expected_b)/max(abs(expected_b), 1.0), 3e-15
                )
                self.assertLess(
                    abs(actual_a-expected_a)/max(abs(expected_a), 1.0), 3e-15
                )
                actual_diff = dmu_b_minus_mu_a_dx(self.branch, x_float)
                self.assertLess(
                    abs(actual_diff-(expected_b-expected_a))
                    / max(abs(expected_b-expected_a), 1.0), 3e-15,
                )

    def test_branch_turning_point_is_retained_explicitly(self):
        self.assertLess(self.branch.x_upper_monotone, self.branch.x_upper_full)
        self.assertGreater(dmu_b_dx(self.branch, 0.99*self.branch.x_upper_monotone), 0.0)
        self.assertLess(dmu_b_dx(self.branch, 1.01*self.branch.x_upper_monotone), 0.0)

    def test_safeguarded_capillary_and_kinetic_roots_match_frozen(self):
        for radius in (3.0, 4.8, 6.0, 9.0, 12.0):
            cap, cap_metrics = solve_capillary_root(
                self.branch, radius, self.gamma, self.volume, 2
            )
            frozen_cap = solve_interface_composition_x(
                self.temperature, radius, self.gamma, self.volume,
                self.scale, 2, 0.0, 0.0,
            )
            self.assertLess(abs(cap-frozen_cap), 2e-15)
            self.assertLess(cap_metrics.iterations, 12)
            for velocity in (-0.006, 0.0, 0.006):
                increment = self.scale*self.beta*velocity
                value, metrics = solve_kinetic_root(
                    self.branch, cap, increment, cap
                )
                frozen = solve_interface_composition_x(
                    self.temperature, radius, self.gamma, self.volume,
                    self.scale, 2, self.beta, velocity,
                )
                self.assertLess(abs(value-frozen), 3e-15)
                self.assertLess(metrics.iterations, 12)

    def test_coupled_newton_matches_frozen_nested_solve(self):
        from scripts.next2_finite_box_sharp_oracle import solve_stefan_surface_state

        radius = 6.0
        dr = 0.02
        unknown = np.array([0.0082, 0.0083, 0.0084], dtype=np.float64)

        def boundary(value, velocity):
            return solve_interface_composition_x(
                self.temperature, value, self.gamma, self.volume,
                self.scale, 2, self.beta, velocity,
            )

        x_old, v_old, _ = solve_stefan_surface_state(
            radius, unknown, dr, 9.0, boundary
        )
        metrics = SolverMetrics.empty()
        _, x_new, v_new, f1, f2 = coupled_surface_velocity_newton_v1(
            self.branch, radius, unknown, dr, 9.0, self.gamma,
            self.volume, self.scale, 2, self.beta, 1.0,
            None, None, 0.0, metrics,
        )
        self.assertLess(abs(x_new-x_old), 3e-15)
        self.assertLess(abs(v_new-v_old), 2e-13)
        self.assertLess(abs(f1)/self.scale, 5e-14)
        self.assertLess(abs(f2), 5e-14)
        self.assertEqual(metrics.coupled_fallback_count, 0)

    def test_numba_root_kernels_match_validated_python_roots(self):
        radius = 6.0
        capillary, _ = solve_capillary_root(
            self.branch, radius, self.gamma, self.volume, 2
        )
        x_eq = unit.xAg2Te_eq_from_T(self.temperature)
        target = (
            mu_b(self.branch, x_eq) - mu_a(self.branch, x_eq)
            + self.gamma*self.volume/(radius*1e-9)
        )
        cap_jit = capillary_root_jit(
            self.branch.mu_b_base, self.branch.mu_a_base,
            self.branch.rt, self.branch.l0, x_eq, target,
            x_eq, self.branch.x_upper_monotone, x_eq, 1e5,
        )
        self.assertEqual(cap_jit[4], 0)
        self.assertLess(abs(cap_jit[0]-capillary), 2e-15)
        increment = self.scale*self.beta*0.004
        kinetic, _ = solve_kinetic_root(
            self.branch, capillary, increment, capillary
        )
        kin_jit = kinetic_root_jit(
            self.branch.mu_b_base, self.branch.rt, self.branch.l0,
            capillary, increment, capillary,
            self.branch.x_upper_monotone, capillary, 1e5,
        )
        self.assertEqual(kin_jit[4], 0)
        self.assertLess(abs(kin_jit[0]-kinetic), 2e-15)

    def test_numba_coupled_kernel_matches_python_coupled_newton(self):
        radius, dr = 6.0, 0.02
        unknown = np.array([0.0082, 0.0083, 0.0084])
        metrics = SolverMetrics.empty()
        capillary, x_python, v_python, _, _ = coupled_surface_velocity_newton_v1(
            self.branch, radius, unknown, dr, 9.0, self.gamma,
            self.volume, self.scale, 2, self.beta, 1.0,
            None, None, 0.0, metrics,
        )
        compiled = coupled_surface_velocity_jit(
            self.branch.mu_b_base, self.branch.rt, self.branch.l0,
            capillary, self.scale*self.beta, unknown[0], unknown[1],
            dr, 9.0, capillary, 0.0, self.branch.x_lower,
            self.branch.x_upper_monotone, self.scale,
        )
        self.assertEqual(compiled[4], 0)
        self.assertLess(abs(compiled[0]-x_python), 2e-15)
        self.assertLess(abs(compiled[1]-v_python), 2e-13)

    def test_short_optimized_case_matches_frozen_path(self):
        import json
        from scripts.next2_finite_box_sharp_oracle import run_case

        state = json.loads(Path(
            "/tmp/pf_next4_inputs_20260714/moving/shared/r10/"
            "moving_state_manifest.json"
        ).read_text())
        params = Path(
            "/tmp/pf_next4_inputs_20260714/stationary/r10/benchmark.params"
        )
        args = (
            params, float(state["radius_h_nm"]), float(state["box_nm"]),
            float(state["far_xB_requested"]), float(state["diffusion_age_code"]),
            1.5625e-5, 1.5625e-6, 256, float(state["pf_inventory_nm2"]),
        )
        frozen = run_case(*args, finite_lphi_kinetics=True)
        optimized = run_case_optimized(*args, finite_lphi_kinetics=True)
        self.assertLess(abs(
            optimized["velocity_full_nm_per_code_time"]
            - frozen["velocity_full_nm_per_code_time"]
        ), 2e-10)
        self.assertLess(abs(
            optimized["radius_final_nm"] - frozen["radius_final_nm"]
        ), 2e-14)
        self.assertEqual(
            optimized["solver_metrics"]["coupled_fallback_count"], 0
        )

    def test_checkpoint_restart_matches_continuous_run(self):
        import json

        state = json.loads(Path(
            "/tmp/pf_next4_inputs_20260714/moving/shared/r10/"
            "moving_state_manifest.json"
        ).read_text())
        params = Path(
            "/tmp/pf_next4_inputs_20260714/stationary/r10/benchmark.params"
        )
        args = (
            params, float(state["radius_h_nm"]), float(state["box_nm"]),
            float(state["far_xB_requested"]), float(state["diffusion_age_code"]),
            3.125e-5, 1.5625e-6, 128, float(state["pf_inventory_nm2"]),
        )
        continuous = run_case_optimized(
            *args, finite_lphi_kinetics=True, root_backend="numba"
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "restart.npz"
            partial = run_case_optimized(
                *args, finite_lphi_kinetics=True, root_backend="numba",
                checkpoint_path=checkpoint, checkpoint_interval_steps=10,
                checkpoint_input_hash="test-input", stop_at_step=10,
            )
            self.assertAlmostEqual(
                partial["actual_final_time_code"], 1.5625e-5, places=19
            )
            restarted = run_case_optimized(
                *args, finite_lphi_kinetics=True, root_backend="numba",
                checkpoint_path=checkpoint, checkpoint_interval_steps=10,
                resume_checkpoint=checkpoint,
                checkpoint_input_hash="test-input",
            )
            self.assertTrue(np.array_equal(
                continuous["profile_final"], restarted["profile_final"]
            ))
            self.assertEqual(
                continuous["radius_final_nm"], restarted["radius_final_nm"]
            )
            metadata = checkpoint.with_suffix(".npz.json")
            changed = json.loads(metadata.read_text())
            changed["checkpoint_version"] = "bad-version"
            metadata.write_text(json.dumps(changed)+"\n")
            with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                run_case_optimized(
                    *args, finite_lphi_kinetics=True, root_backend="numba",
                    resume_checkpoint=checkpoint,
                    checkpoint_input_hash="test-input",
                )


if __name__ == "__main__":
    unittest.main()
