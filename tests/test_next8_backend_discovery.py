import importlib.util
import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "next8_backend", ROOT / "scripts" / "audit_next8_independent_backend.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Next8BackendDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.base = MODULE.repository_candidates()[0]

    def test_diagnostic_rsmd_is_rejected_as_pf_coupled(self):
        candidate = MODULE.repository_candidates()[0]
        self.assertFalse(candidate.backend_independent)
        self.assertTrue(candidate.circular_calibration)
        self.assertEqual(MODULE.classify_capability(candidate), "NOT_PHYSICALLY_ELIGIBLE")

    def test_all_hard_gates_are_required_for_full_rank(self):
        full = replace(
            self.base,
            backend_independent=True,
            phase_force_controllable=True,
            diffusion_force_controllable=True,
            forces_non_collinear=True,
            Vn_measurable=True,
            JB_measurable=True,
            raw_trajectory_available=True,
        )
        self.assertEqual(MODULE.classify_capability(full), "FULL_RANK_INTERFACE_RESPONSE_BACKEND")
        for gate in MODULE.HARD_GATES:
            failed = replace(full, **{gate: False})
            self.assertNotEqual(MODULE.classify_capability(failed), "FULL_RANK_INTERFACE_RESPONSE_BACKEND")

    def test_partial_backends_are_not_promoted(self):
        phase = replace(self.base, backend_independent=True, phase_force_controllable=True,
                        Vn_measurable=True, diffusion_force_controllable=False,
                        JB_measurable=False, raw_trajectory_available=True)
        diffusion = replace(self.base, backend_independent=True, phase_force_controllable=False,
                            Vn_measurable=False, diffusion_force_controllable=True,
                            JB_measurable=True, raw_trajectory_available=True)
        self.assertEqual(MODULE.classify_capability(phase), "PHASE_MOBILITY_ONLY_BACKEND")
        self.assertEqual(MODULE.classify_capability(diffusion), "DIFFUSION_ONLY_BACKEND")

    def test_repository_has_no_eligible_backend(self):
        eligible = [c for c in MODULE.repository_candidates()
                    if MODULE.classify_capability(c) == "FULL_RANK_INTERFACE_RESPONSE_BACKEND"]
        self.assertEqual(eligible, [])


if __name__ == "__main__":
    unittest.main()
