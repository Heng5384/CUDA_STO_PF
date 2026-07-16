import pathlib
import unittest


SOURCE = (pathlib.Path(__file__).resolve().parents[1] / "scripts" /
          "run_next2_p2_profile_workstation.sh").read_text()


class Next2P2ProfileRunnerTests(unittest.TestCase):
    def test_matched_physics_and_solver_flags(self):
        self.assertIn("for implementation in p1 p2", SOURCE)
        self.assertIn("composition_evolution_mode=ctot_mimetic_be", SOURCE)
        self.assertIn("ctot_phase_semismooth_pdas_enabled=1", SOURCE)
        self.assertIn("ctot_finite_interface_antitrapping_enabled=0", SOURCE)
        self.assertIn("dt=6.25e-6", SOURCE)

    def test_scope_and_resource_guards(self):
        self.assertIn("refusing to overwrite run root", SOURCE)
        self.assertIn("GPU already has a compute process", SOURCE)
        self.assertIn("run_case \"$implementation\" \"$binary\" 64 1 1", SOURCE)
        self.assertNotIn("128", SOURCE)


if __name__ == "__main__":
    unittest.main()
