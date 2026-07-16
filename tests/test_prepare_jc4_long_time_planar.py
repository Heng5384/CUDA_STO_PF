import json
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PrepareJc4LongTimePlanarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(ROOT))
        from scripts import prepare_jc4_long_time_planar as module
        cls.module = module

    def test_inventory_capacity_supports_long_displacement(self):
        for direction, spec in self.module.CASE_SPECS.items():
            config = self.module.OracleConfig(
                cells=int(spec["cells"]),
                initial_beta_half_width_nm=float(spec["half_width_nm"]),
                initial_matrix_xB=float(spec["matrix_xB"]),
            )
            oracle = self.module.FixedCtotJiChen1DOracle(config)
            values = oracle.unpack(oracle.initial_state())
            mean_c = float(values["C"].mean())
            equilibrium_fraction = (mean_c - oracle.x_eq) / (1.0 - oracle.x_eq)
            equilibrium_half_width = 0.5 * equilibrium_fraction * config.length_nm
            capacity_displacement = equilibrium_half_width - config.initial_beta_half_width_nm
            if direction == "growth":
                self.assertGreater(capacity_displacement, 5.0)
            else:
                self.assertLess(capacity_displacement, -5.0)

    def test_pilot_matrix_is_pf_only_and_thin(self):
        with tempfile.TemporaryDirectory() as directory:
            oracle_records = [
                {"time_s": 0.0, "beta_half_width_nm": 20.0},
                {"time_s": 1.0, "beta_half_width_nm": 20.5},
            ]
            with mock.patch.object(
                self.module.FixedCtotJiChen1DOracle,
                "run", return_value={"records": oracle_records},
            ):
                manifest = self.module.build(pathlib.Path(directory), "pilot")
            self.assertEqual(len(manifest["cases"]), 10)
            for case in manifest["cases"]:
                self.assertEqual(case["grid"][1:], [2, 2])
                self.assertFalse(case["formal_3d_production"] if "formal_3d_production" in case else False)
                case_root = pathlib.Path(directory) / "cases" / case["case_id"]
                params = (case_root / "runtime.params").read_text()
                self.assertIn("PF_RESEARCH_MODEL=fixed_ctot_ji_chen_coarse4_gp_v1", params)
                self.assertIn("GP_population_mode=OFF", params)
                self.assertIn("ctot_finite_interface_antitrapping_enabled=0", params)
                self.assertIn("ctot_performance_profile_enabled=0", params)
                metadata = json.loads((case_root / "init_meta.json").read_text())
                self.assertEqual(metadata["authoritative_state"], "Ctot")

    def test_full_dissolution_reference_is_long_but_not_post_equilibrium_padding(self):
        spec = self.module.CASE_SPECS["dissolution"]
        self.assertEqual(spec["full_time_code"], 5000.0)
        self.assertEqual(spec["full_dt_code"], 5.0)

    def test_runtime_preparation_can_fail_closed_without_long_host_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.module.build(
                pathlib.Path(directory), "full", run_host_reference=False,
            )
            self.assertFalse(manifest["host_long_time_reference_requested"])
            self.assertEqual(len(manifest["cases"]), 2)
            for case in manifest["cases"]:
                self.assertIsNone(case["oracle_final_displacement_nm"])
                self.assertEqual(
                    case["host_long_time_oracle_status"],
                    "NOT_RUN_ACTIVE_SET_BACKEND_PERFORMANCE_BLOCKED",
                )
                self.assertGreater(
                    abs(case["finite_box_equilibrium_displacement_nm"]), 5.0
                )

    def test_event_heavy_profiler_is_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.module.build(
                pathlib.Path(directory), "stability",
                run_host_reference=False,
                performance_profile_enabled=True,
            )
            self.assertTrue(manifest["performance_profile_enabled"])
            for case in manifest["cases"]:
                params = (
                    pathlib.Path(directory) / "cases" / case["case_id"] /
                    "runtime.params"
                ).read_text()
                self.assertIn("ctot_performance_profile_enabled=1", params)

    def test_single_cell_transverse_embedding_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.module.build(
                pathlib.Path(directory), "stability_refined",
                run_host_reference=False,
                transverse_cells=1,
            )
            self.assertEqual(manifest["transverse_cells"], 1)
            for case in manifest["cases"]:
                self.assertEqual(case["grid"][1:], [1, 1])
                metadata = json.loads((
                    pathlib.Path(directory) / "cases" / case["case_id"] /
                    "init_meta.json"
                ).read_text())
                self.assertEqual([metadata["Ny"], metadata["Nz"]], [1, 1])

    def test_transverse_embedding_rejects_zero_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.module.build(
                    pathlib.Path(directory), "stability_refined",
                    run_host_reference=False,
                    transverse_cells=0,
                )

    def test_fixed_dt_stability_pilot_has_exact_equal_time(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.module.build(
                pathlib.Path(directory), "stability", run_host_reference=False,
            )
            self.assertEqual(len(manifest["cases"]), 2)
            for case in manifest["cases"]:
                self.assertEqual(case["dt_code"], 0.00625)
                self.assertEqual(case["nsteps"], 100)
                self.assertEqual(case["final_time_code"], 0.625)

    def test_refined_stability_pilot_doubles_steps_at_same_time(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.module.build(
                pathlib.Path(directory), "stability_refined",
                run_host_reference=False,
            )
            for case in manifest["cases"]:
                self.assertEqual(case["dt_code"], 0.003125)
                self.assertEqual(case["nsteps"], 200)
                self.assertEqual(case["final_time_code"], 0.625)

    def test_research_gate_uses_safe_fixed_dt_and_long_displacement_times(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.module.build(
                pathlib.Path(directory), "research_gate",
                run_host_reference=False,
                transverse_cells=1,
            )
            by_direction = {case["direction"]: case for case in manifest["cases"]}
            self.assertEqual(by_direction["growth"]["dt_code"], 0.003125)
            self.assertEqual(by_direction["growth"]["nsteps"], 16000)
            self.assertEqual(by_direction["dissolution"]["dt_code"], 0.003125)
            self.assertEqual(by_direction["dissolution"]["nsteps"], 640000)
            self.assertFalse(manifest["performance_profile_enabled"])

    def test_crossing_probe_records_explicit_matrix_support_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.module.build(
                pathlib.Path(directory), "crossing_probe",
                run_host_reference=False,
                transverse_cells=1,
                matrix_support_eps=1.0e-8,
            )
            self.assertEqual(manifest["matrix_support_eps"], 1.0e-8)
            for case in manifest["cases"]:
                self.assertEqual(case["nsteps"], 1000)
                params = (
                    pathlib.Path(directory) / "cases" / case["case_id"] /
                    "runtime.params"
                ).read_text()
                self.assertIn(
                    "ctot_matrix_support_eps=1.00000000000000002e-08", params
                )


if __name__ == "__main__":
    unittest.main()
