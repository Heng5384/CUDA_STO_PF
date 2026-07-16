from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import csv


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Research2PlateauToolsTest(unittest.TestCase):
    def test_reservoir_schema_contract(self) -> None:
        schema = json.loads((ROOT / "schemas/research2_gp_reservoir_state.schema.json").read_text())
        reservoir = schema["$defs"]["reservoir"]
        self.assertIn("current_inventory", reservoir["required"])
        self.assertIn("cumulative_released_inventory", reservoir["required"])
        self.assertNotIn("growth_rate", reservoir["properties"])
        self.assertEqual(
            schema["properties"]["population_mode"]["const"],
            "FIXED_POPULATION_DEPLETION_ONLY",
        )

    def test_mimetic_planar_input_is_finite_interface_off(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "case"
            command = [
                sys.executable,
                str(ROOT / "scripts/prepare_one_sided_planar_benchmark.py"),
                "--base-params",
                str(ROOT / "reports/pf_ctot_production_candidate/prompt8_inputs/tiny_base.params"),
                "--out-dir", str(output),
                "--dx-nm", "0.1",
                "--lphi-factor", "3",
                "--mode", "ctot_mimetic_be",
                "--dt", "1e-5",
                "--nsteps", "2",
            ]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            manifest = json.loads((output / "benchmark_manifest.json").read_text())
            params = (output / "benchmark.params").read_text()
            self.assertEqual(manifest["mode"], "ctot_mimetic_be")
            self.assertFalse(manifest["finite_interface_antitrapping_enabled"])
            self.assertAlmostEqual(manifest["L_phi_code"], 3 * 5.34408846543238347)
            self.assertIn("composition_evolution_mode=ctot_mimetic_be", params)
            self.assertIn("ctot_finite_interface_antitrapping_enabled=0", params)
            self.assertIn("elastic_enabled=0", params)
            self.assertIn("ctot_automatic_dt_growth=0", params)

    def test_stiffness_dt_schedule_decreases(self) -> None:
        runner = load_module(
            "research2_plateau_runner",
            ROOT / "scripts/run_research2_fast_interface_plateau_workstation.py",
        )
        self.assertGreater(runner.choose_dt(1.0, 0.1), runner.choose_dt(100.0, 0.1))
        self.assertEqual(runner.T400_LPHI_REF_CODE, 5.34408846543238347)
        values = runner.parse_params(
            ROOT / "reports/pf_ctot_production_candidate/prompt8_inputs/tiny_base.params"
        )
        self.assertEqual(float(values["L_phi_code_value"]), runner.T400_LPHI_REF_CODE)

    def test_plateau_primitives(self) -> None:
        analyzer = load_module(
            "research2_plateau_analyzer",
            ROOT / "scripts/analyze_research2_fast_interface_plateau.py",
        )
        self.assertAlmostEqual(analyzer.relative_change(1.0, 1.05), 0.05 / 1.05)
        values = analyzer.np.array([0.0, 0.25, 0.75, 1.0, 0.75, 0.25])
        self.assertEqual(len(analyzer.crossings(values, 0.1)), 2)
        phi = analyzer.np.array([0.0, 0.5, 1.0])
        self.assertTrue(analyzer.np.allclose(analyzer.h(phi), [0.0, 0.5, 1.0]))

    def test_candidate_mu_reconstructs_runtime_mu0(self) -> None:
        analyzer = load_module(
            "research2_plateau_analyzer_mu0",
            ROOT / "scripts/analyze_research2_fast_interface_plateau.py",
        )
        params = {}
        for raw in (ROOT / "reports/pf_ctot_production_candidate/prompt8_inputs/tiny_base.params").read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                params[key.strip()] = value.strip()
        mu, mobility = analyzer.candidate_mu_and_mobility(
            analyzer.np.array([0.0, 0.5, 1.0]),
            analyzer.np.array([0.02, 0.02, 0.02]),
            params,
        )
        self.assertTrue(analyzer.np.all(analyzer.np.isfinite(mu)))
        self.assertGreater(mobility[0], 0.0)
        self.assertEqual(mobility[-1], 0.0)

    def test_face_flux_handles_zero_mobility_without_warning(self) -> None:
        analyzer = load_module(
            "research2_plateau_analyzer_flux",
            ROOT / "scripts/analyze_research2_fast_interface_plateau.py",
        )
        mu = analyzer.np.array([0.0, 1.0, 2.0])
        mobility = analyzer.np.array([1.0, 0.0, 1.0])
        with analyzer.np.errstate(all="raise"):
            flux = analyzer.face_flux(mu, mobility, 1.0)
        self.assertTrue(analyzer.np.all(analyzer.np.isfinite(flux)))
        self.assertTrue(analyzer.np.allclose(flux[:2], 0.0))
        self.assertAlmostEqual(float(flux[2]), -2.0)

    def test_accepted_state_gate_uses_production_evidence(self) -> None:
        analyzer = load_module(
            "research2_plateau_analyzer_gates",
            ROOT / "scripts/analyze_research2_fast_interface_plateau.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write_csv(name: str, header: list[str], rows: list[list[object]]) -> None:
                with (root / name).open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(header)
                    writer.writerows(rows)

            write_csv(
                "ctot_acceptance_predicate.csv",
                ["physical_step_id", "attempt_id", "no_nan_inf", "accepted"],
                [[1, 1, 1, 1], [2, 1, 1, 1]],
            )
            write_csv(
                "ctot_outer_iterations.csv",
                ["physical_step", "phase_KKT_residual",
                 "local_phase_storage_residual", "status"],
                [[1, "1e-12", "2e-16", "CONVERGED"],
                 [2, "3e-12", "4e-16", "CONVERGED"]],
            )
            write_csv(
                "ctot_energy_work.csv",
                ["physical_step_id", "monotone_pass", "balance_pass", "accepted"],
                [[1, 1, 1, 1], [2, 1, 1, 1]],
            )
            passed, kkt, storage, rows = analyzer.accepted_state_gate_metrics(root, 2)
            self.assertTrue(passed)
            self.assertEqual(kkt, 3e-12)
            self.assertEqual(storage, 4e-16)
            self.assertEqual(rows, 2)

    def test_research2_temperature_base_keeps_sources_off(self) -> None:
        generator = load_module(
            "research2_temperature_base",
            ROOT / "scripts/prepare_research2_temperature_base.py",
        )
        self.assertEqual(generator.NUMERICAL_OVERLAY["diagnostic_rsmd_enabled"], 0)
        self.assertEqual(generator.NUMERICAL_OVERLAY["gp_nuc_enabled"], 0)
        self.assertEqual(generator.NUMERICAL_OVERLAY["gp_growth_enabled"], 0)
        self.assertEqual(generator.NUMERICAL_OVERLAY["pf_params_schema_version"], 2)
        self.assertEqual(generator.NUMERICAL_OVERLAY["model_mode"], "two_phase")
        self.assertEqual(generator.NUMERICAL_OVERLAY["beta_staged_conversion_enabled"], 0)
        self.assertEqual(
            generator.NUMERICAL_OVERLAY["composition_evolution_mode"],
            "ctot_mimetic_be",
        )

    def test_evidence_plateau_pair(self) -> None:
        assembler = load_module(
            "research2_evidence_assembler",
            ROOT / "scripts/assemble_research2_fast_interface_evidence.py",
        )
        rows = [
            {"L_phi_factor": "1000", "full_window_velocity_nm_s": "1.0",
             "beta_h_inventory_gain_cell_units": "2.0",
             "matrix_flux_from_stefan_code": "3.0",
             "numerical_hard_gates_pass": "True"},
            {"L_phi_factor": "3000", "full_window_velocity_nm_s": "1.05",
             "beta_h_inventory_gain_cell_units": "2.1",
             "matrix_flux_from_stefan_code": "3.15",
             "numerical_hard_gates_pass": "True"},
        ]
        pair = assembler.plateau_pair(rows, 1000.0, 3000.0)
        self.assertTrue(pair["plateau_10pct_pass"])
        self.assertTrue(pair["all_hard_gates_pass"])


if __name__ == "__main__":
    unittest.main()
