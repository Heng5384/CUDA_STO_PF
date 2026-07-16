import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PrepareJc4DtContinuationProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(ROOT))
        from scripts import prepare_jc4_dt_continuation_probe as module
        cls.module = module

    def create_source(self, root, direction):
        case = root / "cases" / f"T400_{direction}_startup"
        result = case / "run/results"
        result.mkdir(parents=True)
        (case / "runtime_manifest.json").write_text(json.dumps({
            "case_id": case.name,
            "grid": [32, 2, 2],
            "dx_nm": 1.0,
            "lambda_nm": 4.0,
            "temperature_C": 400.0,
        }))
        (case / "runtime.params").write_text(
            "dt=3.125e-3\ninit_case_tag=old\nctot_performance_profile_enabled=1\n"
        )
        (case / "workstation_status.json").write_text(json.dumps({
            "returncode": 0, "accepted_steps": 10, "expected_steps": 10,
            "retry_count": 0, "reject_count": 0,
        }))
        for suffix in ("phi.raw", "xB_alpha.raw", "Ctot.raw"):
            (result / f"ctot_checkpoint_step000010_{suffix}").write_bytes(b"raw")
        (result / "ctot_checkpoint_step000010_meta.json").write_text(json.dumps({
            "schema": "ctot_checkpoint_v1", "time_code": 0.03125,
        }))

    def test_build_uses_authoritative_accepted_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            source = base / "source"
            for direction in ("growth", "dissolution"):
                self.create_source(source, direction)
            output = base / "output"
            manifest = self.module.build(source, output, (0.01, 0.02), 5)
            self.assertEqual(len(manifest["cases"]), 4)
            for row in manifest["cases"]:
                case = output / "cases" / row["case_id"]
                self.assertEqual(row["authoritative_restart"], "Ctot")
                self.assertEqual(row["source_checkpoint_step"], 10)
                self.assertTrue((case / "Ctot_init.raw").is_file())
                params = (case / "runtime.params").read_text()
                self.assertIn(f"init_case_tag={row['case_id']}", params)
                self.assertIn("ctot_performance_profile_enabled=0", params)

    def test_profiler_can_be_enabled_only_for_matched_engineering_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            source = base / "source"
            for direction in ("growth", "dissolution"):
                self.create_source(source, direction)
            output = base / "output"
            manifest = self.module.build(
                source, output, (0.003125,), 2,
                performance_profile_enabled=True,
            )
            self.assertTrue(manifest["performance_profile_enabled"])
            for row in manifest["cases"]:
                params = (
                    output / "cases" / row["case_id"] / "runtime.params"
                ).read_text()
                self.assertIn("ctot_performance_profile_enabled=1", params)

    def test_replace_param_appends_missing_key(self):
        text = self.module.replace_param("a=1\n", "dt", "1e-2")
        self.assertIn("dt=1e-2", text)


if __name__ == "__main__":
    unittest.main()
