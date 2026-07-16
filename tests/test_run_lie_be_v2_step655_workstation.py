import json
import pathlib
import tempfile
import unittest

from scripts.run_lie_be_v2_step655_workstation import build_command, select_cases


class RunLieBeV2Step655WorkstationTests(unittest.TestCase):
    def test_empty_selection_runs_all_and_unknown_selection_fails(self):
        cases = [{"case_id": "a"}, {"case_id": "b"}]
        self.assertEqual(select_cases(cases, set()), cases)
        self.assertEqual(select_cases(cases, {"b"}), [{"case_id": "b"}])
        with self.assertRaises(KeyError):
            select_cases(cases, {"missing"})

    def test_command_uses_immutable_raw_ctot_and_equal_time_case(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            binary = root / "main_cuda"
            binary.write_text("", encoding="utf-8")
            params = root / "runtime.params"
            params.write_text("elastic_enabled=0\n", encoding="utf-8")
            meta = root / "init_meta.json"
            meta.write_text(
                json.dumps({"Nx": 512, "Ny": 1, "Nz": 1}),
                encoding="utf-8",
            )
            frozen = root / "frozen"
            case = {
                "case_id": "case_half",
                "run_dt": 0.0015625,
                "steps": 2,
            }
            command = build_command(binary, case, params, meta, frozen)
            self.assertEqual(command[1:4], ["512", "1", "1"])
            self.assertEqual(command[5], "2")
            self.assertIn("--init-Ctot-raw", command)
            ctot = command[command.index("--init-Ctot-raw") + 1]
            self.assertTrue(ctot.endswith("ctot_checkpoint_step000054_Ctot.raw"))
            self.assertIn("--pf-param-file", command)


if __name__ == "__main__":
    unittest.main()
