import csv
import json
import pathlib
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class TransportDefectGateV2ObserverTest(unittest.TestCase):
    def compile_and_run(self, source: str, output: pathlib.Path) -> None:
        cpp = output.parent / "observer_test.cpp"
        exe = output.parent / "observer_test"
        cpp.write_text(source)
        subprocess.run(
            [
                "c++",
                "-std=c++14",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(ROOT),
                str(cpp),
                "-o",
                str(exe),
            ],
            check=True,
        )
        subprocess.run([str(exe), str(output)], check=True)

    def test_exact_window_metrics_and_transactional_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = textwrap.dedent(
                r"""
                #include "transport_defect_gate_v2_observer.h"
                #include <string>
                #include <vector>

                int main(int argc, char **argv) {
                    if (argc != 2) return 10;
                    TransportDefectGateV2Observer observer;
                    if (!observer.open(argv[1], 4, 1.0)) return 11;
                    std::vector<double> r{1.0, -1.0, 0.0, 0.0};
                    std::vector<double> c0{0.0, 0.0, 0.0, 0.0};
                    std::vector<double> c1{0.1, 0.1, 0.0, 0.0};
                    std::vector<double> c2{0.2, 0.2, 0.0, 0.0};
                    std::vector<double> phi{0.0, 0.5, 1.0, 0.0};
                    // This rejected event transaction must leave no trace.
                    if (!observer.accumulate_accepted(
                            0.5, r, c0, c1, phi, true)) return 12;
                    observer.reset_event_transaction();
                    if (!observer.accumulate_accepted(
                            0.5, r, c0, c1, phi, false)) return 13;
                    if (!observer.accumulate_accepted(
                            0.5, r, c1, c2, phi, true)) return 14;
                    if (!observer.commit_event_transaction()) return 15;
                    if (!observer.finalize()) return 16;
                    return 0;
                }
                """
            )
            self.compile_and_run(source, root)
            csv_path = root / "ctot_transport_defect_v2_window_metrics.csv"
            with csv_path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["record_type"] for row in rows],
                             ["WINDOW", "FULL_TRAJECTORY"])
            window = rows[0]
            self.assertAlmostEqual(float(window["delta_time_code"]), 1.0)
            self.assertAlmostEqual(float(window["sum_abs_D"]), 2.0)
            self.assertAlmostEqual(float(window["sum_A"]), 0.4)
            self.assertAlmostEqual(float(window["eta_global"]), 5.0)
            self.assertAlmostEqual(float(window["eta_interface"]), 5.0)
            self.assertAlmostEqual(float(window["eta_cell_max"]), 5.0)
            self.assertAlmostEqual(float(window["b_signed"]), 0.0)
            self.assertAlmostEqual(float(window["b_interface"]), 1.0)
            self.assertEqual(int(window["interface_cells"]), 1)
            meta = json.loads(
                (root / "ctot_transport_defect_v2_meta.json").read_text()
            )
            self.assertEqual(meta["accepted_rows"], 2)
            self.assertEqual(meta["complete_windows"], 1)
            self.assertFalse(meta["solver_state_modified"])

    def test_source_contract_is_default_off_and_requires_old_observer(self):
        params = (ROOT / "pf_params.h").read_text()
        main = (ROOT / "main_cuda.cu").read_text()
        self.assertIn("int ctot_transport_defect_v2_diagnostics;", params)
        self.assertIn(
            "P->ctot_transport_defect_v2_diagnostics = 0;", main
        )
        self.assertIn(
            "!P->ctot_transport_gate_trajectory_diagnostics", main
        )
        self.assertIn(
            "ctot_transport_defect_v2_observer.accumulate_accepted", main
        )
        self.assertIn(
            "ctot_transport_defect_v2_observer.reset_event_transaction", main
        )


if __name__ == "__main__":
    unittest.main()
