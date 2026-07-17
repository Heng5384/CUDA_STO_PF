import csv
import pathlib
import tempfile
import unittest
from unittest import mock

from scripts import run_transport_defect_gate_v2_holdout as runner


FIELDS = (
    "gate_id",
    "dt_id",
    "long_window_status",
    "long_hard_gates_pass",
    "long_retry_efficiency_gate_pass",
    "all_registered_windows_accuracy_pass",
    "signed_residual_bias_growth_pass",
    "physical_s_per_GPU_hour",
)


def write_rows(root: pathlib.Path, rows: list[dict[str, str]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "long_window_candidate_summary.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def candidate(gate, dt, throughput, **overrides):
    row = {
        "gate_id": gate,
        "dt_id": dt,
        "long_window_status": "FAIL",
        "long_hard_gates_pass": "True",
        "long_retry_efficiency_gate_pass": "True",
        "all_registered_windows_accuracy_pass": "True",
        "signed_residual_bias_growth_pass": "True",
        "physical_s_per_GPU_hour": str(throughput),
    }
    row.update(overrides)
    return row


class HoldoutSelectionTest(unittest.TestCase):
    def test_v1_local_peak_failure_does_not_exclude_v2_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            v1 = pathlib.Path(tmp)
            write_rows(
                v1,
                [
                    candidate("G10", "dt8", 1000),
                    candidate("G9", "dt16", 950),
                    candidate("G8", "dt8", 700),
                ],
            )
            with mock.patch.object(runner, "V1", v1):
                self.assertEqual(
                    runner.auto_candidates(),
                    [("G10", "dt8"), ("G9", "dt16")],
                )

    def test_runner_up_below_ninety_percent_is_not_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            v1 = pathlib.Path(tmp)
            write_rows(
                v1,
                [
                    candidate("G8", "dt8", 1000),
                    candidate("G9", "dt16", 899),
                ],
            )
            with mock.patch.object(runner, "V1", v1):
                self.assertEqual(runner.auto_candidates(), [("G8", "dt8")])

    def test_incomplete_formal_queue_blocks_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            v1 = pathlib.Path(tmp)
            rows = [candidate("G8", "dt8", 1000) for _ in range(13)]
            rows[4]["long_window_status"] = "PENDING"
            write_rows(v1, rows)
            with mock.patch.object(runner, "V1", v1):
                with self.assertRaisesRegex(RuntimeError, "must finish"):
                    runner.formal_queue_complete()


if __name__ == "__main__":
    unittest.main()
