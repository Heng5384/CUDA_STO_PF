import pathlib
import sys
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_jc4_long_time_workstation as runner  # noqa: E402


class RunJc4WorkstationTests(unittest.TestCase):
    def test_source_contract_includes_phase_and_transaction_headers(self):
        for required in (
            "main_cuda.cu", "cuda_kernels.cu", "pf_params.h",
            "phase_kkt_utils.h", "phase_pdas_reduction.h",
            "ctot_performance_profiler.h",
        ):
            self.assertIn(required, runner.SOURCE_FILES)

    @mock.patch.object(runner, "execute")
    def test_busy_workstation_fails_closed(self, execute):
        execute.return_value = mock.Mock(
            returncode=0,
            stdout="PROCESSES\n123 main_cuda\nGPU_PIDS\n123\n",
            stderr="",
        )
        with self.assertRaises(RuntimeError):
            runner.remote_idle("workstation-tail")


if __name__ == "__main__":
    unittest.main()
