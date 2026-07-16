import unittest

from scripts.run_correction2_matrix_workstation import case_failed


class RunCorrection2MatrixWorkstationTests(unittest.TestCase):
    @staticmethod
    def status(**overrides):
        row = {
            "returncode": 0,
            "download_returncode": 0,
            "accepted_steps": 2,
            "expected_steps": 2,
            "retry_count": 0,
            "reject_count": 0,
        }
        row.update(overrides)
        return row

    def test_complete_case_passes(self):
        self.assertFalse(case_failed(self.status()))

    def test_each_fail_closed_condition_is_preserved(self):
        for override in (
            {"returncode": 1},
            {"download_returncode": 1},
            {"accepted_steps": 1},
            {"retry_count": 1},
            {"reject_count": 1},
        ):
            with self.subTest(override=override):
                self.assertTrue(case_failed(self.status(**override)))


if __name__ == "__main__":
    unittest.main()
