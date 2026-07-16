import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class FinalizeJc4QualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(ROOT))
        from scripts import finalize_jc4_qualification as module
        cls.module = module

    def test_find_line_uses_current_source(self):
        line = self.module.find_line(
            ROOT / "main_cuda.cu",
            "__global__ void ctot_local_storage_preconditioner_kernel",
        )
        self.assertGreater(line, 0)

    def test_empty_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.module.write_rows(pathlib.Path(directory) / "x.csv", [])

    def test_status_row_does_not_claim_unrun_values(self):
        row = self.module.status_row("projection", None, "dissolution", "NOT_RUN", "blocked")
        self.assertEqual(row["actual_elapsed_s"], "")
        self.assertEqual(row["qualification_status"], "NOT_RUN")


if __name__ == "__main__":
    unittest.main()
