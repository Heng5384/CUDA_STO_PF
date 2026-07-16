import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_next_cubic_profile.py"
SPEC = importlib.util.spec_from_file_location("cubic_analysis", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class NextCubicProfileAnalysisTests(unittest.TestCase):
    def test_nsys_preamble_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.csv"
            path.write_text(
                "Generating SQLite file\nProcessing report\n"
                "Time (%),Total Time (ns),Num Calls,Name\n"
                "100,42,3,cudaDeviceSynchronize\n"
            )
            data = module.nsys_rows(path)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["Num Calls"], "3")

    def test_source_memory_scales_and_elastic_adds_state(self):
        _, off64 = module.source_memory(64, False)
        _, on64 = module.source_memory(64, True)
        _, off128 = module.source_memory(128, False)
        self.assertGreater(on64, off64)
        self.assertGreater(off128, off64)


if __name__ == "__main__":
    unittest.main()
