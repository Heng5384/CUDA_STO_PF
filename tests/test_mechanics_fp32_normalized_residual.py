import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_mechanics_normalization_manufactured_tests.py"
SPEC = importlib.util.spec_from_file_location("mechanics_manufactured", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MechanicsNormalizedResidualManufacturedTest(unittest.TestCase):
    def test_mechanics_normalized_residual_manufactured_contract(self):
        _, summary = MODULE.run()
        self.assertTrue(summary["pass"], summary)


if __name__ == "__main__":
    unittest.main()
